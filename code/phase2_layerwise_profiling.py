# phase2_layerwise_profiling.py
"""
ANCHOR Phase 2：逐层特征剖析与谄媚向量持久化（Layer-wise Profiling）。

学术动机:
  1) 在 Think: OFF（系统1 / 快思考）下蒸馏「高浓度」谄媚语义方向，
     为创新点 2（跨认知模式零样本迁移到 Think: ON）提供血清。
  2) 逐层 PCA 的 Explained Variance Ratio (EVR) 证明谄媚特征在深层聚集，
     为创新点 1（多层联合洗涤，如 18–20）提供选层依据。
  3) 提取仅使用 extraction_set_mmlu（Phase1 隔离），绝不碰测试集。

对比对（Contrastive Pair）——与既有 step2 一致:
  Truthful   : 客观 baseline prompt
  Sycophantic: baseline + 助手已答对 + 用户 strong_vanity 高压诱导
  ΔH_L = H_syc[L] - H_truth[L]  （取 prompt 最后一个 token 的层 L 输出）

输出（默认写到提取集同目录的 vectors/）:
  layerwise_evr.csv              —— 每层 EVR（Phase4 折线图）
  sycophancy_vector_layer{L}.pt  —— 每一层的第一主成分（符号已对齐）
  sycophancy_vector_best_layer.pt—— EVR 最高单层（兼容旧 step3）
  sycophancy_vectors_bundle.pt   —— 全部层向量 + EVR + 推荐连续层窗口
  profiling_summary.json         —— 可读摘要（best / top-k 连续窗口）

用法:
  python phase2_layerwise_profiling.py \\
      --extraction-set ../repeng_data/Qwen3-8B_multi_nothink/03_splits/extraction_set_mmlu.json \\
      --output-dir ../repeng_data/Qwen3-8B_multi_nothink/04_vectors \\
      --model-path /path/to/Qwen3-8B \\
      --no-enable-thinking
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

import config
from backend import LocalModel
from model_adapt import get_hidden_size, get_num_hidden_layers
from prompts import (
    build_drift_followup_flow,
    build_single_round_prompt,
    with_answer_suffix,
)


def load_items(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def build_contrastive_messages(
    item: dict, *, enable_thinking: bool
) -> tuple[list[dict], list[dict]]:
    """构造 Truthful vs Sycophantic 对话消息。"""
    question_prompt = build_single_round_prompt(
        item, "baseline", enable_thinking=enable_thinking
    )
    truthful = [{"role": "user", "content": question_prompt}]

    # 模拟「模型已给出正确答案」后，用户施加最高档 strong_vanity
    strong_vanity_text = build_drift_followup_flow(item)[-1]
    sycophantic = [
        {"role": "user", "content": question_prompt},
        {"role": "assistant", "content": f"答案：{item['answer']}"},
        {
            "role": "user",
            "content": with_answer_suffix(strong_vanity_text, enable_thinking),
        },
    ]
    return truthful, sycophantic


@torch.no_grad()
def last_token_hidden_by_layer(
    model: LocalModel, messages: list[dict], *, enable_thinking: bool
) -> list[np.ndarray]:
    """
    返回每个解码器层在「读完 prompt 最后一 token」处的激活。

    层索引约定（与 SycophancyIntervention 钩子对齐）:
      hidden_states[0]     = embedding
      hidden_states[L+1]   = decoder layer L 的输出
      钩子挂在 decoder layer L 上（见 model_adapt.get_decoder_layers）→ 保存的 layer=L 可直接用于 Phase3
    """
    prompt = model.build_prompt(messages, enable_thinking=enable_thinking)
    inputs = model.tokenizer(prompt, return_tensors="pt").to(model.model.device)
    out = model.model(**inputs, output_hidden_states=True)
    hs = out.hidden_states
    return [hs[i][0, -1, :].float().cpu().numpy() for i in range(1, len(hs))]


def find_best_consecutive_window(
    evr: list[float],
    window: int,
    *,
    exclude_first: int = 2,
    exclude_last: int = 2,
) -> tuple[list[int], float]:
    """
    在 EVR 曲线上找平均 EVR 最高的连续 window 层（用于多层联合洗涤选层）。

    默认排除最前/最后若干层：末层 EVR 尖峰常不具干预因果性（实证：33-35 无效，
    中层 18-20 有效），避免推荐窗口被最后一层「拉偏」。
    """
    if window <= 0 or window > len(evr):
        raise ValueError(f"window={window} 非法，层数={len(evr)}")
    lo = max(0, exclude_first)
    hi = max(lo + window, len(evr) - max(0, exclude_last))
    # 合法起点: start ∈ [lo, hi-window]
    start_max = hi - window
    if start_max < lo:
        # 退化：不做排除
        lo, start_max = 0, len(evr) - window

    best_start, best_mean = lo, -1.0
    for start in range(lo, start_max + 1):
        mean = float(np.mean(evr[start : start + window]))
        if mean > best_mean:
            best_mean = mean
            best_start = start
    layers = list(range(best_start, best_start + window))
    return layers, best_mean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase2：逐层 PCA 剖析与向量持久化")
    p.add_argument(
        "--extraction-set",
        type=Path,
        required=True,
        help="Phase1 产出的 extraction_set_mmlu.json",
    )
    p.add_argument("--model-path", default=config.MODEL_PATH)
    p.add_argument(
        "--device",
        default=config.DEVICE,
        help='GPU，如 "cuda:1" 或 1；默认 config.DEVICE',
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认：提取集同目录下的 vectors/）",
    )
    p.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="默认 --no-enable-thinking（系统1蒸馏，支撑跨认知迁移）",
    )
    p.add_argument(
        "--window",
        type=int,
        default=3,
        help="推荐连续洗涤窗口宽度（默认 3，对应如 18,19,20）",
    )
    p.add_argument(
        "--top-k-layers",
        type=int,
        default=5,
        help="摘要中额外列出 EVR 最高的 K 层",
    )
    p.add_argument("--limit", type=int, default=None, help="仅用前 N 题（调试）")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    extraction_path = args.extraction_set.resolve()
    if not extraction_path.exists():
        raise FileNotFoundError(
            f"未找到提取集: {extraction_path}\n请先运行 phase1_split_diagnostic.py。"
        )

    items = load_items(extraction_path)
    if args.limit:
        items = items[: args.limit]

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else extraction_path.parent / "vectors"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 规范：bf16 + sdpa 由 LocalModel/config 保证；种子在加载前固定
    device = config.normalize_device(args.device)
    config.set_seed(args.seed)
    model = LocalModel(args.model_path, device=device)
    num_layers = get_num_hidden_layers(model.model)
    hidden = get_hidden_size(model.model)

    print(
        f"提取集: {extraction_path} ({len(items)} 题)\n"
        f"模型: {args.model_path} | device={device} | layers={num_layers} | hidden={hidden}\n"
        f"Think: {'ON' if args.enable_thinking else 'OFF'} | seed={args.seed}\n"
        f"输出: {output_dir}"
    )
    if args.enable_thinking:
        print(
            "⚠️ 警告: 创新点2要求在 Think:OFF 下蒸馏；当前为 ON，"
            "跨认知迁移实验的「源分布」将改变。"
        )

    # ---------- 1) 逐题累积各层 ΔH ----------
    diffs_per_layer: list[list[np.ndarray]] = [[] for _ in range(num_layers)]
    for item in tqdm(items, desc="contrastive_forward"):
        truthful, sycophantic = build_contrastive_messages(
            item, enable_thinking=args.enable_thinking
        )
        h_truth = last_token_hidden_by_layer(
            model, truthful, enable_thinking=args.enable_thinking
        )
        h_syc = last_token_hidden_by_layer(
            model, sycophantic, enable_thinking=args.enable_thinking
        )
        for layer in range(num_layers):
            diffs_per_layer[layer].append(h_syc[layer] - h_truth[layer])

    # ---------- 2) 逐层 PCA + 符号对齐 ----------
    # 符号对齐：让第一主成分与 mean(ΔH) 同向，保证 Phase3「减去 alpha*v」= 去谄媚
    evr_list: list[float] = []
    vectors: list[np.ndarray] = []
    best_layer, best_evr = -1, -1.0

    print("\n[逐层 PCA] Layer → EVR")
    for layer in range(num_layers):
        matrix = np.stack(diffs_per_layer[layer], axis=0)  # [N, H]
        pca = PCA(n_components=1)
        pca.fit(matrix)
        ratio = float(pca.explained_variance_ratio_[0])
        component = pca.components_[0].astype(np.float64)

        mean_diff = matrix.mean(axis=0)
        if float(np.dot(component, mean_diff)) < 0:
            component = -component

        evr_list.append(ratio)
        vectors.append(component.astype(np.float32))
        marker = ""
        if ratio > best_evr:
            best_evr = ratio
            best_layer = layer
            marker = "  ← best"
        print(f"  Layer {layer:>2d}: {ratio:.4f}{marker}")

    # ---------- 3) 推荐连续多层窗口（创新点1）----------
    window_layers, window_mean = find_best_consecutive_window(evr_list, args.window)
    top_k = sorted(range(num_layers), key=lambda i: evr_list[i], reverse=True)[
        : args.top_k_layers
    ]

    # ---------- 4) 持久化 ----------
    # 4a) EVR 表（Phase4 折线图）
    evr_df = pd.DataFrame(
        {
            "layer": list(range(num_layers)),
            "explained_variance_ratio": evr_list,
            "is_best": [i == best_layer for i in range(num_layers)],
            "in_recommended_window": [i in window_layers for i in range(num_layers)],
        }
    )
    evr_csv = output_dir / "layerwise_evr.csv"
    evr_df.to_csv(evr_csv, index=False, encoding="utf-8-sig")

    # 4b) 逐层 .pt（便于 step3 --layers 任意组合）
    for layer, vec in enumerate(vectors):
        payload = {
            "layer": layer,
            "vector": torch.tensor(vec, dtype=torch.float32),
            "explained_variance_ratio": evr_list[layer],
            "num_extraction_items": len(items),
            "model_path": args.model_path,
            "enable_thinking": bool(args.enable_thinking),
            "hidden_size": hidden,
            "seed": args.seed,
            "source_set": str(extraction_path),
        }
        torch.save(payload, output_dir / f"sycophancy_vector_layer{layer}.pt")

    # 4c) best 单层（兼容旧 step3 默认加载）
    best_payload = {
        "layer": best_layer,
        "vector": torch.tensor(vectors[best_layer], dtype=torch.float32),
        "explained_variance_ratio": best_evr,
        "num_extraction_items": len(items),
        "model_path": args.model_path,
        "enable_thinking": bool(args.enable_thinking),
        "hidden_size": hidden,
        "seed": args.seed,
        "source_set": str(extraction_path),
        "recommended_window": window_layers,
        "recommended_window_mean_evr": window_mean,
    }
    best_path = output_dir / "sycophancy_vector_best_layer.pt"
    torch.save(best_payload, best_path)

    # 4d) bundle：一次加载全部层（Phase3 批跑友好）
    bundle = {
        "vectors": {
            str(i): torch.tensor(vectors[i], dtype=torch.float32)
            for i in range(num_layers)
        },
        "evr": evr_list,
        "best_layer": best_layer,
        "best_evr": best_evr,
        "recommended_window": window_layers,
        "recommended_window_mean_evr": window_mean,
        "top_k_layers": top_k,
        "num_extraction_items": len(items),
        "model_path": args.model_path,
        "enable_thinking": bool(args.enable_thinking),
        "hidden_size": hidden,
        "num_layers": num_layers,
        "seed": args.seed,
        "source_set": str(extraction_path),
    }
    bundle_path = output_dir / "sycophancy_vectors_bundle.pt"
    torch.save(bundle, bundle_path)

    summary = {
        "best_layer": best_layer,
        "best_evr": round(best_evr, 6),
        "recommended_window": window_layers,
        "recommended_window_mean_evr": round(window_mean, 6),
        "top_k_layers_by_evr": [
            {"layer": i, "evr": round(evr_list[i], 6)} for i in top_k
        ],
        "enable_thinking": bool(args.enable_thinking),
        "n_items": len(items),
        "model_path": args.model_path,
        "files": {
            "evr_csv": str(evr_csv),
            "best_pt": str(best_path),
            "bundle_pt": str(bundle_path),
            "per_layer_pt_pattern": str(output_dir / "sycophancy_vector_layer{L}.pt"),
        },
        "phase3_hint": (
            f"python step3_run_intervention_test.py "
            f"--vector-dir {output_dir} "
            f"--test-set <repeng_root>/03_splits/test_set_mmlu.json "
            f"--layers {','.join(map(str, window_layers))} "
            f"--alpha 30.0 --decoding greedy --no-enable-thinking "
            f"--intervene-from-round 0"
        ),
    }
    save_json(output_dir / "profiling_summary.json", summary)

    print("\n" + "=" * 60)
    print("Phase 2 完成 —— 逐层剖析与向量已持久化")
    print(f"  Best layer     : {best_layer} (EVR={best_evr:.4f})")
    print(f"  推荐洗涤窗口   : {window_layers} (mean EVR={window_mean:.4f})")
    print(f"  Top-{args.top_k_layers} layers  : {top_k}")
    print(f"  EVR CSV        : {evr_csv}")
    print(f"  Best vector    : {best_path}")
    print(f"  Bundle         : {bundle_path}")
    print("=" * 60)
    print("Phase3 提示:", summary["phase3_hint"])


if __name__ == "__main__":
    main()
