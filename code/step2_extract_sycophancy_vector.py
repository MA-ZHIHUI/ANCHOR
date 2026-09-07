# step2_extract_sycophancy_vector.py
"""
表征工程 - 步骤 2：逐层分析并提取「谄媚向量」。

对提取集中的每道题构造一对对比样本（Contrastive Pair）：
  - Truthful   ：客观原始问题（模型即将客观作答的状态）
  - Sycophantic：客观问题 + 模拟模型先给出正确答案 + 用户施加最高级别诱导（strong_vanity）

分别前向（output_hidden_states=True）取「读完 prompt 最后一个 token」的各层激活，
计算差值 ΔH = H_syc - H_truth。对每个解码器层，把 40 道题的 ΔH 堆成矩阵做 PCA(n_components=1)，
用 explained_variance_ratio_[0] 衡量该层谄媚方向的「一致性」，选出最显著的一层并保存其第一主成分。

层索引说明（关键）：
  output_hidden_states 返回 num_layers+1 个张量，hidden_states[0] 是 embedding 输出，
  hidden_states[i] (i>=1) 是「解码器层 i-1」的输出。SycophancyIntervention 的钩子挂在
  model.model.layers[layer_idx] 上、修改其输出，对应 hidden_states[layer_idx+1]。
  因此本脚本按「解码器层 L -> hidden_states[L+1]」遍历，保存的 layer 即可直接用于步骤 3 的钩子。

用法:
  python step2_extract_sycophancy_vector.py
  python step2_extract_sycophancy_vector.py --extraction-set <...> --output <...>
"""

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

import config
from backend import LocalModel
from model_adapt import get_hidden_size, get_num_hidden_layers
from prompts import (
    build_drift_followup_flow,
    build_single_round_prompt,
    with_answer_suffix,
)

DEFAULT_EXTRACTION_SET = "flipped_data/mmlu_dev/Qwen3-8B_single_think/extraction_set.json"

MODEL_PATH=os.environ.get("ANCHOR_MODEL_PATH", "")
ENABLE_THINK_MODE=True


def load_items(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_contrastive_messages(
    item: dict, *, enable_thinking: bool
) -> tuple[list[dict], list[dict]]:
    """构造对比对：Truthful（客观问题） vs Sycophantic（答对后遭最高级别施压）。"""
    question_prompt = build_single_round_prompt(
        item, "baseline", enable_thinking=enable_thinking
    )
    truthful = [{"role": "user", "content": question_prompt}]

    # 模拟模型已给出正确答案，随后用户施加第 5 级（strong_vanity）高压诱导
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
    """返回读完 prompt 最后一个 token 时、每个解码器层的激活向量（长度 = num_layers）。"""
    prompt = model.build_prompt(messages, enable_thinking=enable_thinking)
    inputs = model.tokenizer(prompt, return_tensors="pt").to(model.model.device)
    out = model.model(**inputs, output_hidden_states=True)
    hs = out.hidden_states  # tuple(num_layers+1)，hs[0]=embedding
    # 解码器层 L 的输出 = hs[L+1]，取最后一个 token
    return [hs[i][0, -1, :].float().cpu().numpy() for i in range(1, len(hs))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="表征工程步骤2：逐层PCA提取谄媚向量")
    parser.add_argument(
        "--extraction-set", type=Path, default=DEFAULT_EXTRACTION_SET, help="提取集 JSON"
    )
    parser.add_argument("--model-path", default=MODEL_PATH, help="本地模型权重目录")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="向量输出 .pt 路径（默认：提取集同目录 sycophancy_vector_best_layer.pt）",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_THINK_MODE,
        help="构造 prompt 时是否走 think 模板",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅用前 N 题（调试）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extraction_path = args.extraction_set.resolve()
    if not extraction_path.exists():
        raise FileNotFoundError(f"未找到提取集: {extraction_path}\n请先运行 step1_split_dataset.py。")

    items = load_items(extraction_path)
    if args.limit:
        items = items[: args.limit]
    output_path = (
        args.output.resolve()
        if args.output
        else extraction_path.parent / "sycophancy_vector_best_layer.pt"
    )

    config.set_seed()  # 须在 LocalModel / 首次 CUDA 之前
    model = LocalModel(args.model_path)
    num_layers = get_num_hidden_layers(model.model)
    print(f"提取集: {extraction_path} ({len(items)} 题) | 模型层数: {num_layers}")

    # 逐题累积各层 ΔH
    diffs_per_layer: list[list[np.ndarray]] = [[] for _ in range(num_layers)]
    for idx, item in enumerate(items):
        truthful, sycophantic = build_contrastive_messages(
            item, enable_thinking=args.enable_thinking
        )
        h_truth = last_token_hidden_by_layer(model, truthful, enable_thinking=args.enable_thinking)
        h_syc = last_token_hidden_by_layer(model, sycophantic, enable_thinking=args.enable_thinking)
        for layer in range(num_layers):
            diffs_per_layer[layer].append(h_syc[layer] - h_truth[layer])
        print(f"  [{idx + 1}/{len(items)}] 已提取对比激活", end="\r")
    print()

    # 逐层 PCA，选出谄媚方向最集中的一层
    print("\n[逐层 PCA] explained_variance_ratio_[0]:")
    best_layer = -1
    best_ratio = -1.0
    best_vector: np.ndarray | None = None
    for layer in range(num_layers):
        matrix = np.stack(diffs_per_layer[layer], axis=0)  # [n_items, hidden]
        pca = PCA(n_components=1)
        pca.fit(matrix)
        ratio = float(pca.explained_variance_ratio_[0])
        component = pca.components_[0]

        # 符号对齐：让向量统一指向「更谄媚」方向（与平均 ΔH 同向），
        # 这样步骤3 中「减去 alpha*vector」才是正确的去谄媚方向。
        mean_diff = matrix.mean(axis=0)
        if float(np.dot(component, mean_diff)) < 0:
            component = -component

        print(f"  Layer {layer:>2d}: {ratio:.4f}")
        if ratio > best_ratio:
            best_ratio = ratio
            best_layer = layer
            best_vector = component

    assert best_vector is not None
    print(f"\n>>> Best Layer = {best_layer} | explained_variance_ratio = {best_ratio:.4f}")

    payload = {
        "layer": best_layer,
        "vector": torch.tensor(best_vector, dtype=torch.float32),
        "explained_variance_ratio": best_ratio,
        "num_extraction_items": len(items),
        "model_path": args.model_path,
        "enable_thinking": bool(args.enable_thinking),
        "hidden_size": get_hidden_size(model.model),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"已保存谄媚向量: {output_path}")
    print(f"  layer={best_layer}, vector.shape={tuple(payload['vector'].shape)}")


if __name__ == "__main__":
    main()
