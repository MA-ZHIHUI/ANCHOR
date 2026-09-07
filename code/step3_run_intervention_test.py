# step3_run_intervention_test.py
"""
表征工程 - 步骤 3：干预效果验证（批量 Baseline vs Intervention）。

流程:
  1. 加载步骤 2 保存的谄媚向量 .pt（含 best layer 与向量），默认沿用提取时的 model_path。
  2. 遍历 test_set.json 中的每道题，执行 5 轮多轮高压诱导（与 run_drift_multi 一致）。
  3. 每题跑两遍：
       - Baseline    ：不干预；
       - Intervention：默认从 Round0 起全程用 SycophancyIntervention 钩子，
                       在一层或多层上同时扣除 alpha * 谄媚向量（多层联合干预）。
                       Round0 含干预用于检验「无诱导时血清是否扰动正常作答」；
                       可用 --intervene-from-round 1 复现旧行为（仅诱导轮）。
  4. 输出与 results/ 对齐的详细 CSV（便于调试与论文制表）：
       detail.csv
       item_summary_{baseline,intervention}.csv
       overall_summary_{baseline,intervention}.csv
       summary_by_round_name_{baseline,intervention}.csv
       summary_by_subject_{baseline,intervention}.csv
       summary_by_round1_group_{baseline,intervention}.csv
       comparison_by_round.csv   # 两臂逐轮正确率对比
       run_config.json           # 运行超参存档

用法:
  python step3_run_intervention_test.py
  python step3_run_intervention_test.py --alpha 5.0 --decoding greedy --limit 10
  python step3_run_intervention_test.py --layers 18,19,20 --alpha 3.0 --no-enable-thinking

  # 推荐：同一测集先冻结一次 baseline，再扫 α / 层（避免 GPU 非完全确定性导致 baseline 漂移）
  python step3_run_intervention_test.py ... --alpha 50 --output-dir ../results/.../REF_BASELINE
  python step3_run_intervention_test.py ... --alpha 30 \\
    --reuse-baseline-from ../results/.../REF_BASELINE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import config
from backend import LocalModel
from model_adapt import get_decoder_layers, get_hidden_size, get_num_hidden_layers
from metrics import (
    build_item_summary,
    compute_overall_summary,
    compute_round1_group_summary,
    compute_subject_summary,
    compute_summary_by_round_name,
    step_row_to_detail,
)
from prompts import (
    DEFENSE_SYSTEM_PROMPT,
    DRIFT_ROUND_NAMES,
    build_drift_followup_flow,
    build_single_round_prompt,
    with_answer_suffix,
)

DEFAULT_DIR = "flipped_data/mmlu_dev/Qwen3-8B_single_think"
DEFAULT_VECTOR = "flipped_data/mmlu_dev/Qwen3-8B_single_think/sycophancy_vector_best_layer.pt"
DEFAULT_TEST_SET = "flipped_data/mmlu_dev/Qwen3-8B_single_think/test_set.json"
DEFAULT_ALPHA = 3.0
ARMS = ("baseline", "intervention")


# ===================== 表征干预钩子：多层联合干预（Multi-layer Intervention） =====================
class SycophancyIntervention:
    """多层联合干预上下文管理器。

    在指定的一层或多层 Transformer block 输出的最后一个 token 上，
    扣除 alpha * 谄媚向量。

    vectors 支持两种形式:
      - 单个 Tensor: 所有层共用同一向量（旧行为，不推荐跨层）；
      - dict[int, Tensor]: 每层使用该层自己的 PCA 向量（推荐）。

    兼容：tuple / tensor 返回值；动态对齐 device 与 dtype；3D [B,S,H] 与 2D [B*S,H] 维度。
    """

    def __init__(
        self,
        model,
        layer_indices: list[int],
        syc_vector: torch.Tensor | dict[int, torch.Tensor],
        alpha: float,
        *,
        alpha_mode: str = "fixed",
    ):
        self.model = model  # HF CausalLM（可访问 .model.layers）
        self.layer_indices = list(layer_indices)
        self.alpha = alpha
        self.alpha_mode = alpha_mode
        self.handles: list = []
        # 统一成 layer -> vector
        if isinstance(syc_vector, dict):
            self.vectors = {int(k): v.float() for k, v in syc_vector.items()}
        else:
            shared = syc_vector.float()
            self.vectors = {int(l): shared for l in self.layer_indices}

    def __enter__(self):
        self.handles = []
        for layer_idx in self.layer_indices:
            if layer_idx not in self.vectors:
                raise KeyError(f"缺少层 {layer_idx} 的谄媚向量")
            vector_cpu = self.vectors[layer_idx]

            def make_hook(vec_ref: torch.Tensor):
                def hook_fn(module, inputs, output):
                    is_tuple = isinstance(output, tuple)
                    hidden_states = output[0] if is_tuple else output
                    vector = vec_ref.to(
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
                    if hidden_states.dim() == 3:  # [batch, seq, hidden]
                        h = hidden_states[:, -1, :]
                        scale = (
                            self.alpha * h.norm(dim=-1, keepdim=True)
                            if self.alpha_mode == "relative"
                            else self.alpha
                        )
                        hidden_states[:, -1, :] = h - scale * vector
                    elif hidden_states.dim() == 2:  # [batch*seq, hidden]
                        h = hidden_states[-1, :]
                        scale = (
                            self.alpha * h.norm()
                            if self.alpha_mode == "relative"
                            else self.alpha
                        )
                        hidden_states[-1, :] = h - scale * vector
                    if is_tuple:
                        return (hidden_states,) + output[1:]
                    return hidden_states

                return hook_fn

            layer_module = get_decoder_layers(self.model)[layer_idx]
            self.handles.append(
                layer_module.register_forward_hook(make_hook(vector_cpu))
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def load_layer_vectors(
    *,
    layers: list[int],
    vector_path: Path | None,
    vector_dir: Path | None,
    bundle_path: Path | None,
) -> tuple[dict[int, torch.Tensor], dict]:
    """
    按层加载向量。优先级:
      1) --vector-dir 下 sycophancy_vector_layer{L}.pt
      2) --bundle 中的 vectors
      3) 回退 --vector 单文件（所有层共用；若 layer 字段与目标层不一致会告警）
    """
    meta: dict = {"mode": None, "warnings": []}
    vectors: dict[int, torch.Tensor] = {}

    if vector_dir is not None:
        vector_dir = vector_dir.resolve()
        for l in layers:
            p = vector_dir / f"sycophancy_vector_layer{l}.pt"
            if not p.exists():
                raise FileNotFoundError(f"未找到分层向量: {p}")
            payload = torch.load(p, map_location="cpu", weights_only=False)
            vectors[l] = payload["vector"].float()
        meta["mode"] = "per_layer_dir"
        meta["vector_dir"] = str(vector_dir)
        return vectors, meta

    if bundle_path is not None:
        bundle_path = bundle_path.resolve()
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        raw = bundle["vectors"]
        for l in layers:
            key = str(l)
            if key not in raw and l not in raw:
                raise KeyError(f"bundle 中缺少层 {l}")
            vectors[l] = (raw[key] if key in raw else raw[l]).float()
        meta["mode"] = "bundle"
        meta["bundle"] = str(bundle_path)
        return vectors, meta

    if vector_path is None:
        raise ValueError("必须提供 --vector / --vector-dir / --bundle 之一")

    vector_path = vector_path.resolve()
    payload = torch.load(vector_path, map_location="cpu", weights_only=False)
    shared = payload["vector"].float()
    src_layer = payload.get("layer")
    for l in layers:
        vectors[l] = shared
        if src_layer is not None and int(src_layer) != int(l):
            meta["warnings"].append(
                f"用层{src_layer}的向量干预层{l}（跨层共用，可能无效；"
                f"建议改用 --vector-dir 或 --bundle）"
            )
    meta["mode"] = "shared_single_file"
    meta["vector"] = str(vector_path)
    meta["source_layer"] = src_layer
    return vectors, meta


# ===================== 数据与上下文 =====================
def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_context_limit(model: LocalModel) -> int:
    if config.CONTEXT_TOKEN_LIMIT is not None:
        return config.CONTEXT_TOKEN_LIMIT
    max_pos = getattr(model.model.config, "max_position_embeddings", 32768)
    return max(1024, max_pos - config.MAX_NEW_TOKENS_THINK - 256)


def model_short_name(model_path: str) -> str:
    return Path(model_path).name or config.MODEL_NAME


# ===================== 单题多轮实验（可选干预） =====================
def run_multi_round(
    model: LocalModel,
    item: dict,
    *,
    item_index: int,
    model_name: str,
    arm: str,
    enable_thinking: bool,
    context_limit: int,
    sampling: dict | None = None,
    intervention: dict | None = None,
    system_prompt: str | None = None,
) -> list[dict]:
    """
    对单题跑 5 轮多轮诱导。intervention 为 None 时为 Baseline；
    否则为 {vectors|vector, layers, alpha, intervene_from_round}：
      在 round_idx >= intervene_from_round 的轮次做多层联合干预。
      默认 intervene_from_round=0（含无诱导的 Round0，检验血清副作用）。
    system_prompt 非空时在对话最前插入 role=system（用于提示词防御臂）。

    返回与 metrics.step_row_to_detail 对齐的 detail 行（额外含 arm / is_correct 字段）。
    """
    gt = item["answer"]
    misleading = item["misleading"]
    subject = item.get("subject", "")
    choices = list(item["choices"].keys())
    flow = build_drift_followup_flow(item)
    intervene_from = 0
    if intervention is not None:
        intervene_from = int(intervention.get("intervene_from_round", 0))

    history: list[dict] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})
    history.append(
        {
            "role": "user",
            "content": build_single_round_prompt(
                item, "baseline", enable_thinking=enable_thinking
            ),
        }
    )
    rows: list[dict] = []
    is_success = True
    error_msg = None

    for round_idx, user_input in enumerate(flow):
        round_name = DRIFT_ROUND_NAMES[round_idx]
        if round_idx > 0 and user_input is not None:
            history.append(
                {"role": "user", "content": with_answer_suffix(user_input, enable_thinking)}
            )

        # 上下文溢出保护
        if model.count_prompt_tokens(history, enable_thinking=enable_thinking) > context_limit:
            is_success = False
            error_msg = "context_overflow"
            break

        use_hook = intervention is not None and round_idx >= intervene_from
        if use_hook:
            # 多层联合干预：优先使用 per-layer vectors
            vec_arg = intervention.get("vectors", intervention.get("vector"))
            with SycophancyIntervention(
                model.model,
                intervention["layers"],
                vec_arg,
                intervention["alpha"],
                alpha_mode=intervention.get("alpha_mode", "fixed"),
            ):
                step = model.generate_with_belief(
                    history, choices=choices, enable_thinking=enable_thinking, sampling=sampling
                )
        else:
            step = model.generate_with_belief(
                history, choices=choices, enable_thinking=enable_thinking, sampling=sampling
            )

        history.append({"role": "assistant", "content": step["content"]})
        row = step_row_to_detail(
            item_index=item_index,
            subject=subject,
            round_idx=round_idx,
            round_name=round_name,
            gt=gt,
            misleading=misleading,
            step=step,
            model=model_name,
            experiment_mode="multi_round_repeng",
        )
        row["arm"] = arm
        row["is_correct"] = int(step["pred"] == gt)
        row["intervened"] = int(use_hook)
        rows.append(row)

    if not is_success:
        for r in rows:
            r["is_valid_experiment"] = False
            r["error_tag"] = error_msg

    return rows


# ===================== 汇总与落盘 =====================
def compute_arm_tables(detail_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """对单个 arm 的 detail 复用 metrics 流水线，产出与 results/ 同结构的表。"""
    if detail_df.empty:
        empty = pd.DataFrame()
        return {
            "item_summary": empty,
            "overall_summary": empty,
            "summary_by_round_name": empty,
            "summary_by_subject": empty,
            "summary_by_round1_group": empty,
        }

    item_summaries = [
        build_item_summary(detail_df[detail_df["item_index"] == idx].to_dict("records"))
        for idx in detail_df["item_index"].unique()
    ]
    item_df = pd.DataFrame(item_summaries)
    return {
        "item_summary": item_df,
        "overall_summary": compute_overall_summary(item_df, detail_df),
        "summary_by_round_name": compute_summary_by_round_name(detail_df),
        "summary_by_subject": compute_subject_summary(item_df),
        "summary_by_round1_group": compute_round1_group_summary(item_df),
    }


def build_comparison_by_round(detail_df: pd.DataFrame) -> pd.DataFrame:
    """两臂逐轮正确率对比表（论文主结果）。"""
    rows = []
    for r, name in enumerate(DRIFT_ROUND_NAMES):
        row = {"round": r, "condition": name}
        for arm in ARMS:
            sub = detail_df[
                (detail_df["arm"] == arm)
                & (detail_df["round"] == r)
                & (detail_df["is_valid_experiment"] == True)  # noqa: E712
            ]
            n = len(sub)
            acc = float(sub["is_correct"].mean() * 100) if n else 0.0
            flip = float(sub["is_flipped"].mean() * 100) if n else 0.0
            belief_gt = float(sub["belief_gt"].mean() * 100) if n else 0.0
            belief_mis = float(sub["belief_misleading"].mean() * 100) if n else 0.0
            row[f"{arm}_n"] = n
            row[f"{arm}_acc(%)"] = round(acc, 2)
            row[f"{arm}_flip_rate(%)"] = round(flip, 2)
            row[f"{arm}_belief_gt(%)"] = round(belief_gt, 2)
            row[f"{arm}_belief_misleading(%)"] = round(belief_mis, 2)
        row["delta_acc(%)"] = round(
            row["intervention_acc(%)"] - row["baseline_acc(%)"], 2
        )
        row["delta_flip(%)"] = round(
            row["intervention_flip_rate(%)"] - row["baseline_flip_rate(%)"], 2
        )
        rows.append(row)
    return pd.DataFrame(rows)


def save_arm_tables(output_dir: Path, arm: str, tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        path = output_dir / f"{name}_{arm}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="表征工程步骤3：干预效果验证")
    parser.add_argument(
        "--vector",
        type=Path,
        default=None,
        help="单文件向量 .pt（所有层共用；跨层时不推荐）",
    )
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=None,
        help="分层向量目录（内含 sycophancy_vector_layer{L}.pt，推荐）",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="sycophancy_vectors_bundle.pt（按层取向量）",
    )
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET, help="测试集 JSON")
    parser.add_argument(
        "--model-path",
        default=None,
        help="本地模型权重目录（默认沿用向量 .pt 中记录的 model_path，保证与提取一致）",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="干预强度；fixed 模式下为绝对 α；relative 模式下为 c（α_real = c·||H||）",
    )
    parser.add_argument(
        "--alpha-mode",
        choices=["fixed", "relative"],
        default="fixed",
        help="fixed: 减 α·v；relative: 减 (c·||H||)·v，c=--alpha，||v||≈1",
    )
    parser.add_argument(
        "--intervene-from-round",
        type=int,
        default=0,
        help="intervention 臂从该轮起挂钩子（默认 0=含 Round0，检验无诱导副作用；"
        "设为 1 则仅诱导轮干预，复现旧逻辑）",
    )
    parser.add_argument(
        "--device",
        default=config.DEVICE,
        help='GPU，如 "cuda:1" 或 1；默认 config.DEVICE',
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="多层联合干预的层号，逗号分隔，如 '18,19,20'；默认仅用 .pt 中记录的单层",
    )
    parser.add_argument(
        "--decoding",
        choices=["greedy", "sampling"],
        default="greedy",
        help="解码方式：greedy(默认，确定性，受控A/B对比推荐) 或 sampling(官方采样，含随机噪声)",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=config.ENABLE_THINK_MODE,
        help="开启/关闭 think 模式",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅测试前 N 题（调试）")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果输出目录（默认 results/<dataset>/<model>_repeng_layer{L}_alpha{A}_{tag}_{decoding}/）",
    )
    parser.add_argument(
        "--dataset-tag",
        type=str,
        default=None,
        help='results 下数据集子目录名，如 mmlu_dev / commonsenseqa_dev；'
        "默认从 --test-set 路径自动推断",
    )
    parser.add_argument(
        "--reuse-baseline-from",
        type=Path,
        default=None,
        help="复用某次运行目录中的 baseline 行（读其 detail.csv 的 arm=baseline），"
        "本次只跑 intervention。要求同一 --test-set / 题数 / 轮次协议。"
        "用于扫 α、扫层时冻结共享 baseline，消除跨进程 GPU 抖动。",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="只跑 baseline 并写出汇总（不跑 intervention）。"
        "适合先冻结一份 REF baseline，再给后续 --reuse-baseline-from 使用。",
    )
    parser.add_argument(
        "--system-prompt-defense",
        action="store_true",
        help="在对话最前插入中文防迎合 system prompt（prompts.DEFENSE_SYSTEM_PROMPT）。"
        "须与 --baseline-only 联用：不挂钩子，arm 记为 prompt_defense。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="覆盖生成长度上限（默认用 config 分模式预算，通常 4096）。"
        "扫层时建议设 1536，避免浅层大 α 退化为超长乱码拖满预算。",
    )
    return parser.parse_args()


def load_reused_baseline_rows(
    reuse_dir: Path,
    *,
    n_items: int,
    test_set: Path,
) -> list[dict]:
    """从既有 run 目录加载 baseline 明细行，并做基本一致性检查。"""
    reuse_dir = reuse_dir.resolve()
    detail_path = reuse_dir / "detail.csv"
    if not detail_path.exists():
        raise FileNotFoundError(f"--reuse-baseline-from 缺少 detail.csv: {detail_path}")

    detail = pd.read_csv(detail_path)
    if "arm" not in detail.columns:
        raise ValueError(f"{detail_path} 无 arm 列，无法复用 baseline")
    base = detail[detail["arm"].astype(str) == "baseline"].copy()
    if base.empty:
        raise ValueError(f"{detail_path} 中没有 arm=baseline 的行")

    n_base_items = int(base["item_index"].nunique())
    if n_base_items != n_items:
        raise ValueError(
            f"复用 baseline 题数={n_base_items} 与当前 test-set 题数={n_items} 不一致。"
            f" reuse={reuse_dir}"
        )

    cfg_path = reuse_dir / "run_config.json"
    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as f:
            prev = json.load(f)
        prev_test = str(Path(prev.get("test_set", "")).resolve()) if prev.get("test_set") else ""
        cur_test = str(test_set.resolve())
        if prev_test and prev_test != cur_test:
            raise ValueError(
                "复用 baseline 的 test_set 与当前不一致：\n"
                f"  prev: {prev_test}\n"
                f"  curr: {cur_test}\n"
                "请仅在同一测集上复用。"
            )
        if prev.get("decoding") and prev["decoding"] != "greedy":
            print(
                f"⚠️ 复用目录 decoding={prev['decoding']!r}；"
                "建议只复用 greedy baseline。"
            )

    # 保证后续 to_dict 字段干净
    rows = base.to_dict(orient="records")
    print(
        f"✅ 复用 baseline: {reuse_dir} | rows={len(rows)} | "
        f"items={n_base_items} | rounds/item≈{len(rows) / max(n_base_items, 1):.1f}"
    )
    return rows


def parse_layers_arg(layers_str: str | None, default_layer: int) -> list[int]:
    """解析 --layers；未传入时回退为 .pt 中的单层，保证向后兼容。"""
    if layers_str is None or not str(layers_str).strip():
        return [int(default_layer)]
    layers = [int(x.strip()) for x in str(layers_str).split(",") if x.strip()]
    if not layers:
        raise ValueError(f"--layers 解析结果为空: {layers_str!r}")
    return layers


def layers_tag(layers: list[int]) -> str:
    """多层层号拼接，用于输出目录命名，如 18_19_20。"""
    return "_".join(str(l) for l in layers)


def infer_dataset_tag(test_set: Path) -> str:
    """
    从 --test-set 路径推断 results/<dataset>/ 子目录名。
    与 p0 DOMAINS / dataset 目录对齐：mmlu_dev、commonsenseqa_dev。
    """
    s = str(test_set.resolve()).replace("\\", "/").lower()
    # CSQA 优先（路径里常同时出现 multi_nothink 等无关片段）
    csqa_markers = (
        "/test_csqa/",
        "test_set_csqa",
        "csqa_deep_flip",
        "commonsenseqa",
        "/csqa/",
        "_csqa.",
        "_csqa/",
    )
    if any(m in s for m in csqa_markers) or s.rstrip("/").endswith("test_csqa"):
        return "commonsenseqa_dev"

    mmlu_markers = (
        "/test_mmlu/",
        "test_set_mmlu",
        "extraction_mmlu",
        "extraction_set_mmlu",
        "mmlu_deep_flip",
        "/mmlu_dev/",
        "/mmlu/",
        "_mmlu.",
        "_mmlu/",
    )
    if any(m in s for m in mmlu_markers) or s.rstrip("/").endswith("test_mmlu"):
        return "mmlu_dev"

    # 回退：用测试集文件名/父目录名中的关键词
    name = test_set.name.lower()
    parent = test_set.parent.name.lower()
    blob = f"{parent}/{name}"
    if "csqa" in blob or "commonsense" in blob:
        return "commonsenseqa_dev"
    if "mmlu" in blob:
        return "mmlu_dev"

    print(
        f"⚠️ 无法从 test-set 推断数据集标签，回退 mmlu_dev。"
        f" 可用 --dataset-tag 显式指定。path={test_set}"
    )
    return "mmlu_dev"


def default_output_dir(
    *,
    model_name: str,
    layers: list[int],
    alpha: float,
    enable_thinking: bool,
    decoding: str,
    dataset_tag: str,
    mode_suffix: str = "",
    intervene_from_round: int = 0,
) -> Path:
    tag = "think" if enable_thinking else "nothink"
    # 例: ..._alpha30.0_nothink_greedy_fromR0_perlayer
    # fromR{k}: 标明干预起始轮，避免与旧「仅诱导轮」结果目录混淆
    run_name = (
        f"{model_name}_repeng_layer{layers_tag(layers)}_alpha{alpha}_{tag}_{decoding}"
        f"_fromR{intervene_from_round}{mode_suffix}"
    )
    return config.RESULTS_DIR / dataset_tag / run_name


def main() -> None:
    args = parse_args()

    # 默认向量：若未指定任何向量源，回退旧 DEFAULT_VECTOR
    if args.vector is None and args.vector_dir is None and args.bundle is None:
        args.vector = Path(DEFAULT_VECTOR)

    # 先解析层号：若只有单文件且未给 --layers，用文件内 layer
    default_layer = 0
    probe_path = args.vector or args.bundle
    if probe_path is not None and Path(probe_path).exists():
        probe = torch.load(Path(probe_path).resolve(), map_location="cpu", weights_only=False)
        if "layer" in probe:
            default_layer = int(probe["layer"])
        elif "best_layer" in probe:
            default_layer = int(probe["best_layer"])
    layers = parse_layers_arg(args.layers, default_layer)

    vectors, vec_meta = load_layer_vectors(
        layers=layers,
        vector_path=args.vector,
        vector_dir=args.vector_dir,
        bundle_path=args.bundle,
    )
    for w in vec_meta.get("warnings", []):
        print("⚠️", w)
    print(
        f"加载谄媚向量: mode={vec_meta['mode']} | intervene_layers={layers} | "
        f"n_vectors={len(vectors)}"
    )

    # 模型路径：优先 CLI，否则从任一向量文件读
    model_path = args.model_path
    if model_path is None:
        src = args.vector or args.bundle
        if src is None and args.vector_dir is not None:
            src = args.vector_dir / f"sycophancy_vector_layer{layers[0]}.pt"
        if src is not None and Path(src).exists():
            payload = torch.load(Path(src).resolve(), map_location="cpu", weights_only=False)
            model_path = payload.get("model_path", config.MODEL_PATH)
        else:
            model_path = config.MODEL_PATH
    model_name = model_short_name(model_path)

    test_items = load_json(args.test_set.resolve())
    if args.limit:
        test_items = test_items[: args.limit]

    # 种子必须在模型加载 / 首次 CUDA 前设置（CUBLAS_WORKSPACE 也依赖这一点）
    config.set_seed()
    model = LocalModel(
        model_path,
        device=config.normalize_device(args.device),
        max_new_tokens=args.max_new_tokens,
    )
    if args.max_new_tokens is not None:
        print(f"⚠️ max_new_tokens override = {args.max_new_tokens}")

    # 维度一致性校验（Gemma4 等需走 text_config，见 model_adapt.get_hidden_size）
    hidden = get_hidden_size(model.model)
    for l, v in vectors.items():
        if v.shape[0] != hidden:
            raise ValueError(
                f"层{l}向量维度({v.shape[0]}) 与 hidden_size({hidden}) 不一致"
            )

    # 层号合法性校验（多层联合干预）
    n_layers = get_num_hidden_layers(model.model)
    for l in layers:
        if l < 0 or l >= n_layers:
            raise ValueError(f"层号 {l} 越界，模型共有 {n_layers} 层（合法范围 0..{n_layers - 1}）")

    context_limit = resolve_context_limit(model)
    intervene_from_round = int(args.intervene_from_round)
    if intervene_from_round < 0:
        raise ValueError(f"--intervene-from-round 不能为负: {intervene_from_round}")
    intervention_cfg = {
        "vectors": vectors,
        "layers": layers,
        "alpha": args.alpha,
        "alpha_mode": args.alpha_mode,
        "intervene_from_round": intervene_from_round,
    }

    # greedy：确定性解码，使 baseline 与 intervention 在「未挂钩子」的轮次逐 token 可比；
    # 默认从 Round0 起干预时，R0 的 delta 反映血清对无诱导作答的影响（不再恒为 0）。
    sampling = config.GREEDY_DECODING if args.decoding == "greedy" else None

    # 输出目录名区分 per-layer / shared，并按测试集写入对应 dataset 子目录
    out_tag_suffix = ""
    if vec_meta["mode"] == "per_layer_dir":
        out_tag_suffix = "_perlayer"
    elif vec_meta["mode"] == "bundle":
        out_tag_suffix = "_bundle"
    dataset_tag = (args.dataset_tag or infer_dataset_tag(args.test_set)).strip()
    if not dataset_tag:
        raise ValueError("--dataset-tag 不能为空")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else default_output_dir(
            model_name=model_name,
            layers=layers,
            alpha=args.alpha,
            enable_thinking=args.enable_thinking,
            decoding=args.decoding,
            dataset_tag=dataset_tag,
            mode_suffix=out_tag_suffix,
            intervene_from_round=intervene_from_round,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"测试集: {args.test_set} ({len(test_items)} 题) | 模型: {model_path}\n"
        f"多层联合干预层: {layers} | alpha: {args.alpha} ({args.alpha_mode}) | "
        f"intervene_from_round: {intervene_from_round} | "
        f"Think: {'ON' if args.enable_thinking else 'OFF'} | 解码: {args.decoding}\n"
        f"输出目录: {output_dir}"
    )
    if args.baseline_only and args.reuse_baseline_from is not None:
        raise ValueError("--baseline-only 与 --reuse-baseline-from 不能同时使用")
    if args.system_prompt_defense and not args.baseline_only:
        raise ValueError(
            "--system-prompt-defense 须与 --baseline-only 联用（纯提示词防御，不挂钩子）"
        )
    if args.system_prompt_defense and args.reuse_baseline_from is not None:
        raise ValueError("--system-prompt-defense 不能与 --reuse-baseline-from 联用")

    all_detail: list[dict] = []
    reused_baseline: list[dict] | None = None
    if args.reuse_baseline_from is not None:
        reused_baseline = load_reused_baseline_rows(
            args.reuse_baseline_from,
            n_items=len(test_items),
            test_set=args.test_set.resolve(),
        )
        all_detail.extend(reused_baseline)

    defense_system = DEFENSE_SYSTEM_PROMPT if args.system_prompt_defense else None
    no_hook_arm = "prompt_defense" if args.system_prompt_defense else "baseline"

    for item_index, item in enumerate(tqdm(test_items, desc="intervention_test")):
        if reused_baseline is None:
            base_rows = run_multi_round(
                model,
                item,
                item_index=item_index,
                model_name=model_name,
                arm=no_hook_arm,
                enable_thinking=args.enable_thinking,
                context_limit=context_limit,
                sampling=sampling,
                intervention=None,
                system_prompt=defense_system,
            )
            all_detail.extend(base_rows)

        if not args.baseline_only:
            interv_rows = run_multi_round(
                model,
                item,
                item_index=item_index,
                model_name=model_name,
                arm="intervention",
                enable_thinking=args.enable_thinking,
                context_limit=context_limit,
                sampling=sampling,
                intervention=intervention_cfg,
                system_prompt=None,
            )
            all_detail.extend(interv_rows)

    detail_df = pd.DataFrame(all_detail)
    detail_df.to_csv(output_dir / "detail.csv", index=False, encoding="utf-8-sig")

    # 分臂产出与 results/ 对齐的汇总表
    arms_to_save = (no_hook_arm,) if args.baseline_only else ARMS
    for arm in arms_to_save:
        arm_df = detail_df[detail_df["arm"] == arm].copy()
        tables = compute_arm_tables(arm_df)
        save_arm_tables(output_dir, arm, tables)

    if not args.baseline_only:
        comparison = build_comparison_by_round(detail_df)
        comparison.to_csv(
            output_dir / "comparison_by_round.csv", index=False, encoding="utf-8-sig"
        )
    else:
        comparison = None
        # baseline-only / prompt-defense：写出该臂逐轮正确率
        base_only = (
            detail_df[detail_df["arm"] == no_hook_arm]
            .groupby("round", as_index=False)
            .agg(
                **{
                    f"{no_hook_arm}_acc(%)": (
                        "is_correct",
                        lambda s: round(100.0 * s.mean(), 2),
                    )
                }
            )
        )
        base_only.to_csv(
            output_dir / "comparison_by_round.csv", index=False, encoding="utf-8-sig"
        )

    # 运行配置存档（便于复现与调试）
    run_config = {
        "vector_meta": vec_meta,
        "test_set": str(args.test_set.resolve()),
        "dataset_tag": dataset_tag,
        "model_path": model_path,
        "model_name": model_name,
        "layers": layers,
        "alpha": args.alpha,
        "alpha_mode": args.alpha_mode,
        "intervene_from_round": intervene_from_round,
        "enable_thinking": bool(args.enable_thinking),
        "decoding": args.decoding,
        "device": config.normalize_device(args.device),
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "n_items": len(test_items),
        "output_dir": str(output_dir),
        "baseline_only": bool(args.baseline_only),
        "system_prompt_defense": bool(args.system_prompt_defense),
        "defense_system_prompt": DEFENSE_SYSTEM_PROMPT if args.system_prompt_defense else None,
        "reuse_baseline_from": (
            str(args.reuse_baseline_from.resolve())
            if args.reuse_baseline_from is not None
            else None
        ),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=4)

    print("\n" + "█" * 70)
    print(f"📊 ANCHOR 表征干预验证汇总 (Think Mode: {'ON' if args.enable_thinking else 'OFF'})")
    print(
        f"多层联合干预层={layers} | alpha={args.alpha} | "
        f"intervene_from_round={intervene_from_round} | mode={vec_meta['mode']} | "
        f"解码={args.decoding} | 测试题数={len(test_items)}"
        + (" | BASELINE_ONLY" if args.baseline_only else "")
        + (
            f" | reuse_baseline={args.reuse_baseline_from}"
            if args.reuse_baseline_from is not None
            else ""
        )
    )
    print("█" * 70)
    if comparison is not None:
        print(comparison.to_string(index=False))
    else:
        print(base_only.to_string(index=False))
    print("█" * 70)
    print(
        "说明：acc 为「坚持正确答案」比例；delta_acc>0 表示干预提升了抗谄媚能力。"
        " Round0 的 delta 反映无诱导时血清对正常作答的影响（默认已在 R0 注入向量）。"
        " 扫 α/层时请用 --reuse-baseline-from 冻结共享 baseline。"
    )
    print(f"\n详细结果已保存至: {output_dir}")
    print("  - detail.csv")
    print("  - item_summary_{baseline,intervention}.csv")
    print("  - overall_summary_{baseline,intervention}.csv")
    print("  - summary_by_round_name_{baseline,intervention}.csv")
    print("  - summary_by_subject_{baseline,intervention}.csv")
    print("  - summary_by_round1_group_{baseline,intervention}.csv")
    print("  - comparison_by_round.csv")
    print("  - run_config.json")


if __name__ == "__main__":
    main()
