# run_drift_multi.py
"""
多轮认知漂移实验（本地 Transformers 后端）— 与 run_benchmark.py 对齐。

5 轮累积对话：Round 0 baseline + Round 1-4 逐级加强诱导（话术与单轮实验相同）。
区别：本脚本为累积多轮对话（保留历史）；单轮独立实验见 run_benchmark.py。

上下文溢出保护：每轮生成前按 tokenizer 统计 prompt token 数，超过阈值即标记
context_overflow 并跳过后续轮次（替代原 API 版的 openai.BadRequestError 捕获）。

用法:
  python run_drift_multi.py --index 35     # 只跑指定题（详细输出）
  python run_drift_multi.py --limit 10
  python run_drift_multi.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from backend import LocalModel
from metrics import (
    build_item_summary,
    compute_overall_summary,
    compute_round1_group_summary,
    compute_subject_summary,
    compute_summary_by_round_name,
    step_row_to_detail,
)
from prompts import (
    DRIFT_ROUND_NAMES,
    build_drift_followup_flow,
    build_single_round_prompt,
    with_answer_suffix,
)

# ================= 默认配置 =================
CORRECT_JSON = config.FILTERED_DIR / config.MODEL_NAME / "correct.json"
OUTPUT_DIR = config.RESULTS_DIR / "mmlu_dev" / f"{config.MODEL_NAME}_multi"
ENABLE_THINKING = config.ENABLE_THINK_MODE
LIMIT = None
# ============================================


def load_data(
    path: Path, *, index: int | None = None, limit: int | None = None
) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if index is not None:
        if index < 0 or index >= len(data):
            raise IndexError(f"index={index} 超出范围 (共 {len(data)} 题)")
        return [data[index]]
    return data[:limit] if limit else data


def resolve_context_limit(model: LocalModel) -> int:
    """确定触发上下文溢出保护的 prompt token 阈值。"""
    if config.CONTEXT_TOKEN_LIMIT is not None:
        return config.CONTEXT_TOKEN_LIMIT
    max_pos = getattr(model.model.config, "max_position_embeddings", 32768)
    # 预留生成空间与安全余量（按 think 模式的较大生成上限预留）
    reserve = config.MAX_NEW_TOKENS_THINK
    return max(1024, max_pos - reserve - 256)


def run_drift_experiment(
    item: dict,
    flow: list[str | None],
    model: LocalModel,
    model_name: str,
    *,
    enable_thinking: bool,
    context_limit: int,
    item_index: int | None = None,
    verbose: bool = False,
    sampling: dict | None = None,
) -> list[dict]:
    gt = item["answer"]
    misleading = item["misleading"]
    choices = list(item["choices"].keys())

    history: list[dict] = [
        {
            "role": "user",
            "content": build_single_round_prompt(
                item, "baseline", enable_thinking=enable_thinking
            ),
        }
    ]

    rows: list[dict] = []
    is_success = True
    error_msg = None

    if verbose:
        print("\n🚀 开始认知漂移实验 (多轮)...")
        print(f"题目 [{item_index}] subject={item.get('subject', '')}")
        print(f"正确答案 answer={gt} | 误导选项 misleading={misleading}")

    for round_idx, user_input in enumerate(flow):
        round_name = DRIFT_ROUND_NAMES[round_idx]
        if verbose:
            print(f"\n--- Round {round_idx} ({round_name}) ---")

        if round_idx > 0 and user_input is not None:
            history.append(
                {
                    "role": "user",
                    "content": with_answer_suffix(user_input, enable_thinking),
                }
            )

        # 上下文溢出保护：生成前检查 prompt token 数
        prompt_tokens = model.count_prompt_tokens(
            history, enable_thinking=enable_thinking
        )
        if prompt_tokens > context_limit:
            print(
                f"\n⚠️ 题目 [{item_index}] 第 {round_idx} 轮 prompt={prompt_tokens} "
                f"token 超过上限 {context_limit}，跳过后续轮次。"
            )
            is_success = False
            error_msg = "context_overflow"
            break

        step = model.generate_with_belief(
            history,
            choices=choices,
            enable_thinking=enable_thinking,
            verbose=verbose,
            sampling=sampling,
        )

        # content 已是 think 之后的正式回答，多轮历史只保留 content（不含思维链）
        history.append({"role": "assistant", "content": step["content"]})

        rows.append(
            step_row_to_detail(
                item_index=item_index,
                subject=item.get("subject", ""),
                round_idx=round_idx,
                round_name=round_name,
                gt=gt,
                misleading=misleading,
                step=step,
                model=model_name,
                experiment_mode="multi_round",
            )
        )

    if not is_success:
        for r in rows:
            r["is_valid_experiment"] = False
            r["error_tag"] = error_msg

    return rows


def print_single_item_report(item_rows: list[dict]) -> None:
    if not item_rows:
        print("（无有效轮次，可能首轮即上下文溢出）")
        return
    gt = item_rows[0]["gt"]
    misleading = item_rows[0]["misleading"]
    summary = build_item_summary(item_rows)

    print("\n" + "=" * 50)
    print("📊 认知漂移 (Epistemic Drift) 汇总报告")
    print("=" * 50)
    for row in item_rows:
        gt_prob = row["belief_gt"] * 100
        misleading_prob = row["belief_misleading"] * 100
        print(
            f"轮次 {row['round']} ({row['round_name']}): 坚持真理({gt})={gt_prob:>6.2f}% | "
            f"转向诱导({misleading})={misleading_prob:>6.2f}% | "
            f"pred={row['pred']} valid={row['decision_valid']} "
            f"match={row['pred_matches_parse']} method={row['decision_method']}"
        )

    flip_at = summary["first_flip_round"]
    flip_text = "从未翻转" if flip_at == -1 else f"Round {flip_at}"
    print(f"\n首次 pred==misleading: {flip_text}")
    print(f"EDI: {summary['edi'] * 100:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多轮认知漂移（本地 Transformers）")
    parser.add_argument("--model-path", default=config.MODEL_PATH, help="本地模型权重目录")
    parser.add_argument(
        "--device",
        default=config.DEVICE,
        help='GPU，如 "cuda:1" 或 1；默认 config.DEVICE',
    )
    parser.add_argument("--model", default=config.MODEL_NAME, help="模型短名（用于命名）")
    parser.add_argument("--correct-json", type=Path, default=CORRECT_JSON)
    parser.add_argument("--index", type=int, default=None, help="只跑指定题目（详细输出）")
    parser.add_argument("--limit", type=int, default=LIMIT, help="批量时最多跑 N 题")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_THINKING,
        help="开启/关闭 thinking 模式",
    )
    parser.add_argument(
        "--decoding",
        choices=["greedy", "sampling"],
        default="sampling",
        help="decoding=sampling 用官方采样；greedy 确定性（与 step3 对齐时用）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_data(args.correct_json.resolve(), index=args.index, limit=args.limit)
    verbose = args.index is not None

    # 种子须在 LocalModel / 首次 CUDA 之前（CUBLAS_WORKSPACE 才生效）
    config.set_seed()
    model = LocalModel(args.model_path, device=config.normalize_device(args.device))
    context_limit = resolve_context_limit(model)
    sampling = config.GREEDY_DECODING if args.decoding == "greedy" else None
    print(
        f"上下文溢出阈值: {context_limit} tokens | 解码: {args.decoding} | "
        f"device={config.normalize_device(args.device)}"
    )

    all_detail: list[dict] = []
    iterator = items if verbose else tqdm(items, desc="drift_multi")

    start_index = args.index if args.index is not None else 0
    for offset, item in enumerate(iterator):
        item_index = args.index if args.index is not None else start_index + offset
        flow = build_drift_followup_flow(item)
        rows = run_drift_experiment(
            item,
            flow,
            model,
            args.model,
            enable_thinking=args.enable_thinking,
            context_limit=context_limit,
            item_index=item_index,
            verbose=verbose,
            sampling=sampling,
        )
        all_detail.extend(rows)

        if verbose:
            print_single_item_report(rows)

    if len(items) == 1 and verbose:
        return

    detail_df = pd.DataFrame(all_detail)
    item_summaries = [
        build_item_summary(detail_df[detail_df["item_index"] == idx].to_dict("records"))
        for idx in detail_df["item_index"].unique()
    ]
    item_df = pd.DataFrame(item_summaries)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = compute_overall_summary(item_df, detail_df)
    by_round_name = compute_summary_by_round_name(detail_df)
    by_subject = compute_subject_summary(item_df)
    by_round1 = compute_round1_group_summary(item_df)

    detail_df.to_csv(output_dir / "detail.csv", index=False, encoding="utf-8-sig")
    item_df.to_csv(output_dir / "item_summary.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output_dir / "overall_summary.csv", index=False, encoding="utf-8-sig")
    by_round_name.to_csv(
        output_dir / "summary_by_round_name.csv", index=False, encoding="utf-8-sig"
    )
    by_subject.to_csv(
        output_dir / "summary_by_subject.csv", index=False, encoding="utf-8-sig"
    )
    by_round1.to_csv(
        output_dir / "summary_by_round1_group.csv", index=False, encoding="utf-8-sig"
    )

    print("\n=== Overall ===")
    print(overall.to_string(index=False))
    print("\n=== By Round Name ===")
    print(by_round_name.to_string(index=False))
    print("\n=== By Subject (top flip rate) ===")
    if not by_subject.empty:
        print(by_subject.head(10).to_string(index=False))
    print("\n=== By Round1 P(misleading) Group ===")
    if not by_round1.empty:
        print(by_round1.to_string(index=False))
    print(f"\n结果已保存至 {output_dir}")


if __name__ == "__main__":
    main()
