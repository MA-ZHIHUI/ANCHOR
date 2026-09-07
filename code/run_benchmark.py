# run_benchmark.py
"""
单轮 prompt 对比实验（本地 Transformers 后端）。

5 档条件（baseline / praise_hint / vanity / pressure / strong_vanity）每档独立一问一答，
无对话历史。与 run_drift_multi.py 使用相同话术与指标，便于单轮 vs 多轮对比。

输出 (默认 results/<data>/<model>_single[_think]/):
  detail.csv / item_summary.csv / overall_summary.csv
  summary_by_round_name.csv / summary_by_subject.csv / summary_by_round1_group.csv
"""

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from backend import LocalModel
from filter_mmlu import model_run_name
from metrics import (
    build_item_summary,
    compute_overall_summary,
    compute_round1_group_summary,
    compute_subject_summary,
    compute_summary_by_round_name,
    step_row_to_detail,
)
from prompts import DRIFT_ROUND_NAMES, build_single_round_prompt

# ================= 可调参数 =================
MODEL_PATH = os.environ.get("ANCHOR_QWEN4B_PATH", "")
MODEL_NAME = "Qwen3-4B"
CORRECT_JSON = "filtered_data/mmlu_dev/Qwen3-4B/correct.json"
ENABLE_THINKING = True
OUTPUT_DIR = "results/mmlu_dev/Qwen3-4B_single"
LIMIT = None
# ============================================


def load_data(path: Path, limit: int | None = None) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit] if limit else data


def run_single_round_benchmark(
    data: list[dict],
    model: LocalModel,
    model_name: str,
    *,
    enable_thinking: bool,
) -> list[dict]:
    results: list[dict] = []

    print(f"\nRunning {model_name} (single-round, aligned with drift)\n")
    for item_index, item in enumerate(tqdm(data, desc=model_name)):
        gt = item["answer"]
        misleading = item["misleading"]
        subject = item.get("subject", "")
        choices = list(item["choices"].keys())

        for round_idx, round_name in enumerate(DRIFT_ROUND_NAMES):
            prompt = build_single_round_prompt(
                item, round_name, enable_thinking=enable_thinking
            )
            messages = [{"role": "user", "content": prompt}]
            step = model.generate_with_belief(
                messages, choices=choices, enable_thinking=enable_thinking
            )
            results.append(
                step_row_to_detail(
                    item_index=item_index,
                    subject=subject,
                    round_idx=round_idx,
                    round_name=round_name,
                    gt=gt,
                    misleading=misleading,
                    step=step,
                    model=model_name,
                    experiment_mode="single_round",
                )
            )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单轮 5 档 prompt 对比（本地 Transformers）")
    parser.add_argument("--model-path", default=MODEL_PATH, help="本地模型权重目录")
    parser.add_argument("--model", default=MODEL_NAME, help="模型短名（用于命名）")
    parser.add_argument("--correct-json", type=Path, default=CORRECT_JSON, help="correct.json 路径")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_THINKING,
        help="开启/关闭模型 thinking 模式",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="结果输出目录")
    parser.add_argument("--limit", type=int, default=LIMIT, help="仅测试前 N 题")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    correct_path = args.correct_json.resolve()
    if not correct_path.exists():
        raise FileNotFoundError(f"未找到筛选结果: {correct_path}\n请先运行 filter_mmlu.py。")

    data = load_data(correct_path, limit=args.limit)
    print(f"数据: {correct_path} ({len(data)} 题)")
    print(f"模型: {args.model} ({args.model_path})")
    print(f"thinking: {'开启' if args.enable_thinking else '关闭'}")
    print(f"条件: {DRIFT_ROUND_NAMES}")
    print("模式: 单轮独立（每档条件单独一问一答）")

    # 种子须在 LocalModel / 首次 CUDA 之前
    config.set_seed()
    model = LocalModel(args.model_path)
    detail_rows = run_single_round_benchmark(
        data, model, args.model, enable_thinking=args.enable_thinking
    )
    detail_df = pd.DataFrame(detail_rows)

    item_summaries = [
        build_item_summary(detail_df[detail_df["item_index"] == idx].to_dict("records"))
        for idx in detail_df["item_index"].unique()
    ]
    item_df = pd.DataFrame(item_summaries)

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

    print("\n=== Overall (final=strong_vanity vs baseline) ===")
    print(overall.to_string(index=False))
    print("\n=== By Round Name / Condition ===")
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
