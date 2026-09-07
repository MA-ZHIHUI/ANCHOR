# extract_flipped.py
"""
从实验结果 item_summary.csv 中挑出发生翻转的题目（默认 final_flipped==1，即最终屈服于
诱导），按 item_index 从对应的 correct.json 取回原题，另存为新文件（保持 correct.json 结构，
可直接回灌到后续实验/表征工程干预）。

下一步（闭环推荐）:
  将输出作为 validate_candidates.py 的 --candidates，在 Transformers 上准入复现后再 step1 切分。

默认路径推断（可用参数覆盖）：
  item_summary:  results/<dataset>/<run>/item_summary.csv
  correct.json:  filtered_data/<dataset>/<model>/correct.json
                 （<model> 由 <run> 去掉 _single/_multi 推断，保留 _think 后缀）
  输出:          flipped_data/<dataset>/<run>/<column>.json

用法:
  python extract_flipped.py \
      --item-summary ../results/mmlu_dev/Qwen3-8B_single/item_summary.csv
  # 指定翻转口径为“任意一轮曾翻转”
  python extract_flipped.py --item-summary <...> --column ever_flipped
  # 显式指定输入输出
  python extract_flipped.py --item-summary <...> --correct-json <...> --output <...>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config

FLIP_COLUMNS = ["final_flipped", "ever_flipped"]


def infer_correct_json(item_summary: Path) -> Path:
    """由 results/<dataset>/<run>/item_summary.csv 推断 correct.json 路径。"""
    run = item_summary.parent.name
    dataset = item_summary.parent.parent.name
    # <run> 去掉 _single / _multi，保留 _think 得到 filtered_data 下的 <model> 目录名
    model_dir = run.replace("_single", "").replace("_multi", "")
    return config.PROJECT_ROOT / "filtered_data" / dataset / model_dir / "correct.json"


def infer_output(item_summary: Path, column: str) -> Path:
    run = item_summary.parent.name
    dataset = item_summary.parent.parent.name
    return config.PROJECT_ROOT / "flipped_data" / dataset / run / f"{column}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取发生翻转的题目并另存")
    parser.add_argument(
        "--item-summary", type=Path, required=True, help="item_summary.csv 路径"
    )
    parser.add_argument(
        "--correct-json",
        type=Path,
        default=None,
        help="对应的 correct.json 路径（默认按目录结构推断）",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="输出 JSON 路径（默认自动命名）"
    )
    parser.add_argument(
        "--column",
        choices=FLIP_COLUMNS,
        default="final_flipped",
        help="翻转口径：final_flipped(最终翻转) 或 ever_flipped(任意一轮翻转)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    item_summary = args.item_summary.resolve()
    if not item_summary.exists():
        raise FileNotFoundError(f"未找到 item_summary: {item_summary}")

    correct_json = (
        args.correct_json.resolve()
        if args.correct_json
        else infer_correct_json(item_summary)
    )
    if not correct_json.exists():
        raise FileNotFoundError(
            f"未找到 correct.json: {correct_json}\n请用 --correct-json 显式指定。"
        )

    output = args.output.resolve() if args.output else infer_output(item_summary, args.column)

    df = pd.read_csv(item_summary)
    if args.column not in df.columns:
        raise ValueError(f"item_summary 缺少列 '{args.column}'，现有列: {list(df.columns)}")

    with correct_json.open("r", encoding="utf-8") as f:
        correct_items = json.load(f)

    flipped = df[df[args.column] == 1]
    selected: list[dict] = []
    missing: list[int] = []
    for item_index in flipped["item_index"].astype(int).tolist():
        if 0 <= item_index < len(correct_items):
            selected.append(correct_items[item_index])
        else:
            missing.append(item_index)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=4)

    total = len(df)
    print(f"item_summary : {item_summary}")
    print(f"correct.json : {correct_json} (共 {len(correct_items)} 题)")
    print(f"翻转口径     : {args.column}")
    print(f"命中题目     : {len(selected)} / {total}")
    if missing:
        print(f"⚠️ 有 {len(missing)} 个 item_index 越界，已跳过: {missing[:10]}...")
    print(f"已保存       : {output}")


if __name__ == "__main__":
    main()
