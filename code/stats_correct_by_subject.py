# stats_correct_by_subject.py
"""
统计 correct.json 中按 subject 分类的题目数量，输出详细表格。

用法:
  python stats_correct_by_subject.py
  python stats_correct_by_subject.py --input <path/to/correct.json>
  python stats_correct_by_subject.py --output <path/to/subject_stats.csv>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

import config

# 跨平台默认路径，指向已存在的筛选结果
DEFAULT_INPUT = config.FILTERED_DIR / config.MODEL_NAME / "correct.json"


def load_items(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 JSON 数组: {path}")
    return data


def build_subject_table(items: list[dict]) -> pd.DataFrame:
    subjects = [item.get("subject", "") or "(empty)" for item in items]
    counts = Counter(subjects)
    total = len(items)

    rows = [
        {
            "rank": 0,
            "subject": subject,
            "count": count,
            "pct": round(count / total * 100, 2) if total else 0.0,
        }
        for subject, count in counts.items()
    ]
    df = pd.DataFrame(rows).sort_values(["count", "subject"], ascending=[False, True])
    df["rank"] = range(1, len(df) + 1)
    df = df[["rank", "subject", "count", "pct"]]

    summary = pd.DataFrame(
        [
            {"rank": "", "subject": "__TOTAL__", "count": total, "pct": 100.0},
            {"rank": "", "subject": "__NUM_SUBJECTS__", "count": len(counts), "pct": ""},
        ]
    )
    return pd.concat([df, summary], ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 subject 统计 correct.json 题目分布")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="correct.json 路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 CSV 路径（默认: 与输入同目录下的 subject_stats.csv）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"未找到文件: {input_path}")

    items = load_items(input_path)
    df = build_subject_table(items)

    output_path = (
        args.output.resolve() if args.output else input_path.parent / "subject_stats.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    num_subjects = len([s for s in df["subject"] if not str(s).startswith("__")])
    total = int(df.loc[df["subject"] == "__TOTAL__", "count"].iloc[0])

    print(f"数据文件: {input_path}")
    print(f"总题数: {total}")
    print(f"学科类别数: {num_subjects}")
    print("\n=== 按 subject 统计（前 20）===\n")
    detail = df[~df["subject"].astype(str).str.startswith("__")]
    print(detail.head(20).to_string(index=False))
    if len(detail) > 20:
        print(f"\n... 共 {len(detail)} 个 subject，完整表格见 CSV")
    print(f"\n已保存: {output_path}")


if __name__ == "__main__":
    main()
