# filter_sciq.py
"""
在 baseline（无干扰）条件下过滤 SciQ，输出格式与 filter_mmlu.py 对齐。

正确题 -> {output_dir}/{run_name}/correct.json
错误题 -> {output_dir}/{run_name}/incorrect.json
无法解析 -> {output_dir}/{run_name}/unparsable.json
运行明细 -> {output_dir}/{run_name}/detail.jsonl
统计摘要 -> {output_dir}/{run_name}/summary.json

run_name 规则: {model} 或 {model}_think

SciQ JSON 字段:
  question: str
  correct_answer: str
  distractor1 / distractor2 / distractor3: str
  support: str（不作为选项，不写入 correct.json）

说明:
  将 correct_answer + 3 个 distractor 组成 A/B/C/D 四选项；
  正确答案位置用题干 MD5 确定性打乱，避免永远落在 A。
  subject 固定为 "sciq"（原数据无学科字段）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from filter_mmlu import evaluate_model, model_run_name, save_outputs


# ================= 可调参数 =================
SOURCE = Path("raw_data/sciq/valid.json")
MODELS = ["gemma-4-12B-it"]
OUTPUT_DIR = Path("filtered_data/sciq_valid")
ENABLE_THINKING = True
LIMIT = None
# ============================================


LETTERS = "ABCD"


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 JSON 数组: {path}")
    return data


def row_to_item(row: dict) -> dict:
    """将 SciQ 原始行转为与 MMLU 对齐的内部格式（A–D 四选一）。"""
    required = (
        "question",
        "correct_answer",
        "distractor1",
        "distractor2",
        "distractor3",
    )
    for key in required:
        if key not in row or row[key] in (None, ""):
            raise ValueError(f"缺少字段 {key}: {str(row.get('question', ''))[:80]}")

    options = [
        str(row["correct_answer"]),
        str(row["distractor1"]),
        str(row["distractor2"]),
        str(row["distractor3"]),
    ]

    # 确定性打乱，保证相同题目可复现，且正确答案不全在 A
    seed = hashlib.md5(row["question"].encode("utf-8")).hexdigest()
    order = list(range(4))
    random.Random(seed).shuffle(order)
    shuffled = [options[i] for i in order]
    answer_idx = order.index(0)  # 0 对应 correct_answer 原位置

    return {
        "question": row["question"],
        "subject": "sciq",
        "choices": {LETTERS[i]: shuffled[i] for i in range(4)},
        "answer": LETTERS[answer_idx],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="baseline 条件下过滤 SciQ 题目")
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE,
        help="SciQ JSON 文件（train.json / valid.json / test.json）",
    )
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="输出根目录 {output_dir}/{model} 或 {model}_think",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_THINKING,
    )
    parser.add_argument("--limit", type=int, default=LIMIT, help="仅测试前 N 题")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()

    print(f"数据源: {source}")
    print(f"thinking 模式: {'开启' if args.enable_thinking else '关闭'}")

    raw_rows = load_json(source)
    items = [row_to_item(row) for row in raw_rows]
    print(f"共加载 {len(items)} 题（已转为 A–D 四选项）\n")

    # 快速自检：答案字母分布应大致均匀
    from collections import Counter

    dist = Counter(item["answer"] for item in items)
    print(f"正确答案字母分布: {dict(sorted(dist.items()))}\n")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        run_name = model_run_name(model, args.enable_thinking)
        print(f"开始测试模型: {model} -> 输出目录: {run_name}")
        correct, incorrect, unparsable, details = evaluate_model(
            model,
            items,
            enable_thinking=args.enable_thinking,
            limit=args.limit,
        )
        save_outputs(
            output_dir,
            model,
            source,
            args.enable_thinking,
            correct,
            incorrect,
            unparsable,
            details,
        )

    print("\n全部完成。")


if __name__ == "__main__":
    main()
