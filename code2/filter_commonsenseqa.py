# filter_commonsenseqa.py
"""
在 baseline（无干扰）条件下过滤 CommonsenseQA，输出格式与 filter_mmlu.py 对齐。

正确题 -> {output_dir}/{run_name}/correct.json
错误题 -> {output_dir}/{run_name}/incorrect.json
无法解析 -> {output_dir}/{run_name}/unparsable.json
运行明细 -> {output_dir}/{run_name}/detail.jsonl
统计摘要 -> {output_dir}/{run_name}/summary.json

run_name 规则: {model} 或 {model}_think

CommonsenseQA JSONL 字段:
  question: str
  question_concept: str   # 映射为 subject
  choices.label: ["A".."E"]
  choices.text: [str, ...]
  answerKey: "A".."E"     # test 集通常无此字段，不可用于过滤

注意: CommonsenseQA 为 5 选一（A–E），correct.json 的 choices 将含 E。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from filter_mmlu import evaluate_model, model_run_name, save_outputs


# ================= 可调参数 =================
SOURCE = Path("raw_data/commonsenseqa/dev_rand_split.jsonl")
MODELS = ["gemma-4-12B-it"]
OUTPUT_DIR = Path("filtered_data/commonsenseqa_dev")
ENABLE_THINKING = True
LIMIT = None
# ============================================


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败 ({path}:{line_no}): {e}") from e
    return rows


def row_to_item(row: dict) -> dict:
    """将 CommonsenseQA 原始行转为与 MMLU 对齐的内部格式。"""
    if "answerKey" not in row or row["answerKey"] in ("", None):
        raise ValueError(
            "缺少 answerKey，无法过滤。"
            "CommonsenseQA test 集通常无答案，请使用 train/dev（validation）。"
        )

    choices_obj = row["choices"]
    labels = choices_obj["label"]
    texts = choices_obj["text"]
    if len(labels) != len(texts):
        raise ValueError(
            f"choices label/text 长度不一致: {row.get('question', '')[:80]}"
        )
    if len(labels) < 2:
        raise ValueError(f"选项过少: {row.get('question', '')[:80]}")

    choices = {str(label).upper(): text for label, text in zip(labels, texts)}
    answer = str(row["answerKey"]).upper()
    if answer not in choices:
        raise ValueError(
            f"answerKey={answer} 不在选项 {list(choices)} 中: "
            f"{row.get('question', '')[:80]}"
        )

    return {
        "question": row["question"],
        "subject": row.get("question_concept") or "commonsenseqa",
        "choices": choices,
        "answer": answer,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="baseline 条件下过滤 CommonsenseQA 题目"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE,
        help="CommonsenseQA JSONL（需含 answerKey，如 train/dev）",
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

    raw_rows = load_jsonl(source)
    items = [row_to_item(row) for row in raw_rows]
    print(f"共加载 {len(items)} 题（含 A–E 五选项）\n")

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
