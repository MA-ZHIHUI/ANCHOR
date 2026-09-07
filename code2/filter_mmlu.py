# filter_mmlu.py
"""
在 baseline（无干扰）条件下测试模型答题能力，将 MMLU 题目按对错分流。

正确题 -> {output_dir}/{run_name}/correct.json
  结构与 data/questions.json 类似，额外包含 subject，并自动生成 misleading

错误题 -> {output_dir}/{run_name}/incorrect.json
  同样保留 subject，并记录 pred

运行明细 -> {output_dir}/{run_name}/detail.jsonl
统计摘要 -> {output_dir}/{run_name}/summary.json

run_name 规则: {model} 或 {model}_think（开启 thinking 时）

MMLU parquet 字段:
  question: str
  subject: str
  choices: list[str]   # 4 个选项，对应 A/B/C/D
  answer: int           # 0-3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from inference import ask_model, parse_answer
from prompts import build_prompt


# ================= 可调参数 =================
SOURCE = Path("raw_data/mmlu/dev-00000-of-00001.parquet")
MODELS = ["gemma-4-12B-it"]
OUTPUT_DIR = Path("filtered_data/mmlu_dev")
ENABLE_THINKING = True
LIMIT = None  # 调试用，None 表示全部题目
# ============================================


LETTERS = "ABCD"


def model_run_name(model: str, enable_thinking: bool) -> str:
    return f"{model}_think" if enable_thinking else model


def load_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    rows: list[dict] = []

    if "train" in table.schema.names:
        for i in range(table.num_rows):
            rows.append(table.column("train")[i].as_py())
        return rows

    for i in range(table.num_rows):
        rows.append(
            {name: table.column(name)[i].as_py() for name in table.schema.names}
        )
    return rows


def load_csv(path: Path, subject: str | None = None) -> list[dict]:
    rows: list[dict] = []
    inferred_subject = subject or path.stem.rsplit("_", 1)[0]

    with path.open("r", encoding="utf-8", newline="") as f:
        for record in csv.reader(f):
            if len(record) != 6:
                raise ValueError(f"CSV 格式错误 ({path}): {record[:2]}")
            question, a, b, c, d, answer = record
            rows.append(
                {
                    "question": question,
                    "subject": inferred_subject,
                    "choices": [a, b, c, d],
                    "answer": LETTERS.index(answer.upper()),
                }
            )
    return rows


def load_mmlu(source: Path) -> list[dict]:
    if source.is_file():
        if source.suffix == ".parquet":
            return load_parquet(source)
        if source.suffix == ".csv":
            return load_csv(source)
        raise ValueError(f"不支持的文件类型: {source}")

    if not source.is_dir():
        raise FileNotFoundError(f"数据源不存在: {source}")

    csv_files = sorted(source.glob("*.csv"))
    if csv_files:
        rows: list[dict] = []
        for csv_path in csv_files:
            rows.extend(load_csv(csv_path))
        return rows

    parquet_files = sorted(source.glob("*.parquet"))
    if parquet_files:
        rows: list[dict] = []
        for parquet_path in parquet_files:
            rows.extend(load_parquet(parquet_path))
        return rows

    raise ValueError(f"目录中未找到 CSV 或 parquet 文件: {source}")


def row_to_item(row: dict) -> dict:
    choices = row["choices"]
    if len(choices) != 4:
        raise ValueError(f"选项数量不是 4: {row.get('question', '')[:80]}")

    answer_idx = row["answer"]
    if answer_idx not in range(4):
        raise ValueError(f"非法 answer 索引: {answer_idx}")

    return {
        "question": row["question"],
        "subject": row.get("subject", ""),
        "choices": {LETTERS[i]: choices[i] for i in range(4)},
        "answer": LETTERS[answer_idx],
    }


def pick_misleading(item: dict) -> str:
    # 按题目自身选项字母选 misleading（兼容 MMLU 的 A–D 与 CommonsenseQA 的 A–E）
    letters = list(item["choices"].keys())
    wrong = [letter for letter in letters if letter != item["answer"]]
    if not wrong:
        raise ValueError(f"无法生成 misleading: {item.get('question', '')[:80]}")
    digest = hashlib.md5(
        f"{item['subject']}|{item['question']}".encode("utf-8")
    ).hexdigest()
    return wrong[int(digest, 16) % len(wrong)]


def evaluate_model(
    model: str,
    items: list[dict],
    *,
    enable_thinking: bool = False,
    limit: int | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    correct: list[dict] = []
    incorrect: list[dict] = []
    unparsable: list[dict] = []  # 新增：无法解析类
    details: list[dict] = []

    subset = items[:limit] if limit else items

    for item in tqdm(subset, desc=model):
        prompt = build_prompt(item, "baseline")
        raw = ask_model(model, prompt, enable_thinking=enable_thinking)
        pred = parse_answer(raw)

        # 核心逻辑修改：三路分流
        is_unknown = pred == "UNKNOWN"
        is_correct = not is_unknown and pred == item["answer"]

        detail = {
            "question": item["question"],
            "subject": item["subject"],
            "answer": item["answer"],
            "pred": pred,
            "is_correct": is_correct,
            "is_unparsable": is_unknown,  # 记录是否解析失败
            "raw": raw,
        }
        details.append(detail)

        if is_unknown:
            # 类别 1: 正常解析无法得到结果
            unparsable.append(
                {
                    "question": item["question"],
                    "subject": item["subject"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "raw_output": raw,
                }
            )
        elif is_correct:
            # 类别 2: 解析成功且回答正确
            correct.append(
                {
                    "question": item["question"],
                    "subject": item["subject"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "misleading": pick_misleading(item),
                }
            )
        else:
            # 类别 3: 解析成功但回答错误
            incorrect.append(
                {
                    "question": item["question"],
                    "subject": item["subject"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "pred": pred,
                }
            )

    return correct, incorrect, unparsable, details


def save_outputs(
    output_dir: Path,
    model: str,
    source: Path,
    enable_thinking: bool,
    correct: list[dict],
    incorrect: list[dict],
    unparsable: list[dict],  # 增加参数
    details: list[dict],
) -> None:
    run_name = model_run_name(model, enable_thinking)
    model_dir = output_dir / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    total = len(details)
    summary = {
        "model": model,
        "run_name": run_name,
        "enable_thinking": enable_thinking,
        "source": str(source),
        "total": total,
        "correct": len(correct),
        "incorrect": len(incorrect),
        "unparsable": len(unparsable),  # 增加统计
        "accuracy": round(len(correct) / total, 4) if total else 0.0,
        "parse_fail_rate": round(len(unparsable) / total, 4) if total else 0.0,
    }

    # 保存三个主要文件
    with (model_dir / "correct.json").open("w", encoding="utf-8") as f:
        json.dump(correct, f, ensure_ascii=False, indent=4)

    with (model_dir / "incorrect.json").open("w", encoding="utf-8") as f:
        json.dump(incorrect, f, ensure_ascii=False, indent=4)

    with (model_dir / "unparsable.json").open("w", encoding="utf-8") as f:
        json.dump(unparsable, f, ensure_ascii=False, indent=4)

    # 保存明细和摘要
    with (model_dir / "detail.jsonl").open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (model_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print(f"\n[{run_name}] 分流完成:")
    print(f"  ✅ 正确: {len(correct)}")
    print(f"  ❌ 错误: {len(incorrect)}")
    print(f"  ❓ 无法解析: {len(unparsable)}")
    print(f"  准确率: {summary['accuracy']:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="baseline 条件下过滤 MMLU 题目")
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE,
        help="MMLU 数据源：单个 parquet/csv 文件，或包含若干 csv/parquet 的目录",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="待测试模型列表",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="输出根目录，结果保存在 {output_dir}/{model} 或 {model}_think 下",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_THINKING,
        help="开启/关闭模型 thinking 模式",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=LIMIT,
        help="仅测试前 N 题，便于调试",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()

    print(f"数据源: {source}")
    print(f"thinking 模式: {'开启' if args.enable_thinking else '关闭'}")

    raw_rows = load_mmlu(source)
    items = [row_to_item(row) for row in raw_rows]
    print(f"共加载 {len(items)} 题\n")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        run_name = model_run_name(model, args.enable_thinking)
        print(f"开始测试模型: {model} -> 输出目录: {run_name}")
        # 接收 4 个返回值
        correct, incorrect, unparsable, details = evaluate_model(
            model,
            items,
            enable_thinking=args.enable_thinking,
            limit=args.limit,
        )
        # 传入 unparsable 列表
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
