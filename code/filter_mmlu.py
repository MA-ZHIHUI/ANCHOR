# filter_mmlu.py
"""
baseline（无干扰）条件下测试本地模型答题能力，将题目三路分流。

正确题   -> {output_dir}/{run_name}/correct.json   （含 subject，并确定性生成 misleading）
错误题   -> {output_dir}/{run_name}/incorrect.json （含 pred）
无法解析 -> {output_dir}/{run_name}/unparsable.json（含原始输出）
运行明细 -> {output_dir}/{run_name}/detail.jsonl
统计摘要 -> {output_dir}/{run_name}/summary.json

run_name 规则: {model} 或 {model}_think（开启 thinking 时）

支持数据源:
  - MMLU parquet / csv（4 选一，answer 为 0-3）
  - CommonsenseQA jsonl（5 选一，choices={label,text}, answerKey）
  - SciQ json（correct_answer + distractor1/2/3，确定性排布为 4 选一）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

import config
from backend import LocalModel, parse_answer
from prompts import build_single_round_prompt

# ================= 可调参数 =================
SOURCE = config.DEFAULT_SOURCE
OUTPUT_DIR = config.FILTERED_DIR
ENABLE_THINKING = config.ENABLE_THINK_MODE
LIMIT = None  # 调试用，None 表示全部题目
# ============================================

LETTERS = config.CHOICES_MAX


def model_run_name(model: str, enable_thinking: bool) -> str:
    return f"{model}_think" if enable_thinking else model


# ===================== 数据加载 =====================
def load_parquet(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "读取 MMLU parquet 需要 pyarrow，请先安装: pip install pyarrow"
        ) from exc

    table = pq.read_table(path)
    rows: list[dict] = []

    if "train" in table.schema.names:
        for i in range(table.num_rows):
            rows.append(table.column("train")[i].as_py())
        return rows

    for i in range(table.num_rows):
        rows.append({name: table.column(name)[i].as_py() for name in table.schema.names})
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


def load_csqa_jsonl(path: Path) -> list[dict]:
    """CommonsenseQA：5 选一，choices={label,text}，answerKey 为字母。"""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            labels = record["choices"]["label"]
            texts = record["choices"]["text"]
            answer_key = record.get("answerKey", "")
            if answer_key not in labels:
                # 测试集可能无标签，跳过无答案样本
                continue
            rows.append(
                {
                    "question": record["question"],
                    "subject": "commonsenseqa",
                    "choices": list(texts),
                    "answer": labels.index(answer_key),
                }
            )
    return rows


def load_sciq_json(path: Path) -> list[dict]:
    """SciQ：correct_answer + distractor1/2/3，按题目哈希确定性排布成 4 选一。"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows: list[dict] = []
    for record in data:
        options = [
            record["correct_answer"],
            record["distractor1"],
            record["distractor2"],
            record["distractor3"],
        ]
        # 确定性打乱：按题目哈希生成一个固定排列
        digest = int(hashlib.md5(record["question"].encode("utf-8")).hexdigest(), 16)
        order = sorted(range(4), key=lambda i: (digest >> (i * 8)) & 0xFF)
        shuffled = [options[i] for i in order]
        answer_idx = shuffled.index(record["correct_answer"])
        rows.append(
            {
                "question": record["question"],
                "subject": "sciq",
                "choices": shuffled,
                "answer": answer_idx,
            }
        )
    return rows


def _load_single_file(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return load_parquet(path)
    if suffix == ".csv":
        return load_csv(path)
    if suffix == ".jsonl":
        return load_csqa_jsonl(path)
    if suffix == ".json":
        return load_sciq_json(path)
    raise ValueError(f"不支持的文件类型: {path}")


def load_dataset(source: Path) -> list[dict]:
    if source.is_file():
        return _load_single_file(source)

    if not source.is_dir():
        raise FileNotFoundError(f"数据源不存在: {source}")

    for pattern in ("*.csv", "*.parquet", "*.jsonl", "*.json"):
        files = sorted(source.glob(pattern))
        if files:
            rows: list[dict] = []
            for file_path in files:
                rows.extend(_load_single_file(file_path))
            return rows

    raise ValueError(f"目录中未找到受支持的数据文件: {source}")


def row_to_item(row: dict) -> dict:
    """归一化为 {question, subject, choices:{字母:文本}, answer:字母}，支持 4/5 选一。"""
    choices = row["choices"]
    n = len(choices)
    if n < 2 or n > len(LETTERS):
        raise ValueError(f"选项数量非法({n}): {str(row.get('question', ''))[:80]}")

    answer_idx = row["answer"]
    if answer_idx not in range(n):
        raise ValueError(f"非法 answer 索引: {answer_idx}")

    letters = LETTERS[:n]
    return {
        "question": row["question"],
        "subject": row.get("subject", ""),
        "choices": {letters[i]: choices[i] for i in range(n)},
        "answer": letters[answer_idx],
    }


def pick_misleading(item: dict) -> str:
    """按题目自身选项字母确定性地挑一个错误选项作为诱导目标。"""
    letters = list(item["choices"].keys())
    wrong = [letter for letter in letters if letter != item["answer"]]
    if not wrong:
        raise ValueError(f"无法生成 misleading: {str(item.get('question', ''))[:80]}")
    digest = hashlib.md5(
        f"{item['subject']}|{item['question']}".encode("utf-8")
    ).hexdigest()
    return wrong[int(digest, 16) % len(wrong)]


# ===================== 评测与分流 =====================
def evaluate_model(
    model: LocalModel,
    items: list[dict],
    *,
    enable_thinking: bool = False,
    limit: int | None = None,
    desc: str = "filter",
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    correct: list[dict] = []
    incorrect: list[dict] = []
    unparsable: list[dict] = []
    details: list[dict] = []

    subset = items[:limit] if limit else items

    for item in tqdm(subset, desc=desc):
        choices = list(item["choices"].keys())
        prompt = build_single_round_prompt(
            item, "baseline", enable_thinking=enable_thinking
        )
        out = model.generate_text(
            [{"role": "user", "content": prompt}], enable_thinking=enable_thinking
        )
        content = out["content"]
        pred = parse_answer(content, choices)

        is_unknown = pred == "UNKNOWN"
        is_correct = not is_unknown and pred == item["answer"]

        details.append(
            {
                "question": item["question"],
                "subject": item["subject"],
                "answer": item["answer"],
                "pred": pred,
                "is_correct": is_correct,
                "is_unparsable": is_unknown,
                "content": content,
                "thinking_content": out["thinking_content"],
            }
        )

        if is_unknown:
            unparsable.append(
                {
                    "question": item["question"],
                    "subject": item["subject"],
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "raw_output": out["full_text"],
                }
            )
        elif is_correct:
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
    model_name: str,
    source: Path,
    enable_thinking: bool,
    correct: list[dict],
    incorrect: list[dict],
    unparsable: list[dict],
    details: list[dict],
) -> None:
    run_name = model_run_name(model_name, enable_thinking)
    model_dir = output_dir / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    total = len(details)
    summary = {
        "model": model_name,
        "run_name": run_name,
        "enable_thinking": enable_thinking,
        "source": str(source),
        "total": total,
        "correct": len(correct),
        "incorrect": len(incorrect),
        "unparsable": len(unparsable),
        "accuracy": round(len(correct) / total, 4) if total else 0.0,
        "parse_fail_rate": round(len(unparsable) / total, 4) if total else 0.0,
    }

    with (model_dir / "correct.json").open("w", encoding="utf-8") as f:
        json.dump(correct, f, ensure_ascii=False, indent=4)
    with (model_dir / "incorrect.json").open("w", encoding="utf-8") as f:
        json.dump(incorrect, f, ensure_ascii=False, indent=4)
    with (model_dir / "unparsable.json").open("w", encoding="utf-8") as f:
        json.dump(unparsable, f, ensure_ascii=False, indent=4)
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
    parser = argparse.ArgumentParser(description="baseline 条件下过滤题目（本地 Transformers）")
    parser.add_argument("--source", type=Path, default=SOURCE, help="数据源文件或目录")
    parser.add_argument(
        "--model-path", default=config.MODEL_PATH, help="本地模型权重目录"
    )
    parser.add_argument(
        "--model", default=config.MODEL_NAME, help="用于输出目录命名的模型短名"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="输出根目录")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_THINKING,
        help="开启/关闭模型 thinking 模式",
    )
    parser.add_argument("--limit", type=int, default=LIMIT, help="仅测试前 N 题")
    parser.add_argument(
        "--device",
        default=config.DEVICE,
        help='GPU，如 "cuda:2" 或 2；默认 config.DEVICE',
    )
    return parser.parse_args()


def main() -> None:
    config.set_seed()
    args = parse_args()
    source = args.source.resolve()

    print(f"数据源: {source}")
    print(f"thinking 模式: {'开启' if args.enable_thinking else '关闭'}")

    raw_rows = load_dataset(source)
    items = [row_to_item(row) for row in raw_rows]
    print(f"共加载 {len(items)} 题\n")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = LocalModel(args.model_path, device=args.device)
    run_name = model_run_name(args.model, args.enable_thinking)
    print(f"开始测试模型: {args.model} -> 输出目录: {run_name}")

    correct, incorrect, unparsable, details = evaluate_model(
        model,
        items,
        enable_thinking=args.enable_thinking,
        limit=args.limit,
        desc=args.model,
    )
    save_outputs(
        output_dir,
        args.model,
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
