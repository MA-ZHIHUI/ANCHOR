#!/usr/bin/env python3
"""Build full CommonsenseQA-dev JSON with deterministic misleading (no model filter)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from filter_mmlu import pick_misleading

LETTERS = config.CHOICES_MAX


def load_full_csqa(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            labels = rec["choices"]["label"]
            texts = rec["choices"]["text"]
            answer_key = rec["answerKey"]
            if answer_key not in labels:
                continue
            n = len(labels)
            item = {
                "id": rec.get("id", ""),
                "question": rec["question"],
                "subject": rec.get("question_concept") or "commonsenseqa",
                "choices": {labels[i]: texts[i] for i in range(n)},
                "answer": answer_key,
            }
            item["misleading"] = pick_misleading(item)
            items.append(item)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full CSQA dev with misleading")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("raw_data/commonsenseqa/dev_rand_split.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "filtered_data/commonsenseqa_dev/Qwen3-8B/full_dev_with_misleading.json"
        ),
    )
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.output.resolve()

    items = load_full_csqa(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)

    print(f"wrote {len(items)} items -> {out}")


if __name__ == "__main__":
    main()
