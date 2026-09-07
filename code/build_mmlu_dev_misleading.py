#!/usr/bin/env python3
"""Build MMLU-dev JSON with deterministic misleading (no model baseline filter)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from filter_mmlu import load_parquet, pick_misleading, row_to_item


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 MMLU parquet 生成带 misleading 的全量 JSON（不跑模型）"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("raw_data/mmlu/dev-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "filtered_data/mmlu_dev/Qwen3-4B/full_dev_with_misleading.json"
        ),
    )
    args = parser.parse_args()

    rows = load_parquet(args.source.resolve())
    items = []
    for row in rows:
        item = row_to_item(row)
        item["misleading"] = pick_misleading(item)
        items.append(item)

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)

    print(f"wrote {len(items)} items -> {out}")


if __name__ == "__main__":
    main()
