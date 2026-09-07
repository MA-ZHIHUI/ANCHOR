# step1_split_dataset.py
"""
表征工程 - 步骤 1：数据集切分。

将「正式分析集」按固定随机种子打乱，切分为两份互不重叠的子集：

  extraction_set.json：用于离线提取谄媚向量（默认 40 道）
  test_set.json      ：用于验证干预效果（其余题目）

推荐输入（闭环严谨路径）:
  validate_candidates.py 产出的 valid.json
  （即 Transformers 上再次确认「Round0 答对 ∧ 诱导翻转」的 D_valid）

亦可直接用第一阶段的 final_flipped.json，但跨引擎时存在选样风险，论文主结果不推荐。

提取集与测试集严格隔离，避免「在提取数据上验证」造成的信息泄漏。

用法:
  python step1_split_dataset.py --input ../flipped_data/.../transformers_gate/valid.json
  python step1_split_dataset.py --extraction-size 40 --seed 42
"""


from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import config

# 默认输入：优先使用准入复现后的 D_valid；若不存在请显式传 --input
DEFAULT_INPUT = (
    "flipped_data/mmlu_dev/Qwen3-8B_single_think/transformers_gate/valid.json"
)


def load_items(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 JSON 数组: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="表征工程步骤1：切分提取集/测试集")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="翻转题 JSON 路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认与输入文件同目录）",
    )
    parser.add_argument(
        "--extraction-size", type=int, default=40, help="提取集题目数量（其余归入测试集）"
    )
    parser.add_argument("--seed", type=int, default=config.SEED, help="打乱随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path}")

    output_dir = (args.output_dir.resolve() if args.output_dir else input_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(input_path)
    total = len(items)
    if args.extraction_size >= total:
        raise ValueError(
            f"提取集数量({args.extraction_size}) 必须小于总题数({total})"
        )

    # 固定种子打乱，保证可复现（用独立的 Random 实例，避免影响全局随机状态）
    rng = random.Random(args.seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    extraction_set = shuffled[: args.extraction_size]
    test_set = shuffled[args.extraction_size:]

    extraction_path = output_dir / "extraction_set.json"
    test_path = output_dir / "test_set.json"
    with extraction_path.open("w", encoding="utf-8") as f:
        json.dump(extraction_set, f, ensure_ascii=False, indent=4)
    with test_path.open("w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=4)

    print(f"输入        : {input_path} (共 {total} 题)")
    print(f"随机种子    : {args.seed}")
    print(f"提取集      : {len(extraction_set)} 题 -> {extraction_path}")
    print(f"测试集      : {len(test_set)} 题 -> {test_path}")


if __name__ == "__main__":
    main()
