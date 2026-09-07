# run_benchmark.py
"""
单轮 prompt 对比实验（与 run_drift 对齐）。

支持：
  - 指定条件增量跑（如只跑 neutral_reask，不重跑旧 5 档）
  - 与已有 detail.csv 合并
  - 断点续跑：每完成一条条件即写入 detail.csv，中断后重跑自动跳过已完成

用法示例:
  # 全量 5 档（断点续跑）
  python run_benchmark.py --correct-json ... --output-dir results/.../Qwen3-8B_single

  # 只补 control，合并进旧结果
  python run_benchmark.py --conditions neutral_reask \\
    --correct-json filtered_data/mmlu_dev/Qwen3-8B/correct.json \\
    --output-dir results/mmlu_dev/Qwen3-8B_single \\
    --merge-detail results/mmlu_dev/Qwen3-8B_single/detail.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from checkpoint_io import (
    append_detail_rows,
    backup_file,
    existing_keys,
    load_detail_csv,
    merge_detail_rows,
    save_detail_csv,
)
from filter_mmlu import model_run_name
from inference import ask_with_belief
from metrics import step_row_to_detail, write_result_csvs
from prompts import (
    ALL_SINGLE_ROUND_NAMES,
    DRIFT_ROUND_NAMES,
    ROUND_INDEX,
    build_single_round_prompt,
)

# ================= 可调默认参数 =================
MODELS = ["Qwen3-8B"]
FILTERED_DIR = Path("filtered_data/mmlu_dev")
CORRECT_JSON = Path("filtered_data/mmlu_dev/Qwen3-8B/correct.json")
ENABLE_THINKING = False
OUTPUT_DIR = Path("results/mmlu_dev/Qwen3-8B_single")
LIMIT = None
# =================================================


def resolve_correct_json(filtered_dir: Path, model: str, enable_thinking: bool) -> Path:
    run_name = model_run_name(model, enable_thinking)
    return filtered_dir / run_name / "correct.json"


def load_data(path: Path, limit: int | None = None) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit] if limit else data


def parse_conditions(raw: list[str] | None) -> list[str]:
    if not raw:
        return list(DRIFT_ROUND_NAMES)
    if len(raw) == 1 and raw[0].lower() in ("all", "all_single"):
        return list(ALL_SINGLE_ROUND_NAMES)
    out: list[str] = []
    for name in raw:
        if name not in ALL_SINGLE_ROUND_NAMES:
            raise ValueError(
                f"未知条件 {name!r}；可选: {ALL_SINGLE_ROUND_NAMES}"
            )
        if name not in out:
            out.append(name)
    return out


def run_single_round_benchmark(
    data: list[dict],
    models: list[str],
    *,
    enable_thinking: bool,
    conditions: list[str],
    detail_path: Path,
    done_keys: set[tuple],
    skip_existing: bool = True,
) -> tuple[list[dict], set[tuple]]:
    """逐条推理；每条成功后立即 append 到 detail_path。返回新产生的行。"""
    new_rows: list[dict] = []
    experiment_mode = "single_round"

    for model in models:
        print(f"\nRunning {model} (single-round) conditions={conditions}\n")
        pbar = tqdm(list(enumerate(data)), desc=model)
        for item_index, item in pbar:
            gt = item["answer"]
            misleading = item["misleading"]
            subject = item.get("subject", "")

            for round_name in conditions:
                key = (str(model), experiment_mode, int(item_index), str(round_name))
                if skip_existing and key in done_keys:
                    continue

                round_idx = ROUND_INDEX.get(round_name, 0)
                prompt = build_single_round_prompt(item, round_name)
                messages = [{"role": "user", "content": prompt}]
                step = ask_with_belief(model, messages, enable_thinking=enable_thinking)
                row = step_row_to_detail(
                    item_index=item_index,
                    subject=subject,
                    round_idx=round_idx,
                    round_name=round_name,
                    gt=gt,
                    misleading=misleading,
                    step=step,
                    model=model,
                    experiment_mode=experiment_mode,
                )
                append_detail_rows(detail_path, [row])
                new_rows.append(row)
                done_keys.add(key)
                pbar.set_postfix(item=item_index, cond=round_name)

    return new_rows, done_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单轮条件对比（支持增量条件 / 断点续跑 / 合并旧 detail）"
    )
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--filtered-dir", type=Path, default=FILTERED_DIR)
    parser.add_argument("--correct-json", type=Path, default=CORRECT_JSON)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_THINKING,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help=(
            "要跑的条件名；默认原 5 档。"
            "例: --conditions neutral_reask"
            " 或 --conditions all_single"
        ),
    )
    parser.add_argument(
        "--merge-detail",
        type=Path,
        default=None,
        help="已有 detail.csv；默认使用 output-dir/detail.csv（若存在）",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="主键已存在则跳过（断点续跑 / 增量 control）",
    )
    parser.add_argument(
        "--backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="合并写入前备份已有 detail.csv",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="不调模型，仅根据 output-dir/detail.csv 重算 summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "detail.csv"

    if args.finalize_only:
        detail_df = load_detail_csv(detail_path)
        if detail_df.empty:
            raise FileNotFoundError(f"无 detail 可汇总: {detail_path}")
        result = write_result_csvs(output_dir, detail_df)
        print(result["overall"].to_string(index=False))
        print(result["by_round_name"].to_string(index=False))
        print(f"\n已重算 summary → {output_dir}")
        return

    conditions = parse_conditions(args.conditions)

    if args.correct_json:
        correct_path = args.correct_json.resolve()
    else:
        correct_path = resolve_correct_json(
            args.filtered_dir.resolve(),
            args.models[0],
            args.enable_thinking,
        )
    if not correct_path.exists():
        raise FileNotFoundError(f"未找到筛选结果: {correct_path}")

    merge_path = (
        args.merge_detail.resolve()
        if args.merge_detail is not None
        else (detail_path if detail_path.exists() else None)
    )
    existing = load_detail_csv(merge_path)
    if merge_path is not None and merge_path.exists() and args.backup:
        bak = backup_file(Path(merge_path))
        if bak:
            print(f"已备份: {bak}")

    # 若 merge 来自别处且 output detail 尚不存在，先落一份底稿便于 append
    if not existing.empty and not detail_path.exists():
        save_detail_csv(existing, detail_path)
    elif (
        merge_path is not None
        and Path(merge_path).resolve() != detail_path.resolve()
        and not existing.empty
    ):
        # 统一以 output_dir/detail.csv 为断点文件
        save_detail_csv(existing, detail_path)

    # 重新从 output detail 读，保证 append 目标一致
    existing = load_detail_csv(detail_path)
    done = existing_keys(existing)

    data = load_data(correct_path, limit=args.limit)
    print(f"数据: {correct_path} ({len(data)} 题)")
    print(f"模型: {args.models}")
    print(f"thinking: {'开启' if args.enable_thinking else '关闭'}")
    print(f"条件: {conditions}")
    print(f"断点文件: {detail_path}")
    print(f"已存在键数: {len(done)} | skip_existing={args.skip_existing}")

    new_rows, done = run_single_round_benchmark(
        data,
        args.models,
        enable_thinking=args.enable_thinking,
        conditions=conditions,
        detail_path=detail_path,
        done_keys=done,
        skip_existing=args.skip_existing,
    )
    print(f"本轮新写入: {len(new_rows)} 行")

    detail_df = load_detail_csv(detail_path)
    # 保险：与内存 existing 再合并一次（防中断后仅有 append）
    detail_df = merge_detail_rows(existing, detail_df)
    result = write_result_csvs(output_dir, detail_df)

    print("\n=== Overall ===")
    print(result["overall"].to_string(index=False))
    print("\n=== By Round Name / Condition ===")
    print(result["by_round_name"].to_string(index=False))
    if not result["by_subject"].empty:
        print("\n=== By Subject (top) ===")
        print(result["by_subject"].head(10).to_string(index=False))
    print(f"\n结果已保存至 {output_dir}")


if __name__ == "__main__":
    main()
