"""
多轮认知漂移实验（v3.0）— 与 run_benchmark.py 对齐。

5 轮累积对话。支持断点续跑：
  - 每完成一整题（5 轮）即追加写入 detail.csv
  - 重跑时跳过已齐 5 轮的 item；若某题只有部分轮次则删除残缺行并整题重跑

用法:
  python run_drift.py
  python run_drift.py --limit 10
  python run_drift.py --index 35
  python run_drift.py --finalize-only   # 仅重算 summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import openai
from tqdm import tqdm

from checkpoint_io import (
    append_detail_rows,
    backup_file,
    completed_item_indices,
    drop_partial_items,
    load_detail_csv,
    merge_detail_rows,
    save_detail_csv,
)
from inference import ask_with_belief, strip_thinking
from metrics import build_item_summary, step_row_to_detail, write_result_csvs
from prompts import (
    DRIFT_ROUND_NAMES,
    build_drift_followup_flow,
    build_single_round_prompt,
    with_answer_suffix,
)

# ================= 默认配置 =================
MODEL_NAME = "Qwen3-8B"
CORRECT_JSON = Path("filtered_data/mmlu_dev/Qwen3-8B/correct.json")
OUTPUT_DIR = Path("results/mmlu_dev/Qwen3-8B")
ENABLE_THINKING = False
LIMIT = None
# ============================================


def load_data(
    path: Path, *, index: int | None = None, limit: int | None = None
) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if index is not None:
        if index < 0 or index >= len(data):
            raise IndexError(f"index={index} 超出范围 (共 {len(data)} 题)")
        return [data[index]]
    return data[:limit] if limit else data


def run_drift_experiment(
    item: dict,
    flow: list[str | None],
    *,
    model: str,
    enable_thinking: bool,
    item_index: int | None = None,
    verbose: bool = False,
) -> list[dict]:
    question = build_single_round_prompt(item, "baseline")
    history = [{"role": "user", "content": question}]
    gt = item["answer"]
    misleading = item["misleading"]
    rows: list[dict] = []

    is_success = True
    error_msg = None

    if verbose:
        print("\n开始多轮认知漂移实验...")
        print(f"题目 [{item_index}] subject={item.get('subject', '')}")
        print(f"answer={gt} | misleading={misleading}")

    for round_idx, user_input in enumerate(flow):
        round_name = DRIFT_ROUND_NAMES[round_idx]
        if verbose:
            print(f"\n--- Round {round_idx} ({round_name}) ---")

        if round_idx > 0 and user_input is not None:
            history.append({"role": "user", "content": with_answer_suffix(user_input)})

        try:
            step = ask_with_belief(
                model, history, enable_thinking=enable_thinking, verbose=verbose
            )
            cleaned_content = strip_thinking(step["text"]) or step.get(
                "answer_text", ""
            )
            history.append({"role": "assistant", "content": cleaned_content})
            rows.append(
                step_row_to_detail(
                    item_index=item_index,
                    subject=item.get("subject", ""),
                    round_idx=round_idx,
                    round_name=round_name,
                    gt=gt,
                    misleading=misleading,
                    step=step,
                    model=model,
                    experiment_mode="multi_round",
                )
            )
        except openai.BadRequestError as e:
            if "maximum context length" in str(e):
                print(
                    f"\n题目 [{item_index}] 第 {round_idx} 轮上下文溢出，跳过后续轮次。"
                )
                is_success = False
                error_msg = "context_overflow"
                break
            raise

    if not is_success:
        for r in rows:
            r["is_valid_experiment"] = False
            r["error_tag"] = error_msg

    return rows


def print_single_item_report(item_rows: list[dict]) -> None:
    gt = item_rows[0]["gt"]
    misleading = item_rows[0]["misleading"]
    summary = build_item_summary(item_rows)

    print("\n" + "=" * 50)
    print("认知漂移汇总")
    print("=" * 50)
    for row in item_rows:
        print(
            f"轮次 {row['round']} ({row['round_name']}): "
            f"gt_belief={row['belief_gt']*100:.2f}% | "
            f"mis_belief={row['belief_misleading']*100:.2f}% | "
            f"pred={row['pred']} complete={row.get('is_complete')}"
        )
    flip_at = summary["first_flip_round"]
    print(f"\n首次翻转: {'从未' if flip_at == -1 else f'Round {flip_at}'}")
    print(f"EDI: {summary['edi'] * 100:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多轮认知漂移（断点续跑）")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--correct-json", type=Path, default=CORRECT_JSON)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_THINKING,
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="跳过已齐 5 轮的题目",
    )
    parser.add_argument(
        "--backup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="不调模型，仅重算 summary",
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
            raise FileNotFoundError(f"无 detail: {detail_path}")
        result = write_result_csvs(output_dir, detail_df)
        print(result["overall"].to_string(index=False))
        print(result["by_round_name"].to_string(index=False))
        print(f"\n已重算 → {output_dir}")
        return

    items = load_data(args.correct_json.resolve(), index=args.index, limit=args.limit)
    verbose = args.index is not None
    model = args.model
    experiment_mode = "multi_round"

    existing = load_detail_csv(detail_path)
    if not existing.empty and args.backup:
        bak = backup_file(detail_path)
        if bak:
            print(f"已备份: {bak}")

    # 残缺题整题重跑：先丢掉 partial
    if not existing.empty:
        before = len(existing)
        existing = drop_partial_items(
            existing,
            model=model,
            experiment_mode=experiment_mode,
            required_round_names=list(DRIFT_ROUND_NAMES),
        )
        dropped = before - len(existing)
        if dropped:
            print(f"已删除不完整题行: {dropped}（将整题重跑）")
            save_detail_csv(existing, detail_path)

    done_items = completed_item_indices(
        existing,
        model=model,
        experiment_mode=experiment_mode,
        required_round_names=list(DRIFT_ROUND_NAMES),
    )
    print(f"数据: {args.correct_json} ({len(items)} 题)")
    print(f"模型: {model} | thinking={args.enable_thinking}")
    print(f"已完成题数: {len(done_items)} | skip_existing={args.skip_existing}")
    print(f"断点文件: {detail_path}")

    # 单题 verbose 模式：不走批量断点写 summary 的提前 return 逻辑照旧
    if verbose and args.index is not None:
        if args.skip_existing and args.index in done_items:
            print(f"题目 {args.index} 已完成，跳过。可用 --no-skip-existing 强制重跑。")
            rows = existing[existing["item_index"] == args.index].to_dict("records")
            if rows:
                print_single_item_report(rows)
            return
        flow = build_drift_followup_flow(items[0])
        rows = run_drift_experiment(
            items[0],
            flow,
            model=model,
            enable_thinking=args.enable_thinking,
            item_index=args.index,
            verbose=True,
        )
        append_detail_rows(detail_path, rows)
        print_single_item_report(rows)
        # 单题模式也刷新 summary，便于检查
        detail_df = load_detail_csv(detail_path)
        write_result_csvs(output_dir, detail_df)
        return

    start_index = 0
    new_count = 0
    iterator = tqdm(list(enumerate(items)), desc="drift_v3_multi")
    for offset, item in iterator:
        item_index = start_index + offset
        if args.skip_existing and item_index in done_items:
            continue
        flow = build_drift_followup_flow(item)
        rows = run_drift_experiment(
            item,
            flow,
            model=model,
            enable_thinking=args.enable_thinking,
            item_index=item_index,
            verbose=False,
        )
        append_detail_rows(detail_path, rows)
        new_count += len(rows)
        done_items.add(item_index)
        iterator.set_postfix(item=item_index, new_rows=new_count)

    print(f"本轮新写入: {new_count} 行")
    detail_df = load_detail_csv(detail_path)
    detail_df = merge_detail_rows(existing, detail_df)
    result = write_result_csvs(output_dir, detail_df)

    print("\n=== Overall ===")
    print(result["overall"].to_string(index=False))
    print("\n=== By Round Name ===")
    print(result["by_round_name"].to_string(index=False))
    if not result["by_subject"].empty:
        print("\n=== By Subject (top) ===")
        print(result["by_subject"].head(10).to_string(index=False))
    print(f"\n结果已保存至 {output_dir}")


if __name__ == "__main__":
    main()
