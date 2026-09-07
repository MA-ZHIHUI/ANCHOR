# validate_candidates.py
"""
第二阶段准入复现（Transformers Gate）：闭环严谨性关键补丁。

背景:
  第一阶段（现象量化）筛出的 final_flipped.json 是「候选集」D_cand。
  这些题在发现栈上满足「Round0 答对 ∧ 诱导后翻转」，但换到 Transformers
  （干预实验所用引擎）后不能默认仍成立。

本脚本:
  对候选集在 Transformers 上重跑「无干预」多轮诱导，按与第一阶段相同的口径
  重新判定，产出正式分析集 D_valid。

  论文协议：先对 01_candidates 全池 Gate（本脚本 / p_gate_splits），
  再 p1_split_diagnostic 从 valid 中切分 extract/test。

分类:
  valid.json              : Round0 答对 且 诱导翻转（正式数据集）
  rejected_fail0.json     : Round0 即未答对
  rejected_noflip.json    : Round0 答对，但诱导下未翻转
  gate_summary.json       : 保留率等统计（写论文附录用）

用法:
  python validate_candidates.py \\
      --candidates ../flipped_data/mmlu_dev/Qwen3-8B_single/final_flipped.json \\
      --model-path /path/to/Qwen3-8B --decoding greedy --no-enable-thinking
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from backend import LocalModel
from metrics import build_item_summary, step_row_to_detail
from prompts import (
    DRIFT_ROUND_NAMES,
    build_drift_followup_flow,
    build_single_round_prompt,
    with_answer_suffix,
)
from run_drift_multi import resolve_context_limit

DEFAULT_CANDIDATES = (
    config.PROJECT_ROOT
    / "flipped_data"
    / "mmlu_dev"
    / "Qwen3-8B_single"
    / "final_flipped.json"
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def run_one_item(
    model: LocalModel,
    item: dict,
    *,
    item_index: int,
    model_name: str,
    enable_thinking: bool,
    context_limit: int,
    sampling: dict | None,
) -> list[dict]:
    """无干预多轮诱导，返回 detail 行。"""
    gt = item["answer"]
    misleading = item["misleading"]
    choices = list(item["choices"].keys())
    flow = build_drift_followup_flow(item)

    history: list[dict] = [
        {
            "role": "user",
            "content": build_single_round_prompt(
                item, "baseline", enable_thinking=enable_thinking
            ),
        }
    ]
    rows: list[dict] = []
    is_success = True
    error_msg = None

    for round_idx, user_input in enumerate(flow):
        round_name = DRIFT_ROUND_NAMES[round_idx]
        if round_idx > 0 and user_input is not None:
            history.append(
                {
                    "role": "user",
                    "content": with_answer_suffix(user_input, enable_thinking),
                }
            )

        if model.count_prompt_tokens(history, enable_thinking=enable_thinking) > context_limit:
            is_success = False
            error_msg = "context_overflow"
            break

        step = model.generate_with_belief(
            history,
            choices=choices,
            enable_thinking=enable_thinking,
            sampling=sampling,
        )
        history.append({"role": "assistant", "content": step["content"]})
        rows.append(
            step_row_to_detail(
                item_index=item_index,
                subject=item.get("subject", ""),
                round_idx=round_idx,
                round_name=round_name,
                gt=gt,
                misleading=misleading,
                step=step,
                model=model_name,
                experiment_mode="gate_validate",
            )
        )

    if not is_success:
        for r in rows:
            r["is_valid_experiment"] = False
            r["error_tag"] = error_msg
    return rows


def classify_item(
    item: dict,
    item_rows: list[dict],
    *,
    flip_column: str,
) -> tuple[str, dict]:
    """
    返回 (bucket, meta)。
    bucket ∈ {valid, fail0, noflip, incomplete}
    """
    summary = build_item_summary(item_rows)
    if not summary.get("is_valid", 0):
        return "incomplete", {"reason": "incomplete_or_overflow", **summary}

    round0 = next(r for r in item_rows if r["round_name"] == "baseline")
    correct0 = int(round0["pred"] == round0["gt"])
    flipped = int(summary.get(flip_column, 0) == 1)

    meta = {
        "correct0": correct0,
        "flipped": flipped,
        "first_flip_round": summary.get("first_flip_round", -1),
        "edi": summary.get("edi", 0.0),
        "pred_round0": round0["pred"],
        "gt": round0["gt"],
        "misleading": round0["misleading"],
    }

    if not correct0:
        return "fail0", meta
    if not flipped:
        return "noflip", meta
    return "valid", meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="第二阶段准入：在 Transformers 上复现候选翻转题"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES,
        help="第一阶段产出的候选翻转题 JSON（如 final_flipped.json）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认与 candidates 同目录下的 transformers_gate/）",
    )
    parser.add_argument("--model-path", default=config.MODEL_PATH)
    parser.add_argument(
        "--device",
        default=config.DEVICE,
        help='GPU，如 "cuda:1" 或 1；默认 config.DEVICE',
    )
    parser.add_argument("--model", default=config.MODEL_NAME, help="模型短名（写入 detail）")
    parser.add_argument(
        "--decoding",
        choices=["greedy", "sampling"],
        default="greedy",
        help="建议与 step3 干预实验保持一致（默认 greedy）",
    )
    parser.add_argument(
        "--flip-column",
        choices=["final_flipped", "ever_flipped"],
        default="final_flipped",
        help="翻转口径，需与第一阶段 extract_flipped 一致",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=config.ENABLE_THINK_MODE,
    )
    parser.add_argument("--limit", type=int, default=None, help="仅验证前 N 题（调试）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cand_path = args.candidates.resolve()
    if not cand_path.exists():
        raise FileNotFoundError(f"未找到候选集: {cand_path}")

    candidates = load_json(cand_path)
    if args.limit:
        candidates = candidates[: args.limit]

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else cand_path.parent / "transformers_gate"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = config.normalize_device(args.device)
    config.set_seed()
    model = LocalModel(args.model_path, device=device)
    context_limit = resolve_context_limit(model)
    sampling = config.GREEDY_DECODING if args.decoding == "greedy" else None

    print(
        f"候选集: {cand_path} ({len(candidates)} 题)\n"
        f"模型: {args.model_path} | device={device} | 解码: {args.decoding} | "
        f"Think: {'ON' if args.enable_thinking else 'OFF'} | "
        f"翻转口径: {args.flip_column}\n"
        f"输出: {output_dir}"
    )

    all_detail: list[dict] = []
    buckets: dict[str, list[dict]] = {
        "valid": [],
        "fail0": [],
        "noflip": [],
        "incomplete": [],
    }
    per_item_meta: list[dict] = []

    for idx, item in enumerate(tqdm(candidates, desc="validate_gate")):
        rows = run_one_item(
            model,
            item,
            item_index=idx,
            model_name=args.model,
            enable_thinking=args.enable_thinking,
            context_limit=context_limit,
            sampling=sampling,
        )
        all_detail.extend(rows)
        bucket, meta = classify_item(item, rows, flip_column=args.flip_column)
        buckets[bucket].append(item)
        per_item_meta.append({"cand_index": idx, "bucket": bucket, **meta})

    # 落盘分类集（valid 供 step1 使用）
    save_json(output_dir / "valid.json", buckets["valid"])
    save_json(output_dir / "rejected_fail0.json", buckets["fail0"])
    save_json(output_dir / "rejected_noflip.json", buckets["noflip"])
    save_json(output_dir / "rejected_incomplete.json", buckets["incomplete"])

    detail_df = pd.DataFrame(all_detail)
    detail_df.to_csv(output_dir / "detail.csv", index=False, encoding="utf-8-sig")
    meta_df = pd.DataFrame(per_item_meta)
    meta_df.to_csv(output_dir / "item_gate_labels.csv", index=False, encoding="utf-8-sig")

    n = len(candidates)
    n_valid = len(buckets["valid"])
    summary = {
        "candidates": str(cand_path),
        "n_candidates": n,
        "n_valid": n_valid,
        "n_fail0": len(buckets["fail0"]),
        "n_noflip": len(buckets["noflip"]),
        "n_incomplete": len(buckets["incomplete"]),
        "retention_rate": round(n_valid / n, 4) if n else 0.0,
        "flip_column": args.flip_column,
        "decoding": args.decoding,
        "device": device,
        "enable_thinking": bool(args.enable_thinking),
        "model_path": args.model_path,
        "output_dir": str(output_dir),
        "note": (
            "valid.json 为 Transformers 上确认的正式分析集 D_valid；"
            "全池 Gate 后请运行 p1_split_diagnostic.py 从 02_gate/*/valid.json 切分。"
        ),
    }
    save_json(output_dir / "gate_summary.json", summary)

    print("\n" + "=" * 60)
    print("Transformers 准入复现完成")
    print(f"  候选 D_cand     : {n}")
    print(f"  正式 D_valid    : {n_valid}  (保留率 {summary['retention_rate']*100:.1f}%)")
    print(f"  Round0 失败     : {summary['n_fail0']}")
    print(f"  诱导未翻转      : {summary['n_noflip']}")
    print(f"  不完整/溢出     : {summary['n_incomplete']}")
    print(
        "  → 若由 p_gate_splits 编排：全部池跑完后执行 "
        "python p1_split_diagnostic.py --repeng-root <run_root>"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
