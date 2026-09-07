#!/usr/bin/env python3
# sweep_primary_trilayer_windows.py
"""
主实验：固定 α=50，在 primary MMLU test 上滑动三层窗干预扫描。

向量:  04_vectors（per-layer）
测集:  03_splits/test_set_mmlu.json
默认窗: (10,11,12), ..., (33,34,35)  —— 避开浅层大 α 崩坏区
协议:  Think-OFF / greedy / fromR0 / per-layer
防护:  --max-new-tokens 默认 1536（覆盖正常长尾，防止超长乱码拖满 4096）

可选 --shallow-spotcheck：额外跑 (0,1,2) 与 (3,4,5) 作浅层对照。

用法:
  cd ~/ANCHOR/code && conda activate ANCHOR
  CUDA_VISIBLE_DEVICES=0 python sweep_primary_trilayer_windows.py --device cuda:0
  CUDA_VISIBLE_DEVICES=0 python sweep_primary_trilayer_windows.py --device cuda:0 --shallow-spotcheck
  python sweep_primary_trilayer_windows.py --summary-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

import config

REPENG = config.PROJECT_ROOT / "repeng_data" / "Qwen3-8B_multi_nothink"
VECTOR_DIR = REPENG / "04_vectors"
TEST_SET = REPENG / "03_splits" / "test_set_mmlu.json"
DATASET_TAG = "mmlu_dev"
ALPHA = 50.0
WINDOW = 3
N_LAYERS = 36  # Qwen3-8B: 0..35
DEFAULT_START = 10  # 默认从 10,11,12 起
DEFAULT_MAX_NEW_TOKENS = 1536
SHALLOW_SPOTCHECK_STARTS = (0, 3)  # → (0,1,2) 与 (3,4,5)

RESULTS_ROOT = config.PROJECT_ROOT / "results" / DATASET_TAG
REF_BASELINE_DIR = (
    RESULTS_ROOT
    / "Qwen3-8B_REF_baseline_primary_test_mmlu_nothink_greedy_fromR0"
)
SWEEP_TAG = "PRIMARY_TRIWIN"
OVERVIEW_DIR = REPENG / "05_sweeps"
OVERVIEW_CSV = OVERVIEW_DIR / "primary_trilayer_window_alpha50_overview.csv"
OVERVIEW_JSON = OVERVIEW_DIR / "primary_trilayer_window_alpha50_overview.json"


def windows(n_layers: int = N_LAYERS, width: int = WINDOW) -> list[list[int]]:
    return [list(range(s, s + width)) for s in range(0, n_layers - width + 1)]


def layers_tag(layers: list[int]) -> str:
    return "_".join(str(l) for l in layers)


def out_dir_for(layers: list[int]) -> Path:
    tag = layers_tag(layers)
    return (
        RESULTS_ROOT
        / f"Qwen3-8B_repeng_layer{tag}_alpha{ALPHA}_nothink_greedy_fromR0_perlayer_{SWEEP_TAG}"
    )


def done(path: Path) -> bool:
    return (path / "comparison_by_round.csv").exists()


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    print("\n" + "=" * 72)
    print("RUN:", " ".join(cmd))
    print("=" * 72, flush=True)
    if dry_run:
        print("[dry-run] skip")
        return
    subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parent))


def step3_base(
    *,
    device: str,
    limit: int | None,
    max_new_tokens: int | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "step3_run_intervention_test.py"),
        "--vector-dir",
        str(VECTOR_DIR),
        "--test-set",
        str(TEST_SET),
        "--dataset-tag",
        DATASET_TAG,
        "--alpha",
        str(ALPHA),
        "--decoding",
        "greedy",
        "--no-enable-thinking",
        "--intervene-from-round",
        "0",
        "--device",
        device,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(max_new_tokens)])
    return cmd


def ensure_baseline(
    *,
    device: str,
    limit: int | None,
    dry_run: bool,
    skip_done: bool,
    max_new_tokens: int | None,
) -> Path:
    if skip_done and (REF_BASELINE_DIR / "detail.csv").exists() and not dry_run:
        print(f"SKIP baseline REF already exists: {REF_BASELINE_DIR}")
        return REF_BASELINE_DIR

    cmd = step3_base(device=device, limit=limit, max_new_tokens=max_new_tokens)
    cmd.extend(
        [
            "--baseline-only",
            "--layers",
            "20,21,22",
            "--output-dir",
            str(REF_BASELINE_DIR),
        ]
    )
    run_cmd(cmd, dry_run=dry_run)
    return REF_BASELINE_DIR


def run_one_window(
    layers: list[int],
    *,
    device: str,
    limit: int | None,
    dry_run: bool,
    skip_done: bool,
    reuse_from: Path,
    max_new_tokens: int | None,
) -> Path:
    out = out_dir_for(layers)
    if skip_done and done(out):
        print(f"SKIP already done: {out.name}")
        return out

    cmd = step3_base(device=device, limit=limit, max_new_tokens=max_new_tokens)
    cmd.extend(
        [
            "--layers",
            ",".join(str(l) for l in layers),
            "--reuse-baseline-from",
            str(reuse_from),
            "--output-dir",
            str(out),
        ]
    )
    run_cmd(cmd, dry_run=dry_run)
    return out


def summarize(wins: list[list[int]]) -> pd.DataFrame:
    rows = []
    for layers in wins:
        out = out_dir_for(layers)
        cmp_path = out / "comparison_by_round.csv"
        row = {
            "layers": layers_tag(layers),
            "layer_start": layers[0],
            "layer_mid": layers[1],
            "layer_end": layers[2],
            "alpha": ALPHA,
            "output_dir": str(out),
            "done": cmp_path.exists(),
            "r0_base_acc": None,
            "r0_interv_acc": None,
            "r0_delta_acc": None,
            "r4_base_acc": None,
            "r4_interv_acc": None,
            "r4_delta_acc": None,
            "r4_interv_flip": None,
            "r4_format_gap": None,
        }
        if cmp_path.exists():
            df = pd.read_csv(cmp_path)
            rcol = "round" if "round" in df.columns else df.columns[0]
            for r in (0, 4):
                sub = df[df[rcol] == r]
                if sub.empty:
                    continue
                x = sub.iloc[0]
                pref = f"r{r}"
                row[f"{pref}_base_acc"] = float(x["baseline_acc(%)"])
                row[f"{pref}_interv_acc"] = float(x["intervention_acc(%)"])
                row[f"{pref}_delta_acc"] = float(x["delta_acc(%)"])
                if r == 4 and "intervention_flip_rate(%)" in x:
                    flip = float(x["intervention_flip_rate(%)"])
                    acc = float(x["intervention_acc(%)"])
                    row["r4_interv_flip"] = flip
                    row["r4_format_gap"] = round(100.0 - acc - flip, 2)
        rows.append(row)

    overview = pd.DataFrame(rows)
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    overview.to_csv(OVERVIEW_CSV, index=False, encoding="utf-8-sig")
    OVERVIEW_JSON.write_text(
        json.dumps(
            {
                "vector_dir": str(VECTOR_DIR),
                "test_set": str(TEST_SET),
                "alpha": ALPHA,
                "window": WINDOW,
                "default_start": DEFAULT_START,
                "n_windows": len(wins),
                "ref_baseline": str(REF_BASELINE_DIR),
                "sweep_tag": SWEEP_TAG,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    done_df = overview[overview["done"]].copy()
    print("\n" + "█" * 72)
    print(
        f"Primary trilayer window overview | α={ALPHA} | "
        f"done={len(done_df)}/{len(overview)}"
    )
    print("█" * 72)
    if not done_df.empty:
        show = done_df[
            [
                "layers",
                "r0_delta_acc",
                "r4_base_acc",
                "r4_interv_acc",
                "r4_delta_acc",
                "r4_format_gap",
            ]
        ].sort_values("r4_delta_acc", ascending=False)
        print(show.to_string(index=False))
        best = show.iloc[0]
        print(
            f"\nBest by R4 ΔAcc: layers={best['layers']}  "
            f"ΔAcc={best['r4_delta_acc']}  interv={best['r4_interv_acc']}"
        )
        hit = show[show["layers"] == "20_21_22"]
        if not hit.empty:
            r = hit.iloc[0]
            print(
                f"Primary window 20_21_22: ΔAcc={r['r4_delta_acc']}  "
                f"interv={r['r4_interv_acc']}  R0Δ={r['r0_delta_acc']}"
            )
    print(f"\nCSV : {OVERVIEW_CSV}")
    print(f"JSON: {OVERVIEW_JSON}")
    return overview


def build_window_list(start: int, end: int, shallow_spotcheck: bool) -> list[list[int]]:
    all_wins = windows()
    wins = [w for w in all_wins if start <= w[0] <= end]
    if shallow_spotcheck:
        extra = [list(range(s, s + WINDOW)) for s in SHALLOW_SPOTCHECK_STARTS]
        merged: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for w in extra + wins:
            key = tuple(w)
            if key not in seen:
                seen.add(key)
                merged.append(w)
        wins = merged
    return wins


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Primary vectors: sliding 3-layer windows @ α=50 on primary MMLU test"
    )
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-skip-done", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument(
        "--start",
        type=int,
        default=DEFAULT_START,
        help=f"默认 {DEFAULT_START} → 从 {DEFAULT_START},{DEFAULT_START+1},{DEFAULT_START+2} 起",
    )
    p.add_argument("--end", type=int, default=N_LAYERS - WINDOW)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"默认 {DEFAULT_MAX_NEW_TOKENS}；设 0 表示不覆盖（用 config 4096）",
    )
    p.add_argument(
        "--shallow-spotcheck",
        action="store_true",
        help="额外跑浅层窗 (0,1,2) 与 (3,4,5)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not VECTOR_DIR.exists():
        raise FileNotFoundError(f"缺少向量目录: {VECTOR_DIR}")
    if not TEST_SET.exists():
        raise FileNotFoundError(f"缺少测集: {TEST_SET}")

    max_new_tokens = None if args.max_new_tokens == 0 else args.max_new_tokens
    wins = build_window_list(args.start, args.end, args.shallow_spotcheck)
    if not wins:
        raise ValueError(f"无窗格: start={args.start} end={args.end}")

    print(
        f"Primary trilayer sweep\n"
        f"  vectors : {VECTOR_DIR}\n"
        f"  test    : {TEST_SET}\n"
        f"  alpha   : {ALPHA}\n"
        f"  windows : {len(wins)}  ({layers_tag(wins[0])} ... {layers_tag(wins[-1])})\n"
        f"  start/end: {args.start}..{args.end}"
        f"{' + shallow spotcheck' if args.shallow_spotcheck else ''}\n"
        f"  max_new_tokens: {max_new_tokens}\n"
        f"  device  : {args.device}\n"
        f"  results : {RESULTS_ROOT}/*_{SWEEP_TAG}\n"
        f"  baseline REF: {REF_BASELINE_DIR}"
    )

    if args.summary_only:
        summarize(wins)
        return

    skip_done = not args.no_skip_done
    reuse = ensure_baseline(
        device=args.device,
        limit=args.limit,
        dry_run=args.dry_run,
        skip_done=skip_done,
        max_new_tokens=max_new_tokens,
    )

    for i, layers in enumerate(wins, 1):
        print(f"\n>>> [{i}/{len(wins)}] layers={layers}")
        run_one_window(
            layers,
            device=args.device,
            limit=args.limit,
            dry_run=args.dry_run,
            skip_done=skip_done,
            reuse_from=reuse,
            max_new_tokens=max_new_tokens,
        )

    summarize(wins)
    print("\n全部完成。用 --summary-only 可随时重新汇总。")


if __name__ == "__main__":
    main()
