#!/usr/bin/env python3
# sweep_single_layer.py
"""
单层 × alpha × {MMLU, CSQA} 扫描（模型只加载一次；baseline 每数据集只跑一次）。

协议与 step3 对齐:
  Think OFF / greedy / per-layer 向量 / intervene_from_round=0（含 Round0）

用法:
  cd code
  python sweep_single_layer.py --device 3
  python sweep_single_layer.py --device 3 --limit 5          # 调试
  python sweep_single_layer.py --summary-only               # 只汇总
  python sweep_single_layer.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


def _lazy_imports():
    global config, torch, tqdm, LocalModel
    global ARMS, build_comparison_by_round, compute_arm_tables, default_output_dir
    global load_json, load_layer_vectors, model_short_name, resolve_context_limit
    global run_multi_round, save_arm_tables, ROOT, VECTOR_DIR, SWEEP_DIR, DATASETS

    import config as _config
    import torch as _torch
    from tqdm import tqdm as _tqdm
    from backend import LocalModel as _LocalModel
    from step3_run_intervention_test import (
        ARMS as _ARMS,
        build_comparison_by_round as _build_comparison_by_round,
        compute_arm_tables as _compute_arm_tables,
        default_output_dir as _default_output_dir,
        load_json as _load_json,
        load_layer_vectors as _load_layer_vectors,
        model_short_name as _model_short_name,
        resolve_context_limit as _resolve_context_limit,
        run_multi_round as _run_multi_round,
        save_arm_tables as _save_arm_tables,
    )

    config = _config
    torch = _torch
    tqdm = _tqdm
    LocalModel = _LocalModel
    ARMS = _ARMS
    build_comparison_by_round = _build_comparison_by_round
    compute_arm_tables = _compute_arm_tables
    default_output_dir = _default_output_dir
    load_json = _load_json
    load_layer_vectors = _load_layer_vectors
    model_short_name = _model_short_name
    resolve_context_limit = _resolve_context_limit
    run_multi_round = _run_multi_round
    save_arm_tables = _save_arm_tables

    ROOT = config.PROJECT_ROOT
    VECTOR_DIR = ROOT / "repeng_data" / "Qwen3-8B_multi_nothink" / "04_vectors"
    SWEEP_DIR = ROOT / "repeng_data" / "Qwen3-8B_multi_nothink" / "05_sweeps"
    DATASETS = {
        "mmlu_dev": ROOT
        / "repeng_data"
        / "Qwen3-8B_multi_nothink"
        / "03_splits"
        / "test_set_mmlu.json",
        "commonsenseqa_dev": ROOT
        / "repeng_data"
        / "Qwen3-8B_multi_nothink"
        / "03_splits"
        / "test_set_csqa.json",
    }


# 占位路径；_lazy_imports 后会用 config.PROJECT_ROOT 覆盖
ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "repeng_data" / "Qwen3-8B_multi_nothink" / "04_vectors"
SWEEP_DIR = ROOT / "repeng_data" / "Qwen3-8B_multi_nothink" / "05_sweeps"
DATASETS = {
    "mmlu_dev": ROOT
    / "repeng_data"
    / "Qwen3-8B_multi_nothink"
    / "03_splits"
    / "test_set_mmlu.json",
    "commonsenseqa_dev": ROOT
    / "repeng_data"
    / "Qwen3-8B_multi_nothink"
    / "03_splits"
    / "test_set_csqa.json",
}

# 单层网格：浅/中/深 + 加密中层与末层 EVR 峰
DEFAULT_LAYERS = [0, 5, 10, 15, 18, 20, 22, 25, 28, 30, 33, 35]
DEFAULT_ALPHAS = [10.0, 20.0, 30.0, 40.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="单层干预网格扫描（单次加载模型）")
    p.add_argument("--device", default="3")
    p.add_argument("--vector-dir", type=Path, default=VECTOR_DIR)
    p.add_argument(
        "--layers",
        type=str,
        default=",".join(map(str, DEFAULT_LAYERS)),
        help="逗号分隔单层列表",
    )
    p.add_argument(
        "--alphas",
        type=str,
        default=",".join(str(a) for a in DEFAULT_ALPHAS),
        help="逗号分隔 alpha 列表",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="mmlu_dev,commonsenseqa_dev",
        help="逗号分隔 dataset_tag",
    )
    p.add_argument("--decoding", choices=["greedy", "sampling"], default="greedy")
    p.add_argument("--intervene-from-round", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-done", dest="skip_done", action="store_true", default=True)
    p.add_argument("--no-skip-done", dest="skip_done", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument(
        "--model-path",
        default=None,
        help="默认从向量 .pt 读取",
    )
    return p.parse_args()


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def out_dir_for(
    *,
    model_name: str,
    layer: int,
    alpha: float,
    enable_thinking: bool,
    decoding: str,
    dataset_tag: str,
    intervene_from_round: int,
) -> Path:
    return default_output_dir(
        model_name=model_name,
        layers=[layer],
        alpha=alpha,
        enable_thinking=enable_thinking,
        decoding=decoding,
        dataset_tag=dataset_tag,
        mode_suffix="_perlayer",
        intervene_from_round=intervene_from_round,
    )


def save_run(
    *,
    output_dir: Path,
    detail_df: pd.DataFrame,
    run_config: dict,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(output_dir / "detail.csv", index=False, encoding="utf-8-sig")
    for arm in ARMS:
        arm_df = detail_df[detail_df["arm"] == arm].copy()
        save_arm_tables(output_dir, arm, compute_arm_tables(arm_df))
    comparison = build_comparison_by_round(detail_df)
    comparison.to_csv(
        output_dir / "comparison_by_round.csv", index=False, encoding="utf-8-sig"
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=4)
    return comparison


def run_baseline_cached(
    *,
    model: LocalModel,
    items: list[dict],
    model_name: str,
    enable_thinking: bool,
    context_limit: int,
    sampling: dict | None,
    cache_path: Path,
    force: bool = False,
) -> list[dict]:
    if cache_path.exists() and not force:
        print(f"  baseline cache hit: {cache_path}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for item_index, item in enumerate(tqdm(items, desc="baseline_once")):
        rows.extend(
            run_multi_round(
                model,
                item,
                item_index=item_index,
                model_name=model_name,
                arm="baseline",
                enable_thinking=enable_thinking,
                context_limit=context_limit,
                sampling=sampling,
                intervention=None,
            )
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"  baseline cached -> {cache_path}")
    return rows


def summarize_results(root: Path = ROOT) -> Path | None:
    results = root / "results"
    out_dir = SWEEP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cmp in results.rglob("comparison_by_round.csv"):
        parent = cmp.parent
        name = parent.name
        if "_repeng_layer" not in name or "_fromR" not in name:
            continue
        m2 = re.search(r"_repeng_layer(\d+(?:_\d+)*)_alpha", name)
        if not m2 or "_" in m2.group(1):
            continue
        layer = int(m2.group(1))
        cfg = {}
        cfg_path = parent / "run_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        layers = cfg.get("layers")
        if layers is not None and list(layers) != [layer]:
            continue
        df = pd.read_csv(cmp)
        alpha = cfg.get("alpha")
        if alpha is None:
            am = re.search(r"_alpha([0-9.]+)_", name)
            alpha = float(am.group(1)) if am else None
        r0 = df[df["round"] == 0].iloc[0]
        r4 = df[df["round"] == 4].iloc[0]
        gap4 = (
            100.0
            - float(r4["intervention_acc(%)"])
            - float(r4["intervention_flip_rate(%)"])
        )
        rows.append(
            {
                "dataset": parent.parent.name,
                "layer": layer,
                "alpha": float(alpha),
                "n_items": cfg.get("n_items"),
                "intervene_from_round": cfg.get("intervene_from_round"),
                "mode": (cfg.get("vector_meta") or {}).get("mode"),
                "r0_delta_acc": float(r0["delta_acc(%)"]),
                "r4_delta_acc": float(r4["delta_acc(%)"]),
                "r4_interv_acc": float(r4["intervention_acc(%)"]),
                "r4_interv_flip": float(r4["intervention_flip_rate(%)"]),
                "r4_unk_gap": round(gap4, 2),
                "run_dir": str(parent),
            }
        )
    if not rows:
        print("未找到单层 fromR* 结果。")
        return None
    out = pd.DataFrame(rows).sort_values(["dataset", "layer", "alpha"])
    csv_path = out_dir / "single_layer_summary.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"wrote {csv_path} ({len(out)} runs)")
    for ds, g in out.groupby("dataset"):
        print(f"\n[{ds}] R4 Δacc Top-5")
        print(
            g.sort_values("r4_delta_acc", ascending=False)
            .head(5)[
                ["layer", "alpha", "r0_delta_acc", "r4_delta_acc", "r4_interv_acc", "r4_unk_gap"]
            ]
            .to_string(index=False)
        )
        safe = g[g["r0_delta_acc"].abs() <= 1.0]
        print(f"[{ds}] |R0 Δ|≤1 的 R4 Top-5")
        if safe.empty:
            print("  (无)")
        else:
            print(
                safe.sort_values("r4_delta_acc", ascending=False)
                .head(5)[
                    [
                        "layer",
                        "alpha",
                        "r0_delta_acc",
                        "r4_delta_acc",
                        "r4_interv_acc",
                        "r4_unk_gap",
                    ]
                ]
                .to_string(index=False)
            )
    g30 = out[out["alpha"] == 30.0]
    if not g30.empty:
        print("\n=== alpha=30 层剖面 ===")
        for ds, g in g30.groupby("dataset"):
            print(f"\n[{ds}]")
            print(
                g.sort_values("layer")[
                    ["layer", "r0_delta_acc", "r4_delta_acc", "r4_interv_acc", "r4_unk_gap"]
                ].to_string(index=False)
            )
    return csv_path


def main() -> None:
    args = parse_args()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        summarize_results()
        return

    layers = parse_int_list(args.layers)
    alphas = parse_float_list(args.alphas)
    dataset_tags = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for t in dataset_tags:
        if t not in DATASETS:
            raise SystemExit(f"未知 dataset_tag={t}，可选: {list(DATASETS)}")

    jobs = [(d, L, a) for d in dataset_tags for L in layers for a in alphas]
    print(
        f"网格: datasets={dataset_tags} layers={layers} alphas={alphas}\n"
        f"任务数={len(jobs)} | device={args.device} | fromR={args.intervene_from_round} | "
        f"skip_done={args.skip_done}"
    )
    if args.dry_run:
        for d, L, a in jobs:
            print(f"  would run {d} L{L} α={a}")
        return

    _lazy_imports()
    vector_dir = Path(args.vector_dir)
    if not vector_dir.is_absolute():
        vector_dir = (Path.cwd() / vector_dir).resolve()
    args.vector_dir = vector_dir

    probe = args.vector_dir / f"sycophancy_vector_layer{layers[0]}.pt"
    payload = torch.load(probe, map_location="cpu", weights_only=False)
    model_path = args.model_path or payload.get("model_path", config.MODEL_PATH)
    model_name = model_short_name(model_path)

    device = config.normalize_device(args.device)
    config.set_seed()
    model = LocalModel(model_path, device=device)
    context_limit = resolve_context_limit(model)
    sampling = config.GREEDY_DECODING if args.decoding == "greedy" else None
    enable_thinking = False

    # 预加载本扫描用到的全部单层向量
    all_vecs, vec_meta = load_layer_vectors(
        layers=layers,
        vector_path=None,
        vector_dir=args.vector_dir,
        bundle_path=None,
    )
    print(f"向量: mode={vec_meta['mode']} | n_layers_loaded={len(all_vecs)}")

    overview = []
    for dataset_tag in dataset_tags:
        test_path = DATASETS[dataset_tag]
        items = load_json(test_path.resolve())
        if args.limit:
            items = items[: args.limit]
        print(f"\n==== dataset={dataset_tag} n={len(items)} ====")

        cache_name = f"baseline_{dataset_tag}_n{len(items)}_nothink_{args.decoding}.json"
        baseline_rows = run_baseline_cached(
            model=model,
            items=items,
            model_name=model_name,
            enable_thinking=enable_thinking,
            context_limit=context_limit,
            sampling=sampling,
            cache_path=SWEEP_DIR / "baseline_cache" / cache_name,
            force=bool(args.limit),  # limit 调试不复用全量 cache
        )

        for layer in layers:
            for alpha in alphas:
                out_dir = out_dir_for(
                    model_name=model_name,
                    layer=layer,
                    alpha=alpha,
                    enable_thinking=enable_thinking,
                    decoding=args.decoding,
                    dataset_tag=dataset_tag,
                    intervene_from_round=args.intervene_from_round,
                )
                if args.skip_done and (out_dir / "comparison_by_round.csv").exists():
                    print(f"SKIP {dataset_tag} L{layer} α={alpha}")
                    overview.append(
                        {
                            "dataset": dataset_tag,
                            "layer": layer,
                            "alpha": alpha,
                            "status": "skip",
                            "out_dir": str(out_dir),
                        }
                    )
                    continue

                print(f"\n>>> {dataset_tag} | layer={layer} | alpha={alpha}")
                print(f"    out: {out_dir}")
                intervention_cfg = {
                    "vectors": {layer: all_vecs[layer]},
                    "layers": [layer],
                    "alpha": alpha,
                    "intervene_from_round": args.intervene_from_round,
                }
                interv_rows: list[dict] = []
                for item_index, item in enumerate(
                    tqdm(items, desc=f"interv L{layer} a{alpha}")
                ):
                    interv_rows.extend(
                        run_multi_round(
                            model,
                            item,
                            item_index=item_index,
                            model_name=model_name,
                            arm="intervention",
                            enable_thinking=enable_thinking,
                            context_limit=context_limit,
                            sampling=sampling,
                            intervention=intervention_cfg,
                        )
                    )

                detail_df = pd.DataFrame(baseline_rows + interv_rows)
                run_config = {
                    "vector_meta": vec_meta,
                    "test_set": str(test_path.resolve()),
                    "dataset_tag": dataset_tag,
                    "model_path": model_path,
                    "model_name": model_name,
                    "layers": [layer],
                    "alpha": alpha,
                    "intervene_from_round": args.intervene_from_round,
                    "enable_thinking": enable_thinking,
                    "decoding": args.decoding,
                    "device": device,
                    "limit": args.limit,
                    "n_items": len(items),
                    "output_dir": str(out_dir),
                    "sweep": "single_layer",
                    "baseline_cache": str(
                        SWEEP_DIR / "baseline_cache" / cache_name
                    ),
                }
                comparison = save_run(
                    output_dir=out_dir, detail_df=detail_df, run_config=run_config
                )
                r0 = comparison[comparison["round"] == 0].iloc[0]
                r4 = comparison[comparison["round"] == 4].iloc[0]
                print(
                    f"    R0 Δacc={r0['delta_acc(%)']} | "
                    f"R4 Δacc={r4['delta_acc(%)']} interv_acc={r4['intervention_acc(%)']}"
                )
                overview.append(
                    {
                        "dataset": dataset_tag,
                        "layer": layer,
                        "alpha": alpha,
                        "status": "ok",
                        "r0_delta_acc": float(r0["delta_acc(%)"]),
                        "r4_delta_acc": float(r4["delta_acc(%)"]),
                        "r4_interv_acc": float(r4["intervention_acc(%)"]),
                        "out_dir": str(out_dir),
                    }
                )

    overview_path = SWEEP_DIR / "single_layer_overview.json"
    overview_path.write_text(
        json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\noverview -> {overview_path}")
    summarize_results()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断：可用 --skip-done 续跑；再 --summary-only 汇总。", file=sys.stderr)
        raise
