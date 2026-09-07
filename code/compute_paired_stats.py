#!/usr/bin/env python3
"""Paired intervention / EDI statistics for the ANCHOR paper (no new GPU runs).

Outputs:
  results/stats/paired_intervention_stats.csv
  results/stats/edi_mt_st_permutation.csv
  results/stats/STATS_REPORT.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "stats"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
N_BOOT = 10000
N_PERM = 10000


def mcnemar_exact(b_correct: np.ndarray, i_correct: np.ndarray) -> dict:
    """Exact McNemar on paired binary outcomes (baseline vs intervention)."""
    b = b_correct.astype(bool)
    i = i_correct.astype(bool)
    n01 = int((~b & i).sum())  # baseline wrong, interv correct (rescues)
    n10 = int((b & ~i).sum())  # baseline correct, interv wrong
    n11 = int((b & i).sum())
    n00 = int((~b & ~i).sum())
    discordant = n01 + n10
    if discordant == 0:
        p = 1.0
    else:
        # exact binomial test under H0: P(rescue)=P(harm) among discordant
        p = float(binomtest(n01, discordant, 0.5, alternative="two-sided").pvalue)
    return {
        "n": int(len(b)),
        "n11": n11,
        "n00": n00,
        "n01_rescue": n01,
        "n10_harm": n10,
        "mcnemar_p": p,
    }


def bootstrap_delta_pp(
    b_correct: np.ndarray, i_correct: np.ndarray, n_boot: int = N_BOOT
) -> dict:
    """Paired bootstrap CI for ΔAcc in percentage points."""
    b = b_correct.astype(float)
    i = i_correct.astype(float)
    n = len(b)
    point = 100.0 * (i.mean() - b.mean())
    idx = RNG.integers(0, n, size=(n_boot, n))
    deltas = 100.0 * (i[idx].mean(axis=1) - b[idx].mean(axis=1))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta_acc_pp": float(point),
        "boot_ci95_lo": float(lo),
        "boot_ci95_hi": float(hi),
        "boot_mean": float(deltas.mean()),
    }


def load_round_correct(detail_path: Path, arm: str, round_idx: int = 4) -> pd.Series:
    df = pd.read_csv(detail_path)
    sub = df[(df["arm"].astype(str) == arm) & (df["round"] == round_idx)].copy()
    sub = sub.sort_values("item_index")
    return sub.set_index("item_index")["is_correct"].astype(bool)


def paired_from_two_details(
    base_detail: Path,
    interv_detail: Path,
    base_arm: str,
    interv_arm: str,
    round_idx: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    b = load_round_correct(base_detail, base_arm, round_idx)
    i = load_round_correct(interv_detail, interv_arm, round_idx)
    common = b.index.intersection(i.index)
    return b.loc[common].to_numpy(), i.loc[common].to_numpy()


def paired_from_one_detail(detail_path: Path, round_idx: int = 4) -> tuple[np.ndarray, np.ndarray]:
    b = load_round_correct(detail_path, "baseline", round_idx)
    i = load_round_correct(detail_path, "intervention", round_idx)
    common = b.index.intersection(i.index)
    return b.loc[common].to_numpy(), i.loc[common].to_numpy()


def summarize_pair(name: str, b: np.ndarray, i: np.ndarray) -> dict:
    row = {"setting": name}
    row.update(mcnemar_exact(b, i))
    row.update(bootstrap_delta_pp(b, i))
    row["base_acc"] = float(100.0 * b.mean())
    row["interv_acc"] = float(100.0 * i.mean())
    return row


def edi_mt_st_tests() -> pd.DataFrame:
    """Paired permutation / Wilcoxon-style sign on item-level EDI, Think-OFF."""
    rows = []
    pools = [
        ("mmlu_dev", "MMLU-dev"),
        ("commonsenseqa_dev", "CommonsenseQA-dev"),
        ("sciq_test", "SciQ-test"),
        ("sciq_valid", "SciQ-valid"),
    ]
    for pool_dir, label in pools:
        mt_path = ROOT / "dataset" / pool_dir / "Qwen3-8B" / "item_summary.csv"
        st_path = ROOT / "dataset" / pool_dir / "Qwen3-8B_single" / "item_summary.csv"
        if not mt_path.exists() or not st_path.exists():
            continue
        mt = pd.read_csv(mt_path)
        st = pd.read_csv(st_path)
        # align on subject+gt+misleading+round0 when possible; fall back to item_index
        key_cols = [c for c in ("subject", "gt", "misleading") if c in mt.columns and c in st.columns]
        if key_cols:
            mt = mt.copy()
            st = st.copy()
            mt["_k"] = mt[key_cols].astype(str).agg("||".join, axis=1)
            st["_k"] = st[key_cols].astype(str).agg("||".join, axis=1)
            common = sorted(set(mt["_k"]) & set(st["_k"]))
            mt_edi = mt.set_index("_k").loc[common, "edi"].to_numpy(dtype=float)
            st_edi = st.set_index("_k").loc[common, "edi"].to_numpy(dtype=float)
        else:
            common_idx = sorted(set(mt["item_index"]) & set(st["item_index"]))
            mt_edi = mt.set_index("item_index").loc[common_idx, "edi"].to_numpy(dtype=float)
            st_edi = st.set_index("item_index").loc[common_idx, "edi"].to_numpy(dtype=float)

        diff = mt_edi - st_edi  # positive => MT more drift
        obs = float(diff.mean())
        # paired permutation: randomly flip signs of paired diffs
        signs = RNG.choice([-1.0, 1.0], size=(N_PERM, len(diff)))
        null = (signs * diff).mean(axis=1)
        p_perm = float((np.abs(null) >= abs(obs)).mean())
        # bootstrap CI on mean EDI difference
        boot_idx = RNG.integers(0, len(diff), size=(N_BOOT, len(diff)))
        boot_means = diff[boot_idx].mean(axis=1)
        lo, hi = np.quantile(boot_means, [0.025, 0.975])
        rows.append(
            {
                "pool": label,
                "n_paired": int(len(diff)),
                "mean_edi_mt": float(mt_edi.mean()),
                "mean_edi_st": float(st_edi.mean()),
                "mean_diff_mt_minus_st": obs,
                "boot_ci95_lo": float(lo),
                "boot_ci95_hi": float(hi),
                "perm_p_two_sided": p_perm,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    mmlu = RES / "mmlu_dev"
    csqa = RES / "commonsenseqa_dev"
    ref = mmlu / "Qwen3-8B_REF_baseline_primary_test_mmlu_nothink_greedy_fromR0" / "detail.csv"
    primary = mmlu / "Qwen3-8B_repeng_layer20_21_22_alpha50.0_nothink_greedy_fromR0_perlayer" / "detail.csv"
    triwin = mmlu / "Qwen3-8B_repeng_layer20_21_22_alpha50.0_nothink_greedy_fromR0_perlayer_PRIMARY_TRIWIN" / "detail.csv"
    sysdef = mmlu / "Qwen3-8B_SYSDEF_primary_test_mmlu_nothink_greedy_fromR0" / "detail.csv"
    caa20 = mmlu / "Qwen3-8B_repeng_layer20_alpha50.0_nothink_greedy_fromR0_perlayer_CAA1L" / "detail.csv"
    ood = csqa / "Qwen3-8B_repeng_layer20_21_22_alpha50.0_nothink_greedy_fromR0_perlayer" / "detail.csv"

    rows = []

    # Canonical primary (joint baseline in same detail)
    b, i = paired_from_one_detail(primary)
    rows.append(summarize_pair("MMLU ID ANCHOR canonical (joint Baseline)", b, i))

    # Frozen-baseline view: REF baseline vs primary intervention arm
    b, i = paired_from_two_details(ref, primary, "baseline", "intervention")
    rows.append(summarize_pair("MMLU ID ANCHOR vs frozen REF Baseline", b, i))

    # TRIWIN frozen protocol (same detail has both arms under reused baseline)
    if triwin.exists():
        b, i = paired_from_one_detail(triwin)
        rows.append(summarize_pair("MMLU ID ANCHOR frozen-TRIWIN protocol", b, i))

    # Defense baselines vs frozen REF
    b, i = paired_from_two_details(ref, sysdef, "baseline", "prompt_defense")
    rows.append(summarize_pair("MMLU ID system-prompt vs frozen REF", b, i))

    b, i = paired_from_two_details(ref, caa20, "baseline", "intervention")
    rows.append(summarize_pair("MMLU ID single-layer L20 CAA vs frozen REF", b, i))

    # OOD
    b, i = paired_from_one_detail(ood)
    rows.append(summarize_pair("CSQA OOD ANCHOR canonical", b, i))

    # Extract-size ablation E30/E40/E60/E80 on CSQA (exploratory)
    for n in (30, 40, 60, 80):
        d = csqa / f"Qwen3-8B_E{n}_layer20_21_22_alpha50_nothink_greedy_fromR0_perlayer_EABLATE_csqa300"
        # E40 may be symlink to primary
        det = d / "detail.csv"
        if not det.exists() and n == 40:
            det = ood
        if not det.exists():
            continue
        # ablation reused baseline from primary OOD — intervention-only detail may exist
        df = pd.read_csv(det)
        arms = set(df["arm"].astype(str))
        if "baseline" in arms and "intervention" in arms:
            b, i = paired_from_one_detail(det)
        else:
            b, i = paired_from_two_details(ood, det, "baseline", "intervention")
        rows.append(summarize_pair(f"CSQA OOD extract-size |E|={n} (exploratory)", b, i))

    interv_df = pd.DataFrame(rows)
    interv_df.to_csv(OUT / "paired_intervention_stats.csv", index=False, encoding="utf-8-sig")

    edi_df = edi_mt_st_tests()
    edi_df.to_csv(OUT / "edi_mt_st_permutation.csv", index=False, encoding="utf-8-sig")

    # Markdown report for paper editing
    def df_block(df: pd.DataFrame) -> str:
        return df.to_string(index=False)

    lines = [
        "# ANCHOR paired statistics (seed=42)",
        "",
        f"Bootstrap resamples: {N_BOOT}; EDI permutation: {N_PERM}.",
        "",
        "## Intervention / defense (Round-4 Acc)",
        "",
        "```",
        df_block(interv_df),
        "```",
        "",
        "## EDI MT − ST (Think-OFF, item-paired)",
        "",
        "```",
        df_block(edi_df) if len(edi_df) else "no EDI files found",
        "```",
        "",
        "## Paper numbers to use",
        "",
    ]
    # pull key rows
    def pick(substr: str):
        hit = interv_df[interv_df["setting"].str.contains(substr, regex=False)]
        return hit.iloc[0] if len(hit) else None

    canon = pick("canonical (joint")
    frozen = pick("frozen REF Baseline")
    sysp = pick("system-prompt")
    caa = pick("single-layer")
    oodr = pick("CSQA OOD ANCHOR canonical")
    if frozen is not None:
        lines.append(
            f"- Defense table ANCHOR ΔAcc (frozen REF): "
            f"**{frozen['delta_acc_pp']:+.1f} pp** "
            f"(95% CI [{frozen['boot_ci95_lo']:.1f}, {frozen['boot_ci95_hi']:.1f}]; "
            f"McNemar p={frozen['mcnemar_p']:.2e}; "
            f"rescues={int(frozen['n01_rescue'])}, harms={int(frozen['n10_harm'])})."
        )
    if canon is not None:
        lines.append(
            f"- Canonical headline ΔAcc: **{canon['delta_acc_pp']:+.1f} pp** "
            f"(95% CI [{canon['boot_ci95_lo']:.1f}, {canon['boot_ci95_hi']:.1f}]; "
            f"McNemar p={canon['mcnemar_p']:.2e})."
        )
    if sysp is not None:
        lines.append(
            f"- System-prompt ΔAcc (frozen): **{sysp['delta_acc_pp']:+.1f} pp** "
            f"(95% CI [{sysp['boot_ci95_lo']:.1f}, {sysp['boot_ci95_hi']:.1f}]; "
            f"McNemar p={sysp['mcnemar_p']:.2e})."
        )
    if caa is not None:
        lines.append(
            f"- Best single-layer CAA ΔAcc (frozen): **{caa['delta_acc_pp']:+.1f} pp** "
            f"(95% CI [{caa['boot_ci95_lo']:.1f}, {caa['boot_ci95_hi']:.1f}]; "
            f"McNemar p={caa['mcnemar_p']:.2e})."
        )
    if oodr is not None:
        lines.append(
            f"- CSQA OOD ΔAcc: **{oodr['delta_acc_pp']:+.1f} pp** "
            f"(95% CI [{oodr['boot_ci95_lo']:.1f}, {oodr['boot_ci95_hi']:.1f}]; "
            f"McNemar p={oodr['mcnemar_p']:.2e})."
        )
    (OUT / "STATS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(interv_df.to_string(index=False))
    print()
    print(edi_df.to_string(index=False) if len(edi_df) else "no EDI")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
