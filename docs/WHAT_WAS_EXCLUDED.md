# What is missing vs the full ANCHOR repo

This public tree was copied from the full local experiment repo and then **slimmed**. The items below remain in the full repo (or can be regenerated) and are **not** shipped here.

## Intentionally excluded

| Excluded | Why | How to obtain / reproduce |
|----------|-----|---------------------------|
| `detail.csv` / `detail.jsonl` / `code2_private_archive/` | Per-item trajectories are huge (GB-scale) | Re-run `code2/run_drift.py` or `code/step3_*.py` |
| `raw_data/`, `*.parquet` | Third-party corpus license and size | Hugging Face / `code2/cache_datasets.py` |
| Multi-model white-box trees (Gemma / Qwen3-4B / 14B / single / fullpool) | Outside the primary protocol | Full repo or rebuild via `PIPELINE.md` |
| E-ablation, swap, window-sweep CSV/result trees | Exploratory / appendix noise | Full repo or re-run `sweep_*.py` |
| Gemma / defense-baseline / OOD full-dev scripts and results | Not part of the minimal primary set | Full repo |
| LaTeX manuscript sources, compiled PDFs, draft figures | Not part of this code release | Keep in the private full repo if needed |
| Model weights | Size and license | Point `ANCHOR_MODEL_PATH` at local HF weights |

## Still included (for primary-table comparison)

- `repeng_data/Qwen3-8B_multi_nothink/`: candidates, gate, splits, `04_vectors` (incl. L20–22)
- Qwen3-8B summaries under `dataset/` and `filtered_data/`
- Primary-run summaries under `results/.../Qwen3-8B_repeng_layer20_21_22_alpha50.0_*`
- Runnable entry points in `code/` and `code2/`

**Bottom line**: Enough to match primary-table numbers and re-run Gate → vector → intervention on local weights; not enough to offline-reproduce every appendix sweep or multi-model table.
