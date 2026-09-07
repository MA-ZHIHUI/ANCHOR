# GitHub public release notes

## What is uploaded

| Included | Notes |
|----------|-------|
| `code/` | Transformers primary pipeline (slimmed) |
| `code2/` | Phenomenon-measurement entry points (no local result piles) |
| `repeng_data/Qwen3-8B_multi_nothink/` | Primary gate / splits / vectors |
| `dataset/`, `filtered_data/`, `results/` | Primary summaries |
| `assets/` | Framework overview figure for README |
| Root `README` / `LICENSE` / `requirements.txt` / `.env.example` / `docs/` | Docs and license |

## What is not uploaded

See [`WHAT_WAS_EXCLUDED.md`](WHAT_WAS_EXCLUDED.md). `.gitignore` already blocks `detail.csv`, sweep trees, and similar artifacts.

## Size expectation

After slimming, the working tree is roughly tens of MB. Vector `.pt` files are only a few MB; Git LFS is usually unnecessary.

## Pre-push checks

```bash
cd /path/to/ANCHOR_public
export ANCHOR_MODEL_PATH=/path/to/Qwen3-8B

rg '/apps/users/|/home/' -g '!.env.example' || true
git init   # if not already initialized
git add -A
git status
git ls-files -z | xargs -0 du -ch | tail -1
```

Push from this public copy only — **do not** push the full experiment repository.
