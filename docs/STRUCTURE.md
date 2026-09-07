# Public repository layout

This tree (`ANCHOR_public`) is the **minimal reproducible** subset for GitHub. The full local experiment repo is larger (multi-model sweeps, per-item trajectories, build artifacts).

## Top-level overview

```text
ANCHOR_public/
├── README.md              # Overview and quick start
├── LICENSE
├── .gitignore
├── .env.example           # Model / API path template (do not commit real secrets)
├── requirements.txt
├── docs/                  # Structure, exclusions, reproduce, release notes
├── assets/                # Framework overview figure for README
├── code/                  # Transformers: Gate → vector → intervention
├── code2/                 # Phenomenon measurement (vLLM/API; no intervention)
├── dataset/               # Phenomenon summary CSVs (Qwen3-8B multi only)
├── filtered_data/         # Filter outputs such as correct.json (Qwen3-8B only)
├── repeng_data/           # White-box primary tree: candidates / gate / splits / vectors
└── results/               # Primary intervention summaries (L20–22 α=50)
```

Data is split across `dataset/`, `filtered_data/`, `repeng_data/`, and `results/` so script default paths stay stable and avoid path-related regressions.

## Frozen primary protocol

| Item | Value |
|------|-------|
| Model | Qwen3-8B |
| Protocol | multi + no-think |
| White-box tree | `repeng_data/Qwen3-8B_multi_nothink/` |
| Primary intervention | layers `20,21,22`, `alpha=50`, `fromR0`, per-layer vectors |

## Directory responsibilities

| Path | Kept | Not kept |
|------|------|----------|
| `code/` | Gate / split / vector / intervention `.py` + primary sweep scripts | Gemma / multi-model shells, ablations, defense baselines, `__pycache__` |
| `code2/` | Filter and drift entry scripts | Local `results/` / `filtered_data/` piles, private doc copies |
| `repeng_data/` | Only `Qwen3-8B_multi_nothink` stages 00–04 | Other model trees, E-ablations, swap, `05_sweeps` |
| `dataset/` | `*/Qwen3-8B` summary CSVs | think / single / Gemma matrices |
| `filtered_data/` | `*/Qwen3-8B` | Other models / think |
| `results/` | Primary-run summaries + `stats/` | Sweeps, ablations, defenses, Gemma grids |
| `assets/` | README framework figure | — |

Commands: `repeng_data/PIPELINE.md` and the root `README.md`.
