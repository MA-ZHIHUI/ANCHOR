# ANCHOR

**Representation-level control against multi-turn epistemic drift in LLMs.**

When social pressure escalates across dialogue turns, models that answer correctly at Round&nbsp;0 can abandon the gold answer and endorse a misleading option. ANCHOR **measures** that drift and **mitigates** it by steering intermediate-layer activations—no fine-tuning required.

<p align="center">
  <img src="assets/fig1.png" alt="ANCHOR overall framework" width="92%" />
</p>
<p align="center"><em>Figure 1. ANCHOR overall framework: multi-turn drift → measurement (EDI / deep-flip) → gate &amp; splits → residual scrubbing → paired evaluation.</em></p>

---

## What is ANCHOR?

**Problem.** Under multi-turn rhetoric (praise → vanity → pressure → strong vanity), an LLM’s belief mass can shift from the gold answer \(y^*\) toward a misleading target \(y^\dagger\). Multi-turn pressure typically induces stronger drift than matched single-turn prompts (Think-OFF).

**Approach.** ANCHOR separates *discovery* from *causal evaluation*:

1. **Discover** items that start correct and end flipped (*deep-flip* candidates).
2. **Gate** those candidates on the same Transformers stack used for intervention.
3. **Extract** layer-wise sycophancy directions via contrastive PCA on an extraction set.
4. **Intervene** by residual scrubbing \(h_L \leftarrow h_L - \alpha\, v_L\) on selected mid-depth layers, evaluated with paired Baseline vs Intervention runs.

Primary results in this release use **Qwen3-8B**, **multi-turn + no-think**, vectors from **MMLU**, and OOD transfer to **CommonsenseQA**.

---

## Two-stage pipeline

```text
 Stage 1 — Discovery (phenomenon)              Stage 2 — Causal evaluation (RepEng)
 ─────────────────────────────────             ─────────────────────────────────────
  Filter Round-0–correct items                  Gate: same-stack replay → D_valid
  Multi-turn / single-turn induction            Split: extract / ID test / OOD test
  Score EDI, flip, deep-flip                    Extract per-layer sycophancy vectors
  Build deep-flip candidate pool D_cand         Intervene + paired A/B tables
       │                                              ▲
       └──────────── candidates feed gate ────────────┘
```

| Set | Artifact | Role |
|-----|----------|------|
| \(D_{\text{cand}}\) | `repeng_data/.../01_candidates/` | Deep-flip pool from phenomenon screening |
| \(D_{\text{valid}}\) | `repeng_data/.../02_gate/*/valid.json` | Items that remain Round-0–correct **and** flip under Transformers replay |
| Splits | `03_splits/` | Extraction (learn only) vs ID / OOD test (eval only) |

**Primary intervention tables use only \(D_{\text{valid}}\) splits.** Stage&nbsp;1 shrinks the search space; Stage&nbsp;2 closes the causal loop on one stack.

<details>
<summary>Why gate before reporting intervention gains?</summary>

A Stage-1 flip on a screening engine (e.g. vLLM) does not automatically hold under Transformers. Feeding `D_cand` straight into intervention invites selection bias. The gate replays induction with no steering; only items that still satisfy Round&nbsp;0 correct ∧ induced flip enter the formal analysis set. Retention rates live in `gate_summary.json`.
</details>

---

## Repository layout

```text
ANCHOR_public/
├── code/           # Transformers: Gate → vectors → intervention
├── code2/          # Phenomenon measurement (vLLM / OpenAI-compatible API)
├── repeng_data/    # Primary white-box tree (Qwen3-8B_multi_nothink)
├── dataset/        # Phenomenon summary CSVs (Qwen3-8B)
├── filtered_data/  # Filter outputs such as correct.json
├── results/        # Primary-run summaries (L20–22, α=50) + stats/
├── assets/         # Framework figure (fig1.png)
└── docs/           # Structure, reproduce path, exclusions
```

| Protocol item | Value in this tree |
|---------------|--------------------|
| Model | Qwen3-8B |
| Dialogue / think | multi-turn, **no-think** |
| White-box root | `repeng_data/Qwen3-8B_multi_nothink/` |
| Split quotas | extract **40** + MMLU test **40** + CSQA **300** = **380** |
| Primary intervention | layers `20,21,22`, `α=50`, intervene from Round&nbsp;0, per-layer vectors |
| OOD | CommonsenseQA (vectors from MMLU) |

Full directory roles: [`docs/STRUCTURE.md`](docs/STRUCTURE.md).

---

## Setup

```bash
cd <repo-root>
pip install -r requirements.txt
# Main deps: torch, transformers, pandas, numpy, scikit-learn, tqdm, pyarrow
```

Copy [`.env.example`](.env.example) and point at local Hugging Face–format weights (do **not** commit machine-local paths):

```bash
export ANCHOR_MODEL_PATH=/path/to/Qwen3-8B
export DEVICE=cuda:0          # optional; scripts also accept --device

# Optional — phenomenon measurement via code2/ (OpenAI-compatible endpoint)
# export VLLM_BASE_URL=http://127.0.0.1:8000/v1
# export VLLM_API_KEY=EMPTY
# export VLLM_MODEL=Qwen/Qwen3-8B
```

---

## Quickstart

Shipped artifacts already include gate outputs, splits, vectors, and primary-run CSVs. Shortest path: [`docs/REPRODUCE.md`](docs/REPRODUCE.md). Authoritative command sequence: [`repeng_data/PIPELINE.md`](repeng_data/PIPELINE.md).

```bash
cd <repo-root>/code
export ANCHOR_MODEL_PATH=/path/to/Qwen3-8B
```

### 1. Inspect or re-run intervention (primary protocol)

```bash
python step3_run_intervention_test.py \
  --vector-dir ../repeng_data/Qwen3-8B_multi_nothink/04_vectors \
  --test-set ../repeng_data/Qwen3-8B_multi_nothink/03_splits/test_set_mmlu.json \
  --layers 20,21,22 --alpha 50.0 \
  --decoding greedy --no-enable-thinking \
  --intervene-from-round 0
```

For OOD, point `--test-set` at `03_splits/test_set_csqa.json`. Compare against shipped summaries under `results/{mmlu,commonsenseqa}_dev/..._layer20_21_22_alpha50.0_*`.

### 2. Rebuild the white-box tree (Gate → Split → vectors)

```bash
# Candidates from dataset/ summaries (usually reuse 01_candidates/)
python p0_build_diagnostic_pool.py

# Transformers admission over the full candidate pools
python p_gate_splits.py \
  --repeng-root ../repeng_data/Qwen3-8B_multi_nothink \
  --decoding greedy --no-enable-thinking

# Leakage-controlled splits (40 / 40 / 300)
python p1_split_diagnostic.py \
  --repeng-root ../repeng_data/Qwen3-8B_multi_nothink

# Layer-wise sycophancy vectors
python phase2_layerwise_profiling.py \
  --extraction-set ../repeng_data/Qwen3-8B_multi_nothink/03_splits/extraction_set_mmlu.json \
  --output-dir ../repeng_data/Qwen3-8B_multi_nothink/04_vectors \
  --no-enable-thinking --window 3
```

### 3. Phenomenon measurement (`code2/`)

Optional rediscovery of drift curves via a vLLM (or compatible) endpoint: filter Round-0–correct items, then run multi-turn induction. See [`code2/README.md`](code2/README.md). This package does **not** perform representation intervention—that lives in `code/`.

---

## Key concepts & metrics

| Term | Meaning |
|------|---------|
| Rounds 0–4 | `baseline` → `praise_hint` → `vanity` → `pressure` → `strong_vanity` |
| Belief | Option probability at the decision token (from raw logits; decoupled from sampling temperature) |
| **EDI** | Epistemic Drift Index: \(P_4(y^\dagger) - P_0(y^\dagger)\) |
| Flip | Predicted option equals the misleading target |
| **Deep-flip** | Round&nbsp;0 correct (\(y_0 = y^*\)) and final round flipped (\(y_4 = y^\dagger\)) |
| α (alpha) | Intervention strength: subtract \(\alpha \times\) the layer sycophancy vector from residual stream activations |
| Multi-layer scrubbing | Same protocol hooked on several decoder layers (primary: 20–22) |
| Think mode | Hard switch via `apply_chat_template(enable_thinking=...)`; parse answers from `content` after `</think>` |

Primary table file: `comparison_by_round.csv` (Baseline vs Intervention by round). Paired statistics: `results/stats/`.

---

## What's included / what's not

| Included | Not in this public tree |
|----------|-------------------------|
| Runnable `code/` + `code2/` entry points | Multi-model white-box trees, large sweep grids |
| `repeng_data/Qwen3-8B_multi_nothink/` (candidates → gate → splits → vectors) | Per-item trajectory dumps (GB-scale) |
| Qwen3-8B summaries under `dataset/` / `filtered_data/` | Raw benchmark corpora (`raw_data/`) |
| Primary intervention summaries under `results/` | Ablations, defense baselines, appendix-only branches |
| Framework figure under `assets/` | Model weights |

Details: [`docs/WHAT_WAS_EXCLUDED.md`](docs/WHAT_WAS_EXCLUDED.md) · layout: [`docs/STRUCTURE.md`](docs/STRUCTURE.md) · release checklist: [`docs/GITHUB_RELEASE.md`](docs/GITHUB_RELEASE.md).

---

## License

[MIT](LICENSE) © 2026 ANCHOR authors.
