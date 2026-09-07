# ANCHOR / RepEng formal pipeline

> Protocol notes (2026-08): **Gate the full pool, then Split**; unified answer suffix; No-Think `max_new_tokens=4096`; intervention injects from Round0 by default.

Frozen: do **not** re-run the macroscopic vLLM matrix under `dataset/` unless you intentionally refresh phenomenon screening.

---

## 0. Environment

```bash
cd <repo-root>/code
conda activate ANCHOR
# set GPU as needed
export DEVICE=3
```

Root variables used below:

```bash
REPENG=../repeng_data/Qwen3-8B_multi_nothink
export ANCHOR_MODEL_PATH=/path/to/Qwen3-8B
MODEL="$ANCHOR_MODEL_PATH"
```

---

## 1. P0 — Candidate pool (usually reuse existing `01_candidates`)

Re-run only when changing model / protocol / flip definition:

```bash
python p0_build_diagnostic_pool.py \
  --model Qwen3-8B --protocol multi --no-enable-thinking
```

Output: `$REPENG/01_candidates/{mmlu,csqa}_deep_flip.json`

---

## 2. Gate — Transformers full-pool admission (**must re-run** if prompts / token budgets changed)

Greedy replay on the full MMLU (~108) + CSQA (~819) pools (this is the main runtime cost):

```bash
python p_gate_splits.py \
  --repeng-root "$REPENG" \
  --model-path "$MODEL" \
  --device "$DEVICE" \
  --decoding greedy \
  --no-enable-thinking
```

Smoke test:

```bash
python p_gate_splits.py --repeng-root "$REPENG" --device "$DEVICE" --limit-per-pool 2
```

Output: `$REPENG/02_gate/{mmlu,csqa}/valid.json` + `gate_overview.json`

---

## 3. Split — carve \(D_{\text{valid}}\) (**must re-run**)

Targets 40 / 60 / 300; if MMLU valid &lt; 100, add `--fill-available` (prefer extract first):

```bash
python p1_split_diagnostic.py \
  --repeng-root "$REPENG" \
  --n-extract-mmlu 40 \
  --n-test-mmlu 60 \
  --n-test-csqa 300 \
  --fill-available
```

Output: `$REPENG/03_splits/{extraction_set_mmlu,test_set_mmlu,test_set_csqa}.json`

> Legacy dirs `02_splits/` and `03_gate/` are historical — **do not** feed them into primary tables.

---

## 4. Phase 2 — Extract sycophancy vectors (**must re-run** if extract set / prompts changed)

```bash
python phase2_layerwise_profiling.py \
  --extraction-set "$REPENG/03_splits/extraction_set_mmlu.json" \
  --output-dir "$REPENG/04_vectors" \
  --model-path "$MODEL" \
  --device "$DEVICE" \
  --no-enable-thinking \
  --window 3
```

Output: `$REPENG/04_vectors/sycophancy_vector_layer{L}.pt` + `layerwise_evr.csv`, etc.

---

## 5. Phase 3 — Primary intervention (**must re-run** for fromR0 + new splits + new vectors)

### 5.1 ID primary config (MMLU)

```bash
python step3_run_intervention_test.py \
  --vector-dir "$REPENG/04_vectors" \
  --test-set "$REPENG/03_splits/test_set_mmlu.json" \
  --layers 18,19,20 \
  --alpha 30.0 \
  --decoding greedy \
  --no-enable-thinking \
  --intervene-from-round 0 \
  --device "$DEVICE"
```

### 5.2 OOD (CSQA) — suggested α grid 30 / 40

```bash
python step3_run_intervention_test.py \
  --vector-dir "$REPENG/04_vectors" \
  --test-set "$REPENG/03_splits/test_set_csqa.json" \
  --layers 18,19,20 \
  --alpha 40.0 \
  --decoding greedy \
  --no-enable-thinking \
  --intervene-from-round 0 \
  --device "$DEVICE"
```

### 5.3 Single-layer grid (optional; model loaded once)

```bash
python sweep_single_layer.py --device "$DEVICE"
# or
DEVICE=$DEVICE bash sweep_single_layer.sh
```

Test-set paths already point at `03_splits/`.

---

## 6. Protocol change log (vs older runs)

| Change | Impact |
|--------|--------|
| Gate → Split | Enough valid items to hit target N; paths are `02_gate` / `03_splits` |
| Unified answer suffix (no “forbid analysis”) | Generation distribution shifts; old Gate/intervention not directly comparable |
| `MAX_NEW_TOKENS_NO_THINK=4096` | Longer No-Think bodies allowed |
| `intervene_from_round=0` | Serum also at R0; run dirs contain `_fromR0` |
| `--vector-dir` per-layer | No sharing one best-layer vector across layers |

Therefore: older `03_gate` results, older `04_vectors`, and older `results/*` primary tables should be treated as exploratory; treat this pipeline as the source of truth for primary numbers.

---

## 7. Suggested archival

```bash
# Optional: rename legacy protocol dirs so they are not reused by mistake
cd "$REPENG"
# mv 02_splits 02_splits_LEGACY_split_then_gate
# mv 03_gate   03_gate_LEGACY_split_then_gate
# mv 04_vectors 04_vectors_LEGACY_pre_prompt_fix
```

(Only after confirming; prefer rename over delete for appendix contrasts.)

---

## Appendix: Split-swap robustness (side branch)

After the main split, you may run an MMLU extract↔test swap branch for an objective contrast (does not overwrite `03_splits`):

- Data / notes: `repeng_data/Qwen3-8B_multi_nothink/SWAP_ROBUSTNESS.md` (if present in your full tree)
- One-shot: `DEVICE=3 bash ../repeng_data/Qwen3-8B_multi_nothink/RUN_SWAP.sh`
