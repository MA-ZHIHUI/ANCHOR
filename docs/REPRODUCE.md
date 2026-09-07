# Reproduce primary results (shortest path)

Prerequisites: local Qwen3-8B (HuggingFace format) and packages from `requirements.txt`.

```bash
cd <repo-root>
export ANCHOR_MODEL_PATH=/path/to/Qwen3-8B
export DEVICE=cuda:0
cd code
```

## When shipped data is enough (compare primary tables)

This public tree already includes:

- splits / gate / vectors: `../repeng_data/Qwen3-8B_multi_nothink/`
- primary-run summaries: `../results/{mmlu,commonsenseqa}_dev/Qwen3-8B_repeng_layer20_21_22_alpha50.0_nothink_greedy_fromR0_perlayer/`

Read those CSVs directly, or re-run intervention locally with the commands below.

## Re-run intervention (primary protocol)

```bash
python step3_run_intervention_test.py \
  --vector-dir ../repeng_data/Qwen3-8B_multi_nothink/04_vectors \
  --test-set ../repeng_data/Qwen3-8B_multi_nothink/03_splits/test_set_mmlu.json \
  --layers 20,21,22 --alpha 50.0 \
  --decoding greedy --no-enable-thinking \
  --intervene-from-round 0
```

For CSQA OOD, set `--test-set` to `03_splits/test_set_csqa.json`.

## Rebuild the white-box tree from scratch

Full commands: `../repeng_data/PIPELINE.md` (Gate → Split → vector → intervention). Phenomenon measurement can use `../code2/` (requires your own vLLM and benchmark corpora).
