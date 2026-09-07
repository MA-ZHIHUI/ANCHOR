# code2/ — Phenomenon measurement (filter / multi-turn drift)

Use vLLM (or another OpenAI-compatible API) to measure **epistemic drift / sycophancy**.  
This package does **not** include representation intervention; intervention lives in sibling `../code/`.

## Dependencies

```bash
pip install -r ../requirements.txt
# Also need a reachable OpenAI-compatible chat endpoint (e.g. vLLM)
```

Environment variables: repo-root `.env.example` (`VLLM_BASE_URL` / `VLLM_API_KEY` / `VLLM_MODEL`).

## Typical flow

1. Download benchmarks yourself: `python cache_datasets.py` (writes local `raw_data/`, gitignored)
2. Keep Round0-correct items: `filter_mmlu.py` / `filter_commonsenseqa.py` / `filter_sciq.py`
3. Multi-turn drift: `run_drift.py`; single-turn control: `run_benchmark.py`

This public tree does **not** ship `code2/results` or a local `filtered_data` pile. Primary filter outputs live at repo-root `../filtered_data/Qwen3-8B/`; phenomenon summaries live in `../dataset/`.

More exclusions: `../docs/WHAT_WAS_EXCLUDED.md`.
