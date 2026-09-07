# repeng_data/

Root for white-box representation-engineering artifacts. This public tree keeps **only** the primary protocol:

```text
Qwen3-8B_multi_nothink/
  00_meta/
  01_candidates/     # Deep-flip candidate pools
  02_gate/           # Transformers admission: valid / rejected / summary
  03_splits/         # extraction_set_mmlu / test_set_mmlu / test_set_csqa
  04_vectors/        # Per-layer sycophancy vectors (.pt) + EVR summaries
```

Authoritative re-run commands: [`PIPELINE.md`](PIPELINE.md).

Other model trees, E-ablations, swap branches, and sweep overviews were removed from the public release; see `docs/WHAT_WAS_EXCLUDED.md`.
