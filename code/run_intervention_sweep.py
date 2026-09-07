# run_intervention_sweep.py
"""小批量消融编排：在 03_splits/test_set_mmlu 上扫层与 alpha（per-layer + fromR0）。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import config

VECTOR_DIR = (
    config.PROJECT_ROOT
    / "repeng_data"
    / "Qwen3-8B_multi_nothink"
    / "04_vectors"
)
TEST_SET = (
    config.PROJECT_ROOT
    / "repeng_data"
    / "Qwen3-8B_multi_nothink"
    / "03_splits"
    / "test_set_mmlu.json"
)

# (layers_str, alpha)
SWEEP = [
    ("20", 5.0),
    ("20", 15.0),
    ("20", 30.0),
    ("18,19,20", 5.0),
    ("18,19,20", 15.0),
    ("18,19,20", 30.0),
    ("28,29,30", 15.0),
    ("28,29,30", 30.0),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="3")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    code = Path(__file__).resolve().parent
    overview = []
    for layers, alpha in SWEEP:
        cmd = [
            sys.executable,
            str(code / "step3_run_intervention_test.py"),
            "--vector-dir",
            str(VECTOR_DIR),
            "--test-set",
            str(TEST_SET),
            "--layers",
            layers,
            "--alpha",
            str(alpha),
            "--decoding",
            "greedy",
            "--no-enable-thinking",
            "--intervene-from-round",
            "0",
            "--device",
            str(args.device),
        ]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        print("\n" + "=" * 70)
        print("RUN:", " ".join(cmd))
        print("=" * 70)
        subprocess.run(cmd, check=True, cwd=str(code))
        overview.append({"layers": layers, "alpha": alpha})

    out = VECTOR_DIR.parent / "05_sweeps" / "intervention_sweep_overview.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(overview, indent=2), encoding="utf-8")
    print("sweep done", out)


if __name__ == "__main__":
    main()
