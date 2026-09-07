# p_gate_splits.py
"""
ANCHOR 流水线第 2 步：对 01_candidates 全池做 Transformers 准入（Gate）。

协议顺序（论文口径）:
  P0 候选池 → **Gate** → Split → Phase2 提向量 → Phase3 干预

输出:
  repeng_data/<run>/02_gate/
    mmlu/     # validate_candidates 标准产出（含 valid.json）
    csqa/
    gate_overview.json

建议与 Phase3 一致: --decoding greedy --no-enable-thinking

用法:
  python p_gate_splits.py --device 3
  python p_gate_splits.py --limit-per-pool 2          # 冒烟
  python p_gate_splits.py --pools mmlu                # 只跑 MMLU
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import config

# =====================================================================
REPENG_ROOT = config.PROJECT_ROOT / "repeng_data" / "Qwen3-8B_multi_nothink"
MODEL_PATH = config.MODEL_PATH
DEVICE = config.DEVICE
DECODING = "greedy"
ENABLE_THINKING = False
LIMIT_PER_POOL = None

# 相对 01_candidates/ 的候选文件 → 02_gate/ 子目录
POOLS = (
    ("mmlu_deep_flip.json", "mmlu"),
    ("csqa_deep_flip.json", "csqa"),
)
# =====================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="对 01_candidates 全池做 Transformers Gate（先于 Split）"
    )
    p.add_argument("--repeng-root", type=Path, default=REPENG_ROOT)
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument(
        "--model",
        default=None,
        help="模型短名写入 detail（默认取 model-path 末级目录名）",
    )
    p.add_argument("--device", default=DEVICE, help='GPU，如 "cuda:3" 或 3')
    p.add_argument("--decoding", choices=["greedy", "sampling"], default=DECODING)
    p.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    p.add_argument(
        "--no-enable-thinking",
        dest="enable_thinking",
        action="store_false",
    )
    p.set_defaults(enable_thinking=ENABLE_THINKING)
    p.add_argument(
        "--limit-per-pool",
        type=int,
        default=LIMIT_PER_POOL,
        help="每个候选池只跑前 N 题（冒烟）；全量保持默认 None",
    )
    p.add_argument(
        "--pools",
        nargs="*",
        default=None,
        help="只跑指定子目录名，如 mmlu csqa",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repeng_root.resolve()
    cand_dir = root / "01_candidates"
    gate_root = root / "02_gate"
    gate_root.mkdir(parents=True, exist_ok=True)
    device = config.normalize_device(args.device)
    model_name = args.model or Path(str(args.model_path).rstrip("/")).name
    code_dir = Path(__file__).resolve().parent

    if not cand_dir.exists():
        raise FileNotFoundError(
            f"未找到候选池目录: {cand_dir}\n请先运行 p0_build_diagnostic_pool.py"
        )

    overview = {
        "protocol": "gate_before_split",
        "repeng_root": str(root),
        "device": device,
        "decoding": args.decoding,
        "enable_thinking": bool(args.enable_thinking),
        "model_path": args.model_path,
        "runs": [],
    }
    print(f"Gate（全池准入）设备: {device}")
    print(f"输出根目录: {gate_root}")

    for fname, sub in POOLS:
        if args.pools and sub not in args.pools:
            continue
        cand = cand_dir / fname
        if not cand.exists():
            raise FileNotFoundError(cand)
        out = gate_root / sub
        cmd = [
            sys.executable,
            str(code_dir / "validate_candidates.py"),
            "--candidates",
            str(cand),
            "--output-dir",
            str(out),
            "--model-path",
            args.model_path,
            "--model",
            model_name,
            "--device",
            device,
            "--decoding",
            args.decoding,
            "--flip-column",
            "final_flipped",
        ]
        if args.enable_thinking:
            cmd.append("--enable-thinking")
        else:
            cmd.append("--no-enable-thinking")
        if args.limit_per_pool is not None:
            cmd.extend(["--limit", str(args.limit_per_pool)])

        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(code_dir))

        summary_path = out / "gate_summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else {}
        )
        overview["runs"].append({"pool": sub, "candidates": str(cand), "summary": summary})

    overview["next_step"] = (
        f"python p1_split_diagnostic.py --repeng-root {root}"
    )
    overview_path = gate_root / "gate_overview.json"
    overview_path.write_text(
        json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nGate 编排完成:", overview_path)
    print("下一步:", overview["next_step"])


if __name__ == "__main__":
    main()
