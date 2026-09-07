# p1_split_diagnostic.py
"""
ANCHOR 流水线第 3 步：从 Gate 后的 D_valid 做诊断子集隔离切分（防泄漏）。

协议顺序（论文口径）:
  P0 候选池 → Gate → **Split** → Phase2 → Phase3

默认配额（合计 400；仅在对应 valid 池足够时保证）:
  extraction_set_mmlu.json : 40   仅 MMLU，提向量
  test_set_mmlu.json       : 60   同分布测试
  test_set_csqa.json       : 300  CSQA OOD 测试（永不进提取）

输入:
  02_gate/mmlu/valid.json
  02_gate/csqa/valid.json

输出:
  03_splits/{extraction_set_mmlu,test_set_mmlu,test_set_csqa}.json
  03_splits/split_manifest.json

若 Gate 后 MMLU valid < 100，可用:
  --fill-available   # 按「先抽满 extract，余量给 test」尽量填满并记录实际配额

用法:
  python p1_split_diagnostic.py --repeng-root ../repeng_data/Qwen3-8B_multi_nothink
  python p1_split_diagnostic.py --fill-available
  python p1_split_diagnostic.py --n-extract-mmlu 40 --n-test-mmlu 60 --n-test-csqa 300
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import config

# =====================================================================
REPENG_ROOT = config.PROJECT_ROOT / "repeng_data" / "Qwen3-8B_multi_nothink"

N_EXTRACT_MMLU = 40
N_TEST_MMLU = 60
N_TEST_CSQA = 300
SEED = config.SEED

GATE_MMLU = "mmlu/valid.json"
GATE_CSQA = "csqa/valid.json"
# =====================================================================


def load_items(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 JSON 数组: {path}")
    return data


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def qhash(item: dict) -> str:
    return hashlib.md5(item["question"].strip().encode("utf-8")).hexdigest()


def take_n(
    pool: list[dict],
    n: int,
    *,
    seed: int,
    pool_name: str,
    fill_available: bool,
) -> tuple[list[dict], list[dict], int]:
    """返回 (selected, unused, requested_or_actual_n)。"""
    if n < 0:
        raise ValueError(f"[{pool_name}] n 不能为负: {n}")
    if len(pool) < n:
        if not fill_available:
            raise ValueError(
                f"[{pool_name}] Gate-valid 池大小={len(pool)} < 需要={n}。"
                f" 请降低配额，或加 --fill-available，或扩大上游候选/提高 Gate 留存。"
            )
        print(
            f"⚠️ [{pool_name}] valid={len(pool)} < 请求={n}，"
            f"--fill-available：实际取 {len(pool)}"
        )
        n = len(pool)
    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    return shuffled[:n], shuffled[n:], n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="从 02_gate/*/valid.json 切分诊断子集（Gate→Split）"
    )
    p.add_argument("--repeng-root", type=Path, default=REPENG_ROOT)
    p.add_argument("--n-extract-mmlu", type=int, default=N_EXTRACT_MMLU)
    p.add_argument("--n-test-mmlu", type=int, default=N_TEST_MMLU)
    p.add_argument("--n-test-csqa", type=int, default=N_TEST_CSQA)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--fill-available",
        action="store_true",
        help="valid 不足目标配额时尽量填满（MMLU：先保证 extract，余量给 test）",
    )
    p.add_argument(
        "--gate-mmlu",
        default=GATE_MMLU,
        help="相对 02_gate/ 的 MMLU valid 路径",
    )
    p.add_argument(
        "--gate-csqa",
        default=GATE_CSQA,
        help="相对 02_gate/ 的 CSQA valid 路径",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repeng_root.resolve()
    gate_root = root / "02_gate"
    out = root / "03_splits"
    out.mkdir(parents=True, exist_ok=True)

    mmlu_path = gate_root / args.gate_mmlu
    csqa_path = gate_root / args.gate_csqa
    if not mmlu_path.exists():
        raise FileNotFoundError(
            "请先运行 p_gate_splits.py（Gate 全池准入）\n"
            f"缺少: {mmlu_path}\n"
            "旧协议目录 03_gate/（先 Split 再 Gate）已弃用，勿混用。"
        )
    if not csqa_path.exists():
        if args.n_test_csqa == 0:
            print(
                f"ℹ️ CSQA valid 尚未就绪 ({csqa_path})；"
                f"--n-test-csqa 0 → 先切 MMLU，OOD 留空待 Gate 后再补。"
            )
        else:
            raise FileNotFoundError(
                "请先运行 p_gate_splits.py（Gate 全池准入）\n"
                f"缺少: {csqa_path}\n"
                "若仅做 MMLU ID 线，可先加 --n-test-csqa 0。"
            )

    # 提示：若仍存在旧目录，提醒不要混用
    legacy_splits = root / "02_splits"
    legacy_gate = root / "03_gate"
    if legacy_splits.exists() or legacy_gate.exists():
        print(
            "ℹ️ 检测到旧协议目录 02_splits/ 和/或 03_gate/（Split→Gate）。"
            " 新协议使用 02_gate/ + 03_splits/；旧目录仅作历史归档，勿作主表输入。"
        )

    config.set_seed(args.seed)
    mmlu_pool = load_items(mmlu_path)
    csqa_pool = load_items(csqa_path) if csqa_path.exists() else []

    print(
        f"Gate-valid 池: MMLU={len(mmlu_pool)} | CSQA={len(csqa_pool)} | "
        f"目标配额 extract={args.n_extract_mmlu} "
        f"test_mmlu={args.n_test_mmlu} test_csqa={args.n_test_csqa}"
    )

    # ---- MMLU：先抽 extract+test 总量，再切段；不足时优先 extract ----
    n_mmlu_need = args.n_extract_mmlu + args.n_test_mmlu
    if args.fill_available and len(mmlu_pool) < n_mmlu_need:
        n_extract = min(args.n_extract_mmlu, len(mmlu_pool))
        n_test_m = max(0, len(mmlu_pool) - n_extract)
        print(
            f"⚠️ MMLU valid={len(mmlu_pool)} < 请求合计={n_mmlu_need}；"
            f"fill 策略 → extract={n_extract}, test_mmlu={n_test_m}"
        )
    else:
        n_extract = args.n_extract_mmlu
        n_test_m = args.n_test_mmlu
        n_mmlu_need = n_extract + n_test_m

    mmlu_selected, mmlu_unused, _ = take_n(
        mmlu_pool,
        n_extract + n_test_m,
        seed=args.seed,
        pool_name="MMLU_valid",
        fill_available=args.fill_available,
    )
    extract = mmlu_selected[:n_extract]
    test_mmlu = mmlu_selected[n_extract:]

    test_csqa, csqa_unused, n_csqa_actual = take_n(
        csqa_pool,
        args.n_test_csqa,
        seed=args.seed + 1,
        pool_name="CSQA_valid",
        fill_available=args.fill_available,
    )

    for it in extract:
        it["split"] = "extraction_mmlu"
        it["gate_source"] = "02_gate/mmlu/valid.json"
    for it in test_mmlu:
        it["split"] = "test_mmlu"
        it["gate_source"] = "02_gate/mmlu/valid.json"
    for it in test_csqa:
        it["split"] = "test_csqa"
        it["gate_source"] = "02_gate/csqa/valid.json"

    paths = {
        "extraction_set_mmlu": out / "extraction_set_mmlu.json",
        "test_set_mmlu": out / "test_set_mmlu.json",
        "test_set_csqa": out / "test_set_csqa.json",
    }
    save_json(paths["extraction_set_mmlu"], extract)
    save_json(paths["test_set_mmlu"], test_mmlu)
    save_json(paths["test_set_csqa"], test_csqa)

    leaks = {
        "extract∩test_mmlu": len(
            {qhash(x) for x in extract} & {qhash(x) for x in test_mmlu}
        ),
        "extract∩test_csqa": len(
            {qhash(x) for x in extract} & {qhash(x) for x in test_csqa}
        ),
        "test_mmlu∩test_csqa": len(
            {qhash(x) for x in test_mmlu} & {qhash(x) for x in test_csqa}
        ),
    }
    if any(leaks.values()):
        raise RuntimeError(f"集合泄漏: {leaks}")

    manifest = {
        "protocol": "gate_before_split",
        "repeng_root": str(root),
        "seed": args.seed,
        "fill_available": bool(args.fill_available),
        "quotas_requested": {
            "extraction_mmlu": args.n_extract_mmlu,
            "test_mmlu": args.n_test_mmlu,
            "test_csqa": args.n_test_csqa,
        },
        "quotas_actual": {
            "extraction_mmlu": len(extract),
            "test_mmlu": len(test_mmlu),
            "test_csqa": len(test_csqa),
            "total": len(extract) + len(test_mmlu) + len(test_csqa),
        },
        "pool_sizes": {
            "mmlu_gate_valid": len(mmlu_pool),
            "csqa_gate_valid": len(csqa_pool),
            "mmlu_unused_valid": len(mmlu_unused),
            "csqa_unused_valid": len(csqa_unused),
        },
        "sources": {
            "mmlu_valid": str(mmlu_path),
            "csqa_valid": str(csqa_path),
        },
        "leak_check": {**leaks, "passed": True},
        "paths": {k: str(v) for k, v in paths.items()},
        "policy": {
            "extraction_domains": ["mmlu"],
            "ood_test_domain": "csqa",
            "splits_are_gate_validated": True,
        },
        "next_step": (
            f"python phase2_layerwise_profiling.py "
            f"--extraction-set {paths['extraction_set_mmlu']} "
            f"--output-dir {root / '04_vectors'} "
            f"--no-enable-thinking"
        ),
    }
    save_json(out / "split_manifest.json", manifest)

    print("=" * 60)
    print("Split 完成（Gate→Split）：诊断子集已从 D_valid 隔离切分")
    print(f"  extraction_mmlu : {len(extract)}  (请求 {args.n_extract_mmlu})")
    print(f"  test_mmlu       : {len(test_mmlu)}  (请求 {args.n_test_mmlu})")
    print(f"  test_csqa       : {len(test_csqa)}  (请求 {args.n_test_csqa})")
    print(f"  total           : {manifest['quotas_actual']['total']}")
    print(f"  泄漏自检        : PASSED")
    print(f"  输出            : {out}")
    print("=" * 60)
    print("下一步:", manifest["next_step"])


if __name__ == "__main__":
    main()
