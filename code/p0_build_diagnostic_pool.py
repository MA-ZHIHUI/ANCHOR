# p0_build_diagnostic_pool.py
"""
ANCHOR 第1步：从宏观矩阵捞「深度翻转」题，回灌 correct.json，建成白盒候选池。

推荐逻辑（不再解析 messages）:
  1) 读 dataset/.../item_summary.csv，筛 is_valid ∧ Round0对 ∧ final_flipped
  2) 用 item_index 从 filtered_data/.../correct.json 取原题
  3) 写入 repeng_data/<model>_<protocol>[_think]/01_candidates/

切换实验只需改下方「可配置区」或命令行参数，例如:
  # 多轮 + 不思考（默认）
  python p0_build_diagnostic_pool.py

  # 单轮 + 不思考
  python p0_build_diagnostic_pool.py --protocol single

  # 多轮 + 思考
  python p0_build_diagnostic_pool.py --enable-thinking

  # Gemma
  python p0_build_diagnostic_pool.py --model gemma-4-12B-it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config

# =====================================================================
# 可配置区（改这里即可切换模型 / 协议 / 思考模式 / 路径；命令行可覆盖）
# =====================================================================
MODEL_NAME = "Qwen3-8B"  # 例: "Qwen3-8B" / "gemma-4-12B-it"
PROTOCOL = "multi"  # "multi" | "single"
ENABLE_THINKING = False  # False=no-think, True=think

# 要捞的数据域：逻辑名 → dataset / filtered_data 子目录名
DOMAINS: dict[str, str] = {
    "mmlu": "mmlu_dev",
    "csqa": "commonsenseqa_dev",
}

DATASET_ROOT = config.PROJECT_ROOT / "dataset"
FILTERED_ROOT = config.PROJECT_ROOT / "filtered_data"
OUTPUT_ROOT = config.PROJECT_ROOT / "repeng_data"

# 翻转口径
REQUIRE_VALID = True
REQUIRE_ROUND0_CORRECT = True
FLIP_COLUMN = "final_flipped"  # 或 "ever_flipped"
# =====================================================================


def run_dirname(model: str, protocol: str, think: bool) -> str:
    """
    与 dataset/ 目录命名对齐:
      multi  + nothink → Qwen3-8B
      single + nothink → Qwen3-8B_single
      multi  + think   → Qwen3-8B_think
      single + think   → Qwen3-8B_single_think
    """
    name = model
    if protocol == "single":
        name += "_single"
    elif protocol != "multi":
        raise ValueError(f"protocol 仅支持 multi|single，收到: {protocol!r}")
    if think:
        name += "_think"
    return name


def filtered_dirname(model: str, think: bool) -> str:
    """
    filtered_data 按「筛选时是否 think」分目录，与 single/multi 无关:
      nothink → Qwen3-8B
      think   → Qwen3-8B_think
    """
    return f"{model}_think" if think else model


def repeng_run_tag(protocol: str, think: bool) -> str:
    """输出根目录后缀，如 multi_nothink / single_think。"""
    return f"{protocol}_{'think' if think else 'nothink'}"


def load_json_list(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"期望 JSON 数组: {path}")
    return data


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def select_deep_flip_indices(
    item_summary: Path,
    *,
    require_valid: bool,
    require_round0_correct: bool,
    flip_column: str,
) -> tuple[list[int], dict]:
    """从 item_summary 选出深度翻转题的 item_index 列表。"""
    df = pd.read_csv(item_summary)
    if flip_column not in df.columns:
        raise ValueError(f"{item_summary} 缺少列 {flip_column}")

    mask = df[flip_column] == 1
    if require_valid and "is_valid" in df.columns:
        mask &= df["is_valid"] == 1

    if require_round0_correct:
        if "round0_pred" in df.columns and "gt" in df.columns:
            # 注意：勿用 df.gt（与 DataFrame.gt 方法冲突）
            mask &= df["round0_pred"].astype(str).str.upper() == df["gt"].astype(str).str.upper()
        else:
            raise ValueError(
                f"{item_summary} 缺少 round0_pred/gt，无法检查 Round0 正确性"
            )

    sub = df.loc[mask]
    indices = sub["item_index"].astype(int).tolist()
    stats = {
        "n_total": int(len(df)),
        "n_selected": len(indices),
        "flip_column": flip_column,
        "require_valid": require_valid,
        "require_round0_correct": require_round0_correct,
    }
    return indices, stats


def build_domain_pool(
    *,
    domain_key: str,
    dataset_subdir: str,
    model: str,
    protocol: str,
    think: bool,
    dataset_root: Path,
    filtered_root: Path,
    require_valid: bool,
    require_round0_correct: bool,
    flip_column: str,
) -> tuple[list[dict], dict]:
    """单域：item_summary 定序号 → correct.json 回灌。"""
    run = run_dirname(model, protocol, think)
    filt = filtered_dirname(model, think)

    summary_path = dataset_root / dataset_subdir / run / "item_summary.csv"
    correct_path = filtered_root / dataset_subdir / filt / "correct.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"[{domain_key}] 未找到 item_summary: {summary_path}")
    if not correct_path.exists():
        raise FileNotFoundError(
            f"[{domain_key}] 未找到 correct.json: {correct_path}\n"
            f"请确认筛选目录名是否为 {filt!r}（think 时带 _think）。"
        )

    indices, stats = select_deep_flip_indices(
        summary_path,
        require_valid=require_valid,
        require_round0_correct=require_round0_correct,
        flip_column=flip_column,
    )
    correct_items = load_json_list(correct_path)
    n_correct = len(correct_items)

    selected: list[dict] = []
    missing: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= n_correct:
            missing.append(idx)
            continue
        item = dict(correct_items[idx])  # 浅拷贝，避免改到原列表
        # 补齐溯源字段（不覆盖 correct.json 已有核心字段）
        item.setdefault("answer", item.get("answer"))
        item["source"] = domain_key
        item["model"] = model
        item["protocol"] = protocol
        item["enable_thinking"] = bool(think)
        item["vllm_run"] = run
        item["vllm_item_index"] = idx
        item["filtered_correct"] = str(correct_path)
        selected.append(item)

    meta = {
        **stats,
        "domain": domain_key,
        "dataset_run_dir": str(summary_path.parent),
        "item_summary": str(summary_path),
        "correct_json": str(correct_path),
        "n_correct_json": n_correct,
        "n_missing_index": len(missing),
        "missing_indices_head": missing[:10],
    }
    if missing:
        print(f"  ⚠️ [{domain_key}] {len(missing)} 个 item_index 越界，已跳过")
    return selected, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="第1步：item_summary 定序 + correct.json 回灌 → 诊断候选池"
    )
    p.add_argument("--model", default=MODEL_NAME, help="模型短名，如 Qwen3-8B")
    p.add_argument(
        "--protocol",
        choices=["multi", "single"],
        default=PROTOCOL,
        help="multi=多轮目录无_single；single=带_single",
    )
    p.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_THINKING,
        help="是否对应 dataset/filtered 的 _think 目录",
    )
    p.add_argument(
        "--domains",
        nargs="+",
        default=list(DOMAINS.keys()),
        help=f"域逻辑名，默认 {' '.join(DOMAINS)}；映射见脚本内 DOMAINS",
    )
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--filtered-root", type=Path, default=FILTERED_ROOT)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument(
        "--flip-column",
        choices=["final_flipped", "ever_flipped"],
        default=FLIP_COLUMN,
    )
    p.add_argument(
        "--require-valid",
        action=argparse.BooleanOptionalAction,
        default=REQUIRE_VALID,
    )
    p.add_argument(
        "--require-round0-correct",
        action=argparse.BooleanOptionalAction,
        default=REQUIRE_ROUND0_CORRECT,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model
    protocol = args.protocol
    think = bool(args.enable_thinking)
    run_tag = repeng_run_tag(protocol, think)
    out_root = args.output_root.resolve() / f"{model}_{run_tag}"
    cand_dir = out_root / "01_candidates"
    meta_dir = out_root / "00_meta"

    print(
        f"配置: model={model} | protocol={protocol} | think={think}\n"
        f"dataset run 名: {run_dirname(model, protocol, think)}\n"
        f"filtered 名  : {filtered_dirname(model, think)}\n"
        f"输出根目录   : {out_root}"
    )

    all_meta: dict[str, dict] = {}
    outputs: dict[str, str] = {}

    for domain_key in args.domains:
        if domain_key not in DOMAINS:
            raise KeyError(
                f"未知域 {domain_key!r}；请在脚本顶部 DOMAINS 中注册，"
                f"现有: {list(DOMAINS)}"
            )
        dataset_subdir = DOMAINS[domain_key]
        items, meta = build_domain_pool(
            domain_key=domain_key,
            dataset_subdir=dataset_subdir,
            model=model,
            protocol=protocol,
            think=think,
            dataset_root=args.dataset_root.resolve(),
            filtered_root=args.filtered_root.resolve(),
            require_valid=bool(args.require_valid),
            require_round0_correct=bool(args.require_round0_correct),
            flip_column=args.flip_column,
        )
        out_path = cand_dir / f"{domain_key}_deep_flip.json"
        save_json(out_path, items)
        outputs[domain_key] = str(out_path)
        all_meta[domain_key] = meta
        print(f"  [{domain_key}] 选出 {len(items)} 题 → {out_path}")

    card = {
        "model": model,
        "protocol": protocol,
        "enable_thinking": think,
        "dataset_run_dirname": run_dirname(model, protocol, think),
        "filtered_dirname": filtered_dirname(model, think),
        "flip_column": args.flip_column,
        "require_valid": bool(args.require_valid),
        "require_round0_correct": bool(args.require_round0_correct),
        "domains": args.domains,
        "domain_map": {k: DOMAINS[k] for k in args.domains},
        "method": "item_summary_indices -> filtered_data correct.json",
        "outputs": outputs,
        "per_domain": all_meta,
        "notes": [
            "宏观矩阵只读自 dataset/；题面回灌自 filtered_data/。",
            "切换 single/multi/think/model 只需改可配置区或 CLI。",
            "论文协议顺序：p0 → p_gate_splits（02_gate）→ p1_split（03_splits）→ phase2 → step3。",
        ],
    }
    save_json(meta_dir / "DATASET_CARD.json", card)
    save_json(cand_dir / "build_manifest.json", card)

    for d in ("02_gate", "03_splits", "04_vectors"):
        (out_root / d).mkdir(parents=True, exist_ok=True)
        keep = out_root / d / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")

    print("=" * 60)
    print("第1步完成：候选池已重建（correct.json 回灌）")
    print(f"  DATASET_CARD: {meta_dir / 'DATASET_CARD.json'}")
    print(f"下一步: python p_gate_splits.py --repeng-root {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
