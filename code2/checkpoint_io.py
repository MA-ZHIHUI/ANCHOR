# checkpoint_io.py
"""detail.csv 断点读写 / 合并（单轮与多轮共用）。"""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    csv.field_size_limit(10**7)
except Exception:
    pass


DETAIL_KEY_COLS = ("model", "experiment_mode", "item_index", "round_name")


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, bak)
    return bak


def load_detail_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "model" not in out.columns:
        out["model"] = ""
    if "experiment_mode" not in out.columns:
        out["experiment_mode"] = ""
    out["item_index"] = out["item_index"].astype(int)
    out["round_name"] = out["round_name"].astype(str)
    out["model"] = out["model"].astype(str)
    out["experiment_mode"] = out["experiment_mode"].astype(str)
    return out


def existing_keys(df: pd.DataFrame) -> set[tuple]:
    if df.empty:
        return set()
    df = normalize_key_frame(df)
    keys: set[tuple] = set()
    for model, mode, item_index, round_name in zip(
        df["model"], df["experiment_mode"], df["item_index"], df["round_name"]
    ):
        keys.add((str(model), str(mode), int(item_index), str(round_name)))
    return keys


def completed_item_indices(
    df: pd.DataFrame,
    *,
    model: str,
    experiment_mode: str,
    required_round_names: list[str],
) -> set[int]:
    """返回已具备全部 required_round_names 的 item_index。"""
    if df.empty:
        return set()
    df = normalize_key_frame(df)
    sub = df[
        (df["model"] == str(model)) & (df["experiment_mode"] == str(experiment_mode))
    ]
    if sub.empty:
        return set()
    need = set(required_round_names)
    done: set[int] = set()
    for item_index, g in sub.groupby("item_index"):
        names = set(g["round_name"].astype(str))
        if need.issubset(names):
            done.add(int(item_index))
    return done


def drop_partial_items(
    df: pd.DataFrame,
    *,
    model: str,
    experiment_mode: str,
    required_round_names: list[str],
) -> pd.DataFrame:
    """删掉同 model/mode 下未凑齐 required 轮次的题（便于多轮整题重跑）。"""
    if df.empty:
        return df
    df = normalize_key_frame(df)
    mask_scope = (df["model"] == str(model)) & (
        df["experiment_mode"] == str(experiment_mode)
    )
    other = df[~mask_scope]
    scoped = df[mask_scope]
    if scoped.empty:
        return df

    need = set(required_round_names)
    keep_idx: list[int] = []
    for item_index, g in scoped.groupby("item_index"):
        names = set(g["round_name"].astype(str))
        if need.issubset(names):
            keep_idx.extend(g.index.tolist())
        # else: drop partial — will be re-run
    kept = scoped.loc[keep_idx] if keep_idx else scoped.iloc[0:0]
    return pd.concat([other, kept], ignore_index=True)


def merge_detail_rows(
    existing: pd.DataFrame, new_rows: list[dict] | pd.DataFrame
) -> pd.DataFrame:
    """旧行优先；按主键去重。"""
    new_df = pd.DataFrame(new_rows) if not isinstance(new_rows, pd.DataFrame) else new_rows
    if existing is None or existing.empty:
        return new_df.reset_index(drop=True)
    if new_df is None or new_df.empty:
        return existing.reset_index(drop=True)

    existing = normalize_key_frame(existing)
    new_df = normalize_key_frame(new_df)
    merged = pd.concat([existing, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=list(DETAIL_KEY_COLS), keep="first")
    return merged.reset_index(drop=True)


def save_detail_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def append_detail_rows(path: Path, rows: list[dict]) -> None:
    """逐条/逐批追加到 detail.csv（断点落盘）。若文件不存在则写表头。"""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )
