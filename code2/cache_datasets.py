# cache_datasets.py
"""
将 CommonsenseQA、SciQ 下载并缓存到 data/ 目录，尽量保持各数据集原始格式。

MMLU 请使用独立脚本: python cache_mmlu.py

输出结构:
  data/commonsenseqa/
    train_rand_split.jsonl
    dev_rand_split.jsonl
    test_rand_split.jsonl

  data/sciq/
    train.json
    valid.json
    test.json

依赖: huggingface_hub, pyarrow (仅 CommonsenseQA 转换需要)
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


SCIQ_URL = "https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip"

CSQA_REPO = "tau/commonsense_qa"
CSQA_SPLITS = {
    "train_rand_split.jsonl": "train",
    "dev_rand_split.jsonl": "validation",
    "test_rand_split.jsonl": "test",
}

SCIQ_JSON_FILES = {
    "train.json": "train.json",
    "valid.json": "valid.json",
    "test.json": "test.json",
}


def download_file(url: str, dest: Path, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"  已存在，跳过下载: {dest}")
        return

    print(f"  下载: {url}")
    print(f"  保存到: {dest}")

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = block_num * block_size
        pct = min(done * 100.0 / total_size, 100.0)
        print(f"\r  进度: {pct:5.1f}% ({done / 1024 / 1024:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_report)
    print()


def _is_cached(marker_files: list[Path]) -> bool:
    return all(p.exists() for p in marker_files)


def cache_sciq(data_dir: Path, force: bool = False) -> None:
    target = data_dir / "sciq"
    marker_files = [target / name for name in SCIQ_JSON_FILES]
    if _is_cached(marker_files) and not force:
        print(f"[SciQ] 已缓存: {target}")
        return

    print("[SciQ] 开始缓存 (原始 JSON, AllenAI SciQ.zip)...")
    if target.exists() and force:
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "sciq.zip"
        download_file(SCIQ_URL, archive, force=True)

        with zipfile.ZipFile(archive, "r") as zf:
            names = zf.namelist()
            for out_name, pattern in SCIQ_JSON_FILES.items():
                match = next((n for n in names if n.endswith(f"/{pattern}") or n.endswith(pattern)), None)
                if match is None:
                    raise FileNotFoundError(f"SciQ zip 中未找到 {pattern}")
                dest = target / out_name
                dest.write_bytes(zf.read(match))
                print(f"  写入: {dest}")

    print(f"[SciQ] 完成: {target}")


def _parquet_to_jsonl_rows(parquet_path: str) -> Iterable[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    columns = table.schema.names

    for i in range(table.num_rows):
        row = {name: table.column(name)[i].as_py() for name in columns}
        answer_key = row.get("answerKey")
        if answer_key in ("", None):
            row.pop("answerKey", None)
        yield row


def cache_commonsenseqa(data_dir: Path, force: bool = False) -> None:
    from huggingface_hub import hf_hub_download

    target = data_dir / "commonsenseqa"
    marker_files = [target / name for name in CSQA_SPLITS]
    if _is_cached(marker_files) and not force:
        print(f"[CommonsenseQA] 已缓存: {target}")
        return

    print("[CommonsenseQA] 开始缓存 (原始 JSONL, 由 tau/commonsense_qa 转换)...")
    if target.exists() and force:
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    for filename, split in CSQA_SPLITS.items():
        parquet_name = f"data/{split}-00000-of-00001.parquet"
        parquet_path = hf_hub_download(
            repo_id=CSQA_REPO,
            filename=parquet_name,
            repo_type="dataset",
        )
        out_path = target / filename
        count = 0
        with out_path.open("w", encoding="utf-8") as f:
            for row in _parquet_to_jsonl_rows(parquet_path):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        print(f"  写入: {out_path} ({count} 条)")

    print(f"[CommonsenseQA] 完成: {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="缓存 CommonsenseQA / SciQ 到 data/ 目录")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="数据根目录 (默认: data)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["all", "commonsenseqa", "sciq"],
        default=["all"],
        help="要缓存的数据集 (默认: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载并覆盖已有缓存",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.datasets)
    if "all" in selected:
        selected = {"commonsenseqa", "sciq"}

    print(f"数据目录: {data_dir}\n")

    if "sciq" in selected:
        cache_sciq(data_dir, force=args.force)
        print()

    if "commonsenseqa" in selected:
        cache_commonsenseqa(data_dir, force=args.force)
        print()

    print("全部完成。")


if __name__ == "__main__":
    main()
