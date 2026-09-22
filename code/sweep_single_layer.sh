#!/usr/bin/env bash
# =============================================================================
# 单层谄媚向量扫描入口（推荐走 Python：模型只加载一次，baseline 每数据集缓存一次）
#
# 网格默认:
#   layers = 0 5 10 15 18 20 22 25 28 30 33 35
#   alphas = 10 20 30 40
#   datasets = mmlu_dev + commonsenseqa_dev
#   → 约 96 个配置；协议: Think OFF / greedy / per-layer / fromR0
#
# 用法:
#   cd ~/ANCHOR/code
#   conda activate ANCHOR
#   bash sweep_single_layer.sh                       # 全量
#   DEVICE=1 bash sweep_single_layer.sh
#   LIMIT=3 bash sweep_single_layer.sh               # 调试
#   DRY_RUN=1 bash sweep_single_layer.sh
#   bash sweep_single_layer.sh --summary-only
#
# 预估（Qwen3-8B，单卡）:
#   MMLU n=52 baseline ~1–2min；CSQA n=275 baseline ~5–8min；
#   其后每个 (layer,alpha) 约等于只跑 intervention 臂。
#   全量大约数小时量级；可中断，默认 --skip-done 续跑。
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE="${ROOT}/code"
PY="${PYTHON:-python}"
DEVICE="${DEVICE:-3}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"

cd "${CODE}"

cmd=(
  "${PY}" sweep_single_layer.py
  --device "${DEVICE}"
)

if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi
if [[ "${1:-}" == "--summary-only" ]]; then
  cmd+=(--summary-only)
fi

echo "[sweep] ${cmd[*]}"
exec "${cmd[@]}"
