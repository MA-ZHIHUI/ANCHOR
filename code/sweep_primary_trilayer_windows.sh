#!/usr/bin/env bash
# =============================================================================
# 主实验：primary 向量滑动三层窗（默认从 10,11,12 到 33,34,35，α=50）
#
# 用法:
#   cd ~/ANCHOR/code && conda activate ANCHOR
#   CUDA_VISIBLE_DEVICES=0 bash sweep_primary_trilayer_windows.sh
#   SHALLOW=1 CUDA_VISIBLE_DEVICES=0 bash sweep_primary_trilayer_windows.sh
#   LIMIT=2 CUDA_VISIBLE_DEVICES=0 bash sweep_primary_trilayer_windows.sh
#   bash sweep_primary_trilayer_windows.sh --summary-only
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE="${ROOT}/code"
PY="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"
SHALLOW="${SHALLOW:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"

cd "${CODE}"

cmd=(
  "${PY}" sweep_primary_trilayer_windows.py
  --device "${DEVICE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi
if [[ "${SHALLOW}" == "1" ]]; then
  cmd+=(--shallow-spotcheck)
fi
if [[ "${1:-}" == "--summary-only" ]]; then
  cmd+=(--summary-only)
fi

echo "[sweep_primary_trilayer] ${cmd[*]}"
exec "${cmd[@]}"
