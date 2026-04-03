#!/usr/bin/env bash
# 把 ~/runs/ptv3_test_sub10/vis 与 SUB10_REPORT.md 同步进本仓库（不删源）
# 用法: bash ~/pointv3/scripts/sync_sub10_vis_from_home.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_BASE="${HOME}/runs/ptv3_test_sub10"
DST_BASE="${ROOT}/runs/ptv3_test_sub10"
SRC_VIS="${SRC_BASE}/vis"
DST_VIS="${DST_BASE}/vis"
SRC_MD="${SRC_BASE}/SUB10_REPORT.md"
DST_MD="${DST_BASE}/SUB10_REPORT.md"

log() { echo "[sync_sub10_vis] $*"; }

if [ ! -d "$SRC_VIS" ]; then
  log "跳过：不存在 $SRC_VIS"
else
  mkdir -p "$DST_VIS"
  log "同步 $SRC_VIS -> $DST_VIS"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$SRC_VIS/" "$DST_VIS/"
  else
    cp -a "$SRC_VIS/." "$DST_VIS/"
  fi
fi

if [ -f "$SRC_MD" ]; then
  mkdir -p "$(dirname "$DST_MD")"
  log "复制 $SRC_MD -> $DST_MD"
  cp -a "$SRC_MD" "$DST_MD"
else
  log "跳过：不存在 $SRC_MD"
fi

log "完成。仓库内路径: $DST_BASE"
