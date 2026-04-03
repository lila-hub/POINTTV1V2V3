#!/usr/bin/env bash
# 将 ~/datasets/scannet_processed、~/models/ptv3、整个 ~/runs 迁入本仓库 runs/
# 用法: bash ~/pointv3/relocate_assets.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/data" "$ROOT/runs" "$ROOT/models" "$ROOT/logs"

log() { echo "[relocate] $*"; }

# 若目标已存在且非空：合并后删除源；若目标不存在：直接 mv
merge_or_move() {
  local src="$1"
  local dst="$2"
  if [ ! -e "$src" ]; then
    log "跳过（源不存在）: $src"
    return 0
  fi
  if [ -L "$src" ]; then
    log "跳过（源为符号链接，请手动处理）: $src"
    return 0
  fi
  if [ ! -e "$dst" ]; then
    log "mv -> $dst"
    mv "$src" "$dst"
    return 0
  fi
  # 目标已存在：合并目录内容
  if [ -d "$src" ] && [ -d "$dst" ]; then
    log "合并目录: $src -> $dst"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "$src/" "$dst/"
    else
      cp -a "$src/." "$dst/"
    fi
    rm -rf "$src"
    log "已删除源目录: $src"
    return 0
  fi
  log "跳过（目标已存在且类型冲突）: $dst"
}

merge_or_move "$HOME/datasets/scannet_processed" "$ROOT/data/scannet_processed"
merge_or_move "$HOME/models/ptv3" "$ROOT/models/ptv3"
# 整个 ~/runs（含 ptv3_* 及任意其它实验目录）-> 本仓库 runs/
merge_or_move "$HOME/runs" "$ROOT/runs"

# 若 ~/runs 已迁走但仍有残留子路径，再尝试同步 sub10 的 vis/ 与报告
if [ -f "$ROOT/scripts/sync_sub10_vis_from_home.sh" ]; then
  bash "$ROOT/scripts/sync_sub10_vis_from_home.sh"
fi

# Pointcept 数据目录 -> 本仓库 data
mkdir -p "$ROOT/Pointcept/data"
if [ -d "$ROOT/data/scannet_processed" ]; then
  ln -sfn "$ROOT/data/scannet_processed" "$ROOT/Pointcept/data/scannet"
  log "软链: Pointcept/data/scannet -> $ROOT/data/scannet_processed"
else
  log "警告: 未找到 $ROOT/data/scannet_processed，未创建 scannet 软链"
fi

log "完成。布局:"
ls -la "$ROOT/data" 2>/dev/null || true
ls -la "$ROOT/models" 2>/dev/null || true
ls -la "$ROOT/runs" 2>/dev/null || true
