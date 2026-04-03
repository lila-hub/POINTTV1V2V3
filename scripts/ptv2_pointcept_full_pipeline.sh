#!/usr/bin/env bash
# PTv2：优先 Pointcept + pointcept；冲突则 legacy + pcr_ptv2
set -euo pipefail

BASE="${HOME}/ptv3_easy"
POINTCEPT="${BASE}/Pointcept"
LOG="${BASE}/ptv2_pipeline.log"
mkdir -p "$BASE"
exec > >(tee -a "$LOG") 2>&1

echo "==== 日志: $LOG ===="

fail() { echo "FAILED: $*"; }
ok() { echo "SUCCESS: $*"; }

# conda activate 的 hook 会引用未定义变量，与 set -u 冲突（如 ADDR2LINE）
conda_activate_safe() {
  set +u
  conda activate "$1"
  local _e=$?
  set -u
  return "$_e"
}

# ==================== 阶段0 ====================
echo "==== 阶段0：基础检查 ===="
command -v conda >/dev/null || { echo "FAILED: conda 不存在"; exit 1; }
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "FAILED: nvidia-smi 不可用"
  exit 1
fi
echo "---- GPU ----"
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "---- conda env list ----"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list

mkdir -p "$BASE"
if [ ! -e "$POINTCEPT" ] && [ -d "${HOME}/pointv3/Pointcept" ]; then
  echo "==== 链接 Pointcept -> ${HOME}/pointv3/Pointcept ===="
  ln -sfn "${HOME}/pointv3/Pointcept" "$POINTCEPT"
fi
if [ ! -d "$POINTCEPT" ]; then
  echo "FAILED: 无 Pointcept：$POINTCEPT（且未找到 ${HOME}/pointv3/Pointcept）"
  exit 1
fi

DATA_ROOT=""
for d in "${HOME}/datasets/scannet_processed" "${HOME}/pointv3/data/scannet_processed"; do
  [ -d "$d" ] && DATA_ROOT="$d" && break
done
if [ -z "$DATA_ROOT" ]; then
  echo "FAILED: 无 ScanNet 数据目录"
  exit 1
fi
echo "数据目录: $DATA_ROOT"
ok "阶段0"
POINTCEPT_FAIL=0
TEST_RAN=0
TEST_OK=0
CKPT=""
ROUTE="Pointcept"

# ==================== 阶段1 ====================
echo "==== 阶段1：Pointcept + conda pointcept ===="
conda_activate_safe pointcept
echo "python: $(python --version 2>&1)"
echo "which: $(which python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd "$POINTCEPT"
export PYTHONPATH=./
CFG_REL="configs/scannet/semseg-pt-v2m2-0-base.py"
if [ ! -f "$CFG_REL" ] || [ ! -f tools/test.py ] || [ ! -f scripts/train.sh ]; then
  echo "FAILED: PTv2 配置或脚本缺失"
  POINTCEPT_FAIL=1
else
  mkdir -p data
  if [ ! -e data/scannet ]; then
    ln -sfn "$DATA_ROOT" data/scannet
  fi
  ls -l data/scannet
  python - <<'PY'
from pathlib import Path
need = [
    Path("configs/scannet/semseg-pt-v2m2-0-base.py"),
    Path("tools/test.py"),
    Path("scripts/train.sh"),
    Path("data/scannet"),
]
for p in need:
    print(p, "exists =", p.exists())
PY
  if [ -d libs/pointops ]; then
    echo "==== 编译 libs/pointops ===="
    (cd libs/pointops && pip install -e . -q) || (cd libs/pointops && python setup.py install)
  fi
fi

if [ "$POINTCEPT_FAIL" -eq 1 ]; then
  echo "FAILED 阶段1（配置缺失）"
else
  ok "阶段1"
fi

# ==================== 阶段2：找权重 ====================
echo "==== 阶段2：查找 PTv2 checkpoint ===="
CKPT=""
BEST_SCORE=-1
while IFS= read -r f; do
  [ -f "$f" ] || continue
  lc=$(echo "$f" | tr '[:upper:]' '[:lower:]')
  sc=0
  [[ "$lc" == *"v2m2"* ]] && sc=$((sc + 20))
  [[ "$lc" == *"pt-v2"* || "$lc" == *"ptv2"* ]] && sc=$((sc + 15))
  [[ "$lc" == *"semseg-pt-v2"* ]] && sc=$((sc + 10))
  [[ "$lc" == *"scannet"* ]] && sc=$((sc + 5))
  if [ "$sc" -gt "$BEST_SCORE" ]; then BEST_SCORE=$sc; CKPT="$f"; fi
done < <(find "$HOME/models" "$HOME/runs" "$POINTCEPT/exp" "$BASE" -name 'model_best.pth' 2>/dev/null)

for try in \
  "$POINTCEPT/exp/scannet/semseg-pt-v2m2-0-base/model/model_best.pth" \
  "$HOME/models/ptv2/scannet-semseg-pt-v2m2-0-base/model/model_best.pth"; do
  if [ -z "$CKPT" ] && [ -f "$try" ]; then CKPT="$try"; BEST_SCORE=100; break; fi
done

if [ -n "$CKPT" ]; then
  echo "选用 checkpoint (score=$BEST_SCORE): $CKPT"
else
  echo "未在本地找到明显 PTv2 的 model_best.pth"
fi

# ==================== 阶段3 / 4 ====================
mkdir -p "${HOME}/runs/ptv2_scannet_val"
RUN_TEST_CMD() {
  cd "$POINTCEPT"
  export PYTHONPATH=./
  conda_activate_safe pointcept
  python tools/test.py \
    --config-file "$CFG_REL" \
    --num-gpus 1 \
    --options \
      save_path="${HOME}/runs/ptv2_scannet_val" \
      weight="$CKPT" \
      num_worker=4
}

if [ "$POINTCEPT_FAIL" -eq 0 ] && [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
  echo "==== 阶段3：Pointcept PTv2 测试 ===="
  if RUN_TEST_CMD; then
    TEST_RAN=1
    TEST_OK=1
    ok "阶段3"
  else
    echo "阶段3 首次失败，重编译 pointops 后重试一次"
    if [ -d libs/pointops ]; then
      (cd libs/pointops && python setup.py install)
    fi
    if RUN_TEST_CMD; then
      TEST_RAN=1
      TEST_OK=1
      ok "阶段3（重试成功）"
    else
      echo "FAILED 阶段3"
      TEST_RAN=1
      TEST_OK=0
    fi
  fi
elif [ "$POINTCEPT_FAIL" -eq 0 ]; then
  echo "==== 阶段4：无 checkpoint，就绪检查 ===="
  cd "$POINTCEPT"
  export PYTHONPATH=./
  python - <<'PY'
from pathlib import Path
cfg = Path("configs/scannet/semseg-pt-v2m2-0-base.py")
print("config_exists =", cfg.exists())
print("data_scannet_exists =", Path("data/scannet").exists())
print("tools_test_exists =", Path("tools/test.py").exists())
print("scripts_train_exists =", Path("scripts/train.sh").exists())
PY
  echo "Pointcept 路线 PTv2 已就绪，但当前未找到可用 checkpoint，因此未执行测试"
  ok "阶段4"
fi

# 成功则总结退出
if [ "$POINTCEPT_FAIL" -eq 0 ] && [ "$TEST_OK" -eq 1 ]; then
  echo "==== 第7阶段：总结 ===="
  echo "1) 路线: Pointcept 复用"
  echo "2) conda: pointcept"
  echo "3) 仓库: $POINTCEPT"
  echo "4) 配置: $POINTCEPT/$CFG_REL"
  echo "5) 数据: $DATA_ROOT"
  echo "6) checkpoint: 找到"
  echo "7) 路径: $CKPT"
  echo "8) 测试: 成功"
  echo "9) 最短复测:"
  echo "   cd $POINTCEPT && conda activate pointcept && export PYTHONPATH=./ && python tools/test.py --config-file $CFG_REL --num-gpus 1 --options save_path=${HOME}/runs/ptv2_scannet_val weight=$CKPT num_worker=4"
  echo "SUCCESS 全部结束"
  exit 0
fi

if [ "$POINTCEPT_FAIL" -eq 0 ] && [ -z "$CKPT" ]; then
  echo "==== 第7阶段：总结 ===="
  echo "1) 路线: Pointcept 复用"
  echo "2) conda: pointcept"
  echo "3) 仓库: $POINTCEPT"
  echo "4) 配置: $POINTCEPT/$CFG_REL"
  echo "5) 数据: $DATA_ROOT"
  echo "6) checkpoint: 未找到"
  echo "8) 测试: 未执行（无权重）"
  echo "训练命令:"
  echo "  cd $POINTCEPT && export PYTHONPATH=./ && sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base"
  echo "测试模板:"
  echo "  cd $POINTCEPT && export PYTHONPATH=./ && python tools/test.py --config-file $CFG_REL --num-gpus 1 --options save_path=${HOME}/runs/ptv2_scannet_val weight=你的checkpoint路径 num_worker=4"
  echo "SUCCESS 结束"
  exit 0
fi

# ==================== 阶段5 fallback ====================
echo "==== 阶段5：legacy PointTransformerV2 + pcr_ptv2 ===="
LEGACY="${BASE}/PointTransformerV2"
if [ ! -d "$LEGACY/.git" ]; then
  git clone --depth 1 https://github.com/Pointcept/PointTransformerV2.git "$LEGACY" || { echo "FAILED git clone"; exit 1; }
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -qE '^pcr_ptv2[[:space:]]'; then
  echo "==== 创建 conda 环境 pcr_ptv2（耗时）===="
  conda create -n pcr_ptv2 python=3.8 -y
fi
set +e
conda_activate_safe pcr_ptv2
conda install ninja -y
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
conda install -c anaconda h5py pyyaml -y
conda install -c conda-forge sharedarray tensorboardx yapf addict einops scipy plyfile termcolor timm -y
conda install -c pyg pytorch-cluster pytorch-scatter pytorch-sparse -y
pip install -q torch-geometric
set -e

cd "$LEGACY/libs/pointops" && python setup.py install
cd "$LEGACY"
mkdir -p data
ln -sfn "$DATA_ROOT" data/scannet
ls -l data/scannet
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ok "阶段5"

# ==================== 阶段6 legacy 测试 ====================
echo "==== 阶段6：legacy 找权重并测试 ===="
CKPT_L=""
BEST=-1
while IFS= read -r f; do
  [ -f "$f" ] || continue
  lc=$(echo "$f" | tr '[:upper:]' '[:lower:]')
  sc=0
  [[ "$lc" == *"v2m2"* ]] && sc=$((sc + 20))
  [[ "$lc" == *"ptv2"* || "$lc" == *"pt-v2"* ]] && sc=$((sc + 15))
  [ "$sc" -gt "$BEST" ] && BEST=$sc && CKPT_L="$f"
done < <(find "$HOME/models" "$HOME/runs" "$LEGACY" -name 'model_best.pth' 2>/dev/null)

LCFG="configs/scannet/semseg-pt-v2m2-0-base.py"
if [ ! -f "$LEGACY/$LCFG" ]; then
  LCFG="configs/scannet/semseg-ptv2m2-0-base.py"
fi
if [ ! -f "$LEGACY/$LCFG" ]; then
  echo "列出 configs/scannet:"
  ls -la "$LEGACY/configs/scannet/" 2>/dev/null | head -20
fi

if [ -n "$CKPT_L" ] && [ -f "$CKPT_L" ]; then
  echo "legacy 使用权重: $CKPT_L"
  export PYTHONPATH=./
  mkdir -p "$LEGACY/exp/scannet/semseg-ptv2m2-0-base"
  if python tools/test.py \
    --config-file "$LCFG" \
    --num-gpus 1 \
    --options \
      save_path="$LEGACY/exp/scannet/semseg-ptv2m2-0-base" \
      weight="$CKPT_L" \
      num_worker=4; then
    ROUTE="legacy"
    echo "==== 第7阶段：总结 ===="
    echo "1) 路线: legacy PointTransformerV2"
    echo "2) conda: pcr_ptv2"
    echo "3) 仓库: $LEGACY"
    echo "4) 配置: $LEGACY/$LCFG"
    echo "5) 数据: $DATA_ROOT"
    echo "7) checkpoint: $CKPT_L"
    echo "8) 测试: 成功"
    echo "9) 复测: cd $LEGACY && conda activate pcr_ptv2 && export PYTHONPATH=./ && python tools/test.py --config-file $LCFG --num-gpus 1 --options save_path=$LEGACY/exp/scannet/semseg-ptv2m2-0-base weight=$CKPT_L num_worker=4"
    echo "SUCCESS"
    exit 0
  fi
fi

echo "==== 第7阶段：总结（legacy 就绪 / 测试未跑或无权重）===="
echo "1) 路线: legacy fallback"
echo "2) conda: pcr_ptv2"
echo "3) 仓库: $LEGACY"
echo "4) 配置: $LEGACY/$LCFG"
for p in "$LCFG" tools/test.py data/scannet; do
  [ -e "$LEGACY/$p" ] && echo "  OK $p" || echo "  MISSING $p"
done
echo "训练: cd $LEGACY && export PYTHONPATH=./ && sh scripts/train.sh -g 4 -d scannet -c semseg-ptv2m2-0-base -n semseg-ptv2m2-0-base"
echo "测试模板: python tools/test.py --config-file $LCFG --num-gpus 1 --options save_path=$LEGACY/exp/scannet/semseg-ptv2m2-0-base weight=你的路径"
echo "SUCCESS 结束"
