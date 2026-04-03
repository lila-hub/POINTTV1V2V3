#!/usr/bin/env bash
# Pointcept PTv3 ScanNet val 测试 — 从 pointops 编译到 test（在已装好 conda 环境 pointcept 后执行）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${ROOT}/logs"
LOG="${ROOT}/logs/complete_ptv3.log"
exec > >(tee -a "$LOG") 2>&1

echo "==== 日志: $LOG ===="

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pointcept

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

echo "==== 第6步：pointops ===="
cd "${ROOT}/Pointcept/libs/pointops"
python setup.py install || { echo "FAILED setup.py"; pip install -e .; }
cd "${ROOT}/Pointcept"
python -c "import pointops; print('pointops import OK')"

echo "==== 第7步：Hugging Face ===="
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "FAILED: 缺少 HUGGINGFACE_HUB_TOKEN。请先执行："
  echo "  export HUGGINGFACE_HUB_TOKEN=你的token"
  exit 2
fi
hf auth login --token "${HUGGINGFACE_HUB_TOKEN}"

echo "==== 第8步：下载 ScanNet 预处理数据 ===="
mkdir -p "${ROOT}/data/scannet_processed"
hf download Pointcept/scannet-compressed --repo-type dataset --local-dir "${ROOT}/data/scannet_processed"

echo "==== 第9步：软链接 ===="
mkdir -p "${ROOT}/Pointcept/data"
ln -sfn "${ROOT}/data/scannet_processed" "${ROOT}/Pointcept/data/scannet"
ls -l "${ROOT}/Pointcept/data/scannet"

echo "==== 第10步：下载权重 ===="
mkdir -p "${ROOT}/models/ptv3"
hf download Pointcept/PointTransformerV3 \
  --repo-type model \
  --local-dir "${ROOT}/models/ptv3" \
  --include "scannet-semseg-pt-v3m1-0-base/**"
CFG="${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/config.py"
WGT="${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth"
test -f "$CFG" && test -f "$WGT"

echo "==== 第11步：修改 config ===="
python3 << PY
from pathlib import Path
p = Path("${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/config.py")
t = p.read_text(encoding="utf-8")
t = t.replace("enable_flash=True", "enable_flash=False")
t = t.replace("enable_flash = True", "enable_flash = False")
t = t.replace(
    "enc_patch_size=(1024, 1024, 1024, 1024, 1024)",
    "enc_patch_size=(128, 128, 128, 128, 128)",
)
t = t.replace(
    "dec_patch_size=(1024, 1024, 1024, 1024)",
    "dec_patch_size=(128, 128, 128, 128)",
)
p.write_text(t, encoding="utf-8")
for line in p.read_text(encoding="utf-8").splitlines():
    if "enable_flash" in line or "enc_patch_size" in line or "dec_patch_size" in line:
        print(line)
PY

echo "==== 第12步：测试 ScanNet val ===="
mkdir -p "${ROOT}/runs/ptv3_scannet_val"
cd "${ROOT}/Pointcept"
export PYTHONPATH=./
python tools/test.py \
  --config-file "${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/config.py" \
  --num-gpus 1 \
  --options \
    save_path="${ROOT}/runs/ptv3_scannet_val" \
    weight="${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth"

echo "==== 完成 ===="
