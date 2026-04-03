#!/usr/bin/env bash
# 在 export HUGGINGFACE_HUB_TOKEN=你的token 后执行本脚本，完成阶段 7–14
# 项目根目录：~/pointv3
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pointcept
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${ROOT}/Pointcept/"

echo "==== 阶段7：登录 Hugging Face ===="
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  echo "FAILED: 缺少 HUGGINGFACE_HUB_TOKEN。请先执行："
  echo "  export HUGGINGFACE_HUB_TOKEN=你的token"
  echo "然后重新运行: bash $0"
  exit 1
fi
hf auth login --token "$HUGGINGFACE_HUB_TOKEN"
echo "SUCCESS: 阶段7"
echo "下一步: 下载 ScanNet 预处理数据"

echo "==== 阶段8：下载预处理 ScanNet ===="
mkdir -p "${ROOT}/data/scannet_processed"
if hf download Pointcept/scannet-compressed --repo-type dataset --local-dir "${ROOT}/data/scannet_processed"; then
  echo "SUCCESS: 阶段8"
else
  echo "FAILED: 阶段8（若为 403，请到 Hugging Face 数据集页面同意条款后重试）"
  exit 1
fi
echo "==== 阶段8b：解压 scannet.tar.gz（若尚未解压）===="
if [ ! -d "${ROOT}/data/scannet_processed/train" ] && [ -f "${ROOT}/data/scannet_processed/scannet.tar.gz" ]; then
  echo "正在解压（时间较长）..."
  (cd "${ROOT}/data/scannet_processed" && tar -xzf scannet.tar.gz)
fi
test -d "${ROOT}/data/scannet_processed/train" || { echo "FAILED: 未找到 train/，请检查 scannet.tar.gz"; exit 1; }
echo "==== 阶段8c：npy 子目录 -> Pointcept 所需 *.pth ===="
python "${ROOT}/scripts/convert_scannet_npy_to_pth.py" "${ROOT}/data/scannet_processed"
echo "SUCCESS: 阶段8c"
echo "下一步: 数据软链接"

echo "==== 阶段9：建立数据软链接 ===="
mkdir -p "${ROOT}/Pointcept/data"
ln -sfn "${ROOT}/data/scannet_processed" "${ROOT}/Pointcept/data/scannet"
ls -l "${ROOT}/Pointcept/data/scannet"
echo "SUCCESS: 阶段9"
echo "下一步: 下载权重"

echo "==== 阶段10：下载 PTv3 实验与权重 ===="
mkdir -p "${ROOT}/models/ptv3"
hf download Pointcept/PointTransformerV3 \
  --repo-type model \
  --local-dir "${ROOT}/models/ptv3" \
  --include "scannet-semseg-pt-v3m1-0-base/**"
CFG="${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/config.py"
PTH="${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth"
if [ -f "$CFG" ] && [ -f "$PTH" ]; then
  echo "SUCCESS: 阶段10（config 与 model_best.pth 已存在）"
else
  echo "FAILED: 缺少预期文件，目录列表："
  find "${ROOT}/models/ptv3" -maxdepth 4 -type f 2>/dev/null | head -50
  exit 1
fi
echo "下一步: 修改 config"

echo "==== 阶段11：自动修改 config ===="
python << PY
from pathlib import Path
p = Path("${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/config.py")
t = p.read_text()
t = t.replace("enable_flash=True", "enable_flash=False")
t = t.replace("enable_flash = True", "enable_flash = False")
t = t.replace("enc_patch_size=(1024, 1024, 1024, 1024, 1024)", "enc_patch_size=(128, 128, 128, 128, 128)")
t = t.replace("dec_patch_size=(1024, 1024, 1024, 1024)", "dec_patch_size=(128, 128, 128, 128)")
p.write_text(t)
lines = [ln for ln in p.read_text().splitlines() if any(k in ln for k in ("enable_flash", "enc_patch_size", "dec_patch_size"))]
print("--- 关键行 ---")
for ln in lines[:30]:
    print(ln)
PY
echo "SUCCESS: 阶段11"
echo "下一步: 测试"

echo "==== 阶段12：启动测试 ===="
pip install -q spconv-cu118
mkdir -p "${ROOT}/runs/ptv3_scannet_val"
cd "${ROOT}/Pointcept"
export PYTHONPATH=./
python tools/test.py \
  --config-file "${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/config.py" \
  --num-gpus 1 \
  --options \
    save_path="${ROOT}/runs/ptv3_scannet_val" \
    weight="${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth"
echo "SUCCESS: 阶段12（测试命令已执行完毕）"

echo "==== 阶段14：收尾信息 ===="
echo "conda 环境: pointcept"
cd "${ROOT}/Pointcept" && echo "Pointcept: $(git describe --tags --always) $(git rev-parse --short HEAD)"
echo "项目根: ${ROOT}"
echo "数据: ${ROOT}/data/scannet_processed"
echo "权重: ${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base"
echo "输出: ${ROOT}/runs/ptv3_scannet_val"
echo "重复运行测试（最短）："
echo "  conda activate pointcept && cd ${ROOT}/Pointcept && export PYTHONPATH=./ && python tools/test.py --config-file ${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/config.py --num-gpus 1 --options save_path=${ROOT}/runs/ptv3_scannet_val weight=${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth"
