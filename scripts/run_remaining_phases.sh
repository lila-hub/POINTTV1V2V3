#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${ROOT}/logs"
LOG="${ROOT}/logs/ptv3_run.log"
exec > >(tee -a "$LOG") 2>&1

echo "==== 第6步：安装 pointops ===="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pointcept
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
cd "${ROOT}/Pointcept/libs/pointops"
if python setup.py install; then
  echo "SUCCESS: pointops via setup.py install"
else
  echo "Trying pip install -e ."
  pip install -e .
fi
cd "${ROOT}/Pointcept"
python -c "import pointops; print('pointops ok')"
echo "SUCCESS 第6步 | 下一步: 第7步 HF"

echo "==== 第7步：Hugging Face 登录 ===="
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "FAILED: 缺少 HUGGINGFACE_HUB_TOKEN。请先执行："
  echo "export HUGGINGFACE_HUB_TOKEN=你的token"
  echo "然后重新运行本脚本。"
  exit 2
fi
hf auth login --token "$HUGGINGFACE_HUB_TOKEN"
echo "SUCCESS 第7步 | 下一步: 第8步 下载数据"

echo "==== 第8步：下载预处理 ScanNet ===="
mkdir -p "${ROOT}/data/scannet_processed"
set +e
hf download Pointcept/scannet-compressed --repo-type dataset --local-dir "${ROOT}/data/scannet_processed" 2>&1
HF_DL=$?
set -e
if [[ $HF_DL -ne 0 ]]; then
  echo "FAILED: 数据集下载失败（可能是权限/403）。请在 Hugging Face 页面同意 ScanNet 数据使用条款后重试。"
  exit 3
fi
echo "SUCCESS 第8步 | 下一步: 第9步 软链接"

echo "==== 第9步：数据软链接 ===="
mkdir -p "${ROOT}/Pointcept/data"
ln -sfn "${ROOT}/data/scannet_processed" "${ROOT}/Pointcept/data/scannet"
ls -l "${ROOT}/Pointcept/data/scannet"
echo "SUCCESS 第9步 | 下一步: 第10步 权重"

echo "==== 第10步：下载 PTv3 权重 ===="
mkdir -p "${ROOT}/models/ptv3"
hf download Pointcept/PointTransformerV3 \
  --repo-type model \
  --local-dir "${ROOT}/models/ptv3" \
  --include "scannet-semseg-pt-v3m1-0-base/**"
CFG="${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/config.py"
WGT="${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth"
if [[ -f "$CFG" && -f "$WGT" ]]; then
  echo "OK: config 与权重存在"
  ls -lh "$WGT"
else
  echo "FAILED: 缺少文件，目录树："
  find "${ROOT}/models/ptv3" -maxdepth 4 -type f 2>/dev/null | head -80
  exit 4
fi
echo "SUCCESS 第10步 | 下一步: 第11步 修改 config"

echo "==== 第11步：修改 config ===="
python3 << PY
from pathlib import Path
p = Path("${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/config.py")
text = p.read_text(encoding="utf-8")
text = text.replace("enable_flash=True", "enable_flash=False")
text = text.replace("enable_flash = True", "enable_flash = False")
text = text.replace(
    "enc_patch_size=(1024, 1024, 1024, 1024, 1024)",
    "enc_patch_size=(128, 128, 128, 128, 128)",
)
text = text.replace(
    "dec_patch_size=(1024, 1024, 1024, 1024)",
    "dec_patch_size=(128, 128, 128, 128)",
)
p.write_text(text, encoding="utf-8")
lines = p.read_text(encoding="utf-8").splitlines()
for line in lines:
    if "enable_flash" in line or "enc_patch_size" in line or "dec_patch_size" in line:
        print(line)
PY
echo "SUCCESS 第11步 | 下一步: 第12步 测试"

echo "==== 第12步：启动测试 ===="
mkdir -p "${ROOT}/runs/ptv3_scannet_val"
cd "${ROOT}/Pointcept"
export PYTHONPATH=./
set +e
python tools/test.py \
  --config-file "${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/config.py" \
  --num-gpus 1 \
  --options \
    save_path="${ROOT}/runs/ptv3_scannet_val" \
    weight="${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth"
TEST_EXIT=$?
set -e
if [[ $TEST_EXIT -ne 0 ]]; then
  echo "==== 第13步：自动排查 ===="
  which python
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  python -c "import pointops; print('pointops ok')" || true
  ls -la data/scannet 2>/dev/null | head -20
  ls -lh "${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth"
  python3 -c "
from pathlib import Path
t = Path('${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/config.py').read_text()
for k in ['data_root', 'enable_flash', 'enc_patch_size', 'dec_patch_size']:
    for line in t.splitlines():
        if k in line and not line.strip().startswith('#'):
            print(line.strip())
            break
"
  echo "测试退出码: $TEST_EXIT"
  exit "$TEST_EXIT"
fi

echo "==== 第14步：收尾 ===="
cd "${ROOT}/Pointcept"
echo "conda 环境: pointcept"
echo "Pointcept: $(git describe --tags --exact-match 2>/dev/null || true) commit $(git rev-parse HEAD)"
echo "项目根: ${ROOT}"
echo "数据: ${ROOT}/data/scannet_processed"
echo "权重: ${ROOT}/models/ptv3"
echo "输出: ${ROOT}/runs/ptv3_scannet_val"
echo "SUCCESS 全流程"
