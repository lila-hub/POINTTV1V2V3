#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pointcept
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

echo "==== 1/10 测试子集：准备 test_sub10 ===="
python "${ROOT}/scripts/make_test_sub10_split.py" "${ROOT}/data/scannet_processed"

mkdir -p "${ROOT}/Pointcept/data"
ln -sfn "${ROOT}/data/scannet_processed" "${ROOT}/Pointcept/data/scannet"

OUT="${ROOT}/runs/ptv3_test_sub10"
mkdir -p "$OUT"
LOG="${OUT}/test_sub10.log"

echo "==== 运行 tools/test.py（10 个 test 场景）===="
cd "${ROOT}/Pointcept"
export PYTHONPATH=./
python tools/test.py \
  --config-file "${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/config.py" \
  --num-gpus 1 \
  --options \
    save_path="${OUT}" \
    weight="${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth" \
    data.test.split=test_sub10 \
    num_worker=4 \
  2>&1 | tee "$LOG"

echo "==== 可视化 + 报告 ===="
pip install -q matplotlib
python "${ROOT}/scripts/make_sub10_vis_and_report.py"
echo "完成。报告: ${OUT}/SUB10_REPORT.md"
