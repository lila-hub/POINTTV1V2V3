#!/usr/bin/env bash
# 先确保已 convert，再后台跑测试，日志写入 test_bg.log
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${ROOT}/logs/test_bg.log"
PIDF="${ROOT}/logs/test_bg.pid"
mkdir -p "${ROOT}/logs"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pointcept
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${ROOT}/Pointcept/"

echo "==== $(date -Iseconds) 转换 npy -> pth ====" | tee -a "$LOG"
python "${ROOT}/scripts/convert_scannet_npy_to_pth.py" "${ROOT}/data/scannet_processed" 2>&1 | tee -a "$LOG"

echo "==== $(date -Iseconds) 启动后台测试 ====" | tee -a "$LOG"
nohup bash -c "
  source \"\$(conda info --base)/etc/profile.d/conda.sh\"
  conda activate pointcept
  export CUDA_HOME=\"\${CONDA_PREFIX}\"
  cd \"${ROOT}/Pointcept\"
  export PYTHONPATH=./
  exec python tools/test.py \\
    --config-file \"${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/config.py\" \\
    --num-gpus 1 \\
    --options \\
      save_path=\"${ROOT}/runs/ptv3_scannet_val\" \\
      weight=\"${ROOT}/models/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth\"
" >> "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "PID=$(cat "$PIDF") 日志: $LOG"
echo "查看: tail -f $LOG"
