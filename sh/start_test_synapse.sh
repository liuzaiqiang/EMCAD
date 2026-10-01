#!/usr/bin/env bash
# Bash严格模式：测试准备阶段出现失败、未定义变量或失败管道时立即退出。
set -euo pipefail


PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


cd "${PROJECT_DIR}"


# 创建测试日志目录；所有后台stdout/stderr会追加到该目录中的本次日志。
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"


# Synapse测试脚本使用固定conda安装位置和环境名，不读取外部覆盖值。
CONDA_BASE="/base/mambaforge"
CONDA_ENV_PREFIX="/root/shared-nvme/lzq_conda/envs/sld_emcad"
#CONDA_BASE="/home/mlf/anaconda3"
#CONDA_ENV_PREFIX="/home/mlf/anaconda3/envs/sld_emcad"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"


export CUDA_VISIBLE_DEVICES=0



IMG_SIZE=224
DATASET="Synapse"



SEED=2222
TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
LOG_FILE="${LOG_DIR}/test_${DATASET}__img${IMG_SIZE}_${TS}.log"


RUN_ID="test_${DATASET}_imgSize_${IMG_SIZE}_${TS}_RAND${RAND}"
PID_FILE="${RUN_ID}.pid"

VOLUME_PATH="../data/Synapse/test_vol_h5"
LIST_DIR="../data/Synapse/lists/lists_Synapse"

CHECKPOINT="./model_pth/Synapse/encoder_pvt_v2_b2/img_size_224/seed2222/batch_size_16/lr_0.0001/maxEpochs_400/best.pth"

num_classes=9
# 当前项目支持 pvt_v2_b0 pvt_v2_b1 pvt_v2_b2 pvt_v2_b3 pvt_v2_b4 pvt_v2_b5 resnet18 resnet34 resnet50 resnet101 resnet152
encoder=pvt_v2_b2
#它控制 MSCB 模块中间隐藏通道的扩展倍数。
expansion_factor=2
kernel_sizes=[1, 3, 5]
lgag_ks=3
activation_mscb=relu6


PARAM_NAMES=(
  CONDA_BASE
  CONDA_ENV_PREFIX
  PROJECT_DIR
  LOG_DIR
  CUDA_VISIBLE_DEVICES
  SEED
  DATASET
  IMG_SIZE
  TS
  RAND
  LOG_FILE
  RUN_ID
  PID_FILE

  VOLUME_PATH
  LIST_DIR

  CHECKPOINT
)

{
  echo "[INFO] parameters:"
  for name in "${PARAM_NAMES[@]}"; do
    printf '[INFO] %-24s=%s\n' "$name" "${!name}"
  done
  echo "---------------------------ready to run---------------------------------"
} | tee -a "${LOG_FILE}"


test -d "${VOLUME_PATH}" || { echo "[ERROR] VOLUME_PATH not found: ${VOLUME_PATH}" | tee -a "${LOG_FILE}"; exit 1; }

# 整个反斜杠块是一条测试命令：nohup抵抗终端断开，env写入RUN_ID供停止时核验进程身份。
# stdout追加日志且stderr合并；末尾&转入后台。此脚本未显式写< /dev/null，stdin处理由nohup实现决定。
nohup env RUN_ID="${RUN_ID}"   python -u test_synapse.py \
  --dataset "${DATASET}" \
  --img_size "${IMG_SIZE}" \
  --seed "${SEED}" \
  --volume_path "${VOLUME_PATH}" \
  --list_dir "${LIST_DIR}" \
  --checkpoint "${CHECKPOINT}" \
   --num_classes "${num_classes}" \
  --encoder "${encoder}" \
  --expansion_factor "${expansion_factor}" \
  --kernel_sizes "${kernel_sizes}" \
  --lgag_ks "${lgag_ks}" \
  --activation_mscb "${activation_mscb}" \
  >> "${LOG_FILE}" 2>&1 &


PID=$!
echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
