#!/usr/bin/env bash
# Bash严格模式：测试准备阶段出现失败、未定义变量或失败管道时立即退出。
set -euo pipefail


PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


cd "${PROJECT_DIR}"


# 创建测试日志目录；所有后台stdout/stderr会追加到该目录中的本次日志。
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"


# Synapse测试脚本使用固定conda安装位置和环境名，不读取外部覆盖值。
#CONDA_BASE="/base/mambaforge"
#CONDA_ENV_PREFIX="/root/shared-nvme/lzq_conda/envs/sld_emcad"


CONDA_BASE="/home/mlf/anaconda3"
CONDA_ENV_PREFIX="/home/mlf/anaconda3/envs/sld_emcad"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"


export CUDA_VISIBLE_DEVICES=0



IMG_SIZE=224
DATASET="Synapse"



SEED=2222
TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
LOG_FILE="${LOG_DIR}/test_${DATASET}__img${IMG_SIZE}_${TS}.log"


RUN_ID="test_${DATASET}_imgSize_${IMG_SIZE}_supervision_${SUPERVISION}_batchSize_seed${seed}_maxepochs_${TS}_RAND${RAND}"
PID_FILE="${RUN_ID}.pid"

FUSION_MODE="pixel_reliability"
FUSION_LOSS_WEIGHT="1"
RELIABILITY_LOSS_WEIGHT="1"
CAA_MODE="caa"
CAA_RESIDUAL_SCALE="0.1"

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

  NUM_WORKERS

  FUSION_MODE
  FUSION_LOSS_WEIGHT
  RELIABILITY_LOSS_WEIGHT
  CAA_MODE
  CAA_RESIDUAL_SCALE

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
  --fusion_mode "${FUSION_MODE}" \
  --fusion_loss_weight "${FUSION_LOSS_WEIGHT}" \
  --reliability_loss_weight "${RELIABILITY_LOSS_WEIGHT}" \
  --caa_mode "${CAA_MODE}" \
  --caa_residual_scale "${CAA_RESIDUAL_SCALE}" \


  >> "${LOG_FILE}" 2>&1 &


PID=$!
echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
