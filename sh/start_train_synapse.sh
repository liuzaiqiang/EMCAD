#!/usr/bin/env bash
# 使用 Bash 解释器；本脚本依赖 pipefail、BASH_SOURCE 和参数展开等 Bash 行为。
# -e 遇到未处理的非零退出码即停止，-u 拒绝未定义变量，pipefail 让管道任一环节失败都算失败。
set -euo pipefail




# CONDA_BASE="/base/mambaforge"
# CONDA_ENV_PREFIX="/root/shared-nvme/lzq_conda/envs/sld_emcad"
CONDA_BASE="/home/mlf/anaconda3"
CONDA_ENV_PREFIX="/home/mlf/anaconda3/envs/sld_emcad"


source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"


PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 后续相对路径和 PID 文件均以项目根目录为当前工作目录。
cd "${PROJECT_DIR}"

# 统一日志目录；-p 会在目录已存在时保持成功，并递归创建缺失的父目录。
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"


export CUDA_VISIBLE_DEVICES=0
#windows环境下运行时，设置为0（0 表示由主进程加载数据，最稳定）;linux环境下运行时，设置为8。
#NUM_WORKERS=8



SEED=2222
DATASET="Synapse"
IMG_SIZE=224
BATCH_SIZE=16
SUPERVISION="mutation"
MAX_EPOCHS=400
BASE_LR=1e-4


FUSION_MODE="pixel_reliability"
FUSION_LOSS_WEIGHT="1"
RELIABILITY_LOSS_WEIGHT="1"


RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
TS="$(date +%F_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${DATASET}_imgSize_${IMG_SIZE}_supervision_${SUPERVISION}_bs_${BATCH_SIZE}_seed_${SEED}_lr_${BASE_LR}_maxepo_${MAX_EPOCHS}_ts_${TS}_RAND_${RAND}.log"
RUN_ID="$(basename "${LOG_FILE}" .log)"
PID_FILE="${LOG_DIR}/${RUN_ID}.pid"


PARAM_NAMES=(
  CONDA_BASE
  CONDA_ENV_PREFIX
  PROJECT_DIR
  LOG_DIR
  CUDA_VISIBLE_DEVICES 
  SEED
  MAX_EPOCHS
  DATASET
  IMG_SIZE
  BATCH_SIZE
  SUPERVISION
  BASE_LR
  TS
  RAND
  LOG_FILE
  RUN_ID
  PID_FILE

  FUSION_MODE
  FUSION_LOSS_WEIGHT
  RELIABILITY_LOSS_WEIGHT

)

{
  echo "[INFO] parameters:"
  for name in "${PARAM_NAMES[@]}"; do
    printf '[INFO] %-24s=%s\n' "$name" "${!name}"
  done
} | tee -a "${LOG_FILE}"
echo "---------------------------ready to train---------------------------------" | tee -a "${LOG_FILE}" > /dev/null

# 启动命令块：nohup使进程忽略终端挂断信号；env把RUN_ID写入子进程环境，停止脚本会读取它防止PID复用误杀。
# 各反斜杠续行共同组成一条命令，不能在续行之间插入注释。
# >> 追加标准输出，2>&1把标准错误并入同一日志，< /dev/null断开标准输入，末尾&放入后台。
nohup env RUN_ID="${RUN_ID}" python -u train_synapse.py \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --base_lr "${BASE_LR}" \
  --seed "${SEED}" \
  --supervision "${SUPERVISION}" \
  --fusion_mode "${FUSION_MODE}" \
  --fusion_loss_weight "${FUSION_LOSS_WEIGHT}" \
  --reliability_loss_weight "${RELIABILITY_LOSS_WEIGHT}" \
  --semantic_feedback \
  --feedback_init_scale 0.1 \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

# $! 是当前 shell 最近启动的后台进程PID，即 nohup/env/python 进程链最终跟踪的训练进程。
PID=$!
echo "[INFO] PID=${PID}"
# PID文件只保存数字PID；与RUN_ID环境变量双重校验后，停止脚本才会发送终止信号。
echo "${PID}" > "${PID_FILE}"
