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


#这条命令得到的是项目路径值，而不是sh/下的路径值
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


cd "${PROJECT_DIR}"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES=0


SEED=2222
MAX_EPOCHS=300
DATASET="Synapse"
IMG_SIZE=224
BATCH_SIZE=16
SUPERVISION="mutation"
BASE_LR=1e-4


num_classes=9
# 当前项目支持 pvt_v2_b0 pvt_v2_b1 pvt_v2_b2 pvt_v2_b3 pvt_v2_b4 pvt_v2_b5 resnet18 resnet34 resnet50 resnet101 resnet152
encoder=pvt_v2_b2
#它控制 MSCB 模块中间隐藏通道的扩展倍数。
expansion_factor=2
kernel_sizes=[1, 3, 5]
lgag_ks=3
activation_mscb=relu6



RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
TS="$(date +%F_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${DATASET}_imgSize_${IMG_SIZE}_supervision_${SUPERVISION}_bs_${BATCH_SIZE}_seed_${SEED}_lr_${BASE_LR}_maxepo_${MAX_EPOCHS}_ts_${TS}_RAND_${RAND}.log"
RUN_ID="$(basename "${LOG_FILE}" .log)"
PID_FILE="${LOG_DIR}/${RUN_ID}.pid"

LIST_DIR="${PROJECT_DIR}/../data/Synapse/lists/lists_Synapse"
ROOT_PATH="../data/Synapse/train_npz"
VOLUME_PATH="../data/Synapse/test_vol_h5"


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

  NUM_WORKERS

  LIST_DIR
  ROOT_PATH
  VOLUME_PATH

  num_classes
  encoder
  expansion_factor
  kernel_sizes
  lgag_ks
  activation_mscb
)

{
  echo "[INFO] parameters:"
  for name in "${PARAM_NAMES[@]}"; do
    printf '[INFO] %-24s=%s\n' "$name" "${!name}"
  done
  echo "---------------------------ready to run---------------------------------"
} | tee -a "${LOG_FILE}"


# 启动命令块：nohup使进程忽略终端挂断信号；env把RUN_ID写入子进程环境，停止脚本会读取它防止PID复用误杀。  >> 追加标准输出，2>&1把标准错误并入同一日志，< /dev/null断开标准输入
nohup env RUN_ID="${RUN_ID}" python -u train_synapse.py \
  --dataset "${DATASET}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --base_lr "${BASE_LR}" \
  --seed "${SEED}" \
  --supervision "${SUPERVISION}" \
  --list_dir "${LIST_DIR}" \
  --root_path "${ROOT_PATH}" \
  --volume_path "${VOLUME_PATH}" \
  --num_classes "${num_classes}" \
  --encoder "${encoder}" \
  --expansion_factor "${expansion_factor}" \
  --kernel_sizes "${kernel_sizes}" \
  --lgag_ks "${lgag_ks}" \
  --activation_mscb "${activation_mscb}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &


PID=$!
echo "[INFO] PID=${PID}"
#PID 文件在项目根目录
echo "${PID}" > "${PID_FILE}"
