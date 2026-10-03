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


export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"

# Synapse 测试必须使用与训练一致的模型配置；启动时从 best.pth 同目录读取训练配置。
# VOLUME_PATH 和 LIST_DIR 可在调用脚本时覆盖，用于指向服务器实际存放的完整病例及划分列表。
VOLUME_PATH="${VOLUME_PATH:-${PROJECT_DIR}/../data/Synapse/test_vol_h5}"
LIST_DIR="${LIST_DIR:-${PROJECT_DIR}/../data/Synapse/lists/lists_Synapse}"
DATASET="Synapse"

# 默认在Synapse对应的训练输出目录中查找 best.pth；CKPT 非空时优先使用调用者明确指定的路径。
CKPT="${CKPT:-}"

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[ERROR] CKPT is missing or does not exist: ${CKPT}"
  echo "[ERROR] Example: CKPT=/absolute/path/to/best.pth bash sh/start_test_synapse.sh"
  exit 1
fi

# 将 checkpoint 转为绝对路径，确保其相邻 config.json 可从任意工作目录稳定定位。
CKPT="$(realpath "${CKPT}")"
# 每个训练实验目录由 train_synapse.py 同时保存 best.pth 和 config.json。
TRAIN_CONFIG="${TRAIN_CONFIG:-$(dirname "${CKPT}")/config.json}"
test -f "${TRAIN_CONFIG}" || {
  echo "[ERROR] Training config not found: ${TRAIN_CONFIG}"
  echo "[ERROR] Expected config.json next to the selected checkpoint, or set TRAIN_CONFIG explicitly."
  exit 1
}

# 在启动 Python 前验证完整病例数据目录和 test_vol.txt 列表，错误时给出明确路径。
test -d "${VOLUME_PATH}" || { echo "[ERROR] VOLUME_PATH not found: ${VOLUME_PATH}"; exit 1; }
test -f "${LIST_DIR}/test_vol.txt" || { echo "[ERROR] test_vol.txt not found: ${LIST_DIR}/test_vol.txt"; exit 1; }


TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
LOG_FILE="${LOG_DIR}/test_${DATASET}_${TS}_${RAND}.log"


RUN_ID="test_${DATASET}_${TS}_gpu_${CUDA_VISIBLE_DEVICES}_RAND_${RAND}"
PID_FILE="${RUN_ID}.pid"

PARAM_NAMES=(
  CONDA_BASE
  CONDA_ENV_PREFIX
  PROJECT_DIR
  LOG_DIR
  CUDA_VISIBLE_DEVICES
  CKPT
  TRAIN_CONFIG
  VOLUME_PATH
  LIST_DIR
  DATASET
  TS
  RAND
  LOG_FILE
  RUN_ID
  PID_FILE
)

{
  echo "[INFO] parameters:"
  for name in "${PARAM_NAMES[@]}"; do
    printf '[INFO] %-24s=%s\n' "$name" "${!name}"
  done
  echo "---------------------------ready to run---------------------------------"
} | tee -a "${LOG_FILE}"


# 整个反斜杠块是一条测试命令：nohup抵抗终端断开，env写入RUN_ID供停止时核验进程身份。
# 测试模型参数和输入尺寸从 TRAIN_CONFIG 自动恢复；本启动器只传 checkpoint 与测试数据位置。
# stdout/stderr 写入同一测试日志，stdin 断开，末尾 & 让启动器立即返回。
nohup env RUN_ID="${RUN_ID}" python -u test_synapse.py \
  --dataset "${DATASET}" \
  --checkpoint "${CKPT}" \
  --train_config "${TRAIN_CONFIG}" \
  --volume_path "${VOLUME_PATH}" \
  --list_dir "${LIST_DIR}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &


PID=$!
echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
