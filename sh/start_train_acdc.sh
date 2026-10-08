#!/usr/bin/env bash
# Bash 严格模式：命令失败、未定义变量或管道中间失败时立即退出，避免带错误路径继续启动任务。
set -euo pipefail



#CONDA_BASE="/home/mlf/anaconda3"
#CONDA_ENV_PREFIX="/home/mlf/anaconda3/envs/sld_emcad"
#source "${CONDA_BASE}/etc/profile.d/conda.sh"
#conda activate "${CONDA_ENV_PREFIX}"


CONDA_BASE="/data/miniconda3"
CONDA_ENV_PREFIX="/data/miniconda3/envs/sld_emcad"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"


# 解析脚本所在目录为项目绝对路径，并切换过去，使后续相对路径和PID文件位置稳定。
#PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_DIR}"


# 所有后台输出集中写入项目 logs 目录。
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"



export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
N_GPU="${N_GPU:-1}"


export PYTHONUNBUFFERED=1

SEED="${SEED:-2222}"
IMG_SIZE="${IMG_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-12}"
MAX_EPOCHS="${MAX_EPOCHS:-400}"

# ACDC 是4类心脏MRI分割；这些环境变量可在调用脚本前覆盖。
DATASET="ACDC"
BASE_LR="${BASE_LR:-1e-4}"
SUPERVISION="${SUPERVISION:-mutation}"
NUM_WORKERS="${NUM_WORKERS:-8}"


DETERMINISTIC="${DETERMINISTIC:-1}"


# 数据根目录包含 train/valid/test；列表目录保存各划分病例名；模型输出固定写入 model_pth/ACDC。
LIST_DIR="${PROJECT_DIR}/../data/ACDC/lists_ACDC"
ROOT_PATH="${PROJECT_DIR}/../data/ACDC"
OUTPUT_DIR="${PROJECT_DIR}/model_pth/ACDC"

# 启动前逐项验证训练/验证目录、列表和PVTv2-B2预训练权重；失败统一返回退出码1。
test -d "${ROOT_PATH}/train" || { echo "[ERROR] ROOT_PATH/train not found: ${ROOT_PATH}/train"; exit 1; }
test -d "${ROOT_PATH}/valid" || { echo "[ERROR] ROOT_PATH/valid not found: ${ROOT_PATH}/valid"; exit 1; }
test -f "${LIST_DIR}/train.txt" || { echo "[ERROR] train list not found: ${LIST_DIR}/train.txt"; exit 1; }
test -f "${LIST_DIR}/valid.txt" || { echo "[ERROR] valid list not found: ${LIST_DIR}/valid.txt"; exit 1; }
test -f "${PROJECT_DIR}/pretrained_pth/pvt/pvt_v2_b2.pth" || {
  echo "[ERROR] pretrained model not found: ${PROJECT_DIR}/pretrained_pth/pvt/pvt_v2_b2.pth"
  echo "[ERROR] This Synapse-style launcher uses the pretrained EMCAD encoder."
  exit 1
}

# 秒级时间戳加随机十六进制后缀组成唯一运行标识，避免并发启动相互覆盖日志/PID文件。
TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
# 日志名记录关键训练规模，RUN_ID额外记录GPU和随机种子。
LOG_FILE="${LOG_DIR}/train_${DATASET}_imgSize_${IMG_SIZE}_batchSize_${BATCH_SIZE}_lr_${BASE_LR}_max_epochs_${MAX_EPOCHS}_${TS}.log"
RUN_ID="train_${DATASET}_${TS}_gpu_${CUDA_VISIBLE_DEVICES}_SEED_${SEED}_RAND_${RAND}"
# 脚本已cd到项目根，因此相对PID文件位于项目根目录。
PID_FILE="${RUN_ID}.pid"


# 两个模块分别消融；设置为0时参数明确传为p1或off，相关网络结构/损失不会启用。
USE_PIXEL_RELIABILITY_FUSION="${USE_PIXEL_RELIABILITY_FUSION:-0}"
USE_CONTENT_AWARE_ANTIALIAS="${USE_CONTENT_AWARE_ANTIALIAS:-0}"

FUSION_MODE="${FUSION_MODE:-pixel_reliability}"
FUSION_LOSS_WEIGHT="${FUSION_LOSS_WEIGHT:-1}"
RELIABILITY_LOSS_WEIGHT="${RELIABILITY_LOSS_WEIGHT:-1}"
CAA_MODE="${CAA_MODE:-caa}"

CAA_RESIDUAL_SCALE="${CAA_RESIDUAL_SCALE:-0.1}"
case "${USE_PIXEL_RELIABILITY_FUSION}" in
  0) FUSION_MODE="p1"; FUSION_LOSS_WEIGHT="0" ;;
  1) ;;
  *) echo "[ERROR] USE_PIXEL_RELIABILITY_FUSION must be 0 or 1"; exit 1 ;;
esac
case "${USE_CONTENT_AWARE_ANTIALIAS}" in
  0) CAA_MODE="off" ;;
  1) ;;
  *) echo "[ERROR] USE_CONTENT_AWARE_ANTIALIAS must be 0 or 1"; exit 1 ;;
esac


# tee -a把配置写入日志；RUN_ID同时打印到终端，供stop_train_acdc.sh作为参数使用。
PARAM_NAMES=(CONDA_BASE CONDA_ENV_PREFIX PROJECT_DIR LOG_DIR CUDA_VISIBLE_DEVICES SEED IMG_SIZE BATCH_SIZE MAX_EPOCHS DATASET BASE_LR SUPERVISION NUM_WORKERS N_GPU DETERMINISTIC LIST_DIR ROOT_PATH OUTPUT_DIR USE_PIXEL_RELIABILITY_FUSION USE_CONTENT_AWARE_ANTIALIAS FUSION_MODE FUSION_LOSS_WEIGHT RELIABILITY_LOSS_WEIGHT CAA_MODE CAA_RESIDUAL_SCALE TS RAND LOG_FILE RUN_ID PID_FILE)
echo "[INFO] RUN_ID=${RUN_ID}"

{
  echo "[INFO] parameters:"
  for name in "${PARAM_NAMES[@]}"; do
    printf '[INFO] %-24s=%s\n' "$name" "${!name}"
  done
  echo "---------------------------ready to train---------------------------------"
} | tee -a "${LOG_FILE}"

# 整个反斜杠块是一条后台命令：nohup抵抗终端断开，env注入RUN_ID用于停止时核验进程身份。
# -u和PYTHONUNBUFFERED共同减少日志缓冲；stdout追加到日志，stderr合并，stdin断开，&立即返回控制权。
nohup env RUN_ID="${RUN_ID}" python -u train_acdc.py \
  --root_path "${ROOT_PATH}" \
  --list_dir "${LIST_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_ID}" \
  --encoder pvt_v2_b2 \
  --kernel_sizes 1 3 5 \
  --expansion_factor 2 \
  --lgag_ks 3 \
  --activation_mscb relu6 \
  --supervision "${SUPERVISION}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --base_lr "${BASE_LR}" \
  --num_workers "${NUM_WORKERS}" \
  --n_gpu "${N_GPU}" \
  --deterministic "${DETERMINISTIC}" \
  --caa_mode "${CAA_MODE}" \
  --caa_residual_scale "${CAA_RESIDUAL_SCALE}" \
  --seed "${SEED}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

# 保存刚启动后台任务的PID；脚本本身成功结束不代表训练完成，只表示启动和PID记录成功。
PID=$!
echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
# 输出PID文件和日志位置，方便监控及后续停止。
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
