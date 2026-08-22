#!/usr/bin/env bash
# Bash严格模式防止缺参数、坏路径或失败管道被静默忽略。
set -euo pipefail

# 获取项目根绝对路径并切换过去，统一相对路径解析。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# conda和Python入口支持环境变量覆盖；conda初始化文件缺失时不激活环境。
CONDA_BASE="${CONDA_BASE:-/base/mambaforge}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sld_emcad}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

# 限定GPU并使测试日志无缓冲写出。
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export PYTHONUNBUFFERED=1

# 选择ISIC版本和val/test划分。
DATASET="ISIC"
DATASET_NAME="${DATASET_NAME:-ISIC2018}"
SPLIT="${SPLIT:-test}"

# 推理批量、工作进程、阈值、病例上限、可复现性和设备选择。
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
THRESHOLD="${THRESHOLD:-0.5}"
MAX_CASES="${MAX_CASES:-0}"
DETERMINISTIC="${DETERMINISTIC:-1}"
SEED="${SEED:-2222}"
DEVICE="${DEVICE:-auto}"

# CKPT必须显式提供；prepared数据位于DATA_ROOT/<ISIC版本>/<划分>。
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../data/isic/target}"
CKPT="${CKPT:-}"

# 两个case块分别限定合法数据集和划分，非法输入统一退出1。
case "${DATASET_NAME}" in
  ISIC2017|ISIC2018)
    ;;
  *)
    echo "[ERROR] DATASET_NAME must be ISIC2017 or ISIC2018"
    exit 1
    ;;
esac

case "${SPLIT}" in
  val|test)
    ;;
  *)
    echo "[ERROR] SPLIT must be val or test"
    exit 1
    ;;
esac

# 检查点为空或不存在时打印可直接参考的启动示例。
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[ERROR] CKPT is missing or does not exist: ${CKPT}"
  echo "[ERROR] Example:"
  echo "CKPT=/absolute/path/to/best.pth DATASET_NAME=ISIC2018 bash start_test_isic.sh"
  exit 1
fi

# 检查所选划分的图像和掩膜目录。
test -d "${DATA_ROOT}/${DATASET_NAME}/${SPLIT}/images" || {
  echo "[ERROR] images directory not found:"
  echo "${DATA_ROOT}/${DATASET_NAME}/${SPLIT}/images"
  exit 1
}

test -d "${DATA_ROOT}/${DATASET_NAME}/${SPLIT}/masks" || {
  echo "[ERROR] masks directory not found:"
  echo "${DATA_ROOT}/${DATASET_NAME}/${SPLIT}/masks"
  exit 1
}

# 规范检查点绝对路径，并要求同目录config.json记录训练时模型配置。
CKPT_DIR="$(cd "$(dirname "${CKPT}")" && pwd)"
CKPT="${CKPT_DIR}/$(basename "${CKPT}")"
CONFIG_FILE="${CKPT_DIR}/config.json"

# 缺少config.json时无法可靠重建训练架构，因此退出1。
test -f "${CONFIG_FILE}" || {
  echo "[ERROR] checkpoint config not found:"
  echo "${CONFIG_FILE}"
  exit 1
}

# 默认输出目录和CSV放在检查点目录旁，环境变量可覆盖。
TEST_SAVE_DIR="${TEST_SAVE_DIR:-${CKPT_DIR}/${SPLIT}_${DATASET_NAME}_outputs}"
OUTPUT_CSV="${OUTPUT_CSV:-${TEST_SAVE_DIR}/${SPLIT}_metrics.csv}"

# 生成唯一RUN_ID并建立日志/PID关联。
TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"

LOG_FILE="${LOG_DIR}/test_${DATASET}_${DATASET_NAME}_${SPLIT}_${TS}.log"
RUN_ID="test_${DATASET}_${DATASET_NAME}_${SPLIT}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"
PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

# 将路径、阈值和RUN_ID追加到日志，方便复核本次评估配置。
echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATASET_NAME=${DATASET_NAME}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] SPLIT=${SPLIT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATA_ROOT=${DATA_ROOT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] CKPT=${CKPT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] CONFIG_FILE=${CONFIG_FILE}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] TEST_SAVE_DIR=${TEST_SAVE_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] THRESHOLD=${THRESHOLD}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"

echo "---------------------------ready to test----------------------------------" | tee -a "${LOG_FILE}" > /dev/null

# 整个续行块是一条nohup后台命令；模型结构由检查点旁config.json在test_isic.py中恢复。
# 输出、错误、输入和RUN_ID的处理方式与其他启动器一致；--save_probabilities额外导出概率图。
nohup env RUN_ID="${RUN_ID}" "${PYTHON_BIN}" -u test_isic.py \
  --checkpoint "${CKPT}" \
  --data_root "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --output_dir "${TEST_SAVE_DIR}" \
  --output_csv "${OUTPUT_CSV}" \
  --inference_batch_size "${INFERENCE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --threshold "${THRESHOLD}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --max_cases "${MAX_CASES}" \
  --device "${DEVICE}" \
  --save_probabilities \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

# 获取后台PID、写PID文件并报告日志及结果目录。
PID=$!

echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV}"
echo "[INFO] PREDICTIONS=${TEST_SAVE_DIR}"
