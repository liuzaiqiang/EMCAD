#!/usr/bin/env bash
# Bash严格模式确保参数准备和路径校验失败时不会继续启动测试。
set -euo pipefail

# 切换到脚本所在项目根，使数据、日志和PID路径可预测。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# conda/Python入口允许由外部环境变量覆盖；初始化脚本不存在时沿用当前环境。
CONDA_BASE="${CONDA_BASE:-/base/mambaforge}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sld_emcad}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

# 选择GPU并关闭Python标准流缓冲。
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export PYTHONUNBUFFERED=1

# DATASET_NAME选择具体息肉数据集，SPLIT只允许val或test。
DATASET="Polyp"
DATASET_NAME="${DATASET_NAME:-ClinicDB}"
SPLIT="${SPLIT:-test}"

# 推理尺寸、批量、DataLoader、二值阈值、病例上限、随机性和设备参数。
IMG_SIZE="${IMG_SIZE:-352}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
THRESHOLD="${THRESHOLD:-0.5}"
MAX_CASES="${MAX_CASES:-0}"
DETERMINISTIC="${DETERMINISTIC:-1}"
SEED="${SEED:-2222}"
DEVICE="${DEVICE:-auto}"

# 必须与检查点训练时架构一致的EMCAD配置；本脚本直接把这些值传给test_polyp.py。
ENCODER="${ENCODER:-pvt_v2_b2}"
EXPANSION_FACTOR="${EXPANSION_FACTOR:-2}"
LGAG_KS="${LGAG_KS:-3}"
ACTIVATION_MSCB="${ACTIVATION_MSCB:-relu6}"

# CKPT默认空，必须由调用者显式提供；DATA_ROOT指向prepared数据根。
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../data/polyp/target}"
CKPT="${CKPT:-}"

# 数据集名只允许安全文件名字符，防止拼接出意外路径。
[[ "${DATASET_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "[ERROR] invalid DATASET_NAME: ${DATASET_NAME}"
  exit 1
}

# case块限制划分名；非法值以退出码1结束。
case "${SPLIT}" in
  val|test)
    ;;
  *)
    echo "[ERROR] SPLIT must be val or test"
    exit 1
    ;;
esac

# 检查点为空或文件不存在都会打印用法示例并退出1。
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[ERROR] CKPT is missing or does not exist: ${CKPT}"
  echo "[ERROR] Example:"
  echo "CKPT=/absolute/path/to/best.pth DATASET_NAME=ClinicDB bash start_test_polyp.sh"
  exit 1
fi

# 验证所选划分的图像和掩膜目录存在。
test -d "${DATA_ROOT}/${DATASET_NAME}/${SPLIT}/images" || {
  echo "[ERROR] images directory not found"
  exit 1
}

test -d "${DATA_ROOT}/${DATASET_NAME}/${SPLIT}/masks" || {
  echo "[ERROR] masks directory not found"
  exit 1
}

# 把检查点目录和文件名规范成绝对路径，避免后台进程受工作目录变化影响。
CKPT_DIR="$(cd "$(dirname "${CKPT}")" && pwd)"
CKPT="${CKPT_DIR}/$(basename "${CKPT}")"

# 默认把预测、概率图和CSV写在检查点旁边，便于模型与结果一一对应。
TEST_SAVE_DIR="${TEST_SAVE_DIR:-${CKPT_DIR}/${SPLIT}_${DATASET_NAME}_outputs}"
OUTPUT_CSV="${OUTPUT_CSV:-${TEST_SAVE_DIR}/test_metrics.csv}"

# 时间戳和随机后缀生成唯一RUN_ID、日志名和PID文件。
TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"

LOG_FILE="${LOG_DIR}/test_${DATASET}_${DATASET_NAME}_${SPLIT}__img${IMG_SIZE}_${TS}.log"
RUN_ID="test_${DATASET}_${DATASET_NAME}_${SPLIT}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"
PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

# 把实际解析后的路径写入日志；RUN_ID同时显示在终端。
echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATASET_NAME=${DATASET_NAME}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] SPLIT=${SPLIT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATA_ROOT=${DATA_ROOT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] CKPT=${CKPT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] TEST_SAVE_DIR=${TEST_SAVE_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"

echo "---------------------------ready to test----------------------------------" | tee -a "${LOG_FILE}" > /dev/null

# 单个多行命令块：nohup+后台执行，RUN_ID注入进程环境，-u关闭Python缓冲。
# --save_probabilities要求同时保存归一化概率图；stdout/stderr追加日志，stdin断开。
nohup env RUN_ID="${RUN_ID}" "${PYTHON_BIN}" -u test_polyp.py \
  --checkpoint "${CKPT}" \
  --data_root "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --output_dir "${TEST_SAVE_DIR}" \
  --output_csv "${OUTPUT_CSV}" \
  --encoder "${ENCODER}" \
  --kernel_sizes 1 3 5 \
  --expansion_factor "${EXPANSION_FACTOR}" \
  --lgag_ks "${LGAG_KS}" \
  --activation_mscb "${ACTIVATION_MSCB}" \
  --img_size "${IMG_SIZE}" \
  --inference_batch_size "${INFERENCE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --threshold "${THRESHOLD}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --max_cases "${MAX_CASES}" \
  --device "${DEVICE}" \
  --save_probabilities \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

# 保存后台PID；stop_test_polyp.sh会核验PID文件和/proc中的RUN_ID后再发送信号。
PID=$!

echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV}"
echo "[INFO] PREDICTIONS=${TEST_SAVE_DIR}"
