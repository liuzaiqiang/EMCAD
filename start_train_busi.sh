#!/usr/bin/env bash
# 严格模式使未定义变量、命令失败和管道中间失败都能立即暴露。
set -euo pipefail

# 获取脚本目录绝对路径并切换到项目根，统一后续相对路径语义。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# 后台任务的标准输出和错误统一保存在logs目录。
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 可由外部覆盖的conda和Python入口；未覆盖时使用服务器默认值。
CONDA_BASE="${CONDA_BASE:-/base/mambaforge}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sld_emcad_251}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# conda初始化脚本存在才激活环境，否则使用当前shell已有环境。
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

# 选择GPU并让Python输出不缓冲，减少日志延迟。
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export PYTHONUNBUFFERED=1

# BUSI启动器只接受固定数据集名BUSI。
DATASET="BUSI"
DATASET_NAME="${DATASET_NAME:-BUSI}"

# BUSI论文设置使用256输入并关闭多尺度；其余训练、优化、阈值和调度参数可外部覆盖。
IMG_SIZE="${IMG_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
BASE_LR="${BASE_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLIP="${CLIP:-0.5}"
THRESHOLD="${THRESHOLD:-0.5}"
SCHEDULER="${SCHEDULER:-constant}"
MIN_LR="${MIN_LR:-1e-6}"

# DataLoader、随机性、验证/保存周期、调试上限和设备选择。
NUM_WORKERS="${NUM_WORKERS:-0}"
N_GPU="${N_GPU:-1}"
DETERMINISTIC="${DETERMINISTIC:-1}"
SEED="${SEED:-2222}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-50}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VALID_CASES="${MAX_VALID_CASES:-0}"
DEVICE="${DEVICE:-auto}"

# EMCAD结构及二分类监督参数。
ENCODER="${ENCODER:-pvt_v2_b2}"
EXPANSION_FACTOR="${EXPANSION_FACTOR:-2}"
LGAG_KS="${LGAG_KS:-3}"
ACTIVATION_MSCB="${ACTIVATION_MSCB:-relu6}"
SUPERVISION="${SUPERVISION:-paper}"

# 数据、模型输出和PVT权重目录；prepared BUSI数据固定位于DATA_ROOT/BUSI。
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../data/busi/target}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/model_pth}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${PROJECT_DIR}/pretrained_pth/pvt}"

# 非BUSI名称立即以退出码1停止，避免路径和数据协议不一致。
if [[ "${DATASET_NAME}" != "BUSI" ]]; then
  echo "[ERROR] DATASET_NAME must be BUSI"
  exit 1
fi

# 整个for续行块列出训练/验证图像掩膜及划分元数据；循环逐项检查存在性，缺一即退出1。
for REQUIRED in \
  "${DATA_ROOT}/BUSI/train/images" \
  "${DATA_ROOT}/BUSI/train/masks" \
  "${DATA_ROOT}/BUSI/val/images" \
  "${DATA_ROOT}/BUSI/val/masks" \
  "${DATA_ROOT}/BUSI/manifest.csv" \
  "${DATA_ROOT}/BUSI/split_summary.json"
do
  if [[ ! -e "${REQUIRED}" ]]; then
    echo "[ERROR] required BUSI path not found:"
    echo "${REQUIRED}"
    exit 1
  fi
done

# 仅PVTv2编码器执行预训练文件检查。
# 注意：下一条if的现有原文混入了PowerShell/Windows路径片段，按Bash解析可能报语法或路径错误；本次只加注释，不改原命令。
if [[ "${ENCODER}" == pvt_v2_* ]]; then
  if [[ ! -f "$th = "D:\install\python_3.12.4\Scripts;$env:Path"{PRETRAINED_DIR}/${ENCODER}.pth" ]]; then
    echo "[ERROR] pretrained model not found:"
    echo "${PRETRAINED_DIR}/${ENCODER}.pth"
    exit 1
  fi
fi

# 确保模型输出根目录存在。
mkdir -p "${OUTPUT_DIR}"

# 时间戳和系统随机后缀共同生成不会轻易冲突的RUN_ID。
TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"

# RUN_ID关联日志、PID、Python输出目录和/proc中的进程环境标识。
RUN_ID="train_${DATASET}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"
LOG_FILE="${LOG_DIR}/${RUN_ID}__imgSize${IMG_SIZE}_batchSize${BATCH_SIZE}_lr${BASE_LR}_epo${MAX_EPOCHS}.log"
PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

# 把实际启动配置记录到日志；BUSI多尺度禁用状态也显式写入。
echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATA_ROOT=${DATA_ROOT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] IMG_SIZE=${IMG_SIZE}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] BATCH_SIZE=${BATCH_SIZE} VAL_BATCH_SIZE=${VAL_BATCH_SIZE}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] MAX_EPOCHS=${MAX_EPOCHS} BASE_LR=${BASE_LR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] WEIGHT_DECAY=${WEIGHT_DECAY} CLIP=${CLIP}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] SUPERVISION=${SUPERVISION}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] MULTI_SCALE=disabled_for_BUSI" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] NUM_WORKERS=${NUM_WORKERS} SEED=${SEED}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}" | tee -a "${LOG_FILE}" > /dev/null

echo "[INFO] RUN_ID=${RUN_ID}"
echo "---------------------------ready to train---------------------------------" | tee -a "${LOG_FILE}" > /dev/null

# 单个多行后台命令：nohup防终端断开，env RUN_ID供stop脚本核验，python -u关闭解释器缓冲。
# stdout追加日志、stderr并入stdout、stdin断开；没有传scale_rates且train_busi.py会强制禁用多尺度。
nohup env RUN_ID="${RUN_ID}" "${PYTHON_BIN}" -u train_busi.py \
  --data_root "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_ID}" \
  --encoder "${ENCODER}" \
  --kernel_sizes 1 3 5 \
  --expansion_factor "${EXPANSION_FACTOR}" \
  --lgag_ks "${LGAG_KS}" \
  --activation_mscb "${ACTIVATION_MSCB}" \
  --supervision "${SUPERVISION}" \
  --pretrained_dir "${PRETRAINED_DIR}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --val_batch_size "${VAL_BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --base_lr "${BASE_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --clip "${CLIP}" \
  --scheduler "${SCHEDULER}" \
  --min_lr "${MIN_LR}" \
  --num_workers "${NUM_WORKERS}" \
  --n_gpu "${N_GPU}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --validate_every "${VALIDATE_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --threshold "${THRESHOLD}" \
  --max_train_batches "${MAX_TRAIN_BATCHES}" \
  --max_valid_cases "${MAX_VALID_CASES}" \
  --device "${DEVICE}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

# 保存后台PID；停止脚本会同时核对PID文件和RUN_ID环境，降低PID复用误杀风险。
PID=$!

echo "${PID}" > "${PID_FILE}"

echo "[INFO] PID=${PID}"
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] RUN_OUTPUT=${OUTPUT_DIR}/BUSI/${RUN_ID}"
