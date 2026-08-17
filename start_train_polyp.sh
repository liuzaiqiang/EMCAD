#!/usr/bin/env bash
# 使用Bash严格模式；任一未处理错误、未定义变量或失败管道都会终止启动，避免后台任务带错参数运行。
set -euo pipefail

# 把脚本目录解析为项目根绝对路径并切换过去，保证从任意目录执行时路径含义一致。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# 创建集中日志目录；已有目录不会报错。
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 三个变量均允许由外部环境覆盖；未设置时使用服务器默认conda位置、环境名和python命令。
CONDA_BASE="${CONDA_BASE:-/base/mambaforge}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sld_emcad_251}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# conda初始化文件存在才激活环境；不存在时保留调用脚本前的当前Python环境。
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

# CUDA_DEVICE默认0；PYTHONUNBUFFERED确保nohup日志尽快落盘。
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export PYTHONUNBUFFERED=1

# DATASET用于日志分类，DATASET_NAME选择target目录下的具体息肉数据集，默认ClinicDB。
DATASET="Polyp"
DATASET_NAME="${DATASET_NAME:-ClinicDB}"

# 训练尺寸、训练/验证批量、epoch、AdamW学习率/权重衰减和梯度值裁剪均可由环境变量覆盖。
IMG_SIZE="${IMG_SIZE:-352}"
BATCH_SIZE="${BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
BASE_LR="${BASE_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLIP="${CLIP:-0.5}"

# 运行与调试控制：0通常表示“不限制”批次数/验证病例数；DEVICE=auto由Python选择CUDA或CPU。
NUM_WORKERS="${NUM_WORKERS:-0}"
N_GPU="${N_GPU:-1}"
DETERMINISTIC="${DETERMINISTIC:-1}"
SEED="${SEED:-2222}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-50}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VALID_CASES="${MAX_VALID_CASES:-0}"
DEVICE="${DEVICE:-auto}"

# EMCAD结构参数：编码器、MSCB扩张率、LGAG核、激活和二分类监督策略。
# supervision=paper对应四个单头损失加四头logits求和后的第五项损失。
ENCODER="${ENCODER:-pvt_v2_b2}"
EXPANSION_FACTOR="${EXPANSION_FACTOR:-2}"
LGAG_KS="${LGAG_KS:-3}"
ACTIVATION_MSCB="${ACTIVATION_MSCB:-relu6}"
SUPERVISION="${SUPERVISION:-paper}"

# 数据根目录应包含 <dataset>/{train,val}/{images,masks}；输出和PVT预训练权重目录也可外部覆盖。
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../data/polyp/target}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/model_pth/Polyp}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${PROJECT_DIR}/pretrained_pth/pvt}"

# 白名单正则只允许安全文件名字符，防止DATASET_NAME把路径拼接到意外目录；失败退出码为1。
[[ "${DATASET_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "[ERROR] invalid DATASET_NAME: ${DATASET_NAME}"
  exit 1
}

# 四个test块分别确认训练和验证图像/掩膜目录存在；任一缺失都以退出码1停止。
test -d "${DATA_ROOT}/${DATASET_NAME}/train/images" || {
  echo "[ERROR] train images not found"
  exit 1
}

test -d "${DATA_ROOT}/${DATASET_NAME}/train/masks" || {
  echo "[ERROR] train masks not found"
  exit 1
}

test -d "${DATA_ROOT}/${DATASET_NAME}/val/images" || {
  echo "[ERROR] val images not found"
  exit 1
}

test -d "${DATA_ROOT}/${DATASET_NAME}/val/masks" || {
  echo "[ERROR] val masks not found"
  exit 1
}

# PVTv2编码器需要与名称对应的本地预训练权重；ResNet等其他编码器跳过此文件检查。
if [[ "${ENCODER}" == pvt_v2_* ]]; then
  test -f "${PRETRAINED_DIR}/${ENCODER}.pth" || {
    echo "[ERROR] pretrained model not found:"
    echo "${PRETRAINED_DIR}/${ENCODER}.pth"
    exit 1
  }
fi

# 创建模型输出根目录，具体运行目录由Python根据dataset_name和run_name继续组织。
mkdir -p "${OUTPUT_DIR}"

# 时间戳和12位随机十六进制共同保证并发运行标识不重复。
TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"

# LOG_FILE编码主要超参数；RUN_ID还承担“日志/PID/输出目录/进程环境”之间的关联键。
LOG_FILE="${LOG_DIR}/train_${DATASET}_${DATASET_NAME}__imgSize${IMG_SIZE}_batchSize${BATCH_SIZE}_lr${BASE_LR}_epo${MAX_EPOCHS}_${TS}.log"
RUN_ID="train_${DATASET}_${DATASET_NAME}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"
PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

# 参数快照追加到日志；终端只保留便于复制的RUN_ID和后续PID/路径信息。
echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATASET_NAME=${DATASET_NAME}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATA_ROOT=${DATA_ROOT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] IMG_SIZE=${IMG_SIZE}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] BATCH_SIZE=${BATCH_SIZE} VAL_BATCH_SIZE=${VAL_BATCH_SIZE}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] MAX_EPOCHS=${MAX_EPOCHS} BASE_LR=${BASE_LR} WEIGHT_DECAY=${WEIGHT_DECAY}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] NUM_WORKERS=${NUM_WORKERS} SEED=${SEED}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"

echo "---------------------------ready to train---------------------------------" | tee -a "${LOG_FILE}" > /dev/null

# 单个多行命令块：nohup忽略挂断，env注入RUN_ID供stop脚本从/proc校验，python -u关闭解释器输出缓冲。
# 参数固定使用[1,3,5]并行尺度、constant调度和0.75/1/1.25多尺度训练；末尾重定向日志、断开stdin并放入后台。
nohup env RUN_ID="${RUN_ID}" "${PYTHON_BIN}" -u train_polyp.py \
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
  --scheduler constant \
  --scale_rates 0.75 1.0 1.25 \
  --num_workers "${NUM_WORKERS}" \
  --n_gpu "${N_GPU}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --validate_every "${VALIDATE_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --max_train_batches "${MAX_TRAIN_BATCHES}" \
  --max_valid_cases "${MAX_VALID_CASES}" \
  --device "${DEVICE}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

# $!取得最近后台进程PID；写入PID文件后，stop_train_polyp.sh会同时校验PID和RUN_ID再停止。
PID=$!

echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] RUN_OUTPUT=${OUTPUT_DIR}/${DATASET_NAME}/${RUN_ID}"
