#!/usr/bin/env bash
# 使用Bash严格模式；任一未处理错误、未定义变量或失败管道都会终止启动，避免后台任务带错参数运行。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"



# CONDA_BASE="/base/mambaforge"
# CONDA_ENV_PREFIX="/root/shared-nvme/lzq_conda/envs/sld_emcad"
CONDA_BASE="/home/mlf/anaconda3"
CONDA_ENV_PREFIX="/home/mlf/anaconda3/envs/sld_emcad"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate "${CONDA_ENV_PREFIX}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# CUDA_DEVICE默认0；PYTHONUNBUFFERED确保nohup日志尽快落盘。
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"

export PYTHONUNBUFFERED=1

# DATASET用于日志分类，DATASET_NAME选择target目录下的具体息肉数据集，默认ClinicDB。
DATASET="${DATASET:-Polyp}"
DATASET_NAME="${DATASET_NAME:-ClinicDB}"
# DATASET同时决定默认输出父目录，只允许用作单层目录名的字符。
[[ "${DATASET}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "[ERROR] invalid DATASET output label: ${DATASET}"
  exit 1
}

# 训练尺寸、训练/验证批量、epoch、AdamW学习率/权重衰减和梯度值裁剪均可由环境变量覆盖。
IMG_SIZE="${IMG_SIZE:-352}"
BATCH_SIZE="${BATCH_SIZE:-16}"


VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
BASE_LR="${BASE_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
#在训练脚本里通常指**梯度裁剪阈值 gradient clipping 它通常用于限制反向传播时的梯度大小，防止梯度爆炸。CLIP=0.5 表示：把模型所有参数梯度的总范数限制在不超过 `0.5`。
#梯度裁剪主要用于：
# - 防止梯度爆炸；
# - 稳定训练过程；
# - 减少loss突然飙升；
# - 在 RNN、Transformer、医学图像分割等容易出现梯度波动的任务中常用。
# 在医学图像分割里，比如多数据集训练、难样本较多、loss 权重较大时，CLIP=0.5可以避免某一批难样本把参数更新幅度拉得过大。
#总计：CLIP=0.5 通常决定的是：反向传播后，模型梯度允许的最大更新幅度上限。它主要用来稳定训练、防止梯度爆炸；值越小更新越保守，值越大更新越激进。
CLIP="${CLIP:-0.5}"

# 运行与调试控制：0通常表示“不限制”批次数/验证病例数；DEVICE=auto由Python选择CUDA或CPU。
NUM_WORKERS="${NUM_WORKERS:-0}"

DETERMINISTIC="${DETERMINISTIC:-1}"
SEED="${SEED:-2222}"
#每隔多少个 epoch 做一次验证。1 表示每个 epoch 都验证，并用于更新 best.pth
VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-0}"
#每个 epoch 最多训练多少个 batch。0 表示不限制；大于 0 通常用于快速调试或冒烟测试。
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
#每次验证最多评估多少个病例。0 表示使用完整验证集；大于 0 会截断验证集，不适合正式实验。
MAX_VALID_CASES="${MAX_VALID_CASES:-0}"
#运算设备。auto 时由 Python 自动选择 CUDA 或 CPU；也可以写 cuda、cuda:0 或 cpu。属于硬件运行选项，论文可能报告 GPU 型号，但不会把 DEVICE=auto 当作模型超参数。
#DEVICE="${DEVICE:-auto}"
DEVICE="${DEVICE:-cuda}"

# EMCAD结构参数：编码器、MSCB扩张率、LGAG核、激活和二分类监督策略。
# supervision=paper对应四个单头损失加四头logits求和后的第五项损失。
ENCODER="${ENCODER:-pvt_v2_b2}"
EXPANSION_FACTOR="${EXPANSION_FACTOR:-2}"
LGAG_KS="${LGAG_KS:-3}"
ACTIVATION_MSCB="${ACTIVATION_MSCB:-relu6}"

#二分类任务，深监督方式是paper
SUPERVISION="${SUPERVISION:-paper}"


#####################################################################


# 独立消融开关；关闭时分别映射到 p1 与 EUCB 原始上采样路径。
#像素级可靠性多头融合的总开关。 1：允许使用像素可靠性融合  0：关闭该创新点，强制回到 p1 输出
USE_PIXEL_RELIABILITY_FUSION="${USE_PIXEL_RELIABILITY_FUSION:-1}"
#它决定四个 EMCAD 输出如何融合。当前代码支持四种模式：
#p1:只使用 p1，即 outputs[-1] 这是原始单输出推理方式，也是关闭像素可靠性融合时的模式。
# fixed_sum  outputs[0] + outputs[1] + outputs[2] + outputs[3]  四个输出直接相加，权重固定为 1。
#global_scalar   为四个输出学习四个全局标量权重： softmax(w1, w2, w3, w4) 每个输出在整张图上使用同一个权重。
#pixel_reliability  为每个像素、每个输出预测可靠性： 每个像素位置分别计算四个输出的权重  也就是说，模型不是简单平均四个输出，而是针对每个像素决定更信任哪个预测头。
FUSION_MODE="${FUSION_MODE:-pixel_reliability}"



#这是可靠性预测头校准损失的权重。 它只影响 pixel_reliability 模式中的可靠性监督部分，不影响原始 EMCAD 主损失。
#当前逻辑可以简化为：
# fusion_auxiliary_loss =
#     0.3 × fused_BCE
#   + 0.7 × fused_Dice
#   + RELIABILITY_LOSS_WEIGHT × reliability_BCE

#RELIABILITY_LOSS_WEIGHT=1  表示按当前默认强度加入可靠性校准损失。
# 注意它只有在以下条件同时满足时才有效：
# FUSION_MODE=pixel_reliability
# FUSION_LOSS_WEIGHT>0
RELIABILITY_LOSS_WEIGHT="${RELIABILITY_LOSS_WEIGHT:-1}"

#####################################################################




USE_CONTENT_AWARE_ANTIALIAS="${USE_CONTENT_AWARE_ANTIALIAS:-1}"




#融合辅助损失的总权重。
# 0	不计算融合辅助损失
# 1	按原始比例加入融合辅助损失
# 0.5	融合辅助损失影响减半
# 2	融合辅助损失影响加倍
FUSION_LOSS_WEIGHT="${FUSION_LOSS_WEIGHT:-1}"


#它决定 EUCB 使用哪种上采样方式。
# 值	         实际含义
# off	          原始 EMCAD EUCB 上采样
# aa_only	        只使用抗混叠上采样
# content_only	只使用内容感知残差，不使用抗混叠基底
# caa	抗混叠基底 + 内容感知残差
CAA_MODE="${CAA_MODE:-caa}"



#CAA 内容感知残差的初始缩放系数。
# CAA 的核心计算可以简化为：output = base + scale * gate * residual
#其中：
# - base：抗混叠或最近邻上采样结果；
# - gate：内容感知门控；
# - residual：深度卷积残差；
# - scale：残差强度。
# 当前初始值为：
# CAA_RESIDUAL_SCALE=0.1
# 这表示模型开始训练时，内容感知残差以较小强度加入基础上采样路径，避免新模块一开始完全压过原始上采样结果。
#0.1 是初始值，不是训练期间永远固定为 0.1。
CAA_RESIDUAL_SCALE="${CAA_RESIDUAL_SCALE:-0.1}"


#这是多尺度训练总开关。
USE_MULTI_SCALE_TRAINING="${USE_MULTI_SCALE_TRAINING:-1}"



#####################################################################

INPUT_CHANNELS="${INPUT_CHANNELS:-3}"
MERGE_INSTANCE_MASKS="${MERGE_INSTANCE_MASKS:-0}"

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
case "${USE_MULTI_SCALE_TRAINING}" in
  0|1) ;;
  *) echo "[ERROR] USE_MULTI_SCALE_TRAINING must be 0 or 1"; exit 1 ;;
esac
case "${INPUT_CHANNELS}" in
  1|3) ;;
  *) echo "[ERROR] INPUT_CHANNELS must be 1 or 3"; exit 1 ;;
esac
case "${MERGE_INSTANCE_MASKS}" in
  0|1) ;;
  *) echo "[ERROR] MERGE_INSTANCE_MASKS must be 0 or 1"; exit 1 ;;
esac
MULTI_SCALE_ARGS=()
if [[ "${USE_MULTI_SCALE_TRAINING}" == "0" ]]; then MULTI_SCALE_ARGS+=(--no_multi_scale); fi
IMAGE_MODE_ARGS=()
if [[ "${INPUT_CHANNELS}" == "1" ]]; then IMAGE_MODE_ARGS+=(--grayscale); fi

# 数据根目录应包含 <dataset>/{train,val}/{images,masks}；默认输出按 DATASET 隔离并可外部覆盖。
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../data/polyp/target}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/model_pth/${DATASET}}"
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
PARAM_NAMES=(PROJECT_DIR DATASET_NAME DATA_ROOT OUTPUT_DIR IMG_SIZE BATCH_SIZE VAL_BATCH_SIZE MAX_EPOCHS BASE_LR WEIGHT_DECAY NUM_WORKERS SEED FUSION_MODE FUSION_LOSS_WEIGHT RELIABILITY_LOSS_WEIGHT CAA_MODE CAA_RESIDUAL_SCALE USE_MULTI_SCALE_TRAINING INPUT_CHANNELS MERGE_INSTANCE_MASKS RUN_ID)
{
  echo "[INFO] parameters:"
  for name in "${PARAM_NAMES[@]}"; do printf '[INFO] %-24s=%s\n' "$name" "${!name}"; done
  echo "---------------------------ready to train---------------------------------"
} | tee -a "${LOG_FILE}"


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
  --fusion_mode "${FUSION_MODE}" \
  --fusion_loss_weight "${FUSION_LOSS_WEIGHT}" \
  --reliability_loss_weight "${RELIABILITY_LOSS_WEIGHT}" \
  --caa_mode "${CAA_MODE}" \
  --caa_residual_scale "${CAA_RESIDUAL_SCALE}" \
  --merge_instance_masks "${MERGE_INSTANCE_MASKS}" \
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
  "${MULTI_SCALE_ARGS[@]}" \
  "${IMAGE_MODE_ARGS[@]}" \
  --num_workers "${NUM_WORKERS}" \
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
