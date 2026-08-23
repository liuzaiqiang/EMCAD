#!/usr/bin/env bash
# 使用 Bash 解释器；本脚本依赖 pipefail、BASH_SOURCE 和参数展开等 Bash 行为。
# -e 遇到未处理的非零退出码即停止，-u 拒绝未定义变量，pipefail 让管道任一环节失败都算失败。
set -euo pipefail

#conda路径、环境名参数化
# 这里是固定默认值，不读取同名外部环境变量；服务器安装位置或环境名变化时需要相应调整。
CONDA_BASE="/base/mambaforge"
CONDA_ENV_PREFIX="/root/shared-nvme/lzq_conda/envs/sld_emcad"


CONDA_ENV_NAME="sld_emcad"


# BASH_SOURCE[0] 指向当前脚本；进入其目录后取绝对路径，保证从任意工作目录启动都定位到本项目。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 后续相对路径和 PID 文件均以项目根目录为当前工作目录。
cd "${PROJECT_DIR}"

# 统一日志目录；-p 会在目录已存在时保持成功，并递归创建缺失的父目录。
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 加载 conda 的 shell 函数后激活指定环境；任一步失败都会因 set -e 终止启动。
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"


# 只向本进程及其子进程暴露编号0的GPU；PYTHONUNBUFFERED让训练日志尽快写入文件。
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# 以下变量描述本次 Synapse 训练超参数；数值会传给 train_synapse.py 的同名命令行参数。
DATASET="Synapse"
IMG_SIZE=256
BATCH_SIZE=20
MAX_EPOCHS=300
BASE_LR=1e-4
# 当前脚本定义了 SUPERVISION，但下方 Python 命令没有传 --supervision，因此实际使用 Python 脚本默认值。
SUPERVISION="mutation"

# LIST_DIR 使用项目绝对路径；ROOT_PATH/VOLUME_PATH 在已 cd 到项目目录后按相对路径解析。
# train_npz 是二维训练切片，test_vol_h5 是逐体积验证数据。
LIST_DIR="${PROJECT_DIR}/../data/Synapse/lists/lists_Synapse"
ROOT_PATH="../data/Synapse/train_npz"
VOLUME_PATH="../data/Synapse/test_vol_h5"
# DETERMINISTIC=1 请求确定性训练；SEED固定 Python/NumPy/PyTorch 随机序列。
DETERMINISTIC=1
SEED=2222

# 从系统随机源读取6字节并转为12位十六进制，避免同一秒启动多个任务时 RUN_ID 冲突。
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
#TS="$(date +%F_%H%M)"
# 时间戳精确到秒，同时参与日志名和运行标识。
TS="$(date +%F_%H%M%S)"
# LOG_FILE 集中保存训练标准输出和错误；RUN_ID还包含GPU、种子和随机后缀，供停止脚本核验进程身份。
LOG_FILE="${LOG_DIR}/train_${DATASET}__imgSize${IMG_SIZE}_batchSize${BATCH_SIZE}_lr${BASE_LR}_epo${MAX_EPOCHS}_${TS}.log"
RUN_ID="train_${DATASET}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"

# 因为脚本已进入 PROJECT_DIR，这个相对 PID 文件实际写在项目根目录。
PID_FILE="${RUN_ID}.pid"

# tee -a 先把运行参数追加到日志，随后 > /dev/null 抑制大多数参数在终端重复显示。
# RUN_ID 单独再次输出到终端，便于复制给对应 stop 脚本。
echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null 
echo "[INFO] DATASET=${DATASET}"  | tee -a "${LOG_FILE}" > /dev/null
#echo "[INFO] IMG_SIZE=${IMG_SIZE} NUM_CLASSES=${NUM_CLASSES}"  | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] BATCH_SIZE=${BATCH_SIZE} MAX_EPOCHS=${MAX_EPOCHS} BASE_LR=${BASE_LR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] LIST_DIR=${LIST_DIR}"  | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] SEED=${SEED}"  | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"  | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"

echo "---------------------------ready to train---------------------------------" | tee -a "${LOG_FILE}" > /dev/null

# 启动命令块：nohup使进程忽略终端挂断信号；env把RUN_ID写入子进程环境，停止脚本会读取它防止PID复用误杀。
# 各反斜杠续行共同组成一条命令，不能在续行之间插入注释。
# >> 追加标准输出，2>&1把标准错误并入同一日志，< /dev/null断开标准输入，末尾&放入后台。
nohup env RUN_ID="${RUN_ID}" python train_synapse.py \
  --root_path "${ROOT_PATH}" \
  --list_dir "${LIST_DIR}" \
  --volume_path "${VOLUME_PATH}" \
  --dataset "${DATASET}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --base_lr "${BASE_LR}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

# $! 是当前 shell 最近启动的后台进程PID，即 nohup/env/python 进程链最终跟踪的训练进程。
PID=$!
echo "[INFO] PID=${PID}"
# PID文件只保存数字PID；与RUN_ID环境变量双重校验后，停止脚本才会发送终止信号。
echo "${PID}" > "${PID_FILE}"
