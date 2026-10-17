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


cd "${PROJECT_DIR}"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 默认使用 GPU 0；连续队列也可通过 CUDA_DEVICE 覆盖，供启动器之间统一选卡。
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
# 单卡队列默认 n_gpu=1；确定性模式沿用 train_synapse.py 原来的默认值 1。
N_GPU="${N_GPU:-1}"
#windows环境下运行时，设置为0（0 表示由主进程加载数据，最稳定）;linux环境下运行时，设置为8。
NUM_WORKERS=8

SEED="${SEED:-2222}"
MAX_EPOCHS=400
DATASET="Synapse"
IMG_SIZE=224
BATCH_SIZE=16
SUPERVISION="mutation"
BASE_LR=1e-4

# 编码器是特征提取网络的结构名称；默认 PVTv2-B2 与 train_synapse.py 原有默认值一致。
# 可在启动队列前写 ENCODER=pvt_v2_b0 或 ENCODER=resnet34 来切换整个队列的结构。
# 仅填写 lib/networks.py 实现的名称，避免该文件对未知名称静默回退到 B2。
ENCODER="${ENCODER:-pvt_v2_b2}"
# networks.py 对未知名称会回退到 B2，却保留原名称；启动前拦截拼写错误，保证目录名与实际结构一致。
case "${ENCODER}" in
  pvt_v2_b0|pvt_v2_b1|pvt_v2_b2|pvt_v2_b3|pvt_v2_b4|pvt_v2_b5|resnet18|resnet34|resnet50|resnet101|resnet152) ;;
  *) echo "[ERROR] Unsupported ENCODER: ${ENCODER}" >&2; exit 1 ;;
esac
# 1 表示加载编码器预训练权重，0 表示随机初始化；结构选择与权重选择是两件事。
# PVTv2 权重从 PRETRAINED_DIR/<ENCODER>.pth 读取；ResNet 由其实现使用 PyTorch 缓存/下载。
USE_PRETRAIN="${USE_PRETRAIN:-1}"
PRETRAINED_DIR="${PRETRAINED_DIR:-./pretrained_pth/pvt/}"
# EMCAD 解码器的主要结构参数；这里显式传给 Python，避免以后 Python 默认值改动造成实验漂移。
EXPANSION_FACTOR="${EXPANSION_FACTOR:-2}"
#MSDC 的三个深度卷积核；数组展开后对应 --kernel_sizes 1 3 5。
KERNEL_SIZES=(1 3 5)
# 普通字符串供参数清单打印完整数组；Bash 对数组执行 ${!name} 时只会打印首项。
KERNEL_SIZES_DISPLAY="${KERNEL_SIZES[*]}"

LGAG_KS="${LGAG_KS:-3}"

ACTIVATION_MSCB="${ACTIVATION_MSCB:-relu6}"


DETERMINISTIC="${DETERMINISTIC:-1}"
# 0 为默认的并行深度卷积与多尺度相加；1 分别启用串行深度卷积/通道拼接。
NO_DW_PARALLEL="${NO_DW_PARALLEL:-0}"

CONCATENATION="${CONCATENATION:-0}"

# Python的布尔开关必须按是否出现来传递，不能传 --no_pretrain 0 这种字符串。
ARCH_FLAGS=()
case "${USE_PRETRAIN}" in
  0) ARCH_FLAGS+=(--no_pretrain) ;;
  1) ;;
  *) echo "[ERROR] USE_PRETRAIN must be 0 or 1" >&2; exit 1 ;;
esac
case "${NO_DW_PARALLEL}" in
  0) ;;
  1) ARCH_FLAGS+=(--no_dw_parallel) ;;
  *) echo "[ERROR] NO_DW_PARALLEL must be 0 or 1" >&2; exit 1 ;;
esac
case "${CONCATENATION}" in
  0) ;;
  1) ARCH_FLAGS+=(--concatenation) ;;
  *) echo "[ERROR] CONCATENATION must be 0 or 1" >&2; exit 1 ;;
esac


RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
TS="$(date +%F_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${DATASET}_encoder_${ENCODER}_imgSize_${IMG_SIZE}_supervision_${SUPERVISION}_bs_${BATCH_SIZE}_seed_${SEED}_lr_${BASE_LR}_maxepo_${MAX_EPOCHS}_ts_${TS}_RAND_${RAND}.log"
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
  ENCODER
  USE_PRETRAIN
  PRETRAINED_DIR
  EXPANSION_FACTOR
  KERNEL_SIZES_DISPLAY
  LGAG_KS
  ACTIVATION_MSCB
  N_GPU
  DETERMINISTIC
  NO_DW_PARALLEL
  CONCATENATION
  TS
  RAND
  LOG_FILE
  RUN_ID
  PID_FILE
  NUM_WORKERS
)

{
  echo "[INFO] parameters:"
  for name in "${PARAM_NAMES[@]}"; do
    printf '[INFO] %-24s=%s\n' "$name" "${!name}"
  done
  echo "---------------------------ready to run---------------------------------"
} | tee -a "${LOG_FILE}"


# 启动命令块：nohup使进程忽略终端挂断信号；env把RUN_ID写入子进程环境，停止脚本会读取它防止PID复用误杀。  >> 追加标准输出，2>&1把标准错误并入同一日志，< /dev/null断开标准输入
# 参数列表和日志重定向必须属于同一条 shell 命令；最后一个参数行用反斜杠续接重定向行，
# 这样 $! 才会记录真正运行 Python 训练的后台进程，且 stdout/stderr 会进入本次训练日志。
nohup env RUN_ID="${RUN_ID}" python -u train_synapse.py \
  --dataset "${DATASET}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --base_lr "${BASE_LR}" \
  --seed "${SEED}" \
  --encoder "${ENCODER}" \
  --pretrained_dir "${PRETRAINED_DIR}" \
  --expansion_factor "${EXPANSION_FACTOR}" \
  --kernel_sizes "${KERNEL_SIZES[@]}" \
  --lgag_ks "${LGAG_KS}" \
  --activation_mscb "${ACTIVATION_MSCB}" \
  "${ARCH_FLAGS[@]}" \
  --supervision "${SUPERVISION}" \
  --num_workers "${NUM_WORKERS}" \
  --n_gpu "${N_GPU}" \
  --deterministic "${DETERMINISTIC}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &


PID=$!
echo "[INFO] PID=${PID}"
#PID 文件在项目根目录
echo "${PID}" > "${PID_FILE}"
