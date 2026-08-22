#!/usr/bin/env bash
# Bash严格模式：测试准备阶段出现失败、未定义变量或失败管道时立即退出。
set -euo pipefail

# 将工作目录固定为脚本所在项目根，后续相对数据路径和PID文件不受调用位置影响。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"


# 创建测试日志目录；所有后台stdout/stderr会追加到该目录中的本次日志。
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"


# Synapse测试脚本使用固定conda安装位置和环境名，不读取外部覆盖值。
CONDA_BASE="/base/mambaforge"
CONDA_ENV_NAME="sld_emcad"


# 加载conda shell函数并激活环境；失败会因set -e终止。
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

# 只暴露GPU0并关闭Python输出缓冲，便于实时查看测试日志。
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1


# 测试输入尺寸、数据集名、体数据目录和病例列表目录。
IMG_SIZE=256
DATASET="Synapse"
VOLUME_PATH="../data/Synapse/test_vol_h5"
LIST_PATH="../data/Synapse/lists/lists_Synapse"

#本项目ckpt无需作为启动参数，项目中写死了位置，best.pth文件必须放在项目的根目录下
#CKPT="${PROJECT_DIR}/model_pth/SimMPNetSynapse....."

# 种子进入RUN_ID用于区分记录；时间戳和随机后缀避免并发测试重名。
SEED=2222
#TS="$(date +%F_%H%M)"	
TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
# 日志名记录数据集和输入尺寸。
LOG_FILE="${LOG_DIR}/test_${DATASET}__img${IMG_SIZE}_${TS}.log"

# RUN_ID会注入Python进程环境，PID文件位于已cd到的项目根目录。
RUN_ID="test_${DATASET}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"
PID_FILE="${RUN_ID}.pid"
# tee -a把关键路径和运行标识写入日志，RUN_ID另行输出到终端供停止脚本使用。
echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] VOLUME_PATH=${VOLUME_PATH}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"  | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"

# 硬性检查：路径不存在就直接退出（比跑半天才发现强太多）  ||：逻辑或，含义是：如果左边失败，就执行右边
#test -f "${CKPT}" || { echo "[ERROR] CKPT not found: ${CKPT}" | tee -a "${LOG_FILE}"; exit 1; }
# VOLUME_PATH不存在时右侧错误块执行并返回退出码1，不会启动后台测试。
test -d "${VOLUME_PATH}" || { echo "[ERROR] VOLUME_PATH not found: ${VOLUME_PATH}" | tee -a "${LOG_FILE}"; exit 1; }

# 整个反斜杠块是一条测试命令：nohup抵抗终端断开，env写入RUN_ID供停止时核验进程身份。
# stdout追加日志且stderr合并；末尾&转入后台。此脚本未显式写< /dev/null，stdin处理由nohup实现决定。
nohup env RUN_ID="${RUN_ID}"   python test_synapse.py \
  --volume_path "${VOLUME_PATH}" \
  --dataset "${DATASET}" \
  --img_size "${IMG_SIZE}" \
  --list_dir "${LIST_PATH}" \
  >> "${LOG_FILE}" 2>&1 &

# $!取得最近后台任务PID并写入与RUN_ID同名文件；成功启动后脚本本身随即结束。
PID=$!
echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
# 至此启动器已把RUN_ID与后台PID建立映射，后续停止操作应使用终端打印的同一个RUN_ID。

# 脚本在记录PID后结束；后台test_synapse.py继续运行并把输出追加到本次日志。
