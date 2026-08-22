#!/usr/bin/env bash
# Bash 严格模式：遇到失败命令、未定义变量或失败管道即退出，避免错误配置被带入后台测试。
set -euo pipefail

# 定位脚本所在项目根并切换过去，使相对PID文件以及后续路径具有固定基准。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# 创建集中保存评估日志的目录；目录已存在时 mkdir -p 仍返回成功。
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# conda根目录、环境名和Python入口均允许用环境变量覆盖服务器默认值。
CONDA_BASE="${CONDA_BASE:-/base/mambaforge}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sld_emcad}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# conda初始化文件存在时才激活环境，否则继续使用调用者已经准备好的Python环境。
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

# 默认暴露0号GPU；关闭Python输出缓冲，让nohup日志及时写出。
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export PYTHONUNBUFFERED=1

# ACDC 是4类心脏MRI分割；IMG_SIZE控制逐切片网络输入尺寸。
# INFERENCE_BATCH_SIZE是切片推理批量，Z_SPACING用于三维距离类指标的体素间距换算，MAX_CASES=0表示全部病例。
DATASET="ACDC"
IMG_SIZE="${IMG_SIZE:-224}"
NUM_WORKERS="${NUM_WORKERS:-0}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
Z_SPACING="${Z_SPACING:-10.0}"
MAX_CASES="${MAX_CASES:-0}"
SEED="${SEED:-2222}"

# 病例列表和数据根目录采用项目旁的固定ACDC布局；CKPT必须由调用者明确指定。
LIST_DIR="${PROJECT_DIR}/../data/ACDC/lists/lists_ACDC"
ROOT_PATH="${PROJECT_DIR}/../data/ACDC"
CKPT="${CKPT:-}"

# 先验证权重文件，再验证测试体数据目录和病例清单；任何缺项都以状态码1终止启动。
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[ERROR] CKPT is missing or does not exist: ${CKPT}"
  echo "[ERROR] Example: CKPT=/absolute/path/to/best.pth bash start_test_acdc.sh"
  exit 1
fi
test -d "${ROOT_PATH}/test" || { echo "[ERROR] test directory not found: ${ROOT_PATH}/test"; exit 1; }
test -f "${LIST_DIR}/test.txt" || { echo "[ERROR] test list not found: ${LIST_DIR}/test.txt"; exit 1; }

# 由检查点目录派生NIfTI/NPZ预测输出目录和逐病例指标CSV，确保结果与对应权重放在一起。
CKPT_DIR="$(cd "$(dirname "${CKPT}")" && pwd)"
TEST_SAVE_DIR="${CKPT_DIR}/predictions"
OUTPUT_CSV="${CKPT_DIR}/test_metrics.csv"

# 时间戳和随机十六进制后缀共同用于区分并发测试；RAND来自系统随机源的6字节数据。
TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
# 日志名记录数据集与输入尺寸；RUN_ID额外记录GPU、种子和随机后缀，PID文件位于已cd到的项目根目录。
LOG_FILE="${LOG_DIR}/test_${DATASET}__img${IMG_SIZE}_${TS}.log"
RUN_ID="test_${DATASET}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"
PID_FILE="${RUN_ID}.pid"

# 将数据、权重、模型输入和运行标识追加到日志；RUN_ID另行打印到终端，供stop_test_acdc.sh使用。
echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATASET=${DATASET}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] ROOT_PATH=${ROOT_PATH}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] LIST_DIR=${LIST_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] VOLUME_PATH=${ROOT_PATH}/test" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] CKPT=${CKPT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] IMG_SIZE=${IMG_SIZE} NUM_CLASSES=4" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] INFERENCE_BATCH_SIZE=${INFERENCE_BATCH_SIZE} MAX_CASES=${MAX_CASES}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"

echo "---------------------------ready to test----------------------------------" | tee -a "${LOG_FILE}" > /dev/null

# 整个续行块是一条后台命令；nohup抵抗终端断开，env把RUN_ID写入子进程环境供停止脚本核验。
# 模型结构参数与EMCAD设计对应：PVTv2-B2编码器提取四尺度特征，1/3/5多尺度深度卷积核和扩展因子2配置MSCB，
# lgag_ks=3配置LGAG门控卷积核，relu6是MSCB激活。它们必须与训练该CKPT时的结构一致，否则权重形状或语义会不匹配。
# 这些模块关系见项目所附EMCAD论文的方法与整体架构部分；Shell仅传参，真正组网发生在test_ACDC.py及lib网络模块中。
# --save_nii和--save_npz分别请求保存医学影像格式预测与数组结果；输出追加日志、错误合并、输入断开并转入后台。
nohup env RUN_ID="${RUN_ID}" "${PYTHON_BIN}" -u test_ACDC.py \
  --checkpoint "${CKPT}" \
  --root_path "${ROOT_PATH}" \
  --list_dir "${LIST_DIR}" \
  --output_dir "${TEST_SAVE_DIR}" \
  --output_csv "${OUTPUT_CSV}" \
  --encoder pvt_v2_b2 \
  --kernel_sizes 1 3 5 \
  --expansion_factor 2 \
  --lgag_ks 3 \
  --activation_mscb relu6 \
  --img_size "${IMG_SIZE}" \
  --inference_batch_size "${INFERENCE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --z_spacing "${Z_SPACING}" \
  --seed "${SEED}" \
  --max_cases "${MAX_CASES}" \
  --device auto \
  --save_nii \
  --save_npz \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

# $!取得刚启动的后台测试进程PID；先报告，再写入与RUN_ID同名的PID文件。
PID=$!
echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
# 输出停止任务、查看日志和定位指标所需的文件路径。
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV}"
