#!/usr/bin/env bash
# Bash 严格模式：遇到失败命令、未定义变量或失败管道即退出，避免错误配置被带入后台测试。
set -euo pipefail

# 定位脚本所在项目根并切换过去，使相对PID文件以及后续路径具有固定基准。
#PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


cd "${PROJECT_DIR}"

# 创建集中保存评估日志的目录；目录已存在时 mkdir -p 仍返回成功。
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Conda 路径和 Python 可执行文件可由服务器环境覆盖；没有 Conda 时使用当前 Python。
CONDA_BASE="${CONDA_BASE:-/home/mlf/anaconda3}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${CONDA_BASE}/envs/sld_emcad}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PREFIX}"
fi

# 默认暴露0号GPU；关闭Python输出缓冲，让nohup日志及时写出。
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export PYTHONUNBUFFERED=1

# ACDC 是4类心脏MRI分割；模型结构与输入尺寸从 checkpoint 同目录的 config.json 恢复。
# INFERENCE_BATCH_SIZE是切片推理批量，Z_SPACING用于三维距离类指标的体素间距换算，MAX_CASES=0表示全部病例。
DATASET="ACDC"
NUM_WORKERS="${NUM_WORKERS:-0}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
Z_SPACING="${Z_SPACING:-10.0}"
MAX_CASES="${MAX_CASES:-0}"

# 病例列表和数据根目录采用项目旁的固定ACDC布局；CKPT必须由调用者明确指定。
LIST_DIR="${LIST_DIR:-${PROJECT_DIR}/../data/ACDC/lists_ACDC}"
ROOT_PATH="${PROJECT_DIR}/../data/ACDC"
# 默认在 ACDC 对应的训练输出目录中查找 best.pth；CKPT 非空时优先使用调用者明确指定的路径。
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
CKPT="${CKPT_DIR}/$(basename "${CKPT}")"
CONFIG_FILE="${CKPT_DIR}/config.json"
test -f "${CONFIG_FILE}" || { echo "[ERROR] checkpoint config not found: ${CONFIG_FILE}"; exit 1; }
TEST_SAVE_DIR="${CKPT_DIR}/predictions"
OUTPUT_CSV="${CKPT_DIR}/test_metrics.csv"

# 时间戳和随机十六进制后缀共同用于区分并发测试；RAND来自系统随机源的6字节数据。
TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
# 日志名记录数据集与唯一运行标识；模型参数由Python从CONFIG_FILE读取。
LOG_FILE="${LOG_DIR}/test_${DATASET}_${TS}_${RAND}.log"
RUN_ID="test_${DATASET}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_RAND${RAND}"
PID_FILE="${RUN_ID}.pid"

# 将测试专属数据路径、推理批量和运行标识写入日志；模型配置由配置文件统一记录。
PARAM_NAMES=(CONDA_BASE CONDA_ENV_PREFIX PROJECT_DIR LOG_DIR CUDA_VISIBLE_DEVICES DATASET NUM_WORKERS INFERENCE_BATCH_SIZE Z_SPACING MAX_CASES LIST_DIR ROOT_PATH CKPT CKPT_DIR CONFIG_FILE TEST_SAVE_DIR OUTPUT_CSV TS RAND LOG_FILE RUN_ID PID_FILE)
echo "[INFO] RUN_ID=${RUN_ID}"

{
  echo "[INFO] parameters:"
  for name in "${PARAM_NAMES[@]}"; do
    printf '[INFO] %-24s=%s\n' "$name" "${!name}"
  done
  echo "---------------------------ready to test----------------------------------"
} | tee -a "${LOG_FILE}"

# 整个续行块是一条后台命令；nohup抵抗终端断开，env把RUN_ID写入子进程环境供停止脚本核验。
# 模型结构、融合/CAA选项和输入尺寸均由 test_acdc.py 从 CONFIG_FILE 自动恢复，启动器不再重复填写。
# --save_nii和--save_npz分别请求保存医学影像格式预测与数组结果；输出追加日志、错误合并、输入断开并转入后台。
nohup env RUN_ID="${RUN_ID}" "${PYTHON_BIN}" -u test_acdc.py \
  --checkpoint "${CKPT}" \
  --root_path "${ROOT_PATH}" \
  --list_dir "${LIST_DIR}" \
  --output_dir "${TEST_SAVE_DIR}" \
  --output_csv "${OUTPUT_CSV}" \
  --inference_batch_size "${INFERENCE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --z_spacing "${Z_SPACING}" \
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
