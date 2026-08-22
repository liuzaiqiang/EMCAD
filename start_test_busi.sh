#!/usr/bin/env bash
# Bash 严格模式：任一未处理错误、未定义变量或失败管道都会终止启动，避免用残缺参数执行评估。
set -euo pipefail

# 取得当前脚本所在的项目根目录并切换过去，使数据、日志和辅助脚本的相对路径不受调用位置影响。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# 测试进程的标准输出和标准错误统一写入项目 logs 目录；-p 允许目录已存在。
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 三个入口均可由同名环境变量覆盖，便于适配不同服务器的 conda 安装位置、环境名和 Python 命令。
CONDA_BASE="${CONDA_BASE:-/base/mambaforge}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sld_emcad}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# conda 初始化脚本存在时才加载并激活指定环境；不存在时沿用调用者当前的 Python 环境。
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

# CUDA_DEVICE 默认选择0号GPU；无缓冲输出使 nohup 日志尽快落盘，方便实时观察评估进度。
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export PYTHONUNBUFFERED=1

# BUSI 是乳腺超声二分类分割任务；DATASET_NAME 被限制为 BUSI，SPLIT 可选 val 或 test。
DATASET="BUSI"
DATASET_NAME="${DATASET_NAME:-BUSI}"
SPLIT="${SPLIT:-test}"

# 推理批量、数据加载进程、病例上限、确定性开关、随机种子和运行设备均支持启动前覆盖。
# MAX_CASES=0 表示不人为截断病例数，DEVICE=auto 交由 Python 入口选择可用设备。
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_CASES="${MAX_CASES:-0}"
DETERMINISTIC="${DETERMINISTIC:-1}"
SEED="${SEED:-2222}"
DEVICE="${DEVICE:-auto}"

# DATA_ROOT 默认指向预处理后的 BUSI target 目录；CKPT 不设默认权重，必须由调用者明确提供。
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../data/busi/target}"
CKPT="${CKPT:-}"

# 拒绝其他数据集名，防止目录结构或类别定义与 test_busi.py 的 BUSI 评估逻辑不一致。
if [[ "${DATASET_NAME}" != "BUSI" ]]; then
  echo "[ERROR] DATASET_NAME must be BUSI"
  exit 1
fi

# 只接受验证集或测试集；case 的 *) 分支负责捕获拼写错误和未支持的划分名。
case "${SPLIT}" in
  val|test)
    ;;
  *)
    echo "[ERROR] SPLIT must be val or test"
    exit 1
    ;;
esac

# 检查点变量为空或文件不存在时打印示例并以状态码1退出，避免后台任务启动后才失败。
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[ERROR] CKPT is missing or does not exist:"
  echo "${CKPT}"
  echo "[ERROR] Example:"
  echo 'CKPT="/absolute/path/to/best.pth" bash start_test_busi.sh'
  exit 1
fi

# 逐项检查图像、掩膜、样本清单和划分摘要；任一路径缺失都会立即指出具体缺项。
# 反斜杠连接的是同一个 for 头部，四个路径依次赋给 REQUIRED。
for REQUIRED in \
  "${DATA_ROOT}/BUSI/${SPLIT}/images" \
  "${DATA_ROOT}/BUSI/${SPLIT}/masks" \
  "${DATA_ROOT}/BUSI/manifest.csv" \
  "${DATA_ROOT}/BUSI/split_summary.json"
do
  if [[ ! -e "${REQUIRED}" ]]; then
    echo "[ERROR] required BUSI path not found:"
    echo "${REQUIRED}"
    exit 1
  fi
done

# 将检查点目录解析成绝对路径，再由它定位训练时一同保存的 config.json。
# test_busi.py 依据该配置重建与权重匹配的模型结构，而不是在本启动器里重复填写编码器和解码器参数。
CKPT_DIR="$(cd "$(dirname "${CKPT}")" && pwd)"
CKPT="${CKPT_DIR}/$(basename "${CKPT}")"
CONFIG_FILE="${CKPT_DIR}/config.json"

# 权重和结构配置必须同目录存在；缺少 config.json 时无法可靠确认权重对应的网络配置。
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[ERROR] checkpoint config not found:"
  echo "${CONFIG_FILE}"
  echo "[ERROR] best.pth and config.json must be in the same directory"
  exit 1
fi

# 默认把预测、概率图和逐病例指标放在检查点目录旁；两个路径都可由环境变量单独覆盖。
TEST_SAVE_DIR="${TEST_SAVE_DIR:-${CKPT_DIR}/${SPLIT}_BUSI_outputs}"
OUTPUT_CSV="${OUTPUT_CSV:-${TEST_SAVE_DIR}/${SPLIT}_metrics.csv}"

# 秒级时间戳配合6字节随机数生成唯一后缀，降低同一秒并发评估时文件重名的概率。
TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"

# RUN_ID 把任务类型、数据集、划分、时间、GPU、种子和随机后缀串联起来，并关联日志与PID文件。
RUN_ID="test_${DATASET}_${SPLIT}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

# tee -a 将本次评估的关键路径和参数追加到日志；重定向到 /dev/null 避免每项在终端重复显示。
# MODEL_CONFIG 这一记录强调网络结构来自检查点旁的 config.json；RUN_ID 单独打印供停止脚本使用。
echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATA_ROOT=${DATA_ROOT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] SPLIT=${SPLIT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] CKPT=${CKPT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] CONFIG_FILE=${CONFIG_FILE}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] TEST_SAVE_DIR=${TEST_SAVE_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] MODEL_CONFIG=loaded_from_checkpoint_config" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] MAX_CASES=${MAX_CASES}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}" | tee -a "${LOG_FILE}" > /dev/null

echo "[INFO] RUN_ID=${RUN_ID}"
echo "---------------------------ready to test----------------------------------" | tee -a "${LOG_FILE}" > /dev/null

# 整个反斜杠续行块是一条后台评估命令：nohup 允许终端断开后继续运行，env 注入 RUN_ID 供停止时核验身份。
# test_busi.py 从 checkpoint/config.json 恢复模型；其余参数控制数据划分、输出位置、加载并行度和可复现性。
# --save_probabilities 要求同时保存像素概率；stdout 追加日志，stderr 合并，stdin 断开，末尾 & 放入后台。
nohup env RUN_ID="${RUN_ID}" "${PYTHON_BIN}" -u test_busi.py \
  --checkpoint "${CKPT}" \
  --data_root "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --output_dir "${TEST_SAVE_DIR}" \
  --output_csv "${OUTPUT_CSV}" \
  --inference_batch_size "${INFERENCE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --max_cases "${MAX_CASES}" \
  --device "${DEVICE}" \
  --save_probabilities \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

# $! 是当前 shell 最近启动的后台任务PID，必须在启动命令后立即读取。
PID=$!

# PID 文件建立 RUN_ID 到操作系统进程号的映射，供 stop_test_busi.sh 精确定位并二次核验目标进程。
echo "${PID}" > "${PID_FILE}"

# 向终端报告监控、停止和查找结果所需的位置；预测掩膜与概率图分别进入对应子目录。
echo "[INFO] PID=${PID}"
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV}"
echo "[INFO] PREDICTIONS=${TEST_SAVE_DIR}/predictions"
echo "[INFO] PROBABILITIES=${TEST_SAVE_DIR}/probabilities"
