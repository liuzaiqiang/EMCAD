#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# EMCAD 多数据集连续训练调度器
#
# 用途：依次调用 sh/start_train_*.sh，并等待每个训练 Python 进程真正退出，
#       然后才启动下一个 seed。各训练启动脚本仍负责 Conda、数据集参数、
#       独立训练日志、PID 文件和模型输出目录；本脚本只负责队列和状态记录。
#
# 用法：
#   bash sh/run_seed_queue.sh synapse 2222 3407 5678
#   DATASET_NAME=Kvasir bash sh/run_seed_queue.sh polyp 2222 3407
#   DATASET_NAME=ISIC2017 bash sh/run_seed_queue.sh isic 2222 3407
#   bash sh/run_seed_queue.sh acdc 2222 3407
#   bash sh/run_seed_queue.sh busi 2222 3407
#
# 可选环境变量：
#   CUDA_DEVICE=0       选择物理 GPU；不设置时由各启动脚本采用自己的默认值。
#   DATASET_NAME=...    选择 Polyp/ISIC 等启动器支持的具体子数据集。
#   CONTINUE_ON_ERROR=1 单个 seed 失败后继续队列；默认失败即停止，避免漏看错误。
# -----------------------------------------------------------------------------

# -e：未处理的错误时停止；-u：拒绝空的未定义变量；pipefail：管道任一环节失败即失败。
set -euo pipefail

# 通过本调度器自身的位置解析项目根目录，因此从项目根目录或其他目录调用都可用。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
QUEUE_LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${QUEUE_LOG_DIR}"

# 至少需要一个启动器名称和一个 seed；在执行任何训练前先验证所有 seed 格式。
if (( $# < 2 )); then
  echo "用法: bash sh/run_seed_queue.sh <synapse|acdc|polyp|busi|isic> <seed1> [seed2 ...]" >&2
  echo "示例: DATASET_NAME=Kvasir bash sh/run_seed_queue.sh polyp 2222 3407 5678" >&2
  exit 2
fi

DATASET_KEY="${1,,}"
shift
case "${DATASET_KEY}" in
  synapse) LAUNCHER="${SCRIPT_DIR}/start_train_synapse.sh" ;;
  acdc)    LAUNCHER="${SCRIPT_DIR}/start_train_acdc.sh" ;;
  polyp)   LAUNCHER="${SCRIPT_DIR}/start_train_polyp.sh" ;;
  busi)    LAUNCHER="${SCRIPT_DIR}/start_train_busi.sh" ;;
  isic)    LAUNCHER="${SCRIPT_DIR}/start_train_isic.sh" ;;
  *)
    echo "[ERROR] 不支持的训练启动器: ${DATASET_KEY}" >&2
    echo "[ERROR] 可选值: synapse, acdc, polyp, busi, isic" >&2
    exit 2
    ;;
esac

# 校验所有 seed 后再开跑，避免队列运行一半才发现后面的参数拼错。
for requested_seed in "$@"; do
  if [[ ! "${requested_seed}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] seed 必须是非负整数，收到: ${requested_seed}" >&2
    exit 2
  fi
done

# 每次队列运行单独写一个摘要日志；模型训练的详细 stdout/stderr 仍在各启动脚本生成的日志文件。
QUEUE_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
QUEUE_LOG="${QUEUE_LOG_DIR}/seed_queue_${DATASET_KEY}_${QUEUE_TIMESTAMP}_$$.log"

# tee 同时把队列状态显示到当前终端并保存到文件，方便 nohup 后用 tail -f 查看。
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "[QUEUE] 启动器=${LAUNCHER}"
echo "[QUEUE] seeds=$*"
if [[ -n "${DATASET_NAME:-}" ]]; then
  echo "[QUEUE] DATASET_NAME=${DATASET_NAME}"
fi
echo "[QUEUE] 队列日志=${QUEUE_LOG}"

# 记录当前训练 PID。若用户向队列进程发送 TERM/INT，转发 TERM 给正在训练的进程，
# 以便停止队列时不会悄悄留下一个无人调度的后台训练；训练脚本的 stop 脚本仍可单独使用。
ACTIVE_TRAIN_PID=""
handle_queue_signal() {
  local signal_name="$1"
  echo "[QUEUE] 收到 ${signal_name}，停止当前训练 PID=${ACTIVE_TRAIN_PID:-none} 并结束队列。"
  if [[ -n "${ACTIVE_TRAIN_PID}" ]] && kill -0 "${ACTIVE_TRAIN_PID}" 2>/dev/null; then
    kill -TERM "${ACTIVE_TRAIN_PID}" 2>/dev/null || true
    wait "${ACTIVE_TRAIN_PID}" 2>/dev/null || true
  fi
  exit 130
}
trap 'handle_queue_signal INT' INT
trap 'handle_queue_signal TERM' TERM

# 默认遇到失败立即停止；显式设 CONTINUE_ON_ERROR=1 时记录失败并继续后续 seed。
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
if [[ "${CONTINUE_ON_ERROR}" != "0" && "${CONTINUE_ON_ERROR}" != "1" ]]; then
  echo "[ERROR] CONTINUE_ON_ERROR 只能为 0 或 1，收到: ${CONTINUE_ON_ERROR}" >&2
  exit 2
fi

FAILED_SEEDS=()
for requested_seed in "$@"; do
  # 由环境变量把 seed 交给现有启动脚本；这些脚本会把 seed 继续传给 Python CLI，
  # 并在训练日志和实验配置中记录实际使用值。export 也让 nohup 子进程继承该值。
  export SEED="${requested_seed}"
  echo "[QUEUE] 开始 dataset=${DATASET_KEY} seed=${SEED} 时间=$(date '+%F %T')"

  # source 而不是启动一个独立 shell：训练启动器设置的 PID 因此仍对应当前 shell
  # 的后台子进程，下面的 wait 能准确取得 Python 训练进程的结束状态。
  # 各启动器使用 BASH_SOURCE 计算自身路径，source 时仍会指向对应的 start_train 文件。
  source "${LAUNCHER}"

  # 每个现有 start_train 脚本都会将 nohup 后台任务 PID 保存到 PID 变量。
  # 检查 PID 格式，防止启动器变更后 wait 意外等待错误对象或跳过等待。
  if [[ ! "${PID:-}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] 启动脚本未返回有效 PID：${LAUNCHER}，seed=${SEED}" >&2
    exit 3
  fi
  ACTIVE_TRAIN_PID="${PID}"
  echo "[QUEUE] 等待训练 PID=${ACTIVE_TRAIN_PID} seed=${SEED}"

  # wait 会阻塞到本次训练真正退出。捕获非零状态后按策略停止或继续；
  # 成功退出后才进入下一轮，因此同一队列不会同时占用 GPU 训练多个 seed。
  if wait "${ACTIVE_TRAIN_PID}"; then
    echo "[QUEUE] 完成 dataset=${DATASET_KEY} seed=${SEED} 时间=$(date '+%F %T')"
  else
    training_status=$?
    FAILED_SEEDS+=("${requested_seed}:${training_status}")
    echo "[QUEUE] 失败 dataset=${DATASET_KEY} seed=${SEED} exit_code=${training_status}"
    ACTIVE_TRAIN_PID=""
    if [[ "${CONTINUE_ON_ERROR}" == "0" ]]; then
      echo "[QUEUE] 默认失败即停止；后续 seed 未启动。"
      exit "${training_status}"
    fi
  fi
  ACTIVE_TRAIN_PID=""
done

# 汇总队列结果：只要某个 seed 失败，最终退出码非零，便于服务器任务系统检测失败。
if (( ${#FAILED_SEEDS[@]} > 0 )); then
  echo "[QUEUE] 队列结束但有失败 seed：${FAILED_SEEDS[*]}"
  exit 1
fi

echo "[QUEUE] 全部 seed 完成 dataset=${DATASET_KEY} 时间=$(date '+%F %T')"
