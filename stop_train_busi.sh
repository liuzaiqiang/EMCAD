#!/usr/bin/env bash
# 严格模式要求命令、变量和管道状态都被正确处理，避免停止脚本带着异常输入继续运行。
set -euo pipefail

# 定位并进入项目根目录；绝对PID_FILE路径进一步保证从其他目录调用时仍指向本项目。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# 待停止任务的RUN_ID优先取第一个参数，未传时回退到调用环境中的RUN_ID变量。
RUN_ID="${1:-${RUN_ID:-}}"

# 两种来源都为空时打印用法并以1退出；此时不会尝试猜测进程。
if [[ -z "${RUN_ID}" ]]; then
  echo "[ERROR] usage: bash stop_train_busi.sh <RUN_ID>"
  exit 1
fi

# BUSI训练启动器把后台PID写到项目根的“RUN_ID.pid”，这里建立同一绝对文件路径。
PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

# PID文件缺失以2退出，表示无法从指定RUN_ID取得可信的目标进程号。
if [[ ! -f "${PID_FILE}" ]]; then
  echo "[ERROR] PID file not found: ${PID_FILE}"
  exit 2
fi

# 读取PID文件并去掉空格、换行，得到规范化的候选PID。
PID="$(tr -d ' \n' < "${PID_FILE}")"

# Bash正则要求PID完全由数字构成；验证失败打印内容并以3退出，不执行任何kill。
if [[ ! "${PID}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] invalid PID in ${PID_FILE}: ${PID}"
  exit 3
fi

# kill -0仅探测进程是否存在以及当前用户是否有发信号权限，并不真正终止进程。
# 若探测失败，脚本删除陈旧PID文件并以0退出；重复停止已结束任务因此是幂等的。
if ! kill -0 "${PID}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "[INFO] process already ended; PID file removed"
  exit 0
fi

# 默认空值表示尚未验证目标身份；/proc不可读时后面的比较会拒绝停止，而不是冒险继续。
ENV_RUN_ID=""

# Linux /proc中的环境项由NUL分隔；管道将其逐行化、提取RUN_ID并只保留第一项。
# 最后的true避免无匹配或管道提前关闭在pipefail下直接结束，确保统一走显式的返回码4分支。
if [[ -r "/proc/${PID}/environ" ]]; then
  ENV_RUN_ID="$(
    tr '\0' '\n' < "/proc/${PID}/environ" |
      sed -n 's/^RUN_ID=//p' |
      head -n 1 || true
  )"
fi

# 当前进程的RUN_ID必须与PID文件名中的请求值完全一致；否则返回4，防止PID复用造成误杀。
if [[ "${ENV_RUN_ID}" != "${RUN_ID}" ]]; then
  echo "[ERROR] RUN_ID does not match PID ${PID}"
  exit 4
fi

# 先发送SIGTERM请求正常退出；是否保存中间状态取决于train_busi.py自身是否注册信号处理。
# 若发送时目标已结束，就清除PID文件并按成功停止返回。
kill -TERM "${PID}" 2>/dev/null || {
  rm -f "${PID_FILE}"
  exit 0
}

# 最多等待10轮，每轮探测后间隔1秒；若已退出就清理PID文件、报告普通停止并返回0。
for _ in {1..10}; do
  if ! kill -0 "${PID}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "[INFO] stopped train_busi.py: RUN_ID=${RUN_ID} PID=${PID}"
    exit 0
  fi

  sleep 1
done

# 约10秒后仍存活才发送SIGKILL；|| true容忍最后探测与发送之间的自然退出竞态。
kill -KILL "${PID}" 2>/dev/null || true
# 移除已经完成生命周期的PID文件，避免遗留映射影响后续运行。
rm -f "${PID_FILE}"

# 报告走到了强制停止分支，并输出RUN_ID/PID供日志核对。
echo "[INFO] force-stopped train_busi.py: RUN_ID=${RUN_ID} PID=${PID}"
