#!/usr/bin/env bash
# 停止脚本同样启用严格模式，避免缺失参数、失败命令或失败管道被静默忽略。
set -euo pipefail

# 解析脚本所在项目根并切换过去，使相对命令上下文稳定；本文件的PID_FILE本身还使用绝对路径。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# RUN_ID优先取第一个位置参数；未传参数时再读取外部RUN_ID环境变量，二者都没有则得到空串。
RUN_ID="${1:-${RUN_ID:-}}"

# RUN_ID为空时打印用法并返回1；没有实验标识就不能安全确定要读取哪个PID文件。
[ -n "${RUN_ID}" ] || {
  echo "[ERROR] usage: bash stop_train_polyp.sh <RUN_ID>"
  exit 1
}

# 启动器以“RUN_ID.pid”记录后台进程号；显式拼上PROJECT_DIR后不受调用者当前目录影响。
PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

# PID文件不存在时返回2，表示任务标识错误、启动未成功记录PID，或PID文件已被清理。
[ -f "${PID_FILE}" ] || {
  echo "[ERROR] PID file not found: ${PID_FILE}"
  exit 2
}

# 从PID文件读取内容并删除空格和换行，得到后续系统调用使用的纯PID字符串。
PID="$(tr -d ' \n' < "${PID_FILE}")"

# 正则要求PID只含一个或多个十进制数字；空文件或其他内容均返回3，避免把异常文本交给kill。
[[ "${PID}" =~ ^[0-9]+$ ]] || {
  echo "[ERROR] invalid PID in ${PID_FILE}: ${PID}"
  exit 3
}

# kill -0只检查目标进程是否存在且当前用户是否有发信号权限，不发送真正的终止信号。
# 检查失败时删除已经失效的PID文件并成功退出，使重复停止已结束任务成为幂等操作。
kill -0 "${PID}" 2>/dev/null || {
  rm -f "${PID_FILE}"
  exit 0
}

# 先置空，确保/proc环境文件不可读时后面的身份比较必然失败，而不会沿用其他值。
ENV_RUN_ID=""

# Linux将进程环境变量以NUL字节分隔；tr转成逐行文本，sed提取RUN_ID，head只取第一项。
# 末尾true用于容忍管道中“没有匹配项”等非零状态，避免pipefail在身份比较前直接终止脚本。
if [[ -r "/proc/${PID}/environ" ]]; then
  ENV_RUN_ID="$(tr '\0' '\n' < "/proc/${PID}/environ" |
    sed -n 's/^RUN_ID=//p' |
    head -n 1 || true)"
fi

# 只有进程环境中的RUN_ID与请求停止的RUN_ID完全一致才继续；不匹配返回4，防止PID复用导致误杀。
[ "${ENV_RUN_ID}" = "${RUN_ID}" ] || {
  echo "[ERROR] RUN_ID does not match PID ${PID}"
  exit 4
}

# 先发送SIGTERM，请求进程正常退出；是否保存检查点取决于Python程序自身是否实现对应信号处理。
# 若发送时进程已结束，则清理PID文件并按“目标已停止”成功返回。
kill -TERM "${PID}" 2>/dev/null || {
  rm -f "${PID_FILE}"
  exit 0
}

# 最多轮询10次，每次仍存活就等待1秒；任一次发现进程结束便清理PID文件、报告成功并退出。
for _ in {1..10}; do
  kill -0 "${PID}" 2>/dev/null || {
    rm -f "${PID_FILE}"
    echo "[INFO] stopped train_polyp.py: RUN_ID=${RUN_ID} PID=${PID}"
    exit 0
  }
  sleep 1
done

# 等待约10秒后仍存活则发送不可捕获的SIGKILL；|| true容忍检查与发送之间进程自行结束的竞态。
kill -KILL "${PID}" 2>/dev/null || true
# 无论KILL是否刚好遇到进程已结束，都移除本次运行的PID文件，避免后续把它当成活动任务。
rm -f "${PID_FILE}"

# 明确报告本次经过强制停止路径及其RUN_ID/PID，便于与日志记录对应。
echo "[INFO] force-stopped train_polyp.py: RUN_ID=${RUN_ID} PID=${PID}"
