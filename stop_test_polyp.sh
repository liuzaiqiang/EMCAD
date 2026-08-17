#!/usr/bin/env bash
# 严格模式让参数、文件或管道错误尽早终止，避免停止流程在不完整状态下继续执行。
set -euo pipefail

# 固定到脚本所在项目根；PID_FILE还显式使用该绝对路径，因此可从任意目录调用本脚本。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# 优先采用命令行第一个参数，缺省时采用环境变量RUN_ID；这两种方式都能指定待停止的测试任务。
RUN_ID="${1:-${RUN_ID:-}}"

# 没有RUN_ID就显示正确调用形式并返回1，阻止构造含糊的PID文件名。
[ -n "${RUN_ID}" ] || {
  echo "[ERROR] usage: bash stop_test_polyp.sh <RUN_ID>"
  exit 1
}

# start_test_polyp.sh把后台PID写入项目根的“RUN_ID.pid”，这里用同一规则定位它。
PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

# 找不到对应PID文件时返回2；此时没有足够证据确定目标进程，不执行任何kill。
[ -f "${PID_FILE}" ] || {
  echo "[ERROR] PID file not found: ${PID_FILE}"
  exit 2
}

# 读取PID并移除文件结尾换行及可能的空格，使后续正则和kill接收规范字符串。
PID="$(tr -d ' \n' < "${PID_FILE}")"

# 仅允许纯数字PID；格式非法返回3，从而把损坏或错误PID文件挡在信号发送之前。
[[ "${PID}" =~ ^[0-9]+$ ]] || {
  echo "[ERROR] invalid PID in ${PID_FILE}: ${PID}"
  exit 3
}

# 信号0只探测进程/权限，不会停止任务；失败时清除陈旧PID文件并返回成功，支持重复执行stop。
kill -0 "${PID}" 2>/dev/null || {
  rm -f "${PID_FILE}"
  exit 0
}

# 默认空值代表尚未从目标进程环境中确认身份。
ENV_RUN_ID=""

# /proc/<PID>/environ以NUL分隔环境变量；转换为行后提取启动器通过env注入的RUN_ID。
# head限定只取第一项，|| true避免无匹配结果触发严格模式的pipefail提前退出。
if [[ -r "/proc/${PID}/environ" ]]; then
  ENV_RUN_ID="$(tr '\0' '\n' < "/proc/${PID}/environ" |
    sed -n 's/^RUN_ID=//p' |
    head -n 1 || true)"
fi

# RUN_ID不完全一致时返回4且拒绝发终止信号，这是防止操作系统复用旧PID后误停其他进程的关键保护。
[ "${ENV_RUN_ID}" = "${RUN_ID}" ] || {
  echo "[ERROR] RUN_ID does not match PID ${PID}"
  exit 4
}

# 身份确认后先发SIGTERM请求正常退出；发送失败通常表示进程刚刚结束，因此删除PID文件并成功返回。
kill -TERM "${PID}" 2>/dev/null || {
  rm -f "${PID_FILE}"
  exit 0
}

# 以1秒间隔最多检查10次；进程一旦消失就清理PID文件、打印普通停止结果并立即结束脚本。
for _ in {1..10}; do
  kill -0 "${PID}" 2>/dev/null || {
    rm -f "${PID_FILE}"
    echo "[INFO] stopped test_polyp.py: RUN_ID=${RUN_ID} PID=${PID}"
    exit 0
  }
  sleep 1
done

# 超过等待窗口后用SIGKILL强制结束；true吞掉“进程恰好已退出”等可接受竞态的非零状态。
kill -KILL "${PID}" 2>/dev/null || true
# 删除已消费的PID映射，避免它在下一次停止操作中成为陈旧引用。
rm -f "${PID_FILE}"

# 记录走到了强制停止分支，便于区分正常响应TERM与超时后KILL。
echo "[INFO] force-stopped test_polyp.py: RUN_ID=${RUN_ID} PID=${PID}"
