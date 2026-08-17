#!/usr/bin/env bash
# 严格模式让任何未处理的失败、未定义变量或失败管道立即终止，减少错误停止目标的可能。
set -euo pipefail

# 解析脚本所在项目目录并切换过去；后续PID_FILE使用绝对路径，调用位置不会改变目标。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# RUN_ID可由第一个位置参数给出，也可在未传参数时通过环境变量提供；命令行参数优先。
RUN_ID="${1:-${RUN_ID:-}}"

# RUN_ID为空时显示用法并返回1；停止脚本不会按Python进程名做不安全的模糊匹配。
if [[ -z "${RUN_ID}" ]]; then
  echo "[ERROR] usage: bash stop_test_busi.sh <RUN_ID>"
  exit 1
fi

# 根据测试启动器的命名规则定位项目根下唯一的“RUN_ID.pid”文件。
PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

# 没有对应文件时返回2，因为无法证明某个PID属于用户指定的这次测试。
if [[ ! -f "${PID_FILE}" ]]; then
  echo "[ERROR] PID file not found: ${PID_FILE}"
  exit 2
fi

# 读取文件并删除空格、换行，把文本记录还原为用于系统信号调用的PID字符串。
PID="$(tr -d ' \n' < "${PID_FILE}")"

# 只接受完整的十进制数字；格式不合法时返回3并报告损坏内容。
if [[ ! "${PID}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] invalid PID in ${PID_FILE}: ${PID}"
  exit 3
fi

# 信号0只做进程存在性/权限检查，不会结束进程；失败时PID文件已失效，因此删除并返回成功。
# 这使脚本可重复调用：目标本就不在时，“停止”这一最终状态已经满足。
if ! kill -0 "${PID}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "[INFO] process already ended; PID file removed"
  exit 0
fi

# 在/proc身份信息读取成功前保持空串，遵循无法确认就拒绝停止的默认策略。
ENV_RUN_ID=""

# 将/proc/<PID>/environ的NUL分隔项转为文本行，再提取启动时通过env写入的RUN_ID。
# head只取第一个匹配，|| true避免无匹配触发pipefail，从而让下一块统一给出身份不符错误。
if [[ -r "/proc/${PID}/environ" ]]; then
  ENV_RUN_ID="$(
    tr '\0' '\n' < "/proc/${PID}/environ" |
      sed -n 's/^RUN_ID=//p' |
      head -n 1 || true
  )"
fi

# RUN_ID不同则以4退出且不发信号；这是PID文件之外的第二重身份校验，可阻断PID复用误杀。
if [[ "${ENV_RUN_ID}" != "${RUN_ID}" ]]; then
  echo "[ERROR] RUN_ID does not match PID ${PID}"
  exit 4
fi

# 先发送SIGTERM给进程正常退出机会；发送失败通常意味着它在检查后已结束，此时清PID并返回0。
kill -TERM "${PID}" 2>/dev/null || {
  rm -f "${PID_FILE}"
  exit 0
}

# 每秒检查一次、最多10次；一旦目标消失，就清理PID文件并报告由TERM完成的普通停止。
for _ in {1..10}; do
  if ! kill -0 "${PID}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "[INFO] stopped test_busi.py: RUN_ID=${RUN_ID} PID=${PID}"
    exit 0
  fi

  sleep 1
done

# 等待期满仍存活才发送不可捕获的SIGKILL；true容忍发送前进程刚好退出的竞态。
kill -KILL "${PID}" 2>/dev/null || true
# 删除RUN_ID到旧PID的映射，避免未来把它误认为仍在运行。
rm -f "${PID_FILE}"

# 终端信息明确区分这是超时后的强制停止结果。
echo "[INFO] force-stopped test_busi.py: RUN_ID=${RUN_ID} PID=${PID}"
