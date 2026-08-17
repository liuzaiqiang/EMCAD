#!/usr/bin/env bash
# 开启Bash严格模式，让未定义变量、失败命令或失败管道不会被静默忽略。
set -euo pipefail

# 获取脚本所在项目根并进入该目录；相对PID_FILE因而稳定解析到项目根。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# RUN_ID先取命令行第一个参数，缺省时再从调用环境读取同名变量。
RUN_ID="${1:-${RUN_ID:-}}"
# RUN_ID为空时打印正确调用方式并返回1，避免无法确认目标时继续执行。
[ -n "${RUN_ID}" ] || {
  echo "[ERROR] usage: bash stop_test_acdc.sh <RUN_ID>"
  exit 1
}

# start_test_acdc.sh同样在项目根写“RUN_ID.pid”；前面的cd使这个相对路径与启动器一致。
PID_FILE="${RUN_ID}.pid"
# 找不到PID文件时返回2，不尝试用进程名进行模糊查找。
[ -f "${PID_FILE}" ] || {
  echo "[ERROR] PID file not found: ${PID_FILE}"
  exit 2
}

# 从文本文件读取PID并去掉空白行尾，得到待校验的进程号。
PID="$(tr -d ' \n' < "${PID_FILE}")"
# 只接受一个或多个数字；非法内容返回3且不会发送任何信号。
[[ "${PID}" =~ ^[0-9]+$ ]] || {
  echo "[ERROR] invalid PID in ${PID_FILE}: ${PID}"
  exit 3
}

# 信号0仅探测进程是否存在和当前用户是否有权限；失败时清除旧PID文件并成功返回，支持重复stop。
kill -0 "${PID}" 2>/dev/null || { rm -f "${PID_FILE}"; exit 0; }

# 默认空值代表尚未证明当前PID仍属于指定RUN_ID。
ENV_RUN_ID=""
# 从Linux /proc进程环境中读取启动时注入的RUN_ID：NUL转换行、sed提取值、head取首个匹配。
# 当前管道未使用“|| true”；在set -e和pipefail下，读取/匹配异常可能直接结束脚本而不进入比较块。
if [[ -r "/proc/${PID}/environ" ]]; then
  ENV_RUN_ID="$(tr '\0' '\n' < "/proc/${PID}/environ" | sed -n 's/^RUN_ID=//p' | head -n 1)"
fi
# RUN_ID完全一致才允许继续；不一致以4退出，这是抵御PID复用误停其他任务的核心校验。
[ "${ENV_RUN_ID}" = "${RUN_ID}" ] || {
  echo "[ERROR] RUN_ID does not match PID ${PID}"
  exit 4
}

# 先发送SIGTERM请求测试进程退出；此处未吞掉失败，kill失败会依照严格模式终止脚本。
kill -TERM "${PID}"
# 最多等待约10秒，每秒用kill -0检测一次；一旦结束便删除PID文件并以0退出。
for _ in {1..10}; do
  kill -0 "${PID}" 2>/dev/null || { rm -f "${PID_FILE}"; exit 0; }
  sleep 1
done

# 等待结束仍存活时发送SIGKILL强制终止；|| true处理进程在竞态窗口中已退出的情况。
kill -KILL "${PID}" 2>/dev/null || true
# 删除本次运行的PID记录，防止后续再引用已结束任务的进程号。
rm -f "${PID_FILE}"
# 输出停止结果；该echo位于SIGKILL回退之后，因此表示走完了最长等待路径。
echo "[INFO] stopped test_ACDC.py: RUN_ID=${RUN_ID} PID=${PID}"
