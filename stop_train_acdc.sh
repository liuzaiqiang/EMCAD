#!/usr/bin/env bash
# 严格模式使未定义变量、失败命令和失败管道立即终止，避免异常状态继续进入信号发送阶段。
set -euo pipefail

# 解析脚本所在项目根并切换过去；后面的相对PID_FILE因此实际位于项目根目录。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# RUN_ID优先读取第一个位置参数，未传时回退到外部环境变量RUN_ID。
RUN_ID="${1:-${RUN_ID:-}}"
# 两种来源均为空时打印用法并返回1，不对任何进程执行操作。
[ -n "${RUN_ID}" ] || {
  echo "[ERROR] usage: bash stop_train_acdc.sh <RUN_ID>"
  exit 1
}

# 虽然这里未显式拼PROJECT_DIR，但前面已经cd到项目根，所以“RUN_ID.pid”在项目根解析。
PID_FILE="${RUN_ID}.pid"
# PID文件不存在返回2，表示无法找到指定运行对应的进程号记录。
[ -f "${PID_FILE}" ] || {
  echo "[ERROR] PID file not found: ${PID_FILE}"
  exit 2
}

# 读取PID并去除空格、换行；随后用正则确认它是纯数字。
PID="$(tr -d ' \n' < "${PID_FILE}")"
# 非数字或空PID返回3，防止损坏文件内容被作为kill参数。
[[ "${PID}" =~ ^[0-9]+$ ]] || {
  echo "[ERROR] invalid PID in ${PID_FILE}: ${PID}"
  exit 3
}

# kill -0仅检查进程存在性/信号权限；失败时删除陈旧PID文件并以0退出，实现重复停止的幂等性。
kill -0 "${PID}" 2>/dev/null || { rm -f "${PID_FILE}"; exit 0; }

# 先设为空值；只有成功读取目标进程环境并找到RUN_ID后，身份校验才可能通过。
ENV_RUN_ID=""
# /proc/<PID>/environ是Linux提供的NUL分隔进程环境；tr逐行化、sed取RUN_ID、head取第一项。
# 本管道没有追加“|| true”，因此在严格模式下读取或管道失败可能直接终止脚本，这是当前代码的实际行为。
if [[ -r "/proc/${PID}/environ" ]]; then
  ENV_RUN_ID="$(tr '\0' '\n' < "/proc/${PID}/environ" | sed -n 's/^RUN_ID=//p' | head -n 1)"
fi
# 环境中的RUN_ID必须与请求值完全一致；否则打印错误并返回4，防止旧PID被复用后误杀其他进程。
[ "${ENV_RUN_ID}" = "${RUN_ID}" ] || {
  echo "[ERROR] RUN_ID does not match PID ${PID}"
  exit 4
}

# 身份通过后发送SIGTERM，请求训练进程退出；该命令未包容失败，失败会因set -e直接终止当前脚本。
# SIGTERM是否触发保存或清理取决于train_ACDC.py自身的信号处理，Shell本身不保证保存检查点。
kill -TERM "${PID}"
# 最多等待10轮，每轮先用kill -0探测；发现目标消失便删除PID文件并成功退出，否则等待1秒。
for _ in {1..10}; do
  kill -0 "${PID}" 2>/dev/null || { rm -f "${PID_FILE}"; exit 0; }
  sleep 1
done

# 约10秒后进程仍在则发送SIGKILL；|| true容忍进程恰好在发送前自行结束。
kill -KILL "${PID}" 2>/dev/null || true
# 强制停止路径结束后清除PID文件，避免保留过期的RUN_ID到PID映射。
rm -f "${PID_FILE}"
# 报告停止对象；当前原始输出文字没有区分TERM成功与KILL路径，此处实际位于KILL回退之后。
echo "[INFO] stopped train_ACDC.py: RUN_ID=${RUN_ID} PID=${PID}"
