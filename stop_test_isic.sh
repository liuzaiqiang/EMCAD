#!/usr/bin/env bash
# Bash严格模式使参数、文件读取和管道错误不会被静默跳过，适合包含kill操作的控制脚本。
set -euo pipefail

# 解析项目根并切换过去；随后又用PROJECT_DIR构造绝对PID路径，确保定位不依赖调用目录。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# 第一个位置参数优先级高于外部RUN_ID环境变量；二者都未提供时结果为空字符串。
RUN_ID="${1:-${RUN_ID:-}}"

# 未指定RUN_ID时打印调用格式并返回1，不进行任何进程查询或信号发送。
[ -n "${RUN_ID}" ] || {
  echo "[ERROR] usage: bash stop_test_isic.sh <RUN_ID>"
  exit 1
}

# 测试启动器将后台进程号写在项目根的“RUN_ID.pid”，此处按相同约定读取。
PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

# 文件不存在返回2，说明没有可验证的RUN_ID到PID映射。
[ -f "${PID_FILE}" ] || {
  echo "[ERROR] PID file not found: ${PID_FILE}"
  exit 2
}

# 读出PID并去除空格、换行，消除文本文件行尾对校验和系统调用的影响。
PID="$(tr -d ' \n' < "${PID_FILE}")"

# 纯数字正则验证失败时返回3；这同时覆盖空文件、损坏内容和误写入其他文本。
[[ "${PID}" =~ ^[0-9]+$ ]] || {
  echo "[ERROR] invalid PID in ${PID_FILE}: ${PID}"
  exit 3
}

# kill -0不终止进程，只检查存在性/发信号权限；失败即删除陈旧PID文件并成功结束。
# 这样目标已经停止时再次执行本脚本仍得到成功结果，而不会把“已达成”当作错误。
kill -0 "${PID}" 2>/dev/null || {
  rm -f "${PID_FILE}"
  exit 0
}

# 目标进程身份尚未验证时保持空值，保证后续采用默认拒绝策略。
ENV_RUN_ID=""

# 从Linux的/proc进程环境读取启动器注入的RUN_ID：NUL转换行、sed取值、head取首项。
# || true让无匹配等管道状态不绕过下面的显式安全比较。
if [[ -r "/proc/${PID}/environ" ]]; then
  ENV_RUN_ID="$(
    tr '\0' '\n' < "/proc/${PID}/environ" |
      sed -n 's/^RUN_ID=//p' |
      head -n 1 || true
  )"
fi

# 环境RUN_ID与请求值不相等时返回4，不发送TERM；该检查专门防止旧PID被系统复用后误停新进程。
[ "${ENV_RUN_ID}" = "${RUN_ID}" ] || {
  echo "[ERROR] RUN_ID does not match PID ${PID}"
  exit 4
}

# 身份匹配后先发送可捕获的SIGTERM；若目标在此刻已经退出，则只清PID文件并返回成功。
kill -TERM "${PID}" 2>/dev/null || {
  rm -f "${PID_FILE}"
  exit 0
}

# 给予进程最多约10秒响应TERM；每秒探测一次，发现结束便清理映射并报告正常停止。
for _ in {1..10}; do
  kill -0 "${PID}" 2>/dev/null || {
    rm -f "${PID_FILE}"
    echo "[INFO] stopped test_isic.py: RUN_ID=${RUN_ID} PID=${PID}"
    exit 0
  }

  sleep 1
done

# 十轮后仍存活则以SIGKILL强制终止；|| true吸收进程恰好退出造成的竞态错误。
kill -KILL "${PID}" 2>/dev/null || true
# 清除PID文件，避免后续根据过期映射操作已被复用的进程号。
rm -f "${PID_FILE}"

# 打印强制停止分支结果，便于从终端确认停止对象。
echo "[INFO] force-stopped test_isic.py: RUN_ID=${RUN_ID} PID=${PID}"
