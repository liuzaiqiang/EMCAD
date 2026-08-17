#!/usr/bin/env bash
# 严格模式让未定义变量、失败命令和失败管道立即暴露，降低错误PID进入停止流程的风险。
set -euo pipefail

# 获取脚本所在项目根并切换过去；PID_FILE还使用绝对路径，调用位置不会影响目标文件。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# RUN_ID优先来自第一个命令行参数，否则回退到同名环境变量；两者都缺失时为空。
RUN_ID="${1:-${RUN_ID:-}}"

# 空RUN_ID返回1并打印用法，因为缺少唯一运行标识时不能安全选择训练进程。
[ -n "${RUN_ID}" ] || {
  echo "[ERROR] usage: bash stop_train_isic.sh <RUN_ID>"
  exit 1
}

# 与启动器的命名规则对应，项目根下每个运行标识拥有独立的PID文件。
PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

# PID文件缺失返回2；脚本不会退而使用进程名模糊匹配，以免停止其他训练任务。
[ -f "${PID_FILE}" ] || {
  echo "[ERROR] PID file not found: ${PID_FILE}"
  exit 2
}

# 删除PID文件内容中的空格和换行，得到待验证的进程号。
PID="$(tr -d ' \n' < "${PID_FILE}")"

# 只接受十进制数字组成的PID；空值或其他字符返回3，不向系统发送信号。
[[ "${PID}" =~ ^[0-9]+$ ]] || {
  echo "[ERROR] invalid PID in ${PID_FILE}: ${PID}"
  exit 3
}

# kill -0只做存在性和权限探测；失败时认为任务已经结束，清除陈旧PID文件并以0返回。
# 因而重复运行停止命令不会因为目标早已消失而报错，这就是这里的幂等语义。
kill -0 "${PID}" 2>/dev/null || {
  rm -f "${PID_FILE}"
  exit 0
}

# 在读取目标进程环境前初始化为空，无法读取/proc时会在下一步身份比较中被安全拒绝。
ENV_RUN_ID=""

# Linux进程环境以NUL分隔：tr转成逐行形式，sed提取RUN_ID，head限制为首个匹配。
# 管道末尾的true避免“未找到RUN_ID”等状态在pipefail下直接中断，交由显式比较返回4。
if [[ -r "/proc/${PID}/environ" ]]; then
  ENV_RUN_ID="$(
    tr '\0' '\n' < "/proc/${PID}/environ" |
      sed -n 's/^RUN_ID=//p' |
      head -n 1 || true
  )"
fi

# 只有启动时注入进程环境的RUN_ID与用户指定值相同才继续；不匹配返回4以防PID复用误杀。
[ "${ENV_RUN_ID}" = "${RUN_ID}" ] || {
  echo "[ERROR] RUN_ID does not match PID ${PID}"
  exit 4
}

# 先发SIGTERM请求进程退出；Python是否执行额外清理或保存动作取决于其自身信号处理实现。
# 如果发信号时目标已消失，则清理PID文件并把任务视为已经停止。
kill -TERM "${PID}" 2>/dev/null || {
  rm -f "${PID_FILE}"
  exit 0
}

# 最多等待约10秒：每轮先用kill -0探测，仍存活才sleep 1；退出后立即清PID并报告正常停止。
for _ in {1..10}; do
  kill -0 "${PID}" 2>/dev/null || {
    rm -f "${PID_FILE}"
    echo "[INFO] stopped train_isic.py: RUN_ID=${RUN_ID} PID=${PID}"
    exit 0
  }

  sleep 1
done

# TERM等待窗口结束后仍存活则发送SIGKILL；|| true容忍进程在最后一次探测后自行结束。
kill -KILL "${PID}" 2>/dev/null || true
# 删除PID文件，结束RUN_ID到该进程号的生命周期映射。
rm -f "${PID_FILE}"

# 输出强制停止结果，保留RUN_ID和PID供日志排查。
echo "[INFO] force-stopped train_isic.py: RUN_ID=${RUN_ID} PID=${PID}"
