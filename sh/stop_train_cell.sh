#!/usr/bin/env bash
# Cell 训练的专用停止入口；进程仍由 train_polyp.py 执行，故复用已有身份核验和信号处理实现。
set -euo pipefail

# 依据本文件位置推导 sh 目录，从任何当前工作目录都能找到项目内的通用停止脚本。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 把调用者给出的 RUN_ID 原样传入；底层会核验 PID 文件与 /proc/<pid>/environ 的 RUN_ID。
# 使用示例：bash sh/stop_train_cell.sh <start_train_cell.sh 输出的 RUN_ID>。
exec bash "${SCRIPT_DIR}/stop_train_polyp.sh" "$@"
