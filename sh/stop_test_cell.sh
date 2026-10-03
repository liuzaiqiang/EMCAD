#!/usr/bin/env bash
# Cell 测试专用停止入口；实际测试进程由 test_polyp.py 运行，复用其 PID/RUN_ID 安全核验实现。
set -euo pipefail

# 依据当前脚本位置找到项目 sh 目录，避免依赖调用者的工作目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 原样传递测试 RUN_ID；通用脚本验证 PID 文件和进程环境后才发送 SIGTERM。
# 使用示例：bash sh/stop_test_cell.sh <start_test_cell.sh 输出的 RUN_ID>。
exec bash "${SCRIPT_DIR}/stop_test_polyp.sh" "$@"
