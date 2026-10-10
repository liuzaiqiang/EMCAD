#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# DSB18 / EM 细胞二分类训练启动器
#
# 本脚本提供 Cell 专用数据集默认值和校验；训练主体复用已存在的 train_polyp.py，
# 它本身实现通用二值分割损失、增强、验证和 checkpoint 保存，避免重复维护训练循环。
# run_seed_queue.sh 会 source 本脚本并等待这里返回的 PID，所以必须在当前 Bash 进程
# 中 source 通用启动器，不能用后台调用把真正训练 PID 脱离队列。
#
# 数据目录：DATA_ROOT/DATASET_NAME/{train,val,test}/{images,masks}/...
# DSB18 可将每张图像的多个实例掩膜合成一个语义前景；EM 使用配对的语义掩膜。
# -----------------------------------------------------------------------------

# -e 遇到未处理失败即停，-u 拒绝空未定义变量，pipefail 检查管道中每个环节的退出状态。
set -euo pipefail

# 根据当前启动文件位置计算项目根，使用户从项目任意目录调用时路径仍保持正确。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Cell 数据根默认位于仓库同级的 data/cell/target；集群使用其他挂载点时可传 DATA_ROOT 覆盖。
export DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../data/cell/target}"

# 固定系列标签为 Cell，让训练日志和模型输出不会混入 Polyp 目录。
export DATASET="Cell"

# 允许一份启动器服务两套任务；未指定时默认 DSB18。
export DATASET_NAME="${DATASET_NAME:-DSB18}"

# 只允许 EMCAD 论文中的两个细胞数据集名称，避免拼错后指向不存在的子目录。
case "${DATASET_NAME}" in
  DSB18|EM) ;;
  *)
    echo "[ERROR] DATASET_NAME must be DSB18 or EM; got: ${DATASET_NAME}" >&2
    exit 2
    ;;
esac

# 论文对 DSB18 与 EM 都使用 256x256 输入；显式传入 IMG_SIZE 时保留用户设置。
export IMG_SIZE="${IMG_SIZE:-256}"

# Cell 论文协议使用固定尺度；这里关闭息肉/皮肤任务用的多尺度训练，仍允许显式覆盖。
export USE_MULTI_SCALE_TRAINING="${USE_MULTI_SCALE_TRAINING:-0}"

# DSB18 默认将每张图像对应的实例掩膜并成语义前景；EM 默认是一张图像配一张语义掩膜。
# 用 MERGE_INSTANCE_MASKS=0/1 显式覆盖时，两个数据集均尊重调用者选择。
if [[ "${DATASET_NAME}" == "DSB18" ]]; then
  export MERGE_INSTANCE_MASKS="${MERGE_INSTANCE_MASKS:-1}"
else
  export MERGE_INSTANCE_MASKS="${MERGE_INSTANCE_MASKS:-0}"
fi

# PVTv2 入口默认三通道；若 EM 文件按单通道灰度读取，可设置 INPUT_CHANNELS=1。
export INPUT_CHANNELS="${INPUT_CHANNELS:-3}"

# 通用二分类启动器会把 N_GPU 传给 Python 的 --n_gpu；为 Cell 明确给出单 GPU 默认值，
# 以免严格模式 set -u 因变量未定义而在创建训练进程前退出；多卡时可在命令前覆盖。
export N_GPU="${N_GPU:-1}"

# Forward the same isolated controls to the shared binary training launcher.
export USE_PIXEL_RELIABILITY_FUSION="${USE_PIXEL_RELIABILITY_FUSION:-0}"
export FUSION_DICE_SOFTMAX="${FUSION_DICE_SOFTMAX:-0}"

# 在访问数据前检查通道数和掩膜合并开关，避免把非法值传入训练入口。
[[ "${INPUT_CHANNELS}" == "1" || "${INPUT_CHANNELS}" == "3" ]] || {
  echo "[ERROR] INPUT_CHANNELS must be 1 or 3; got: ${INPUT_CHANNELS}" >&2
  exit 2
}
[[ "${MERGE_INSTANCE_MASKS}" == "0" || "${MERGE_INSTANCE_MASKS}" == "1" ]] || {
  echo "[ERROR] MERGE_INSTANCE_MASKS must be 0 or 1; got: ${MERGE_INSTANCE_MASKS}" >&2
  exit 2
}

# 底层通用启动器沿用论文默认200轮、batch 16、AdamW学习率和权重衰减1e-4；
# 项目根固定后 source 它，使本脚本、通用启动器和队列共享相同训练 PID 与停止标识。
cd "${PROJECT_DIR}"
source "${SCRIPT_DIR}/start_train_polyp.sh"
