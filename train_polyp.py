# argparse 定义 Polyp/BUSI/ISIC 共用训练命令行接口。
import argparse
# csv 追加 epoch 级训练历史与逐病例验证指标。
import csv
# json 保存可复现实验配置。
import json
# logging 同时写文件和标准输出。
import logging
# os 负责路径检查、环境变量和文件操作。
import os
# sys.stdout 接入日志处理器。
import sys
# time 统计完整训练耗时。
import time
# datetime 生成默认运行名。
from datetime import datetime

# NumPy 计算 epoch 平均损失和记录数值。
import numpy as np
# PyTorch 负责模型、优化和设备。
import torch
# nn 提供多 GPU DataParallel 类型检查/封装。
import torch.nn as nn
# F.interpolate 支持论文 Polyp 多尺度训练。
import torch.nn.functional as F
# TensorBoard 写入损失、学习率和验证指标。
from tensorboardX import SummaryWriter
# AMP 梯度缩放和自动混合精度上下文。
from torch.cuda.amp import GradScaler, autocast
# tqdm 显示 epoch/batch 进度。
from tqdm import tqdm

# 默认使用通用二分类图像/掩膜严格配对加载器；ISIC/BUSI 包装器会替换它。
from utils.dataloader_polyp import get_loader
# 二分类模型、损失、评测、检查点和设备辅助函数。
from utils.polyp_utils import (
    # 二分类像素级可靠性融合的融合/校准辅助损失。
    # 构建 EMCADNet。
    build_model,
    # 在验证/测试加载器上统计 Dice/IoU 等。
    evaluate_loader,
    # 恢复检查点。
    load_checkpoint,
    # 统一单输出/多输出列表。
    model_outputs,
    model_outputs_with_boundary,
    boundary_loss,
    # 解析 auto/cpu/cuda 设备。
    resolve_device,
    # 保存模型及优化状态。
    save_checkpoint,
    # 固定随机源。
    seed_everything,
    # 论文结构损失及多头监督组合。
    supervised_structure_loss,
)
from lib.model_complexity import log_model_complexity
from lib.experiment_paths import make_experiment_dir
from lib.benchmarking import log_environment, log_json, peak_memory_mb, reset_peak_memory, synchronize


# 解析所有二分类训练参数；默认值对应 EMCAD Polyp 主实验口径。
def parse_args():
    # 创建带用途说明的解析器。
    parser = argparse.ArgumentParser(
        # 显示在 --help 顶部。
        description="Train EMCAD on one Polyp dataset"
    )

    # 准备数据总根，其下按数据集名和 train/val/test 组织。
    parser.add_argument(
        # 选项名。
        "--data_root",
        # 默认路径。
        default="../data/polyp/target",
    )
    # 当前训练的具体数据集子目录。
    parser.add_argument(
        # 选项名。
        "--dataset_name",
        # 论文 Polyp 默认入口使用 ClinicDB。
        default="ClinicDB",
    )
    # 检查点、配置和日志输出根。
    parser.add_argument(
        # 选项名。
        "--output_dir",
        # 默认输出目录。
        default="./model_pth/Polyp",
    )
    # 单次运行名；为空时 main 自动生成。
    parser.add_argument(
        # 选项名。
        "--run_name",
        # 自动命名。
        default=None,
    )
    # 可选断点检查点路径。
    parser.add_argument(
        # 选项名。
        "--checkpoint",
        # 默认从头训练。
        default=None,
    )

    # 层次化编码器名称。
    parser.add_argument(
        # 选项名。
        "--encoder",
        # EMCAD 论文主干 PVTv2-B2。
        default="pvt_v2_b2",
    )
    # MSDC 多尺度深度卷积核列表，对应论文 PDF4 Eq.(5)-(6)。
    parser.add_argument(
        # 选项名。
        "--kernel_sizes",
        # 每项转 int。
        type=int,
        # 接收一个或多个值。
        nargs="+",
        # 论文默认1/3/5。
        default=[1, 3, 5],
    )
    # MSCB 通道扩张倍数，对应论文 PDF4 Eq.(4)。
    parser.add_argument(
        # 选项名。
        "--expansion_factor",
        # 整数。
        type=int,
        # 论文默认2。
        default=2,
    )
    # LGAG 分组卷积核大小。
    parser.add_argument(
        # 选项名。
        "--lgag_ks",
        # 整数。
        type=int,
        # 默认3x3。
        default=3,
    )
    # MSCB 激活函数。
    parser.add_argument(
        # 选项名。
        "--activation_mscb",
        # 论文描述使用 ReLU6。
        default="relu6",
    )
    # 出现该标志时把 MSDC 从论文默认并行切为串行。
    parser.add_argument(
        # 选项名。
        "--no_dw_parallel",
        # 布尔开关。
        action="store_true",
    )
    # 出现时将多尺度分支由相加改为通道拼接；拼接不是论文默认。
    parser.add_argument(
        # 选项名。
        "--concatenation",
        # 布尔开关。
        action="store_true",
    )
    # 关闭编码器预训练参数加载。
    parser.add_argument(
        # 选项名。
        "--no_pretrain",
        # 布尔开关。
        action="store_true",
    )
    # PVT 本地预训练权重目录。
    parser.add_argument(
        # 选项名。
        "--pretrained_dir",
        # 默认目录。
        default="./pretrained_pth/pvt/",
    )
    # CAA 仅改变 EUCB 上采样；off 模式保留基础模型路径。
    parser.add_argument("--caa_mode", choices=["off", "aa_only", "content_only", "caa"], default="off")
    parser.add_argument("--caa_residual_scale", type=float, default=0.1)
    parser.add_argument("--use_boundary_refinement", type=int, choices=[0, 1], default=0)
    parser.add_argument("--boundary_loss_weight", type=float, default=0.1)
    # DSB18 可用该开关把每张图像目录下的多个细胞实例掩膜并为前景。
    parser.add_argument("--merge_instance_masks", type=int, choices=[0, 1], default=0)

    # 多输出监督策略。
    parser.add_argument(
        # 选项名。
        "--supervision",
        # 限定合法策略。
        choices=[
            # 论文二分类 Eq.(11)：四个单头加四头和，共5项结构损失。
            "paper",
            # 四个输出分别监督。
            "deep_supervision",
            # 仅最终 P[-1]。
            "last_layer",
            # 所有15个非空输出组合。
            "mutation",
        ],
        # Polyp 采用论文策略。
        default="paper",
    )
    # 网络输入边长。
    parser.add_argument(
        # 选项名。
        "--img_size",
        # 整数像素。
        type=int,
        # 论文 Polyp 默认352。
        default=352,
    )
    # 训练批大小。
    parser.add_argument(
        # 选项名。
        "--batch_size",
        # 整数。
        type=int,
        # 论文默认16。
        default=16,
    )
    # 验证批大小；原尺寸GT由自定义collate保留。
    parser.add_argument(
        # 选项名。
        "--val_batch_size",
        # 整数。
        type=int,
        # 默认8。
        default=8,
    )
    # 最大训练轮数。
    parser.add_argument(
        # 选项名。
        "--max_epochs",
        # 整数。
        type=int,
        # Polyp 论文设置200。
        default=200,
    )
    # AdamW 初始学习率。
    parser.add_argument(
        # 选项名。
        "--base_lr",
        # 浮点。
        type=float,
        # 默认1e-4。
        default=1e-4,
    )
    # AdamW 权重衰减。
    parser.add_argument(
        # 选项名。
        "--weight_decay",
        # 浮点。
        type=float,
        # 默认1e-4。
        default=1e-4,
    )
    # 按元素梯度裁剪阈值。
    parser.add_argument(
        # 选项名。
        "--clip",
        # 浮点。
        type=float,
        # 论文 Polyp 设置0.5。
        default=0.5,
    )
    # 学习率调度模式。
    parser.add_argument(
        # 选项名。
        "--scheduler",
        # 常数或余弦。
        choices=["constant", "cosine"],
        # 论文代码默认常数学习率。
        default="constant",
    )
    # 余弦退火最小学习率。
    parser.add_argument(
        # 选项名。
        "--min_lr",
        # 浮点。
        type=float,
        # 默认1e-6。
        default=1e-6,
    )
    # 多尺度训练比例列表。
    parser.add_argument(
        # 选项名。
        "--scale_rates",
        # 浮点列表。
        type=float,
        # 至少一个。
        nargs="+",
        # 论文0.75/1/1.25。
        default=[0.75, 1.0, 1.25],
    )
    # 出现时只用1.0比例；BUSI包装器强制启用。
    parser.add_argument(
        # 选项名。
        "--no_multi_scale",
        # 布尔开关。
        action="store_true",
    )
    # 关闭数据增强。
    parser.add_argument(
        # 选项名。
        "--no_augmentation",
        # 布尔开关。
        action="store_true",
    )
    # 使用单通道输入；ISIC/BUSI包装器禁止该模式。
    parser.add_argument(
        # 选项名。
        "--grayscale",
        # 布尔开关。
        action="store_true",
    )
    # DataLoader worker数。
    parser.add_argument(
        # 选项名。
        "--num_workers",
        # 整数。
        type=int,
        # Windows友好的默认0。
        default=0,
    )
    # 期望使用的GPU数量。
    parser.add_argument(
        # 选项名。
        "--n_gpu",
        # 整数。
        type=int,
        # 默认单卡。
        default=1,
    )
    # 全局随机种子。
    parser.add_argument(
        # 选项名。
        "--seed",
        # 整数。
        type=int,
        # 默认2222。
        default=2222,
    )
    # cuDNN确定性开关。
    parser.add_argument(
        # 选项名。
        "--deterministic",
        # 0/1整数。
        type=int,
        # 限定范围。
        choices=[0, 1],
        # 默认可复现。
        default=1,
    )
    # 每多少epoch验证一次。
    parser.add_argument(
        # 选项名。
        "--validate_every",
        # 整数。
        type=int,
        # 默认每轮。
        default=1,
    )
    # 每多少epoch额外保存周期检查点。
    parser.add_argument(
        # 选项名。
        "--save_every",
        # 整数。
        type=int,
        # 默认50轮。
        default=50,
    )
    # 验证概率二值化阈值。
    parser.add_argument(
        # 选项名。
        "--threshold",
        # 浮点。
        type=float,
        # 默认0.5。
        default=0.5,
    )
    # 调试时限制每epoch训练批数；0表示不限。
    parser.add_argument(
        # 选项名。
        "--max_train_batches",
        # 整数。
        type=int,
        # 不限制。
        default=0,
    )
    # 调试时限制验证病例数；0表示不限。
    parser.add_argument(
        # 选项名。
        "--max_valid_cases",
        # 整数。
        type=int,
        # 不限制。
        default=0,
    )
    # 启用CUDA自动混合精度。
    parser.add_argument(
        # 选项名。
        "--amp",
        # 布尔开关。
        action="store_true",
    )
    # auto/cpu/cuda设备字符串。
    parser.add_argument(
        # 选项名。
        "--device",
        # 自动选择。
        default="auto",
    )

    # 解析实际命令行并返回 Namespace。
    return parser.parse_args()


# 向 history.csv 追加一个 epoch 汇总；首次写入时自动添加表头。
def append_history(path, row):
    # 固定列顺序便于跨运行比较。
    fieldnames = [
        # epoch编号。
        "epoch",
        # 训练平均损失。
        "train_loss",
        # 验证平均Dice。
        "val_dice",
        # 验证平均IoU。
        "val_iou",
        # 当前学习率。
        "learning_rate",
        # 累计耗时秒。
        "elapsed_seconds",
    ]

    # 写入前判断文件是否已存在，以决定是否写表头。
    exists = os.path.isfile(path)

    # 追加模式打开CSV；newline=""交给csv模块管理换行。
    with open(
            # 目标路径。
            path,
            # 追加模式。
            "a",
            # CSV推荐设置。
            newline="",
            # UTF-8。
            encoding="utf-8",
    ) as stream:
        # 按固定字段创建字典写入器。
        writer = csv.DictWriter(
            # 文件流。
            stream,
            # 列顺序。
            fieldnames=fieldnames,
        )

        # 新文件先写列名。
        if not exists:
            # 写表头。
            writer.writeheader()

        # 追加当前epoch字典。
        writer.writerow(row)


# 将一次验证的逐病例Dice/IoU追加到 validation_cases.csv。
def append_validation_rows(path, epoch, rows):
    # 固定逐病例列。
    fieldnames = [
        # 评测发生的epoch。
        "epoch",
        # 病例/图像名。
        "case_name",
        # Dice。
        "dice",
        # IoU。
        "iou",
    ]

    # 判断是否需要表头。
    exists = os.path.isfile(path)

    # 追加打开文件。
    with open(
            # 路径。
            path,
            # 追加。
            "a",
            # CSV换行。
            newline="",
            # 编码。
            encoding="utf-8",
    ) as stream:
        # 创建写入器。
        writer = csv.DictWriter(
            # 流。
            stream,
            # 列定义。
            fieldnames=fieldnames,
        )

        # 首次写表头。
        if not exists:
            # 输出字段名。
            writer.writeheader()

        # 遍历评测函数返回的每个病例记录。
        for row in rows:
            # 只选择稳定字段并附加epoch。
            writer.writerow(
                # 新行字典。
                {
                    # 当前epoch。
                    "epoch": epoch,
                    # 病例名。
                    "case_name": row["case_name"],
                    # Dice值。
                    "dice": row["dice"],
                    # IoU值。
                    "iou": row["iou"],
                }
            )


# 按多尺度比例同步调整图像和掩膜；目标边长对齐到32的倍数以适配四级编码器。
def resized_batch(images, masks, image_size, rate):
    # 比例1.0直接复用原张量，避免无意义插值。
    if rate == 1.0:
        # 返回原图和mask。
        return images, masks

    # 计算最接近 image_size*rate 的32倍数。
    scaled = int(
        # 先除32四舍五入，再乘32。
        round(image_size * rate / 32.0) * 32
    )

    # images = F.interpolate(
    #     images,
    #     size=(scaled, scaled),
    #     mode="bilinear",
    #     align_corners=False,
    # ) 为修复BKAI数据集指标差了8.4个点而修改
    # 连续RGB/灰度图使用双线性插值到scaled正方形。
    images = F.interpolate(
        # 输入[B,C,H,W]。
        images,
        # 目标尺寸。
        size=(scaled, scaled),
        # 双线性模式。
        mode="bilinear",
        # 当前代码为修复BKAI实验显式使用True；这是工程实现，不是论文公式参数。
        align_corners=True,
    )

    # 二值掩膜必须使用最近邻插值，保持0/1标签。
    masks = F.interpolate(
        # 输入[B,1,H,W]。
        masks,
        # 与图像相同目标尺寸。
        size=(scaled, scaled),
        # 最近邻。
        mode="nearest",
    )

    # 返回同步缩放批次。
    return images, masks


# 完整训练入口：数据校验、实验记录、模型构建、训练、验证和检查点选择。
def main():
    # 解析参数；ISIC/BUSI包装器可替换该函数。
    args = parse_args()

    # 验证间隔必须是正整数。
    if args.validate_every < 1:
        # 拒绝0/负值。
        raise ValueError(
            # 错误说明。
            "--validate_every must be at least 1"
        )

    # 二值概率阈值必须严格位于0和1之间。
    if not 0.0 < args.threshold < 1.0:
        # 报告范围。
        raise ValueError(
            # 错误说明。
            "--threshold must be between 0 and 1"
        )

    # 数据集根为 data_root/dataset_name。
    dataset_root = os.path.join(
        # 公共根。
        args.data_root,
        # 具体数据集。
        args.dataset_name,
    )
    # 训练划分根。
    train_root = os.path.join(
        # 数据集根。
        dataset_root,
        # 子目录。
        "train",
    )
    # 验证划分根。
    val_root = os.path.join(
        # 数据集根。
        dataset_root,
        # 子目录。
        "val",
    )

    # 训练前必须存在的四个目录。
    required = [
        # 训练图像。
        os.path.join(train_root, "images"),
        # 训练掩膜。
        os.path.join(train_root, "masks"),
        # 验证图像。
        os.path.join(val_root, "images"),
        # 验证掩膜。
        os.path.join(val_root, "masks"),
    ]

    # 收集缺失目录。
    missing = [
        # 返回路径本身。
        path
        # 遍历要求。
        for path in required
        # 只保留非目录。
        if not os.path.isdir(path)
    ]

    # 任一缺失就拒绝训练。
    if missing:
        # 多行列出路径。
        raise FileNotFoundError(
            # 标题。
            "Missing Polyp train/val directories:\n"
            # 拼每行路径。
            + "\n".join(missing)
        )

    # 固定所有随机源和cuDNN策略。
    seed_everything(
        # 种子。
        args.seed,
        # 0/1转bool。
        bool(args.deterministic),
    )

    # 解析目标设备。
    device = resolve_device(args.device)
    # CUDA时启用锁页内存加速传输。
    pin_memory = device.type == "cuda"

    # 构造训练DataLoader。
    train_loader = get_loader(
        # 训练图像目录。
        image_root=os.path.join(
            # 根。
            train_root,
            # 子目录。
            "images",
        ),
        # 训练掩膜目录。
        gt_root=os.path.join(
            # 根。
            train_root,
            # 子目录。
            "masks",
        ),
        # 批大小。
        batchsize=args.batch_size,
        # 输入尺寸。
        trainsize=args.img_size,
        # 训练打乱。
        shuffle=True,
        # worker数。
        num_workers=args.num_workers,
        # 锁页内存。
        pin_memory=pin_memory,
        # 默认开启增强，除非显式关闭。
        augmentation=not args.no_augmentation,
        # train分支返回固定尺寸mask。
        split="train",
        # 默认RGB，grayscale时单通道。
        color_image=not args.grayscale,
        # 随机种子。
        seed=args.seed,
        # 只在 DSB18 显式启用时把实例文件夹传给通用二分类加载器。
        **({"merge_instance_masks": True} if args.merge_instance_masks else {}),
    )

    # 构造验证DataLoader。
    val_loader = get_loader(
        # 验证图像目录。
        image_root=os.path.join(
            # 根。
            val_root,
            # 子目录。
            "images",
        ),
        # 验证掩膜目录。
        gt_root=os.path.join(
            # 根。
            val_root,
            # 子目录。
            "masks",
        ),
        # 验证批大小。
        batchsize=args.val_batch_size,
        # 模型输入尺寸。
        trainsize=args.img_size,
        # 禁止打乱便于复核顺序。
        shuffle=False,
        # worker数。
        num_workers=args.num_workers,
        # 锁页内存。
        pin_memory=pin_memory,
        # 验证不增强。
        augmentation=False,
        # val分支保留原始GT。
        split="val",
        # 通道模式与训练一致。
        color_image=not args.grayscale,
        # 种子。
        seed=args.seed,
        # 与训练集使用同一实例掩膜解释。
        **({"merge_instance_masks": True} if args.merge_instance_masks else {}),
    )

    overlap = (
            set(train_loader.dataset.stems)
            & set(val_loader.dataset.stems)
    )

    if overlap:
        raise RuntimeError(
            "Train/val leakage detected: {}".format(
                sorted(overlap)[:10]
            )
        )

    # BUSI/ISIC 复用本入口的训练循环；根据数据集名选择统一的顶层目录。
    # 这里只改变保存位置，不改变模型、损失或训练计算。
    output_family = {
        "BUSI": "BUSI",
        "ISIC2017": "ISIC",
        "ISIC2018": "ISIC",
        # DSB18 和 EM 通过本通用二分类训练循环读取，但归档到独立 Cell 实验族。
        "DSB18": "Cell",
        "EM": "Cell",
    }.get(args.dataset_name, "Polyp")
    run_dir = make_experiment_dir(args, output_family)

    logging.basicConfig(
        filename=os.path.join(
            run_dir,
            "train.log",
        ),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger().addHandler(
        logging.StreamHandler(sys.stdout)
    )

    configuration = {
        **vars(args),
        "data_root": os.path.abspath(
            args.data_root
        ),
        "dataset_root": os.path.abspath(
            dataset_root
        ),
        "run_dir": run_dir,
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "train_manifest_sha256": (
            train_loader.dataset.manifest_sha256
        ),
        "val_manifest_sha256": (
            val_loader.dataset.manifest_sha256
        ),
        "command": " ".join(sys.argv),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }

    with open(
            os.path.join(run_dir, "config.json"),
            "w",
            encoding="utf-8",
    ) as stream:
        json.dump(
            configuration,
            stream,
            ensure_ascii=False,
            indent=2,
        )

    logging.info("args=%s", args)
    logging.info("device=%s", device)
    # 只观测首次完整训练更新后的峰值显存，不参与 loss、optimizer 或 checkpoint 决策。
    train_memory_logged = False
    reset_peak_memory(device)
    train_environment = log_environment(logging, device, prefix="BENCHMARK_TRAIN_ENV")
    logging.info(
        "train_images=%d val_images=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    model = build_model(
        args,
        pretrain=not args.no_pretrain,
    )

    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(
                "Checkpoint not found: {}".format(
                    args.checkpoint
                )
            )
        load_checkpoint(
            model,
            args.checkpoint,
        )

    model.to(device)

    if device.type == "cuda" and args.n_gpu > 1:
        available = torch.cuda.device_count()

        if args.n_gpu > available:
            raise RuntimeError(
                "Requested {} GPUs, but only {} are visible".format(
                    args.n_gpu,
                    available,
                )
            )

        model = nn.DataParallel(
            model,
            device_ids=list(
                range(args.n_gpu)
            ),
        )

    logging.info(
        "model_parameters=%d",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )

    # Polyp、BUSI 和 ISIC 共用此入口；复杂度统计只观察已构建模型一次。
    log_model_complexity(
        model,
        (1, 3, args.img_size, args.img_size),
        output_path=os.path.join(run_dir, "model_complexity.txt"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs,
            eta_min=args.min_lr,
        )

    scaler = GradScaler(
        enabled=args.amp and device.type == "cuda"
    )

    writer = SummaryWriter(
        os.path.join(run_dir, "tensorboard")
    )

    history_path = os.path.join(
        run_dir,
        "train_history.csv",
    )
    validation_path = os.path.join(
        run_dir,
        "validation_metrics.csv",
    )

    best_dice = float("-inf")
    best_epoch = 0
    global_step = 0
    scale_rates = (
        [1.0]
        if args.no_multi_scale
        else args.scale_rates
    )
    started = time.time()

    for epoch in range(
            1,
            args.max_epochs + 1,
    ):
        model.train()
        epoch_losses = []

        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc="epoch {}/{}".format(
                epoch,
                args.max_epochs,
            ),
        )

        for batch_index, (images, masks) in progress:
            if (
                    args.max_train_batches
                    and batch_index >= args.max_train_batches
            ):
                break

            images = images.to(
                device=device,
                dtype=torch.float32,
            )
            masks = masks.to(
                device=device,
                dtype=torch.float32,
            )

            for rate in scale_rates:
                scaled_images, scaled_masks = resized_batch(
                    images,
                    masks,
                    args.img_size,
                    float(rate),
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                with autocast(
                        enabled=scaler.is_enabled()
                ):
                    outputs, boundary_logits = model_outputs_with_boundary(
                        model,
                        scaled_images,
                        mode="train",
                    )

                    loss = supervised_structure_loss(
                        outputs,
                        scaled_masks,
                        args.supervision,
                    )
                    if boundary_logits is not None:
                        loss = loss + args.boundary_loss_weight * boundary_loss(boundary_logits, scaled_masks)
                    # 仅非 p1 配置追加融合监督；p1 消融保持原始 loss 完全不变。

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    args.clip,
                )

                scaler.step(optimizer)
                scaler.update()
                if not train_memory_logged:
                    synchronize(device)
                    log_json(logging, "BENCHMARK_TRAIN_PEAK_GPU_MEMORY", {
                        "measurement_policy": "peak after first complete forward/backward/optimizer step",
                        "dataset": args.dataset_name,
                        "batch_size": args.batch_size,
                        "input_size": args.img_size,
                        "amp": bool(args.amp),
                        "optimizer": "AdamW",
                        "environment": train_environment,
                        **peak_memory_mb(device),
                    })
                    train_memory_logged = True

                global_step += 1
                loss_value = float(
                    loss.item()
                )
                epoch_losses.append(loss_value)

                writer.add_scalar(
                    "train/loss",
                    loss_value,
                    global_step,
                )
                writer.add_scalar(
                    "train/lr",
                    optimizer.param_groups[0]["lr"],
                    global_step,
                )

            progress.set_postfix(
                loss="{:.4f}".format(
                    float(
                        np.mean(
                            epoch_losses[
                                -len(scale_rates):
                            ]
                        )
                    )
                )
            )

        if not epoch_losses:
            raise RuntimeError(
                "No Polyp training batches were processed"
            )

        train_loss = float(
            np.mean(epoch_losses)
        )
        learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        save_checkpoint(
            model,
            os.path.join(
                run_dir,
                "last.pth",
            ),
        )

        val_dice = ""
        val_iou = ""

        if (
                epoch % args.validate_every == 0
                or epoch == args.max_epochs
        ):
            val_rows, val_mean, _ = evaluate_loader(
                model=model,
                loader=val_loader,
                device=device,
                threshold=args.threshold,
                max_cases=args.max_valid_cases,
                output_dir=None,
                compute_surface=False,
                # 用 dataset_name 显示实际数据集，避免 Cell/BUSI/ISIC 训练时进度标题仍写 Polyp。
                description="{} val".format(args.dataset_name),
            )

            val_dice = val_mean["dice"]
            val_iou = val_mean["iou"]

            append_validation_rows(
                validation_path,
                epoch,
                val_rows,
            )

            writer.add_scalar(
                "val/dice",
                val_dice,
                epoch,
            )
            writer.add_scalar(
                "val/iou",
                val_iou,
                epoch,
            )

            logging.info(
                "epoch=%d train_loss=%.6f val_dice=%.6f val_iou=%.6f",
                epoch,
                train_loss,
                val_dice,
                val_iou,
            )

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch

                save_checkpoint(
                    model,
                    os.path.join(
                        run_dir,
                        "best.pth",
                    ),
                )

                with open(
                        os.path.join(
                            run_dir,
                            "best_validation.json",
                        ),
                        "w",
                        encoding="utf-8",
                ) as stream:
                    json.dump(
                        {
                            "epoch": best_epoch,
                            "val_dice": best_dice,
                            "val_iou": val_iou,
                        },
                        stream,
                        indent=2,
                    )

                logging.info(
                    "saved best.pth epoch=%d val_dice=%.6f",
                    best_epoch,
                    best_dice,
                )
        else:
            logging.info(
                "epoch=%d train_loss=%.6f",
                epoch,
                train_loss,
            )

        append_history(
            history_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "learning_rate": learning_rate,
                "elapsed_seconds": time.time() - started,
            },
        )

        writer.add_scalar(
            "train/epoch_loss",
            train_loss,
            epoch,
        )

        if args.save_every and (
                epoch % args.save_every == 0
                or epoch == args.max_epochs
        ):
            save_checkpoint(
                model,
                os.path.join(
                    run_dir,
                    "epoch_{}.pth".format(epoch),
                ),
            )

        if scheduler is not None:
            scheduler.step()

    writer.close()

    logging.info(
        "training finished best_epoch=%d best_val_dice=%.6f elapsed=%.2fs",
        best_epoch,
        best_dice,
        time.time() - started,
    )

    print("RUN_DIR=" + run_dir)
    print(
        "BEST_CHECKPOINT="
        + os.path.join(run_dir, "best.pth")
    )


if __name__ == "__main__":
    main()
