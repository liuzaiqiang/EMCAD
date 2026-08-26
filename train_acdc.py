# ============================== 初学者阅读总览 ==============================
# 本文件是 ACDC 心脏分割的独立训练入口，训练样本是二维切片，验证样本按病例重组成三维体。
# 类别固定为 4：0=背景，1=右心室(RV)，2=心肌(MYO)，3=左心室(LV)。
# 训练数据流：[B,1,H,W] 图像 + [B,H,W] 整数标签 -> EMCAD 四个 [B,4,H,W]
# logits -> mutation/deep/last supervision -> 0.3*CE+0.7*Dice -> AMP/反向传播。
# 验证数据流：[1,D,H,W] -> predict_volume 分批逐切片推理 -> [D,H,W] 类别预测 ->
# RV/MYO/LV Dice -> 病例均值 -> 全验证集均值 -> 选择 best.pth。
# 论文对应：整体网络见第 3.2 节与图 2，多阶段监督见第 3.3 节，ACDC 结果见第 4.2.3 节；
# 第 4.1 节论文设置为 224x224、400 epoch、batch size 12、AdamW、lr/wd=1e-4、
# 0.3*CE+0.7*Dice。当前脚本默认 150 epoch、batch size 6，因此默认命令不是论文原设置的逐项复刻。
# ========================================================================

# argparse 定义命令行接口。
import argparse
# csv 把每个验证病例、每个器官的 Dice 按 epoch 追加到表格。
import csv
# json 保存本次实验完整参数 config.json。
import json
# logging 写训练日志。
import logging
# os 处理数据、输出和 checkpoint 路径。
import os
# sys.stdout 用于把日志同步显示到终端。
import sys
# datetime 为自动 run_name 生成时间戳。
from datetime import datetime

# NumPy 负责验证均值、训练 epoch 平均损失及 worker 随机种子。
import numpy as np
# torch 提供模型训练、设备与权重序列化。
import torch
# nn 用于检测/包装 DataParallel。
import torch.nn as nn
# optim 提供 AdamW 优化器。
import torch.optim as optim
# SummaryWriter 记录 TensorBoard 标量。
from tensorboardX import SummaryWriter
# GradScaler/autocast 实现可选的 CUDA 自动混合精度。
from torch.cuda.amp import GradScaler, autocast
# 多类交叉熵处理每像素 4 类分类。
from torch.nn.modules.loss import CrossEntropyLoss
# DataLoader 负责切片批处理和病例迭代。
from torch.utils.data import DataLoader
# Compose 串联 ACDC 的训练增强/缩放变换。
from torchvision import transforms
# tqdm 展示训练和验证进度。
from tqdm import tqdm

# acdc_utils 汇总 ACDC 专用模型构建、损失、推理、指标、种子与 checkpoint 兼容逻辑。
from utils.acdc_utils import (
    # 三个前景类别名；本文件当前没有直接使用，是保留导入。
    ACDC_CLASS_NAMES,
    # 总类别数常量 4，包含背景。
    ACDC_NUM_CLASSES,
    # 多类 Dice loss 实现。
    DiceLoss,
    # 按 args 构建 num_classes=4 的 EMCADNet。
    build_model,
    # 兼容若干 checkpoint 字典格式并去除 module. 前缀。
    load_checkpoint,
    # 把模型单张量/元组输出统一转换为 logits 列表。
    model_outputs,
    # 对完整体数据按深度分批推理并拼回类别图。
    predict_volume,
    # 同时固定 Python、NumPy、PyTorch、CUDA 与 cuDNN 行为。
    seed_everything,
    # 按监督策略组合四级输出并计算 0.3*CE+0.7*Dice。
    supervised_loss,
    # 只计算 RV/MYO/LV 的病例 Dice。
    validation_dice,
)
# ACDCdataset 读取训练二维切片；ACDCVolumeDataset 读取/重组验证体；RandomGenerator 做同步增强。
from utils.dataset_ACDC import ACDCVolumeDataset, ACDCdataset, RandomGenerator


# 集中定义全部训练参数；函数返回 Namespace，不在 import 阶段直接解析命令行。
def parse_args():
    # description 会显示在 python train_ACDC.py --help 的标题中。
    parser = argparse.ArgumentParser(description="Train EMCAD on ACDC")
    # ACDC 根目录，预期含 train/、valid/ 等子目录。
    parser.add_argument("--root_path", default="./data/ACDC")
    # 列表目录，至少需要 train.txt 和 valid.txt。
    parser.add_argument("--list_dir", default="./data/ACDC/lists/lists_ACDC")
    # 所有 ACDC 实验目录的根位置。
    parser.add_argument("--output_dir", default="./model_pth/ACDC")
    # 单次实验名；None 时主函数用时间戳自动生成。
    parser.add_argument("--run_name", default=None)
    # 可选初始化 checkpoint；只加载模型参数，不恢复优化器/epoch。
    parser.add_argument("--checkpoint", default=None)

    # 编码器类型，默认论文主干之一 PVTv2-B2。
    parser.add_argument("--encoder", default="pvt_v2_b2")
    # MSDC 多尺度核；nargs='+' 允许传入 --kernel_sizes 1 3 5。
    parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[1, 3, 5])
    # MSCB 通道扩张倍数。
    parser.add_argument("--expansion_factor", type=int, default=2)
    # LGAG 的卷积核尺寸。
    parser.add_argument("--lgag_ks", type=int, default=3)
    # MSCB 激活函数名称。
    parser.add_argument("--activation_mscb", default="relu6")
    # 出现该旗标后关闭并行深度卷积，即 build_model 中 dw_parallel=False。
    parser.add_argument("--no_dw_parallel", action="store_true")
    # 出现该旗标后多尺度特征采用拼接，build_model 中 add=False。
    parser.add_argument("--concatenation", action="store_true")
    # 出现该旗标后不加载 ImageNet 编码器预训练权重。
    parser.add_argument("--no_pretrain", action="store_true")
    # PVT 预训练权重所在目录。
    parser.add_argument("--pretrained_dir", default="./pretrained_pth/pvt/")

    # 限制监督策略只能取三个已实现值，非法字符串会由 argparse 直接拒绝。
    parser.add_argument(
        # 命令行选项名。
        "--supervision",
        # mutation=15个非空组合；deep=4个单输出；last=只监督 P[-1]。
        choices=["mutation", "deep_supervision", "last_layer"],
        # 默认使用 mutation 监督。
        default="mutation",
        # 结束该参数定义。
    )
    # 训练/推理切片目标尺寸，默认 224x224。
    parser.add_argument("--img_size", type=int, default=224)
    # DataLoader 每批二维切片数；当前默认 6 与论文 ACDC 的 12 不同。
    parser.add_argument("--batch_size", type=int, default=6)
    # 训练轮数；当前默认 150 与论文 ACDC 的 400 不同。
    parser.add_argument("--max_epochs", type=int, default=150)
    # AdamW 初始学习率。
    parser.add_argument("--base_lr", type=float, default=0.0001)
    # AdamW 解耦权重衰减。
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    # 训练 DataLoader 子进程数；Windows 环境若多进程异常可显式设 0。
    parser.add_argument("--num_workers", type=int, default=4)
    # 请求 GPU 数；大于 1 时启用 DataParallel。
    parser.add_argument("--n_gpu", type=int, default=1)
    # 全局随机种子。
    parser.add_argument("--seed", type=int, default=2222)
    # 1 要求确定性，0 允许 cuDNN benchmark；在 main 中转为 bool。
    parser.add_argument("--deterministic", type=int, default=1)
    # 每隔多少 epoch 运行一次完整验证。
    parser.add_argument("--validate_every", type=int, default=1)
    # 每隔多少 epoch 保存 epoch_N.pth。
    parser.add_argument("--save_every", type=int, default=50)
    # 完整体推理时一次送入 GPU 的切片数，不等同于训练 batch_size。
    parser.add_argument("--inference_batch_size", type=int, default=8)
    # 大于 0 时每 epoch 只训练前 N 个 batch，主要用于 smoke test；0 表示不限制。
    parser.add_argument("--max_train_batches", type=int, default=0)
    # 大于 0 时验证只处理前 N 个病例，主要用于快速检查；0 表示全部。
    parser.add_argument("--max_valid_volumes", type=int, default=0)
    # 出现该旗标后在 CUDA 上启用自动混合精度；CPU 时 scaler 仍被禁用。
    parser.add_argument("--amp", action="store_true")
    # auto 自动选 CUDA/CPU，也可显式传 cpu、cuda、cuda:1 等 torch device 字符串。
    parser.add_argument("--device", default="auto")
    # 实际解析命令行并返回配置对象。
    return parser.parse_args()


# 把用户设备字符串转换为 torch.device。
def resolve_device(requested):
    # auto 优先 CUDA，不可用时回退 CPU。
    if requested == "auto":
        # torch.cuda.is_available() 同时检查 PyTorch 构建和驱动可用性。
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 显式值直接交给 torch.device 校验/解析。
    return torch.device(requested)


# 取得不带 DataParallel 外壳的 state_dict，保证单卡测试也能直接加载。
def get_state_dict(model):
    # 多卡模型参数实际位于 model.module；单卡则直接 model.state_dict()。
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


# 统一保存裸模型参数，不保存优化器、scaler、epoch 或随机状态。
def save_state_dict(model, path):
    # torch.save 使用 PyTorch 序列化格式写入指定 .pth。
    torch.save(get_state_dict(model), path)


# 把一次验证的所有病例行追加到 CSV；同一文件可累积多个 epoch。
def append_validation_csv(path, epoch, rows):
    # 列顺序固定，便于后续按列分析。
    fieldnames = ["epoch", "case_name", "RV_dice", "MYO_dice", "LV_dice", "mean_dice"]
    # 先判断文件是否存在；不存在时需要写表头。
    exists = os.path.isfile(path)
    # 追加模式 a 保留之前 epoch；newline='' 避免 Windows CSV 空行。
    with open(path, "a", newline="", encoding="utf-8") as stream:
        # DictWriter 按 fieldnames 顺序写字典。
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        # 仅首次创建时写列名。
        if not exists:
            # 写入 header 行。
            writer.writeheader()
        # rows 中每个字典代表一个验证病例。
        for row in rows:
            # 在病例指标前补上当前 epoch，再写为一行。
            writer.writerow({"epoch": epoch, **row})


# 执行一次完整验证，返回“所有病例 mean_dice 的平均值”，用于选择 best.pth。
def validate(args, model, device, csv_path, epoch):
    # valid 列表中的二维切片会由 ACDCVolumeDataset 按病例和 ED/ES 相位重新堆叠为 [D,H,W]。
    dataset = ACDCVolumeDataset(args.root_path, args.list_dir, split="valid")
    # 每批一个体数据；num_workers=0 避免 H5/NPZ 多进程与病例变长带来的复杂性。
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # 收集每个病例的器官 Dice 与病例均值，稍后写 CSV 并做总体平均。
    rows = []
    # 切换评估模式；下一 epoch 开头 main 会再次 model.train()。
    model.eval()
    # 逐病例验证，leave=False 表示完成后移除该内部进度条。
    for index, sampled in enumerate(tqdm(loader, desc="ACDC valid", leave=False)):
        # 调试上限：设为正数时只处理前 max_valid_volumes 个病例。
        if args.max_valid_volumes and index >= args.max_valid_volumes:
            # 达到上限后停止病例循环。
            break
        # DataLoader 增加的 batch 维为 1；取 [0] 后转回 NumPy，形状 [D,H,W]。
        image = sampled["image"][0].numpy()
        # 标签同样为 [D,H,W]，值域 0..3。
        label = sampled["label"][0].numpy()
        # case_name 是单元素字符串列表，取第一个名字。
        case_name = sampled["case_name"][0]
        # predict_volume 内部使用 no_grad，按深度批量送入模型并取最后输出 argmax。
        prediction = predict_volume(
            # 当前模型，可能是 DataParallel。
            model,
            # 三维 NumPy 图像 [D,H,W]。
            image,
            # 推理设备。
            device=device,
            # 每张切片的网络输入尺寸。
            img_size=args.img_size,
            # 一次处理多少张切片。
            batch_size=args.inference_batch_size,
            # 返回 [D,H,W] int16 类别图。
        )
        # 分别计算标签 1、2、3 的 Dice，返回 {1:...,2:...,3:...}。
        per_class = validation_dice(prediction, label)
        # 对 RV/MYO/LV 三个 Dice 做算术平均，得到该病例 mean Dice。
        mean_dice = float(np.mean(list(per_class.values())))
        # 追加结构化病例结果。
        rows.append(
            # 单病例字典开始。
            {
                # 病例/相位名。
                "case_name": case_name,
                # 标签 1 右心室 Dice。
                "RV_dice": per_class[1],
                # 标签 2 心肌 Dice。
                "MYO_dice": per_class[2],
                # 标签 3 左心室 Dice。
                "LV_dice": per_class[3],
                # 三器官平均 Dice。
                "mean_dice": mean_dice,
                # 单病例字典结束。
            }
            # append 调用结束。
        )
    # 若数据集为空或调试上限导致一例未处理，拒绝生成无意义的平均值。
    if not rows:
        # 显式抛错比 np.mean([])=NaN 更容易定位数据问题。
        raise RuntimeError("ACDC validation produced no volumes")
    # 将本 epoch 所有病例明细追加进 validation_metrics.csv。
    append_validation_csv(csv_path, epoch, rows)
    # 对病例 mean_dice 再取平均，作为验证集模型选择指标。
    return float(np.mean([row["mean_dice"] for row in rows]))


# ACDC 训练总控函数，负责从参数检查一直运行到打印 best checkpoint 路径。
def main():
    # 读取命令行配置。
    args = parse_args()
    # 防御性检查：ACDC 标签约定必须始终是背景+3个结构共 4 类。
    if ACDC_NUM_CLASSES != 4:
        # 常量被意外改动时立即停止，避免用错误输出通道静默训练。
        raise RuntimeError("ACDC must use four classes including background")

    # 列出训练开始前必须存在的目录和划分文件。
    required = [
        # 二维训练切片目录。
        os.path.join(args.root_path, "train"),
        # 二维验证切片目录，验证时会按病例重组。
        os.path.join(args.root_path, "valid"),
        # 训练切片清单。
        os.path.join(args.list_dir, "train.txt"),
        # 验证切片清单。
        os.path.join(args.list_dir, "valid.txt"),
        # 必需路径列表结束。
    ]
    # 过滤出不存在的路径。
    missing = [path for path in required if not os.path.exists(path)]
    # 任一缺失都停止，避免训练到 DataLoader 阶段才出现模糊报错。
    if missing:
        # 每个缺失路径占一行，便于直接补齐数据结构。
        raise FileNotFoundError("Missing ACDC paths:\n" + "\n".join(missing))

    # 一次性固定各随机源，并根据 deterministic 设置 cuDNN 行为。
    seed_everything(args.seed, bool(args.deterministic))
    # auto/cpu/cuda 字符串解析为实际设备。
    device = resolve_device(args.device)
    # 用户若显式要求 CUDA，但当前 PyTorch 检测不到 GPU，则给出明确错误。
    if device.type == "cuda" and not torch.cuda.is_available():
        # 防止后面 model.to(device) 才抛出更难读的底层异常。
        raise RuntimeError("CUDA was requested but is unavailable")

    # 未指定 run_name 时用当前时间生成唯一实验目录名。
    if args.run_name is None:
        # 时间格式精确到秒；同一秒并发启动仍可能重名。
        args.run_name = "acdc_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    # 本次实验目录=<output_dir>/<run_name>。
    snapshot_path = os.path.join(args.output_dir, args.run_name)
    # 递归创建目录；已存在时复用。
    os.makedirs(snapshot_path, exist_ok=True)

    # 配置文件日志，写入本实验 train.log。
    logging.basicConfig(
        # 日志文件路径。
        filename=os.path.join(snapshot_path, "train.log"),
        # INFO 及更高级别消息会被记录。
        level=logging.INFO,
        # 每条日志带时间和毫秒。
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        # 只显示时分秒，日期可由实验目录/文件时间辅助确定。
        datefmt="%H:%M:%S",
        # logging 配置结束。
    )
    # 同时把根 logger 输出到终端。
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # 保存全部参数的人类可读日志。
    logging.info("args=%s", args)
    # 保存实际运行设备。
    logging.info("device=%s", device)
    # 以写模式创建 config.json；这里是可复现实验的重要配置快照。
    with open(os.path.join(snapshot_path, "config.json"), "w", encoding="utf-8") as stream:
        # vars(args) 把 Namespace 转字典；ensure_ascii=False 保留可能的非 ASCII 路径。
        json.dump(vars(args), stream, ensure_ascii=False, indent=2)

    # 构造二维切片训练集；每个样本从 train/<name>.npz 读取 img 和 label。
    train_dataset = ACDCdataset(
        # 数据根目录。
        args.root_path,
        # train.txt 所在目录。
        args.list_dir,
        # 明确读取训练划分。
        split="train",
        # Compose 允许后续继续添加变换；当前仅一个 RandomGenerator。
        transform=transforms.Compose(
            # 同步随机旋转/翻转图像与标签，并缩放至 img_size x img_size。
            [RandomGenerator(output_size=[args.img_size, args.img_size])]
            # Compose 结束。
        ),
        # 数据集构造结束。
    )

    # 为每个 DataLoader worker 设置独立 NumPy 随机种子。
    def worker_init_fn(worker_id):
        # worker 0=seed，worker 1=seed+1，保证并行增强不会完全重复。
        np.random.seed(args.seed + worker_id)

    # 训练 DataLoader 把二维样本堆成 batch。
    trainloader = DataLoader(
        # 数据来源。
        train_dataset,
        # 每批切片数。
        batch_size=args.batch_size,
        # 每个 epoch 重排训练切片顺序。
        shuffle=True,
        # 并行取样进程数。
        num_workers=args.num_workers,
        # CUDA 时锁页内存，可配合异步传输；CPU 时关闭。
        pin_memory=device.type == "cuda",
        # 每个 worker 启动时固定随机状态。
        worker_init_fn=worker_init_fn,
        # DataLoader 构造结束。
    )
    # 记录切片总数和每 epoch 批次数。
    logging.info("train slices=%d batches=%d", len(train_dataset), len(trainloader))

    # 按结构参数创建 4 类 EMCADNet；pretrain=True 时加载编码器 ImageNet 权重。
    model = build_model(args, pretrain=not args.no_pretrain)
    # 若提供 checkpoint，则用其模型参数覆盖当前初始化。
    if args.checkpoint:
        # 只恢复模型 state_dict；优化器、scaler、global_step 和 best_dice 均重新开始。
        load_checkpoint(model, args.checkpoint)
    # 把模型移动到选定设备。
    model.to(device)
    # 仅 CUDA 且请求多于 1 张 GPU 时启用数据并行。
    if device.type == "cuda" and args.n_gpu > 1:
        # device_ids=[0,...,n_gpu-1] 明确限制使用前 n_gpu 张可见卡。
        model = nn.DataParallel(model, device_ids=list(range(args.n_gpu)))

    # 交叉熵负责像素级 4 类分类。
    rbce_loss = CrossEntropyLoss()
    # DiceLoss 内部 softmax，并对背景/RV/MYO/LV 四类求平均。
    dice_loss = DiceLoss(ACDC_NUM_CLASSES)
    # AdamW 更新全部可训练参数。
    optimizer = optim.AdamW(
        # 参数迭代器。
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay
        # 优化器构造结束；当前没有学习率 scheduler，学习率保持常数。
    )
    # 仅 args.amp 且 CUDA 时启用梯度缩放；否则 scaler 透明地按普通精度执行。
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    # TensorBoard 日志保存到本次实验的 tensorboard 子目录。
    writer = SummaryWriter(os.path.join(snapshot_path, "tensooard"))

    # 最好验证 Dice 从 -1 开始，确保第一次有效验证一定能保存 best.pth。
    best_dice = -1.0
    # 全局更新步数用于 TensorBoard x 轴。
    global_step = 0
    # 验证明细 CSV 的固定路径。
    valid_csv = os.path.join(snapshot_path, "validation_metrics.csv")

    # epoch 使用零基索引 0..max_epochs-1；日志和文件名通常再加 1 显示人类轮次。
    for epoch in range(args.max_epochs):
        # 每轮开头切回 train 模式；这也修复上一轮 validate 留下的 eval 状态。
        model.train()
        # 收集本 epoch 每个已处理 batch 的标量 loss，用于计算 epoch 均值。
        epoch_losses = []
        # 创建批次进度条，显示当前人类轮次 epoch+1/max_epochs。
        progress = tqdm(trainloader, desc="epoch {}/{}".format(epoch + 1, args.max_epochs))
        # 逐批读取字典；sampled['image']=[B,1,H,W]，sampled['label']=[B,H,W]。
        for batch_index, sampled in enumerate(progress):
            # smoke test 上限：正数 N 表示仅处理索引 0..N-1 的 N 个 batch。
            if args.max_train_batches and batch_index >= args.max_train_batches:
                # 达到限制后提前结束当前 epoch。
                break
            # 图像移动到设备并强制 float32；形状通常 [B,1,224,224]。
            images = sampled["image"].to(device=device, dtype=torch.float32)
            # 标签移动到设备并强制 int64，供 CrossEntropy 和 one_hot 使用；形状 [B,224,224]。
            labels = sampled["label"].to(device=device, dtype=torch.long)
            # set_to_none=True 释放/置空旧梯度，通常比逐元素清零更省内存。
            optimizer.zero_grad(set_to_none=True)
            # 进入自动混合精度上下文；scaler 未启用时等同普通前向。
            with autocast(enabled=scaler.is_enabled()):
                # EMCAD 训练前向并统一转成 list；典型 4 个输出均为 [B,4,224,224] logits。
                outputs = model_outputs(model, images, mode="train")
                # 按监督策略组合输出，每组计算 0.3*CE+0.7*Dice，再把所有组相加。
                loss = supervised_loss(
                    # 四级 logits 列表。
                    outputs,
                    # 整数像素标签。
                    labels,
                    # mutation/deep_supervision/last_layer。
                    supervision=args.supervision,
                    # 已构造的交叉熵对象。
                    ce_loss=ce_loss,
                    # 已构造的 4 类 DiceLoss 对象。
                    dice_loss=dice_loss,
                    # 损失调用结束，loss 为带梯度的标量 Tensor。
                )
            # AMP 时先按缩放因子放大 loss 再反向，降低 float16 梯度下溢风险；普通模式不缩放。
            scaler.scale(loss).backward()
            # 若本步梯度有效，scaler.step 内部反缩放并调用 optimizer.step；溢出时可跳过更新。
            scaler.step(optimizer)
            # 根据本步是否溢出动态调整下一步缩放因子。
            scaler.update()
            # 一个训练 batch 完成后全局步数加 1。
            global_step += 1
            # loss.item() 把 GPU 标量同步取回 Python，并追加到本 epoch 列表。
            epoch_losses.append(float(loss.item()))
            # 记录逐步训练损失。
            writer.add_scalar("train/loss", loss.item(), global_step)
            # 记录常数基础学习率；代码没有 scheduler。
            writer.add_scalar("train/lr", args.base_lr, global_step)
            # 在 tqdm 尾部显示当前 batch loss，保留 4 位小数。
            progress.set_postfix(loss="{:.4f}".format(loss.item()))

        # 如果数据集为空或 max_train_batches 配置使一批都未处理，则拒绝继续保存无效模型。
        if not epoch_losses:
            # 明确抛错定位训练数据/调试参数问题。
            raise RuntimeError("No ACDC training batches were processed")
        # 计算本 epoch 所有已处理 batch 的算术平均 loss。
        mean_loss = float(np.mean(epoch_losses))
        # 写入文本日志，epoch+1 使用一基轮次。
        logging.info("epoch=%d train_loss=%.6f", epoch + 1, mean_loss)
        # 写入 TensorBoard epoch 级 loss 曲线。
        writer.add_scalar("train/epoch_loss", mean_loss, epoch + 1)
        # 每轮覆盖 last.pth，始终代表最新完成 epoch 的裸模型参数。
        save_state_dict(model, os.path.join(snapshot_path, "last.pth"))

        # 达到验证间隔或最后一轮时执行完整病例验证。
        if (epoch + 1) % args.validate_every == 0 or epoch + 1 == args.max_epochs:
            # validate 返回病例宏平均后的总体 mean Dice，并把每病例明细追加到 CSV。
            mean_dice = validate(args, model, device, valid_csv, epoch + 1)
            # 记录验证指标。
            logging.info("epoch=%d validation_mean_dice=%.6f", epoch + 1, mean_dice)
            # 写入 TensorBoard 验证曲线。
            writer.add_scalar("valid/mean_dice", mean_dice, epoch + 1)
            # 当前结果大于或等于历史最好时更新；相等也会覆盖 best.pth。
            if mean_dice >= best_dice:
                # 更新内存中的最好分数。
                best_dice = mean_dice
                # 保存不带 module. 前缀的模型 state_dict，便于单卡测试加载。
                save_state_dict(model, os.path.join(snapshot_path, "best.pth"))
                # 记录本次最优保存事件。
                logging.info("saved best.pth validation_mean_dice=%.6f", best_dice)

        # 到达定期保存间隔或最后一轮时创建不会被覆盖的 epoch_N.pth。
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.max_epochs:
            # 调用跨单卡/多卡统一的保存函数。
            save_state_dict(
                # 当前模型。
                model,
                # 文件名使用一基轮次，例如 epoch_50.pth。
                os.path.join(snapshot_path, "epoch_{}.pth".format(epoch + 1)),
                # 保存调用结束。
            )

    # 训练全部完成后刷新并关闭 TensorBoard 文件。
    writer.close()
    # 记录全程最好验证 Dice。
    logging.info("training finished best_dice=%.6f", best_dice)
    # 打印 best.pth 的绝对路径，便于直接复制给 test_ACDC.py --checkpoint。
    print("BEST_CHECKPOINT=" + os.path.abspath(os.path.join(snapshot_path, "best.pth")))


# 只有直接运行本文件时启动训练，被 import 时只暴露辅助函数。
if __name__ == "__main__":
    # 进入训练总控。
    main()
