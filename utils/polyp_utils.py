import csv
import itertools
import os
import random

# OpenCV 只负责把预测二值图和可选概率图写入磁盘，不参与模型前向或损失计算。
import cv2
# NumPy 用于布尔掩膜统计、逐病例指标汇总以及有限值筛选。
import numpy as np
# torch/nn/F 分别提供张量、模型包装判断和二分类损失/插值等函数式接口。
import torch
import torch.nn as nn
import torch.nn.functional as F
# MedPy 在此文件中用于真正的 HD95 和 ASSD 表面距离计算。
from medpy import metric
# tqdm 仅包装数据加载器以显示评估进度。
from tqdm import tqdm

# EMCADNet 是编码器、EMCAD 解码器和四个分割头的总封装。
from lib.networks import EMCADNet


# 这些名称决定逐病例结果、均值和标准差中需要统一汇总的核心指标列。
# 前景像素数和表面距离是否有定义属于诊断字段，因此在写 CSV 时另外追加。
TEST_METRIC_NAMES = (
    "dice",
    "iou",
    "sensitivity",
    "specificity",
    "precision",
    "accuracy",
    "hd95",
    "assd",
)


# 固定 Python、NumPy、PyTorch CPU 与全部 CUDA 设备的随机数状态。
# deterministic=True 同时关闭 cuDNN benchmark，减少算法自动选择带来的运行差异；
# 这仍不等同于 torch.use_deterministic_algorithms(True)，所以不能保证所有算子绝对确定。
def seed_everything(seed, deterministic=True):
# Python random 影响由标准库完成的抽样或增强。
    random.seed(seed)
# NumPy 随机状态影响基于 NumPy 的划分、增强或采样。
    np.random.seed(seed)
# 设置当前 PyTorch 进程的 CPU 随机种子。
    torch.manual_seed(seed)
# 同时设置所有可见 CUDA 设备的随机种子，兼容单卡和 DataParallel。
    torch.cuda.manual_seed_all(seed)

# 确定性模式倾向于选择可复现的 cuDNN 实现。
    torch.backends.cudnn.deterministic = bool(deterministic)
# benchmark 会按输入测试最快算法，开启时可能在不同运行中选择不同实现。
    torch.backends.cudnn.benchmark = not bool(deterministic)


# 把命令行的设备字符串规范为 torch.device，并对显式 CUDA 请求做可用性检查。
def resolve_device(requested):
# auto 表示优先使用 CUDA；没有可用 CUDA 时自动回落到 CPU。
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

# 非 auto 值可以是 cpu、cuda、cuda:0 等 torch.device 支持的写法。
    device = torch.device(requested)

# 用户明确要求 CUDA 但当前 PyTorch 看不到 CUDA 时立即报错，避免静默改用 CPU。
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable"
        )

    return device


# 根据训练/测试参数创建二分类 EMCAD 网络。
# num_classes=1 表示每个像素只输出一个“前景”logit；背景概率隐式等于 1-sigmoid(logit)，
# 因而二分类任务不需要额外建立一个背景输出通道。
def build_model(args, pretrain):
    return EMCADNet(
# 二分类分割头的输出形状为 [B,1,H,W]。
        num_classes=1,
# MSDC 多尺度深度卷积核；论文主文 PDF 第6页/印刷第6页 §4.1 使用 [1,3,5]。
        kernel_sizes=args.kernel_sizes,
# MSCB 的通道扩张倍率；论文方法使用 2。
        expansion_factor=args.expansion_factor,
# --no_dw_parallel 被取反：默认采用论文实验所用的并行 MSDC。
        dw_parallel=not args.no_dw_parallel,
# --concatenation 被取反：默认把多尺度分支相加，对应论文 Eq.(5) 的求和形式。
        add=not args.concatenation,
# LGAG 的局部分组卷积核大小；论文默认是 3x3。
        lgag_ks=args.lgag_ks,
# 该激活参数主要传给 MSCB；EUCB 和 LGAG 各自仍使用其构造函数默认激活。
        activation=args.activation_mscb,
# 编码器可选 PVTv2 或仓库扩展的 ResNet；论文正式实验使用 PVTv2-B0/B2。
        encoder=args.encoder,
# 是否加载 ImageNet 预训练编码器权重，与编码器结构选择相互独立。
        pretrain=pretrain,
# PVTv2 本地预训练权重所在目录；ResNet 采用自身加载路径。
        pretrained_dir=args.pretrained_dir,
    )


# 把模型返回值统一规范成 Python list，供各种监督策略按输出索引组合。
# EMCADNet 当前返回顺序固定为 [p4,p3,p2,p1]：从最低分辨率解码级到最高分辨率解码级，
# 四个分割头的结果已经在 lib/networks.py 中分别上采样 32/16/8/4 倍到输入尺寸。
# 这里得到的仍然是未经过 Sigmoid 的 logits，不是概率图。
def model_outputs(model, images, mode="test"):
# mode 原样传给模型；当前 EMCADNet 的 train/test 分支返回内容相同，但保留该接口便于兼容其他模型。
    outputs = model(images, mode=mode)

# EMCAD 的四输出 list/tuple 被复制成普通 list，避免下游依赖具体容器类型。
    if isinstance(outputs, (list, tuple)):
        return list(outputs)

# 对单输出模型也包装成一元素列表，使监督和评估代码无需另写分支。
    return [outputs]


# 提取适合保存的参数字典；DataParallel 会在真实模型外再包一层 module。
def _model_state_dict(model):
# 避免把 DataParallel 自动添加的 "module." 前缀写入新检查点。
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()

# 非 DataParallel 模型直接返回自身 state_dict。
    return model.state_dict()


# 只保存模型参数，不附带优化器、调度器、epoch 或随机数状态。
def save_checkpoint(model, path):
    torch.save(_model_state_dict(model), path)


# 从磁盘加载模型参数，兼容纯 state_dict 和两种常见的外层字典格式。
def load_checkpoint(model, path):
# 先映射到 CPU，避免检查点记录的原 GPU 编号与当前机器不一致。
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

# 某些训练脚本把模型参数保存在 model_state_dict 键下。
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
# 兼容另一些框架常用的 state_dict 键。
    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]
# 否则假定文件本身就是“参数名 -> 张量”的映射。
    else:
        state_dict = checkpoint

# 非字典对象不可能直接作为 PyTorch 参数映射使用，因此显式拒绝。
    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "Unsupported checkpoint format: {}".format(path)
        )

# 兼容由 DataParallel 保存的旧检查点：只移除键开头的 "module."。
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }

# 当前待加载对象若仍由 DataParallel 包装，则把参数装入内部真实模型。
    target = (
        model.module
        if isinstance(model, nn.DataParallel)
        else model
    )

# strict=True 要求所有参数键和形状完全匹配，防止漏载层时静默继续。
    target.load_state_dict(
        state_dict,
        strict=True,
    )


# 论文对应：主文 PDF 第6页/印刷第6页 §4.1 说明二分类采用“加权 BCE + 加权 IoU”。
# 论文只给出损失类型，没有写出这里的 31x31 局部平均、边界权重系数 5、平滑常数 1；
# 这些数值属于仓库的具体实现细节。logits/mask 预期形状均为 [B,1,H,W]。
def structure_loss(logits, mask):
# 31x31 平均池化估计每个像素邻域内的前景比例，输出尺寸因 stride=1、padding=15 保持不变。
# abs(局部平均-mask) 在均匀区域接近 0，在前景/背景交界附近较大。
    weight = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(
            mask,
            kernel_size=31,
            stride=1,
            padding=15,
        )
        - mask
    )

# binary_cross_entropy_with_logits 在数值稳定的形式中内部合并 Sigmoid 与 BCE；
# 因此传入的必须是原始 logits，不能在调用前再手动做一次 Sigmoid。
# reduction="none" 保留逐像素损失，便于随后施加空间权重。
    weighted_bce = F.binary_cross_entropy_with_logits(
        logits,
        mask,
        reduction="none",
    )

# 对每张图分别计算空间加权平均：分子是加权损失总和，分母是权重总和。
# 这里只沿 H/W 求和，batch 和单输出通道维仍被保留。
    weighted_bce = (
        weight * weighted_bce
    ).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

# IoU 项需要 [0,1] 概率，所以 Sigmoid 在这里显式出现；这也是训练阶段概率化的位置。
    probability = torch.sigmoid(logits)

# 加权软交集：概率、0/1真值和空间权重逐像素相乘后沿 H/W 累加。
    intersection = (
        probability * mask * weight
    ).sum(dim=(2, 3))

# 先计算加权的 probability+mask；后面减去 intersection 得到软并集。
    union = (
        (probability + mask) * weight
    ).sum(dim=(2, 3))

# 1-IoU 是要最小化的损失；分子分母各加 1 是代码采用的平滑策略。
    weighted_iou = 1.0 - (
        (intersection + 1.0)
        / (union - intersection + 1.0)
    )

# 先把每张图的加权 BCE 与加权 IoU 相加，再对 batch/通道取均值。
    return (weighted_bce + weighted_iou).mean()


# 根据监督名称决定四个尺度 logits 怎样组合，再对每组调用 structure_loss。
# 论文定位：主文 PDF 第5页/印刷第5页 §3.3 与 Eq.(11)；Fig.2 位于主文第4页。
def supervised_structure_loss(outputs, mask, supervision):
# 正常 EMCAD 有四个输出，顺序为 [p4,p3,p2,p1]，索引 3 即最高分辨率解码头 p1。
    count = len(outputs)
# indices 通常为 [0,1,2,3]，后续每个内层列表代表一组需要先相加的 logits。
    indices = list(range(count))

# paper 是论文对二分类任务明确给出的五项加法监督。
    if supervision == "paper":
        # EMCAD 二分类论文设置：
        # 4 个单输出损失 + 4 个输出相加后的损失。
# 前四组分别是 [p4]、[p3]、[p2]、[p1]。
        groups = [[index] for index in indices]
# 第五组把四个 logits 逐元素相加，对应 Eq.(11) 的 L_(p1+p2+p3+p4)；五项权重均为 1。
        groups.append(indices)
# deep_supervision 只监督四个单头并把四项损失相加。
# 补充材料 PDF 第14页/补充印刷第3页 Table 10 只报告是否使用深监督，
# 未明确把“DS=Yes”定义成这个四单头模式，因此这里应理解为代码提供的独立实验选项。
    elif supervision == "deep_supervision":
        groups = [[index] for index in indices]
# last_layer 只使用列表最后一个输出，即代码中的 p1；推理阶段同样使用 outputs[-1]。
# 注意论文 §3.3 文字称 p4 为最终图，但 Fig.2 与当前代码都把最高分辨率末级输出命名为 p1。
    elif supervision == "last_layer":
        groups = [[count - 1]]
# mutation 枚举全部非空子集；四输出时共有 2^4-1=15 组。
# 论文在多分类任务中明确采用 MUTATION，本文件把它额外开放给二分类实验。
    elif supervision == "mutation":
        groups = [
# itertools.combinations 按组合长度 1、2、3、4 依次生成索引组，不包含空集。
            list(group)
            for length in range(1, count + 1)
            for group in itertools.combinations(
                indices,
                length,
            )
        ]
# 未知字符串会改变监督语义，因此直接报错而不静默回落。
    else:
        raise ValueError(
            "Unknown supervision: {}".format(supervision)
        )

# 在 mask 所在设备上创建标量零，确保后续累加不会发生 CPU/CUDA 设备冲突。
    loss = mask.new_tensor(0.0)

# 各组损失直接求和，不除以组数；所以 paper/deep/last/mutation 的原始损失量级不可横向比较。
    for group in groups:
# 组内先相加的是未激活 logits，而不是 Sigmoid 概率；随后 structure_loss 才负责概率化。
        logits = sum(outputs[index] for index in group)
# 每一组都计算完整的加权 BCE + 加权 IoU。
        loss = loss + structure_loss(logits, mask)

# 返回当前监督策略所有组合损失之和。
    return loss


def binary_metrics(prediction, target, compute_surface=True):
# 统一转为布尔数组：非零视为前景，零视为背景；后续逻辑不再使用概率值。
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

# 指标要求逐像素一一对应，尺寸不一致通常意味着恢复原图尺寸或数据配对出了问题。
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction/target shape mismatch: {} vs {}".format(
                prediction.shape,
                target.shape,
            )
        )

# TP：预测为前景且真值为前景的像素数。
    tp = int(
        np.logical_and(prediction, target).sum()
    )
# TN：预测为背景且真值为背景的像素数。
    tn = int(
        np.logical_and(~prediction, ~target).sum()
    )
# FP：把真实背景误判为前景的像素数。
    fp = int(
        np.logical_and(prediction, ~target).sum()
    )
# FN：把真实前景漏判为背景的像素数。
    fn = int(
        np.logical_and(~prediction, target).sum()
    )

# 预先整理各指标反复使用的分母，避免公式中重复拼接计数项。
    dice_denominator = 2 * tp + fp + fn
# IoU 并集像素数等于 TP+FP+FN。
    union = tp + fp + fn
# 真值前景总数，用作 sensitivity/recall 的分母。
    positive_target = tp + fn
# 真值背景总数，用作 specificity 的分母。
    negative_target = tn + fp
# 预测前景总数，用作 precision 的分母。
    positive_prediction = tp + fp
# 总像素数，用作 accuracy 的分母。
    total = tp + tn + fp + fn

# 补充材料 PDF 第12页/补充印刷第1页 Eq.(12) 给出 Dice 定义。
# 当预测和真值都没有前景时分母为 0；当前代码把这种完全一致的空掩膜记为 Dice=1。
    dice = (
        1.0
        if dice_denominator == 0
        else (2.0 * tp) / dice_denominator
    )

# 补充材料同页 Eq.(13) 给出 IoU；双方都为空时同样按完全匹配记为 1。
    iou = (
        1.0
        if union == 0
        else tp / union
    )

# sensitivity=TP/(TP+FN)。没有真实前景时，只有预测也为空才记为 1，否则记为 0。
# 这一空类别约定是评估代码细节，EMCAD 论文没有说明。
    if positive_target == 0:
        sensitivity = (
            1.0
            if positive_prediction == 0
            else 0.0
        )
    else:
        sensitivity = tp / positive_target

# specificity=TN/(TN+FP)；若图像根本没有背景像素，则当前实现返回 1。
    specificity = (
        1.0
        if negative_target == 0
        else tn / negative_target
    )

# precision=TP/(TP+FP)。没有预测前景时，根据真值是否也为空返回 1 或 0。
    if positive_prediction == 0:
        precision = (
            1.0
            if positive_target == 0
            else 0.0
        )
    else:
        precision = tp / positive_prediction

# accuracy=(TP+TN)/总像素数；空数组这一防御性分支返回 1。
    accuracy = (
        1.0
        if total == 0
        else (tp + tn) / total
    )

# 表面距离只有在判断前景集合是否为空后才能安全调用 MedPy。
    prediction_nonempty = bool(prediction.any())
    target_nonempty = bool(target.any())

# 双方均非空时距离正常定义；双方均为空时本实现也把距离定义为 0。
# 只有一方为空时没有可配对的表面，后面保留 NaN，并把该标志置 0。
    surface_defined = int(
        (
            prediction_nonempty
            and target_nonempty
        )
        or (
            not prediction_nonempty
            and not target_nonempty
        )
    )

# NaN 让汇总函数能够明确排除未定义的单边空掩膜，而不是伪造一个距离值。
    hd95 = float("nan")
    assd = float("nan")

# compute_surface=False 可跳过成本较高的边界距离，Dice/IoU 等区域指标仍会计算。
    if (
        compute_surface
        and prediction_nonempty
        and target_nonempty
    ):
# MedPy hd95 计算双向表面距离的第95百分位。
# 论文补充 Eq.(14) 的排版更接近普通最大 Hausdorff 距离；这里实现的是指标名称所指的真正 HD95。
        hd95 = float(
            metric.binary.hd95(
                prediction,
                target,
            )
        )
# ASSD 是预测表面和真值表面之间的双向平均对称表面距离，论文表格未把它列为主要二分类指标。
        assd = float(
            metric.binary.assd(
                prediction,
                target,
            )
        )
# 双方均为空时视作表面完全一致，距离置 0。
    elif (
        compute_surface
        and not prediction_nonempty
        and not target_nonempty
    ):
        hd95 = 0.0
        assd = 0.0

# 返回逐病例完整指标和诊断字段；所有区域指标显式转成 Python float，便于 CSV 序列化。
    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "accuracy": float(accuracy),
        "hd95": hd95,
        "assd": assd,
# compute_surface=False 时不声称表面距离已经计算。
        "surface_distance_defined": (
            surface_defined
            if compute_surface
            else 0
        ),
# 记录预测/真值前景面积，便于排查空预测、异常大掩膜或病例难度。
        "pred_foreground_pixels": int(
            prediction.sum()
        ),
        "gt_foreground_pixels": int(
            target.sum()
        ),
    }


# 从逐病例结果中提取某一列，删除 NaN/Inf 后调用指定统计函数。
def _finite_stat(rows, name, reducer):
# dtype=float64 统一数值类型，便于 np.isfinite 处理。
    values = np.asarray(
        [row[name] for row in rows],
        dtype=np.float64,
    )
# HD95/ASSD 在单边空掩膜时为 NaN，因此汇总前必须筛掉非有限值。
    values = values[np.isfinite(values)]

# 一列没有任何有效值时继续返回 NaN，明确表示“无法汇总”。
    if values.size == 0:
        return float("nan")

# reducer 通常是 np.mean 或 np.std，结果转成 Python float。
    return float(reducer(values))


# 计算所有病例的逐指标均值与标准差，生成可直接追加到 CSV 的两行。
def summarize_rows(rows):
# 空评估集通常意味着路径、数据过滤或 max_cases 设置错误，不能生成有意义的汇总。
    if not rows:
        raise RuntimeError(
            "Evaluation produced no images"
        )

# case_name 使用固定标识，便于人与脚本区分病例行和汇总行。
    mean_row = {"case_name": "MEAN"}
    std_row = {"case_name": "STD"}

# 核心指标逐列统计；_finite_stat 会忽略未定义的表面距离 NaN。
    for name in TEST_METRIC_NAMES:
        mean_row[name] = _finite_stat(
            rows,
            name,
            np.mean,
        )
# np.std 默认 ddof=0，这里报告的是当前病例集合的总体标准差而非无偏样本标准差。
        std_row[name] = _finite_stat(
            rows,
            name,
            np.std,
        )

# 均值行中的该字段实际保存“有定义的病例数量”，不是比例或算术均值。
    mean_row["surface_distance_defined"] = int(
        sum(
            row["surface_distance_defined"]
            for row in rows
        )
    )
# 对计数字段不计算标准差，CSV 中保留空字符串。
    std_row["surface_distance_defined"] = ""

# 前景像素数也报告均值和标准差，帮助解释指标波动是否来自目标面积差异。
    for name in (
        "pred_foreground_pixels",
        "gt_foreground_pixels",
    ):
        mean_row[name] = _finite_stat(
            rows,
            name,
            np.mean,
        )
        std_row[name] = _finite_stat(
            rows,
            name,
            np.std,
        )

# 返回两个与病例行字段兼容的字典。
    return mean_row, std_row


# 在完整数据加载器上执行二分类评估，并可选保存二值预测和归一化概率图。
# 论文定位：主文 PDF 第5页/印刷第5页 §3.3 说明二分类最终使用 Sigmoid；
# 但选择哪一个预测头、逐图 min-max 归一化和固定阈值均是当前代码的评估细节。
def evaluate_loader(
    model,
    loader,
    device,
    threshold=0.5,
    max_cases=0,
    output_dir=None,
    save_probabilities=False,
    compute_surface=True,
    description="Polyp evaluation",
):
# 只有请求输出目录时才创建文件夹；纯指标评估不会写预测图。
    if output_dir:
# 二值掩膜和概率可视化分开放置，避免下游混淆。
        mask_dir = os.path.join(
            output_dir,
            "predictions",
        )
        probability_dir = os.path.join(
            output_dir,
            "probabilities",
        )

# exist_ok=True 允许向已经存在的运行目录继续写入，但同名文件可能被 cv2.imwrite 覆盖。
        os.makedirs(mask_dir, exist_ok=True)

# 概率图仅在显式请求时建立目录。
        if save_probabilities:
            os.makedirs(
                probability_dir,
                exist_ok=True,
            )

# rows 按图像保存逐病例指标，循环结束后再统一计算均值和标准差。
    rows = []
# 评估模式关闭 Dropout 的随机失活，并让 BatchNorm 使用已学习的运行统计量。
    model.eval()

# 禁止构建反向传播图，降低显存占用和评估开销。
    with torch.no_grad():
# loader 每批提供网络尺寸图像、对应掩膜、原始高宽和文件名。
        for (
            images,
            targets,
            original_sizes,
            names,
        ) in tqdm(loader, desc=description):
# 图像移到目标设备并统一为 float32；目标在逐病例计算时仍从 CPU 数组读取。
            images = images.to(
                device=device,
                dtype=torch.float32,
            )

# model_outputs 返回 [p4,p3,p2,p1] 的未激活 logits；[-1] 选择最高分辨率解码头 p1。
# 论文 §3.3 文字称 p4 为最终输出，但 Fig.2 的尺度标注及仓库推理代码均实际选择 p1。
            logits = model_outputs(
                model,
                images,
                mode="test",
            )[-1]

# 一个 batch 内逐图恢复各自原始尺寸、计算指标和保存结果。
            for index, name in enumerate(names):
# max_cases=0 表示不限制；正数达到上限后停止继续处理。
                if max_cases and len(rows) >= max_cases:
                    break

# 数据加载器记录的原始高宽用于把统一训练/推理尺寸结果恢复到病例真实分辨率。
                height = int(
                    original_sizes[index, 0]
                )
                width = int(
                    original_sizes[index, 1]
                )

                # probability = torch.sigmoid(
                #     F.interpolate(
                #         logits[index:index + 1],
                #         size=(height, width),
                #         mode="bilinear",
                #         align_corners=False,
                #     )
                # )[0, 0].cpu().numpy()

                # target = (
                #     targets[index]
                #     .squeeze(0)
                #     .cpu()
                #     .numpy()
                #     >= 0.5
                # )
                # prediction = probability >= float(
                #     threshold
                # )
# 先在 logit 空间进行双线性插值，再做 Sigmoid；这与“先 Sigmoid 再插值”数值上不完全相同。
# align_corners=False 使用像素中心对齐规则，适合常规分割概率图尺寸恢复。
                probability = torch.sigmoid(
                    F.interpolate(
                        logits[index:index + 1],
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
# [0,0] 取当前单病例、单前景通道，并转为 CPU NumPy 概率数组。
                )[0, 0].cpu().numpy()

                # Match the official EMCAD Polyp evaluation:
                # per-image min-max normalization before thresholding.
# 这是逐图动态范围拉伸，不是模型 Sigmoid 的组成部分，也不是校准概率。
# EMCAD 本地论文没有给出该归一化公式；它会让同一个固定阈值具有“图内相对阈值”的含义。
                probability = (
                    probability - probability.min()
                ) / (
                    probability.max()
                    - probability.min()
                    + 1e-8
                )

# 数据加载器目标通常已是 0/1 浮点掩膜；>=0.5 将其稳健转换为布尔真值。
                target = (
                    targets[index]
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    >= 0.5
                )

# 对归一化后的单图概率应用固定阈值，得到送入 binary_metrics 的布尔预测。
                prediction = probability >= float(
                    threshold
                )






# 区域指标始终计算；HD95/ASSD 是否计算由 compute_surface 控制。
                metrics = binary_metrics(
                    prediction,
                    target,
                    compute_surface=compute_surface,
                )

# 保留文件名作为病例主键，并展开所有指标字段。
                rows.append(
                    {
                        "case_name": name,
                        **metrics,
                    }
                )

# 请求输出时把布尔预测转换为 OpenCV 可写的 0/255 单通道图像。
                if output_dir:
                    prediction_image = (
                        prediction.astype(np.uint8) * 255
                    )

# cv2.imwrite 返回布尔成功标志，因此不能只依赖“函数未抛异常”。
                    saved = cv2.imwrite(
                        os.path.join(
                            mask_dir,
                            name,
                        ),
                        prediction_image,
                    )

# 写盘失败立即报错，防止指标完成但预测文件缺失时仍宣称导出成功。
                    if not saved:
                        raise RuntimeError(
                            "Failed to save prediction: "
                            + name
                        )

# 概率图保存的是经过逐图 min-max 后的值，不是原始 Sigmoid 概率。
                    if save_probabilities:
# 乘255、裁剪并转uint8，得到只用于观察的8位灰度图；该量化结果不参与指标。
                        probability_image = np.clip(
                            probability * 255.0,
                            0,
                            255,
                        ).astype(np.uint8)

# 使用与二值掩膜相同的病例文件名写入 probabilities 子目录。
                        saved = cv2.imwrite(
                            os.path.join(
                                probability_dir,
                                name,
                            ),
                            probability_image,
                        )

# 概率图写入也单独检查，避免只成功保存二值掩膜。
                        if not saved:
                            raise RuntimeError(
                                "Failed to save probability: "
                                + name
                            )

# 达到全局 max_cases 后终止外层 batch 循环。
            if max_cases and len(rows) >= max_cases:
                break

# 汇总仅基于实际处理并加入 rows 的病例。
    mean_row, std_row = summarize_rows(rows)
# 同时返回逐病例数据和两类汇总，调用方可记录日志并写 CSV。
    return rows, mean_row, std_row


# 把逐病例、均值和标准差按统一字段顺序写入 UTF-8 CSV。
def write_metrics_csv(
    path,
    rows,
    mean_row,
    std_row,
):
# 先确保 CSV 的父目录存在；path 可以是相对路径或绝对路径。
    os.makedirs(
        os.path.dirname(os.path.abspath(path)),
        exist_ok=True,
    )

# 列顺序固定，保证不同实验生成的 CSV 可以直接纵向或横向比较。
    fieldnames = [
        "case_name",
        *TEST_METRIC_NAMES,
        "surface_distance_defined",
        "pred_foreground_pixels",
        "gt_foreground_pixels",
    ]

# newline="" 避免 Windows csv.writer 产生额外空行，UTF-8 保留非 ASCII 病例名。
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
# DictWriter 会按 fieldnames 从每个字典取值并序列化。
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
# 首行写字段名，随后依次写病例、均值和标准差。
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(mean_row)
        writer.writerow(std_row)
