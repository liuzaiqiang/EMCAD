# itertools.combinations 枚举 mutation supervision 中所有非空输出组合。
import itertools
# os 用于创建结果目录和拼接 NIfTI 输出路径。
import os
# random 用于统一设置 Python 随机种子。
import random

# NumPy 负责体数据分批、类别掩膜和指标均值计算。
import numpy as np
# PyTorch 负责模型输入、张量运算、检查点加载和推理设备管理。
import torch
# nn 提供自定义 DiceLoss 的 Module 基类。
import torch.nn as nn
# functional 提供 one_hot 与 interpolate 等无状态操作。
import torch.nn.functional as F
# MedPy 提供 ACDC 的 Dice、HD95、Jaccard 和 ASSD 指标。
from medpy import metric

# 复用项目统一的编码器 + EMCAD 解码器封装。
from lib.networks import EMCADNet

# ACDC 的网络输出共四类：背景、右心室、心肌、左心室。
ACDC_NUM_CLASSES = 4
# 三个前景类别的显示名称，顺序对应类别索引 1、2、3。
ACDC_CLASS_NAMES = ("RV", "MYO", "LV")
# 所有病例级评测字典统一使用的指标键顺序。
METRIC_NAMES = ("dice", "hd95", "jaccard", "asd")


# 同时固定 Python、NumPy、CPU/CUDA PyTorch 随机源，并设置 cuDNN 可复现策略。
def seed_everything(seed, deterministic=True):
    # 固定 Python 标准库随机序列。
    random.seed(seed)
    # 固定 NumPy 随机序列。
    np.random.seed(seed)
    # 固定 PyTorch CPU 及当前通用随机状态。
    torch.manual_seed(seed)
    # 固定所有可见 CUDA 设备的随机序列。
    torch.cuda.manual_seed_all(seed)
    # True 时强制 cuDNN 优先选择确定性实现。
    torch.backends.cudnn.deterministic = bool(deterministic)
    # benchmark 会按输入搜索最快算法，通常与严格确定性设置相反。
    torch.backends.cudnn.benchmark = not bool(deterministic)


# 根据命令行参数构造 ACDC 四分类 EMCADNet。
def build_model(args, pretrain):
    # 直接返回统一模型实例；本函数只集中参数映射，不改变网络结构。
    return EMCADNet(
        # ACDC 固定输出 4 个互斥类别 logits。
        num_classes=ACDC_NUM_CLASSES,
        # MSDC 各深度卷积分支的卷积核尺寸。
        kernel_sizes=args.kernel_sizes,
        # MSCB 中第一次 1x1 卷积的通道扩张倍数。
        expansion_factor=args.expansion_factor,
        # 命令行 no_dw_parallel=True 时切换为串行多尺度深度卷积。
        dw_parallel=not args.no_dw_parallel,
        # concatenation=False 时采用论文默认的逐分支相加聚合。
        add=not args.concatenation,
        # LGAG 分组卷积的空间核大小。
        lgag_ks=args.lgag_ks,
        # MSCB 使用的激活函数名称。
        activation=args.activation_mscb,
        # 选择 PVTv2 或 ResNet 编码器。
        encoder=args.encoder,
        # 控制是否加载编码器预训练权重，与编码器结构选择相互独立。
        pretrain=pretrain,
        # 本地 PVT 权重目录。
        pretrained_dir=args.pretrained_dir,
    )


# 把模型输出规范为列表，统一处理单输出与 EMCAD 多尺度输出。
def model_outputs(model, images, mode="test"):
    # mode 传给 EMCADNet，决定返回训练监督输出还是测试输出。
    outputs = model(images, mode=mode)
    # 已是 list/tuple 时复制为列表；单张量则包成单元素列表。
    return list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]


# ACDC 多类别 soft Dice 损失，直接使用 PyTorch one_hot 实现。
class DiceLoss(nn.Module):
    # num_classes 应与 logits 通道数一致，ACDC 中为 4。
    def __init__(self, num_classes):
        # 初始化 Module 内部参数/缓冲区注册机制。
        super().__init__()
        # 保存类别数。
        self.num_classes = num_classes

    # logits: [B,C,H,W]，target: [B,H,W]。
    def forward(self, logits, target):
        # softmax 在互斥类别维上把 logits 转成逐像素概率。
        probabilities = torch.softmax(logits, dim=1)
        # one_hot 初始布局为 [B,H,W,C]。
        target_one_hot = F.one_hot(
            # 标签转 long，并显式声明输出类别数以保留缺失类别通道。
            target.long(), num_classes=self.num_classes
            # 把类别维移到通道位置，得到 [B,C,H,W]，再转浮点。
        ).permute(0, 3, 1, 2).float()
        # 对 batch、高、宽求和，保留每个类别各自 Dice。
        dims = (0, 2, 3)
        # 每类软交集。
        intersection = torch.sum(probabilities * target_one_hot, dim=dims)
        # 每类预测概率平方和与独热标签平方和。
        denominator = torch.sum(
            # 平方形式与仓库 Synapse DiceLoss 的定义保持一致。
            probabilities * probabilities + target_one_hot * target_one_hot,
            # 只在 batch 和空间维归约。
            dim=dims,
        )
        # 平滑项 1e-5 防止空类别导致 0/0。
        dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
        # 对四类 Dice 求均值并转换成最小化损失。
        return 1.0 - dice.mean()


# 根据监督策略返回需要相加后计算损失的输出索引组。
def _supervision_groups(output_count, supervision):
    # 输出索引通常为 [0,1,2,3]，对应从低到高分辨率的四个预测头。
    indices = list(range(output_count))
    # last_layer 只监督最后的最高分辨率输出。
    if supervision == "last_layer":
        # output_count-1 是末输出索引。
        return [[output_count - 1]]
    # deep_supervision 分别对每个尺度单独计算损失。
    if supervision == "deep_supervision":
        # 每个内层列表只有一个输出索引。
        return [[index] for index in indices]
    # mutation 枚举所有非空输出子集，并先把组内 logits 相加再监督。
    if supervision == "mutation":
        # 返回二维列表，例如四输出时共有 2^4-1=15 个组合。
        return [
            # combinations 返回元组，转换成列表供下游索引。
            list(group)
            # 组合长度从 1 到全部输出数，明确排除空集。
            for length in range(1, output_count + 1)
            # 枚举当前长度的所有不重复索引组合。
            for group in itertools.combinations(indices, length)
        ]
    # 未知字符串会导致训练含义不明确，因此显式报错。
    raise ValueError("Unknown supervision: " + supervision)


# 按指定监督策略累计交叉熵与 Dice 混合损失。
def supervised_loss(outputs, target, supervision, ce_loss, dice_loss):
    # 在 target 所在设备创建标量浮点零，避免 CPU/CUDA 设备不一致。
    total = target.new_tensor(0.0, dtype=torch.float32)
    # 遍历 last/deep/mutation 策略生成的每个输出组合。
    for group in _supervision_groups(len(outputs), supervision):
        # 将组合中的多尺度 logits 逐元素相加；各输出已由 EMCADNet 上采样到同一尺寸。
        logits = sum(outputs[index] for index in group)
        # 交叉熵权重 0.3，监督离散互斥类别。
        total = total + 0.3 * ce_loss(logits, target.long())
        # Dice 权重 0.7，直接优化区域重叠。
        total = total + 0.7 * dice_loss(logits, target)
    # 返回所有监督组合损失之和，原实现没有再除以组合数量。
    return total


# 兼容多种常见检查点包装格式并严格加载到模型。
def load_checkpoint(model, checkpoint_path):
    # 先映射到 CPU，避免保存时 GPU 编号与当前机器不一致。
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # 兼容包含 model_state_dict 键的训练快照。
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # 取出真正参数字典。
        state_dict = checkpoint["model_state_dict"]
    # 兼容包含通用 state_dict 键的快照。
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # 取出参数字典。
        state_dict = checkpoint["state_dict"]
    # 否则假设 checkpoint 自身就是参数名到张量的映射。
    else:
        # 不额外包装或转换值。
        state_dict = checkpoint
    # DataParallel 保存的键以 module. 开头，这里统一去掉该前缀。
    state_dict = {
        # 仅对确实以 module. 开头的键切掉前七个字符。
        key[7:] if key.startswith("module.") else key: value
        # 保留每个参数张量 value 不变。
        for key, value in state_dict.items()
    }
    # strict=True 要求模型全部参数键与检查点完全匹配，防止静默漏载。
    model.load_state_dict(state_dict, strict=True)


# 对一个 [D,H,W] ACDC 体数据按切片批量推理并重组为三维类别图。
def predict_volume(model, image, device, img_size, batch_size=8):
    # 统一为 float32 NumPy，避免后续 from_numpy 使用不受支持或过高精度类型。
    image = np.asarray(image, dtype=np.float32)
    # 本函数只接受深度、高、宽三维体。
    if image.ndim != 3:
        # 报错时显示实际 shape，便于定位切片/通道布局问题。
        raise ValueError("Expected [D,H,W] volume, got {}".format(image.shape))
    # 记录原始深度和空间尺寸。
    depth, height, width = image.shape
    # 暂存每个深度批次的预测类别数组。
    predictions = []
    # 使用 BatchNorm/Dropout 的评估行为。
    model.eval()
    # 完整体推理不需要构建梯度图。
    with torch.no_grad():
        # 沿深度轴按 batch_size 切分，减少显存占用。
        for start in range(0, depth, batch_size):
            # 取 [b,H,W] 切片批次并增加单通道维成 [b,1,H,W]。
            batch = torch.from_numpy(image[start:start + batch_size]).unsqueeze(1)
            # 搬到指定 CPU/GPU，并确保 float32。
            batch = batch.to(device=device, dtype=torch.float32)
            # 原始 ACDC 切片尺寸不等于训练尺寸时先缩放输入。
            if (height, width) != (img_size, img_size):
                # 对连续图像张量使用双线性插值。
                batch = F.interpolate(
                    # 输入 [b,1,H,W]。
                    batch,
                    # 目标为正方形训练尺寸。
                    size=(img_size, img_size),
                    # bilinear 适合二维连续灰度图。
                    mode="bilinear",
                    # align_corners=False 采用 PyTorch 常用的像素中心对齐策略。
                    align_corners=False,
                )
            # 统一模型输出为列表并选最后的最终分割 logits。
            logits = model_outputs(model, batch, mode="test")[-1]
            # 如果模型最终输出仍不是原始体数据空间尺寸，则把连续 logits 缩回原尺寸。
            if logits.shape[-2:] != (height, width):
                # 在 argmax 前插值 logits，避免对离散类别图做不合理的双线性插值。
                logits = F.interpolate(
                    # 输入多类别 logits [b,C,h,w]。
                    logits,
                    # 恢复原始 H、W。
                    size=(height, width),
                    # 每个类别通道独立双线性插值。
                    mode="bilinear",
                    # 使用默认像素中心几何约定。
                    align_corners=False,
                )
            # 在类别通道取最大 logit，转 CPU NumPy并加入深度批次列表。
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    # 沿深度轴拼接批次，最终以 int16 返回 [D,H,W] 类别编号体。
    return np.concatenate(predictions, axis=0).astype(np.int16)


# 计算单个二值类别的四项病例级指标，并返回具名字典。
def calculate_metric_percase(prediction, target, voxelspacing=None):
    """Local copy of the existing Synapse metric policy; Synapse stays untouched."""
    # 强制转换为布尔预测掩膜；非零位置视为前景。
    prediction = np.asarray(prediction).astype(bool)
    # 强制转换为布尔真值掩膜。
    target = np.asarray(target).astype(bool)
    # 预测和真值均含前景时，表面距离指标才可正常计算。
    if prediction.any() and target.any():
        # 返回固定四键字典，便于按名称汇总。
        return {
            # Dice 区域重叠系数。
            "dice": float(metric.binary.dc(prediction, target)),
            # HD95 使用可选 voxelspacing 将像素距离换算到物理空间。
            "hd95": float(
                # 调用 MedPy 的 95% Hausdorff 距离。
                metric.binary.hd95(
                    # 传入二值预测、真值和空间间距。
                    prediction, target, voxelspacing=voxelspacing
                )
            ),
            # Jaccard/IoU 区域重叠系数。
            "jaccard": float(metric.binary.jc(prediction, target)),
            # 平均对称表面距离，同样考虑可选体素间距。
            "asd": float(
                # MedPy 名称 assd 对应 average symmetric surface distance。
                metric.binary.assd(
                    # 传入二值掩膜及体素物理间距。
                    prediction, target, voxelspacing=voxelspacing
                )
            ),
        }
    # 延续仓库现有 Synapse 指标策略：预测有前景但真值为空时返回这组值。
    if prediction.any() and not target.any():
        # 这里只解释原始策略，不改变其数学含义或数值。
        return {"dice": 1.0, "hd95": 0.0, "jaccard": 1.0, "asd": 0.0}
    # 其余空前景组合统一返回四项零值。
    return {"dice": 0.0, "hd95": 0.0, "jaccard": 0.0, "asd": 0.0}


# 对完整多类别体数据分别计算每个前景类别的病例级指标。
def volume_metrics(prediction, target, num_classes=ACDC_NUM_CLASSES, voxelspacing=None):
    # 返回类别索引到四指标字典的映射。
    return {
        # 将第 class_index 类转换为二值问题后计算指标。
        class_index: calculate_metric_percase(
            # 当前类别预测掩膜。
            prediction == class_index,
            # 当前类别真实掩膜。
            target == class_index,
            # 传递物理体素间距给距离指标。
            voxelspacing=voxelspacing,
        )
        # 从 1 开始跳过背景类别 0。
        for class_index in range(1, num_classes)
    }


# 把一个病例的逐前景类别指标平均成四个宏平均值。
def mean_metrics(per_class):
    # 返回与 METRIC_NAMES 同键的均值字典。
    return {
        # 对所有前景类别的同名指标取算术平均。
        name: float(np.mean([values[name] for values in per_class.values()]))
        # 固定迭代 dice、hd95、jaccard、asd。
        for name in METRIC_NAMES
    }


# 验证阶段只计算三个前景类别的 Dice，供选择最佳检查点。
def validation_dice(prediction, target, num_classes=ACDC_NUM_CLASSES):
    # 保存类别索引到 Dice 的映射。
    values = {}
    # ACDC 遍历 1=RV、2=MYO、3=LV，跳过背景。
    for class_index in range(1, num_classes):
        # 构造当前类别预测布尔掩膜。
        pred_mask = np.asarray(prediction == class_index).astype(bool)
        # 构造当前类别真值布尔掩膜。
        target_mask = np.asarray(target == class_index).astype(bool)
        # 双方都有前景时正常计算 Dice。
        if pred_mask.any() and target_mask.any():
            # 转 float 便于 JSON/日志序列化。
            values[class_index] = float(metric.binary.dc(pred_mask, target_mask))
        # 延续项目既有空类策略：仅预测非空、真值为空记为 1。
        elif pred_mask.any() and not target_mask.any():
            # 保存固定值。
            values[class_index] = 1.0
        # 其他空类组合记为 0。
        else:
            # 保存固定值。
            values[class_index] = 0.0
    # 返回三个前景类别的 Dice 字典。
    return values


# 将 ACDC 图像、预测、真值三个体数据保存为同空间间距的 NIfTI 文件。
def save_nifti_triplet(image, prediction, target, output_dir, case_name, z_spacing):
    # 延迟导入 SimpleITK，使不需要保存 NIfTI 的训练流程无需在模块导入阶段使用它。
    import SimpleITK as sitk

    # 创建输出目录及缺失父目录；已有目录不会报错。
    os.makedirs(output_dir, exist_ok=True)
    # 依次处理原图、预测和真值，统一写盘逻辑。
    for suffix, array in (
            # 原始图像后缀 img。
            ("img", image),
            # 模型预测后缀 pred。
            ("pred", prediction),
            # 人工真值后缀 gt。
            ("gt", target),
    ):
        # 转 float32 NumPy 后从 [D,H,W] 构造 SimpleITK 图像。
        itk_image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
        # x、y 间距设为 1，z 使用调用方给出的切片间距。
        itk_image.SetSpacing((1.0, 1.0, float(z_spacing)))
        # 以压缩 NIfTI 格式写入磁盘。
        sitk.WriteImage(
            # 要写出的 SimpleITK 图像对象。
            itk_image,
            # 文件名形如 case001_pred.nii.gz。
            os.path.join(output_dir, "{}_{}.nii.gz".format(case_name, suffix)),
        )
