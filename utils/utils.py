
# PyTorch 张量与设备操作。
import torch
# nn 提供 DiceLoss 继承的 Module 基类。
import torch.nn as nn
# NumPy 用于体数据推理、类别掩膜和指标数组处理。
import numpy as np
# MedPy 提供医学分割常用的 Dice、HD95、Jaccard 和平均表面距离实现。
from medpy import metric
# scipy.zoom 在二维切片推理前后进行连续图像和离散预测的尺寸变换。
from scipy.ndimage import zoom
# seaborn 当前未在活动代码中调用；保留原导入。
import seaborn as sns
# PIL.Image 只在下方保留的可视化注释代码中引用。
from PIL import Image 
# matplotlib 用于掩膜叠加图的绘制/保存依赖链。
import matplotlib.pyplot as plt
# overlay_masks 把多类别布尔掩膜以颜色覆盖在原始 CT 切片上。
from segmentation_mask_overlay import overlay_masks
# CSS4_COLORS 为各器官掩膜选择可读颜色。
import matplotlib.colors as mcolors

# SimpleITK 把 NumPy 体数据写成带空间间距的 NIfTI 文件。
import SimpleITK as sitk
# pandas 当前未在活动代码中调用；保留原导入。
import pandas as pd

# THOP profile 统计一次前向传播的计算量和参数量。
from thop import profile
# clever_format 把大数转换成 K/M/G 等可读单位。
from thop import clever_format
# ptflops 提供另一套逐层 MACs 和参数量统计接口。
from ptflops import get_model_complexity_info

# 递归生成 seq 的全部子集；训练器用它构造 mutation supervision 的输出组合。
def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    # 空序列或单元素序列是递归终止条件。
    if len(seq) <= 1:
        # 先产生序列自身；单元素时即包含该元素的子集。
        yield seq
        # 再产生空集，保证幂集定义完整。
        yield []
    # 多元素序列递归计算尾部 seq[1:] 的幂集。
    else:
        # item 依次代表不含首元素的每个尾部子集。
        for item in powerset(seq[1:]):
            # 在 item 前加入 seq[0]，得到包含首元素的对应子集。
            yield [seq[0]]+item
            # 同时产生不包含首元素的原 item。
            yield item

# 按元素把所有参数梯度截断到 [-grad_clip, grad_clip]，避免极端梯度值。
def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    # 优化器可能包含多个具有不同超参数的参数组。
    for group in optimizer.param_groups:
        # 遍历当前组里的每个可训练参数。
        for param in group['params']:
            # 未参与当前计算图的参数 grad 为 None，必须跳过。
            if param.grad is not None:
                # clamp_ 原地截断梯度张量；这不是全局范数裁剪。
                param.grad.data.clamp_(-grad_clip, grad_clip)

# 按固定 epoch 周期计算学习率衰减因子并作用到优化器参数组。
def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    # 每经过 decay_epoch 个 epoch，指数 floor 值加 1，decay 再乘一次 decay_rate。
    decay = decay_rate ** (epoch // decay_epoch)
    # 对优化器中的每个参数组更新学习率。
    for param_group in optimizer.param_groups:
        # 原实现使用 *=，如果每个 epoch 都调用会在现有 lr 上继续累乘；init_lr 参数未直接使用。
        param_group['lr'] *= decay

# 维护标量的当前值、累计平均值和最近若干次记录。
class AvgMeter(object):
    # num 决定 show() 最多平均最近多少个张量值。
    def __init__(self, num=40):
        # 保存滑动显示窗口大小。
        self.num = num
        # 统一初始化所有统计字段。
        self.reset()

    # 清空累计状态，开始新的统计区间。
    def reset(self):
        # 最近一次传入值。
        self.val = 0
        # 从 reset 起的加权平均值。
        self.avg = 0
        # 加权总和。
        self.sum = 0
        # 累计样本权重。
        self.count = 0
        # 保存每次 val，供 show() 计算近期张量均值。
        self.losses = []

    # 用值 val 和权重/样本数 n 更新统计量。
    def update(self, val, n=1):
        # 记录当前批次值。
        self.val = val
        # 累加 val 对 n 个样本的贡献。
        self.sum += val * n
        # 累加样本数。
        self.count += n
        # 更新全历史加权平均。
        self.avg = self.sum / self.count
        # 保存原始 val；通常是标量 PyTorch 张量。
        self.losses.append(val)

    # 返回最近 num 个记录的均值，用于显示更平滑的短期趋势。
    def show(self):
        # 起点不小于 0；torch.stack 要求 losses 中各张量形状一致。
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))

# 使用 THOP 计算给定 model 和实际 input_tensor 的 FLOPs/参数量并打印。
def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    # profile 会执行模型前向钩子统计运算，inputs 必须是元组。
    flops, params = profile(model, inputs=(input_tensor,))
    # 将原始整数统计转换为例如 12.345G、26.789M 的字符串。
    flops, params = clever_format([flops, params], "%.3f")
    # 输出统一格式的模型复杂度摘要。
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
    
# 把整数类别标签 [B,H,W] 转成独热张量 [B,C,H,W]。
def one_hot_encoder(input_tensor,dataset,n_classes = None):
    # 暂存每个类别的单通道布尔掩膜。
    tensor_list = []
    # MMWHS 原始标签不是连续 0..C-1，而是特定灰度编码。
    if dataset == 'MMWHS':  
        # 这些值依次代表 MMWHS 的背景/解剖类别编码；变量名 dict 沿用原实现。
        dict = [0,205,420,500,550,600,820,850]
        # 针对每个原始标签值构造一个通道。
        for i in dict:
            # 相等比较得到 [B,H,W] 布尔掩膜。
            temp_prob = input_tensor == i  
            # 在通道位置 1 增维成 [B,1,H,W] 并加入列表。
            tensor_list.append(temp_prob.unsqueeze(1))
        # 沿通道维拼接全部类别掩膜。
        output_tensor = torch.cat(tensor_list, dim=1)
        # 损失计算需要浮点 0/1，而非 bool。
        return output_tensor.float()
    # 其他数据集假定类别编号连续为 0..n_classes-1。
    else:
        # 遍历每个连续类别索引。
        for i in range(n_classes):
            # 得到当前类别的布尔掩膜。
            temp_prob = input_tensor == i  
            # 增加类别通道维。
            tensor_list.append(temp_prob.unsqueeze(1))
        # 合并为完整独热编码。
        output_tensor = torch.cat(tensor_list, dim=1)
        # 转浮点返回。
        return output_tensor.float()    

# 多类别 soft Dice 损失；训练器将它与交叉熵按 0.7/0.3 加权。
class DiceLoss(nn.Module):
    # n_classes 必须等于网络输出 logits 的通道数。
    def __init__(self, n_classes):
        # 初始化 nn.Module 内部状态。
        super(DiceLoss, self).__init__()
        # 保存类别数供独热编码和逐类循环使用。
        self.n_classes = n_classes

    # 类内部的连续标签到独热标签转换。
    def _one_hot_encoder(self, input_tensor):
        # 存放各类别 [B,1,H,W] 掩膜。
        tensor_list = []
        # 类别索引包含背景 0。
        for i in range(self.n_classes):
            # 比较得到属于第 i 类的位置。
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            # 增加通道维后加入列表。
            tensor_list.append(temp_prob.unsqueeze(1))
        # 拼接成与网络输出相同的 [B,C,H,W] 布局。
        output_tensor = torch.cat(tensor_list, dim=1)
        # 转成浮点参与乘法和求和。
        return output_tensor.float()

    # 计算单个类别通道的 Dice loss。
    def _dice_loss(self, score, target):
        # 确保目标掩膜与概率 score 使用兼容浮点类型。
        target = target.float()
        # 平滑项防止预测和标签都为空时分母为 0。
        smooth = 1e-5
        # 概率与目标逐元素乘积之和，对应软交集。
        intersect = torch.sum(score * target)
        # 标签平方和；二值目标时等于前景像素数量。
        y_sum = torch.sum(target * target)
        # 预测概率平方和。
        z_sum = torch.sum(score * score)
        # 计算 soft Dice 系数：2*交集 / 两侧能量和。
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        # 从相似度转为最小化目标。
        loss = 1 - loss
        # 返回标量张量并保留梯度。
        return loss

    # inputs 是 logits [B,C,H,W]，target 是类别索引 [B,H,W]。
    def forward(self, inputs, target, weight=None, softmax=False):
        # 训练调用 softmax=True 时，把互斥类别 logits 转成逐像素概率。
        if softmax:
            # dim=1 是类别通道维。
            inputs = torch.softmax(inputs, dim=1)
        # 把整数标签转换为与 inputs 同形状的独热张量。
        target = self._one_hot_encoder(target)
        # 未提供类别权重时，各类等权。
        if weight is None:
            # 创建长度为 C 的 Python 权重列表。
            weight = [1] * self.n_classes
        # 在逐类计算前严格检查 batch、通道和空间形状一致。
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        # 保存每类 Dice 数值用于调试；当前函数最终未返回该列表。
        class_wise_dice = []
        # 初始化总损失为浮点零。
        loss = 0.0
        # 包含背景在内逐类计算损失。
        for i in range(0, self.n_classes):
            # 取第 i 个概率通道和独热目标通道。
            dice = self._dice_loss(inputs[:, i], target[:, i])
            # dice 变量实际是 loss，因此 1-dice 才是 Dice 系数。
            class_wise_dice.append(1.0 - dice.item())
            # 按调用方给定权重累计该类损失。
            loss += dice * weight[i]
        # 对类别数取平均；若 weight 非等权，这里仍除以类别总数。
        return loss / self.n_classes

# 计算单个二值类别的 Dice、HD95、Jaccard 和 ASSD。
def calculate_metric_percase(pred, gt):
    # 把所有正值统一成前景 1；该操作会原地修改传入数组。
    pred[pred > 0] = 1
    # 对真实标签执行同样二值化。
    gt[gt > 0] = 1
    # 预测与真值都含前景时，几何距离指标才有正常定义。
    if pred.sum() > 0 and gt.sum()>0:
        # Dice = 2|P∩G|/(|P|+|G|)。
        dice = metric.binary.dc(pred, gt)
        # 95% Hausdorff 距离降低极端离群边界点对最大距离的影响。
        hd95 = metric.binary.hd95(pred, gt)
        # Jaccard/IoU = |P∩G|/|P∪G|。
        jaccard = metric.binary.jc(pred, gt)
        # assd 是预测和真值表面之间的对称平均距离。
        asd = metric.binary.assd(pred, gt)
        # 按测试代码期望的固定顺序返回四项指标。
        return dice, hd95, jaccard, asd
    # 原策略把“有预测但真值为空”返回为完美重叠；这里只注释现状，不改变其语义。
    elif pred.sum() > 0 and gt.sum()==0:
        # 返回 (Dice, HD95, Jaccard, ASSD)。
        return 1, 0, 1, 0
    # 其余情况包括预测为空且真值有前景，以及双方都为空。
    else:
        # 原策略统一返回零重叠和零距离。
        return 0, 0, 0, 0

# 验证阶段的轻量版本，只计算单个二值类别 Dice。
def calculate_dice_percase(pred, gt):
    # 原地将预测二值化。
    pred[pred > 0] = 1
    # 原地将真值二值化。
    gt[gt > 0] = 1
    # 两侧都含前景时调用 MedPy Dice。
    if pred.sum() > 0 and gt.sum()>0:
        # 计算二值 Dice 系数。
        dice = metric.binary.dc(pred, gt)
        # 返回 Python/NumPy 标量 Dice。
        return dice
    # 保持与上一个函数一致的原始空类策略。
    elif pred.sum() > 0 and gt.sum()==0:
        # 原代码把该情况记为 1。
        return 1
    # 其他空前景组合返回 0。
    else:
        # 返回零 Dice。
        return 0

# 对一个 Synapse 病例执行逐切片推理、逐类指标计算，并可保存可视化和 NIfTI。
def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, class_names=None):
    # DataLoader 增加了 batch 维；去掉 batch 后搬到 CPU、断开计算图并转为 NumPy。
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # 未提供器官名称时，用类别索引 1..C-1 作为图例标签。
    if class_names==None:
        # 背景类别 0 不参与器官图例。
        mask_labels = np.arange(1,classes)
    # 提供名称时直接用名称列表作为叠加图标签。
    else:
        # 例如 Synapse 的 spleen、right kidney 等八个前景名称。
        mask_labels = class_names
    # 取得 Matplotlib CSS4 颜色名称到色值的映射。
    cmaps = mcolors.CSS4_COLORS
    # 为最多十三个前景类别预设对比明显的颜色顺序。
    my_colors=['red','darkorange','yellow','forestgreen','blue','purple','magenta','cyan','deeppink', 'chocolate', 'olive','deepskyblue','darkviolet']
    # 只保留前 C-1 个指定颜色，构造 overlay_masks 接收的颜色字典。
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes-1]}
    # 三维输入按 [D,H,W] 逐轴向切片送入二维 EMCAD 网络。
    if len(image.shape) == 3:
        # 创建与标签同形状的整卷预测缓存，初值为背景 0。
        prediction = np.zeros_like(label)
        # 沿深度 D 遍历每一张二维切片。
        for ind in range(image.shape[0]):
            # 取当前轴向 CT 切片 [H,W]。
            slice = image[ind, :, :]
            # 保存原始高宽，推理后需要把预测缩回该尺寸。
            x, y = slice.shape[0], slice.shape[1]
            # 网络输入尺寸与原切片不一致时先重采样。
            if x != patch_size[0] or y != patch_size[1]:
                # CT 是连续信号，使用三次插值缩放到模型训练尺寸。
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            # 从 [H,W] 增加 batch、channel 两维成 [1,1,H,W]，转 float 并送到默认 CUDA。
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            # 切换评估模式，关闭 BatchNorm 统计更新等训练行为。
            net.eval()
            # 推理无需构建反向传播图，节省显存和计算。
            with torch.no_grad():
                # EMCADNet 返回多尺度分割输出列表。
                P = net(input)
                # 最后一个输出 P[-1] 是最高空间分辨率的最终预测头。
                outputs = P[-1]
                # 先在类别维 softmax，再 argmax 得到每像素类别索引，并去掉 batch 维。
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                # 搬回 CPU 并转 NumPy，供 SciPy 缩放和病例缓存写入。
                out = out.cpu().detach().numpy()
                # 若推理前调整过尺寸，就把离散类别图恢复到原始切片大小。
                if x != patch_size[0] or y != patch_size[1]:
                    # order=0 最近邻插值避免类别编号被插出小数或新类别。
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                # 原尺寸已经等于 patch_size 时无需二次插值。
                else:
                    # 直接沿用网络预测类别图。
                    pred = out
                # 把当前二维预测放回整卷对应深度位置。
                prediction[ind] = pred
                # saving the final output as a PNG file
                #print(test_save_path + '/'+case + '' +str(ind))
                #Image.fromarray((pred/8 * 255).astype(np.uint8)).save(test_save_path + '/'+case + '' +str(ind)+'_pred.png')
                #Image.fromarray((image[ind, :, :] * 255).astype(np.uint8)).save(test_save_path + '/'+case + '' +str(ind)+'_img.png')
                #Image.fromarray((label[ind, :, :]/8 * 255).astype(np.uint8)).save(test_save_path + '/'+case + '' +str(ind)+'_gt.png')
                #cmap = plt.cm.tab20(np.arange(len(mask_labels)))
                
                # 取当前切片的真实多类别标签。
                lbl = label[ind, :, :]
                # 收集每个前景类别的真实二值掩膜。
                masks = []
                # 跳过背景 0，遍历类别 1..C-1。
                for i in range(1, classes):
                    # 与 i 比较得到当前器官的布尔真值掩膜。
                    masks.append(lbl==i)
                # 收集每个前景类别的预测二值掩膜。
                preds_o = []
                # 与真值使用相同类别顺序，确保图例颜色对应。
                for i in range(1, classes):
                    # 当前器官预测位置为 True。
                    preds_o.append(pred==i)
                
                # 把真实器官掩膜半透明叠加到原始 CT 切片，返回 Matplotlib figure。
                fig_gt = overlay_masks(image[ind, :, :], masks, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                # 以完全相同颜色规则叠加预测掩膜，便于并排比较。
                fig_pred = overlay_masks(image[ind, :, :], preds_o, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                # Do with that image whatever you want to do.
                # 以病例名和切片序号保存真值叠加图；当前代码要求 test_save_path 非 None 且目录已存在。
                fig_gt.savefig(test_save_path + '/' + case + '_' +str(ind) + '_gt.png', bbox_inches="tight", dpi=300)
                # 保存对应预测叠加图，300 dpi 用于较清晰的结果检查。
                fig_pred.savefig(test_save_path + '/' + case + '_' +str(ind) + '_pred.png', bbox_inches="tight", dpi=300)

    # 二维输入分支直接推理一张图，不执行逐深度循环。
    else:
        # 增加 batch 和 channel 维，并将 float 输入送到 CUDA。
        input = torch.from_numpy(image).unsqueeze(
            # 两个 unsqueeze 最终形成 [1,1,H,W]。
            0).unsqueeze(0).float().cuda()
        # 使用评估模式。
        net.eval()
        # 关闭梯度记录。
        with torch.no_grad():
            # 获取模型多尺度输出。
            P = net(input)
            # 选择最终最高分辨率预测头。
            outputs = P[-1]
            # softmax 后按类别取最大概率索引。
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            # 去掉设备和计算图依赖，得到二维预测数组。
            prediction = out.cpu().detach().numpy()
    # 保存当前病例每个前景类别的四指标元组。
    metric_list = []    
    # 背景不纳入论文常见的器官平均指标，故从类别 1 开始。
    for i in range(1, classes):
        # 将多类别图转成第 i 类二值掩膜后计算指标。
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # 指定输出目录时，额外保存 CT、预测和真值 NIfTI 体数据。
    if test_save_path is not None:
        # 从 NumPy [D,H,W] 构造 SimpleITK CT 图像。
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        # 构造预测标签图像。
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        # 构造真实标签图像。
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        # 设定 x、y 间距为 1，z 间距使用调用方数据集配置。
        img_itk.SetSpacing((1, 1, z_spacing))
        # 预测必须使用同一 spacing，几何评测/查看时才与 CT 对齐。
        prd_itk.SetSpacing((1, 1, z_spacing))
        # 真值也设置同样 spacing。
        lab_itk.SetSpacing((1, 1, z_spacing))
        # 写出预测标签压缩 NIfTI。
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        # 写出归一化 CT 压缩 NIfTI。
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        # 写出真实标签压缩 NIfTI。
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    # 返回长度 C-1 的逐类指标列表，外层 test_synapse.py 再按病例求平均。
    return metric_list

# 验证阶段的病例级推理：流程与 test_single_volume 类似，但只返回逐类 Dice 且不保存图像。
def val_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    # 去掉 batch 维、转到 CPU，并转换为 NumPy 体数据。
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    # 完整体数据按深度逐切片推理。
    if len(image.shape) == 3:
        # 初始化整卷预测标签。
        prediction = np.zeros_like(label)
        # 遍历轴向切片。
        for ind in range(image.shape[0]):
            # 取当前 CT 切片。
            slice = image[ind, :, :]
            # 记录原始高宽。
            x, y = slice.shape[0], slice.shape[1]
            # 如有必要，将连续 CT 缩放到训练 patch 尺寸。
            if x != patch_size[0] or y != patch_size[1]:
                # order=3 对连续灰度图做三次插值。
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            # 组成 [1,1,H,W] CUDA float 张量。
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            # 关闭训练态行为。
            net.eval()
            # 验证不求梯度。
            with torch.no_grad():
                # 前向得到多尺度输出。
                P = net(input)
                # 原代码先建立浮点占位；下一行立即以最终输出覆盖。
                outputs = 0.0
                # 采用最高分辨率的最终预测头。
                outputs = P[-1]
                # 得到每像素类别索引。
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                # 转为 NumPy。
                out = out.cpu().detach().numpy()
                # 若推理尺寸与原图不同，恢复到原空间尺寸。
                if x != patch_size[0] or y != patch_size[1]:
                    # 离散预测使用最近邻插值。
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                # 尺寸一致时直接使用输出。
                else:
                    # 不创建额外缩放结果。
                    pred = out
                # 写入整卷预测缓存。
                prediction[ind] = pred
    # 二维样本直接单次前向。
    else:
        # 增加 batch/channel 维。
        input = torch.from_numpy(image).unsqueeze(
            # 转 float 并送入 CUDA。
            0).unsqueeze(0).float().cuda()
        # 切换评估模式。
        net.eval()
        # 关闭梯度。
        with torch.no_grad():
            # 获取多尺度输出。
            P = net(input)
            # 取最终头。
            outputs = P[-1]
            # 转成类别图。
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            # 搬回 CPU NumPy。
            prediction = out.cpu().detach().numpy()
    # 收集前景类别 Dice。
    metric_list = []
    # 跳过背景类别 0。
    for i in range(1, classes):
        # 对第 i 类预测/真值二值掩膜计算 Dice。
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    # 返回逐类 Dice，trainer.py 再求病例及类别平均。
    return metric_list

# 对 HWC 数组做左右镜像；用于下面的测试时增强。
def horizontal_flip(image):
    # 高度和通道轴不变，宽度轴使用 ::-1 反序。
    image = image[:, ::-1, :]
    # 返回翻转视图/数组。
    return image

# 对 HWC 数组做上下镜像。
def vertical_flip(image):
    # 高度轴反序，宽度和通道保持。
    image = image[::-1, :, :]
    # 返回翻转结果。
    return image

# Keras 风格 predict 接口的三视图测试时增强；当前 PyTorch Synapse 路径没有调用。
def tta_model(model, image):
    # 原始方向图像。
    n_image = image
    # 水平翻转图像。
    h_image = horizontal_flip(image)
    # 垂直翻转图像。
    v_image = vertical_flip(image)

    # 增加 batch 维并预测原始方向，取 batch 中第一项。
    n_mask = model.predict(np.expand_dims(n_image, axis=0))[0]
    # 预测水平翻转输入。
    h_mask = model.predict(np.expand_dims(h_image, axis=0))[0]
    # 预测垂直翻转输入。
    v_mask = model.predict(np.expand_dims(v_image, axis=0))[0]

    # 原方向预测无需逆变换；该赋值保留原实现。
    n_mask = n_mask
    # 把水平翻转预测翻回原坐标系。
    h_mask = horizontal_flip(h_mask)
    # 把垂直翻转预测翻回原坐标系。
    v_mask = vertical_flip(v_mask)

    # 对三个已对齐概率/掩膜逐像素取均值。
    mean_mask = (n_mask + h_mask + v_mask) / 3.0
    # 返回融合后的测试时增强结果。
    return mean_mask

# 使用随机输入统计模型 FLOPs、THOP 参数量和直接求和参数量。
def cal_params_flops(model, size, logger):
    # 构造 [1,3,size,size] 的随机 CUDA 输入；适用于三通道模型接口。
    input = torch.randn(1, 3, size, size).cuda()
    # THOP 执行前向钩子统计操作数和参数量。
    flops, params = profile(model, inputs=(input,))
    # 以十亿为单位打印 FLOPs。
    print('flops',flops/1e9)			## 打印计算量
    # 以百万为单位打印 THOP 参数量。
    print('params',params/1e6)			## 打印参数量

    # 直接累计 model.parameters() 中每个张量的元素数量。
    total = sum(p.numel() for p in model.parameters())
    # 打印百万参数规模。
    print("Total params: %.2fM" % (total/1e6))
    # 把同一统计写入调用方日志。
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')

# Example function to calculate and print GMACs and parameter count for a given model
# 使用 ptflops 打印模型参数量和 MACs 的辅助函数。
def print_model_stats(model, input_size=(3, 224, 224)):
    # Print model parameter count
    # 遍历全部参数张量并累计元素数量，不区分是否 requires_grad。
    total_params = sum(p.numel() for p in model.parameters())
    # 输出精确参数个数。
    print(f'Model created, param count: {total_params}')
    
    # Calculate GMACs using ptflops
    # ptflops 按给定 CHW 输入尺寸分析模型，并输出逐层统计。
    macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)
    
    # Display GMACs and params
    # 打印 ptflops 返回的可读 MACs 和参数量字符串。
    print(f'Model: {macs} GMACs, {params} parameters')
