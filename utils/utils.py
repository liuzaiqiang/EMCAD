
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

# ============================== 本文件阅读总览 ==============================
# 这个文件不是一个单一用途的“工具箱”，而是把 EMCAD 项目中几个不同阶段的公共小功能集中放在了一起。阅读时可以先按下面四条链路建立地图：
#
# 1. 训练辅助链路：powerset -> 生成多级输出的监督组合；
#    clip_gradient -> 在 optimizer.step() 之前限制梯度；
#    adjust_lr -> 在每个 epoch 调整优化器中的学习率；
#    AvgMeter -> 在终端显示近期 loss。
#
# 2. 标签和损失链路：one_hot_encoder / DiceLoss。
#    网络的多分类输出通常是 [B,C,H,W] 的 logits，而标签通常是 [B,H,W]的整数类别图；
#    DiceLoss 需要把后者转换为 [B,C,H,W] 的 one-hot 掩膜，然后逐类别计算 soft Dice loss。这里的 C 必须与网络输出通道数一致。
#
# 3. Synapse 病例评估链路：test_single_volume / val_single_volume。
#    训练样本可以是二维切片，但一个病例往往是 [D,H,W] 的三维体数据；这两个函数负责把体数据逐切片送进二维网络，再把每张切片的预测拼回三维体，最后按前景类别计算 Dice、HD95、Jaccard 和 ASD（或只算 Dice）。
#
# 4. 模型统计链路：CalParams / cal_params_flops / print_model_stats。它们只用于估计参数量、FLOPs 或 MACs，不参与训练，也不会提升模型精度。

# 递归生成 seq 的全部子集；训练器用它构造 mutation supervision 的输出组合。
# 调用位置：trainer.py 的 mutation supervision 分支会先构造 out_idxs=[0,1,2,3]，
# 再调用 powerset(out_idxs)，得到四个输出头的所有非空组合。每个组合会把相应的 logits 相加并计算一次损失，因此该函数间接决定了 mutation 分支一轮 batch要计算多少条监督路径。四个元素共有 2^4=16 个子集，去掉空集后是 15 个。
# 为什么使用生成器：
# 如果直接返回一个完整列表，短序列问题不大；
# 使用 yield 可以让调用方逐个消费组合，避免为更长的序列一次性额外复制全部结果。这里的 seq通常是 Python list，返回的子集也是 list；函数不会修改传入的 seq。
def powerset(seq):
    """
        Returns all the subsets of this set. This is a generator.
    """
    # 空序列或单元素序列是递归终止条件。
    if len(seq) <= 1:
        # 当 seq=[] 时，幂集只有一个元素：空集本身；当 seq=[x] 时，幂集为
        # [[x], []]。把这两个情况统一处理，可以让递归在缩短到空列表时自然结束。
        # 先产生序列自身；单元素时即包含该元素的子集。
        yield seq
        # 再产生空集，保证幂集定义完整。
        yield []
    # 多元素序列递归计算尾部 seq[1:] 的幂集。
    else:
        # 递归思想：先求“去掉首元素后的尾部”的幂集，再对尾部的每个子集
        # 产生两个版本：一个加入首元素，一个不加入首元素，因此不会漏组合。
        # item 依次代表不含首元素的每个尾部子集。
        for item in powerset(seq[1:]):
            # 在 item 前加入 seq[0]，得到包含首元素的对应子集。
            yield [seq[0]]+item
            # 同时产生不包含首元素的原 item。
            yield item

# 按元素把所有参数梯度截断到 [-grad_clip, grad_clip]，避免极端梯度值。
#
# 调用位置：旧版/SLDGroup 息肉训练脚本在 loss.backward() 之后、
# optimizer.step() 之前调用 clip_gradient(optimizer, opt.clip)。此时梯度已经写入每个 Parameter.grad，但参数还没有更新，所以这里可以安全地先处理梯度。
#
# 为什么需要梯度裁剪：某个batch、某个尺度或某条深监督路径可能产生异常大的梯度；若直接交给优化器，单步参数变化可能过大，表现为 loss 突然爆炸、NaN，或训练被一个异常样本破坏。
# 裁剪只能限制更新的输入，不能修复错误标签或不合理学习率，也不能保证梯度方向正确。
#
# 重要区别：当前实现是“逐元素裁剪”（element-wise clamp），每个梯度元素独立限制在 [-grad_clip, grad_clip]；
# 它不是按整个梯度向量的 L2 范数裁剪。两者对梯度方向和大小的影响不同，阅读实验配置时不要把它们当成同一种策略。
def clip_gradient(optimizer, grad_clip):
    """
        利用裁剪梯度技术标定不对准梯度
        :param optimizer:
        :param grad_clip:
        :return:
    """
    # 优化器可能包含多个具有不同超参数的参数组。
    for group in optimizer.param_groups:
        # param_groups 是 PyTorch 优化器的统一接口；即使当前实验只有一个参数组，
        # 也要从这里遍历，才能兼容“编码器和解码器使用不同超参数”的写法。
        # 遍历当前组里的每个可训练参数。
        for param in group['params']:
            # 参数本身是卷积核、偏置等张量；param.grad 是 backward() 产生的同形状梯度。
            # 未参与当前计算图的参数 grad 为 None，必须跳过。
            if param.grad is not None:
                # data 让裁剪直接作用于梯度存储值；clamp_ 末尾的下划线表示原地修改，
                # 因而不会创建一个新的梯度张量，也不会再次建立自动求导图。
                # clamp_ 原地截断梯度张量；这不是全局范数裁剪。
                param.grad.data.clamp_(-grad_clip, grad_clip)


# 按固定 epoch 周期计算学习率衰减因子并作用到优化器参数组。
#
# 调用位置：train_polyp_SLDGroup.py 在每个 epoch 开始处调用本函数，然后才进入train(...)。
# 它属于旧的阶梯式学习率策略；同一脚本随后还会在 epoch 末执行CosineAnnealingLR，因此如果两个策略同时有效，学习率会叠加变化。
#
# 参数含义：
# optimizer 是待修改的优化器；
# init_lr 在当前实现中没有被使用；
# epoch  当前轮次；
# decay_epoch 表示每隔多少轮衰减一次；
# decay_rate 是每次衰减乘上的比例。按公式，epoch=0..29 时 decay=1，epoch=30..59 时 decay=decay_rate。
#
# 代码风险提示：这里使用 param_group['lr'] *= decay，而不是基于 init_lr 重新计算。
# 如果调用方每个 epoch 都调用一次，且 epoch//decay_epoch 在同一阶段保持不变，
# 学习率会被重复乘以相同的 decay；这是当前代码行为，注释只帮助理解，不改逻辑。
def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    # 每经过 decay_epoch 个 epoch，指数 floor 值加 1，decay 再乘一次 decay_rate。
    decay = decay_rate ** (epoch // decay_epoch)
    # decay 是 Python 浮点数；例如 decay_rate=0.1、epoch=60、decay_epoch=30 时为 0.01。
    # 对优化器中的每个参数组更新学习率。
    for param_group in optimizer.param_groups:
        # 每个参数组各自保存学习率；逐组更新可以保留不同参数组的相对配置。
        # 原实现使用 *=，如果每个 epoch 都调用会在现有 lr 上继续累乘；init_lr 参数未直接使用。
        param_group['lr'] *= decay



# 维护标量的当前值、累计平均值和最近若干次记录。
#
# 调用位置：train_polyp_SLDGroup.py 创建 AvgMeter() 记录训练损失。每个原始 batch可能包含多个尺度更新，但旧脚本只在 rate==1 时调用 update；因此这里显示的是代码选择记录的那些 loss，不一定等于所有优化器 step 的严格平均值。
#
# 这个类同时维护两种“平均”：
# avg 是从 reset() 开始的全历史加权平均，
# show() 是最近 num 次记录的滑动平均。终端进度条通常使用 show()，因为短窗口能减少单个batch 的噪声；最终汇总时可使用 avg。val、sum、count 等字段不是模型参数，只是
# 日志统计状态，不会参与反向传播。
class AvgMeter(object):
    # num 决定 show() 最多平均最近多少个张量值。
    def __init__(self, num=40):
        # num 是显示窗口长度；默认只看最近 40 次 update，避免终端曲线过于抖动。
        # 保存滑动显示窗口大小。
        self.num = num
        # 统一初始化所有统计字段。
        self.reset()

    # 清空累计状态，开始新的统计区间。
    def reset(self):
        # reset 通常在开始一个新的 epoch、一个新的 run 或重新统计一项指标时调用；
        # 它不会改变 num，因此窗口设置会保留。
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
        # n 允许按 batch 中样本数加权。例如 batch loss 是 8 个样本的平均值时，
        # 用 n=8 累加后，avg 才等价于按所有样本展开后的平均，而不是“各 batch 等权”。
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
        # losses 中通常保存的是 0 维 PyTorch Tensor；因此 torch.stack 后仍是张量，
        # 调用方可以继续 .item() 或让 tqdm/日志系统读取。若从未调用 update，
        # losses 为空，torch.stack 会报错；当前训练脚本默认会先 update 再 show。
        # 起点不小于 0；torch.stack 要求 losses 中各张量形状一致。
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))

# 使用 THOP 计算给定 model 和实际 input_tensor 的 FLOPs/参数量并打印。
#
# 适用场景：在模型正式训练前，给一个已经构造好的网络和一份“代表性输入”，
# 快速估计一次前向传播的计算量与参数量。它是实验记录/模型对比工具，不会计算loss、不会调用 backward，也不会更新权重。
# input_tensor 的形状和设备必须与 model.forward() 的真实接口一致；如果网络要求 [B,1,H,W]，就不能随便传 [B,3,H,W]。
# THOP 是通过 forward hook 观察模块执行来估算 FLOPs，因此含有自定义算子、动态分支或多个输出时，统计值可能只是近似值。
# 不同工具（THOP、ptflops）对一次乘加是否算 1 次或 2 次操作的口径也可能不同，论文中比较复杂度时要保持工具和输入尺寸一致。
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
    # 这里使用单元素元组而不是直接传 input_tensor，是 THOP profile 的调用约定：
    # profile(model, inputs=(x,)) 会把 x 作为 forward 的第一个位置参数。
    flops, params = profile(model, inputs=(input_tensor,))
    # 将原始整数统计转换为例如 12.345G、26.789M 的字符串。
    flops, params = clever_format([flops, params], "%.3f")
    # 输出统一格式的模型复杂度摘要。
    # 函数没有 return；如果其他代码需要原始数值，必须自行调用 profile，不能从
    # CalParams 的返回值获得，因为当前函数返回值是 None。
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
    
# 把整数类别标签 [B,H,W] 转成张量 [B,C,H,W]。
# 这个函数服务于多分类 Dice 计算：
# CrossEntropyLoss 可以直接接收 [B,H,W] 的整数标签，但 Dice 需要对每个类别分别做交集，因此要把一个像素的类别编号转换为 C 个 0/1 通道。
# 例如标签像素值为 2 时，one-hot 的四类通道应为 [0,0,1,0]。
#
# input_tensor 预期是整数类别图，常见形状为 [B,H,W]；返回值形状为 [B,C,H,W]，类型为 float32。返回浮点而不是 bool，是因为后续要和网络概率逐元素相乘并求和。
# dataset == 'MMWHS' 是特殊分支：该数据集的标签值不是连续的 0,1,2,...，而是 [0,205,420,...] 这样的编码；其他数据集必须提供 n_classes，并且标签值连续。
# 当前 EMCAD 的 Synapse 训练路径主要使用 DiceLoss 类内部的同类逻辑；这个公开函数是通用/历史接口，trainer.py 虽然导入了它，但当前训练循环不一定直接调用。
def one_hot_encoder(input_tensor,dataset,n_classes = None):
    # 暂存每个类别的单通道布尔掩膜。
    tensor_list = []
    # MMWHS 原始标签不是连续 0..C-1，而是特定灰度编码。
    if dataset == 'MMWHS':  
        # 不能直接对 i in range(n_classes) 比较，否则 205、420 等真实标签会被
        # 当作“未知类别”全部变成全零；因此先显式列出数据集的原始编码。
        # 这些值依次代表 MMWHS 的背景/解剖类别编码；变量名 dict 沿用原实现。
        dict = [0,205,420,500,550,600,820,850]
        # 针对每个原始标签值构造一个通道。
        for i in dict:
            # 每次循环创建一个类别通道；比较运算不会改变 input_tensor 本身。
            # 相等比较得到 [B,H,W] 布尔掩膜。
            temp_prob = input_tensor == i  
            # 在通道位置 1 增维成 [B,1,H,W] 并加入列表。
            tensor_list.append(temp_prob.unsqueeze(1))
        # 沿通道维拼接全部类别掩膜。
        # dim=1 对应类别通道；如果误用 dim=0，会把 batch 维拼坏，后续 Dice 的
        # shape assert 会失败，且不同样本的类别信息会互相混在一起。
        output_tensor = torch.cat(tensor_list, dim=1)
        # 损失计算需要浮点 0/1，而非 bool。
        return output_tensor.float()
    # 其他数据集假定类别编号连续为 0..n_classes-1。
    else:
        # 非 MMWHS 分支假定标签已经经过数据集预处理，类别编号从 0 连续到 C-1；
        # n_classes=None 时 range(None) 会报错，这是调用方必须满足的前置条件。
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
#
# 调用位置：trainer.py 为 Synapse 构造 DiceLoss(num_classes)，在每个监督输出上调用 dice_loss(logits, label, softmax=True)；
# 其他多分类训练代码也可以复用它。
# 它是 nn.Module，因此可以像普通损失函数一样放入训练循环，但它本身没有可学习参数。
# 它的输入通常是：
#   inputs: [B,C,H,W]，网络输出的原始 logits，而不是已经 argmax 的类别图；
#   target: [B,H,W]，每个像素是 0..C-1 的整数类别编号；
#   weight: 可选的长度为 C 的类别权重；
#   softmax: 是否在函数内部把 logits 转为互斥类别概率。
#
# 为什么 Dice 要保留概率而不是先 argmax：argmax 是离散操作，几乎处处没有可用梯度；
# 训练阶段必须用 softmax 后的连续概率计算“软交集”，这样 loss.backward()才能把误差传回 EMCAD 的解码器和编码器。推理阶段才使用 softmax+argmax 得到最终类别索引。
# 该实现把背景类别 0 也纳入平均，最终由调用方决定是否再与 CE加权组合。
class DiceLoss(nn.Module):
    # n_classes 必须等于网络输出 logits 的通道数。
    def __init__(self, n_classes):
        # nn.Module 的初始化会建立 _parameters、_modules 等内部字典；不调用它，
        # 这个类虽然可能能算一次数值，但不能可靠地作为 PyTorch 模块使用。
        # 初始化 nn.Module 内部状态。
        super(DiceLoss, self).__init__()
        # 保存类别数供独热编码和逐类循环使用。
        self.n_classes = n_classes

    # 类内部的连续标签到独热标签转换。
    #
    # 这里与上面的公开 one_hot_encoder 逻辑相似，但不接收 dataset 参数；它专门
    # 服务当前 DiceLoss，假设 target 的类别编号已经是连续的 0..n_classes-1。
    # target [B,H,W] 经过循环后变为 [B,C,H,W]，从而能与 inputs 的每个类别通道
    # 一一对应。输入 target 不会被原地改写。
    def _one_hot_encoder(self, input_tensor):
        # 存放各类别 [B,1,H,W] 掩膜。
        tensor_list = []
        # 类别索引包含背景 0。
        for i in range(self.n_classes):
            # 每次只识别一个类别；比较结果是 bool mask，值为 True 的像素属于 i 类。
            # 比较得到属于第 i 类的位置。
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            # 增加通道维后加入列表。
            tensor_list.append(temp_prob.unsqueeze(1))
        # 拼接成与网络输出相同的 [B,C,H,W] 布局。
        output_tensor = torch.cat(tensor_list, dim=1)
        # 转成浮点参与乘法和求和。
        return output_tensor.float()

    # 计算单个类别通道的 Dice loss。
    #
    # score 是一个类别的连续概率图，形状通常为 [B,H,W]；target 是同形状的 0/1
    # 独热掩膜。这里把整个 batch 和空间维一起求和，所以返回一个代表“该类别在
    # 当前 batch 上整体重叠程度”的标量，而不是每张图各返回一个值。
    # 公式是：Dice = (2 * sum(score * target) + smooth) /
    #                  (sum(score^2) + sum(target^2) + smooth)
    # loss = 1 - Dice。使用平方和是 soft Dice 的常见写法，允许 score 保持连续。
    def _dice_loss(self, score, target):
        # 确保目标掩膜与概率 score 使用兼容浮点类型。
        target = target.float()
        # 平滑项防止预测和标签都为空时分母为 0。
        # smooth 只是在分子、分母都很小时提供数值保护；它不是医学意义上的
        # 平滑图像，也不会改变预测边界。值很小是为了不明显改变正常样本的 Dice。
        smooth = 1e-5
        # 概率与目标逐元素乘积之和，对应软交集。
        # torch.sum 默认把 batch 和空间维全部压成一个标量，因此当前实现计算的是
        # batch-level Dice，而不是先逐病例算 Dice 再平均；两种统计方式并不完全相同。
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
    #
    # forward 是 PyTorch 在执行 dice_loss(...) 时自动调用的入口。训练调用通常把
    # softmax=True，因为 inputs 是网络的 logits；如果调用方已经提前做了 softmax，
    # 就应传 softmax=False，避免重复归一化。注意：如果把已经 argmax 的整数类别图
    # 传进来，形状和梯度语义都会不符合这个函数的设计。
    def forward(self, inputs, target, weight=None, softmax=False):
        # 训练调用 softmax=True 时，把互斥类别 logits 转成逐像素概率。
        if softmax:
            # 多分类互斥任务沿 dim=1（类别维）归一化；每个像素 C 个概率之和为 1。
            # dim=1 是类别通道维。
            inputs = torch.softmax(inputs, dim=1)
        # 把整数标签转换为与 inputs 同形状的独热张量。
        # 这个转换只作用于监督目标，不会对 inputs 做离散化，因此 inputs 的梯度仍能
        # 通过后续乘法和求和回到网络。
        target = self._one_hot_encoder(target)
        # 未提供类别权重时，各类等权。
        if weight is None:
            # 使用 Python list 足够，因为权重只参与标量乘法；它不是需要反向传播的
            # Tensor。若传入 weight，调用方必须至少提供 n_classes 个元素。
            # 创建长度为 C 的 Python 权重列表。
            weight = [1] * self.n_classes
        # 在逐类计算前严格检查 batch、通道和空间形状一致。
        # 这个 assert 能尽早暴露“模型输出通道数与标签独热通道数不一致”或尺寸还原
        # 错误；否则错误可能延迟到逐元素乘法，报错位置会更难理解。
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        # 保存每类 Dice 数值用于调试；当前函数最终未返回该列表。
        class_wise_dice = []
        # 初始化总损失为浮点零。
        # Python float 与第一个 Tensor 相加后会自动变成带梯度的 Tensor；这样无需预先
        # 知道 inputs 的 device。循环至少执行一次，因为 n_classes 应为正数。
        loss = 0.0
        # 包含背景在内逐类计算损失。
        for i in range(0, self.n_classes):
            # 取第 i 个概率通道和独热目标通道。
            # [:, i] 去掉类别维，得到 [B,H,W]；第 i 类的 Dice 不会与其他类别直接
            # 混在一起，最后再通过 loss 累加形成多类平均。
            dice = self._dice_loss(inputs[:, i], target[:, i])
            # dice 变量实际是 loss，因此 1-dice 才是 Dice 系数。
            class_wise_dice.append(1.0 - dice.item())
            # 按调用方给定权重累计该类损失。
            loss += dice * weight[i]
        # 对类别数取平均；若 weight 非等权，这里仍除以类别总数。
        # 这里的除数固定为 C，而不是 sum(weight)。因此传入非全 1 权重时，权重会
        # 改变总损失尺度；调用方比较实验 loss 数值时要记录权重设置。
        return loss / self.n_classes

# 计算单个二值类别的 Dice、HD95、Jaccard 和 ASSD。
#
# 调用位置：test_single_volume 在每个前景类别上把 prediction==i 和 label==i
# 转成布尔掩膜后调用本函数；test_synapse.py 再把每个病例返回的四元组汇总。
# 这里的“percase”指一个病例、一个类别，而不是整个数据集一次性计算。
#
# 输入 pred、gt 应具有相同形状，通常是 [D,H,W] 或单张 [H,W] 的 NumPy 数组，
# 值可以是 bool、0/1 或任意正数标签。函数会把正值原地改成 1，因此传入的是视图或仍要复用的数组时要注意副作用。
# MedPy 的 hd95/assd 还依赖前景边界；如果数组没有合适的体素 spacing，本文件调用的是默认像素间距，结果单位是像素而不是真实毫米。test_single_volume 保存 NIfTI 时会写入 z_spacing，但这里计算指标时并没有把 spacing 传给 MedPy，这是当前代码的实际语义。
#
# 空前景分支是历史约定，不能简单按数学公式理解：当前代码对“预测有前景、真值为空”返回 Dice=1、Jaccard=1，这与通常把假阳性判为 0 的评价约定相反；这里保留
# 原行为，只把它明确写出来，避免阅读者误以为该分支是 MedPy 自动得出的结果。
def calculate_metric_percase(pred, gt):
    # 把所有正值统一成前景 1；该操作会原地修改传入数组。
    # 二值化的目的，是让指标函数只区分“前景/背景”，而不是把不同正整数当作
    # 不同类别。调用方已经按某个 i 构造了 prediction==i，因此这里通常只是保险。
    pred[pred > 0] = 1
    # 对真实标签执行同样二值化。
    gt[gt > 0] = 1
    # 预测与真值都含前景时，几何距离指标才有正常定义。
    if pred.sum() > 0 and gt.sum()>0:
        # 两个 sum 都在检查“至少有一个前景像素”；如果一侧为空，HD95/ASSD 的
        # 表面距离可能没有定义，所以不能无条件调用 MedPy。
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
        # 该分支的四个返回值是人为设定的常量，不是由 pred/gt 的几何关系计算出来。
        # 如果后续研究要修正评价协议，应在实验记录中说明，否则不同版本的结果不可比。
        # 返回 (Dice, HD95, Jaccard, ASSD)。
        return 1, 0, 1, 0
    # 其余情况包括预测为空且真值有前景，以及双方都为空。
    else:
        # 原策略统一返回零重叠和零距离。
        return 0, 0, 0, 0



# 验证阶段的轻量版本，只计算单个二值类别 Dice。
# 调用位置：trainer.py 的 inference/验证循环调用 val_single_volume，后者只需要Dice 来选择 best.pth，因此使用此轻量函数，避免每个 epoch 都计算较昂贵的 HD95、Jaccard 和 ASSD。
# 它与 calculate_metric_percase 使用相同的二值化和空类约定，所以验证 Dice 与最终测试 Dice 的空类处理是一致的，但测试时的其他几何指标不会在这里产生。
def calculate_dice_percase(pred, gt):
    # 原地将预测二值化。
    pred[pred > 0] = 1
    # 原地将真值二值化。
    gt[gt > 0] = 1
    # 两侧都含前景时调用 MedPy Dice。
    if pred.sum() > 0 and gt.sum()>0:
        # 只有 Dice 需要时，不调用表面距离算法，可以显著减少验证阶段的耗时。
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
# 这是本文件最重要的“测试集/整病例”函数。
# 训练阶段通常把二维切片作为样本，但医学论文的病例级结果要把同一个病人的所有切片重新组合起来；本函数就是这个桥梁。典型调用来自 test_synapse.py：
#
#   image: DataLoader 输出的 [1,D,H,W]，其中第一个 1 是 batch 维；
#   label: 同样是 [1,D,H,W] 的整数标签；
#   net:   已加载 checkpoint 的 EMCADNet，输入通常要求 [B,1,H,W]；
#   classes: 类别总数，包含背景 0，例如 Synapse 的 9；
#   patch_size: 网络训练/推理使用的二维尺寸，例如 [224,224] 或 [256,256]；
#   test_save_path: 输出目录；不为 None 时会写 NIfTI，并且三维分支还会保存 PNG；
#   z_spacing: 保存 NIfTI 时使用的 z 方向体素间距；
#   class_names: 可选的前景器官名称，用于叠加图标签。
#
# 核心顺序是：去 batch -> 逐切片 resize -> CUDA 前向 -> softmax/argmax ->还原切片尺寸 -> 拼回 prediction -> 对每个前景类别计算指标 -> 可选保存结果。
# resize 图像时用三次插值，因为 CT 灰度是连续值；resize 离散类别预测时用最近邻插值，因为线性/三次插值会生成 1.3 之类不存在的类别编号。
#
# 当前实现有几个必须知道的运行前提：函数内部硬编码 .cuda()，所以没有 CUDA 时即使调用方选择 CPU 也会失败；
# net.eval() 在每张切片循环内重复调用，语义正确但有少量额外开销；
# 三维分支的 PNG 保存语句没有用 test_save_path 做条件保护，因而若 test_save_path=None，实际运行到 fig_gt.savefig 时可能报错。这里不修改
# 这些历史行为，只在注释中把它们标明，便于你沿调用链排查问题。
def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, class_names=None):
    # DataLoader 增加了 batch 维；去掉 batch 后搬到 CPU、断开计算图并转为 NumPy。
    # 评估函数不需要继续建立 autograd 图；detach() 解除历史计算图引用，cpu() 让
    # 后面的 NumPy、SciPy 和 SimpleITK 接口可以使用。若传入的是 [D,H,W] 而不是
    # [1,D,H,W]，squeeze(0) 仍可能误删深度维，调用方必须保持约定的 batch 形状。
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # 未提供器官名称时，用类别索引 1..C-1 作为图例标签。
    if class_names==None:
        # class_names 只影响可视化图例，不影响 prediction 的类别编号和指标计算。
        # 因此没有名称时使用 1..classes-1 作为占位标签，跳过背景 0。
        # 背景类别 0 不参与器官图例。
        mask_labels = np.arange(1,classes)
    # 提供名称时直接用名称列表作为叠加图标签。
    else:
        # 例如 Synapse 的 spleen、right kidney 等八个前景名称。
        mask_labels = class_names
    # 取得 Matplotlib CSS4 颜色名称到色值的映射。
    # overlay_masks 接收的是颜色字典，而不是类别编号到颜色的直接数组；下面会
    # 用同一套 cmap 同时绘制 ground truth 和 prediction，保证颜色能逐类对应。
    cmaps = mcolors.CSS4_COLORS
    # 为最多十三个前景类别预设对比明显的颜色顺序。
    my_colors=['red','darkorange','yellow','forestgreen','blue','purple','magenta','cyan','deeppink', 'chocolate', 'olive','deepskyblue','darkviolet']
    # 只保留前 C-1 个指定颜色，构造 overlay_masks 接收的颜色字典。
    # sorted(cmaps.keys()) 只是建立稳定的字典顺序；真正选中的颜色由 my_colors
    # 过滤。classes 超过预设颜色数量时，颜色字典不会自动扩容，可能导致图例颜色
    # 不足，这是当前可视化配置的边界条件。
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes-1]}
    # 三维输入按 [D,H,W] 逐轴向切片送入二维 EMCAD 网络。
    if len(image.shape) == 3:
        # 三维分支是病例级主要路径。此时 image/label 通常为 [D,H,W]；网络本身仍
        # 是二维模型，所以不能把整个 [D,H,W] 直接当作 [B,C,H,W] 输入。
        # 创建与标签同形状的整卷预测缓存，初值为背景 0。
        prediction = np.zeros_like(label)
        # 沿深度 D 遍历每一张二维切片。
        for ind in range(image.shape[0]):
            # ind 代表轴向切片编号；prediction[ind] 最终与 label[ind] 一一对应，
            # 这样病例级指标可以在完整三维坐标中比较预测和真值。
            # 取当前轴向 CT 切片 [H,W]。
            slice = image[ind, :, :]
            # 保存原始高宽，推理后需要把预测缩回该尺寸。
            x, y = slice.shape[0], slice.shape[1]
            # 网络输入尺寸与原切片不一致时先重采样。
            if x != patch_size[0] or y != patch_size[1]:
                # 缩放比例分别按高度和宽度计算，避免非正方形原始切片被强行按一个
                # 比例拉伸。order=3 会读取邻域像素估计连续灰度值，适合输入图像。
                # CT 是连续信号，使用三次插值缩放到模型训练尺寸。
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            # 从 [H,W] 增加 batch、channel 两维成 [1,1,H,W]，转 float 并送到默认 CUDA。
            # 第一次 unsqueeze(0) 是 batch=1，第二次是单通道医学图像；float() 满足
            # 卷积层通常要求浮点输入。这里没有显式指定 device，而是依赖当前默认
            # CUDA 设备，和模型所在 GPU 必须一致，否则会出现 device mismatch。
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            # 切换评估模式，关闭 BatchNorm 统计更新等训练行为。
            net.eval()
            # 推理无需构建反向传播图，节省显存和计算。
            with torch.no_grad():
                # no_grad 不会改变 net 的参数，只是停止保存反向传播所需的中间激活；
                # 评估时这样可以显著降低显存，并避免误把测试过程接入训练图。
                # EMCADNet 返回多尺度分割输出列表。
                P = net(input)
                # 最后一个输出 P[-1] 是最高空间分辨率的最终预测头。
                # EMCAD 的训练/推理接口返回多个尺度或多个解码头；这里选择列表最后
                # 一个作为最终结果，而不是把所有输出平均。训练时的深监督仍可能使用
                # 全部输出，但本测试函数只使用最终头。
                outputs = P[-1]
                # 先在类别维 softmax，再 argmax 得到每像素类别索引，并去掉 batch 维。
                # softmax 把 logits 变成概率，argmax 再选概率最大的类别；由于只需要
                # 离散标签而不需要概率值，最后得到 [H,W] 的整数类别图。
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                # 搬回 CPU 并转 NumPy，供 SciPy 缩放和病例缓存写入。
                out = out.cpu().detach().numpy()
                # 若推理前调整过尺寸，就把离散类别图恢复到原始切片大小。
                if x != patch_size[0] or y != patch_size[1]:
                    # 预测图是离散类别编号，必须使用 order=0；否则插值会把类别编号
                    # 混合成小数，后续 prediction==i 会漏掉这些像素。
                    # order=0 最近邻插值避免类别编号被插出小数或新类别。
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                # 原尺寸已经等于 patch_size 时无需二次插值。
                else:
                    # 直接沿用网络预测类别图。
                    pred = out
                # 把当前二维预测放回整卷对应深度位置。
                # pred 的空间尺寸已经恢复为 x*y；赋值把本切片结果写到与标签同形状的
                # 三维缓存中，循环结束后 prediction 就是整个病例的预测标签体。
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
                # overlay_masks 不直接接受整数标签图，而是希望收到“每个类别一个
                # bool mask”的列表；因此这里把一张多类 lbl 拆成 C-1 张前景掩膜。
                masks = []
                # 跳过背景 0，遍历类别 1..C-1。
                for i in range(1, classes):
                    # 与 i 比较得到当前器官的布尔真值掩膜。
                    masks.append(lbl==i)
                # 收集每个前景类别的预测二值掩膜。
                # preds_o 与 masks 必须使用完全相同的类别顺序，否则图例中的颜色会
                # 对不上器官，造成看图时的误判；因此两个循环都从 1 遍历到 classes-1。
                preds_o = []
                # 与真值使用相同类别顺序，确保图例颜色对应。
                for i in range(1, classes):
                    # 当前器官预测位置为 True。
                    preds_o.append(pred==i)
                
                # 把真实器官掩膜半透明叠加到原始 CT 切片，返回 Matplotlib figure。
                # mask_alpha=0.5 让底层灰度 CT 仍可见；这一步只生成检查用图，不参与
                # 指标计算，也不会改变 image、lbl 或 prediction。
                fig_gt = overlay_masks(image[ind, :, :], masks, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                # 以完全相同颜色规则叠加预测掩膜，便于并排比较。
                fig_pred = overlay_masks(image[ind, :, :], preds_o, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                # Do with that image whatever you want to do.
                # 以病例名和切片序号保存真值叠加图；当前代码要求 test_save_path 非 None 且目录已存在。
                # 文件名包含 case 和 ind，便于把图重新对应到具体病例/切片；bbox_inches
                # 去除多余边缘，dpi=300 适合论文前的人工质量检查，但会增加磁盘写入时间。
                fig_gt.savefig(test_save_path + '/' + case + '_' +str(ind) + '_gt.png', bbox_inches="tight", dpi=300)
                # 保存对应预测叠加图，300 dpi 用于较清晰的结果检查。
                fig_pred.savefig(test_save_path + '/' + case + '_' +str(ind) + '_pred.png', bbox_inches="tight", dpi=300)

    # 二维输入分支直接推理一张图，不执行逐深度循环。
    else:
        # 如果调用方传入的是单张 [H,W] 图像，就走这里；prediction 最终是 [H,W]，
        # 与三维分支的 prediction[ind] 具有相同的“整数类别图”语义。这个分支不生成
        # PNG 叠加图，只在函数末尾计算指标并按需保存 NIfTI（对二维数据写出的其实
        # 是二维图像对象）。
        # 增加 batch 和 channel 维，并将 float 输入送到 CUDA。
        input = torch.from_numpy(image).unsqueeze(
            # 两个 unsqueeze 最终形成 [1,1,H,W]。
            0).unsqueeze(0).float().cuda()
        # 使用评估模式。
        net.eval()
        # 关闭梯度记录。
        with torch.no_grad():
            # 与三维分支保持同一推理规则：只取 P[-1]，softmax 后 argmax；这样两种
            # 输入维度的评估结果使用同一输出头和同一类别决策方式。
            # 获取模型多尺度输出。
            P = net(input)
            # 选择最终最高分辨率预测头。
            outputs = P[-1]
            # softmax 后按类别取最大概率索引。
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            # 去掉设备和计算图依赖，得到二维预测数组。
            prediction = out.cpu().detach().numpy()
    # 保存当前病例每个前景类别的四指标元组。
    # prediction 和 label 都是整数类别图；prediction == i 会生成第 i 类的 bool
    # 掩膜，再由 calculate_metric_percase 计算一个 (Dice, HD95, Jaccard, ASD) 元组。
    metric_list = []    
    # 背景不纳入论文常见的器官平均指标，故从类别 1 开始。
    for i in range(1, classes):
        # 从 1 开始是有意跳过背景：医学分割报告通常关心器官/病灶前景，背景面积
        # 很大，纳入平均会掩盖前景分割质量。
        # 将多类别图转成第 i 类二值掩膜后计算指标。
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # 指定输出目录时，额外保存 CT、预测和真值 NIfTI 体数据。
    if test_save_path is not None:
        # 这里的保存条件只包住 NIfTI 写出；三维分支前面保存 PNG 的两行位于循环中，
        # 当前代码并没有用相同条件保护它们。输出目录应事先创建，并且 case 不能为 None，
        # 否则路径拼接会失败。
        # 从 NumPy [D,H,W] 构造 SimpleITK CT 图像。
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        # 构造预测标签图像。
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        # 构造真实标签图像。
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        # 设定 x、y 间距为 1，z 间距使用调用方数据集配置。
        # SimpleITK 的 spacing 顺序是 (x,y,z)，而 NumPy 数组通常按 [z,y,x] 访问；
        # 这里把面内间距固定成 1，把调用方给出的 z_spacing 写入第三个坐标。
        # 这只影响保存文件的物理元数据，不会改变 prediction 的数组索引。
        img_itk.SetSpacing((1, 1, z_spacing))
        # 预测必须使用同一 spacing，几何评测/查看时才与 CT 对齐。
        prd_itk.SetSpacing((1, 1, z_spacing))
        # 真值也设置同样 spacing。
        lab_itk.SetSpacing((1, 1, z_spacing))
        # 写出预测标签压缩 NIfTI。
        # 三个文件使用同一病例前缀，方便在 ITK-SNAP 等软件中同时打开并核对 CT、
        # 预测和 ground truth；预测/标签使用 float32 保存，读取时仍应按离散类别解释。
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        # 写出归一化 CT 压缩 NIfTI。
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        # 写出真实标签压缩 NIfTI。
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    # 返回长度 C-1 的逐类指标列表，外层 test_synapse.py 再按病例求平均。
    return metric_list



# 验证阶段的病例级推理：流程与 test_single_volume 类似，但只返回逐类 Dice 且不保存图像。
#
# 调用位置：trainer.py 的 inference() 在每个 epoch 后调用本函数，得到一个病例中每个前景器官的 Dice，再跨病例平均，并用 mean Dice 决定是否保存 best.pth。
#
# 与 test_single_volume 的差别是“验证阶段只保留选择 checkpoint 所需的最小结果”：不创建 overlay_masks 图、不写 PNG/NIfTI、不计算 HD95/Jaccard/ASD，只返回长度classes-1 的 Dice 列表。
# 此它通常比完整测试更快，但不能替代最终的病例级指标报告。
# 参数 test_save_path、case、z_spacing 为了兼容旧调用接口而保留，在当前函数体中没有参与保存或 Dice 计算；看到它们不要误以为验证阶段会写文件。
#
# 输入形状约定与 test_single_volume 相同：DataLoader 的 [1,D,H,W] 会先变成 [D,H,W]，
# 然后逐切片 resize、前向、还原尺寸并拼回 prediction；若输入已经是 [H,W]，则只做一次前向。函数同样硬编码 .cuda()，所以当前验证实现要求 CUDA 环境。
def val_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    # 去掉 batch 维、转到 CPU，并转换为 NumPy 体数据。
    # 验证只做前向和指标，不需要保留 PyTorch 图；把数据转 NumPy 后可以直接使用
    # scipy.zoom 和本文件的 NumPy 比较逻辑。这里的 squeeze(0) 依赖调用方保留 batch=1。
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    # 完整体数据按深度逐切片推理。
    if len(image.shape) == 3:
        # 对三维病例逐切片执行二维网络推理；prediction 先分配完整体积，避免每张
        # 切片预测结束后还要重新组织深度维度。
        # 初始化整卷预测标签。
        prediction = np.zeros_like(label)
        # 遍历轴向切片。
        for ind in range(image.shape[0]):
            # 这一层循环的 ind 是病例内的深度索引，不是 batch 索引；每次只处理一个
            # [H,W] 切片，最后写回 prediction[ind]。
            # 取当前 CT 切片。
            slice = image[ind, :, :]
            # 记录原始高宽。
            x, y = slice.shape[0], slice.shape[1]
            # 如有必要，将连续 CT 缩放到训练 patch 尺寸。
            if x != patch_size[0] or y != patch_size[1]:
                # 输入灰度值是连续数据，用三次插值；保持与 test_single_volume 完全
                # 相同的前向尺寸，验证结果才与最终测试结果具有可比性。
                # order=3 对连续灰度图做三次插值。
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            # 组成 [1,1,H,W] CUDA float 张量。
            # 网络需要 batch 和 channel 两个维度；医学 CT 在这里按单通道处理，不能
            # 把深度 D 误当成 channel C，否则卷积权重和语义都会不匹配。
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            # 关闭训练态行为。
            net.eval()
            # 验证不求梯度。
            with torch.no_grad():
                # 验证阶段不保存激活、不求梯度；这既节省显存，也避免验证调用影响
                # 训练优化器的梯度状态。
                # 前向得到多尺度输出。
                P = net(input)
                # 原代码先建立浮点占位；下一行立即以最终输出覆盖。
                # 这个占位对最终数值没有作用，只是历史代码遗留；真正使用的对象必须
                # 是 P[-1]，否则后面的 softmax/argmax 无法得到网络输出。
                outputs = 0.0
                # 采用最高分辨率的最终预测头。
                outputs = P[-1]
                # 得到每像素类别索引。
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                # 转为 NumPy。
                out = out.cpu().detach().numpy()
                # 若推理尺寸与原图不同，恢复到原空间尺寸。
                if x != patch_size[0] or y != patch_size[1]:
                    # 类别图只能最近邻还原；连续插值会生成不存在的类别编号，导致
                    # prediction == i 的统计不可靠。
                    # 离散预测使用最近邻插值。
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                # 尺寸一致时直接使用输出。
                else:
                    # 不创建额外缩放结果。
                    pred = out
                # 写入整卷预测缓存。
                # 此时 pred 的尺寸已经与原始切片一致，赋值后 prediction 与 label 的
                # 形状相同，后续每个类别的布尔比较可以逐像素对齐。
                prediction[ind] = pred
    # 二维样本直接单次前向。
    else:
        # 单张二维图像的快捷路径；它与三维分支只差一个深度循环，输出仍是 [H,W]
        # 的整数类别图，便于统一走下面的 Dice 统计代码。
        # 增加 batch/channel 维。
        input = torch.from_numpy(image).unsqueeze(
            # 转 float 并送入 CUDA。
            0).unsqueeze(0).float().cuda()
        # 切换评估模式。
        net.eval()
        # 关闭梯度。
        with torch.no_grad():
            # 使用与三维分支相同的最终输出头和类别决策规则，避免因输入维度不同而
            # 产生两套不一致的验证定义。
            # 获取多尺度输出。
            P = net(input)
            # 取最终头。
            outputs = P[-1]
            # 转成类别图。
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            # 搬回 CPU NumPy。
            prediction = out.cpu().detach().numpy()
    # 收集前景类别 Dice。
    # 这里不调用 calculate_metric_percase，是为了避免验证每个 epoch 都计算表面
    # 距离指标；返回值只服务 trainer.py 的 checkpoint 选择和 TensorBoard/日志记录。
    metric_list = []
    # 跳过背景类别 0。
    for i in range(1, classes):
        # 第 i 类单独转成 bool mask；跳过背景后，列表第 0 项对应类别 1，而不是类别 0。
        # 对第 i 类预测/真值二值掩膜计算 Dice。
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    # 返回逐类 Dice，trainer.py 再求病例及类别平均。
    return metric_list




# 对 HWC 数组做左右镜像；用于下面的测试时增强。
#
# 输入约定是 NumPy 的 HWC（Height, Width, Channel）布局，而不是 PyTorch 常见的CHW。
# 切片表达式只反转第 2 个轴，也就是左右方向；通道轴保持不变。函数不改变数值内容，只改变像素的空间顺序。
# 它属于早期/兼容性 TTA 辅助函数，当前 EMCAD的主要 Synapse 测试路径使用 test_single_volume，并没有调用这里的 tta_model。
def horizontal_flip(image):
    # 高度和通道轴不变，宽度轴使用 ::-1 反序。
    # `::-1` 的步长为 -1，表示从最后一列向第一列读取；这样左侧像素会移动到右侧。
    # 对 NumPy 而言，这通常产生负 stride 的视图，而不是立刻复制全部数据。
    image = image[:, ::-1, :]
    # 返回翻转视图/数组。
    return image



# 对 HWC 数组做上下镜像。
# 这与 horizontal_flip 对称：只反转高度轴，宽度和通道不变。TTA 使用它来构造一个与训练样本不同方向的输入，检验模型是否对简单的空间翻转更稳健。
def vertical_flip(image):
    # 高度轴反序，宽度和通道保持。
    # 第一个 `::-1` 反转行顺序；如果图像是 HWC，结果形状仍为 HWC。
    image = image[::-1, :, :]
    # 返回翻转结果。
    return image




# Keras 风格 predict 接口的三视图测试时增强；当前 PyTorch Synapse 路径没有调用。
#
# TTA（test-time augmentation）的思想是：同一张图以原始、水平翻转、垂直翻转三种形式预测，再把所有结果变回原坐标并平均。
# 若模型输出的是概率图，平均可以降低某一次方向预测的偶然误差；
# 若输出已经是离散 0/1 类别图，平均后的值则需要调用方再阈值化，不能直接当作类别编号。
#
# 该函数使用 model.predict(...)，这是 Keras 风格接口；EMCAD 的 PyTorch 模型通常直接调用 model(tensor)，所以不能把它未经改造地接到当前 PyTorch 网络上。它还假设image 和 model 输出都是 HWC，并依赖 NumPy 翻转结果能被 model.predict 接受。
def tta_model(model, image):
    # 原始方向图像。
    # 保留原引用即可，因为函数不会修改 n_image 的元素；翻转函数只产生新的视图。
    n_image = image
    # 水平翻转图像。
    h_image = horizontal_flip(image)
    # 垂直翻转图像。
    v_image = vertical_flip(image)

    # 增加 batch 维并预测原始方向，取 batch 中第一项。
    # np.expand_dims(..., axis=0) 把 HWC 变成 NHWC；[0] 去掉模型返回的 batch 维，
    # 使三个 mask 可以按像素相加。
    n_mask = model.predict(np.expand_dims(n_image, axis=0))[0]
    # 预测水平翻转输入。
    h_mask = model.predict(np.expand_dims(h_image, axis=0))[0]
    # 预测垂直翻转输入。
    v_mask = model.predict(np.expand_dims(v_image, axis=0))[0]

    # 原方向预测无需逆变换；该赋值保留原实现。
    # 这是一条“保持原坐标”的显式赋值，不会产生新数据；写出来是为了让三条分支
    # 的后续处理形式一致。
    n_mask = n_mask
    # 把水平翻转预测翻回原坐标系。
    h_mask = horizontal_flip(h_mask)
    # 把垂直翻转预测翻回原坐标系。
    v_mask = vertical_flip(v_mask)

    # 对三个已对齐概率/掩膜逐像素取均值。
    # 翻转预测已经先逆变换，所以三个数组的每个坐标指向原图同一像素；逐元素平均
    # 才有意义。这里没有加权，也没有再次 sigmoid/threshold。
    mean_mask = (n_mask + h_mask + v_mask) / 3.0
    # 返回融合后的测试时增强结果。
    return mean_mask




# 使用随机输入统计模型 FLOPs、THOP 参数量和直接求和参数量。
#
# 调用位置：train_polyp_SLDGroup.py 启动每个 run 时调用 cal_params_flops(model,opt.img_size, logging)，用于把模型复杂度写入日志。
# 它和 CalParams 的区别是： CalParams 接收调用方准备好的真实 input_tensor；本函数自己创建一个 [1,3,S,S]的随机 CUDA 输入，并同时打印 THOP 统计和直接计数得到的参数量。
#
# 当前实现假设模型接受 3 通道输入且 CUDA 可用。EMCAD 的某些医学图像路径可能是单通道 [B,1,H,W]，这种情况下直接调用此函数可能在 forward 阶段报通道数错误；
# 复杂度统计的输入通道数必须与实际模型入口保持一致，否则 FLOPs 没有可比性。
def cal_params_flops(model, size, logger):
    # 构造 [1,3,size,size] 的随机 CUDA 输入；适用于三通道模型接口。
    # 使用随机值只是为了触发一次与真实尺寸相同的 forward；THOP 统计的是结构和
    # 空间尺寸，不依赖这组随机像素的医学含义。
    input = torch.randn(1, 3, size, size).cuda()
    # THOP 执行前向钩子统计操作数和参数量。
    # profile 可能真的执行模型 forward，因此模型所在 device、输入 dtype 和模型
    # 当前模式都必须兼容；统计前最好确保没有正在进行的训练梯度累积。
    flops, params = profile(model, inputs=(input,))
    # 以十亿为单位打印 FLOPs。
    print('flops',flops/1e9)			## 打印计算量
    # 以百万为单位打印 THOP 参数量。
    print('params',params/1e6)			## 打印参数量

    # 直接累计 model.parameters() 中每个张量的元素数量。
    # 这是最直观的参数量统计：每个标量权重算一个参数；它不依赖 THOP 是否认识
    # 某个自定义层，因此可与 THOP 返回的 params 互相核对。
    total = sum(p.numel() for p in model.parameters())
    # 打印百万参数规模。
    print("Total params: %.2fM" % (total/1e6))
    # 把同一统计写入调用方日志。
    # logger 通常是 logging 模块；写日志后可在实验结果中追溯当时使用的模型规模，
    # 避免只在终端打印而丢失记录。
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')



# Example function to calculate and print GMACs and parameter count for a given model
# 使用 ptflops 打印模型参数量和 MACs 的辅助函数。这是另一套复杂度统计入口，使用 ptflops 而不是 THOP。
# input_size 使用不含 batch的 CHW 格式（默认 [3,224,224]），ptflops 会自行添加 batch=1 并尝试分析各层。
# 该函数主要用于交互式比较模型，不参与训练或测试；若模型含有 ptflops 不认识的自定义模块，可能需要额外的 hook 或会出现统计警告。
def print_model_stats(model, input_size=(3, 224, 224)):
    # Print model parameter count
    # 遍历全部参数张量并累计元素数量，不区分是否 requires_grad。
    # 这里与 cal_params_flops 的直接计数相同，包含被冻结的参数；如果只想统计可训练
    # 参数，需要额外筛选 requires_grad，但当前代码保留“模型总参数量”的口径。
    total_params = sum(p.numel() for p in model.parameters())
    # 输出精确参数个数。
    print(f'Model created, param count: {total_params}')
    
    # Calculate GMACs using ptflops
    # ptflops 按给定 CHW 输入尺寸分析模型，并输出逐层统计。
    # as_strings=True 让返回值直接带 K/M/G 等单位，便于打印；
    # print_per_layer_stat=True 会输出每层详情，适合定位计算量主要来自哪里，但会
    # 产生较多终端输出。这里的 params 字符串可能与上面的精确整数略有格式差异。
    macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)
    
    # Display GMACs and params
    # 打印 ptflops 返回的可读 MACs 和参数量字符串。
    print(f'Model: {macs} GMACs, {params} parameters')
