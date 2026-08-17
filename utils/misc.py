# os 仅供 check_mkdir 查询和创建目录。
import os
# ceil 用于把滑窗步数向上取整，保证图像右侧和下侧边缘也能被窗口覆盖。
from math import ceil

# NumPy 用于生成双线性插值核、混淆矩阵以及规则采样网格。
import numpy as np
# torch/F/nn 提供张量、函数式插值采样和神经网络层。
import torch
# F 提供 log_softmax、softmax、padding、插值和 grid_sample 等无状态算子。
import torch.nn.functional as F
# nn 提供 Module、Conv2d、Linear、BatchNorm2d、NLLLoss2d 和 Parameter。
from torch import nn
# Variable 是旧版 PyTorch 的张量自动求导包装；现代 PyTorch 中 Tensor 已具备该能力。
from torch.autograd import Variable


# 仅在目录不存在时创建单级目录；父目录必须已经存在。
# 这是通用文件辅助函数，与 EMCAD 论文的模型结构无直接对应关系。
def check_mkdir(dir_name):
# exists 对文件和目录都返回 True，因此同名普通文件存在时本函数不会主动报出更明确的目录错误。
    if not os.path.exists(dir_name):
# os.mkdir 不递归创建父目录，也没有 exist_ok 参数。
        os.mkdir(dir_name)


# 对传入的一个或多个模型统一初始化卷积、全连接和二维 BatchNorm。
# EMCAD 主网络在 lib/decoders.py、lib/pvtv2.py 等文件中有各自初始化流程；
# 本函数只有被其他调用方显式执行时才会覆盖那些参数。
def initialize_weights(*models):
# 支持一次传入多个独立模型或子模块。
    for model in models:
# modules() 会递归遍历模型本身及所有已注册子模块。
        for module in model.modules():
# 卷积和全连接权重使用适合 ReLU 网络的 Kaiming 正态初始化。
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
# 当前代码使用旧式无下划线接口 nn.init.kaiming_normal；保留其现有行为。
                nn.init.kaiming_normal(module.weight)
# 只有确实定义 bias 的层才把偏置清零。
                if module.bias is not None:
# 直接写入参数 data，保持这段旧初始化代码的原始行为。
                    module.bias.data.zero_()
# BatchNorm 的缩放参数初始化为 1、平移参数初始化为 0，使初始变换接近恒等映射。
            elif isinstance(module, nn.BatchNorm2d):
# gamma=1 保留归一化后的幅度。
                module.weight.data.fill_(1)
# beta=0 不引入额外平移。
                module.bias.data.zero_()


# 构造可用于转置卷积初始化的二维双线性上采样核。
# 返回形状为 [in_channels,out_channels,kernel_size,kernel_size] 的 float32 张量。
def get_upsampling_weight(in_channels, out_channels, kernel_size):
# factor 是三角插值函数从中心衰减到零所需的尺度。
    factor = (kernel_size + 1) // 2
# 奇数核的中心正好落在整数像素上。
    if kernel_size % 2 == 1:
        center = factor - 1
# 偶数核的几何中心落在两个像素之间，因此使用半像素中心。
    else:
        center = factor - 0.5
# ogrid 生成可广播的纵向和横向坐标，不创建完整坐标网格副本。
    og = np.ogrid[:kernel_size, :kernel_size]
# 两个一维线性帐篷函数相乘，形成可分离的二维双线性插值核。
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
# 先创建全零的转置卷积权重数组。
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
# 只把输入/输出通道的对应“对角项”填成相同插值核，通道之间不混合。
# 该高级索引通常要求 in_channels 与 out_channels 可一一对应。
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
# NumPy float64 最终转换为神经网络常用的 PyTorch float32。
    return torch.from_numpy(weight).float()


# 旧式二维多分类交叉熵包装：先对类别通道做 log_softmax，再交给 NLLLoss2d。
# 该类不是当前 EMCAD 二分类 structure_loss，也不对应论文的加权 BCE+IoU。
class CrossEntropyLoss2d(nn.Module):
# weight 是类别权重，ignore_index 指定不参与损失的标签值。
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
# NLLLoss2d 和位置参数 size_average 属于旧版 PyTorch API，代码保持原有接口。
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

# inputs 应为 [B,C,H,W] logits，targets 应为 [B,H,W] 整数类别索引。
    def forward(self, inputs, targets):
# log_softmax 把 logits 转成对数概率；当前旧代码未显式写 dim，行为依赖所用 PyTorch 版本。
        return self.nll_loss(F.log_softmax(inputs), targets)


# 在逐像素多分类 NLL 前乘以 (1-p)^gamma，使高置信度易样本的贡献衰减。
# 这是通用 focal-loss 风格实现，与 EMCAD 论文主损失没有直接对应关系。
class FocalLoss2d(nn.Module):
# gamma 越大，对易分类像素的抑制越强；其余参数传给 NLLLoss2d。
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

# inputs 是未归一化 logits，targets 是类别索引图。
    def forward(self, inputs, targets):
# softmax 概率产生调制因子，log_softmax 提供对数概率；两者逐元素相乘后由 NLL 聚合。
# 这里同样沿用未显式指定 dim 的旧式函数调用。
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)


# 用一次 bincount 高效构造 num_classes x num_classes 混淆矩阵。
def _fast_hist(label_pred, label_true, num_classes):
# 只保留合法真值类别；负标签和大于等于类别数的 ignore 标签被排除。
    mask = (label_true >= 0) & (label_true < num_classes)
# 把二维坐标 (真实类别,预测类别) 编码成一维索引 true*C+pred，再恢复成矩阵。
    hist = np.bincount(
# 把真值类别作为“行”、预测类别作为“列”编码到一维桶编号中。
        num_classes * label_true[mask].astype(int) +
# minlength 即使某些类别完全没出现，也保证最终可 reshape 为固定方阵。
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
# 返回单个预测/真值对的混淆矩阵，供 evaluate 跨样本累加。
    return hist


# 汇总一组预测/真值标签图，返回像素准确率、平均类别准确率、mIoU 和频率加权IoU。
def evaluate(predictions, gts, num_classes):
# 混淆矩阵使用浮点数组，便于后面的除法和跨病例累加。
    hist = np.zeros((num_classes, num_classes))
# zip 按位置配对预测和真值；若两列表长度不同，超出的部分不会参与评估。
    for lp, lt in zip(predictions, gts):
# flatten 把任意空间形状展开成逐像素一维序列。
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
# 对角线是预测正确像素；除以全部有效像素得到总体像素准确率。
    acc = np.diag(hist).sum() / hist.sum()
# 每个真实类别的对角元素除以该行总数，得到逐类召回/类别准确率。
    acc_cls = np.diag(hist) / hist.sum(axis=1)
# 没有真值样本的类别会产生 NaN，nanmean 将其排除。
    acc_cls = np.nanmean(acc_cls)
# 每类 IoU=交集/(真实总数+预测总数-交集)。
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
# mIoU 对有定义的类别取平均。
    mean_iu = np.nanmean(iu)
# 类别频率是真值中各类像素占比。
    freq = hist.sum(axis=1) / hist.sum()
# 频率加权IoU只累加真实频率大于0的类别。
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
# 返回顺序由旧调用方约定，修改顺序会改变外部解包语义。
    return acc, acc_cls, mean_iu, fwavacc


# 保存流式标量的当前值、加权总和、样本计数和全程平均值。
class AverageMeter(object):
# 构造时复用 reset，避免初始化逻辑重复。
    def __init__(self):
# 构造时立即把当前值、累计和、计数和平均值全部归零。
        self.reset()

# 清空全部累计状态，开始新的统计区间。
    def reset(self):
# 最近一次 update 输入值。
        self.val = 0
# 从 reset 起的加权平均值。
        self.avg = 0
# val*n 的累计和。
        self.sum = 0
# 权重或样本数累计和。
        self.count = 0

# 用当前值 val 代表 n 个样本更新统计量。
    def update(self, val, n=1):
# 保存最近一次传入的标量，便于日志展示当前 batch。
        self.val = val
# 累计加权总和，而不是简单累计 batch 均值。
        self.sum += val * n
# 累加本次值所代表的样本数或权重。
        self.count += n
# count 为正时得到从 reset 至今的总体加权平均。
        self.avg = self.sum / self.count


# 多项式学习率调度器：lr=初始lr*(1-curr_iter/max_iter)^lr_decay。
class PolyLR(object):
# 当前迭代数由调用方传入；本类 step() 不会自行递增 curr_iter。
    def __init__(self, optimizer, curr_iter, max_iter, lr_decay):
# 转成浮点避免旧版 Python/张量环境中的整数除法歧义。
        self.max_iter = float(max_iter)
# 分别记录每个优化器参数组的初始学习率，支持不同组使用不同基准值。
        self.init_lr_groups = []
# 逐参数组保存各自的初始学习率，避免假设所有组使用同一数值。
        for p in optimizer.param_groups:
# append 保留参数组顺序，step 中用相同索引配对。
            self.init_lr_groups.append(p['lr'])
# 保留优化器参数组对象的引用，step 会原地修改其中 lr。
        self.param_groups = optimizer.param_groups
# curr_iter 是本次计算学习率所用的全局迭代编号。
        self.curr_iter = curr_iter
# lr_decay 是多项式指数，常见取值如0.9。
        self.lr_decay = lr_decay

# 按构造时保存的 curr_iter 更新所有参数组的学习率。
    def step(self):
# 每个参数组都在自己的初始学习率基础上应用同一衰减比例。
        for idx, p in enumerate(self.param_groups):
# 若调用方不更新 self.curr_iter 或重新创建调度器，多次 step 会得到同一个学习率。
            p['lr'] = self.init_lr_groups[idx] * (1 - self.curr_iter / self.max_iter) ** self.lr_decay


# just a try, not recommend to use
# 实验性“可变形卷积”包装器：先预测每个输入通道的二维偏移，再重采样后执行普通卷积。
# 这不是 EMCAD 论文提出的模块，当前主模型也不从 lib/decoders.py 调用它。
class Conv2dDeformable(nn.Module):
# regular_filter 是最终作用在重采样特征上的普通 Conv2d；cuda 控制缓存网格所在设备。
    def __init__(self, regular_filter, cuda=True):
# 注册父类状态，确保内部卷积和参数被 PyTorch 跟踪。
        super(Conv2dDeformable, self).__init__()
# 该包装器只接受二维卷积，其他可调用层会在构造时触发断言。
        assert isinstance(regular_filter, nn.Conv2d)
# 保存调用方提供的实际卷积核；输入先变形，再交给它做卷积。
        self.regular_filter = regular_filter
# 每个输入通道预测水平和垂直两个偏移，所以输出通道数为 2*C_in。
        self.offset_filter = nn.Conv2d(regular_filter.in_channels, 2 * regular_filter.in_channels, kernel_size=3,
# padding=1 保持偏移场的 H、W 与输入一致；不使用偏置。
                                       padding=1, bias=False)
# 用很小的正态噪声初始化偏移卷积，使初始采样位置接近规则网格。
        self.offset_filter.weight.data.normal_(0, 0.0005)
# 缓存上一次输入形状；形状改变时重新创建规则采样网格。
        self.input_shape = None
# 水平归一化坐标网格，首次 forward 时创建。
        self.grid_w = None
# 垂直归一化坐标网格，首次 forward 时创建。
        self.grid_h = None
# 旧实现用布尔值决定是否直接调用 .cuda()；它不支持任意 device 参数。
        self.cuda = cuda

# 输入 x 形状应为 (B,C,H,W)，返回普通卷积 regular_filter 的输出。
    def forward(self, x):
# 保存四维形状，后续把 B、C 合并为 grid_sample 的批维。
        x_shape = x.size()  # (b, c, h, w)
# 学习得到 (B,2C,H,W) 偏移场，前 C 个通道和后 C 个通道分别表示两个坐标方向。
        offset = self.offset_filter(x)  # (b, 2*c, h, w)
# 沿通道维等分为水平与垂直偏移，各自形状均为 (B,C,H,W)。
        offset_w, offset_h = torch.split(offset, self.regular_filter.in_channels, 1)  # (b, c, h, w)
# 把 B 和 C 合并，使每个原输入通道作为独立的 grid_sample 样本。
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
# 垂直偏移采用相同的展平方式，保持与水平偏移逐元素对齐。
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
# 第一次调用或 B/C/H/W 任一维变化时，重建 [-1,1] 规则网格。
        if not self.input_shape or self.input_shape != x_shape:
# 更新缓存形状，后续同形状输入复用网格。
            self.input_shape = x_shape
# meshgrid 生成 HxW 的水平/垂直坐标；grid_sample 使用归一化坐标范围 [-1,1]。
            grid_w, grid_h = np.meshgrid(np.linspace(-1, 1, x_shape[3]), np.linspace(-1, 1, x_shape[2]))  # (h, w)
# 把 NumPy 水平网格转为 PyTorch 浮点张量。
            grid_w = torch.Tensor(grid_w)
# 把 NumPy 垂直网格转为 PyTorch 浮点张量。
            grid_h = torch.Tensor(grid_h)
# 旧代码在 cuda=True 时把两个网格直接移到默认 CUDA 设备。
            if self.cuda:
# 水平网格迁移到 GPU。
                grid_w = grid_w.cuda()
# 垂直网格迁移到 GPU。
                grid_h = grid_h.cuda()
# 把规则网格包装为 Parameter；注意它在 forward 中创建并会被优化器视为可训练参数。
            self.grid_w = nn.Parameter(grid_w)
# 垂直网格同样注册为 Parameter。
            self.grid_h = nn.Parameter(grid_h)
# 广播规则水平坐标到 B*C 个采样平面，并叠加学习偏移。
        offset_w = offset_w + self.grid_w  # (b*c, h, w)
# 广播规则垂直坐标并叠加学习偏移。
        offset_h = offset_h + self.grid_h  # (b*c, h, w)
# 把原输入也折叠为 (B*C,1,H,W)，与逐通道采样网格对应。
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3])).unsqueeze(1)  # (b*c, 1, h, w)
# 把两个坐标堆成网格最后一维并做双线性重采样；未显式参数沿用当前 PyTorch 默认值。
        x = F.grid_sample(x, torch.stack((offset_h, offset_w), 3))  # (b*c, h, w)
# 将折叠的 B*C 恢复为原来的 (B,C,H,W) 通道布局。
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))  # (b, c, h, w)
# 对完成空间变形的特征执行调用方传入的普通卷积。
        x = self.regular_filter(x)
# 返回卷积结果；本包装器不额外返回偏移场。
        return x


# 装饰器：把一次整图 forward 改造成“多尺度缩放 + 必要时滑窗切片 + 重叠区域平均”的 forward。
# 它依赖被包装模型提供 use_aux、scales、crop_size、stride_rate 和 num_classes 等属性；EMCAD 主流程未使用它。
def sliced_forward(single_forward):
# 内部补边函数只在右侧和底部补零，使小图至少达到 crop_size x crop_size。
    def _pad(x, crop_size):
# 读取当前张量空间高宽，忽略 batch 和 channel 维。
        h, w = x.size()[2:]
# 高度不足时计算需补的行数；已足够大时为 0。
        pad_h = max(crop_size - h, 0)
# 宽度不足时计算需补的列数。
        pad_w = max(crop_size - w, 0)
# F.pad 的四元组顺序为 left,right,top,bottom，因此这里只补右、下两边。
        x = F.pad(x, (0, pad_w, 0, pad_h))
# 同时返回补边量，调用方在推理后裁掉人造区域。
        return x, pad_h, pad_w

# wrapper 将替代原单尺度 forward；self 是被装饰的模型实例。
    def wrapper(self, x):
# 保存 batch 和原始 H/W，最终累计画布按该尺寸创建。
        batch_size, _, ori_h, ori_w = x.size()
# 训练且开启辅助头时，single_forward 预期返回主输出和辅助输出两个张量。
        if self.training and self.use_aux:
# 创建主输出的全尺度累加画布；旧代码无条件放到默认 CUDA 设备。
            outputs_all_scales = Variable(torch.zeros((batch_size, self.num_classes, ori_h, ori_w))).cuda()
# 辅助输出使用另一张同尺寸累加画布。
            aux_all_scales = Variable(torch.zeros((batch_size, self.num_classes, ori_h, ori_w))).cuda()
# 依次处理模型配置的每个缩放倍率。
            for s in self.scales:
# 将原始空间尺寸乘以当前比例并取整，形成该尺度目标 H/W。
                new_size = (int(ori_h * s), int(ori_w * s))
# 旧接口 F.upsample 对输入做双线性缩放；未显式 align_corners，行为取决于 PyTorch 版本。
                scaled_x = F.upsample(x, size=new_size, mode='bilinear')
# 再包装为旧式 Variable 并迁移到默认 CUDA 设备。
                scaled_x = Variable(scaled_x).cuda()
# 读取实际缩放后高宽。
                scaled_h, scaled_w = scaled_x.size()[2:]
# 用长边判断该尺度能否一次送入 crop_size 窗口。
                long_size = max(scaled_h, scaled_w)
# 调试输出：打印当前尺度张量形状；这是原实现的可见副作用。
                print(scaled_x.size())

# 任一空间边超过裁剪尺寸时，采用重叠滑窗推理。
                if long_size > self.crop_size:
# count 记录每个像素被多少窗口覆盖，后面用于重叠区域求平均。
                    count = torch.zeros((scaled_h, scaled_w))
# 主输出在当前缩放尺寸上的累计画布。
                    outputs = Variable(torch.zeros((batch_size, self.num_classes, scaled_h, scaled_w))).cuda()
# 辅助输出在当前缩放尺寸上的累计画布。
                    aux_outputs = Variable(torch.zeros((batch_size, self.num_classes, scaled_h, scaled_w))).cuda()
# 步长由裁剪边长乘 stride_rate 后向下取整得到。
                    stride = int(ceil(self.crop_size * self.stride_rate))
# 向上取整并加 1，确保高度方向最后一个窗口覆盖到下边缘。
                    h_step_num = int(ceil((scaled_h - self.crop_size) / stride)) + 1
# 宽度方向使用相同规则覆盖右边缘。
                    w_step_num = int(ceil((scaled_w - self.crop_size) / stride)) + 1
# 外层循环枚举窗口的纵向编号。
                    for yy in range(h_step_num):
# 内层循环枚举横向编号，二者笛卡尔积覆盖整幅图。
                        for xx in range(w_step_num):
# 由窗口编号和 stride 得到左上角坐标。
                            sy, sx = yy * stride, xx * stride
# 右下角按固定 crop_size 推进；靠边窗口可能超出真实尺寸。
                            ey, ex = sy + self.crop_size, sx + self.crop_size
# 从缩放图中截取当前窗口；Python 切片会自动截断超出边界的终点。
                            x_sub = scaled_x[:, :, sy: ey, sx: ex]
# 对边缘处不足 crop_size 的窗口补到固定大小，并记录补边量。
                            x_sub, pad_h, pad_w = _pad(x_sub, self.crop_size)
# 原实现保留的逐窗口形状调试输出。
                            print(x_sub.size())
# 调用原始 forward，同时得到当前窗口的主输出和辅助输出。
                            outputs_sub, aux_sub = single_forward(self, x_sub)

# 当窗口底边越过缩放图时，从预测底部裁掉 _pad 添加的行。
                            if sy + self.crop_size > scaled_h:
# pad_h 在该分支应为正；负切片终点删除相应尾部行。
                                outputs_sub = outputs_sub[:, :, : -pad_h, :]
# 辅助预测必须做完全相同的裁剪，才能与主输出和目标区域对齐。
                                aux_sub = aux_sub[:, :, : -pad_h, :]

# 当窗口右边越过缩放图时，从预测右侧裁掉补出的列。
                            if sx + self.crop_size > scaled_w:
# 只裁宽度维，保持此前可能已完成的高度裁剪。
                                outputs_sub = outputs_sub[:, :, :, : -pad_w]
# 辅助输出同步裁宽度维。
                                aux_sub = aux_sub[:, :, :, : -pad_w]

# 把当前窗口主预测累加到其在整幅缩放图上的对应区域。
                            outputs[:, :, sy: ey, sx: ex] = outputs_sub
# 把辅助预测写入对应区域；原代码使用赋值而非 +=，保持其现有覆盖行为。
                            aux_outputs[:, :, sy: ey, sx: ex] = aux_sub

# 记录这个窗口覆盖的像素；重叠位置最终 count 会大于 1。
                            count[sy: ey, sx: ex] += 1
# 将覆盖计数包装成旧式 Variable 并迁移到 CUDA，便于与预测张量相除。
                    count = Variable(count).cuda()
# 主输出按覆盖次数归一化；由于上面是赋值而非累加，重叠区行为应按现代码理解。
                    outputs = (outputs / count)
# 注意这里原代码用 outputs/count 赋给 aux_outputs，而不是 aux_outputs/count；仅做说明，不改行为。
                    aux_outputs = (outputs / count)
# 若缩放图长边不超过 crop_size，则整图补边后只调用一次 single_forward。
                else:
# 把小图补到固定裁剪尺寸，并保存补边量。
                    scaled_x, pad_h, pad_w = _pad(scaled_x, self.crop_size)
# 一次得到主输出与辅助输出。
                    outputs, aux_outputs = single_forward(self, scaled_x)
# 裁去补边；原表达式在 pad_h 或 pad_w 为 0 时会出现 `:-0` 语义，保持原代码不动。
                    outputs = outputs[:, :, : -pad_h, : -pad_w]
# 辅助输出按同样切片规则恢复到缩放图尺寸。
                    aux_outputs = aux_outputs[:, :, : -pad_h, : -pad_w]
# 把当前尺度主输出加到原尺寸累加画布；调用方需保证不同尺度结果可相加或内部已恢复尺寸。
                outputs_all_scales += outputs
# 辅助输出也跨尺度累加。
                aux_all_scales += aux_outputs
# 主输出除以尺度数得到均值；辅助输出按原实现直接返回累加值，不除尺度数。
            return outputs_all_scales / len(self.scales), aux_all_scales
# 推理模式或未启用辅助头时，只维护一个主输出分支。
        else:
# 创建原始 H/W 尺寸的主输出累计画布并放到默认 CUDA 设备。
            outputs_all_scales = Variable(torch.zeros((batch_size, self.num_classes, ori_h, ori_w))).cuda()
# 逐个尺度执行与训练分支相同的缩放和滑窗策略。
            for s in self.scales:
# 计算当前尺度目标尺寸。
                new_size = (int(ori_h * s), int(ori_w * s))
# 对输入做双线性缩放；这里没有再显式 Variable(...).cuda()，沿用输入设备。
                scaled_x = F.upsample(x, size=new_size, mode='bilinear')
# 读取缩放后高宽。
                scaled_h, scaled_w = scaled_x.size()[2:]
# 用长边决定是否需要切片。
                long_size = max(scaled_h, scaled_w)

# 大于 crop_size 时创建当前尺度画布并滑窗处理。
                if long_size > self.crop_size:
# 保存每个位置的窗口覆盖次数。
                    count = torch.zeros((scaled_h, scaled_w))
# 创建当前尺度的类别 logits/score 画布。
                    outputs = Variable(torch.zeros((batch_size, self.num_classes, scaled_h, scaled_w))).cuda()
# 从裁剪大小与重叠比例得到滑动步长。
                    stride = int(ceil(self.crop_size * self.stride_rate))
# 计算纵向窗口数量，确保覆盖底边。
                    h_step_num = int(ceil((scaled_h - self.crop_size) / stride)) + 1
# 计算横向窗口数量，确保覆盖右边。
                    w_step_num = int(ceil((scaled_w - self.crop_size) / stride)) + 1
# 遍历纵向窗口索引。
                    for yy in range(h_step_num):
# 遍历横向窗口索引。
                        for xx in range(w_step_num):
# 当前窗口左上角。
                            sy, sx = yy * stride, xx * stride
# 当前窗口理论右下角。
                            ey, ex = sy + self.crop_size, sx + self.crop_size
# 从当前尺度输入中截取实际存在的区域。
                            x_sub = scaled_x[:, :, sy: ey, sx: ex]
# 对边缘不足固定大小的区域补零。
                            x_sub, pad_h, pad_w = _pad(x_sub, self.crop_size)

# 单头分支的原始 forward 只返回主预测。
                            outputs_sub = single_forward(self, x_sub)

# 底边越界时去掉补零对应的预测行。
                            if sy + self.crop_size > scaled_h:
# 恢复实际窗口高度。
                                outputs_sub = outputs_sub[:, :, : -pad_h, :]

# 右边越界时去掉补零对应的预测列。
                            if sx + self.crop_size > scaled_w:
# 恢复实际窗口宽度。
                                outputs_sub = outputs_sub[:, :, :, : -pad_w]

# 把窗口结果写回当前尺度画布；这里同样是赋值，不是重叠累加。
                            outputs[:, :, sy: ey, sx: ex] = outputs_sub

# 累计当前窗口覆盖次数。
                            count[sy: ey, sx: ex] += 1
# 迁移覆盖计数到 CUDA 以匹配 outputs。
                    count = Variable(count).cuda()
# 按覆盖次数归一化当前尺度画布。
                    outputs = (outputs / count)
# 小图路径：补到固定大小后整图前向一次。
                else:
# 补右侧和底部并记录补边量。
                    scaled_x, pad_h, pad_w = _pad(scaled_x, self.crop_size)
# 调用原始单输出 forward。
                    outputs = single_forward(self, scaled_x)
# 裁掉补边区域；保留原实现对 0 补边时 `:-0` 的既有切片行为。
                    outputs = outputs[:, :, : -pad_h, : -pad_w]
# 将当前尺度输出累加到全尺度画布；原实现最终不除以尺度数。
                outputs_all_scales += outputs
# 返回所有尺度的累加结果。
            return outputs_all_scales

# 装饰器工厂返回替代原 forward 的 wrapper 函数。
    return wrapper
