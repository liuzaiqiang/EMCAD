# 导入 PyTorch 张量运算；本文件中的特征相加、拼接、转置都依赖 torch。
import torch
# 导入神经网络模块别名，卷积、归一化、激活和 ModuleList 均从 nn 创建。
import torch.nn as nn
# partial 用来预先固定权重初始化函数的 scheme 参数，再交给 named_apply 递归调用。
from functools import partial

# math.sqrt 用于依据卷积层 fan-out 计算默认初始化标准差。
import math
# timm 的截断正态分布初始化函数，对应可选的 trunc_normal 初始化方案。
from timm.models.layers import trunc_normal_tf_
# named_apply 会递归访问当前模块及其子模块，并把模块名称一并传给初始化函数。
from timm.models.helpers import named_apply


# 计算两个正整数的最大公约数；MSCB 用它确定 channel shuffle 的分组数。
def gcd(a, b):
# 欧几里得算法：只要余数除数 b 还不为 0，就继续迭代。
    while b:
# 新的 a 取旧 b，新的 b 取 a 对 b 的余数。
        a, b = b, a % b
# b 为 0 时，a 即最大公约数。
    return a

# Other types of layers can go here (e.g., nn.Linear, etc.)
# named_apply 调用的统一初始化入口；name 是遍历得到的模块路径，当前实现不使用它。
def _init_weights(module, name, scheme=''):
# 卷积层包含主要空间参数，二维和三维卷积共用下面的初始化分支。
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
# normal 对卷积核使用标准差 0.02 的普通正态分布。
        if scheme == 'normal':
# 只原地改写参数值，不替换 Parameter 对象。
            nn.init.normal_(module.weight, std=.02)
# 只有显式启用 bias 的卷积层才需要初始化偏置。
            if module.bias is not None:
# 偏置从 0 开始，避免初始化阶段人为平移激活。
                nn.init.zeros_(module.bias)
# trunc_normal 使用截断正态分布，降低极端初始权重出现的概率。
        elif scheme == 'trunc_normal':
# timm 的 TensorFlow 风格截断正态实现，标准差仍为 0.02。
            trunc_normal_tf_(module.weight, std=.02)
# 与 normal 分支一样，仅在偏置存在时清零。
            if module.bias is not None:
# 清零卷积偏置。
                nn.init.zeros_(module.bias)
# Xavier normal 按输入、输出扇入扇出共同缩放权重。
        elif scheme == 'xavier_normal':
# 对卷积权重应用 Xavier 正态初始化。
            nn.init.xavier_normal_(module.weight)
# 检查偏置是否存在。
            if module.bias is not None:
# 偏置初始化为 0。
                nn.init.zeros_(module.bias)
# Kaiming normal 更适合后接 ReLU 系列激活的卷积层。
        elif scheme == 'kaiming_normal':
# fan_out 模式尽量保持反向传播梯度尺度，nonlinearity 指明按 ReLU 计算增益。
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
# 检查可选偏置。
            if module.bias is not None:
# 偏置初始化为 0。
                nn.init.zeros_(module.bias)
# 未指定上述方案时，采用类似 EfficientNet 的卷积初始化。
        else:
            # efficientnet like
# kernel_h * kernel_w * out_channels 给出未考虑分组时的 fan-out。
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
# 分组卷积中每组独立计算，因此 fan-out 还要除以 groups。
            fan_out //= module.groups
# 以 sqrt(2/fan_out) 为标准差采样，适配 ReLU 类非线性。
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
# 检查卷积是否含偏置。
            if module.bias is not None:
# 偏置清零。
                nn.init.zeros_(module.bias)
# BatchNorm 的可学习缩放和平移按恒等变换初始化。
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
# gamma=1，使初始归一化结果不被额外缩放。
        nn.init.constant_(module.weight, 1)
# beta=0，使初始归一化结果不被额外平移。
        nn.init.constant_(module.bias, 0)
# Transformer/PVT 中的 LayerNorm 也按恒等仿射变换初始化。
    elif isinstance(module, nn.LayerNorm):
# LayerNorm 的缩放参数设为 1。
        nn.init.constant_(module.weight, 1)
# LayerNorm 的偏置参数设为 0。
        nn.init.constant_(module.bias, 0)

# 根据字符串创建激活层，统一供解码器各子模块复用。
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
# 转成小写，允许调用方传入 ReLU、RELU 等不同大小写写法。
    act = act.lower()
# 标准 ReLU：负值截断为 0。
    if act == 'relu':
# inplace 决定是否复用输入张量存储空间。
        layer = nn.ReLU(inplace)
# ReLU6 将正激活上限截断到 6；论文第4页 MSCB 的式(4)-(6)采用 ReLU6。
    elif act == 'relu6':
# 创建 ReLU6 层。
        layer = nn.ReLU6(inplace)
# LeakyReLU 为负区间保留 neg_slope 比例的梯度。
    elif act == 'leakyrelu':
# 创建带指定负斜率的 LeakyReLU。
        layer = nn.LeakyReLU(neg_slope, inplace)
# PReLU 将负斜率变成可训练参数。
    elif act == 'prelu':
# n_prelu 决定共享一个斜率还是为多个通道分别学习斜率。
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
# GELU 常用于 Transformer 前馈网络。
    elif act == 'gelu':
# 创建无额外参数的 GELU。
        layer = nn.GELU()
# Hardswish 是适合轻量网络的分段近似激活。
    elif act == 'hswish':
# 创建 Hardswish 层。
        layer = nn.Hardswish(inplace)
# 对未知字符串立即报错，避免静默使用错误激活。
    else:
# 错误消息包含调用方传入的激活名称。
        raise NotImplementedError('activation layer [%s] is not found' % act)
# 返回构造好的 nn.Module，供 Sequential 直接使用。
    return layer

# ShuffleNet 风格的通道重排：让原本不同 group 的通道发生信息交换。
def channel_shuffle(x, groups):
# 输入约定为 NCHW；提取 B、C、H、W。这里使用 x.data.size() 是原工程写法。
    batchsize, num_channels, height, width = x.data.size()
# 每组通道数必须是整数，因此调用前要求 C 能被 groups 整除。
    channels_per_group = num_channels // groups    
    # reshape
# 将通道维拆成 groups 和 channels_per_group 两维，形状变为 (B,G,C/G,H,W)。
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
# 交换“组”和“组内通道”两维，再 contiguous 生成可安全 view 的连续布局。
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
# 合并两个通道子维，恢复为 (B,C,H,W)，但通道次序已经重排。
    x = x.view(batchsize, -1, height, width)
# 返回形状不变、顺序改变后的特征。
    return x

#   Multi-scale depth-wise convolution (MSDC)
# 论文第4页 Sec.3.1.2、Fig.2(f)、式(5)-(6)：用多个核尺寸的深度卷积提取多尺度局部信息。
class MSDC(nn.Module):
# in_channels 是每个深度卷积分支的通道数；kernel_sizes 通常为 [1,3,5]。
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
# 初始化 nn.Module 的内部注册结构。
        super(MSDC, self).__init__()

# 保存输入通道数；深度卷积令 groups=in_channels，因此输入输出通道相同。
        self.in_channels = in_channels
# 保存多尺度卷积核列表，每个尺寸对应一个分支。
        self.kernel_sizes = kernel_sizes
# 保存每个分支使用的激活类型。
        self.activation = activation
# True 表示所有分支读取同一个输入；False 表示递归更新输入，近似论文式(6)。
        self.dw_parallel = dw_parallel

# ModuleList 负责注册数量由 kernel_sizes 动态决定的多个分支。
        self.dwconvs = nn.ModuleList([
# 每个尺度均按“深度卷积 -> BN -> 激活”组成 DWCB。
            nn.Sequential(
# groups=C 使每个输入通道独立卷积；padding=k//2 在 stride=1 时保持 H、W。
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
# 对每个通道的卷积结果做批归一化。
                nn.BatchNorm2d(self.in_channels),
# 默认 ReLU6，对应论文式(5)中的 R6。
                act_layer(self.activation, inplace=True)
            )
# 对 kernel_sizes 中每个核尺寸建立一条独立分支。
            for kernel_size in self.kernel_sizes
        ])

# 解码器新建时立即初始化本模块及其所有子层。
        self.init_weights('normal')
    
# scheme 可切换初始化方案；当前 EMCAD 构造路径传入 normal。
    def init_weights(self, scheme=''):
# 递归调用 _init_weights，name 参数由 named_apply 自动提供。
        named_apply(partial(_init_weights, scheme=scheme), self)

# 输入 x 形状为 (B,C_ex,H,W)，返回长度为尺度数的特征列表。
    def forward(self, x):
        # Apply the convolution layers in a loop
# 保存每个尺度 DWCB 的独立输出，而不是在此处直接合并。
        outputs = []
# 依次执行 1x1、3x3、5x5 等深度卷积分支。
        for dwconv in self.dwconvs:
# 并行模式下每次读取原始 x；串行模式下读取上一轮残差更新后的 x。
            dw_out = dwconv(x)
# 每个 dw_out 形状均为 (B,C_ex,H/stride,W/stride)。
            outputs.append(dw_out)
# 工程扩展：关闭并行时采用论文式(6)的递归残差输入。
            if self.dw_parallel == False:
# 更新下一分支的输入；默认解码器 stride=1，因此空间尺寸可直接相加。
                x = x+dw_out
        # You can return outputs based on what you intend to do with them
# MSCB 再根据 add 参数对这些分支求和或拼接。
        return outputs

# 论文第4页 Sec.3.1.2、Fig.2(e)、式(4)：MobileNetV2 倒残差思想上的多尺度卷积块。
class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB) 
    """
# 关键参数：C_in、C_out、步幅、多尺度核、扩张倍率、并/串行和分支聚合方式。
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
# 初始化基础模块。
        super(MSCB, self).__init__()
        
# 输入通道数 C_in。
        self.in_channels = in_channels
# 输出通道数 C_out。
        self.out_channels = out_channels
# stride=1 保持分辨率并启用残差；stride=2 时空间下采样且不使用本块残差。
        self.stride = stride
# MSDC 使用的卷积核尺寸列表。
        self.kernel_sizes = kernel_sizes
# 通道扩张倍率 e；论文默认 e=2。
        self.expansion_factor = expansion_factor
# 控制 MSDC 各尺度是并行还是递归串行。
        self.dw_parallel = dw_parallel
# True 对尺度输出逐元素相加；False 是论文外保留的 concat 工程选项。
        self.add = add
# MSCB 内部的激活类型，论文配置为 ReLU6。
        self.activation = activation
# 记录尺度分支数量，concat 时用于计算拼接后的通道数。
        self.n_scales = len(self.kernel_sizes)
        # check stride value
# 当前实现只允许保持尺寸或二倍下采样。
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
# 只有 stride=1 时主分支与捷径分支空间尺寸一致，才允许残差相加。
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
# 扩张后的中间通道 C_ex=int(C_in*e)。默认 e=2 时为 2C_in。
        self.ex_channels = int(self.in_channels * self.expansion_factor)
# 第一个 point-wise convolution 实现倒残差块的通道扩张。
        self.pconv1 = nn.Sequential(
            # pointwise convolution
# 1x1 卷积把 (B,C_in,H,W) 映射为 (B,C_ex,H,W)。
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
# 对 C_ex 个扩张通道做归一化。
            nn.BatchNorm2d(self.ex_channels),
# 默认训练脚本传 ReLU6，对应论文式(4)。
            act_layer(self.activation, inplace=True)
        )
# MSDC 在 C_ex 通道上执行多个深度卷积尺度。
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation, dw_parallel=self.dw_parallel)
# 加法聚合不改变通道数，合并后仍是 C_ex。
        if self.add == True:
# 乘 1 明确记录加法后的通道数。
            self.combined_channels = self.ex_channels*1
# concat 工程选项会沿通道维连接所有尺度。
        else:
# 拼接后通道数为 C_ex*n_scales；这不是论文式(5)的默认求和路径。
            self.combined_channels = self.ex_channels*self.n_scales
# 第二个 point-wise convolution 负责融合通道并投影到 C_out。
        self.pconv2 = nn.Sequential(
            # pointwise convolution
# 输入是合并后的通道，输出形状为 (B,C_out,H/stride,W/stride)。
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False), 
# 输出端 BN；这里没有额外激活，便于随后执行残差相加。
            nn.BatchNorm2d(self.out_channels),
        )
# stride=1 但 C_in!=C_out 时，捷径分支需用 1x1 卷积对齐通道。
        if self.use_skip_connection and (self.in_channels != self.out_channels):
# 捷径只改通道，不改变 H、W。
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
# 初始化当前块全部卷积与归一化层。
        self.init_weights('normal')
    
# 统一权重初始化入口。
    def init_weights(self, scheme=''):
# 递归初始化 MSCB、MSDC 及其内部层。
        named_apply(partial(_init_weights, scheme=scheme), self)

# 输入为 (B,C_in,H,W)，默认解码器中输出为同尺寸、同通道特征。
    def forward(self, x):
# 通道扩张：C_in -> C_ex。
        pout1 = self.pconv1(x)
# 得到多个 (B,C_ex,H/stride,W/stride) 深度卷积结果。
        msdc_outs = self.msdc(pout1)
# 论文式(5)默认采用逐元素求和聚合多尺度响应。
        if self.add == True:
# 从标量 0 开始累计，第一次相加后成为张量。
            dout = 0
# 遍历所有尺度输出。
            for dwout in msdc_outs:
# 尺度间逐元素求和，通道数保持 C_ex。
                dout = dout + dwout
# 工程可选路径：不用求和而沿通道维拼接。
        else:
# concat 后通道为 C_ex*n_scales。
            dout = torch.cat(msdc_outs, dim=1)
# 用 gcd(C_combined,C_out) 选择合法分组数，再重排通道促进跨组交互。
        dout = channel_shuffle(dout, gcd(self.combined_channels,self.out_channels))
# 1x1 投影：C_combined -> C_out。
        out = self.pconv2(dout)
# stride=1 时启用倒残差捷径。
        if self.use_skip_connection:
# 若输入输出通道不同，先对捷径通道进行投影。
            if self.in_channels != self.out_channels:
# x 从 C_in 对齐到 C_out。
                x = self.conv1x1(x)
# 主分支和捷径逐元素相加，输出形状保持 (B,C_out,H,W)。
            return x + out
# stride=2 时不存在同尺寸捷径，直接返回主分支。
        else:
# 输出空间尺寸约为输入的一半。
            return out
        
#   Multi-scale convolution block (MSCB)
# 创建连续 n 个 MSCB；EMCAD 当前每一级都传 n=1，但该工厂保留堆叠能力。
def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        """
        create a series of multi-scale convolution blocks.
        """
# 用普通列表暂存将要注册到 Sequential 的块。
        convs = []
# 第一个块负责 C_in -> C_out，并可使用调用方指定的 stride。
        mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
# 加入第一个 MSCB。
        convs.append(mscb)
# n>1 时继续堆叠保持 C_out 和 stride=1 的块。
        if n > 1:
# 从第二个块循环到第 n 个块。
            for i in range(1, n):
# 后续块均保持通道和空间尺寸。
                mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
# 把后续块加入列表。
                convs.append(mscb)
# Sequential 将列表中的块按顺序执行并注册参数。
        conv = nn.Sequential(*convs)
# 返回可直接调用的 MSCB 序列。
        return conv

#   Efficient up-convolution block (EUCB)
# 论文第5页 Sec.3.1.3、式(9)，结构图见第4页 Fig.2(c)：高效二倍上采样并降低通道。
class EUCB(nn.Module):
# C_in 是当前深层特征通道，C_out 要与下一层 skip 的通道一致。
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
# 初始化基础模块。
        super(EUCB,self).__init__()

# 保存输入通道数。
        self.in_channels = in_channels
# 保存输出通道数。
        self.out_channels = out_channels
# 上采样和深度卷积组合；空间尺寸先乘 2，通道暂时保持 C_in。
        self.up_dwc = nn.Sequential(
# 未指定 mode 时 PyTorch 对 4D 张量采用最近邻插值；输出约为 (2H,2W)。
            nn.Upsample(scale_factor=2),
# 3x3 depth-wise convolution 在每个通道内细化上采样特征。
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=self.in_channels, bias=False),
# 归一化 C_in 个深度卷积输出通道。
	        nn.BatchNorm2d(self.in_channels),
# 论文式(9)在这里使用 ReLU。
            act_layer(activation, inplace=True)
        )
# point-wise convolution 负责从 C_in 投影到下一解码级的 C_out。
        self.pwc = nn.Sequential(
# 1x1 卷积只改变通道，不改变已经放大的空间尺寸。
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        ) 
# 初始化 EUCB 内全部层。
        self.init_weights('normal')
    
# 统一初始化入口。
    def init_weights(self, scheme=''):
# 递归初始化 EUCB。
        named_apply(partial(_init_weights, scheme=scheme), self)

# 输入 (B,C_in,H,W)，输出 (B,C_out,2H,2W)。
    def forward(self, x):
# 上采样、深度卷积、BN、激活，形状变为 (B,C_in,2H,2W)。
        x = self.up_dwc(x)
# 当前 groups=C_in 时每组只有一个通道，这次 shuffle 在现实现中等价于恒等重排。
        x = channel_shuffle(x, self.in_channels)
# 1x1 投影到 C_out，以便与下一层 skip 相加。
        x = self.pwc(x)
# 返回上采样并完成通道对齐的特征。
        return x

#   Large-kernel grouped attention gate (LGAG)
# 论文第3页 Sec.3.1.1、式(1)-(2)，结构见第4页 Fig.2(g)：用深层 gating 信号筛选 skip 特征。
class LGAG(nn.Module):
# F_g 为解码特征通道，F_l 为 skip 通道，F_int 是内部压缩通道，通常等于 C/2。
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
# 初始化基础模块。
        super(LGAG,self).__init__()

# 1x1 卷积无法按当前 C/2 分组约束稳定工作，因此强制退回单组卷积。
        if kernel_size == 1:
# groups=1 即普通卷积。
            groups = 1
# W_g 处理来自更深解码层、已经上采样的 gating 特征 g。
        self.W_g = nn.Sequential(
# 大核分组卷积把 F_g 映射到 F_int；默认 3x3 提供比传统 1x1 AG 更大的局部感受野。
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
# 对 gating 投影结果归一化。
            nn.BatchNorm2d(F_int)
        )
# W_x 处理同分辨率的编码器 skip 特征 x。
        self.W_x = nn.Sequential(
# 使用与 W_g 相同的核、分组数和内部通道，保证两路可逐元素相加。
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
# 对 skip 投影结果归一化。
            nn.BatchNorm2d(F_int)
        )
# psi 把融合后的 F_int 通道压缩成单通道空间注意力图。
        self.psi = nn.Sequential(
# 1x1 卷积执行 F_int -> 1。
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
# 对单通道注意力 logits 做批归一化。
            nn.BatchNorm2d(1),
# Sigmoid 把每个空间位置限制到 [0,1]。
            nn.Sigmoid()
        )
# 论文式(1)中两路相加后使用 ReLU。
        self.activation = act_layer(activation, inplace=True)

# 初始化所有分组卷积、1x1 卷积和 BN。
        self.init_weights('normal')
    
# 统一初始化入口。
    def init_weights(self, scheme=''):
# 递归初始化 LGAG。
        named_apply(partial(_init_weights, scheme=scheme), self)
                
# g 与 x 的空间尺寸必须一致；输出保持 x 的 (B,F_l,H,W)。
    def forward(self, g, x):
# gating 路：F_g -> F_int。
        g1 = self.W_g(g)
# skip 路：F_l -> F_int。
        x1 = self.W_x(x)
# 两路逐元素相加后激活，对应论文式(1)的 q_att。
        psi = self.activation(g1 + x1)
# 压缩为 (B,1,H,W) 并经过 Sigmoid，得到空间门控系数。
        psi = self.psi(psi)

# 单通道 psi 在通道维广播，逐位置抑制或保留原始 skip x，对应论文式(2)。
        return x*psi
    
#   Channel attention block (CAB)
# 论文第5页 Sec.3.1.2、式(7)，结构见第4页 Fig.2(h)：回答“哪些通道更重要”。
class CAB(nn.Module):
# ratio 控制通道瓶颈宽度；out_channels 为空时生成与输入同通道数的权重。
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
# 初始化基础模块。
        super(CAB, self).__init__()

# 输入特征通道 C。
        self.in_channels = in_channels
# 可选输出权重通道，EMCAD 中不传，因此最终等于 C。
        self.out_channels = out_channels
# 当 C<16 时把 ratio 降到 C，避免 C//ratio 变成 0。
        if self.in_channels < ratio:
# 此时瓶颈至少保留 1 个通道。
            ratio = self.in_channels
# 瓶颈通道 C_reduced=C/ratio。
        self.reduced_channels = self.in_channels // ratio
# 默认注意力权重与输入通道一一对应。
        if self.out_channels == None:
# C_out=C_in。
            self.out_channels = in_channels

# 全局平均池化得到 (B,C,1,1)，概括通道的整体响应。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
# 全局最大池化得到 (B,C,1,1)，保留通道最强响应。
        self.max_pool = nn.AdaptiveMaxPool2d(1)
# 两条池化路径共享同一激活类型。
        self.activation = act_layer(activation, inplace=True)
# 共享的第一层 1x1 卷积：C -> C_reduced。
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
# 共享的第二层 1x1 卷积：C_reduced -> C_out。
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        
# 将融合后的通道 logits 映射到 [0,1]。
        self.sigmoid = nn.Sigmoid()

# 初始化 CAB 的两个 1x1 卷积。
        self.init_weights('normal')
    
# 统一初始化入口。
    def init_weights(self, scheme=''):
# 递归初始化 CAB。
        named_apply(partial(_init_weights, scheme=scheme), self)

# 输入 x 为 (B,C,H,W)；本类只返回 (B,C,1,1) 权重，不在类内乘回 x。
    def forward(self, x):
# 平均池化分支压缩空间维。
        avg_pool_out = self.avg_pool(x) 
# 共享 MLP：C -> C_reduced -> C。
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

# 最大池化分支压缩空间维。
        max_pool_out= self.max_pool(x) 
# 与平均分支复用同一 fc1、fc2 参数。
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

# 两种全局统计逐元素相加。
        out = avg_out + max_out
# 返回通道权重；EMCAD.forward 随后执行 self.cab*(feature)，共同等价于论文式(7)。
        return self.sigmoid(out) 
    
#   Spatial attention block (SAB)
# 论文第5页 Sec.3.1.2、式(8)，结构见第4页 Fig.2(i)：回答“哪些空间位置更重要”。
class SAB(nn.Module):
# kernel_size 控制空间注意力感受野；论文和默认实现使用 7x7。
    def __init__(self, kernel_size=7):
# 初始化基础模块。
        super(SAB, self).__init__()

# 只允许三种奇数核，确保可以使用对称 padding 保持 H、W。
        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
# padding=k//2，使 stride=1 卷积前后空间尺寸不变。
        padding = kernel_size//2

# 输入是通道平均图和最大图拼成的 2 通道张量，输出单通道空间 logits。
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
           
# Sigmoid 生成 [0,1] 空间权重。
        self.sigmoid = nn.Sigmoid()

# 初始化 SAB 的大核卷积。
        self.init_weights('normal')
    
# 统一初始化入口。
    def init_weights(self, scheme=''):
# 递归初始化 SAB。
        named_apply(partial(_init_weights, scheme=scheme), self)

# 输入 (B,C,H,W)，返回 (B,1,H,W) 权重；乘回输入发生在 EMCAD.forward。
    def forward(self, x):
# 沿通道维求平均，得到 (B,1,H,W)。
        avg_out = torch.mean(x, dim=1, keepdim=True)
# 沿通道维取最大值；下划线接收不参与后续计算的最大值索引。
        max_out, _ = torch.max(x, dim=1, keepdim=True)
# 拼接成 (B,2,H,W)，同时提供平均响应与最强响应。
        x = torch.cat([avg_out, max_out], dim=1)
# 大核卷积融合相邻空间上下文并压缩到 1 通道。
        x = self.conv(x)
# 返回空间权重；调用方通过广播乘到全部 C 个通道。
        return self.sigmoid(x)

#   Efficient multi-scale convolutional attention decoding (EMCAD)
# 论文第3页 Sec.3.1、整体结构见第4页 Fig.2(b)：从最深特征开始逐级上采样、门控 skip 并细化。
class EMCAD(nn.Module):
# channels 按深到浅排列；PVTv2-B2 默认为 [512,320,128,64]。
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
# 初始化基础模块。
        super(EMCAD,self).__init__()
# EUCB 固定使用 3x3 depth-wise convolution。
        eucb_ks = 3 # kernel size for eucb
# 最深层 d4 只做 MSCAM 细化，不需要先上采样；MSCB 保持 channels[0]。
        self.mscb4 = MSCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
	
# 第三级上采样：channels[0] -> channels[1]，空间尺寸乘 2。
        self.eucb3 = EUCB(in_channels=channels[0], out_channels=channels[1], kernel_size=eucb_ks, stride=eucb_ks//2)
# 用上采样 d3 门控同尺度 x3；内部通道为 channels[1]/2，默认采用大核分组卷积。
        self.lgag3 = LGAG(F_g=channels[1], F_l=channels[1], F_int=channels[1]//2, kernel_size=lgag_ks, groups=channels[1]//2)
# 融合 skip 后执行第三级 MSCAM 中的 MSCB，通道保持 channels[1]。
        self.mscb3 = MSCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

# 第二级上采样：channels[1] -> channels[2]，空间尺寸再乘 2。
        self.eucb2 = EUCB(in_channels=channels[1], out_channels=channels[2], kernel_size=eucb_ks, stride=eucb_ks//2)
# 门控编码器第二级 skip x2。
        self.lgag2 = LGAG(F_g=channels[2], F_l=channels[2], F_int=channels[2]//2, kernel_size=lgag_ks, groups=channels[2]//2)
# 第二级 MSCB 细化，通道保持 channels[2]。
        self.mscb2 = MSCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        
# 第一级上采样：channels[2] -> channels[3]，到达编码器最高分辨率特征层。
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
# 门控最浅层 skip x1；int(...) 与 //2 在正整数通道下结果相同。
        self.lgag1 = LGAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3]/2), kernel_size=lgag_ks, groups=int(channels[3]/2))
# 第一级 MSCB 输出解码器最高分辨率特征 d1。
        self.mscb1 = MSCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        
# 四个解码级分别拥有独立 CAB；权重通道与本级特征一致。
        self.cab4 = CAB(channels[0])
# 第三级 CAB。
        self.cab3 = CAB(channels[1])
# 第二级 CAB。
        self.cab2 = CAB(channels[2])
# 第一级 CAB。
        self.cab1 = CAB(channels[3])
        
# SAB 不依赖通道数，因此四个解码级共享同一个 7x7 空间注意力模块及其参数。
        self.sab = SAB()
       
      
# x 是最深层 x4，skips 必须按 [x3,x2,x1] 从深到浅传入。
    def forward(self, x, skips):
            
        # MSCAM4
# 代码没有单独 MSCAM 类；CAB 权重乘回 x 是论文式(7)的完整通道注意力。
        d4 = self.cab4(x)*x
# SAB 权重乘回 d4 是论文式(8)的完整空间注意力，形状仍为 (B,C4,H/32,W/32)。
        d4 = self.sab(d4)*d4 
# MSCB 完成多尺度深度卷积细化；CAB -> SAB -> MSCB 合起来对应论文式(3)的 MSCAM。
        d4 = self.mscb4(d4)
        
        # EUCB3
# d4 二倍上采样并把 C4 投影为 C3；PVTv2-B2 为 512 -> 320。
        d3 = self.eucb3(d4)
                
        # LGAG3
# skips[0] 是编码器 x3；LGAG 输出被门控的 x3，通道和分辨率不变。
        x3 = self.lgag3(g=d3, x=skips[0])
        
        # Additive aggregation 3
# 解码特征与筛选后的 skip 逐元素相加，不采用通道拼接，因此 d3 仍为 C3。
        d3 = d3 + x3
        
        # MSCAM3
# 第三级通道注意力。
        d3 = self.cab3(d3)*d3
# 第三级空间注意力。
        d3 = self.sab(d3)*d3  
# 第三级多尺度卷积细化；输出形状保持不变。
        d3 = self.mscb3(d3)
        
        # EUCB2
# 再次二倍上采样并执行 C3 -> C2；PVTv2-B2 为 320 -> 128。
        d2 = self.eucb2(d3)
        
        # LGAG2
# 门控编码器 x2，输出保持 skips[1] 的 C2 通道。
        x2 = self.lgag2(g=d2, x=skips[1])
        
        # Additive aggregation 2
# 与门控 skip 相加，形状保持 (B,C2,H/8,W/8)。
        d2 = d2 + x2 
        
        # MSCAM2
# 第二级通道注意力。
        d2 = self.cab2(d2)*d2
# 第二级空间注意力。
        d2 = self.sab(d2)*d2
# 第二级多尺度卷积细化。
        d2 = self.mscb2(d2)
        
        # EUCB1
# 最后一次二倍上采样并执行 C2 -> C1；PVTv2-B2 为 128 -> 64。
        d1 = self.eucb1(d2)
        
        # LGAG1
# 门控最高分辨率编码器特征 x1。
        x1 = self.lgag1(g=d1, x=skips[2])
        
        # Additive aggregation 1
# 相加后 d1 保持 (B,C1,H/4,W/4)。
        d1 = d1 + x1 
        
        # MSCAM1
# 第一级通道注意力。
        d1 = self.cab1(d1)*d1
# 第一级空间注意力。
        d1 = self.sab(d1)*d1
# 最高分辨率解码特征的最终多尺度卷积细化。
        d1 = self.mscb1(d1)
        
# 返回顺序固定为从深到浅 [d4,d3,d2,d1]；网络封装器据此连接四个分割头。
        return [d4, d3, d2, d1]
