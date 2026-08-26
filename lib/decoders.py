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


# ==================================================================================================
# 本文件阅读总览（这一大段只帮助读代码，不参与模型计算）
# --------------------------------------------------------------------------------------------------
# 1. 本文件只实现“解码器”。它不直接读取医学图像，也不直接调用 PVTv2 或 ResNet。
#    真正的总装入口在 lib/networks.py 的 EMCADNet：
#        输入图像
#          -> lib/pvtv2.py 或 lib/resnet.py 产生 [x1, x2, x3, x4]
#          -> lib/networks.py 调用 self.decoder(x4, [x3, x2, x1])
#          -> 本文件 EMCAD.forward 返回 [d4, d3, d2, d1]
#          -> lib/networks.py 的四个 1x1 输出头产生 [p4, p3, p2, p1]
#          -> 再把四个 logits 上采样到原图大小，交给训练损失或测试后处理。
#
# 2. 全文统一使用 NCHW 张量布局：
#        B/N：batch size，一次并行处理的图像数量；
#        C：通道数；
#        H、W：当前特征图的高和宽。
#    因此形如 (B, C, H, W) 的说明，指的就是一个四维 PyTorch 特征张量。
#
# 3. channels 的顺序与编码器输出顺序相反，必须特别留意：
#        编码器输出顺序： [x1, x2, x3, x4]，从浅层高分辨率到深层低分辨率；
#        解码器通道顺序： [C4, C3, C2, C1]，从最深层到最浅层。
#    默认 PVTv2-B2 在输入尺寸能被 32 整除时，对应：
#        x1=(B,  64, H/4,  W/4 )
#        x2=(B, 128, H/8,  W/8 )
#        x3=(B, 320, H/16, W/16)
#        x4=(B, 512, H/32, W/32)
#        channels=[512, 320, 128, 64]
#    例如 Synapse 的 224x224 输入会得到 56x56、28x28、14x14、7x7 四级特征。
#
# 4. 本文件各模块的分工可先记成一句话：
#        CAB：在“通道”维判断哪些语义特征重要；
#        SAB：在“空间”维判断哪些像素位置重要；
#        MSDC：用多个大小的深度卷积核观察不同范围的局部区域；
#        MSCB：先扩通道，再做 MSDC，再压回目标通道，并尽量保留残差；
#        EUCB：把解码特征放大 2 倍，并改成下一层需要的通道数；
#        LGAG：用当前解码信息筛选编码器 skip，减少无关低层信息；
#        EMCAD：把上述模块按 d4 -> d3 -> d2 -> d1 串成完整解码路径。
#
# 5. 这里的“注意力”都不是 PVTv2 中的多头自注意力：
#    CAB、SAB、LGAG 都是卷积式门控模块，最终生成 0~1 的权重并乘到特征上。
#    它们的目标是重新加权已有特征，不是建立 Transformer token 两两之间的注意力矩阵。
#
# 6. 当前工程的常用配置由 lib/networks.py 显式传入：kernel_sizes=[1,3,5]、
#    expansion_factor=2、dw_parallel=True、add=True、lgag_ks=3。虽然下面 EMCAD 类自身的
#    expansion_factor 默认值写成 6，但通过 EMCADNet 正常创建模型时会被上层默认值 2 覆盖。
#
# 7. 三个容易混淆的“加法”不是同一件事：
#        add=True：只控制 MSCB 内多个 MSDC 分支是相加还是沿通道拼接；
#        d3 = d3 + x3：控制解码主路与 LGAG 筛选后的 skip 如何融合，始终是相加；
#        return x + out：是 MSCB 内部的残差连接。
#    因此命令行中的 concatenation 选项不会把编码器 skip 融合改成 concat。
#
# 8. 解码器要求相邻层空间尺寸严格相差 2 倍。项目常用 224、352 等可被 32 整除的输入，
#    因而 EUCB 放大后的尺寸能与对应 skip 对齐；若任意输入尺寸造成奇数层级，LGAG 中 g1+x1
#    或后面的 d+x 可能因 H、W 不一致而报错。本文件没有自动裁剪、补零或插值对齐逻辑。
# ==================================================================================================


# 计算两个正整数的最大公约数；MSCB 用它确定 channel shuffle 的分组数。
# 该辅助函数不会创建网络参数，也不会操作张量；它只在 MSCB.forward 中计算一个整数 groups。
# 例如默认某一级 C_in=C_out=64、expansion_factor=2、add=True 时：
# combined_channels=128，gcd(128,64)=64，于是 channel_shuffle 把 128 个通道视为 64 组、每组 2 个。
# 使用最大公约数的原因是它必然同时整除待重排通道数和输出通道数，能避免非法分组数量。
# 当前调用路径保证 a、b 都是正通道数；若两者都为 0，本函数会返回 0，而卷积分组不能为 0。
def gcd(a, b):
    # 欧几里得算法：只要余数除数 b 还不为 0，就继续迭代。
    while b:
        # 新的 a 取旧 b，新的 b 取 a 对 b 的余数。
        a, b = b, a % b
    # b 为 0 时，a 即最大公约数。
    return a


# Other types of layers can go here (e.g., nn.Linear, etc.)
# named_apply 调用的统一初始化入口；name 是遍历得到的模块路径，当前实现不使用它。
# 调用范围说明：这个函数只被本文件各解码子模块的 init_weights 调用，不会顺带初始化
# lib/pvtv2.py 或 lib/resnet.py 的编码器。虽然这里包含 LayerNorm/Conv3d 分支，但当前 EMCAD
# 实例实际只包含 Conv2d 与 BatchNorm2d；其他分支是为复用或扩展留下的通用能力。
#
# 初始化发生在模块构造阶段，早于训练器创建优化器。初始化只设定训练起点，不会在每次 forward
# 时重复随机化参数。MSDC 在自身构造时初始化一次，随后其外层 MSCB 又递归初始化整块，因此
# 嵌套在 MSCB 中的 MSDC 参数最终以 MSCB 那次初始化结果为准；这不改变网络结构，只改变随机起点。
#
# scheme 的含义：
#   normal         -> N(0, 0.02^2)，是本文件所有实际构造路径显式使用的方案；
#   trunc_normal   -> 截断正态，减少极端初始值；
#   xavier_normal  -> 同时考虑输入和输出 fan；
#   kaiming_normal -> 面向 ReLU 的 fan_out 初始化；
#   其他字符串/空串 -> 进入 EfficientNet 风格分支。
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
        # 补充：当前 decoder 中没有 LayerNorm，所以正常 EMCAD 构造不会实际进入这个分支。
        # LayerNorm 的缩放参数设为 1。
        nn.init.constant_(module.weight, 1)
        # LayerNorm 的偏置参数设为 0。
        nn.init.constant_(module.bias, 0)


# 根据字符串创建激活层，统一供解码器各子模块复用。
# 参数说明：
#   act       是激活名称；调用方用字符串配置，便于命令行做 ReLU/ReLU6 消融实验；
#   inplace   为 True 时允许覆盖输入张量的存储，以减少中间内存，但调用方不能再依赖激活前值；
#   neg_slope 只给 LeakyReLU/PReLU 使用；
#   n_prelu   决定 PReLU 有几个可学习负半轴斜率。
# 返回值是一个 nn.Module，而不是立刻计算的张量；随后会被放进 nn.Sequential 或保存为子模块。
# 当前实际调用关系：MSCB/MSDC 使用上层传入的 activation，EUCB、LGAG、CAB 默认使用 ReLU。
# EMCAD 的 activation 参数只传给 MSCB/MSDC，并不会统一改动 EUCB、LGAG、CAB 的激活函数。
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
# 输入输出形状完全相同，数值也不做加减乘除，只改变通道排列顺序。
# 一个最小例子：若通道顺序是 [a1,a2,b1,b2]，groups=2，则先看成两组
# [[a1,a2],[b1,b2]]，交换“组号”和“组内位置”后会交错成 [a1,b1,a2,b2]。
# 这样，后续 1x1 卷积能同时接触原来不同组中的特征，缓解分组/深度卷积缺少跨通道交流的问题。
# groups 必须为正数且能整除 C；MSCB 通过 gcd 生成合法值，EUCB 则直接传 groups=C。
# EUCB 中 groups=C 意味着每组恰好 1 个通道，当前重排结果实际上与输入顺序相同；代码仍保留
# 此调用，是为了保持 EUCB 的统一结构表达，而不是因为它在该配置下真的混合了通道。
def channel_shuffle(x, groups):
    # 补充：这里只通过 .data 读取 shape，没有拿 .data 参与数值计算，所以本行不会切断后续梯度；
    # 但现代 PyTorch 更常直接写 x.size()，学习其他代码时不要把 .data 当作常规张量运算接口。
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
# --------------------------------------------------------------------------------------------------
# MSDC 可以理解为“同一份特征，分别用不同大小的尺子观察”。普通 3x3 卷积只能看较小邻域，
# 5x5 能看到更宽的上下文，而 1x1 分支保留最局部的逐点变换。这里每条支路都是 depth-wise
# convolution：每个通道独立做空间卷积，不在该步混合不同通道，因此参数量远低于普通卷积。
#
# 输入/输出契约：
#   输入：x，形状 (B, C_ex, H, W)；
#   输出：Python list，长度等于 len(kernel_sizes)；
#   每个元素：形状 (B, C_ex, H_out, W_out)，其中 stride=1 且卷积核为奇数时 H_out=H、W_out=W。
# MSDC 本身不把多尺度结果合成一个张量；求和或 concat 的决定留给外层 MSCB。
#
# dw_parallel=True：所有尺度都读取相同的原始输入 x，分支可以概念上并行；
# dw_parallel=False：后一尺度读取“原输入/上一状态 + 上一分支输出”，形成逐级累积的串行路径。
# 当前 EMCAD 中所有 MSCB 都使用 stride=1，所以串行模式可以做 x+dw_out；如果单独把 MSCB
# 配成 stride=2 且 dw_parallel=False，原 x 与下采样后的 dw_out 空间尺寸不同，会无法相加。
# --------------------------------------------------------------------------------------------------
class MSDC(nn.Module):
    # in_channels 是每个深度卷积分支的通道数；kernel_sizes 通常为 [1,3,5]。
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        # 本构造函数只“搭建”各尺度卷积支路并注册参数，不处理真实 batch；真实张量在 forward 才进入。
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
            # 之所以使用 ModuleList 而不是普通 list，是因为 PyTorch 只有在 Module/ModuleList/
            # Sequential 等容器中才能发现子层参数，并让它们进入 state_dict、optimizer 和 .cuda()。
            # 每个尺度均按“深度卷积 -> BN -> 激活”组成 DWCB。
            nn.Sequential(
                # groups=C 使每个输入通道独立卷积；padding=k//2 在 stride=1 时保持 H、W。
                # 参数量对比：普通 k*k 卷积约有 C_ex*C_ex*k*k 个权重；此处深度卷积只有
                # C_ex*k*k 个权重。跨通道融合随后由 MSCB 的 1x1 point-wise convolution 完成。
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
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
        # forward 没有可学习参数定义；它只是按构造阶段注册好的 self.dwconvs 组织张量流向。
        # Apply the convolution layers in a loop
        # 保存每个尺度 DWCB 的独立输出，而不是在此处直接合并。
        outputs = []
        # 依次执行 1x1、3x3、5x5 等深度卷积分支。
        for dwconv in self.dwconvs:
            # 循环变量 dwconv 依次指向一个完整的“DWConv+BN+activation” Sequential，而非单个卷积核。
            # 并行模式下每次读取原始 x；串行模式下读取上一轮残差更新后的 x。
            dw_out = dwconv(x)
            # 每个 dw_out 形状均为 (B,C_ex,H/stride,W/stride)。
            outputs.append(dw_out)
            # 工程扩展：关闭并行时采用论文式(6)的递归残差输入。
            if self.dw_parallel == False:
                # 只有 False 才进入；写成 not self.dw_parallel 语义相同。这里没有修改 outputs 中已保存的
                # dw_out，只更新下一次循环将读取的局部变量 x。
                # 更新下一分支的输入；默认解码器 stride=1，因此空间尺寸可直接相加。
                x = x + dw_out
        # You can return outputs based on what you intend to do with them
        # MSCB 再根据 add 参数对这些分支求和或拼接。
        return outputs


# 论文第4页 Sec.3.1.2、Fig.2(e)、式(4)：MobileNetV2 倒残差思想上的多尺度卷积块。
# --------------------------------------------------------------------------------------------------
# MSCB 是本解码器的主要“特征加工单元”，完整数据流如下：
#
#   输入 x:(B,C_in,H,W)
#       |
#       +--------------------------- 残差/捷径 ----------------------------+
#       |                                                                  |
#       v                                                                  |
#   pconv1: 1x1 Conv + BN + activation                                    |
#       C_in -> C_ex=int(C_in*expansion_factor)                            |
#       |                                                                  |
#       v                                                                  |
#   MSDC: k1/k2/k3... 多尺度 depth-wise 分支                              |
#       |                                                                  |
#       v                                                                  |
#   add=True 时逐元素求和；add=False 时沿通道维 concat                     |
#       |                                                                  |
#       v                                                                  |
#   channel_shuffle -> pconv2(1x1 Conv + BN): C_combined -> C_out          |
#       |                                                                  |
#       +------------------------- 与对齐后的捷径相加 <---------------------+
#
# “先扩张、后做便宜的深度卷积、最后投影”的目的，是在控制参数量的同时给空间卷积更多中间通道。
# 在 EMCAD.forward 中，四个 MSCB 都是 stride=1 且 C_in=C_out，所以输出形状与输入完全一致，
# 残差分支也无需 self.conv1x1；这个类保留 C_in!=C_out/stride=2 是为了可复用性。
# --------------------------------------------------------------------------------------------------
class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB) 
    """

    # 关键参数：C_in、C_out、步幅、多尺度核、扩张倍率、并/串行和分支聚合方式。
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, activation='relu6'):
        # 参数彼此的影响：kernel_sizes 决定 MSDC 分支数和感受野；expansion_factor 决定中间宽度；
        # dw_parallel 决定分支依赖关系；add 决定分支合并方式；activation 决定 pconv1 与 MSDC 激活。
        # kernel_sizes 的默认 list 在本实现中只读取、不修改，因此不会触发常见的“可变默认参数被污染”问题。
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
        # assert 在构造时尽早拦截其他 stride，避免到 forward 才以难懂的 shape 错误失败。
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        # 只有 stride=1 时主分支与捷径分支空间尺寸一致，才允许残差相加。
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        # 扩张后的中间通道 C_ex=int(C_in*e)。默认 e=2 时为 2C_in。
        # int 会向下截断非整数结果；项目命令行把 expansion_factor 定义为 int，常用配置不会发生截断。
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
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel)
        # 注意：add 只控制“这些 MSDC 尺度分支”的融合，不控制 EMCAD 主路与 encoder skip 的融合。
        # 加法聚合不改变通道数，合并后仍是 C_ex。
        if self.add == True:
            # 三个同形状张量相加仍只有 C_ex 个通道，区别体现在每个位置的数值汇聚了多个感受野。
            # 乘 1 明确记录加法后的通道数。
            self.combined_channels = self.ex_channels * 1
        # concat 工程选项会沿通道维连接所有尺度。
        else:
            # 例如 C_ex=128、三个尺度时，concat 产生 384 通道；表达容量和 pconv2 参数量都会增加。
            # 拼接后通道数为 C_ex*n_scales；这不是论文式(5)的默认求和路径。
            self.combined_channels = self.ex_channels * self.n_scales
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
            # 残差相加要求两边 shape 完全相同。stride=1 已保证 H、W 相同，本层只负责把 C_in 对齐到 C_out。
            # 当前 EMCAD 四级都是 C_in=C_out，因此正常默认网络不会创建这个可选 conv1x1。
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
        # 为便于跟踪，默认 PVTv2-B2 的 d1 级举例：输入 x=(B,64,56,56)，e=2；
        # pout1=(B,128,56,56)，三个 MSDC 输出都为该形状，add 后仍为 128 通道，
        # pconv2 再得到 out=(B,64,56,56)，最后与原 x 相加。
        # 通道扩张：C_in -> C_ex。
        pout1 = self.pconv1(x)
        # 得到多个 (B,C_ex,H/stride,W/stride) 深度卷积结果。
        msdc_outs = self.msdc(pout1)
        # 论文式(5)默认采用逐元素求和聚合多尺度响应。
        if self.add == True:
            # 这里不是 Python 的 sum(msdc_outs)，而是显式循环；梯度仍会沿每次张量加法传回所有分支。
            # 从标量 0 开始累计，第一次相加后成为张量。
            dout = 0
            # 遍历所有尺度输出。
            for dwout in msdc_outs:
                # kernel_sizes 若为空，循环不会执行，dout 会停留为整数 0；项目配置始终至少给一个卷积核。
                # 尺度间逐元素求和，通道数保持 C_ex。
                dout = dout + dwout
        # 工程可选路径：不用求和而沿通道维拼接。
        else:
            # dim=1 明确表示在通道维拼接；不能用 dim=0，否则会错误地把不同尺度当作更多 batch 样本。
            # concat 后通道为 C_ex*n_scales。
            dout = torch.cat(msdc_outs, dim=1)
        # 用 gcd(C_combined,C_out) 选择合法分组数，再重排通道促进跨组交互。
        # shuffle 不改变数值集合和 shape；真正学习如何融合通道的是紧随其后的 pconv2。
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))
        # 1x1 投影：C_combined -> C_out。
        out = self.pconv2(dout)
        # stride=1 时启用倒残差捷径。
        if self.use_skip_connection:
            # 残差连接给梯度提供更短路径，也让本块更容易在需要时学习“只做小修正”而不是重建全部特征。
            # 若输入输出通道不同，先对捷径通道进行投影。
            if self.in_channels != self.out_channels:
                # x 从 C_in 对齐到 C_out。
                x = self.conv1x1(x)
            # 主分支和捷径逐元素相加，输出形状保持 (B,C_out,H,W)。
            # 这里没有再接激活函数：返回的是投影分支 BN 输出与捷径的线性相加，后续模块继续处理。
            return x + out
        # stride=2 时不存在同尺寸捷径，直接返回主分支。
        else:
            # stride=2 时 out 的空间尺寸已经在 MSDC 中下采样；原 x 没有同步下采样，所以不能直接残差相加。
            # 输出空间尺寸约为输入的一半。
            return out


#   Multi-scale convolution block (MSCB)
# 创建连续 n 个 MSCB；EMCAD 当前每一级都传 n=1，但该工厂保留堆叠能力。
# 这是一个“模块工厂函数”，返回 nn.Sequential，而不是定义新的类。调用方无需知道内部有几个 MSCB，
# 只要像普通层一样执行 returned_module(x) 即可。第一个块承担输入到目标通道/步幅的变换，
# 后续块固定为 C_out -> C_out、stride=1，从而在不继续改变尺寸的前提下加深特征加工。
#
# 形状规则：
#   第一个块： (B,C_in,H,W) -> (B,C_out,H/stride,W/stride)，奇数核下 stride=2 约为向上取整一半；
#   后续 n-1 块：始终保持 (B,C_out,H_out,W_out)。
# 当前 EMCAD 的 mscb4/3/2/1 全部传 n=1、stride=1，因此每级只创建一个 MSCB。
def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
              add=True, activation='relu6'):
    """
    create a series of multi-scale convolution blocks.
    """
    # 用普通列表暂存将要注册到 Sequential 的块。
    # 此时普通 list 只是构造期临时容器；最终会用 nn.Sequential(*convs) 注册，参数不会丢失。
    convs = []
    # 第一个块负责 C_in -> C_out，并可使用调用方指定的 stride。
    mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add, activation=activation)
    # 加入第一个 MSCB。
    convs.append(mscb)
    # n>1 时继续堆叠保持 C_out 和 stride=1 的块。
    if n > 1:
        # n=1 时跳过此分支，避免多建不需要的层；n<=0 也仍会保留上面已经创建的第一个块，
        # 因而这个函数的实际最少块数是 1，调用方不应把 n=0 理解为“空层”。
        # 从第二个块循环到第 n 个块。
        for i in range(1, n):
            # 后续块均保持通道和空间尺寸。
            mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                        dw_parallel=dw_parallel, add=add, activation=activation)
            # 把后续块加入列表。
            convs.append(mscb)
    # Sequential 将列表中的块按顺序执行并注册参数。
    # 星号把 Python 列表展开为 Sequential 的位置参数：nn.Sequential(block0, block1, ...)。
    conv = nn.Sequential(*convs)
    # 返回可直接调用的 MSCB 序列。
    return conv


#   Efficient up-convolution block (EUCB)
# 论文第5页 Sec.3.1.3、式(9)，结构图见第4页 Fig.2(c)：高效二倍上采样并降低通道。
# --------------------------------------------------------------------------------------------------
# EUCB 负责在相邻解码阶段之间“升高分辨率、降低通道数”。它没有使用参数量较大的转置卷积，
# 而是采用：最近邻插值 -> depth-wise 空间细化 -> 1x1 point-wise 通道投影。
#
# 输入/输出：
#   输入  (B,C_in,H,W)
#   Upsample 后 (B,C_in,2H,2W)
#   depth-wise Conv/BN/ReLU 后形状不变
#   1x1 Conv 后输出 (B,C_out,2H,2W)
# 这样输出就能与下一层同分辨率、同通道的 encoder skip 一起送入 LGAG 和加法融合。
#
# 默认 PVTv2-B2 的三个实例：
#   eucb3: 512@7x7   -> 320@14x14
#   eucb2: 320@14x14 -> 128@28x28
#   eucb1: 128@28x28 ->  64@56x56
# --------------------------------------------------------------------------------------------------
class EUCB(nn.Module):
    # C_in 是当前深层特征通道，C_out 要与下一层 skip 的通道一致。
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        # 构造期只记录通道并建立两个子序列；scale_factor=2 是写死的，所以每个 EUCB 恢复一级分辨率。
        # 初始化基础模块。
        super(EUCB, self).__init__()

        # 保存输入通道数。
        self.in_channels = in_channels
        # 保存输出通道数。
        self.out_channels = out_channels
        # 上采样和深度卷积组合；空间尺寸先乘 2，通道暂时保持 C_in。
        self.up_dwc = nn.Sequential(
            # 插值本身没有可学习参数，负责几何放大；后面的深度卷积负责学习如何修整复制后的局部响应。
            # 未指定 mode 时 PyTorch 对 4D 张量采用最近邻插值；输出约为 (2H,2W)。
            nn.Upsample(scale_factor=2),
            # 3x3 depth-wise convolution 在每个通道内细化上采样特征。
            # EMCAD 构造中 eucb_ks 固定为 3，传入 stride=eucb_ks//2=1，因此该卷积不会再次改变尺寸。
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            # 归一化 C_in 个深度卷积输出通道。
            nn.BatchNorm2d(self.in_channels),
            # 论文式(9)在这里使用 ReLU。
            act_layer(activation, inplace=True)
        )
        # point-wise convolution 负责从 C_in 投影到下一解码级的 C_out。
        self.pwc = nn.Sequential(
            # 这里只有一个卷积，理论上可以不用 Sequential；保留容器形式便于未来追加 BN/激活而不改 forward。
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
        # 该 forward 不接收 skip；它只处理解码主路。skip 的筛选和融合由后面的 LGAG/EMCAD 完成。
        # 上采样、深度卷积、BN、激活，形状变为 (B,C_in,2H,2W)。
        x = self.up_dwc(x)
        # 当前 groups=C_in 时每组只有一个通道，这次 shuffle 在现实现中等价于恒等重排。
        # 因而不要把本行误解为当前 EUCB 的主要跨通道融合来源；跨通道学习实际发生在下一行 self.pwc。
        x = channel_shuffle(x, self.in_channels)
        # 1x1 投影到 C_out，以便与下一层 skip 相加。
        x = self.pwc(x)
        # 返回上采样并完成通道对齐的特征。
        return x


#   Large-kernel grouped attention gate (LGAG)
# 论文第3页 Sec.3.1.1、式(1)-(2)，结构见第4页 Fig.2(g)：用深层 gating 信号筛选 skip 特征。
# --------------------------------------------------------------------------------------------------
# LGAG 解决的问题：编码器浅层 skip 有清晰边缘和纹理，但也携带大量背景噪声；如果不加选择地融合，
# 可能把与目标无关的细节带回解码器。当前解码特征 g 具有更强语义，可作为“我要找什么”的提示，
# 与 skip x 一起生成一张单通道空间门控图 psi，再用 psi 逐像素筛选原始 skip。
#
# 数据流（标准 EMCAD 中 F_g=F_l=C，F_int=C/2）：
#   g:(B,F_g,H,W) --W_g--> (B,F_int,H,W) --+
#                                                  +--> ReLU --> psi --> (B,1,H,W)
#   x:(B,F_l,H,W) --W_x--> (B,F_int,H,W) --+                    |
#                                                                  v 广播
#   原始 x:(B,F_l,H,W) ---------------------------------------> x*psi
#
# 注意输出是“被门控的 skip x”，不是 g，也不是 g+x；EMCAD.forward 会在门外另做 d3=d3+x3。
# psi 的一个空间位置对该位置的全部 F_l 通道使用同一个权重，因此 LGAG 是空间门控；通道选择由 CAB 做。
#
# 默认分组的实际含义：以 C=320、F_int=160、groups=160 为例，每组读取 2 个输入通道并产生 1 个
# 中间通道，大核卷积参数量远小于 groups=1 的普通 3x3 卷积。groups 必须同时整除输入和输出通道。
# --------------------------------------------------------------------------------------------------
class LGAG(nn.Module):
    # F_g 为解码特征通道，F_l 为 skip 通道，F_int 是内部压缩通道，通常等于 C/2。
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        # F_g 与 F_l 不要求名字上相等，但两条投影结果都必须是 F_int 且 H、W 相同，才能执行 g1+x1。
        # 当前 EMCAD 总是令 F_g=F_l=对应层 channels[i]，并在 EUCB 后再调用，所以条件得到满足。
        # 初始化基础模块。
        super(LGAG, self).__init__()

        # 1x1 卷积无法按当前 C/2 分组约束稳定工作，因此强制退回单组卷积。
        # 更准确地说，这是源码规定的设计分支，而不是 PyTorch 在数学上禁止 1x1 分组卷积；
        # 当 lgag_ks=1 时作者选择 groups=1，使其退化为普通 1x1 注意力门，便于与大核分组版本比较。
        if kernel_size == 1:
            # groups=1 即普通卷积。
            groups = 1
        # W_g 处理来自更深解码层、已经上采样的 gating 特征 g。
        self.W_g = nn.Sequential(
            # W_g 与 W_x 结构相同但参数不共享：模型可以分别学习如何解释解码语义和编码器细节。
            # 大核分组卷积把 F_g 映射到 F_int；默认 3x3 提供比传统 1x1 AG 更大的局部感受野。
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            # 对 gating 投影结果归一化。
            nn.BatchNorm2d(F_int)
        )
        # W_x 处理同分辨率的编码器 skip 特征 x。
        self.W_x = nn.Sequential(
            # 使用与 W_g 相同的核、分组数和内部通道，保证两路可逐元素相加。
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            # 对 skip 投影结果归一化。
            nn.BatchNorm2d(F_int)
        )
        # psi 把融合后的 F_int 通道压缩成单通道空间注意力图。
        self.psi = nn.Sequential(
            # psi Sequential 完成“压缩通道 -> 归一化 -> 概率化”；最终只有一张空间权重图。
            # 1x1 卷积执行 F_int -> 1。
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
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
        # 示例：PVTv2-B2 的 LGAG3 中，g 和 x 都是 (B,320,14,14)，两路投影为
        # (B,160,14,14)，psi 为 (B,1,14,14)，返回仍为 (B,320,14,14)。
        # gating 路：F_g -> F_int。
        g1 = self.W_g(g)
        # skip 路：F_l -> F_int。
        x1 = self.W_x(x)
        # 两路逐元素相加后激活，对应论文式(1)的 q_att。
        # 相加要求 batch、F_int、H、W 全部相同；本类没有在不匹配时自动 resize。
        psi = self.activation(g1 + x1)
        # 压缩为 (B,1,H,W) 并经过 Sigmoid，得到空间门控系数。
        psi = self.psi(psi)

        # 单通道 psi 在通道维广播，逐位置抑制或保留原始 skip x，对应论文式(2)。
        # 因为 Sigmoid 输出在 0 与 1 之间，此门本身只能按位置衰减/保留 x，不能直接把幅度放大到 x 以上。
        return x * psi


#   Channel attention block (CAB)
# 论文第5页 Sec.3.1.2、式(7)，结构见第4页 Fig.2(h)：回答“哪些通道更重要”。
# --------------------------------------------------------------------------------------------------
# CAB 把每个通道整张 HxW 特征压缩成一个描述值，再学习 C 个通道权重。这里同时使用：
#   AdaptiveAvgPool：概括通道整体平均响应；
#   AdaptiveMaxPool：保留通道最显著的局部响应。
# 两路共用 fc1/fc2，既减少参数，也要求模型用同一套通道关系解释两种统计；两路 logits 相加后
# 经 Sigmoid 得到 (B,C,1,1)。CAB 类只返回权重，真正的 feature*weight 在 EMCAD.forward 中执行。
#
# 形状示例（C=64、ratio=16）：
#   x=(B,64,H,W)
#   avg/max pool -> (B,64,1,1)
#   fc1 -> (B,4,1,1)
#   activation -> fc2 -> (B,64,1,1)
#   sigmoid -> 每个样本 64 个 0~1 权重，随后广播到所有 HxW 位置。
#
# 它回答“哪一种特征通道重要”，但同一通道内所有空间位置共享一个权重；空间差异留给 SAB。
# --------------------------------------------------------------------------------------------------
class CAB(nn.Module):
    # ratio 控制通道瓶颈宽度；out_channels 为空时生成与输入同通道数的权重。
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        # in_channels 必须是正整数；ratio 也应为正。项目使用 32 以上通道，通常按固定 16 倍压缩。
        # out_channels 是复用扩展接口；EMCAD 不传它，因此权重通道始终与输入相同并可直接广播相乘。
        # 初始化基础模块。
        super(CAB, self).__init__()

        # 输入特征通道 C。
        self.in_channels = in_channels
        # 可选输出权重通道，EMCAD 中不传，因此最终等于 C。
        self.out_channels = out_channels
        # 当 C<16 时把 ratio 降到 C，避免 C//ratio 变成 0。
        if self.in_channels < ratio:
            # 例如 C=8 时若仍除以 16 会得到 0 个中间通道，Conv2d 无法构造；改为 ratio=8 后得到 1。
            # 此时瓶颈至少保留 1 个通道。
            ratio = self.in_channels
        # 瓶颈通道 C_reduced=C/ratio。
        # 压缩再恢复相当于一个小型共享 MLP，学习通道间依赖，同时避免直接 C->C 的高参数开销。
        self.reduced_channels = self.in_channels // ratio
        # 默认注意力权重与输入通道一一对应。
        if self.out_channels == None:
            # C_out=C_in。
            self.out_channels = in_channels

        # 全局平均池化得到 (B,C,1,1)，概括通道的整体响应。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # “Adaptive” 表示无论输入 H、W 是 7、14 还是 56，输出都自动变为指定的 1x1；
        # 因此同一个 CAB 类逻辑可以用于不同分辨率阶段。
        # 全局最大池化得到 (B,C,1,1)，保留通道最强响应。
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 两条池化路径共享同一激活类型。
        self.activation = act_layer(activation, inplace=True)
        # 共享的第一层 1x1 卷积：C -> C_reduced。
        # 在 1x1 空间张量上，1x1 Conv2d 的作用等价于对通道做全连接，但能保持 NCHW 接口。
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
        # 两条池化支路的 fc1/fc2 是同一对象，forward 调用两次并不会复制参数；梯度会从两路累积到共享权重。
        # 平均池化分支压缩空间维。
        avg_pool_out = self.avg_pool(x)
        # 共享 MLP：C -> C_reduced -> C。
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        # 最大池化分支压缩空间维。
        max_pool_out = self.max_pool(x)
        # 与平均分支复用同一 fc1、fc2 参数。
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        # 两种全局统计逐元素相加。
        # 先相加 logits 再做一次 Sigmoid，不等于分别 Sigmoid 后相加；前者始终把最终权重限制在 [0,1]。
        out = avg_out + max_out
        # 返回通道权重；EMCAD.forward 随后执行 self.cab*(feature)，共同等价于论文式(7)。
        # 具体调用写法是 self.cab4(x) * x、self.cab3(d3) * d3 等；不存在名为 self.cab 的统一成员。
        return self.sigmoid(out)

    #   Spatial attention block (SAB)


# 论文第5页 Sec.3.1.2、式(8)，结构见第4页 Fig.2(i)：回答“哪些空间位置更重要”。
# --------------------------------------------------------------------------------------------------
# SAB 与 CAB 的压缩方向相反：CAB 压掉 H、W 保留 C；SAB 压掉 C 保留 H、W。
# 对每个像素位置，SAB 计算所有通道的平均值和最大值，拼成两张空间图，再用 7x7 卷积结合邻域，
# 输出 (B,1,H,W) 权重。EMCAD 先执行 CAB，再对已做通道加权的特征执行 SAB，形成串行注意力。
#
# 本类与通道数无关，因为卷积永远接收“mean 图 + max 图”共 2 通道。因此 EMCAD 只创建一个
# self.sab，并在 d4、d3、d2、d1 四个阶段反复使用；这意味着四个尺度共享同一套 SAB 卷积参数。
# CAB 则每级独立创建，因为其 fc1/fc2 的输入输出通道随层级变化。
# --------------------------------------------------------------------------------------------------
class SAB(nn.Module):
    # kernel_size 控制空间注意力感受野；论文和默认实现使用 7x7。
    def __init__(self, kernel_size=7):
        # kernel_size 越大，生成某位置权重时能参考的邻域越广，同时参数和计算量也增加。
        # 奇数核配合 padding=k//2 可严格保持 H、W；本实现显式限制为 3、7、11。
        # 初始化基础模块。
        super(SAB, self).__init__()

        # 只允许三种奇数核，确保可以使用对称 padding 保持 H、W。
        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        # padding=k//2，使 stride=1 卷积前后空间尺寸不变。
        padding = kernel_size // 2

        # 输入是通道平均图和最大图拼成的 2 通道张量，输出单通道空间 logits。
        # bias=False 是因为后面虽没有 BN，但源码选择让卷积只学习局部加权；Sigmoid 仍能产生空间门控。
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
        # 示例：输入 d2=(B,128,28,28)，平均图和最大图各为 (B,1,28,28)，拼接后为
        # (B,2,28,28)，卷积/Sigmoid 后为 (B,1,28,28)，广播乘回仍得到 (B,128,28,28)。
        # 沿通道维求平均，得到 (B,1,H,W)。
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 沿通道维取最大值；下划线接收不参与后续计算的最大值索引。
        # torch.max 在指定 dim 时返回 (values, indices)；只需要最大值本身来做注意力，索引无需保存。
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接成 (B,2,H,W)，同时提供平均响应与最强响应。
        x = torch.cat([avg_out, max_out], dim=1)
        # 大核卷积融合相邻空间上下文并压缩到 1 通道。
        # 卷积输出还只是任意实数 logits，下一行 Sigmoid 才把它解释为可广播的软门控系数。
        x = self.conv(x)
        # 返回空间权重；调用方通过广播乘到全部 C 个通道。
        return self.sigmoid(x)


#   Efficient multi-scale convolutional attention decoding (EMCAD)
# 论文第3页 Sec.3.1、整体结构见第4页 Fig.2(b)：从最深特征开始逐级上采样、门控 skip 并细化。
# ==================================================================================================
# EMCAD 是本文件的最终对外模块，lib/networks.py 只直接 import 这个类，其余类都是它的组成零件。
# 本类不产生最终类别 logits：它输出四级“解码特征”，类别通道转换和恢复原图大小由上层 EMCADNet
# 的 out_head4/3/2/1 与 F.interpolate 完成。因此看到 d1 仍为 H/4 是正常的，不代表最终分割少了 4 倍。
#
# 一次完整 forward 的主路径：
#
#   x4 -> [CAB -> SAB -> MSCB] -----------------------------------------------> d4
#                                                                                |
#                                                                                v EUCB3
#   x3(skip) -> LGAG(g=d3_up, x=x3) -> gated_x3 -> (+ d3_up) -> [CAB/SAB/MSCB] -> d3
#                                                                                |
#                                                                                v EUCB2
#   x2(skip) -> LGAG(g=d2_up, x=x2) -> gated_x2 -> (+ d2_up) -> [CAB/SAB/MSCB] -> d2
#                                                                                |
#                                                                                v EUCB1
#   x1(skip) -> LGAG(g=d1_up, x=x1) -> gated_x1 -> (+ d1_up) -> [CAB/SAB/MSCB] -> d1
#
# 这里把“CAB -> SAB -> MSCB”这一串称为 MSCAM，但代码中没有名为 MSCAM 的 Python 类；
# 它是概念模块名。阅读或做消融时，不要搜索不到 class MSCAM 就误以为实现缺失。
#
# 默认 PVTv2-B2、224x224 输入的精确形状：
#   传入 x=x4 : (B,512, 7, 7)      skips[0]=x3: (B,320,14,14)
#   d4         : (B,512, 7, 7)      skips[1]=x2: (B,128,28,28)
#   d3         : (B,320,14,14)      skips[2]=x1: (B, 64,56,56)
#   d2         : (B,128,28,28)
#   d1         : (B, 64,56,56)
# 返回 [d4,d3,d2,d1] 后，lib/networks.py 分别做 C->num_classes 的 1x1 卷积，并放大 32/16/8/4 倍。
#
# 换编码器时 forward 逻辑完全不变，只由 lib/networks.py 换 channels：
#   PVTv2-B0       [256,160, 64, 32]
#   PVTv2-B1~B5    [512,320,128, 64]
#   ResNet18/34    [512,256,128, 64]
#   ResNet50/101/152 [2048,1024,512,256]
# 这就是 decoders.py 能同时服务 pvtv2.py 和 resnet.py 的接口契约。
#
# 参数影响范围再次确认：
#   kernel_sizes / expansion_factor / dw_parallel / add / activation -> 四级 MSCB；
#   lgag_ks -> 三个 LGAG；
#   EUCB 的核固定为 3、激活固定采用其默认 ReLU；CAB/LGAG 的激活也使用各类默认 ReLU；
#   num_classes、encoder、pretrain 都不属于 EMCAD，由 lib/networks.py 管理。
# ==================================================================================================
class EMCAD(nn.Module):
    # channels 按深到浅排列；PVTv2-B2 默认为 [512,320,128,64]。
    def __init__(self, channels=[512, 320, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True,
                 add=True, lgag_ks=3, activation='relu6'):
        # 此处默认 expansion_factor=6 只在直接调用 EMCAD() 且不传参数时生效；项目标准入口
        # EMCADNet(... expansion_factor=2) 会显式把 2 传到这里，所以训练脚本的实际默认值是 2。
        # channels、kernel_sizes 都只读取不修改；尽管写成可变 list 默认参数，当前实现不会原地污染它们。
        # 初始化基础模块。
        super(EMCAD, self).__init__()
        # EUCB 固定使用 3x3 depth-wise convolution。
        # eucb_ks//2 在下面等于 1，所以 EUCB 内卷积保持尺寸；真正的 2 倍放大发生在 nn.Upsample。
        eucb_ks = 3  # kernel size for eucb
        # 最深层 d4 只做 MSCAM 细化，不需要先上采样；MSCB 保持 channels[0]。
        self.mscb4 = MSCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)
        # d4 不与 skip 融合，因为 x 本身就是编码器最深层 x4；先在原始最低分辨率上做注意力和多尺度细化，
        # 可以用较低空间计算量获得最强语义表示，再开始逐级恢复尺寸。

        # 第三级上采样：channels[0] -> channels[1]，空间尺寸乘 2。
        # 名称 eucb3 表示它的输出属于 d3，不是“第 3 个被执行的 EUCB”。
        self.eucb3 = EUCB(in_channels=channels[0], out_channels=channels[1], kernel_size=eucb_ks, stride=eucb_ks // 2)
        # 用上采样 d3 门控同尺度 x3；内部通道为 channels[1]/2，默认采用大核分组卷积。
        self.lgag3 = LGAG(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2, kernel_size=lgag_ks,
                          groups=channels[1] // 2)
        # groups=F_int 使默认 k=3/5 等大核卷积高度分组化；若 lgag_ks=1，LGAG 内部会把 groups 改为 1。
        # 融合 skip 后执行第三级 MSCAM 中的 MSCB，通道保持 channels[1]。
        self.mscb3 = MSCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        # 第二级上采样：channels[1] -> channels[2]，空间尺寸再乘 2。
        # eucb2 输出通道与 x2/skip[1] 完全一致，才能让 LGAG 两路投影和后续逐元素加法成立。
        self.eucb2 = EUCB(in_channels=channels[1], out_channels=channels[2], kernel_size=eucb_ks, stride=eucb_ks // 2)
        # 门控编码器第二级 skip x2。
        self.lgag2 = LGAG(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2, kernel_size=lgag_ks,
                          groups=channels[2] // 2)
        # 第二级 MSCB 细化，通道保持 channels[2]。
        self.mscb2 = MSCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        # 第一级上采样：channels[2] -> channels[3]，到达编码器最高分辨率特征层。
        # “第一级”仍只到 encoder stage1 的 1/4 尺度；解码器内部没有继续到 1/2 或原图尺度。
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks // 2)
        # 门控最浅层 skip x1；int(...) 与 //2 在正整数通道下结果相同。
        self.lgag1 = LGAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3] / 2), kernel_size=lgag_ks,
                          groups=int(channels[3] / 2))
        # 第一级 MSCB 输出解码器最高分辨率特征 d1。
        self.mscb1 = MSCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        # 四个解码级分别拥有独立 CAB；权重通道与本级特征一致。
        # CAB4/3/2/1 的参数不能共享，因为它们的通道数通常分别为 C4/C3/C2/C1，卷积 shape 不同。
        self.cab4 = CAB(channels[0])
        # 第三级 CAB。
        self.cab3 = CAB(channels[1])
        # 第二级 CAB。
        self.cab2 = CAB(channels[2])
        # 第一级 CAB。
        self.cab1 = CAB(channels[3])

        # SAB 不依赖通道数，因此四个解码级共享同一个 7x7 空间注意力模块及其参数。
        # 同一个 Module 多次调用在 PyTorch 中是合法的：每次根据当前输入重新计算权重图，但学习参数是同一份。
        self.sab = SAB()

    # x 是最深层 x4，skips 必须按 [x3,x2,x1] 从深到浅传入。
    def forward(self, x, skips):
        # 本方法没有显式检查 len(skips)、通道或尺寸。若顺序误传为 [x1,x2,x3]，通常会在 LGAG 卷积
        # 因通道不符而报错；若相邻空间尺寸不是严格 2 倍，也会在 g1+x1 或 d+x 时报 shape mismatch。
        # skips 中的张量不会被原地修改；x1/x2/x3 变量接收的是 LGAG 返回的新乘法结果。
        # MSCAM4
        # 代码没有单独 MSCAM 类；CAB 权重乘回 x 是论文式(7)的完整通道注意力。
        # self.cab4(x) 的形状为 (B,C4,1,1)，PyTorch 广播到 H4xW4 后逐元素乘 x，d4 shape 不变。
        d4 = self.cab4(x) * x
        # SAB 权重乘回 d4 是论文式(8)的完整空间注意力，形状仍为 (B,C4,H/32,W/32)。
        # self.sab(d4) 是 (B,1,H4,W4)，广播到全部 C4 通道；CAB 后接 SAB 表示先选通道、再选位置。
        d4 = self.sab(d4) * d4
        # MSCB 完成多尺度深度卷积细化；CAB -> SAB -> MSCB 合起来对应论文式(3)的 MSCAM。
        # MSCB 内部有残差，因此即使注意力压低某些响应，主分支仍能在训练中学习必要的局部修正。
        d4 = self.mscb4(d4)

        # EUCB3
        # d4 二倍上采样并把 C4 投影为 C3；PVTv2-B2 为 512 -> 320。
        # 此时得到的 d3 是“尚未融合 skip 的解码候选特征”，下一步将它作为 LGAG 的 gating 信号。
        d3 = self.eucb3(d4)

        # LGAG3
        # skips[0] 是编码器 x3；LGAG 输出被门控的 x3，通道和分辨率不变。
        # 关键点：LGAG(g=d3,x=skip) 用 d3 决定保留 skip 的哪些空间位置，但返回值内容来源仍是 skip。
        x3 = self.lgag3(g=d3, x=skips[0])

        # Additive aggregation 3
        # 解码特征与筛选后的 skip 逐元素相加，不采用通道拼接，因此 d3 仍为 C3。
        # 相比 concat，相加不会把通道翻倍，也不需要额外卷积立即降维；代价是两路必须提前严格对齐 shape。
        d3 = d3 + x3

        # MSCAM3
        # 第三级通道注意力。
        # 注意力在 skip 融合之后执行，因此其权重同时根据“上采样语义 + 编码细节”的合成特征生成。
        d3 = self.cab3(d3) * d3
        # 第三级空间注意力。
        d3 = self.sab(d3) * d3
        # 第三级多尺度卷积细化；输出形状保持不变。
        d3 = self.mscb3(d3)

        # EUCB2
        # 再次二倍上采样并执行 C3 -> C2；PVTv2-B2 为 320 -> 128。
        # 该操作把 d3 从输入 1/16 尺度恢复到 1/8 尺度。
        d2 = self.eucb2(d3)

        # LGAG2
        # 门控编码器 x2，输出保持 skips[1] 的 C2 通道。
        # x2 变量名在本函数内表示“门控后的第二级 skip”，不是编码器原始 x2 的原地改写。
        x2 = self.lgag2(g=d2, x=skips[1])

        # Additive aggregation 2
        # 与门控 skip 相加，形状保持 (B,C2,H/8,W/8)。
        # 这一层重复 d3 的融合范式，使网络逐级把高层语义与更高分辨率边缘信息结合。
        d2 = d2 + x2

        # MSCAM2
        # 第二级通道注意力。
        d2 = self.cab2(d2) * d2
        # 第二级空间注意力。
        d2 = self.sab(d2) * d2
        # 第二级多尺度卷积细化。
        d2 = self.mscb2(d2)

        # EUCB1
        # 最后一次二倍上采样并执行 C2 -> C1；PVTv2-B2 为 128 -> 64。
        # d1 是本解码器空间分辨率最高的内部特征，但还不是类别概率或二值掩膜。
        d1 = self.eucb1(d2)

        # LGAG1
        # 门控最高分辨率编码器特征 x1。
        # x1 往往包含最丰富的边界/纹理，也最容易带入背景噪声，因此仍需 LGAG 选择后再融合。
        x1 = self.lgag1(g=d1, x=skips[2])

        # Additive aggregation 1
        # 相加后 d1 保持 (B,C1,H/4,W/4)。
        d1 = d1 + x1

        # MSCAM1
        # 第一级通道注意力。
        d1 = self.cab1(d1) * d1
        # 第一级空间注意力。
        d1 = self.sab(d1) * d1
        # 最高分辨率解码特征的最终多尺度卷积细化。
        # 此行结束后 d1 通道仍是 channels[3]；类别数 num_classes 不在本类中出现。
        d1 = self.mscb1(d1)

        # 返回顺序固定为从深到浅 [d4,d3,d2,d1]；网络封装器据此连接四个分割头。
        # 返回 list 而不是只返回 d1，是为了支持 deep supervision/mutation 等多输出训练策略；
        # 在项目测试路径中，最终主预测通常取上层返回列表的最后一个 p1，而不是直接使用这里的 d1。
        return [d4, d3, d2, d1]
