# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""

# PyTorch 神经网络层；该文件实现可替换 PVTv2 的层次化 ResNet 编码器。
import torch.nn as nn
# math.sqrt 用于卷积权重的 He/Kaiming 风格初始化标准差。
import math
# model_zoo 从 PyTorch 官方 URL 下载并缓存 ImageNet 预训练参数。
import torch.utils.model_zoo as model_zoo

# import torchsummary

# 控制 `from lib.resnet import *` 时公开的构造器名称。
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           # 延续上一行并加入 ResNet-152。
           'resnet152']

# PyTorch 官方 ImageNet 权重地址；这些权重来自 ResNet，不是 EMCAD 论文训练产生的。
model_urls = {
    # ResNet-18 权重。
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    # ResNet-34 权重。
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    # ResNet-50 权重。
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # ResNet-101 权重。
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    # ResNet-152 权重。
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# ============================== 本文件阅读地图 ==============================
# 本文件实现的是“编码器”，不是完整 EMCAD 分割网络。它负责把一张 RGB 图像逐级压缩成四张不同尺度的特征图。
# lib/networks.py 中的 EMCADNet 会调用这里的 resnet18/34/50/101/152 工厂函数，再把本文件 forward 返回的
# [x1,x2,x3,x4] 交给 EMCAD 解码器。也就是说，本文件解决“从图像提取多尺度语义”，decoders.py 解决“恢复分辨率并分割”。
#
# 标准输入 x=[B,3,H,W] 的空间流向：
#   7x7 stride=2 卷积 -> [B,64,H/2,W/2]
#   3x3 stride=2 最大池化 -> [B,64,H/4,W/4]
#   layer1 -> x1，保持 H/4
#   layer2 -> x2，变为 H/8
#   layer3 -> x3，变为 H/16
#   layer4 -> x4，变为 H/32。
# 对 352x352 输入，四级空间尺寸就是 88x88、44x44、22x22、11x11，与默认 PVTv2-B2 的尺度完全对齐。
#
# 不同 ResNet 变体的关键区别有两类：
#   1. ResNet18/34 使用 BasicBlock，expansion=1，四级通道为 [64,128,256,512]；
#   2. ResNet50/101/152 使用 Bottleneck，expansion=4，四级通道为 [256,512,1024,2048]。
# 因此替换骨干时，EMCAD 的 channels 参数也必须同步变化；lib/networks.py 已为每个构造器给出对应逆序列表。
#
# 残差块的核心不是简单“多加一条线”，而是让输出等于 F(x)+shortcut(x)。
# 当输入输出形状相同，shortcut 直接传 x；当通道数或空间尺寸变化，shortcut 必须用 1x1 卷积投影后才能逐元素相加。
# 这条捷径既保留原始信息，也为梯度提供更直接的传播路径，是深层 ResNet 能稳定训练的重要原因。
# ===========================================================================


# 创建不带偏置的 3x3 卷积；padding=1 在 stride=1 时保持空间尺寸。
def conv3x3(in_planes, out_planes, stride=1):
    # 参数逐项说明：in_planes 是输入通道，out_planes 是输出通道，stride 决定是否下采样。
    # 对输入 [B,in_planes,H,W]：stride=1 时输出 [B,out_planes,H,W]；stride=2 时通常输出约 [B,out_planes,H/2,W/2]。
    # 把这个常用组合封装成函数，可确保 BasicBlock 两处 3x3 卷积始终使用相同 padding/bias 约定。
    """3x3 convolution with padding"""
    # BatchNorm 紧随卷积时不需要卷积 bias。
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     # 3x3 核使用一圈零填充。
                     padding=1, bias=False)


# ResNet-18/34 使用的两层基础残差块；结构来自 ResNet 原论文，而非 EMCAD 创新模块。
class BasicBlock(nn.Module):
    # ------------------------------ BasicBlock 数据流 ------------------------------
    # 主分支：x -> 3x3卷积 -> BN -> ReLU -> 3x3卷积 -> BN -> 与 residual 相加 -> ReLU。
    # 捷径分支：默认 residual=x；若 shape 变化，则 residual=downsample(x)。
    # 最终相加要求两边四个维度完全相同，这也是 __init__ 必须接收 downsample 的直接原因。
    # 例：layer2 首块可把 [B,64,88,88] 变成 [B,128,44,44]；主分支和捷径分支都必须完成同样的变化。
    # ------------------------------------------------------------------------------
    # 输出通道等于 planes，不做瓶颈扩张。
    expansion = 1

    # stride 通常只在每个 stage 的首块取 2；downsample 对齐残差尺寸/通道。
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # inplanes 描述输入 x 的通道；planes 描述主分支两层卷积的输出通道。
        # downsample 不是布尔值，而是由 ResNet._make_layer 预先构造好的 nn.Sequential，或表示无需投影的 None。
        # 初始化 Module。
        super(BasicBlock, self).__init__()
        # 第一层 3x3 可通过 stride=2 完成空间下采样和通道映射。
        self.conv1 = conv3x3(inplanes, planes, stride)
        # 归一化第一层输出通道。
        self.bn1 = nn.BatchNorm2d(planes)
        # 原地 ReLU 减少额外激活内存。
        self.relu = nn.ReLU(inplace=True)
        # 第二层 3x3 始终 stride=1，输出 planes 通道。
        self.conv2 = conv3x3(planes, planes)
        # 第二次 BN 位于残差相加之前。
        self.bn2 = nn.BatchNorm2d(planes)
        # 保存可选 1x1 残差投影。
        self.downsample = downsample
        # 保存步长用于结构描述/检查。
        self.stride = stride

    # 输入 x 形状通常为 [B,inplanes,H,W]。
    def forward(self, x):
        # forward 不创建新参数，只按 __init__ 注册好的模块执行张量运算；同一块在每次迭代中重复使用同一组权重。
        # 默认捷径分支直接引用输入。
        residual = x

        # 主分支第一层卷积，stride=2 时 H/W 减半。
        out = self.conv1(x)
        # 第一层批归一化。
        out = self.bn1(out)
        # 第一层激活。
        out = self.relu(out)

        # 主分支第二层卷积。
        out = self.conv2(out)
        # 相加前归一化，不在此处先激活。
        out = self.bn2(out)

        # stage 首块或通道变化时，捷径分支需投影到相同 shape。
        if self.downsample is not None:
            # 对原输入执行 1x1 卷积和 BN。
            residual = self.downsample(x)

        # 主分支与捷径分支逐元素相加；此时 shape 必须一致。
        # “逐元素”表示相同批次、通道、行、列位置一一相加，不是通道拼接，所以相加后通道数不会翻倍。
        out += residual
        # 残差相加后执行 ReLU。
        out = self.relu(out)

        # 返回当前残差块输出。
        return out


# ResNet-50/101/152 使用的 1x1-3x3-1x1 瓶颈残差块。
class Bottleneck(nn.Module):
    # ------------------------------ Bottleneck 数据流 ------------------------------
    # 若 planes=P，则主分支通道依次为：inplanes -> P -> P -> 4P。
    # 第一层 1x1 负责降低/整理通道，中间 3x3 负责昂贵的空间建模，最后 1x1 再恢复到 4P；这比全程在 4P 通道做 3x3 更省计算。
    # expansion=4 指“最终输出相对 planes 的倍率”，不是相对任意输入 inplanes 固定放大四倍。
    # stage 内第一个块可能改变通道和分辨率，后续块通常输入输出均为 4P，可以直接使用恒等捷径。
    # -------------------------------------------------------------------------------
    # 最后一层把内部 planes 扩张到 4*planes。
    expansion = 4

    # 参数含义与 BasicBlock 相同，但输出通道由 expansion 决定。
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # 例如 ResNet50 的 layer1 首块参数是 inplanes=64、planes=64，最终输出不是64而是64*4=256通道。
        # 初始化 Module。
        super(Bottleneck, self).__init__()
        # 第一层 1x1 把输入通道压到 planes。
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # 归一化压缩后的通道。
        self.bn1 = nn.BatchNorm2d(planes)
        # 中间 3x3 负责空间建模，并在需要时 stride=2 下采样。
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               # padding=1 保持 stride=1 时尺寸不变。
                               padding=1, bias=False)
        # 归一化中间卷积输出。
        self.bn2 = nn.BatchNorm2d(planes)
        # 最后一层 1x1 扩张到 4*planes 通道。
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        # 归一化扩张后的输出。
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        # 三个位置复用同一无状态 ReLU 模块。
        self.relu = nn.ReLU(inplace=True)
        # 保存残差投影。
        self.downsample = downsample
        # 保存空间步长。
        self.stride = stride

    # 输入 x 通过三层主分支并与捷径相加。
    def forward(self, x):
        # 这里把 stride 放在中间 3x3 卷积；因此需要下采样时，主分支在第二层缩小 H/W，捷径同步按相同步长投影。
        # 默认捷径直接使用输入。
        residual = x

        # 1x1 通道压缩。
        out = self.conv1(x)
        # BN。
        out = self.bn1(out)
        # ReLU。
        out = self.relu(out)

        # 3x3 空间卷积/可选下采样。
        out = self.conv2(out)
        # BN。
        out = self.bn2(out)
        # ReLU。
        out = self.relu(out)

        # 1x1 通道扩张。
        out = self.conv3(out)
        # 相加前最后一次 BN。
        out = self.bn3(out)

        # shape 不一致时投影捷径。
        if self.downsample is not None:
            # 1x1 卷积同时完成通道/步长对齐。
            residual = self.downsample(x)

        # 残差融合。
        # 此处仍是逐元素求和而非 concat；downsample 的职责就是保证 residual 已变成与 out 完全相同的 [B,4P,H',W']。
        out += residual
        # 融合后激活。
        out = self.relu(out)

        # 返回瓶颈块输出。
        return out


# 四级层次化 ResNet；EMCAD 论文主文 PDF5 §3.2 只说明解码器兼容层次化骨干，未报告 ResNet 实验。
class ResNet(nn.Module):

    # ------------------------------ 构造参数与状态 ------------------------------
    # block：传入类本身（BasicBlock 或 Bottleneck），_make_layer 会多次实例化它。
    # layers：长度为4的列表，逐项表示 layer1~layer4 各有多少个残差块，例如 ResNet34 是 [3,4,6,3]。
    # num_classes：仅决定遗留分类头 fc 的输出数；当前分割 forward 不调用 fc，因此它不等于 EMCAD 的分割类别数。
    # deep_base：False 使用经典单层 7x7 stem；True 使用三层 3x3 stem。当前工厂函数默认都传 False。
    # stem_width：只在 deep_base=True 时决定 stem 中间宽度，默认最终仍得到 2*32=64 通道。
    # self.inplanes 是“正在构造的下一个块会接收多少通道”的可变记账值，每建完一个 stage 都会被 _make_layer 更新。
    # --------------------------------------------------------------------------
    # block 决定基础/瓶颈结构，layers 给出四个 stage 的块数。
    def __init__(self, block, layers, num_classes=1000, deep_base=False, stem_width=32):
        # 标准 stem 输出64通道；deep_base 最后输出 2*stem_width。
        self.inplanes = stem_width * 2 if deep_base else 64

        # 初始化 Module；self.inplanes 只是普通属性，可在 super 前赋值。
        super(ResNet, self).__init__()
        # deep_base 用三个 3x3 卷积替代标准 7x7 stem。
        if deep_base:
            # 顺序堆叠三层卷积。
            self.conv1 = nn.Sequential(
                # 第一层把 RGB 映射到 stem_width，并下采样2倍。
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                # 第一层 BN。
                nn.BatchNorm2d(stem_width),
                # 第一层 ReLU。
                nn.ReLU(inplace=True),
                # 第二层保持尺寸和通道。
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                # 第二层 BN。
                nn.BatchNorm2d(stem_width),
                # 第二层 ReLU。
                nn.ReLU(inplace=True),
                # 第三层扩到2*stem_width；外部 bn1/relu 会处理其输出。
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        # 默认走经典 ResNet 7x7 stem。
        else:
            # RGB -> 64，7x7 stride2，空间尺寸减半。
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   # 后接 BN，因此禁用 bias。
                                   bias=False)

        # 归一化 stem 输出，通道数由 self.inplanes 决定。
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        # stem 激活。
        self.relu = nn.ReLU(inplace=True)
        # 3x3 最大池化 stride2，使进入 layer1 前总步幅为4。
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # stage1 保持 1/4 分辨率，基础通道64。
        self.layer1 = self._make_layer(block, 64, layers[0])
        # stage2 首块 stride2，输出 1/8 分辨率。
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # stage3 输出 1/16 分辨率。
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # stage4 输出 1/32 分辨率。
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 下面 avgpool/fc 是原始 ImageNet 分类实现遗留。保留它们可让官方完整 state_dict 严格加载，但分割前向不会产生分类结果。
        # 分类模型遗留的全局池化；当前分割 forward 不调用，但参数外没有可训练状态。
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # 分类头遗留参数仍注册在模型中，但分割 forward 不执行它。
        # 因为 fc 仍是已注册子模块，它的参数会出现在 model.parameters()/state_dict() 中，也会被优化器纳入，尽管没有梯度时不会更新。
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 这里是显式初始化流程：它会访问 stem、四个 stage、投影捷径等所有已注册卷积/BN，也包括 deep_base 中的层。
        # Linear 分类头未在此分支中单独处理，继续使用 nn.Linear 自己的默认初始化。
        # 遍历所有已构造子模块执行原始初始化。
        for m in self.modules():
            # 卷积权重使用与输出通道相关的正态初始化。
            if isinstance(m, nn.Conv2d):
                # 计算 kernel面积 * 输出通道。
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # 原地采样均值0、标准差sqrt(2/n)的权重。
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # BatchNorm 初始为恒等仿射。
            elif isinstance(m, nn.BatchNorm2d):
                # gamma=1。
                m.weight.data.fill_(1)
                # beta=0。
                m.bias.data.zero_()

    # 构建一个 stage；首块可下采样，后续块保持 shape。
    def _make_layer(self, block, planes, blocks, stride=1):
        # 该方法在 __init__ 中调用四次。它返回的是一个完整 stage，而不是只返回一层卷积。
        # planes 是块的“基础通道”；stage 真实输出通道始终为 planes*block.expansion。
        # blocks 决定该 stage 重复多少个残差块；只有第一个块使用传入 stride，余下块的 stride 都保持默认1。
        # 默认捷径无需投影。
        downsample = None
        # 空间步长变化或通道不等于 planes*expansion 时必须对齐残差。
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 两个条件任一成立都不能直接做 F(x)+x：前者表示 H/W 不同，后者表示通道数不同。
            # 投影捷径用最便宜的 1x1 卷积完成对齐，并用 BN 保持与主分支相近的数值尺度。
            # 1x1 卷积加 BN 的投影捷径。
            downsample = nn.Sequential(
                # 同时改变通道和可选空间步长。
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          # 1x1 不需 padding，禁用 bias。
                          kernel_size=1, stride=stride, bias=False),
                # 归一化投影输出。
                nn.BatchNorm2d(planes * block.expansion),
            )

        # 收集当前 stage 的残差块。
        # 先使用普通 Python 列表是为了动态 append；函数末尾再展开成 nn.Sequential，PyTorch 才能注册其中参数。
        layers = []
        # 首块接收旧 self.inplanes，并承担 stride/downsample。
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 更新后续块的输入通道到当前 stage 输出通道。
        # 这一赋值也会影响下一次 _make_layer 调用。例如构造完 layer1 后，layer2 能知道自己接收的是64或256通道。
        self.inplanes = planes * block.expansion
        # 从第2个块开始不再下采样。
        for i in range(1, blocks):
            # 输入输出通道一致，stride 默认1，downsample 默认None。
            layers.append(block(self.inplanes, planes))

        # 将 Python 列表注册成可调用 Sequential stage。
        # 星号把列表元素展开为位置参数，等价于 nn.Sequential(layers[0], layers[1], ...)。
        return nn.Sequential(*layers)

    # ------------------------------ 分割前向接口 ------------------------------
    # 这里与经典 torchvision ResNet 最大的行为差异，是不把 layer4 输出送进 avgpool/fc，而是保存并返回四个 stage。
    # 返回列表必须保持从浅到深顺序：[1/4,1/8,1/16,1/32]，因为 networks.py 会按 x1,x2,x3,x4 解包。
    # 对 352 输入：ResNet18/34 返回 [B,64,88,88]、[B,128,44,44]、[B,256,22,22]、[B,512,11,11]；
    # ResNet50+ 的空间尺寸相同，但通道依次是 256、512、1024、2048。
    # -------------------------------------------------------------------------
    # 分割编码器前向只返回四级特征，不执行 avgpool/fc 分类头。
    def forward(self, x):
        # 输入 [B,3,H,W] 经 stem 卷积得到约 [B,64,H/2,W/2]。
        x = self.conv1(x)
        # stem BN。
        x = self.bn1(x)
        # stem ReLU。
        x = self.relu(x)
        # 最大池化到 H/4、W/4。
        x = self.maxpool(x)

        # 依次保存由浅到深的四级特征，接口与 PVTv2 forward 一致。
        # features 是当前这次 forward 的临时 Python 列表，不是模型参数，也不会写入 state_dict。
        features = []

        # x1：ResNet18/34 为64通道，ResNet50+为256通道，空间1/4。
        x = self.layer1(x)
        # 加入第一级跳连特征。
        features.append(x)
        # x2：128或512通道，空间1/8。
        x = self.layer2(x)
        # 加入第二级特征。
        features.append(x)
        # x3：256或1024通道，空间1/16。
        x = self.layer3(x)
        # 加入第三级特征。
        features.append(x)
        # x4：512或2048通道，空间1/32。
        x = self.layer4(x)
        # 加入最深层语义特征。
        features.append(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # 列表中的张量都保留计算图，所以 EMCAD 从任一级特征反向传播的梯度都能继续进入相应 ResNet 层。
        # 返回 [x1,x2,x3,x4]；lib/networks.py 会逆序传给 EMCAD 解码器。
        return features


# 下列五个工厂函数把“块类型 + 每阶段块数”封装成常用 ResNet 名称，供 networks.py 通过 encoder 字符串选择。
# pretrained=False 只是不加载 ImageNet 权重，并不会删掉任何层；pretrained=True 会从 URL 下载或命中本机 PyTorch 缓存。
# load_state_dict 默认 strict=True，因此官方权重键名和形状必须与当前 ResNet（包括未使用的 fc）一致。


# 构造 ResNet-18：四个 stage 的 BasicBlock 数量为 2/2/2/2。
def resnet18(pretrained=False, **kwargs):
    # **kwargs 原样传给 ResNet，可用于覆盖 num_classes/deep_base/stem_width；当前 EMCADNet 调用没有额外传值。
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 实例化基础块网络；四级通道为 [64,128,256,512]。
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # 调用方明确要求时加载官方 ImageNet 参数。
    if pretrained:
        # load_url 下载/缓存并返回 state_dict，再严格加载。
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    # 返回可作为 EMCAD 编码器的模型。
    return model


# 构造 ResNet-34：BasicBlock 数量为 3/4/6/3。
def resnet34(pretrained=False, **kwargs):
    # 相比 ResNet18，它只增加各 stage 的 BasicBlock 数量，四级接口和通道数保持不变，所以解码器 channels 无需另改。
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 四级输出通道仍为 [64,128,256,512]。
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    # 原代码创建模型参数字典但活动加载逻辑未使用它；只保留现状。
    # 该局部变量不会改变模型，也不会复制底层参数；state_dict() 返回键到张量的有序映射，函数结束后变量即失去引用。
    model_dict = model.state_dict()

    # 可选加载 ImageNet 预训练权重。
    if pretrained:
        # 控制台提示开始加载。
        print('Using pretrained weight!')
        # 从官方 URL 取得参数；行尾英文注释是原代码遗留。
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])  # Modify 'model_dir' according to your own path
        # 控制台提示下载/读取完成。
        print('Petrain Model Have been loaded!')
        # pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # 严格加载完整 state_dict，包括当前未执行的分类 fc 参数。
        model.load_state_dict(pretrained_dict)
    # 返回模型。
    return model


# 构造 ResNet-50：使用扩张系数4的 Bottleneck，块数3/4/6/3。
def resnet50(pretrained=False, **kwargs):
    # 从此变体开始使用 Bottleneck；虽然块数配置也为3/4/6/3，但每块有三层卷积且 expansion=4，不能按 ResNet34 通道理解。
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 四级输出通道为 [256,512,1024,2048]。
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # 可选加载官方 ImageNet 参数。
    if pretrained:
        # 输出提示。
        print('Using pretrained weight!')
        # 下载/缓存并严格加载。
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    # 返回模型。
    return model


# 构造 ResNet-101：第三 stage 加深到23个 Bottleneck。
def resnet101(pretrained=False, **kwargs):
    # 名称中的101来自完整分类网络的层数计数；本函数最显著的结构变化是 layer3 从6个 Bottleneck 增至23个。
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 实例化 3/4/23/3 深度配置。
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    # 可选加载 ImageNet 参数。
    if pretrained:
        # 从官方 URL 加载。
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    # 返回模型。
    return model


# 构造 ResNet-152：四个 stage 的 Bottleneck 数量为3/8/36/3。
def resnet152(pretrained=False, **kwargs):
    # layer2 和 layer3 进一步加深为8与36个 Bottleneck；更深不等于在当前医学数据上必然更好，还会显著增加计算和显存。
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 实例化最深配置；输出通道仍为 [256,512,1024,2048]。
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    # 可选加载官方 ImageNet 参数。
    if pretrained:
        # 从官方 URL 取得并加载完整 state_dict。
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    # 返回模型。
    return model
# net = resnet34(pretrained=False)
# torchsummary.summary(net, (3, 512, 512))
