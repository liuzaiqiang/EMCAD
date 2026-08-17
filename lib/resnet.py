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
#import torchsummary

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


# 创建不带偏置的 3x3 卷积；padding=1 在 stride=1 时保持空间尺寸。
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    # BatchNorm 紧随卷积时不需要卷积 bias。
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     # 3x3 核使用一圈零填充。
                     padding=1, bias=False)


# ResNet-18/34 使用的两层基础残差块；结构来自 ResNet 原论文，而非 EMCAD 创新模块。
class BasicBlock(nn.Module):
    # 输出通道等于 planes，不做瓶颈扩张。
    expansion = 1

    # stride 通常只在每个 stage 的首块取 2；downsample 对齐残差尺寸/通道。
    def __init__(self, inplanes, planes, stride=1, downsample=None):
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
        out += residual
        # 残差相加后执行 ReLU。
        out = self.relu(out)

        # 返回当前残差块输出。
        return out


# ResNet-50/101/152 使用的 1x1-3x3-1x1 瓶颈残差块。
class Bottleneck(nn.Module):
    # 最后一层把内部 planes 扩张到 4*planes。
    expansion = 4

    # 参数含义与 BasicBlock 相同，但输出通道由 expansion 决定。
    def __init__(self, inplanes, planes, stride=1, downsample=None):
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
        out += residual
        # 融合后激活。
        out = self.relu(out)

        # 返回瓶颈块输出。
        return out


# 四级层次化 ResNet；EMCAD 论文主文 PDF5 §3.2 只说明解码器兼容层次化骨干，未报告 ResNet 实验。
class ResNet(nn.Module):

    # block 决定基础/瓶颈结构，layers 给出四个 stage 的块数。
    def __init__(self, block, layers, num_classes=1000,deep_base=False,stem_width=32):
        # 标准 stem 输出64通道；deep_base 最后输出 2*stem_width。
        self.inplanes = stem_width*2 if deep_base else 64
        
        # 初始化 Module；self.inplanes 只是普通属性，可在 super 前赋值。
        super(ResNet, self).__init__()
        # deep_base 用三个 3x3 卷积替代标准 7x7 stem。
        if deep_base:
            # 顺序堆叠三层卷积。
            self.conv1= nn.Sequential(
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
                nn.Conv2d(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False),
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
        # 分类模型遗留的全局池化；当前分割 forward 不调用，但参数外没有可训练状态。
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # 分类头遗留参数仍注册在模型中，但分割 forward 不执行它。
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        # 默认捷径无需投影。
        downsample = None
        # 空间步长变化或通道不等于 planes*expansion 时必须对齐残差。
        if stride != 1 or self.inplanes != planes * block.expansion:
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
        layers = []
        # 首块接收旧 self.inplanes，并承担 stride/downsample。
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 更新后续块的输入通道到当前 stage 输出通道。
        self.inplanes = planes * block.expansion
        # 从第2个块开始不再下采样。
        for i in range(1, blocks):
            # 输入输出通道一致，stride 默认1，downsample 默认None。
            layers.append(block(self.inplanes, planes))

        # 将 Python 列表注册成可调用 Sequential stage。
        return nn.Sequential(*layers)

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

        # 返回 [x1,x2,x3,x4]；lib/networks.py 会逆序传给 EMCAD 解码器。
        return features


# 构造 ResNet-18：四个 stage 的 BasicBlock 数量为 2/2/2/2。
def resnet18(pretrained=False, **kwargs):
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
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 四级输出通道仍为 [64,128,256,512]。
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    # 原代码创建模型参数字典但活动加载逻辑未使用它；只保留现状。
    model_dict = model.state_dict()


    # 可选加载 ImageNet 预训练权重。
    if pretrained:
        # 控制台提示开始加载。
        print('Using pretrained weight!')
        # 从官方 URL 取得参数；行尾英文注释是原代码遗留。
        pretrained_dict=model_zoo.load_url(model_urls['resnet34'])# Modify 'model_dir' according to your own path
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
