# PyTorch 用于 checkpoint 读取、随机输入构造和参数操作。
import torch
# nn 提供卷积、归一化、激活以及 EMCADNet 的 Module 基类。
import torch.nn as nn
# F.interpolate 将四个解码尺度的 logits 上采样到输入分辨率。
import torch.nn.functional as F

# 导入六种 PVTv2 层级编码器。
from lib.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
# 导入五种 ResNet 工程扩展编码器。
from lib.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
# 导入论文提出的 EMCAD 解码器。
from lib.decoders import EMCAD

# 工程定位：当前 train/test 脚本均从 lib.networks 导入 EMCADNet，本文件是根目录副本并带额外预训练诊断逻辑。
# 下方已有字符串是原文件说明，本次仅在其外部补充 # 注释，不改动其内容。
"""
 EMCADNet 总装（encoder + decoder + 输出头 + forward 逻辑）

负责：
    encoder 选择（PVTv2 / ResNet）
    pretrained 权重加载
    decoder 挂载
    多尺度输出（[p4, p3, p2, p1]）


Head & 上采样策略
在 networks.py 中：
out_head4/3/2/1
scale factor 固定为 32 / 16 / 8 / 4
多尺度 supervision 是“结构上支持，loss 决定是否启用”

"""


# 模型总装类：输入通道适配、编码器、EMCAD 解码器和四个 segmentation head。
# 论文对应第4页 Fig.2(a)-(b)及第5页 Sec.3.2；本副本的诊断打印属于工程实现。
class EMCADNet(nn.Module):
    # num_classes 是输出类别数；其余参数控制编码器和 EMCAD 的消融配置。
    def __init__(self, num_classes=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3,
                 activation='relu',
                 # encoder 和 pretrain 分别控制骨干类型与是否使用预训练，二者是相互独立的选择。
                 encoder='pvt_v2_b2', pretrain=True, pretrained_dir='./pretrained_pth/pvt/'):
        # 初始化 nn.Module 的模块注册机制。
        super(EMCADNet, self).__init__()

        # conv block to convert single channel to 3 channels
        # 医学灰度输入需要先适配为 ImageNet 编码器通常使用的 3 通道。
        self.conv = nn.Sequential(
            # 1x1 卷积执行 C:1->3，H、W 不变。
            nn.Conv2d(1, 3, kernel_size=1),
            # 对三个适配通道归一化。
            nn.BatchNorm2d(3),
            # 加入 ReLU 非线性。
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        # PVTv2-B0 是论文 tiny 版本，正向通道为 [32,64,160,256]。
        if encoder == 'pvt_v2_b0':
            # 创建 B0 骨干。
            self.backbone = pvt_v2_b0()
            # 记录 B0 本地权重路径。
            path = pretrained_dir + '/pvt_v2_b0.pth'
            # 解码器按最深到最浅读取通道，因此使用逆序列表。
            channels = [256, 160, 64, 32]
        # PVTv2-B1 与 B2 共享四级通道，主要差异是 block 数量。
        elif encoder == 'pvt_v2_b1':
            # 创建 B1 骨干。
            self.backbone = pvt_v2_b1()
            # B1 本地权重路径。
            path = pretrained_dir + '/pvt_v2_b1.pth'
            # d4->d1 通道配置。
            channels = [512, 320, 128, 64]
        # PVTv2-B2 是论文标准版本和默认编码器。
        elif encoder == 'pvt_v2_b2':
            # 创建 B2 骨干。
            self.backbone = pvt_v2_b2()
            # B2 本地权重路径。
            path = pretrained_dir + '/pvt_v2_b2.pth'
            # 默认 EMCAD 解码通道流 512->320->128->64。
            channels = [512, 320, 128, 64]
        # B3 加深第三阶段，四级接口不变。
        elif encoder == 'pvt_v2_b3':
            # 创建 B3 骨干。
            self.backbone = pvt_v2_b3()
            # B3 本地权重路径。
            path = pretrained_dir + '/pvt_v2_b3.pth'
            # B3 解码通道配置。
            channels = [512, 320, 128, 64]
        # B4 继续增加层数，仍保持相同通道接口。
        elif encoder == 'pvt_v2_b4':
            # 创建 B4 骨干。
            self.backbone = pvt_v2_b4()
            # B4 本地权重路径。
            path = pretrained_dir + '/pvt_v2_b4.pth'
            # B4 解码通道配置。
            channels = [512, 320, 128, 64]
        # B5 是此处最深的 PVTv2 变体。
        elif encoder == 'pvt_v2_b5':
            # 创建 B5 骨干。
            self.backbone = pvt_v2_b5()
            # B5 本地权重路径。
            path = pretrained_dir + '/pvt_v2_b5.pth'
            # B5 解码通道配置。
            channels = [512, 320, 128, 64]
        # ResNet18 是仓库工程扩展，不是 EMCAD 论文的主要实验骨干。
        elif encoder == 'resnet18':
            # ResNet 工厂内部依据 pretrain 决定是否加载 ImageNet 权重。
            self.backbone = resnet18(pretrained=pretrain)
            # ResNet18 正向 [64,128,256,512]，解码器使用逆序。
            channels = [512, 256, 128, 64]
        # ResNet34 通道接口相同，BasicBlock 数量更多。
        elif encoder == 'resnet34':
            # 创建 ResNet34。
            self.backbone = resnet34(pretrained=pretrain)
            # ResNet34 解码通道配置。
            channels = [512, 256, 128, 64]
        # ResNet50 改用 expansion=4 的 Bottleneck。
        elif encoder == 'resnet50':
            # 创建 ResNet50。
            self.backbone = resnet50(pretrained=pretrain)
            # 正向 [256,512,1024,2048] 的逆序配置。
            channels = [2048, 1024, 512, 256]
        # ResNet101 与 ResNet50 通道相同、深度不同。
        elif encoder == 'resnet101':
            # 创建 ResNet101。
            self.backbone = resnet101(pretrained=pretrain)
            # ResNet101 解码通道配置。
            channels = [2048, 1024, 512, 256]
        # ResNet152 是最深的 ResNet 选项。
        elif encoder == 'resnet152':
            # 创建 ResNet152。
            self.backbone = resnet152(pretrained=pretrain)
            # ResNet152 解码通道配置。
            channels = [2048, 1024, 512, 256]
        # 未知编码器名称回退为 PVTv2-B2。
        else:
            # 提示实际回退行为。
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
            # 创建回退骨干，但原 encoder 字符串不变。
            self.backbone = pvt_v2_b2()
            # 保存回退骨干权重路径。
            path = pretrained_dir + '/pvt_v2_b2.pth'
            # 使用 B2 通道契约。
            channels = [512, 320, 128, 64]

        # PVT 使用本地 checkpoint；ResNet 的预训练已在工厂函数内完成。
        if pretrain == True and 'pvt_v2' in encoder:
            # 读取 checkpoint；当前实现未指定 map_location。
            save_model = torch.load(path)
            # 本副本额外打印实际读取路径，用于排查是否选错权重文件。
            print("[Pretrain] path:", path)
            # 本副本额外打印 checkpoint 顶层键数量；它不等价于成功匹配的参数数。
            print("[Pretrain] ckpt params:", len(save_model))
            # 获取当前骨干完整 state_dict。
            model_dict = self.backbone.state_dict()
            # 只按同名键筛选 checkpoint；没有在此显式筛掉形状不一致的张量。
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            # 本副本打印真正匹配到的键数量。
            print("[Pretrain] matched params:", len(state_dict))
            # 本副本的工程保护：若一个键都未匹配则立即失败；lib/networks.py 没有此断言。
            assert len(state_dict) > 0, "No keys matched! Pretrain NOT loaded."
            # 用匹配参数覆盖当前随机初始化值。
            model_dict.update(state_dict)
            # 加载合并后的完整参数字典。
            self.backbone.load_state_dict(model_dict)

        # 打印编码器参数总数。
        print('Model %s created, param count: %d' %
              # ResNet 的 avgpool/fc 虽不参与分割前向，其参数仍包含在该总数中。
              (encoder + ' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))

        #   decoder initialization
        # 按选定骨干的逆序通道创建 EMCAD，并透传多尺度卷积与 LGAG 配置。
        self.decoder = EMCAD(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                             dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation)

        # 打印解码器参数量。
        print('Model %s created, param count: %d' %
              # 该统计仅覆盖 EMCAD，不包含四个输出头。
              ('EMCAD decoder: ', sum([m.numel() for m in self.decoder.parameters()])))

        # 论文第5页 Sec.3.1.4、式(10)：d4 的 1x1 segmentation head，C4->K。
        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        # d3 输出头，C3->K。
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        # d2 输出头，C2->K。
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        # d1 输出头，C1->K。
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)

    # 输入 x 为 NCHW；mode 参数当前不改变前向结果。
    def forward(self, x, mode='test'):

        # if grayscale input, convert to 3 channels
        # 灰度图 C=1 时才启用输入适配器。
        if x.size()[1] == 1:
            # (B,1,H,W)->(B,3,H,W)。
            x = self.conv(x)

        # encoder
        # 默认 B2 输出：x1=64@1/4、x2=128@1/8、x3=320@1/16、x4=512@1/32。
        x1, x2, x3, x4 = self.backbone(x)
        # print(x1.shape, x2.shape, x3.shape, x4.shape)

        # decoder
        # 最深 x4 进入主路，skip 按 [x3,x2,x1] 传入；返回 [d4,d3,d2,d1]。
        dec_outs = self.decoder(x4, [x3, x2, x1])

        # prediction heads  
        # 最深 d4 产生代码命名 p4，原始空间尺度为 1/32。
        p4 = self.out_head4(dec_outs[0])
        # d3 产生 p3，尺度 1/16。
        p3 = self.out_head3(dec_outs[1])
        # d2 产生 p2，尺度 1/8。
        p2 = self.out_head2(dec_outs[2])
        # 最高分辨率 d1 产生代码命名 p1，尺度 1/4。
        p1 = self.out_head1(dec_outs[3])

        # 四个 head 使用固定倍率恢复输入分辨率；这假设编码器步幅为 4/8/16/32。
        p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
        # p3 放大 16 倍。
        p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
        # p2 放大 8 倍。
        p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
        # p1 放大 4 倍，此时四个 logits 形状均为 (B,K,H,W)。
        p1 = F.interpolate(p1, scale_factor=4, mode='bilinear')

        # 论文第5页把最后解码阶段称为 p4；代码编号方向相反，实际推理消费者取 P[-1]，即这里的 p1。
        if mode == 'test':
            # 模型内部不做 sigmoid/softmax，也不聚合预测，直接返回全部 logits。
            return [p4, p3, p2, p1]

        # 训练模式返回相同列表，监督规则由外部 trainer 决定。
        return [p4, p3, p2, p1]


# 仅直接执行本文件时运行的 CUDA 形状演示；当前正式脚本不会从这里启动训练。
if __name__ == '__main__':
    # 创建默认模型并移动到 GPU。
    model = EMCADNet().cuda()
    # 3 通道随机输入绕过灰度适配器。
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    # 完整前向得到四个预测。
    P = model(input_tensor)
    # 预期打印四个 (1,1,352,352) 张量尺寸。
    print(P[0].size(), P[1].size(), P[2].size(), P[3].size())
