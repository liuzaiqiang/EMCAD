# PyTorch 提供权重加载、张量和模型参数统计等基础能力。
import torch
# nn 用于构建输入适配层和四个分割输出头。
import torch.nn as nn
# F.interpolate 用于把四个尺度的预测统一恢复到输入分辨率。
import torch.nn.functional as F

# 导入 PVTv2 的六种层级编码器；论文实验重点使用 B0 和 B2。
from lib.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
# 导入工程扩展支持的 ResNet 层级编码器；它们不是 EMCAD 论文的主要实验骨干。
from lib.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
# EMCAD 是论文提出的解码器，接收四级编码特征并逐级恢复空间分辨率。
from lib.decoders import EMCAD


# 模型总装入口：输入适配 -> 层级编码器 -> EMCAD 解码器 -> 四个分割头。
# 论文对应：第4页 Fig.2(a)-(b)，整体架构说明见第5页 Sec.3.2。
class EMCADNet(nn.Module):
# num_classes 决定每个输出头的通道；其余参数控制 EMCAD 消融配置和编码器选择。
    def __init__(self, num_classes=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu', encoder='pvt_v2_b2', pretrain=True, pretrained_dir='./pretrained_pth/pvt/'):
# 初始化 nn.Module，使后续赋值的子模块和参数被 PyTorch 正确注册。
        super(EMCADNet, self).__init__()

        # conv block to convert single channel to 3 channels
# 医学 CT/灰度图通常为 1 通道，而 ImageNet 编码器期望 3 通道，因此建立可学习的 1->3 适配器。
        self.conv = nn.Sequential(
# 1x1 卷积只混合通道，不改变 H、W；输出形状由 (B,1,H,W) 变为 (B,3,H,W)。
            nn.Conv2d(1, 3, kernel_size=1),
# 对新生成的 3 个通道做批归一化。
            nn.BatchNorm2d(3),
# ReLU 为输入适配器加入非线性。
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
# PVTv2-B0 是论文中的 tiny encoder，四级正向通道为 [32,64,160,256]。
        if encoder == 'pvt_v2_b0':
# 创建 B0 编码器；forward 返回 [x1,x2,x3,x4]。
            self.backbone = pvt_v2_b0()
# 记录本地预训练权重路径，只有 pretrain=True 时才实际读取。
            path = pretrained_dir + '/pvt_v2_b0.pth'
# 解码器要求按“最深到最浅”排列，所以反转为 [256,160,64,32]。
            channels=[256, 160, 64, 32]
# PVTv2-B1 的各级通道与 B2 相同，主要区别是 Transformer block 深度。
        elif encoder == 'pvt_v2_b1':
# 创建 B1 编码器。
            self.backbone = pvt_v2_b1()
# B1 本地权重文件。
            path = pretrained_dir + '/pvt_v2_b1.pth'
# B1 正向通道 [64,128,320,512] 的逆序解码配置。
            channels=[512, 320, 128, 64]
# PVTv2-B2 是论文标准模型和本项目默认编码器。
        elif encoder == 'pvt_v2_b2':
# 创建 B2；默认 224 输入得到 x1..x4 空间尺寸 56、28、14、7。
            self.backbone = pvt_v2_b2()
# B2 本地 ImageNet 预训练权重文件。
            path = pretrained_dir + '/pvt_v2_b2.pth'
# d4->d1 的解码通道流为 512 -> 320 -> 128 -> 64。
            channels=[512, 320, 128, 64]
# B3 保持同一通道接口，但第三阶段 block 数更多。
        elif encoder == 'pvt_v2_b3':
# 创建 B3 编码器。
            self.backbone = pvt_v2_b3()
# B3 本地权重路径。
            path = pretrained_dir + '/pvt_v2_b3.pth'
# 仍使用 [512,320,128,64] 解码接口。
            channels=[512, 320, 128, 64]
# B4 继续增加网络深度，四级通道不变。
        elif encoder == 'pvt_v2_b4':
# 创建 B4 编码器。
            self.backbone = pvt_v2_b4()
# B4 本地权重路径。
            path = pretrained_dir + '/pvt_v2_b4.pth'
# B4 解码通道配置。
            channels=[512, 320, 128, 64]
# B5 是此文件中最深的 PVTv2 变体，仍满足同一四级接口。
        elif encoder == 'pvt_v2_b5':
# 创建 B5 编码器。
            self.backbone = pvt_v2_b5() 
# B5 本地权重路径。
            path = pretrained_dir + '/pvt_v2_b5.pth'
# B5 解码通道配置。
            channels=[512, 320, 128, 64]
# ResNet18 是工程扩展编码器；论文只说明 EMCAD 可接任意层级视觉骨干。
        elif encoder == 'resnet18':
# pretrain 参数直接传给 ResNet 工厂，True 时由 model_zoo 下载/读取 ImageNet 权重。
            self.backbone = resnet18(pretrained=pretrain)
# ResNet18 正向通道 [64,128,256,512]，此处按深到浅反排。
            channels=[512, 256, 128, 64]
# ResNet34 与 ResNet18 通道相同，只增加 BasicBlock 数量。
        elif encoder == 'resnet34':
# 创建 ResNet34。
            self.backbone = resnet34(pretrained=pretrain)
# ResNet34 解码通道配置。
            channels=[512, 256, 128, 64]
# ResNet50 使用 expansion=4 的 Bottleneck，四级通道显著增大。
        elif encoder == 'resnet50':
# 创建 ResNet50。
            self.backbone = resnet50(pretrained=pretrain)
# 正向通道 [256,512,1024,2048] 的逆序配置。
            channels=[2048, 1024, 512, 256]
# ResNet101 与 ResNet50 的通道接口相同，block 深度不同。
        elif encoder == 'resnet101':
# 创建 ResNet101。
            self.backbone = resnet101(pretrained=pretrain)  
# ResNet101 解码通道配置。
            channels=[2048, 1024, 512, 256]
# ResNet152 是最深的 ResNet 工程选项。
        elif encoder == 'resnet152':
# 创建 ResNet152。
            self.backbone = resnet152(pretrained=pretrain)  
# ResNet152 解码通道配置。
            channels=[2048, 1024, 512, 256]
# 未识别的名称不会终止程序，而是回退到 PVTv2-B2。
        else:
# 控制台明确提示调用者实际采用了默认编码器。
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
# 工程注意：这里只替换 backbone，没有改写原始 encoder 字符串。
            self.backbone = pvt_v2_b2()    # 创建编码器结构
# 保存回退模型的权重路径。
            path = pretrained_dir + '/pvt_v2_b2.pth'   # 编码器预训练权重
# 回退模型使用 B2 的解码通道。
            channels=[512, 320, 128, 64]
            
# PVT 权重由本地文件加载；ResNet 已在各自工厂函数内部处理 pretrain。
        if pretrain==True and 'pvt_v2' in encoder:
# torch.load 读取 checkpoint；当前代码未指定 map_location，设备行为由保存文件和运行环境决定。
            save_model = torch.load(path)
# 获取当前 PVT 主干完整参数字典，后面只覆盖 checkpoint 中同名的键。
            model_dict = self.backbone.state_dict()
# 过滤 checkpoint：只保留当前模型存在的键；这里按键名过滤，未单独检查张量形状。
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
# 把匹配到的预训练参数合并进当前随机初始化参数。
            model_dict.update(state_dict)
# 加载合并后的完整字典；未匹配到的参数保留初始化值。
            self.backbone.load_state_dict(model_dict)
        
# 打印编码器参数总数；ResNet 的未使用分类头参数也包含在该统计中。
        print('Model %s created, param count: %d' %
# m.numel() 对所有编码器 Parameter 的元素数求和。
                     (encoder+' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))
        
        #   decoder initialization
# 用所选骨干的四级逆序通道构造 EMCAD，并透传所有消融参数。
# 论文默认配置是 kernel_sizes=[1,3,5]、expansion_factor=2、并行深度卷积、加法聚合和 ReLU6。
        self.decoder = EMCAD(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation)
        
# 打印仅 EMCAD 解码器的参数量，便于核对轻量化设计。
        print('Model %s created, param count: %d' %
# 解码器参数统计不包含编码器和下面的四个 segmentation head。
                     ('EMCAD decoder: ', sum([m.numel() for m in self.decoder.parameters()])))

# 论文第5页 Sec.3.1.4、式(10)：最深层 d4 使用 1x1 卷积把 channels[0] 投影到类别数 K。
        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
# d3 输出头：channels[1] -> K。
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
# d2 输出头：channels[2] -> K。
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
# 最高分辨率 d1 输出头：channels[3] -> K。
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)
        
# 输入 x 约定为 (B,C,H,W)；mode 当前不改变返回值，训练策略由外部 trainer 决定。
    def forward(self, x, mode='test'):
        
        # if grayscale input, convert to 3 channels
# 只在 C=1 时使用可学习适配器；已经是 RGB/C=3 时直接送入编码器。
        if x.size()[1] == 1:
# (B,1,H,W) -> (B,3,H,W)，空间尺寸保持不变。
            x = self.conv(x)
        
        # encoder
# 所有受支持编码器都履行相同契约：返回从浅到深的四级特征。
# 默认 B2：x1=(B,64,H/4,W/4)、x2=(B,128,H/8,W/8)、x3=(B,320,H/16,W/16)、x4=(B,512,H/32,W/32)。
        x1, x2, x3, x4 = self.backbone(x)
        #print(x1.shape, x2.shape, x3.shape, x4.shape)

        # decoder
# x4 进入上采样主路，skip 按深到浅排列为 [x3,x2,x1]。
# 返回 dec_outs=[d4,d3,d2,d1]，默认通道依次为 [512,320,128,64]。
        dec_outs = self.decoder(x4, [x3, x2, x1])
        
        # prediction heads  
# d4 位于 H/32，先产生最深尺度 logits p4，形状 (B,K,H/32,W/32)。
        p4 = self.out_head4(dec_outs[0])
# d3 位于 H/16，产生 p3。
        p3 = self.out_head3(dec_outs[1])
# d2 位于 H/8，产生 p2。
        p2 = self.out_head2(dec_outs[2])
# d1 位于 H/4，产生代码命名的 p1；它是实际推理采用的最高分辨率解码头。
        p1 = self.out_head1(dec_outs[3])

# 固定放大 32 倍，把 p4 恢复到输入分辨率；隐含输入 H、W 与四级步幅兼容。
        p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
# p3 固定放大 16 倍。
        p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
# p2 固定放大 8 倍。
        p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
# p1 固定放大 4 倍；四个结果现在均为 (B,K,H,W)。
        p1 = F.interpolate(p1, scale_factor=4, mode='bilinear')

# 论文第5页 Sec.3.3 把最后解码阶段称为 p4；本代码按反方向编号，外部推理实际取返回列表 P[-1]，即这里的 p1。
        if mode == 'test':
# 测试模式仍返回全部四个 logits，不在模型内部执行 sigmoid、softmax 或多头求和。
            return [p4, p3, p2, p1]
        
# 非 test 模式返回完全相同的列表；mutation/deep supervision/last-layer 由训练器选择。
        return [p4, p3, p2, p1]
               

        
# 直接运行本文件时执行一个 GPU 形状检查；被训练脚本 import 时不会进入该分支。
if __name__ == '__main__':
# 使用默认 PVTv2-B2、1 类输出并移动到 CUDA。
    model = EMCADNet().cuda()
# 构造 352x352 的 3 通道随机输入；因为 C=3，不经过 1->3 适配器。
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

# 执行一次完整前向，P 是四个同尺寸预测组成的列表。
    P = model(input_tensor)
# 打印四个输出形状，预期每个都是 (1,1,352,352)。
    print(P[0].size(), P[1].size(), P[2].size(), P[3].size())

