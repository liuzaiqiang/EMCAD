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


# ===========================================================
# 这个文件不负责定义 PVTv2、ResNet 或 EMCAD 内部的具体算子，而是负责把它们“组装”成一个可训练的分割网络。
# 可以把 EMCADNet 理解为总调度器：它规定数据依次经过哪些模块、四级特征如何传递、最终怎样得到分割图。
#
# 完整数据流如下（B=批大小，K=num_classes，H/W=输入高宽）：
#   输入 x:[B,C,H,W]
#     -> 若 C=1，先经 self.conv 变为 [B,3,H,W]
#     -> backbone 提取 [x1,x2,x3,x4] 四级特征
#     -> EMCAD 以 x4 为主输入，并使用 x3/x2/x1 作为三条跳跃连接
#     -> 得到 [d4,d3,d2,d1] 四级解码特征
#     -> 四个 1x1 输出头把通道数分别压到 K
#     -> 双线性插值恢复到输入分辨率
#     -> 返回 [p4,p3,p2,p1] 四个“原始 logits”供深监督或最终预测使用。
#
# 以项目默认的 PVTv2-B2 和 352x352 输入为例，最容易建立尺寸直觉：
#   x1=[B, 64,88,88]，x2=[B,128,44,44]，x3=[B,320,22,22]，x4=[B,512,11,11]；
#   d4=[B,512,11,11]，d3=[B,320,22,22]，d2=[B,128,44,44]，d1=[B, 64,88,88]；
#   四个输出头先得到 11/22/44/88 四种分辨率的 logits，再分别放大 32/16/8/4 倍到 352x352。
#
# 注意“编码器选择”和“是否使用预训练权重”是两个不同问题：
# encoder 决定网络结构是 PVTv2 还是 ResNet；
# pretrain 只决定该结构是否加载已有 ImageNet 参数。
# 即使 pretrain=False，编码器仍然存在，只是从本文件/对应骨干定义的随机初始化权重开始训练。
#
# 本模型内部不做 sigmoid、softmax、argmax 或阈值化。返回值是 logits：
#   二分类通常在损失或推理代码中对单通道 logits 使用 sigmoid；
#   多分类通常在交叉熵内部或推理代码中沿类别维使用 softmax/argmax。
# 这样设计可以直接配合数值更稳定的 BCEWithLogitsLoss/CrossEntropyLoss，并允许训练器自由组合四个监督头。
# ====================================================================


# 模型总装入口：输入适配 -> 层级编码器 -> EMCAD 解码器 -> 四个分割头。
# 论文对应：第4页 Fig.2(a)-(b)，整体架构说明见第5页 Sec.3.2。
class EMCADNet(nn.Module):
    # ------------------------------ 构造参数怎么读 ------------------------------
    # num_classes：每个像素要输出的类别 logit 数，也是四个 out_head 的输出通道数。
    # kernel_sizes：MSDC 中并行深度卷积分支的卷积核尺寸，默认 1/3/5 用来覆盖不同局部感受野。
    # expansion_factor：MSCB 内部先扩张通道的倍率；更大通常增加表达能力，也增加参数量和计算量。
    # dw_parallel：控制多个深度卷积分支是都看同一输入，还是后一分支继续处理前一分支结果。
    # add：控制多尺度分支结果按元素相加还是沿通道拼接；这会影响后续投影层看到的通道数。
    # lgag_ks：LGAG 注意力门中分组卷积的核尺寸，决定门控计算时观察多大的邻域。
    # activation：传给解码器各模块的激活函数名称；具体支持范围由 decoders.py 的 act_layer 决定。
    # encoder：选择骨干结构。所有可用骨干都必须返回由浅到深的四个特征图，才能接入同一个 EMCAD。
    # pretrain：是否加载骨干预训练权重；它不代表恢复完整分割训练 checkpoint。
    # pretrained_dir：仅供 PVTv2 查找本地 ImageNet 权重；ResNet 使用其文件中 model_zoo 的 URL/cache 逻辑。
    # --------------------------------------------------------------------------
    # num_classes 决定每个输出头的通道；其余参数控制 EMCAD 消融配置和编码器选择。
    def __init__(self, num_classes=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3,
                 activation='relu', encoder='pvt_v2_b2', pretrain=True, pretrained_dir='./pretrained_pth/pvt/'):
        # 初始化 nn.Module，使后续赋值的子模块和参数被 PyTorch 正确注册。
        super(EMCADNet, self).__init__()
        # conv block to convert single channel to 3 channels
        # 医学 CT/灰度图通常为 1 通道，而 ImageNet 编码器期望 3 通道，因此建立可学习的 1->3 适配器。
        # 这里选择“可学习的 1x1 卷积”而不是简单复制三份灰度图，意味着网络可以为三个输出通道学习不同缩放与组合。
        # 该模块虽然总会在 __init__ 中创建并计入整网参数，但 forward 只有看到恰好 1 个输入通道时才会调用它。
        self.conv = nn.Sequential(
            # 1x1 卷积只混合通道，不改变 H、W；输出形状由 (B,1,H,W) 变为 (B,3,H,W)。
            nn.Conv2d(1, 3, kernel_size=1),
            # 对新生成的 3 个通道做批归一化。
            nn.BatchNorm2d(3),
            # ReLU 为输入适配器加入非线性。
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        # 下面每个分支同时确定两件事：self.backbone 是谁，以及 EMCAD 构造时需要的逆序通道列表 channels。
        # 对 PVTv2 还会准备本地权重 path；对 ResNet 不需要 path，因为其工厂函数已经接收并处理 pretrain。
        # channels 必须和骨干真实输出严格匹配，否则 LGAG、EUCB 或 MSCB 的卷积会在运行时出现通道不一致错误。
        # PVTv2-B0 是论文中的 tiny encoder，四级正向通道为 [32,64,160,256]。
        if encoder == 'pvt_v2_b0':
            # 创建 B0 编码器；forward 返回 [x1,x2,x3,x4]。
            self.backbone = pvt_v2_b0()
            # 记录本地预训练权重路径，只有 pretrain=True 时才实际读取。
            path = pretrained_dir + '/pvt_v2_b0.pth'
            # 解码器要求按“最深到最浅”排列，所以反转为 [256,160,64,32]。
            channels = [256, 160, 64, 32]
        # PVTv2-B1 的各级通道与 B2 相同，主要区别是 Transformer block 深度。
        elif encoder == 'pvt_v2_b1':
            # 创建 B1 编码器。
            self.backbone = pvt_v2_b1()
            # B1 本地权重文件。
            path = pretrained_dir + '/pvt_v2_b1.pth'
            # B1 正向通道 [64,128,320,512] 的逆序解码配置。
            channels = [512, 320, 128, 64]
        # PVTv2-B2 是论文标准模型和本项目默认编码器。
        elif encoder == 'pvt_v2_b2':
            # 创建 B2；默认 224 输入得到 x1..x4 空间尺寸 56、28、14、7。
            self.backbone = pvt_v2_b2()
            # B2 本地 ImageNet 预训练权重文件。
            path = pretrained_dir + '/pvt_v2_b2.pth'
            # d4->d1 的解码通道流为 512 -> 320 -> 128 -> 64。
            channels = [512, 320, 128, 64]
        # B3 保持同一通道接口，但第三阶段 block 数更多。
        elif encoder == 'pvt_v2_b3':
            # 创建 B3 编码器。
            self.backbone = pvt_v2_b3()
            # B3 本地权重路径。
            path = pretrained_dir + '/pvt_v2_b3.pth'
            # 仍使用 [512,320,128,64] 解码接口。
            channels = [512, 320, 128, 64]
        # B4 继续增加网络深度，四级通道不变。
        elif encoder == 'pvt_v2_b4':
            # 创建 B4 编码器。
            self.backbone = pvt_v2_b4()
            # B4 本地权重路径。
            path = pretrained_dir + '/pvt_v2_b4.pth'
            # B4 解码通道配置。
            channels = [512, 320, 128, 64]
        # B5 是此文件中最深的 PVTv2 变体，仍满足同一四级接口。
        elif encoder == 'pvt_v2_b5':
            # 创建 B5 编码器。
            self.backbone = pvt_v2_b5()
            # B5 本地权重路径。
            path = pretrained_dir + '/pvt_v2_b5.pth'
            # B5 解码通道配置。
            channels = [512, 320, 128, 64]
        # ResNet18 是工程扩展编码器；论文只说明 EMCAD 可接任意层级视觉骨干。
        elif encoder == 'resnet18':
            # pretrain 参数直接传给 ResNet 工厂，True 时由 model_zoo 下载/读取 ImageNet 权重。
            self.backbone = resnet18(pretrained=pretrain)
            # ResNet18 正向通道 [64,128,256,512]，此处按深到浅反排。
            channels = [512, 256, 128, 64]
        # ResNet34 与 ResNet18 通道相同，只增加 BasicBlock 数量。
        elif encoder == 'resnet34':
            # 创建 ResNet34。
            self.backbone = resnet34(pretrained=pretrain)
            # ResNet34 解码通道配置。
            channels = [512, 256, 128, 64]
        # ResNet50 使用 expansion=4 的 Bottleneck，四级通道显著增大。
        elif encoder == 'resnet50':
            # 创建 ResNet50。
            self.backbone = resnet50(pretrained=pretrain)
            # 正向通道 [256,512,1024,2048] 的逆序配置。
            channels = [2048, 1024, 512, 256]
        # ResNet101 与 ResNet50 的通道接口相同，block 深度不同。
        elif encoder == 'resnet101':
            # 创建 ResNet101。
            self.backbone = resnet101(pretrained=pretrain)
            # ResNet101 解码通道配置。
            channels = [2048, 1024, 512, 256]
        # ResNet152 是最深的 ResNet 工程选项。
        elif encoder == 'resnet152':
            # 创建 ResNet152。
            self.backbone = resnet152(pretrained=pretrain)
            # ResNet152 解码通道配置。
            channels = [2048, 1024, 512, 256]
        # 未识别的名称不会终止程序，而是回退到 PVTv2-B2。
        else:
            # 控制台明确提示调用者实际采用了默认编码器。
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
            # 工程注意：这里只替换 backbone，没有改写原始 encoder 字符串。
            self.backbone = pvt_v2_b2()  # 创建编码器结构
            # 保存回退模型的权重路径。
            path = pretrained_dir + '/pvt_v2_b2.pth'  # 编码器预训练权重
            # 回退模型使用 B2 的解码通道。
            channels = [512, 320, 128, 64]

        # PVT 权重由本地文件加载；ResNet 已在各自工厂函数内部处理 pretrain。
        # Python 的 and 会短路求值：当 pretrain=False 时，不会继续判断字符串；当编码器是 ResNet 时也不会进入本地加载块。
        # 此判断使用的是调用者传入的原始 encoder 字符串，而不是实际 self.backbone 的类型。
        # 因此未知名称回退到 B2 时，若名称本身不含“pvt_v2”，即使 pretrain=True 也不会加载刚才准备的 B2 权重。
        if pretrain == True and 'pvt_v2' in encoder:
            # 这里预期 .pth 文件顶层就是“参数名 -> Tensor”的 state_dict；若 checkpoint 外层还有 model/state_dict 键，需由外部先解包。
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

        # numel() 返回单个参数张量含有多少个标量；对 parameters() 求和得到参数数量，而不是显存字节数或 FLOPs。
        # 这条输出用于确认所选骨干规模，不能直接代表训练速度；特征图尺寸、算子类型和硬件同样影响耗时。
        # 打印编码器参数总数；ResNet 的未使用分类头参数也包含在该统计中。    # m.numel() 对所有编码器 Parameter 的元素数求和。
        print('Model %s created, param count: %d' % (
            encoder + ' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))

        # decoder initialization。用所选骨干的四级逆序通道构造 EMCAD，并透传所有消融参数。
        # 论文默认配置是 kernel_sizes=[1,3,5]、expansion_factor=2、并行深度卷积、加法聚合和 ReLU6。
        # 需要区分论文描述与当前调用默认值：本构造函数的 activation 默认实参是 'relu'，实际建层始终以传入字符串为准。
        # 这里只“创建”解码器各层，尚未流过任何图像；真正的张量计算发生在 forward 的 self.decoder(...) 调用中。
        self.decoder = EMCAD(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                             dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation)

        # 打印仅 EMCAD 解码器的参数量，便于核对轻量化设计。解码器参数统计不包含编码器和下面的四个 segmentation head。
        print('Model %s created, param count: %d' % ('EMCAD decoder: ',
                                                     sum([m.numel() for m in self.decoder.parameters()])))

        # 四个输出头互不共享参数，各自学习怎样把当前尺度的解码特征变成 K 个类别的像素证据。
        # kernel_size=1 表示每个空间位置只做通道线性组合，因此不会改变该特征图的 H/W。
        # 训练时若四个预测都参与损失，这就是深监督：较深层也直接收到分割目标带来的梯度，而不只依赖最后一层反传。
        # 论文第5页 Sec.3.1.4、式(10)：最深层 d4 使用 1x1 卷积把 channels[0] 投影到类别数 K。
        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        # d3 输出头：channels[1] -> K。
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        # d2 输出头：channels[2] -> K。
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        # 最高分辨率 d1 输出头：channels[3] -> K。
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)

    # ------------------------------ forward 契约 ------------------------------
    # 输入：四维 PyTorch 图像张量 x=[B,C,H,W]。代码专门适配 C=1，骨干原生适配 C=3。
    # 若传入 C=2、C=4 等其他通道数，本方法不会自动转换，随后骨干第一层通常会因期望 3 通道而报错。
    # 输出：长度为 4 的 Python 列表，每个元素都是 [B,K,H,W] 形状的 logits；列表末尾 p1 分辨率恢复路径最短。
    # mode：当前只造成一次条件分支，但两个分支返回内容完全一致；它不会自动调用 eval()，也不控制梯度开关。
    # 调用 model.eval()、torch.no_grad() 以及选择最终预测头，仍由测试脚本负责。
    # -------------------------------------------------------------------------
    # 输入 x 约定为 (B,C,H,W)；mode 当前不改变返回值，训练策略由外部 trainer 决定。
    def forward(self, x, mode='test'):
        # 只在 C=1 时使用可学习适配器；已经是 RGB/C=3 时直接送入编码器。
        if x.size()[1] == 1:
            # (B,1,H,W) -> (B,3,H,W)，空间尺寸保持不变。
            x = self.conv(x)

        # encoder
        # 所有受支持编码器都履行相同契约：返回从浅到深的四级特征。
        # 默认 B2：x1=(B,64,H/4,W/4)、x2=(B,128,H/8,W/8)、x3=(B,320,H/16,W/16)、x4=(B,512,H/32,W/32)。
        # 浅层 x1 保留较多边缘/纹理和定位信息，深层 x4 感受野更大、语义更强但空间细节更少；解码器需要两者互补。
        x1, x2, x3, x4 = self.backbone(x)
        # print(x1.shape, x2.shape, x3.shape, x4.shape)

        # decoder
        # x4 进入上采样主路，skip 按深到浅排列为 [x3,x2,x1]。
        # 返回 dec_outs=[d4,d3,d2,d1]，默认通道依次为 [512,320,128,64]。
        # 列表顺序不能改成 [x1,x2,x3]：EMCAD 第一次上采样后空间是 H/16，只能先与同为 H/16 的 x3 对齐融合。
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

        # 这里使用 scale_factor 而非显式 size=x.shape[-2:]，所以代码假设输入尺寸经过四次下采样后能被这些倍率精确还原。
        # 352 可被 32 整除，四个输出都会严格回到 352x352；若 H/W 不是 32 的合适倍数，卷积取整可能使放大后尺寸偏离原图。
        # mode='bilinear' 对每个类别通道独立做二维连续插值，只改变空间采样，不会在类别通道之间混合数值。
        # 未显式传 align_corners，沿用当前 PyTorch 默认行为；训练和推理必须运行同一段代码以保持插值约定一致。
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
            # return 返回的是列表本身，不会复制四个张量；张量仍保留其 autograd 关系，除非外层使用 no_grad 或 detach。
            # 测试模式仍返回全部四个 logits，不在模型内部执行 sigmoid、softmax 或多头求和。
            return [p4, p3, p2, p1]

        # 非 test 模式返回完全相同的列表；mutation/deep supervision/last-layer 由训练器选择。
        return [p4, p3, p2, p1]


# 直接运行本文件时执行一个 GPU 形状检查；被训练脚本 import 时不会进入该分支。
if __name__ == '__main__':
    # 这只是开发者自检入口，不是正式训练入口：它没有数据集、损失、优化器、反向传播或 checkpoint 保存逻辑。
    # 代码显式调用 .cuda()，因此机器没有可用 CUDA 时会失败；默认 pretrain=True 还要求 B2 本地权重路径有效。
    # 使用默认 PVTv2-B2、1类输出并移动到 CUDA。
    model = EMCADNet().cuda()
    # 构造 352x352 的 3 通道随机输入；因为 C=3，不经过 1->3 适配器。
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    # 执行一次完整前向，P 是四个同尺寸预测组成的列表。
    P = model(input_tensor)
    # 打印四个输出形状，预期每个都是 (1,1,352,352)。
    print(P[0].size(), P[1].size(), P[2].size(), P[3].size())
