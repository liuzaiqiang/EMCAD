# EMCAD 候选创新点 1：CGCF 互补跳连融合

- 研究基线：EMCAD（CVPR 2024）
- 候选模块：Correlation-Gated Complementary Fusion，简称 CGCF
- 模块类别：单一网络结构创新
- 文档日期：2026-09-06
- 代码原则：本文档只提供手工复制步骤；本次没有修改任何项目源代码。
- 实验目标：先完成 EMCAD 与 EMCAD + CGCF 的公平、独立对比。不要与此前的 DGEMCAD、动态多尺度路由、蒸馏、辅助损失或后处理同时启用。

## 0. 先看结论

DGEMCAD 没有提高指标，应当作为有效的负实验记录保留，但不能继续与本方案混合。本方案是一个新的、单独可验证的候选结构：在原 EMCAD 的三处 LGAG 后跳连相加位置，引入“相关性门控的互补分量”而不是直接相加。

它不是一个已被证实一定有效的模块，也不能宣称“首次使用正交或去冗余思想”。截至本文档日期的定向检索，没有发现同时满足以下完整定义的高度相同公开论文或公开代码：

1. 位于 EMCAD 的 LGAG 后；
2. 按像素、沿通道维计算 decoder 与 gated skip 的余弦相关性；
3. 只扣除 skip 中与 decoder 平行的分量；
4. 通过相关性门控决定扣除强度；
5. 同一个小模块在三个 decoder 尺度共享两个标量参数。

因此它可以作为小论文候选进行严格验证，但发表价值取决于完整实验结果，而不是模块名称。

## A. 是否已有相同或高度相似论文

### A.1 检索结论

结论：**未发现高度相同的公开实现；发现了若干概念相近但机制不同的工作。**

本次检索使用了 orthogonal skip connection segmentation、orthogonal feature fusion medical image segmentation、feature orthogonalization skip fusion semantic segmentation、correlation-gated feature fusion medical image segmentation 等关键词，并检查公开论文索引、Crossref、arXiv API 与论文主页。没有检索到与本 CGCF 完全同构的论文条目。

这不是“全世界绝对不存在”的证明。正式投稿前必须再用 Google Scholar、IEEE Xplore、PubMed、arXiv、Papers with Code 复核同一组关键词和 CGCF 的全称；若届时发现高度同构工作，应停止把 CGCF 当作独立创新点，并重新设计。

| 公开工作 | 相似点 | 与 CGCF 的关键不同 |
| --- | --- | --- |
| [EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html), CVPR 2024 | 都在 decoder 中处理多尺度特征和 gated skip | 原 EMCAD 的 LGAG 后采用直接逐元素相加；没有按 decoder/skip 的余弦相关性分解并抑制平行 skip 分量。它是本研究的 baseline。 |
| [Medical Image Segmentation via Cascaded Attention Decoding](https://openaccess.thecvf.com/content/WACV2023/html/Rahman_Medical_Image_Segmentation_via_Cascaded_Attention_Decoding_WACV_2023_paper.html), WACV 2023 | 都对 encoder skip 做 attention gate，再与 decoder 融合 | 关注 cascaded attention decoding；没有逐像素通道投影、冗余门控和互补残差。 |
| [Searching Enhanced Decoder with Switchable Skip Connection for Semantic Segmentation](https://doi.org/10.1016/j.patcog.2023.110196), Pattern Recognition 2024 | 都认为 skip connection 的融合方式会影响 decoder 质量 | 它在候选 skip 连接或开关中搜索；CGCF 不搜索连接，也不改网络拓扑，只在同尺度两张特征图之间计算连续的冗余抑制。 |
| [Gated Skip-Connection Network with Adaptive Upsampling for Retinal Vessel Segmentation](https://doi.org/10.3390/s21186177), Sensors 2021 | 都使用 gated skip | 该工作使用门控和上采样；没有利用 decoder 与 gated skip 的通道余弦相关性来保留互补信息。 |
| [FEAST-Mamba: FEAture and SpaTial Aware Mamba Network with Bidirectional Orthogonal Fusion for Cross-Modal Point Cloud Segmentation](https://doi.org/10.1609/aaai.v39i5.32489), AAAI 2025 | 都出现“orthogonal fusion”这一宽泛概念 | 任务是跨模态点云分割，双向特征和空间融合机制不同；不能把 CGCF 宣称为首次提出 orthogonal fusion。 |

论文的新颖性表述必须限定为：

> 提出一种面向 EMCAD 解码器中 gated skip 冗余问题的相关性门控互补融合机制。

不要写“首次提出正交融合”“首次解决 skip redundancy”等全局性表述。

## B. 创新点是否合理

### B.1 EMCAD 中要解决的具体问题

当前解码器的关键路径是：

~~~text
d3 = EUCB3(d4)
x3 = LGAG3(g=d3, x=skip3)
d3 = d3 + x3

d2 = EUCB2(d3)
x2 = LGAG2(g=d2, x=skip2)
d2 = d2 + x2

d1 = EUCB1(d2)
x1 = LGAG1(g=d1, x=skip1)
d1 = d1 + x1
~~~

LGAG 已经在空间位置上选择 skip，但它返回的仍是完整的 gated skip 特征。若某个位置的 skip 与 decoder 特征在通道方向高度一致，直接相加会重复累积同类响应；LGAG 不能区分“有用的新细节”与“已经在 decoder 中存在的重复语义”。

CGCF 的目标不是删除 skip，而是仅在冗余高的位置，从 skip 中扣除与 decoder 同方向的分量，保留更互补的成分。它特别适合被解释为：浅层边界和纹理信息需要被保留，但不应重复叠加已经很强的解码语义。

### B.2 单模块机制

设同尺度 decoder 特征为 d，LGAG 输出的 skip 特征为 x，二者都是 [B, C, H, W]。

1. 在每个空间位置沿通道维归一化，计算余弦相关性 c：
   c = sum_c(normalize(d) * normalize(x))，形状为 [B, 1, H, W]。
2. 通过一个可学习温度和阈值生成冗余权重 r：
   r = sigmoid(temperature * (c - threshold))。
3. 将 x 投影到 d 的通道方向，得到平行分量 p：
   p = sum_c(x * normalize(d)) * normalize(d)。
4. 仅按 r 扣除该平行分量：
   x_complementary = x - r * p。
5. 保持 EMCAD 的残差融合形式：
   output = d + x_complementary。

当 r 接近 0，CGCF 等价于原来的 d + x；当 r 接近 1，主要抑制 skip 中与 decoder 平行的冗余部分。该设计不改变编码器、EUCB、LGAG、CAB、SAB、MSCB 和四个预测头。

### B.3 合理性、参数和计算影响

- 训练目标：不改变。仍使用项目现有的四输出监督和原有损失。
- 新增参数：只有两个全局共享的标量，log_temperature 与 threshold，总计 2 个参数。
- 通道数：不改变。没有 concat、1x1 投影或额外卷积。
- 空间尺寸：不改变。输入输出始终同为 [B, C, H, W]。
- 计算量：增加逐元素乘法、按通道求和、rsqrt 和 sigmoid，理论上是 O(BCHW)，远小于卷积；实际 FLOPs、推理时间和峰值显存仍必须实测。
- 推理显存：推理阶段只会多出少量中间张量；训练阶段还需保存它们反向传播，因此不能口头承诺“零显存增长”。
- 结构创新判断：这是“针对 EMCAD 已有 gated skip 融合点的单一结构替换”，不是损失函数、训练策略或后处理。创新强度是增量式的，不是大模型级方法创新。

## C. 它到底是一个还是多个创新点

它是**一个创新点**：CGCF 跳连融合。

虽然内部有“相关性计算、投影分解、门控扣除”三个运算步骤，但它们共同构成一个不可拆分的融合规则，且共享同一对标量参数。三处 d3、d2、d1 调用的是同一个模块实例和同一组参数，不是三个创新点。

本轮不得加入以下任何内容：

- 新损失或辅助损失；
- EMA 教师、蒸馏或边界蒸馏；
- 动态多尺度路由、DGEMCAD 或 adaptive_msdc；
- 改数据增广、采样、后处理、输入尺寸、backbone；
- 新的边界分支或孔洞修复模块。

否则无法把最终提升归因于 CGCF。

## D. 与 EMCAD 的具体差异

### D.1 公式差异

原 EMCAD：

~~~text
f_original = d + x
~~~

EMCAD + CGCF：

~~~text
d_hat = d / sqrt(sum_c(d*d) + eps)
c = sum_c(d_hat * x_hat)
r = sigmoid(exp(log_temperature) * (c - threshold))
p = sum_c(x * d_hat) * d_hat
f_cgcf = d + (x - r * p)
~~~

### D.2 精确插入位置和张量对齐

以当前项目 PVTv2-B2 和 224 x 224 输入为例：

| 融合位置 | 原 decoder 特征 | LGAG 输出 | CGCF 输出 |
| --- | --- | --- | --- |
| d3 融合 | [B, 320, 14, 14] | [B, 320, 14, 14] | [B, 320, 14, 14] |
| d2 融合 | [B, 128, 28, 28] | [B, 128, 28, 28] | [B, 128, 28, 28] |
| d1 融合 | [B, 64, 56, 56] | [B, 64, 56, 56] | [B, 64, 56, 56] |

因此：

- LGAG 的输出可直接送入 CGCF；
- CGCF 不会改变三处 MSCBLayer 的输入通道；
- EMCAD.forward() 仍返回 [d4, d3, d2, d1]；
- EMCADNet.forward() 仍返回四个 [B, num_classes, 224, 224] 的 logits；
- 原训练代码和原损失的返回值契约保持兼容。

本方案会在运行时显式检查 decoder_feature.shape 与 gated_skip.shape，防止误传 skip 顺序或通道数时静默出错。

## E. 实现位置和实现步骤

以下是手工复制方案。请先备份四个文件；只修改本文列出的锚点。不要复制到此前的 dg_emcad.py 或 dg_losses.py。

### E.1 文件一：lib/decoders.py

#### 第一步：新增 CGCF 类

**插入位置：** 找到文本 “class EMCAD(nn.Module):”，将下列完整类粘贴在它的正上方。

~~~python
class CorrelationGatedComplementaryFusion(nn.Module):
    """
    CGCF: Correlation-Gated Complementary Fusion.

    输入:
        decoder_feature: 当前上采样后的解码特征 d，形状 [B, C, H, W]
        gated_skip:      LGAG 筛选后的跳连特征 x，形状 [B, C, H, W]

    输出:
        与输入同形状的融合结果。模块只改变融合数值，不改变通道数和空间尺寸。
    """
    def __init__(self, init_temperature=4.0, init_threshold=0.5, eps=1e-6):
        super().__init__()

        # 温度必须为正，所以优化 log_temperature，再通过 exp() 还原。
        # 初值 exp(log(4.0)) = 4.0；该标量在 d3/d2/d1 三处共享。
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(float(init_temperature)))
        )

        # 余弦相关性阈值。初值 0.5 表示仅在较高正相关时强力去冗余。
        self.threshold = nn.Parameter(torch.tensor(float(init_threshold)))

        # 防止零向量归一化时除零。
        self.eps = eps

    def forward(self, decoder_feature, gated_skip):
        # CGCF 只能融合已经通道和空间尺寸对齐的同尺度特征。
        # 若 skip 顺序、backbone channels 或输入尺寸错误，立即报出可读错误。
        if decoder_feature.shape != gated_skip.shape:
            raise RuntimeError(
                "CGCF requires decoder_feature and gated_skip to have the same "
                "[B, C, H, W] shape, but got {} and {}.".format(
                    tuple(decoder_feature.shape), tuple(gated_skip.shape)
                )
            )

        # 沿通道维 L2 归一化。每个像素位置都会得到一个单位通道方向。
        decoder_norm = decoder_feature * torch.rsqrt(
            torch.sum(decoder_feature * decoder_feature, dim=1, keepdim=True)
            + self.eps
        )
        skip_norm = gated_skip * torch.rsqrt(
            torch.sum(gated_skip * gated_skip, dim=1, keepdim=True)
            + self.eps
        )

        # 每个像素位置的通道余弦相关性，输出形状 [B, 1, H, W]。
        cosine = torch.sum(decoder_norm * skip_norm, dim=1, keepdim=True)

        # 限制温度上界，避免训练把 sigmoid 推入数值过硬的饱和状态。
        temperature = torch.exp(self.log_temperature).clamp(max=8.0)

        # 相关性高于阈值时 redundancy 接近 1，说明 skip 中平行分量更可能重复。
        redundancy = torch.sigmoid(temperature * (cosine - self.threshold))

        # 将原始 gated_skip 投影到 decoder 的单位通道方向上。
        # 该张量仍是 [B, C, H, W]，不是降维后的单通道图。
        parallel_skip = (
            torch.sum(gated_skip * decoder_norm, dim=1, keepdim=True)
            * decoder_norm
        )

        # 仅按冗余权重扣除平行部分，保留更互补的 skip 细节。
        complementary_skip = gated_skip - redundancy * parallel_skip

        # 保持 EMCAD 原始的“decoder 主路 + skip 残差”接口。
        return decoder_feature + complementary_skip
~~~

不需要新增 import。当前文件已经导入了 torch 和 torch.nn as nn。

#### 第二步：给 EMCAD 构造函数加开关

**替换位置：** 找到当前函数头：

~~~python
def __init__(self, channels=[512, 320, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True,
             add=True, lgag_ks=3, activation='relu6'):
~~~

**替换为：**

~~~python
def __init__(self, channels=[512, 320, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True,
             add=True, lgag_ks=3, activation='relu6', use_cgcf=False):
~~~

**新增位置：** 找到 EMCAD 构造函数末尾已有的一行：

~~~python
self.sab = SAB()
~~~

紧接着粘贴：

~~~python
# 默认 False 时严格保留原 EMCAD 的 d + x 融合行为。
self.use_cgcf = use_cgcf

# 一个 CGCF 实例在 d3、d2、d1 三个尺度共享，避免把同一机制错误包装成三个模块。
if self.use_cgcf:
    self.cgcf = CorrelationGatedComplementaryFusion()
~~~

#### 第三步：替换三处实际跳连相加

在 EMCAD.forward() 中逐处替换。只替换活动代码行，不要改上方说明注释。

~~~python
# 原代码
d3 = d3 + x3

# 替换为
d3 = self.cgcf(d3, x3) if self.use_cgcf else d3 + x3
~~~

~~~python
# 原代码
d2 = d2 + x2

# 替换为
d2 = self.cgcf(d2, x2) if self.use_cgcf else d2 + x2
~~~

~~~python
# 原代码
d1 = d1 + x1

# 替换为
d1 = self.cgcf(d1, x1) if self.use_cgcf else d1 + x1
~~~

这三行是 CGCF 唯一进入网络计算图的位置。不开 --cgcf 时，它们会逐字等价回退为原 EMCAD 的加法。

### E.2 文件二：lib/networks.py

#### 第一步：给 EMCADNet 构造函数加开关

**替换位置：** 找到函数头末尾：

~~~python
activation='relu', encoder='pvt_v2_b2', pretrain=True, pretrained_dir='./pretrained_pth/pvt/'):
~~~

**替换为：**

~~~python
activation='relu', encoder='pvt_v2_b2', pretrain=True, pretrained_dir='./pretrained_pth/pvt/',
use_cgcf=False):
~~~

#### 第二步：将开关传入解码器

**替换位置：** 找到创建 decoder 的两行：

~~~python
self.decoder = EMCAD(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                     dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation)
~~~

**替换为：**

~~~python
# use_cgcf 只影响 LGAG 后三处 skip 融合，不改变 encoder、MSCB 或输出头。
self.decoder = EMCAD(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                     dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks,
                     activation=activation, use_cgcf=use_cgcf)

# 这是运行日志中的第一条模型身份核验信息。
print("CGCF skip fusion: {}".format("enabled" if use_cgcf else "disabled"))
~~~

### E.3 文件三：train_synapse.py

#### 第一步：新增命令行参数

**插入位置：** 找到：

~~~python
parser.add_argument('--concatenation', action='store_true', default=False,
                    help='use this flag to concatenate feature maps in MSDC block')
~~~

紧接着粘贴：

~~~python
parser.add_argument(
    '--cgcf',
    action='store_true',
    default=False,
    help='enable Correlation-Gated Complementary Fusion after each LGAG'
)
~~~

#### 第二步：将参数传入 EMCADNet

**替换位置：** 找到模型构造末尾：

~~~python
encoder=args.encoder, pretrain=not args.no_pretrain, pretrained_dir=args.pretrained_dir)
~~~

**替换为：**

~~~python
encoder=args.encoder, pretrain=not args.no_pretrain,
pretrained_dir=args.pretrained_dir, use_cgcf=args.cgcf)
~~~

#### 第三步：隔离创新模型 checkpoint

当前活动代码有：

~~~python
exp_name = f"run_seed{args.seed}"
snapshot_path = os.path.join("model_pth", exp_name)
~~~

**替换为：**

~~~python
# 防止 CGCF 的 best.pth 覆盖同 seed 的原 EMCAD 基线权重。
# 不开 --cgcf 时目录名称与当前原基线保持完全一致。
exp_name = f"run_seed{args.seed}_cgcf" if args.cgcf else f"run_seed{args.seed}"
snapshot_path = os.path.join("model_pth", exp_name)
~~~

不要修改上方被三引号包围的旧目录命名示例；只有后面实际执行的简化路径生效。

### E.4 文件四：test_synapse.py

训练和测试必须以完全同样的开关重建网络和 checkpoint 路径，否则会发生以下两种问题：

- 模型结构不含 CGCF，却试图加载含 CGCF 的权重；
- 加载到了原 EMCAD 的 best.pth，却误以为测试的是 CGCF。

#### 第一步：新增同名命令行参数

在 --concatenation 参数定义后，粘贴与训练脚本完全相同的代码：

~~~python
parser.add_argument(
    '--cgcf',
    action='store_true',
    default=False,
    help='enable Correlation-Gated Complementary Fusion after each LGAG'
)
~~~

#### 第二步：将参数传入 EMCADNet

找到测试脚本中模型构造末尾：

~~~python
encoder=args.encoder, pretrain=not args.no_pretrain, pretrained_dir=args.pretrained_dir)
~~~

替换为：

~~~python
encoder=args.encoder, pretrain=not args.no_pretrain,
pretrained_dir=args.pretrained_dir, use_cgcf=args.cgcf)
~~~

#### 第三步：同步 checkpoint 路径

找到活动代码：

~~~python
snapshot_path = os.path.join("model_pth", f"run_seed{args.seed}")
~~~

替换为：

~~~python
# 必须与 train_synapse.py 的命名规则逐字一致。
exp_name = f"run_seed{args.seed}_cgcf" if args.cgcf else f"run_seed{args.seed}"
snapshot_path = os.path.join("model_pth", exp_name)
~~~

### E.5 运行前静态检查

在 Linux 服务器的项目根目录执行：

~~~bash
python -m py_compile lib/decoders.py lib/networks.py train_synapse.py test_synapse.py
~~~

没有输出且退出码为 0，才表示 Python 语法通过。它不代表模型的通道、尺寸和 forward 契约已经通过。

### E.6 前向、通道和返回值检查

在服务器根目录运行。这里的 pretrain=False 只避免形状检查时加载预训练权重；正式训练必须恢复为与 baseline 相同的预训练设置。

~~~bash
python - <<'PY'
import torch
from lib.networks import EMCADNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = EMCADNet(
    num_classes=9,
    kernel_sizes=[1, 3, 5],
    expansion_factor=2,
    dw_parallel=True,
    add=True,
    lgag_ks=3,
    activation='relu6',
    encoder='pvt_v2_b2',
    pretrain=False,
    use_cgcf=True,
).to(device).eval()

assert hasattr(model.decoder, 'cgcf'), 'CGCF was not created'
assert sum(p.numel() for p in model.decoder.cgcf.parameters()) == 2

with torch.no_grad():
    outputs = model(torch.randn(1, 1, 224, 224, device=device))

shapes = [tuple(output.shape) for output in outputs]
print('output shapes:', shapes)
assert shapes == [(1, 9, 224, 224)] * 4, shapes
print('CGCF forward contract check: PASS')
PY
~~~

预期日志必须包含：

~~~text
CGCF skip fusion: enabled
output shapes: [(1, 9, 224, 224), (1, 9, 224, 224), (1, 9, 224, 224), (1, 9, 224, 224)]
CGCF forward contract check: PASS
~~~

若出现 CGCF shape error，不要通过 interpolate、裁剪或补零掩盖问题。先检查输入尺寸是否能被 32 整除、PVT 输出通道是否匹配、以及 skips 是否仍按 [x3, x2, x1] 传入。

### E.7 启动命令与“确实跑到 CGCF”的确认

本地工作区没有服务器上的实际 .sh 文件，因此不要凭空替换你的整段启动脚本。保留基线脚本的全部原有参数，只在最终的 python train_synapse.py 参数列表中增加一行：

~~~bash
--cgcf \
~~~

CGCF 实验禁止同时出现：

~~~text
--adaptive_msdc
--dg_router_mode
--dg_route_aux_weight
--dg_route_reg_weight
--bp_distill
~~~

对应 baseline 实验使用同一份脚本、同一切分、同一超参数，但**不加** --cgcf。

训练日志必须同时满足以下三项，才可确认它是 EMCAD + CGCF：

1. Namespace(...) 中出现 cgcf=True；
2. 初始化日志出现 CGCF skip fusion: enabled；
3. 保存位置为 model_pth/run_seed<你的seed>_cgcf/best.pth。

测试时也要在现有测试命令中增加 --cgcf，并确认打印的 snapshot 路径指向上述 _cgcf/best.pth。日志仍出现 Model EMCAD decoder: created 是正常的，因为改进模型仍然以 EMCAD 为基础，而不是一个脱离 EMCAD 的新网络。

## F. 实验和消融设计

### F.1 公平主对比

先只比较下表的 A 与 C：

| 实验 | 模型 | 唯一差异 | 目的 |
| --- | --- | --- | --- |
| A | 原 EMCAD | 不出现 --cgcf | 基线 |
| C | EMCAD + CGCF | 仅增加 --cgcf | 验证单一结构创新 |

必须严格固定：

- 数据集版本、病人级 train/val/test 切分、list 文件；
- 输入尺寸，训练和最终测试均使用 224 x 224；
- PVTv2-B2、是否加载同一预训练权重；
- batch size、学习率、scheduler、最大 epoch、损失、mutation supervision；
- seed、PyTorch/CUDA 版本和数据加载 worker 设置；
- 对比时的 checkpoint 选择规则。

当前 Synapse 代码会在训练中反复使用 volume_path 对应的 test_vol_h5 记录并保存 best。对于**本轮内部筛选**，A 与 C 使用完全相同的现有规则，仍可作控制变量对比；但它不够严格，不能直接作为投稿时的独立测试结论。论文版必须建立独立的 patient-level validation split，用 validation 选 best，test 只在训练完成后运行一次。

### F.2 训练轮数和种子

不要在 70 轮、100 轮或某个中间 epoch 因短暂 Dice 低于 baseline 就宣布模块失败。正确顺序：

1. 用 seed=2222 完整跑到与 baseline 相同的预先设定 max_epochs，例如 300；
2. 两者都用各自验证规则下的 best.pth 进行最终测试；
3. 若 C 的结果有正向信号，再对 A/C 用至少 3 个固定种子，例如 2222、3407、2026，报告 mean +/- std；
4. 若三 seed 无提升或波动覆盖零，不把 CGCF 作为论文主创新。

只在 C 有稳定收益后，才做一个可选的内部消融 B：

| 实验 | 融合式 | 用途 |
| --- | --- | --- |
| A | d + x | 原 EMCAD |
| B | d + (x - parallel_skip) | 验证“无自适应门控的硬去平行分量” |
| C | d + (x - redundancy * parallel_skip) | 完整 CGCF |

B 不是第二个创新模块，而是完整 CGCF 中“相关性门控是否必要”的消融。A 和 C 必须先完成；不要为了填表而先做 B。

### F.3 指标记录

当前项目最终测试日志已经输出：

- Dice：mean_dice；
- IoU：日志中的 mean_jacard，即 Jaccard/IoU；
- HD95：mean_hd95；
- ASD：mean_asd。

注意：当前 metric 调用未显式传入医学图像 spacing，HD95 和 ASD 的数值默认是当前数组坐标单位；论文中不要写成 mm，除非你重构了 spacing 读取并对所有方法统一采用真实物理间距。

每次实验至少记录：

| 实验 ID | 模型 | seed | best epoch | Dice | IoU | HD95 | 参数量 | FLOPs | 推理时间 | 峰值显存 | checkpoint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_s2222 | EMCAD | 2222 |  |  |  |  |  |  |  |  |  |
| cgcf_s2222 | EMCAD+CGCF | 2222 |  |  |  |  |  |  |  |  |  |

### F.4 参数量、FLOPs、推理时间和显存

建议手工新建一个独立文件 profile_cgcf.py，不要改训练代码。先安装或确认统一版本的 profiler，例如：

~~~bash
pip install thop
~~~

然后可使用以下核心逻辑对 baseline 和 CGCF 分别运行。两者必须用同一 GPU、同一个输入 [1, 1, 224, 224]、同一 torch 版本和同一测量脚本。

~~~python
import time
import torch
from thop import profile
from lib.networks import EMCADNet

def build_model(use_cgcf):
    model = EMCADNet(
        num_classes=9,
        kernel_sizes=[1, 3, 5],
        expansion_factor=2,
        dw_parallel=True,
        add=True,
        lgag_ks=3,
        activation='relu6',
        encoder='pvt_v2_b2',
        pretrain=False,  # profile 不需要加载 ImageNet 权重
        use_cgcf=use_cgcf,
    ).cuda().eval()
    return model

@torch.inference_mode()
def benchmark(model, x, warmup=50, repeats=200):
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(x)
    torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    peak_mib = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
    return elapsed_ms, peak_mib

x = torch.randn(1, 1, 224, 224, device='cuda')

for name, enabled in [('EMCAD', False), ('EMCAD+CGCF', True)]:
    model = build_model(enabled)
    params = sum(parameter.numel() for parameter in model.parameters())

    # THOP 对逐元素和归一化算子的统计可能不完整。
    # 论文中须使用同一 profiler、同一版本和同一输入报告相对比较。
    flops, _ = profile(model, inputs=(x,), verbose=False)

    time_ms, peak_mib = benchmark(model, x)
    print(name)
    print('params:', params)
    print('flops:', flops)
    print('latency_ms_per_image:', time_ms)
    print('peak_memory_MiB:', peak_mib)
~~~

不要把 profile 的 pretrain=False 误用于正式公平训练。正式训练必须和 baseline 保持相同的预训练开关。

## G. 可能失败的原因和论文风险

### G.1 机制层面的失败可能

1. **被扣掉的平行分量也可能是正确的分割语义。** 冗余不等于无用，尤其小器官或边界区域可能正需要重复强化。
2. **全尺度共享两个标量可能过于简单。** d3、d2、d1 的语义层次不同，同一阈值不一定都合适。
3. **余弦相关性忽略幅值。** 两个方向相同但强度不同的特征会得到相近相关性，可能不是最优冗余判据。
4. **Synapse 的类别差异很大。** 大器官、小器官和边界模糊器官的收益方向可能不同；必须给出逐器官指标，不能只看均值。
5. **现有 checkpoint 选择方式可能造成测试集调参。** 即使 A/C 都同样使用它，论文层面仍有数据泄漏风险，必须后续修正验证协议。

### G.2 代码和实验层面的失败可能

- 忘记在 test_synapse.py 加 --cgcf，导致结构或权重不匹配；
- 改了训练尺寸却没同步最终测试尺寸，破坏公平性；
- CGCF checkpoint 覆盖 baseline 的 best.pth；
- 把 --cgcf 和旧 DG、蒸馏或额外损失一起启用，归因失效；
- profiler 对自定义逐元素算子统计不全。必须报告工具版本和同工具相对比较；
- 单一随机种子偶然提升或下降。正式结论至少基于三 seed 的均值和标准差。

### G.3 发表价值与停止条件

CGCF 有小论文潜力的前提是：

- 至少一个主数据集上，三 seed 的 Dice/IoU 有稳定、可重复提升；
- HD95 不恶化，或能解释 Dice 与边界指标的权衡；
- 参数量只增加 2 个，实测速度和显存变化很小；
- A/B/C 消融能表明相关性门控本身有作用；
- 最好增加一个外部数据集或不同分割任务，证明不是 Synapse 单一切分偶然现象；
- 文中清楚承认它是 EMCAD decoder skip fusion 的针对性改进，而非提出全新的通用正交理论。

停止条件也应预先定义：若完整 300 epoch 的 A/C 对比无正向信号，且三 seed 的均值不提升或提升落在随机波动内，就把 CGCF 记录为失败候选，不再通过叠加第二模块“救指标”。下一轮应换一个全新的单模块假设，而不是继续在 CGCF 上堆损失或蒸馏。

## 最终执行清单

1. 只按 E 节手工修改 lib/decoders.py、lib/networks.py、train_synapse.py、test_synapse.py；
2. 运行 py_compile 和 224 x 224 forward contract check；
3. 先用原有 baseline 参数跑 A，不加 --cgcf；
4. 再用完全相同参数跑 C，只加 --cgcf；
5. 确认三个日志证据和分离的 _cgcf checkpoint；
6. 跑完整训练后记录 Dice、IoU、HD95、参数量、FLOPs、延迟、显存；
7. 有正向信号后才做三 seed 和可选 B 消融。

本文件定义的是一个新的候选创新点，不代表其已经被实验验证有效。DGEMCAD 的负结果与 CGCF 的结果必须分开记录、分开报告。

