# 当前 EMCAD 项目基础知识点与完整拆解

> 本文不是泛泛介绍 EMCAD 论文，而是依据当前仓库实际代码进行拆解。主要分析入口是 `train_synapse.py`，实际模型链路是 `lib/networks.py -> lib/pvtv2.py / lib/resnet.py -> lib/decoders.py`，训练、数据和评估分别位于 `trainer.py`、`utils/dataset_synapse.py`、`utils/utils.py`。
>
> 当前仓库中已有用户修改，因此本文只解释现状，不修改任何模型、训练或测试代码。

## 0. 先直接回答：涉及 Transformer、RNN、Mamba 吗

| 技术 | 当前项目是否涉及 | 在哪里 | 准确结论 |
|---|---:|---|---|
| CNN / 卷积神经网络 | 是，而且很多 | `lib/decoders.py`、输入适配层、PVTv2 内部局部卷积、可选 ResNet | 即使默认编码器是 Transformer，整个 EMCADNet 仍是明显的 Transformer-CNN 混合架构 |
| Transformer | 是，默认就使用 | `lib/pvtv2.py`，默认编码器 `pvt_v2_b2` | 这是视觉 Transformer，包含 token、Q/K/V、多头自注意力、LayerNorm、残差、MLP、DropPath |
| RNN | 否 | 当前有效运行链中没有 | 没有 `RNN`、`LSTM`、`GRU`，也没有跨切片隐藏状态传递 |
| Mamba | 否 | 当前有效运行链中没有 | 没有 Mamba block、Selective Scan、SSM，也没有 `mamba_ssm` 依赖 |
| U-Net | 结构思想上类似，但不是经典 U-Net | 编码器、逐级解码器、skip connection | 它属于层次化编码器-解码器分割网络，但编码器不是经典双卷积 U-Net，skip 融合也不是原版 concat |
| Attention | 是，而且有两类 | PVTv2 的 self-attention；EMCAD 的 CAB、SAB、LGAG | PVTv2 是 token 自注意力；CAB/SAB/LGAG 是卷积式重标定或门控，不能都叫 Transformer |
| 多尺度特征 | 是，属于项目核心 | PVTv2 四阶段、MSDC、四级解码、四个预测头 | 同时存在跨分辨率多尺度和同一分辨率下的多卷积核多尺度 |
| 3D CNN / 3D Transformer | 否 | 当前 Synapse 模型没有 `Conv3d` 主链 | 模型按二维切片推理，最后把各切片重新拼成三维病例结果 |
| 深监督 | 是 | `trainer.py` | 四个输出头都可以直接参与损失，不只是最终输出头 |
| 预训练迁移学习 | 是 | PVTv2 或 ResNet 编码器 | 编码器可加载 ImageNet 预训练权重，分割解码器和输出头再针对医学数据训练 |
| Diffusion、图神经网络、SAM 提示分割 | 否 | 当前有效运行链中没有 | 不应仅因项目属于当前热门分割研究，就默认这些技术已经存在 |

一句话概括当前默认模型：

> **EMCADNet = 可学习的灰度转 RGB 输入适配器 + PVTv2-B2 金字塔视觉 Transformer 编码器 + 以多尺度深度卷积和卷积注意力为核心的 EMCAD 解码器 + 四个分割预测头。**

因此，最准确的标签不是“纯 Transformer”，也不是“纯 CNN”，而是：

> **层次化 Transformer 编码器 + 轻量卷积解码器的二维医学图像分割网络。**

---

## 1. 当前项目真正运行的主链路

以 Synapse 训练为例，当前代码主链是：

```text
train_synapse.py
    |
    | 创建 EMCADNet，默认 encoder=pvt_v2_b2
    v
lib/networks.py::EMCADNet
    |
    |-- 输入为 1 通道时：1x1 Conv + BN + ReLU，将 1 通道映射为 3 通道
    |
    |-- 编码器：lib/pvtv2.py::pvt_v2_b2
    |       返回 x1、x2、x3、x4 四级特征
    |
    |-- 解码器：lib/decoders.py::EMCAD
    |       输入 x4 和 skips=[x3,x2,x1]
    |       返回 d4、d3、d2、d1 四级解码特征
    |
    |-- 四个 1x1 Conv 分割头
    |       返回 p4、p3、p2、p1 四张全分辨率 logits
    v
trainer.py::trainer_synapse
    |
    |-- Synapse_dataset + DataLoader
    |-- CE + Dice
    |-- mutation / deep_supervision / last_layer
    |-- backward + AdamW.step
    |-- 病例级验证与 checkpoint 保存
    v
utils/utils.py
    |-- 逐切片推理
    |-- softmax + argmax
    |-- 拼回 3D 病例
    |-- Dice、HD95、Jaccard、ASSD
```

关键代码位置：

| 文件 | 作用 | 建议关注点 |
|---|---|---|
| `train_synapse.py` | Synapse 训练入口和实验参数 | 默认编码器、输入尺寸、类别数、监督方式、随机种子 |
| `test_synapse.py` | Synapse 测试入口 | checkpoint 加载、病例级测试、结果保存 |
| `lib/networks.py` | 总模型 `EMCADNet` | 编码器选择、通道契约、四个输出头、完整 forward |
| `lib/pvtv2.py` | PVTv2 编码器 | Transformer token、Q/K/V、SRA、Mix-FFN、四阶段特征 |
| `lib/resnet.py` | 可选 ResNet 编码器 | 纯 CNN 编码器和残差块 |
| `lib/decoders.py` | EMCAD 解码器 | MSDC、MSCB、EUCB、LGAG、CAB、SAB |
| `trainer.py` | Synapse 训练循环 | DataLoader、损失、反向传播、AdamW、验证、保存 |
| `utils/dataset_synapse.py` | Synapse 数据加载与增强 | NPZ/H5、旋转翻转、resize、标签插值 |
| `utils/utils.py` | Dice loss 和病例级评估 | softmax、argmax、逐切片推理、指标定义 |

### 1.1 哪些代码当前没有自动参与基线训练

`DG_EMCAD_reference/` 中存在 disagreement-guided EMCAD、熵、跨尺度 JS disagreement、EMA teacher 和蒸馏损失等参考实现。但是该目录文件自己明确说明需要审查后再复制或集成，当前 `train_synapse.py` 仍然导入：

```python
from lib.networks import EMCADNet
```

而不是 `DGEMCADNet`。所以：

- `DG_EMCAD_reference/` 是参考扩展，不是当前默认基线。
- 它不能被算作当前基线已经实现的创新点。
- 它同样没有引入 RNN 或 Mamba。
- 根目录的 `networks.py` 也不是当前主要训练入口的导入对象；当前训练脚本统一导入 `lib.networks`。

研究时必须建立一个习惯：**仓库里存在某个文件，不等于运行时真的经过它。判断依据是 import、实例化和 forward 调用链。**

---

## 2. 这个项目在解决什么问题

### 2.1 医学图像分割的本质

分类任务回答的是：

```text
这张图是什么？
```

分割任务回答的是：

```text
图中每一个像素属于什么类别？
```

对于 Synapse，模型需要给每个像素分配以下九类之一：

```text
0：背景
1~8：八个前景器官
```

因此，模型不是只输出一个“这张图有肝脏”的标签，而是输出九张像素证据图。每个空间位置有九个原始分数，最后选分数最大的类别。

### 2.2 数学上是什么

设输入切片为：

```text
X ∈ R^(B x C_in x H x W)
```

Synapse 默认：

```text
B：batch size
C_in：1，CT 灰度图
H=W=224
```

网络输出为：

```text
Z ∈ R^(B x K x H x W)
```

其中 `K=9`。`Z` 是 logits，即未经 softmax 的原始类别分数。对每个像素 `(h,w)`：

```text
P(class=k | X,h,w) = softmax(Z[:,k,h,w])
预测类别 = argmax_k P(class=k | X,h,w)
```

所以，从机器学习角度说，Synapse 分割可以理解为：

> **在共享图像上下文的前提下，同时完成 H×W 次九分类。**

它又不等于把每个像素孤立分类，因为卷积、注意力和多尺度特征让一个像素的判断能够使用周围结构以及远距离上下文。

---

## 3. 读这个项目之前必须掌握的张量基础

### 3.1 NCHW

PyTorch 图像张量常用：

```text
[B, C, H, W]
```

例如：

```text
[6, 1, 224, 224]
```

表示一个 batch 有 6 张切片，每张 1 通道，尺寸 224×224。

标签一般是：

```text
[B, H, W]
```

每个位置直接存整数类别编号，而不是 RGB 颜色。

### 3.2 通道不是“图片张数”

在网络内部，`64`、`128`、`320`、`512` 个通道不是 64 张普通图片，而是网络学到的 64、128、320、512 种特征响应。例如某些通道可能偏向边缘、纹理、器官内部、相对位置或更抽象的形状模式。

### 3.3 特征图的两种变化

网络常同时做两件事：

- 空间分辨率逐渐降低：`224 -> 56 -> 28 -> 14 -> 7`。
- 特征通道逐渐增加：`3 -> 64 -> 128 -> 320 -> 512`。

直观理解：

- 浅层特征分辨率高，擅长定位边缘和细节。
- 深层特征分辨率低，单个位置的感受野更大，语义更强。
- 解码器要把深层“它是什么”和浅层“它在哪里”重新融合。

### 3.4 reshape、permute 和广播

PVTv2 会在两种表示间切换：

```text
图像特征：[B,C,H,W]
token 序列：[B,N,C]，其中 N=H×W
```

`reshape` 改变观察张量的维度组织，`permute` 改变维度顺序。CAB 输出 `[B,C,1,1]` 后与 `[B,C,H,W]` 相乘，则依赖广播把每个通道的权重复制到全部空间位置。

如果不理解这些张量操作，就容易把“通道注意力”“空间注意力”和“Transformer token 注意力”混为一谈。

---

## 4. 默认 PVTv2-B2 下的完整形状变化

Synapse 默认输入为 `224×224`，单通道 CT 首先经过可学习适配器：

```text
输入             [B,   1,224,224]
1x1 Conv+BN+ReLU [B,   3,224,224]

PVTv2 stage 1 x1 [B,  64, 56, 56]   输入的 1/4
PVTv2 stage 2 x2 [B, 128, 28, 28]   输入的 1/8
PVTv2 stage 3 x3 [B, 320, 14, 14]   输入的 1/16
PVTv2 stage 4 x4 [B, 512,  7,  7]   输入的 1/32

EMCAD d4          [B, 512,  7,  7]
EMCAD d3          [B, 320, 14, 14]
EMCAD d2          [B, 128, 28, 28]
EMCAD d1          [B,  64, 56, 56]

out_head4         [B,   9,  7,  7]  -> 双线性放大 32 倍
out_head3         [B,   9, 14, 14]  -> 双线性放大 16 倍
out_head2         [B,   9, 28, 28]  -> 双线性放大 8 倍
out_head1         [B,   9, 56, 56]  -> 双线性放大 4 倍

最终返回：
p4,p3,p2,p1       四个 [B,9,224,224] logits
```

这里容易误解的一点是：

- 代码里的 `p4` 来自最深、最低分辨率的 `d4`。
- 代码里的 `p1` 来自最浅、最高分辨率的 `d1`。
- 返回列表顺序是 `[p4,p3,p2,p1]`。
- 测试代码使用 `P[-1]`，也就是代码命名下的 `p1`。

这只是代码编号方向问题，不要仅凭 `p1` 或 `p4` 名字猜谁是最终头。

---

## 5. Transformer 部分：PVTv2 编码器

### 5.1 为什么可以明确说它是 Transformer

`lib/pvtv2.py` 中实际存在：

- token 序列 `[B,N,C]`；
- Q、K、V 线性映射；
- 多头自注意力；
- scaled dot-product attention；
- softmax 注意力权重；
- LayerNorm；
- 残差连接；
- MLP / FFN；
- DropPath；
- 多个 Transformer Block 堆叠。

这些不是名字包装，而是真正执行的 Transformer 计算。

### 5.2 图像怎样变成 token

NLP Transformer 的 token 通常是词。视觉 Transformer 中，可以把一块图像区域映射成一个 token。

PVTv2 使用 `OverlapPatchEmbed`，但它不是先手动裁成互不重叠的小方块，而是使用带步幅和 padding 的卷积生成重叠 patch embedding：

```text
[B,C,H,W]
    |
    | Conv2d 下采样并改变通道
    v
[B,C',H',W']
    |
    | flatten + transpose
    v
[B,N,C']，N=H'×W'
```

“overlap”表示相邻 token 的输入感受区域可以重叠，有助于保留局部连续性。这一点比最原始 ViT 的完全不重叠 patch 更接近 CNN 的局部归纳偏置。

### 5.3 Q、K、V 在做什么

对输入 token `X` 做三组线性变换：

```text
Q = XWq
K = XWk
V = XWv
```

可以把它们理解为：

- Query：当前位置想寻找什么信息。
- Key：各位置声明自己具有什么特征。
- Value：各位置真正准备提供的内容。

注意力核心近似为：

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V
```

`QK^T` 衡量位置之间的匹配程度，softmax 把它变为权重，再对 V 加权求和。这样一个器官内部相距较远的区域，也可以建立直接的信息联系。

### 5.4 多头注意力

PVTv2-B2 四个阶段的通道数和头数为：

| 阶段 | 通道数 | 注意力头数 | 每头维度 |
|---|---:|---:|---:|
| stage 1 | 64 | 1 | 64 |
| stage 2 | 128 | 2 | 64 |
| stage 3 | 320 | 5 | 64 |
| stage 4 | 512 | 8 | 64 |

多个头允许模型在不同表示子空间中学习不同关系。它不保证某个头一定对应某个器官，但提供了同时建模多类关系的能力。

### 5.5 为什么 PVTv2 不直接做完整全局注意力

普通全局注意力对 N 个 token 形成 `N×N` 关系矩阵，复杂度随空间 token 数平方增长。医学分割需要较高分辨率，浅层 token 很多，直接计算代价很高。

PVTv2 使用 Spatial Reduction Attention。默认 B2 的四阶段 `sr_ratios` 是：

```text
[8, 4, 2, 1]
```

其基本思想是：

- Q 仍对应原始全部查询位置。
- 浅层对 K/V 做更强的空间下采样。
- 深层分辨率已经很低，逐渐减少下采样。
- 最后一层 `sr_ratio=1`，K/V 不再压缩。

于是模型保留较丰富的上下文建模能力，同时降低浅层注意力计算量。

### 5.6 PVTv2 的 Mix-FFN

标准 Transformer 的 FFN 通常是：

```text
Linear -> 激活 -> Linear
```

当前 PVTv2 的 MLP 中加入深度卷积，大体是：

```text
Linear 扩张通道
    -> 转回二维结构
    -> 3x3 Depth-wise Conv
    -> GELU
    -> Linear 投影回原通道
```

这使 FFN 不只独立处理每个 token，还显式引入邻域空间信息。所以 PVTv2 本身也不是“完全没有卷积”的 Transformer。

### 5.7 一个 PVTv2 Block 的骨架

可以把单个 Block 简化为：

```text
x = x + DropPath(Attention(LayerNorm(x)))
x = x + DropPath(MixFFN(LayerNorm(x), H, W))
```

这里涉及：

- Pre-Norm：先归一化，再进入子模块。
- Residual connection：学习增量而不是完全重写输入。
- DropPath：训练时随机跳过部分残差分支，属于随机深度正则化。
- GELU：Transformer 中常见的平滑非线性激活。

### 5.8 为什么叫 Pyramid Vision Transformer

原始 ViT 往往主要保留单尺度 token 表示，更自然地适合图像分类。分割需要多层分辨率特征，因此 PVTv2 像 CNN 一样构建金字塔：

```text
高分辨率、低语义：x1
        ↓
中等分辨率：x2
        ↓
更低分辨率：x3
        ↓
低分辨率、高语义：x4
```

这四级输出正好能与解码器和 skip connection 配合。

### 5.9 PVTv2-B0 到 B5 是什么关系

项目支持 `pvt_v2_b0` 到 `pvt_v2_b5`。它们通常在通道规模、Block 深度和 MLP 配置上有所差异。

当前默认 B2 配置的关键值是：

```text
embed_dims = [64,128,320,512]
num_heads  = [1,2,5,8]
depths     = [3,4,6,3]
sr_ratios  = [8,4,2,1]
mlp_ratios = [8,8,4,4]
```

不能只改 `depths` 等配置后继续假设原 B2 预训练权重能够严格匹配。模型结构和权重键、张量形状必须一致。

---

## 6. CNN 部分：它并不是配角

即使默认编码器是 PVTv2，当前模型仍大量依赖卷积。需要掌握以下卷积基础。

### 6.1 普通卷积

普通 `k×k Conv2d` 同时进行：

- 空间邻域特征提取；
- 不同输入通道之间的信息混合；
- 输出新通道。

它具有局部连接和权重共享，天然适合提取边缘、纹理和局部形状。

### 6.2 1×1 point-wise convolution

`1×1` 卷积不扩大空间感受野，主要负责每个像素位置的跨通道线性组合。当前项目中用于：

- 将单通道 CT 学习映射到 3 通道；
- MSCB 内通道扩张和投影；
- EUCB 中通道投影；
- 四个分割头把特征通道映射到 `num_classes`。

例如：

```text
[B,64,H,W] --1x1 Conv--> [B,9,H,W]
```

表示将 64 维特征转换成九类像素证据。

### 6.3 Depth-wise convolution

深度卷积设置 `groups=C`，每个通道独立做空间卷积，不立即混合通道。

普通卷积参数量大致为：

```text
C_in × C_out × k × k
```

深度卷积参数量大致为：

```text
C_in × k × k
```

随后再用 `1×1` point-wise convolution 混合通道，就是深度可分离卷积思想。EMCAD 利用它降低解码器参数量和计算量。

### 6.4 Group convolution

分组卷积将通道分组，每组独立卷积。它介于普通卷积与 depth-wise convolution 之间。LGAG 中使用分组卷积构造较轻量的门控映射。

### 6.5 Channel shuffle

分组或深度卷积可能导致不同通道组之间交流不足。`channel_shuffle` 通过 reshape、转置、再 reshape，重新排列通道，使后续卷积更容易混合原来分属不同组的信息。

注意：channel shuffle 本身没有可学习参数，它改变的是通道排列。

### 6.6 Batch Normalization

BN 根据 batch 和空间位置统计每个通道的均值、方差，并维护运行统计量。其行为取决于：

- `model.train()`：使用当前 batch 统计并更新 running mean/variance。
- `model.eval()`：使用已经保存的运行统计。

这也是为什么训练阶段误留在 eval 模式会真正改变模型行为。

### 6.7 ReLU、ReLU6、GELU、Sigmoid、Softmax

| 函数 | 当前项目用途 | 含义 |
|---|---|---|
| ReLU | 输入适配层、部分解码模块 | 负数截断为 0 |
| ReLU6 | Synapse 默认 MSCB 激活参数 | 将输出限制在 `[0,6]`，是轻量网络中常见激活 |
| GELU | PVTv2 Mix-FFN | Transformer 常用平滑激活 |
| Sigmoid | CAB/SAB/LGAG 权重、二分类任务概率 | 独立映射到 0 到 1 |
| Softmax | Synapse/ACDC 多分类概率 | 同一像素的互斥类别概率和为 1 |

注意：CAB/SAB 内部的 sigmoid 是生成注意力权重，不是生成最终分割概率。

### 6.8 Residual connection

残差结构通常写作：

```text
y = x + F(x)
```

它让网络学习对输入的修正，有助于梯度传播。PVTv2 Block、ResNet 和 MSCB 中都能看到残差思想，但它们的具体模块不同。

---

## 7. 可选 ResNet 编码器

`lib/networks.py` 还支持：

```text
resnet18
resnet34
resnet50
resnet101
resnet152
```

当选择 ResNet 时，编码器是 CNN，不再是 PVTv2 Transformer，但后面的 EMCAD 解码器仍然保留。

### 7.1 BasicBlock 与 Bottleneck

- ResNet18/34 主要使用 BasicBlock。
- ResNet50/101/152 主要使用 Bottleneck，并通过通道压缩、空间卷积、通道扩张构建更深网络。

### 7.2 编码器替换为什么不是改一个名字那么简单

解码器创建时必须知道四级编码特征的通道数：

| 编码器类型 | 浅到深通道 | 传给 EMCAD 的深到浅通道 |
|---|---|---|
| PVTv2-B2 | `[64,128,320,512]` | `[512,320,128,64]` |
| ResNet18/34 | `[64,128,256,512]` | `[512,256,128,64]` |
| ResNet50/101/152 | `[256,512,1024,2048]` | `[2048,1024,512,256]` |

如果编码器输出通道和 `channels` 配置不一致，LGAG、EUCB 或 MSCB 会在运行时出现通道不匹配。

### 7.3 架构与预训练权重是两个独立概念

- `encoder='pvt_v2_b2'` 决定创建什么结构。
- `pretrain=True/False` 决定是否向这个结构加载预训练参数。

关闭预训练不等于换掉 PVTv2，加载预训练也不等于恢复完整分割 checkpoint。ImageNet 预训练一般只初始化编码器；完整分割 checkpoint 则包含训练后的编码器、解码器和预测头。

---

## 8. EMCAD 解码器逐模块拆解

`lib/decoders.py::EMCAD` 接收：

```text
x = x4
skips = [x3,x2,x1]
```

返回：

```text
[d4,d3,d2,d1]
```

整体结构可以简化为：

```text
x4
 |
 | CAB -> SAB -> MSCB
 v
d4 -------------------------------> head4
 |
 | EUCB: 上采样 2 倍，512 -> 320
 v
候选 d3 ----作为 gate----> LGAG(x3)
 |                            |
 +-----------逐元素相加-------+
 |
 | CAB -> SAB -> MSCB
 v
d3 -------------------------------> head3
 |
 | EUCB: 上采样 2 倍，320 -> 128
 v
候选 d2 ----作为 gate----> LGAG(x2)
 |                            |
 +-----------逐元素相加-------+
 |
 | CAB -> SAB -> MSCB
 v
d2 -------------------------------> head2
 |
 | EUCB: 上采样 2 倍，128 -> 64
 v
候选 d1 ----作为 gate----> LGAG(x1)
 |                            |
 +-----------逐元素相加-------+
 |
 | CAB -> SAB -> MSCB
 v
d1 -------------------------------> head1
```

### 8.1 MSDC：Multi-Scale Depth-wise Convolution

MSDC 默认使用：

```text
kernel_sizes = [1,3,5]
```

每个分支是：

```text
Depth-wise Conv -> BN -> activation
```

不同卷积核观察不同大小的局部邻域：

- `1×1`：不扩展空间邻域，更多保留或重新标定当前响应。
- `3×3`：常规局部邻域。
- `5×5`：更大的局部感受野。

当前代码支持两种分支关系：

```text
dw_parallel=True：所有分支读取同一个输入
dw_parallel=False：后一个分支读取经过前一分支残差更新后的输入
```

MSDC 自己返回多个分支结果列表，实际聚合由 MSCB 完成。

### 8.2 MSCB：Multi-Scale Convolutional Block

MSCB 可以简化为：

```text
输入
 -> 1x1 Conv 扩张通道
 -> MSDC 多尺度深度卷积
 -> 分支求和或 concat
 -> channel shuffle
 -> 1x1 Conv 投影到目标通道
 -> 条件允许时与输入做残差相加
```

涉及两个重要参数：

- `expansion_factor`：内部通道扩张倍数。默认 2。
- `add`：多尺度结果是逐元素相加，还是沿通道拼接。

这里的“多尺度”主要是同一层内不同卷积核的局部感受野，不等于 PVTv2 四阶段的分辨率金字塔。当前项目同时使用这两种多尺度。

### 8.3 EUCB：Efficient Up-Convolution Block

EUCB 负责把解码特征放大一级并调整通道，大体流程为：

```text
nearest-neighbor Upsample ×2
 -> Depth-wise Conv
 -> BN + activation
 -> channel shuffle
 -> 1x1 point-wise Conv
```

它的两个目标是：

1. 空间尺寸扩大两倍，逐步恢复定位精度。
2. 通道从深层配置转换到下一级配置，例如 `512 -> 320`。

EUCB 输出还不是分割图，只是更高分辨率的解码特征。

### 8.4 LGAG：Large-kernel Grouped Attention Gate

LGAG 接收两路输入：

- `g`：来自更深层解码器的 gating signal。
- `x`：编码器的 skip feature。

它先将两路特征投影到可比较空间，融合后生成 sigmoid 空间权重，再乘回 skip：

```text
gate = sigmoid(f(g,x))
filtered_skip = x × gate
```

其目的不是凭空创造新特征，而是让解码器的高层语义告诉网络：编码器 skip 中哪些位置值得保留，哪些背景细节应压低。

随后代码执行：

```text
decoder_feature + filtered_skip
```

这是逐元素相加，不是经典 U-Net 的通道 concat。因此两路张量必须在空间尺寸和通道数上对齐。

### 8.5 CAB：Channel Attention Block

CAB 关注“哪些通道重要”。它从空间维度进行全局平均池化和最大池化，生成每个通道的权重：

```text
[B,C,H,W]
 -> 平均池化 / 最大池化
 -> 通道变换
 -> sigmoid
 -> [B,C,1,1]
```

再将权重广播乘回原特征。它对同一通道的全部空间位置使用同一个权重。

### 8.6 SAB：Spatial Attention Block

SAB 关注“哪些空间位置重要”。它沿通道维计算平均图和最大图，拼接后通过 `7×7` 卷积得到：

```text
[B,1,H,W]
```

经 sigmoid 后乘回输入。同一个空间位置的权重会广播到全部通道。

当前 EMCAD 的四级解码共享同一个 SAB 模块参数。共享模块可以在不同分辨率上重复调用，每次权重图根据当前输入重新计算，但卷积参数是同一份。

### 8.7 MSCAM 到底在哪里

论文概念中的 MSCAM 在当前代码中没有单独的 `class MSCAM`。它是以下顺序的组合概念：

```text
CAB -> 特征相乘
 -> SAB -> 特征相乘
 -> MSCB
```

所以搜不到 `class MSCAM` 不代表该思想没有实现。阅读论文和代码时，要区分“论文模块命名”和“代码类拆分方式”。

---

## 9. 三种 Attention 不要混为一谈

| 机制 | 输入关系 | 权重主要表示什么 | 是否 Transformer self-attention |
|---|---|---|---:|
| PVTv2 Attention | token 与 token | 不同空间 token 之间的信息依赖 | 是 |
| CAB | 通道统计 | 哪些特征通道更重要 | 否 |
| SAB | 通道平均/最大图 | 哪些空间位置更重要 | 否 |
| LGAG | 解码 gate 与 skip | skip 的哪些位置应该通过 | 否 |

最重要的判断标准不是模块名称中有没有 `Attention`，而是它是否真正构造 Q、K、V 并计算 token 间注意力矩阵。

因此：

- PVTv2 是 Transformer attention。
- CAB、SAB、LGAG 是注意力式特征重标定或门控。
- “用了 attention”不自动等于“用了 Transformer”。

---

## 10. 为什么这个项目不涉及 RNN

RNN 的核心不是“代码里出现循环”，而是：

```text
h_t = f(x_t, h_(t-1))
```

当前时刻输出依赖上一个时刻的隐藏状态。LSTM 和 GRU 也是围绕这种有状态递归建立的。

当前仓库有效 Python 代码中没有找到：

```text
nn.RNN
nn.LSTM
nn.GRU
```

也没有自定义的跨步隐藏状态更新。

### 10.1 逐切片循环为什么不算 RNN

Synapse 测试时会遍历 3D 病例中的切片：

```python
for ind in range(image.shape[0]):
    slice = image[ind, :, :]
    output = net(slice)
```

这只是普通 Python 循环。第 `t` 张切片推理时，没有接收第 `t-1` 张切片的隐藏状态；每张切片对模型来说都是一次独立输入。

所以：

```text
有 for 循环 != RNN
按顺序处理切片 != 建模切片序列
```

当前模型可能通过单张切片内部的 Transformer 建模远距离二维关系，但不会直接建模相邻 CT 层之间的三维连续性。

---

## 11. 为什么这个项目不涉及 Mamba

Mamba 通常属于 State Space Model 路线，典型关键词包括：

```text
Mamba block
SSM / State Space Model
Selective Scan
状态更新
mamba_ssm
```

当前有效运行链和依赖中没有这些模块。PVTv2 的注意力也不会因为同样能建模长距离依赖，就变成 Mamba。

Transformer 与 Mamba 的核心计算路线不同：

| 项目 | Transformer | Mamba / SSM |
|---|---|---|
| 主要机制 | Q/K/V 注意力 | 状态空间递推与 selective scan |
| 当前 EMCAD 默认编码器 | 有 | 无 |
| 当前仓库主链实现 | `lib/pvtv2.py` | 不存在 |

如果未来引入 VMamba、SegMamba 或其他视觉 SSM 编码器，那将是新的编码器或新模型分支，需要明确实现、依赖和消融实验，不能把当前 PVTv2 基线直接称为 Mamba 模型。

---

## 12. 四个分割头、logits 与概率

### 12.1 为什么需要 1×1 Conv 输出头

解码特征 `d1` 的 64 个通道不是类别。输出头用 `1×1 Conv` 将其映射为 K 个类别分数：

```text
[B,64,56,56] -> [B,K,56,56]
```

再双线性插值到输入大小。

四个头分别连接 `d4,d3,d2,d1`，使不同解码深度都能产生完整分割预测。

### 12.2 为什么模型内部不做 softmax 或 sigmoid

`EMCADNet.forward()` 返回 logits，不在模型内部做最终概率化，原因包括：

- `CrossEntropyLoss` 期望原始 logits，内部会稳定地执行 log-softmax。
- `BCEWithLogitsLoss` 将 sigmoid 与 BCE 合并，数值更稳定。
- mutation supervision 需要先对多个头的 logits 求和，再计算损失。
- 推理时再根据任务类型选择 softmax 或 sigmoid。

### 12.3 多分类和二分类的区别

Synapse/ACDC 是互斥多分类：

```text
K>1 个输出通道
softmax(dim=1)
argmax 得到类别编号
```

Polyp、ISIC、BUSI 当前主要是二分类前景分割：

```text
1 个输出通道
sigmoid 得到前景概率
阈值化得到前景/背景
```

单通道二分类并不是没有背景，而是背景概率隐含为：

```text
P(background) = 1 - P(foreground)
```

---

## 13. 损失函数与四输出监督

### 13.1 Cross Entropy

交叉熵从逐像素分类角度约束正确类别概率。Synapse 中输入是：

```text
logits：[B,9,H,W]
target：[B,H,W]，值域 0~8
```

它关心每个像素的类别是否预测正确，但对小器官和类别不平衡可能不够敏感。

### 13.2 Dice Loss

Dice 系数关注预测区域和真实区域的重叠：

```text
Dice = 2|P∩G| / (|P|+|G|)
DiceLoss ≈ 1 - Dice
```

训练时使用连续 softmax 概率计算 soft Dice，不能先 argmax。因为 argmax 是离散操作，无法提供正常梯度。

当前 Synapse `DiceLoss(num_classes=9)` 会把整数标签转为 one-hot，并按类别计算后平均。类别数必须和模型输出通道一致。

### 13.3 当前组合损失

每个受监督预测组合使用：

```text
Loss = 0.3 × CE + 0.7 × DiceLoss
```

这同时兼顾像素分类与区域重叠。

### 13.4 last_layer

只监督返回列表最后一个输出：

```text
P[-1] = p1
```

计算量最小，但较深解码层只能通过最终路径间接接收监督。

### 13.5 deep_supervision

分别监督四个输出：

```text
Loss(p4) + Loss(p3) + Loss(p2) + Loss(p1)
```

深层特征可以更直接地接收分割目标梯度。这通常有助于优化，但也会改变损失量级和各尺度之间的约束。

### 13.6 mutation supervision

当前默认是 `mutation`。四个输出索引 `[0,1,2,3]` 的所有非空子集共有：

```text
2^4 - 1 = 15
```

例如：

```text
{p4}
{p3}
{p4,p3}
{p4,p2,p1}
{p4,p3,p2,p1}
...
```

对每个子集，代码先逐元素相加 logits：

```text
iout = sum(该子集中的 logits)
```

再计算 CE 和 Dice，最后将 15 组损失相加。

必须知道的细节：

- 相加的是 logits，不是 softmax 概率。
- 15 组损失当前直接累加，没有除以 15。
- 因此 mutation、deep supervision、last layer 的原始 loss 数值不可直接横向比较。
- 如果改变监督方式，优化尺度也可能变化，不能只看日志中的 loss 大小判断方法优劣。

---

## 14. 训练过程涉及的深度学习基础

### 14.1 forward、loss、backward、step

一个训练 batch 的核心过程是：

```text
1. image -> model(image)              前向传播
2. logits + label -> loss             计算误差
3. optimizer.zero_grad()              清空旧梯度
4. loss.backward()                    自动微分，计算梯度
5. optimizer.step()                   更新参数
```

PyTorch autograd 会沿计算图把损失梯度从输出头、解码器一直传回编码器。

### 14.2 AdamW

当前 Synapse 使用：

```text
AdamW
learning rate = 1e-4
weight decay = 1e-4
```

AdamW 使用一阶矩和二阶矩估计自适应调整不同参数的更新尺度，并将 weight decay 与梯度更新更清晰地解耦。

当前训练器实际保持基础学习率，代码中曾考虑的多项式衰减逻辑没有作为当前主逻辑启用。读代码时要以实际执行语句为准，而不是以被注释掉的公式为准。

### 14.3 batch size 与 GPU

当前训练器构造 DataLoader 时使用：

```text
实际 batch_size = args.batch_size × args.n_gpu
```

但这不自动保证多 GPU 并行正确。模型是否被 `DataParallel`/DDP 包装、每卡显存和 checkpoint 键名都要结合实际代码检查。

### 14.4 train 和 eval 模式

`model.train()` 与 `model.eval()` 不控制是否计算梯度；它们主要切换 BN、Dropout、DropPath 等层的行为。是否记录梯度通常由 `torch.no_grad()`、参数 `requires_grad` 和计算图决定。

当前 `trainer.py` 存在一个需要特别注意的实际行为：

1. 进入 epoch 循环前调用一次 `model.train()`。
2. 每个 epoch 末验证函数调用 `model.eval()`。
3. 下一 epoch 开头没有再次调用 `model.train()`。

因此第一个 epoch 验证后，后续 epoch 会继续保持 eval 模式。这会影响 BatchNorm 和 DropPath 等训练行为。它不是 EMCAD 理论结构的一部分，而是当前训练实现状态；正式做可复现实验前应明确修正并重新建立基线。

### 14.5 权重初始化和预训练

解码器卷积层有自己的初始化逻辑，PVTv2 编码器可以加载本地 ImageNet 预训练权重。需要区分：

- 随机初始化：参数从某种分布开始。
- 编码器预训练：使用其他大数据集学到的视觉特征初始化骨干。
- 分割 checkpoint：保存当前任务训练后的完整模型参数。
- optimizer state：AdamW 的动量等状态，当前 `state_dict` 权重文件并不包含这些内容。

只保存 `model.state_dict()` 可以恢复模型推理或继续微调，但不能做到包含 epoch、优化器、随机数状态在内的完全无缝续训。

---

## 15. Synapse 数据管线

### 15.1 训练与测试数据粒度不同

当前 Synapse 路径采用：

| 阶段 | 数据形式 | 典型张量 | 用途 |
|---|---|---|---|
| 训练 | 二维切片 `.npz` | image/label 为 `[H,W]` | 随机抽取和批训练 |
| 验证/测试 | 完整病例 `.npy.h5` | image/label 为 `[D,H,W]` | 逐切片推理后做病例级指标 |

所以它是：

> **二维模型训练 + 三维病例组织与评估。**

不能因为测试输入文件是三维病例，就称当前网络为 3D 网络。

### 15.2 数据增强

`RandomGenerator` 主要包括：

- 随机 `90°` 倍数旋转加翻转；
- 小角度随机旋转；
- 缩放到固定 `img_size`；
- NumPy 转 PyTorch Tensor。

图像和标签必须同步做几何变换，否则位置会错位。

### 15.3 图像与标签为什么使用不同插值

当前 resize 中：

- 连续 CT 图像使用高阶插值。
- 离散类别标签使用最近邻式的 `order=0`。

标签不能用普通平滑插值，否则类别 `1` 和 `2` 之间可能产生 `1.4` 之类没有语义的类别值。

### 15.4 数据划分与泄漏

`trainer.py` 验证时使用的 split 名称是 `test_vol`。这个名字本身不能证明它究竟是验证集还是最终测试集，必须检查实际列表文件和实验协议。

如果每个 epoch 都在真正测试集上选择 `best.pth`，那么测试集已经参与模型选择，最终测试指标会偏乐观。规范研究通常应满足：

```text
训练集：更新模型参数
验证集：调参、早停、选择 best checkpoint
测试集：最终冻结方案后只评估
```

医学数据还必须按患者划分，不能把同一患者的不同切片分到训练和测试两边。

---

## 16. 推理与评估

### 16.1 当前病例级推理流程

对一个 `[D,H,W]` 病例：

```text
for 每个切片：
    取 [H,W]
    resize 到 [224,224]
    增加 batch 和 channel 维
    输入 EMCADNet
    取 outputs[-1]
    softmax + argmax
    resize 回原切片尺寸
    写入 prediction[ind]

所有切片处理完：
    得到 [D,H,W] 预测体
    对每个前景类别计算病例级指标
```

### 16.2 Dice

Dice 衡量区域重叠，越大越好：

```text
Dice = 2TP / (2TP+FP+FN)
```

它对总体重叠很直观，但不能充分反映边界上少量但严重的局部误差。

### 16.3 Jaccard / IoU

```text
IoU = |P∩G| / |P∪G|
```

它和 Dice 都衡量重叠，但数值尺度不同，不能将二者直接当作同一个指标。

### 16.4 HD95

Hausdorff Distance 关注两组边界点之间最坏距离。HD95 使用距离分布的 95% 分位数，减少极端离群点影响。越小越好。

它更能揭示某些区域是否出现远离真实器官的错误分割或边界偏移。

### 16.5 ASSD / ASD

Average Symmetric Surface Distance 衡量预测边界与真实边界之间的平均对称距离，越小越好。

### 16.6 当前指标实现的两个注意事项

第一，当前 Synapse helper 调用 MedPy 的 HD95/ASSD 时没有传入真实 voxel spacing，因此这些距离实际上按数组网格间距计算，不能自动解释为毫米。

第二，当前 `calculate_metric_percase` 的空掩膜分支比较特殊：

```text
预测有前景、真值为空 -> 返回 Dice=1、Jaccard=1
预测为空、真值有前景或双方为空 -> 返回 0
```

这不是通常直觉下的空类规则。正式报告实验前必须统一评价协议，并记录是否修改过该逻辑，否则不同代码版本的指标不可直接比较。

---

## 17. 项目支持的不同分割任务

当前仓库不只包含 Synapse。不同任务帮助理解“模型骨架”和“任务接口”之间的区别。

| 数据/任务 | 模态 | 任务类型 | 输出通道与激活 | 主要损失思路 |
|---|---|---|---|---|
| Synapse | 腹部 CT | 多器官多分类 | 9 通道，softmax | CE + 多类 Dice |
| ACDC | 心脏 MRI | 多结构多分类 | 4 通道，softmax | CE + 多类 Dice |
| Polyp | 内镜 RGB | 二值分割 | 1 通道，sigmoid | 加权 BCE + 加权 IoU 等结构损失 |
| ISIC | 皮肤镜 RGB | 二值病灶分割 | 1 通道，sigmoid | 复用二值分割训练框架 |
| BUSI | 乳腺超声 | 二值病灶分割 | 1 通道，sigmoid | 复用二值分割训练框架 |

从这些任务中需要掌握：

- 灰度与 RGB 输入的差异；
- 多分类与二分类输出设计；
- 背景是显式通道还是隐式概率；
- CT、MRI、内镜、皮肤镜、超声的成像特点不同；
- 同一架构不能不经检查就沿用全部预处理、损失和评价协议。

---

## 18. 多尺度到底有哪几层含义

“多尺度”是医学分割论文中最容易被泛化使用的词。当前项目至少包含四种不同含义：

### 18.1 编码器分辨率金字塔

```text
x1: 1/4
x2: 1/8
x3: 1/16
x4: 1/32
```

这是跨层级、跨空间分辨率的多尺度。

### 18.2 MSDC 多卷积核

```text
1x1、3x3、5x5 depth-wise convolution
```

这是同一分辨率下不同局部感受野的多尺度。

### 18.3 skip connection 融合

解码器逐层融合浅层定位细节和深层语义，是跨语义层次的多尺度融合。

### 18.4 四个预测头

四个解码尺度都产生分割 logits，并通过 deep supervision 或 mutation 参与学习，是输出监督层面的多尺度。

所以论文说“增强多尺度能力”时，必须追问：增强的是哪一种尺度，具体张量从哪里来，如何融合，是否真的进入损失。

---

## 19. 工程层面的基础知识

### 19.1 `nn.Module` 和模块注册

所有可训练层都应作为 `nn.Module` 的属性，或放在 `Sequential`、`ModuleList` 等注册容器中。这样 PyTorch 才能：

- 在 `parameters()` 中发现参数；
- 将参数交给 optimizer；
- 随 `.cuda()` 移动设备；
- 写入 `state_dict()`；
- 在 train/eval 间切换行为。

MSDC 使用 `ModuleList` 动态保存多个卷积分支，就是这一知识点的实际例子。

### 19.2 CPU、CUDA 与显存

模型参数和输入必须位于同一设备。常见错误包括：

```text
模型在 cuda，输入在 cpu
checkpoint 保存自另一设备，加载时未设置 map_location
CUDA/PyTorch/驱动不兼容
batch 太大导致显存不足
```

参数量、FLOPs、显存和实际速度是四个不同概念：

- 参数量影响权重存储，但不是全部显存。
- 训练还要存激活、梯度和优化器状态。
- FLOPs 不完全等于硬件实际耗时。
- depth-wise convolution 参数少，但具体硬件上不一定按同等比例加速。

### 19.3 DataLoader

需要理解：

- `Dataset.__len__` 与 `__getitem__`；
- batch 拼接；
- `shuffle`；
- `num_workers`；
- `pin_memory`；
- worker 随机种子。

数据加载错误、Windows 多进程行为或坏样本，都可能表现为“训练启动失败”，并不一定是模型问题。

### 19.4 随机种子与可复现性

当前入口设置 Python、NumPy、PyTorch 和 CUDA 随机种子，并配置确定性选项。但严格可复现还取决于：

- PyTorch/CUDA/cuDNN 版本；
- GPU 型号；
- DataLoader worker；
- 某些算子是否有确定性实现；
- 数据清单是否完全固定；
- checkpoint 选择协议；
- 是否真的恢复 optimizer 和随机状态。

“设置了 seed”不等于所有环境下逐 bit 一致。

### 19.5 checkpoint

当前训练主要保存：

```text
last.pth
best.pth
阶段性 epoch checkpoint
```

要区分：

- `last.pth`：最后一次覆盖保存的参数。
- `best.pth`：按当前验证指标选择的最好参数。
- 预训练 PVT `.pth`：只用于初始化编码器。

模型结构参数必须和 checkpoint 一致，包括编码器、类别数、kernel sizes、聚合方式和有关通道配置。

### 19.6 日志与 TensorBoard

训练日志不仅用来观察 loss，还应记录：

- 完整命令行参数；
- 数据划分哈希或清单；
- Git commit 与未提交修改；
- 环境版本；
- 每类指标而非只看平均值；
- 失败实验；
- 最优 checkpoint 对应 epoch。

当前 `trainer.py` 在 epoch 末日志中的 loss 是最后一个 batch 的 loss，不是完整 epoch 平均值。阅读曲线或比较实验时要知道这个实际语义。

---

## 20. 当前项目明确没有的技术

下面这些概念可以作为未来学习方向，但不能写成当前基线已经包含：

| 技术 | 当前是否存在 | 容易产生的误判 |
|---|---:|---|
| RNN/LSTM/GRU | 否 | 逐切片 for 循环被误认为序列模型 |
| Mamba/SSM | 否 | 只因模型关注长距离关系就被称为 Mamba |
| Conv3d | 主模型无 | 完整病例 H5 输入被误认为 3D 网络 |
| 3D Transformer | 否 | 二维 PVTv2 被误认为处理体素 token |
| Diffusion model | 否 | “生成分割 mask”被误解为生成式扩散 |
| GNN | 否 | 多尺度特征之间有连接被误叫图网络 |
| SAM/MedSAM 提示分割 | 否 | 一般 segmentation 与 promptable segmentation 混淆 |
| 自监督预训练 | 当前主线无 | ImageNet 有监督预训练被误称自监督 |
| 多模态文本-图像模型 | 否 | 医学图像任务被自动等同视觉语言模型 |
| 时序建模 | 否 | ACDC 心动相位或 CT 切片顺序并未自动形成时序网络 |

---

## 21. 这个项目涉及的基础知识树

### 21.1 第一层：必须掌握，否则读不懂训练代码

- Python 函数、类、继承、列表、字典、循环、条件分支。
- NumPy 数组与 PyTorch Tensor。
- 张量形状 `[B,C,H,W]`。
- CPU 与 GPU 设备。
- `Dataset`、`DataLoader`、batch。
- `nn.Module`、`forward`、参数注册。
- logits、softmax、sigmoid、argmax。
- loss、梯度、反向传播、optimizer。
- train/eval 和 `torch.no_grad()`。

### 21.2 第二层：必须掌握，否则读不懂 EMCAD

- 普通卷积、padding、stride、kernel size。
- 1×1 卷积。
- depth-wise、point-wise、group convolution。
- BN 和激活函数。
- 残差连接。
- 上采样和双线性插值。
- 通道拼接与逐元素相加的区别。
- channel shuffle。
- 感受野与多尺度。
- skip connection。
- channel attention、spatial attention、attention gate。

### 21.3 第三层：必须掌握，否则读不懂 PVTv2

- 图像 patch/token。
- `[B,C,H,W]` 与 `[B,N,C]` 转换。
- Q/K/V。
- scaled dot-product attention。
- multi-head attention。
- LayerNorm。
- MLP/FFN。
- positional/spatial information。
- Spatial Reduction Attention。
- Dropout 与 DropPath。
- 金字塔特征层级。

### 21.4 第四层：必须掌握，否则不会做分割实验

- 二分类与多分类分割。
- one-hot 标签。
- CE、BCE、Dice、IoU 类损失。
- 类别不平衡。
- 深监督和多输出聚合。
- 数据增强与标签插值。
- 患者级数据划分与数据泄漏。
- validation 和 test 的职责。
- Dice、IoU、HD95、ASSD。
- checkpoint 选择。
- 随机种子和复现。
- baseline、ablation、fair comparison。

### 21.5 第五层：做创新时再深入

- 2.5D/3D 网络与跨切片建模。
- Mamba/SSM 与线性复杂度序列建模。
- 边界损失、拓扑损失和小目标优化。
- 不确定性估计与校准。
- 半监督、自监督、领域适应。
- 跨数据集泛化。
- 蒸馏、EMA teacher、一致性学习。
- 计算复杂度、显存、吞吐和部署。
- 统计显著性、置信区间和多随机种子实验。

---

## 22. 建议的代码阅读顺序

不要一上来从 `lib/pvtv2.py` 第一行硬啃到最后。更有效的顺序如下。

### 第 1 步：先看一次完整调用

阅读：

```text
train_synapse.py
```

目标不是看懂每行，而是回答：

- 输入路径在哪里？
- 默认参数是什么？
- 创建了什么模型？
- 最后调用哪个 trainer？

### 第 2 步：看总模型接口

阅读：

```text
lib/networks.py::EMCADNet
```

必须手写出：

```text
输入 -> encoder -> x1/x2/x3/x4 -> decoder -> d4/d3/d2/d1 -> p4/p3/p2/p1
```

先掌握数据流，再钻模块内部。

### 第 3 步：只看 PVTv2 forward

先定位：

```text
pvt_v2_b2()
PyramidVisionTransformerV2.forward_features()
OverlapPatchEmbed.forward()
Block.forward()
Attention.forward()
Mlp.forward()
```

每读一层都记录输入输出 shape。不要先被大量初始化代码淹没。

### 第 4 步：看 EMCAD forward

阅读：

```text
lib/decoders.py::EMCAD.forward
```

再逆向进入：

```text
MSCB -> MSDC
EUCB
LGAG
CAB
SAB
```

重点记录每个模块是否改变：

```text
空间尺寸？
通道数？
特征内容？
是否有可学习参数？
是否有残差？
```

### 第 5 步：看训练器

阅读：

```text
trainer.py
```

把一个 batch 的全部张量 shape 写出来，并分别模拟三种 supervision 的输出选择。

### 第 6 步：看数据和评估

阅读：

```text
utils/dataset_synapse.py
utils/utils.py::DiceLoss
utils/utils.py::val_single_volume
utils/utils.py::test_single_volume
```

最终应能回答：为什么训练样本是二维的，而测试指标是病例级的。

### 第 7 步：最后读其他任务和参考创新

在基线完全理解并成功复现后，再看：

```text
ACDC
Polyp / ISIC / BUSI
DG_EMCAD_reference/
```

否则很容易把不同任务、不同损失和未集成参考代码混成一个模型。

---

## 23. 建议建立的个人拆解表

以后阅读任何新模块，都按下表记录：

| 项目 | 要回答的问题 |
|---|---|
| 模块名称 | 论文名和代码类名是否一致？ |
| 调用位置 | 谁实例化它？谁在 forward 中调用它？ |
| 输入 shape | `[B,C,H,W]` 还是 `[B,N,C]`？ |
| 输出 shape | 空间、通道、序列长度怎样变化？ |
| 核心算子 | Conv、Attention、MLP、Scan 还是简单张量操作？ |
| 参数量来源 | 哪些层有可学习参数？ |
| 作用 | 提取特征、融合、上采样、门控还是输出？ |
| 梯度来源 | 哪个 loss 能直接或间接训练它？ |
| 训练/推理差异 | BN、DropPath、softmax、阈值是否不同？ |
| 是否在主链 | 只是文件存在，还是确实被导入并调用？ |

这张表比只背模块缩写更接近真正的科研能力。

---

## 24. 常见问题快速判断

### 24.1 EMCAD 是 Transformer 模型吗

准确回答：默认完整 EMCADNet 使用 PVTv2 Transformer 编码器，所以整体涉及 Transformer；但 EMCAD 解码器本身主要是卷积、多尺度 depth-wise convolution 和卷积式注意力。

### 24.2 EMCAD 是 CNN 模型吗

准确回答：它大量使用 CNN 算子，但默认编码器不是纯 CNN。因此最好称为 Transformer-CNN 混合分割模型。

### 24.3 CAB、SAB、LGAG 都是 Transformer 吗

不是。它们都能被统称为 attention mechanism，但不是 Q/K/V token self-attention。

### 24.4 逐切片处理是不是 RNN

不是。只有显式把前一切片的隐藏状态传给后一切片，才具备递归序列建模含义。当前切片彼此独立。

### 24.5 项目是不是 Mamba

不是。当前没有 Mamba/SSM/Selective Scan 实现。

### 24.6 项目是不是 3D 分割

任务结果是三维病例分割，但当前神经网络本身是二维网络。更准确地说是“2D slice-based segmentation with 3D case-level evaluation”。

### 24.7 它和 U-Net 的关系是什么

共同点：编码器-解码器、多尺度特征、skip connection、逐级恢复分辨率。

不同点：编码器默认是 PVTv2，解码器是 EMCAD，skip 先经过 LGAG 并采用加法融合，且有 CAB/SAB/MSCB 和四输出监督。因此不能简单说它就是原版 U-Net。

### 24.8 四个输出是不是四次最终预测

它们是从四个解码尺度得到的四组全尺寸 logits。训练时可用于深监督或 mutation，测试主路径通常只取列表最后一个 `p1`。

### 24.9 attention 能自动解决小器官分割吗

不能。attention 只是信息选择或交互机制。小器官表现还取决于数据量、分辨率、类别不平衡、损失、增强、标注质量、边界模糊和评价协议。

### 24.10 模块更多就一定更好吗

不一定。模块可能增加参数、训练不稳定、过拟合或只在某个随机种子上有效。研究结论必须通过固定基线、逐项消融、多随机种子和规范测试来支撑。

---

## 25. 从“会运行”到“能做研究”的学习路线

### 阶段一：能解释一次 forward

完成标准：

- 能从输入写出 x1~x4、d4~d1、p4~p1 shape。
- 能解释哪个地方是 Transformer，哪个地方是 CNN。
- 能说明 CAB/SAB 与 self-attention 的区别。
- 能解释最终为什么输出 9 通道。

### 阶段二：能解释一次训练更新

完成标准：

- 能说清 CE 和 Dice 各自约束什么。
- 能手工列出 mutation 的 15 个非空组合概念。
- 能解释 logits 为什么不能在模型里提前 argmax。
- 能说明 `zero_grad -> backward -> step`。

### 阶段三：能独立验证实验协议

完成标准：

- 能检查训练、验证、测试患者是否重叠。
- 能核对 best checkpoint 是否由验证集选择。
- 能解释 Dice、HD95 的单位和空类规则。
- 能记录 seed、版本、数据清单和 checkpoint 配置。

### 阶段四：能做可信消融

例如只改变一项：

```text
PVTv2-B2 -> ResNet34
kernel_sizes [1,3,5] -> [1,3]
parallel -> series
add -> concat
mutation -> deep_supervision
有 LGAG -> 无 LGAG
```

其他条件保持一致，重复多随机种子，报告均值和波动。此时才是在检验具体科学假设，而不只是堆模块。

### 阶段五：再考虑新技术

如果未来考虑 Mamba、3D、边界约束或不确定性，首先明确它解决当前基线的哪个已验证问题。例如：

- 二维切片缺少层间连续性：考虑 2.5D/3D/跨切片状态建模。
- 长距离建模成本高：比较 Transformer 与 SSM。
- 小器官边界差：研究边界监督或高分辨率分支。
- 多尺度输出互相冲突：研究可靠的跨尺度一致性或不确定性路由。

技术名称不是研究问题。先找到可测量的失败模式，再选择机制。

---

## 26. 最终知识地图

可以把整个当前项目压缩成以下六层：

```text
第 1 层：任务
医学图像语义分割，逐像素分类

第 2 层：数据
二维切片训练，三维病例组织与评估

第 3 层：编码器
默认 PVTv2-B2：真正的视觉 Transformer
可选 ResNet：CNN 编码器

第 4 层：解码器
EMCAD：EUCB 上采样 + LGAG skip 门控
       + CAB/SAB 注意力 + MSCB/MSDC 多尺度深度卷积

第 5 层：输出与学习
四个 logits 头 + CE/Dice
mutation / deep supervision / last layer

第 6 层：评估与工程
逐切片推理、病例级 Dice/HD95/IoU/ASSD
CUDA、DataLoader、checkpoint、复现和数据划分
```

最后再次给出明确结论：

1. **涉及 Transformer**：默认 PVTv2-B2 编码器就是层次化视觉 Transformer。
2. **大量涉及 CNN**：输入适配、PVTv2 的局部卷积、整个 EMCAD 解码器和输出头都使用卷积。
3. **不涉及 RNN**：没有 RNN/LSTM/GRU，也没有切片间隐藏状态。
4. **不涉及 Mamba**：没有 SSM、Selective Scan 或 Mamba 依赖。
5. **不是原版 U-Net**：但继承了编码器-解码器、多尺度和 skip connection 的基本思想。
6. **当前主模型是二维的**：三维病例通过逐切片预测后重新组合，不能称为 3D CNN/Transformer。
7. **真正需要掌握的主轴**：张量形状、卷积、Transformer、特征金字塔、skip 融合、注意力、上采样、分割损失、深监督、数据划分与评价协议。

如果你能不看代码，独立画出默认 B2 的四级 shape、解释四类 attention 的区别，并完整描述一个 batch 从 DataLoader 到 AdamW 更新的过程，就已经跨过了“只会启动项目”的阶段，开始真正理解 EMCAD。
