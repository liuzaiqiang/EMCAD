# 当前自适应多尺度解码模块解决 EMCAD 什么问题

## 1. 先给结论

当前 DGEMCAD 的核心想法是：

> EMCAD 原始解码器同时计算 1x1、3x3、5x5 多尺度深度卷积分支，但对每个像素位置使用固定的相加聚合；DGEMCAD 为每个解码阶段、每个空间位置预测一组分支权重，让不同位置动态混合不同感受野。

它针对的不是 EMCAD 的编码器问题，也不是 EMCAD 的上采样和 skip connection 问题，主要针对的是：

> 固定多尺度融合无法根据局部区域的难度和跨尺度预测是否一致，自适应决定更依赖小感受野还是大感受野。

当前实现可以称为一个“自适应多尺度解码模块”。不过，以下表述必须准确：

- 代码实现的是 **软路由混合**，不是只选择一个分支；
- 代码中的“相邻尺度差异”比较的是相邻解码阶段的辅助预测 logits 概率分布，不是直接比较两个 decoder feature tensor；
- 训练时除了结构替换，还加入了路由辅助损失和路由正则，因此当前实验不是只改变结构；
- 这些机制是否真正改善 EMCAD，需要通过 EMCAD、DGEMCAD、路由消融和多 seed 实验验证，不能仅凭代码宣布有效。

## 2. EMCAD 原始解码器的具体行为

### 2.1 EMCAD 已经具备多尺度分支

原始 `lib/decoders.py` 的 `MSDC` 定义了由 `kernel_sizes` 决定的多个 depth-wise convolution 分支，默认是 `[1, 3, 5]`（见 `lib/decoders.py:241-285`）。

原始 `MSCB` 先用 1x1 卷积把输入通道扩展为 `C_ex = expansion_factor * C_in`，然后将多个尺度分支的结果融合，再用 1x1 卷积投影回输出通道，并加上残差（见 `lib/decoders.py:394-437`、`lib/decoders.py:444-482`）。

当 `add=True` 时，原始分支融合的关键代码是：

```python
dout = 0
for dwout in msdc_outs:
    dout = dout + dwout
```

对应 `lib/decoders.py:453-462`。这意味着默认情况下：

1. 1x1、3x3、5x5 分支都被计算；
2. 每个位置都使用同一套固定融合规则；
3. 没有一个输入相关的空间权重图去表示“当前位置更应该信任哪个尺度”；
4. 这里的“固定”不是说三个分支的数值完全相等，而是说它们没有由当前图像内容产生的显式动态权重。

因此，不能把 EMCAD 说成“没有多尺度能力”。更准确的论文问题表述是：

> EMCAD 具有多尺度提取能力，但其 MSDC 分支聚合规则是静态的，缺少位置自适应的尺度选择机制。

### 2.2 EMCAD 的四级解码流程

原始 `EMCAD` 在 `lib/decoders.py:884-1040` 中按深到浅处理：

```text
x4 -> CAB/SAB -> MSCB4 -> d4
d4 -> EUCB3 -> LGAG(skip x3) -> 相加 -> CAB/SAB -> MSCB3 -> d3
d3 -> EUCB2 -> LGAG(skip x2) -> 相加 -> CAB/SAB -> MSCB2 -> d2
d2 -> EUCB1 -> LGAG(skip x1) -> 相加 -> CAB/SAB -> MSCB1 -> d1
```

具体代码位置是：

- `d4` 的注意力和 `MSCB4`：`lib/decoders.py:956-965`；
- `d3` 的上采样、skip 门控、相加和 `MSCB3`：`lib/decoders.py:967-989`；
- `d2` 的对应流程：`lib/decoders.py:991-1012`；
- `d1` 的对应流程：`lib/decoders.py:1014-1035`；
- 最终返回 `[d4, d3, d2, d1]`：`lib/decoders.py:1037-1040`。

所以 DGEMCAD 最合理的插入边界，就是保留 EMCAD 的 EUCB、LGAG、CAB、SAB 和四级解码顺序，只替换每一级的固定聚合 MSCB。

## 3. DGEMCAD 实际增加了什么

### 3.1 四个原始 MSCB 被替换

`DGEMCADNet` 继承 `EMCADNet`，然后构造 `DisagreementGuidedEMCAD` 并替换 decoder（见 `DG_EMCAD_reference/lib/dg_emcad.py:303-365`）。

新的 decoder 仍然创建四个对应层级：

```python
self.mscb4 = DisagreementGuidedMSCB(c4, c4, ...)
self.mscb3 = DisagreementGuidedMSCB(c3, c3, ...)
self.mscb2 = DisagreementGuidedMSCB(c2, c2, ...)
self.mscb1 = DisagreementGuidedMSCB(c1, c1, ...)
```

代码见 `DG_EMCAD_reference/lib/dg_emcad.py:249-252`。

这说明它不是在 EMCAD 后面额外串接一个完全独立的网络，而是把原来的四级静态 MSCB 聚合替换为四级同位置的自适应 MSCB。论文中可以把它描述为一个统一模块在四个 decoder stage 的实例化。

### 3.2 1x1、3x3、5x5 分支仍然存在

`DisagreementGuidedMSCB` 中的 `self.dwconvs` 仍然按照 `kernel_sizes` 创建 depth-wise 分支（见 `DG_EMCAD_reference/lib/dg_emcad.py:80-102`）。默认配置为 `[1, 3, 5]`。

因此 DGEMCAD 并没有删除 EMCAD 的多尺度信息，而是把原来的：

```text
branch_1 + branch_3 + branch_5
```

改为：

```text
w_1(x,y) * branch_1
 + w_3(x,y) * branch_3
 + w_5(x,y) * branch_5
```

其中权重由 softmax 归一化，且 `w_1 + w_3 + w_5 = 1`。

实际代码在 `DG_EMCAD_reference/lib/dg_emcad.py:193-203`：

```python
expanded = self.pconv1(feature)
branch_outputs = [branch(expanded) for branch in self.dwconvs]
mixed = sum(
    weights[:, index : index + 1] * branch_output
    for index, branch_output in enumerate(branch_outputs)
)
mixed = mixed * float(self.branch_count)
output = self.shortcut(feature) + self.pconv2(mixed)
```

`mixed * branch_count` 是为了让 `equal` 模式下的均匀权重，在数值尺度上接近原始分支直接求和。它不是新的尺度选择机制，只是为了构造可比较的固定路由消融。

## 4. “自适应”是如何产生的

### 4.1 当前特征产生路由信息

在每个 `DisagreementGuidedMSCB` 中，首先由当前 decoder feature 生成辅助预测：

```python
routing_logits = self.routing_prediction(feature)
```

代码见 `DG_EMCAD_reference/lib/dg_emcad.py:113-114` 和 `184-191`。

对于 Synapse，`num_classes=9`，所以某一级特征例如 `[B, C, H, W]` 会经过 1x1 卷积得到：

```text
[B, C, H, W] -> [B, 9, H, W]
```

这个预测不是最终分割输出头，而是路由器用来估计当前层级预测状态的辅助 logits。

### 4.2 熵表示当前层级的不确定性

`_probabilities()` 将二分类 logits 转为 sigmoid 概率，将多分类 logits 转为 softmax 概率（见 `DG_EMCAD_reference/lib/dg_emcad.py:17-22`）。

然后 `_normalized_entropy()` 计算每个像素的归一化熵（见 `DG_EMCAD_reference/lib/dg_emcad.py:25-28`）：

```text
预测概率越平均 -> 熵越高 -> 当前层级越不确定
某一类别概率越集中 -> 熵越低 -> 当前层级越确定
```

这只是一个合理的“不确定性代理”，不是由额外标注直接测得的不确定性。

### 4.3 相邻解码层之间用 JS 差异表示跨尺度不一致

当当前层有上一层更深的 logits 时，代码先将更深层 logits 插值到当前层的空间尺寸，再计算两个概率分布的 Jensen-Shannon 差异：

```python
deeper_logits = F.interpolate(
    deeper_logits,
    size=current_logits.shape[-2:],
    mode="bilinear",
    align_corners=False,
)
deeper_probability = _probabilities(deeper_logits)
disagreement = _normalized_js(current_probability, deeper_probability)
```

见 `DG_EMCAD_reference/lib/dg_emcad.py:140-159`。

四级 decoder 中的实际传递关系是：

```text
q4 -> q3 的相邻尺度差异
q3 -> q2 的相邻尺度差异
q2 -> q1 的相邻尺度差异
```

见 `DG_EMCAD_reference/lib/dg_emcad.py:277-293`。最深层 d4 没有更深层预测，所以只使用自身熵。

当前代码计算的融合不确定性是：

```text
u_i = (entropy(q_i) + lambda * JS(q_i, up(q_{i+1}))) / (1 + lambda)
```

所以论文中应写“当前层预测熵与相邻解码层预测分布差异”，不要写成“直接计算相邻特征图差异”。

### 4.4 不确定性和 feature 一起预测分支权重

在 `router_mode='disagreement'` 时，当前 feature 先经过 1x1 降维、GroupNorm 和 ReLU；然后拼接一张单通道 uncertainty map：

```python
router_feature = self.feature_projection(feature)
router_feature = torch.cat((router_feature, uncertainty), dim=1)
router_logits = self.router(router_feature)
return torch.softmax(router_logits / self.router_temperature, dim=1)
```

见 `DG_EMCAD_reference/lib/dg_emcad.py:161-182`。

对于默认三分支，最终权重张量形状为：

```text
feature       : [B, C, H, W]
uncertainty   : [B, 1, H, W]
router_feature: [B, hidden+1, H, W]
router_logits : [B, 3, H, W]
weights       : [B, 3, H, W]
```

第 2 维的 3 个通道分别对应 `kernel_sizes` 中的 1x1、3x3、5x5 分支。权重在空间位置上可以不同，因此小目标、边界、纹理复杂区和均匀区域可以得到不同的尺度组合。

## 5. 它针对 EMCAD 的问题是否合理

### 合理之处

1. EMCAD 原本已经有多尺度分支，因此从“固定聚合”改为“输入相关的动态聚合”是针对原结构的直接改动，而不是另起炉灶。
2. decoder 各级特征尺寸和通道保持原 EMCAD 契约。DG decoder 要求 `channels=[C4,C3,C2,C1]`，默认 PVTv2-B2 为 `[512,320,128,64]`（见 `DG_EMCAD_reference/lib/dg_emcad.py:225-237`）。
3. EUCB、LGAG 和 skip connection 的层级顺序保持一致（见 `DG_EMCAD_reference/lib/dg_emcad.py:280-292`），因此可以把主要变量归因于 MSCB 聚合方式。
4. DG 网络仍输出 `[d4,d3,d2,d1]`，并经过四个输出头得到 4 个 `[B, num_classes, H, W]` logits；`return_aux=True` 时只额外返回路由辅助信息（见 `DG_EMCAD_reference/lib/dg_emcad.py:367-392`）。这与现有 trainer 的 `unpack_output()` 兼容。

### 不能直接宣称的地方

1. “高不确定性一定应该使用 5x5”只是设计假设。代码通过正则项鼓励尺度权重的期望值与 uncertainty 对齐，但真实数据是否如此必须看路由权重统计和性能。
2. 该模块不是严格的 hard selection。即使某处 5x5 权重最高，1x1 和 3x3 通常仍然参与输出。
3. 路由器的 uncertainty 依赖辅助预测 q_i。q_i 不是独立的真实不确定性标签，而是通过分割标签监督的辅助分类头。
4. 当前训练并非只改变网络结构。`trainer.py:350-389` 在 DG 模式下还加入 `routing_prediction_loss` 和 `routing_regularization`；默认 `dg_route_aux_weight=0.20`、`dg_route_reg_weight=0.05`，参数定义在 `train_synapse.py:137-151`。
5. 因此如果 DGEMCAD 提升，必须进一步做：结构路由但关闭辅助损失、只开辅助损失、以及完整 DG 的消融，才能知道收益来自动态结构还是额外训练约束。

## 6. 它到底是一个还是多个创新点

论文主贡献建议只写一个：

> 面向 EMCAD 固定多尺度聚合问题的预测差异引导自适应多尺度解码模块。

模块内部的组成是实现机制，不宜拆成多个独立创新点：

- 1x1/3x3/5x5 分支：继承 EMCAD 的多尺度基础，不是本次新提出；
- softmax router：实现动态分支混合的核心部件；
- entropy：当前层不确定性代理；
- adjacent-scale JS：跨层预测差异代理；
- routing auxiliary loss：让 q_i 具有分割语义；
- routing regularization：防止路由塌缩并约束尺度顺序。

这些部件共同服务于一个机制。可是实验上必须把结构因素和训练损失因素拆开，否则论文无法证明“自适应多尺度结构”本身有效。

## 7. 当前代码的张量对齐依据

以 PVTv2-B2、输入 `224x224`、Synapse 9 类为例：

```text
encoder:
  x1 [B,  64, 56, 56]
  x2 [B, 128, 28, 28]
  x3 [B, 320, 14, 14]
  x4 [B, 512,  7,  7]

decoder:
  d4 [B, 512,  7,  7]
  d3 [B, 320, 14, 14]
  d2 [B, 128, 28, 28]
  d1 [B,  64, 56, 56]

auxiliary routing logits:
  q4 [B, 9,  7,  7]
  q3 [B, 9, 14, 14]
  q2 [B, 9, 28, 28]
  q1 [B, 9, 56, 56]

router weights:
  a4 [B, 3,  7,  7]
  a3 [B, 3, 14, 14]
  a2 [B, 3, 28, 28]
  a1 [B, 3, 56, 56]

four output heads after interpolation:
  p4,p3,p2,p1 all [B, 9, 224, 224]
```

通道对应关系由 `DGEMCADNet` 从四个输出头反向读取（见 `DG_EMCAD_reference/lib/dg_emcad.py:346-351`），因此没有手写另一套通道表。四级 decoder 的输入输出分别是 `c4->c4`、`c3->c3`、`c2->c2`、`c1->c1`，不会改变层级通道。

## 8. 应如何验证“它确实解决了这个问题”

至少需要以下实验，不能只看最终 Dice：

| 实验 | 目的 |
|---|---|
| EMCAD baseline | 确认原始固定聚合基线 |
| DGEMCAD + `router_mode=equal` | 检验仅替换实现和初始化是否造成变化；应接近固定等权聚合 |
| `router_mode=feature` | 检验仅使用当前 feature 路由的作用 |
| `router_mode=disagreement` | 检验加入熵和相邻预测 JS 后是否进一步有效 |
| disagreement 结构但 `dg_route_aux_weight=0`、`dg_route_reg_weight=0` | 区分结构收益和路由辅助损失收益 |
| 完整 DGEMCAD | 最终模型 |

每个设置至少固定相同数据划分、输入尺寸、训练轮数、学习率、batch size、监督方式和 seed；最终报告验证集选出的 best checkpoint 在测试集上的 Dice、HD95、IoU/Jaccard、参数量、FLOPs、推理时间和峰值显存。

还应保存或统计：

- 1x1、3x3、5x5 三个路由权重的平均值和分位数；
- 边界区域与非边界区域的路由权重；
- 小器官和大器官的路由权重；
- 相邻层 JS 差异与最终分割错误的相关性。

如果高 uncertainty 区域没有更偏向大感受野，或者 `disagreement` 与 `feature` 没有稳定差异，那么“差异引导尺度选择”这个解释就没有得到实验支持，即使某次 Dice 提升，也不能直接把提升归因于该机制。

## 9. 最准确的论文表述

在实验尚未完成前，建议使用：

> 我们针对 EMCAD 中多尺度深度卷积分支采用固定相加聚合、缺少位置自适应尺度分配的问题，设计了一个预测差异引导的自适应多尺度解码模块。该模块在保留 EMCAD 原有四级解码路径、注意力门控和多尺度分支的基础上，根据当前层预测熵与相邻解码层预测分布的 Jensen-Shannon 差异，生成逐像素的尺度路由权重，并对 1x1、3x3、5x5 分支进行软融合。

这段话描述的是当前代码真正做的事情。不要在没有额外代码和实验的情况下写成“直接根据特征不确定性选择唯一卷积核”或“已经证明能修复小目标和边界”。

