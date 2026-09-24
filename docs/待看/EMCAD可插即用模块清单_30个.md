# 面向 EMCAD-Synapse Dice 提升的 30 个可插拔模块

> 目标：以当前 EMCAD + PVTv2-B2 + Synapse、224×224 切片模型为基础，逐个验证能否提高平均 Dice，尤其关注胆囊、胰腺、胃等低分器官。
>
> 说明：这些模块不是“保证涨分”的清单，而是具有公开论文背景、可以形成受控消融的候选组件。建议一次只改一个模块，固定病例级划分、随机种子、训练轮数、损失和评价脚本。当前仓库中 `MSCB/MSDC/CAB/SAB/LGAG/EUCB` 位于 `lib/decoders.py`，模型输出为四级 logits；因此优先从解码器和 skip 融合处试验。

## 一、优先级与实验原则

### 推荐第一批

优先尝试：SE、ECA、CBAM、Coordinate Attention、通道/空间动态尺度融合、ASPP、Pyramid Pooling、边界引导分支、深监督改进、Attention Gate 替换。

这些模块通常能在二维特征图上接入，改动集中在 decoder，比较适合当前 EMCAD。

### 暂缓第一批

Swin/Transformer 全量替换、3D 卷积、跨切片模块、Deformable Conv、复杂状态空间模块需要改输入或显存配置，不宜和普通 decoder 模块混在第一轮比较。

### 统一验收指标

- mean Dice、8 个器官 Dice；
- HD95、ASD；
- 胆囊、胰腺、胃的 per-case Dice；
- 参数量、FLOPs、显存和单病例推理时间；
- 至少 3 个随机种子，或报告多次运行均值±标准差。

## 二、30 个候选模块

表中“接入位置”按当前 EMCAD 的结构描述：`d4→d3→d2→d1`，每一级包含 CAB、SAB、MSCB，并通过 LGAG 过滤 encoder skip。

| 编号 | 模块 | 主要作用 | 推荐接入 EMCAD 的位置 | 接入判断 | 代表论文/代码 |
|---:|---|---|---|---|---|
| 1 | **SE（Squeeze-and-Excitation）** | 全局池化后学习通道权重，抑制无关语义通道 | 替换或并联 CAB | **最容易**；输入输出尺寸不变 | 论文：[CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html)；代码：[作者实现](https://github.com/hujie-frank/SENet) |
| 2 | **ECA-Net** | 用一维局部交互避免 SE 的降维损失，轻量通道注意力 | 替换 CAB | **最容易**；适合小器官通道筛选 | 论文：[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html)；代码：[GitHub](https://github.com/BangguWu/ECANet) |
| 3 | **CBAM** | 顺序执行通道注意力和空间注意力，分别回答“看什么、看哪里” | 替换 CAB+SAB，或只插在 MSCB 前 | **容易**；需防止与现有 CAB/SAB 重复堆叠 | 论文：[ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html)；代码：[Jongchan/attention-module](https://github.com/Jongchan/attention-module) |
| 4 | **Coordinate Attention** | 将二维空间位置信息编码进通道权重，保留长条形位置信息 | 替换 CAB，或放在 skip 融合后 | **容易**；对细长、边界模糊器官有针对性 | 论文：[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Hou_Coordinate_Attention_for_Efficient_Mobile_Network_Design_CVPR_2021_paper.html)；代码：[houqb/CoordAttention](https://github.com/houqb/CoordAttention) |
| 5 | **SimAM** | 无参数能量函数为每个神经元估计重要性 | 放在每一级 MSCB 输出后 | **最容易**；几乎不增加参数 | 论文：[ICML 2021](https://proceedings.mlr.press/v139/yang21o.html)；代码：[GitHub](https://github.com/ZjjConan/SimAM) |
| 6 | **GCNet（Global Context Block）** | 以全局上下文生成通道校正，补充局部卷积的视野 | d4 或 d3 的 MSCB 后 | **容易**；深层特征更合适 | 论文：[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Cao_GCNet_Non-Local_Networks_Meet_Squeeze-Excitation_Networks_and_Beyond_ICCV_2019_paper.html)；代码：[mmcv context blocks](https://github.com/open-mmlab/mmcv) |
| 7 | **Non-local Block** | 建立任意远距离像素之间的相似性，捕捉全局依赖 | 只放 d4/d3，不能四级全放 | **可接入但显存风险高** | 论文：[CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.html)；代码：[官方仓库](https://github.com/facebookresearch/video-nonlocal-net) |
| 8 | **Axial Attention** | 将二维/三维注意力拆成轴向注意力，降低全局计算量 | d3 或 d4，替换部分 SAB | **可接入**；比 Non-local 更适合高分辨率 | 论文：[ICLR 2020](https://openreview.net/forum?id=H1e5GJBtDr)；代码：[lucidrains](https://github.com/lucidrains/axial-attention) |
| 9 | **Criss-Cross Attention（CCA）** | 沿行列聚合远距离上下文，适合结构连续目标 | d3/d2 的空间注意力位置 | **可接入**；需控制迭代次数和显存 | 论文：[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Huang_Criss-Cross_Attention_for_Semantic_Segmentation_ICCV_2019_paper.html)；代码：[speedinghzl/CCNet](https://github.com/speedinghzl/CCNet) |
| 10 | **ASPP** | 多个空洞率并行卷积，扩大感受野而不立即下采样 | d4 或 d3，替换一个 MSCB | **容易**；空洞率需按 7×7/14×14 特征图调整 | 论文：[ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.html)；代码：[PyTorch DeepLab](https://github.com/VainF/DeepLabV3Plus-Pytorch) |
| 11 | **DenseASPP** | 串联不同空洞率，使感受野逐步扩大并减少栅格效应 | d4/d3 深层特征 | **可接入但计算中等** | 论文：[CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.html)；代码：[GitHub 搜索实现](https://github.com/search?q=DenseASPP&type=repositories) |
| 12 | **PSP/Pyramid Pooling Module** | 多尺度池化后上采样并拼接，增强全局场景/器官布局 | d4 后、EUCB3 前 | **容易**；需 1×1 投影回原通道 | 论文：[CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html)；代码：[open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation) |
| 13 | **RFB（Receptive Field Block）** | 用不同膨胀率和分支模拟多尺度感受野，增强形状上下文 | 替换 MSCB 或放 d3 后 | **容易**；与 MSDC 有功能重叠，必须做参数匹配 | 论文：[ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Liu_Receptive_Field_Block_Nets_ECCV_2018_paper.html)；代码：[RFNet](https://github.com/ruinmessi/RFBNet) |
| 14 | **SKNet（Selective Kernel）** | 根据输入动态选择不同卷积核尺度 | 直接替换 MSDC 的固定相加 | **高度推荐研究**；是“自适应 MSDC”的直接候选 | 论文：[CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html)；代码：[impl](https://github.com/developer0hye/SKNet-pytorch) |
| 15 | **Selective Feature Fusion（SFF）** | 对两路或多路特征计算空间/通道选择权重后融合 | LGAG 输出与 decoder 主路相加处 | **容易**；替换当前简单 `d + skip` | 论文：[常用于 FPN/检测的选择性融合思想](https://arxiv.org/abs/1911.09070)；代码：[Detectron2/FPN 参考](https://github.com/facebookresearch/detectron2) |
| 16 | **ASFF（Adaptively Spatial Feature Fusion）** | 每个空间位置自适应决定不同尺度特征的贡献 | 跨级 decoder 特征融合；不建议首轮放四级 | **可接入但需改接口** | 论文：[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6810)；代码：[GOATmessi8/ASFF](https://github.com/GOATmessi8/ASFF) |
| 17 | **BiFPN 加权融合** | 学习不同路径的非负归一化权重，改善多尺度特征融合 | encoder skip 与 decoder 特征融合 | **容易到中等**；需要统一通道和尺度 | 论文：[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html)；代码：[google/automl](https://github.com/google/automl/tree/master/efficientdet) |
| 18 | **FPN/UNet++ Nested Skip** | 重新组织跨层 skip，缩短语义到高分辨率的路径 | 替换现有单次 LGAG skip | **中等**；会改变 decoder 拓扑 | 论文：[UNet++ DLMIA 2018](https://arxiv.org/abs/1807.10165)；代码：[4uiiurz1/pytorch-nested-unet](https://github.com/4uiiurz1/pytorch-nested-unet) |
| 19 | **Attention U-Net Gate** | 用 decoder gate 筛选 encoder skip 的无关区域 | 替换 LGAG 做基线对照 | **容易**；当前已经有 LGAG，创新增量有限 | 论文：[MIDL 2018](https://arxiv.org/abs/1804.03999)；代码：[ozan-oktay/Attention-Gated-Networks](https://github.com/ozan-oktay/Attention-Gated-Networks) |
| 20 | **Gated Skip Connection** | 让网络学习 skip 信息通过比例，抑制低层噪声 | 当前 `d = d + x_skip` 位置 | **最适合当前代码**；接口简单 | 论文：[Gated-SCNN，ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Takikawa_Gated_Spatial_CNNs_for_Semantic_Segmentation_ICCV_2019_paper.html)；代码：[nv-tlabs/GSCNN](https://github.com/nv-tlabs/GSCNN) |
| 21 | **Boundary Refinement / Boundary Head** | 单独预测器官边界，强化轮廓监督，减少 Dice 对边界误差的惩罚 | d1 输出头旁增加边界支路 | **推荐**；需要新增边界标签和辅助损失 | 论文：[BDCN，CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Bi-Directional_Cascade_Network_for_Perceptual_Edge_Detection_CVPR_2019_paper.html)；代码：[官方相关实现](https://github.com/poolio/biseced) |
| 22 | **Gated-SCNN Shape Stream** | 使用独立形状流与语义流交互，显式建模边界 | d2/d1 与 segmentation head | **中等**；适合胃、胰腺边界研究 | 论文：[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Takikawa_Gated_Spatial_CNNs_for_Semantic_Segmentation_ICCV_2019_paper.html)；代码：[nv-tlabs/GSCNN](https://github.com/nv-tlabs/GSCNN) |
| 23 | **Deep Supervision（UNet 侧输出）** | 对 d4/d3/d2/d1 同时施加分割损失，改善梯度和多尺度预测 | 当前四个 out heads 的 loss 组合 | **已有基础，适合重新设计权重** | 论文：[Deeply-Supervised Nets，ICML 2015](https://proceedings.mlr.press/v38/lee15a.html)；代码：[pytorch-UNet](https://github.com/milesial/Pytorch-UNet) |
| 24 | **Lovász-Softmax Loss** | 直接优化 IoU/Jaccard 的排序近似，改善区域重叠 | 替换或加入现有 CE+Dice | **不改网络**；应作为损失消融单独测试 | 论文：[CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_Lovasz-Softmax_Loss_CVPR_2018_paper.html)；代码：[bermanmaxim/LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax) |
| 25 | **Tversky / Focal-Tversky Loss** | 调节 FP/FN 权重，针对小器官漏检和类别不均衡 | 训练 loss，不改 decoder | **最容易验证**；重点观察胆囊/胰腺 | 论文：[Focal Tversky，arXiv](https://arxiv.org/abs/1810.07842)；代码：[nabsabraham/focal-tversky-unet](https://github.com/nabsabraham/focal-tversky-unet) |
| 26 | **Boundary Loss** | 用距离变换约束预测边界与真实边界距离 | 与 CE/Dice 并联 | **容易**；需要生成距离图 | 论文：[MIDL 2019](https://proceedings.mlr.press/v102/kervadec19a.html)；代码：[官方作者页](https://github.com/LIVIAETS/boundary-loss) |
| 27 | **Focal modulation** | 用内容自适应的上下文调制替代显式 self-attention | d3/d2 的空间建模位置 | **中等**；需保持 NCHW 接口 | 论文：[NeurIPS 2022](https://openreview.net/forum?id=2-z9Z1CNRB8)；代码：[microsoft/FocalNet](https://github.com/microsoft/FocalNet) |
| 28 | **ConvNeXt Block** | 大核深度卷积、LayerNorm 和 point-wise MLP，增强局部表示 | 替换 d3/d2 的一个 MSCB | **中等**；需处理 BN/LN 与预训练差异 | 论文：[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)；代码：[facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt) |
| 29 | **Deformable Convolution v2** | 学习采样偏移，适应器官不规则边界和形变 | d2/d1 的 MSCB 中替换 3×3 分支 | **可接入但编译/显存风险中等** | 论文：[CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Deformable_ConvNets_V2_More_Deformable_Better_Results_CVPR_2019_paper.html)；代码：[torchvision.ops](https://pytorch.org/vision/stable/ops.html) |
| 30 | **2.5D Inter-slice Fusion（MOSformer/AFTer-UNet 思路）** | 引入前后 CT 切片，改善 z 轴连续性和小器官定位 | 需要从 Dataset 到 encoder/decoder 改接口 | **长期候选，不是普通即插即用** | MOSformer：[arXiv 2401.11856](https://arxiv.org/abs/2401.11856)；AFTer-UNet：[arXiv 2110.10403](https://arxiv.org/abs/2110.10403) |

## 三、和当前 EMCAD 的具体对应关系

### A. 替换 MSDC 的候选

最直接的比较组是：

```text
原始 MSDC：1×1 + 3×3 + 5×5，逐元素相加
RFB：不同膨胀率分支
SK：输入依赖的尺度权重
动态 MSDC：α1·f1 + α3·f3 + α5·f5
Deformable 3×3：学习采样位置
```

这一组要严格控制中间通道数，否则 Dice 提升可能只是参数量增加。

### B. 替换 CAB/SAB 的候选

```text
CAB/SAB → SE
CAB/SAB → ECA
CAB/SAB → CBAM
CAB → Coordinate Attention
CAB/SAB → SimAM
```

注意当前 EMCAD 已经有通道和空间注意力，直接再叠加 CBAM 很容易重复计算；更好的实验是“替换”而不是无条件串联。

### C. 替换 skip 融合的候选

当前主路与筛选后的 skip 主要是逐元素相加。优先比较：

```text
d + skip
d + gate(d, skip)·skip
BiFPN weighted fusion
SFF/ASFF
Attention U-Net gate
```

这类模块可能比继续增加卷积核更直接地改善 encoder 的低层细节利用。

### D. 不改主干、只改训练目标

Lovász-Softmax、Focal-Tversky、Boundary Loss 和深监督权重都不要求改变 PVTv2 或 EMCAD 的特征接口，适合先做低风险对照。但它们不应被和结构模块放在同一张“网络模块提升”表里，论文中需要单独归类为优化目标改进。

## 四、建议的逐个实验顺序

### 第一轮：低风险、单模块

1. ECA 替换 CAB；
2. Coordinate Attention 替换 CAB；
3. SimAM 替换 CAB/SAB；
4. SK 替换 MSDC 固定相加；
5. BiFPN 或 gated skip 替换 `d+skip`；
6. ASPP 替换 d4 的 MSCB；
7. Boundary Loss；
8. Focal-Tversky；
9. 重新设置四级深监督权重。

### 第二轮：针对低分器官

1. Boundary Head；
2. Gated-SCNN shape stream；
3. d2/d1 使用 Deformable 3×3；
4. 胃/胰腺区域使用动态尺度融合；
5. d3/d4 使用 PSP 或 GCNet。

### 第三轮：需要结构性改动

1. Axial Attention；
2. Criss-Cross Attention；
3. ConvNeXt block；
4. 2.5D 3 张或 5 张相邻切片；
5. MOSformer/AFTer-UNet 式层间融合。

## 五、重要限制

1. 公开论文中的提升幅度不能直接迁移到 EMCAD；数据划分、预处理、输入尺寸、损失和预训练权重都会改变结果。
2. 当前 Synapse 训练是单张二维切片；除第 30 项外，其余模块主要增强 H×W 平面特征，不能解决 z 轴上下文缺失。
3. 当前训练脚本存在使用 `test_vol` 参与每轮模型选择的行为。所有模块消融应改用独立 validation，最终 test 只评估一次，否则会高估 Dice。
4. 先做参数量匹配和单变量实验。若同时更换 MSDC、损失、数据增强和输入尺寸，无法判断 Dice 提升来自哪个因素。
5. 目标应设为“稳定提升平均 Dice 并改善目标器官”，而不是追求某一次运行的最高值。

## 六、推荐记录表

| 实验 ID | 模块 | 插入位置 | 参数量 | seed | mean Dice | Aorta | GB | KL | KR | Liver | PC | SP | Stomach | HD95 | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E00 | 原始 EMCAD | — |  |  | 83.63 |  |  |  |  |  |  |  |  |  | baseline |
| E01 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

