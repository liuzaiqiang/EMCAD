# CCF-B/C 医学图像分割会议创新点与 EMCAD 发论文路线

更新时间：2026-08-29

## 先给结论

你现在缺的不是“再看一百篇论文”，而是没有建立下面这条判断链：

```text
真实且稳定的失败现象
→ 可证伪的研究假设
→ 一个解释机制
→ 公平的控制实验
→ 对正负结果都能解释的结论
```

医学图像分割论文不是“给 U-Net 加一个模块，Dice 提高一点”这么简单。近两年 CCF-B/C 相关会议中，真正容易被接受的工作，通常至少满足以下三点：

1. 针对一个具体困难，而不是笼统地追求更高 Dice。
2. 方法设计与困难之间有明确因果联系，能够解释为什么有效。
3. 实验不仅报告最好分数，还验证稳定性、泛化性、效率或临床/结构约束。

## 1. 检索范围和分区说明

本次检索重点覆盖 2024-2025 年公开论文集或 DBLP 记录中的：

- MICCAI 主会及相关公开论文集；
- IEEE ISBI；
- IEEE ICIP；
- IEEE ICASSP；
- BMVC；
- ACCV；
- AAAI（作为更高竞争强度的参照）。

CCF 目录会随版本更新，且不同学校对会议、workshop、challenge proceedings 的认定可能不同。下文不把“某篇论文出现在会议论文集”直接等同于“你所在学校一定认定为 CCF-B 或 CCF-C”。投稿前必须核对中国计算机学会最新目录和你们学院的认定文件。

医学图像分割的主阵地通常是 MICCAI、ISBI；ICIP 和 ICASSP 更偏通用图像/信号处理，BMVC 与 ACCV 更偏计算机视觉。AAAI 是人工智能综合会议，常作为高强度参考，不应简单当作普通 B/C 会议对待。

## 2. 近两年论文到底在创新什么

下面的“创新点”是根据论文题目、会议记录和公开摘要线索提炼出的研究主线，不等于完整复现了每篇论文的全部技术细节。读论文时仍要打开原文，确认损失、数据集、训练协议和消融设计。

### 2.1 少标签、弱标签和半监督学习

这是当前最稳定的研究问题之一，因为医学标注昂贵，而且问题本身比“换一个注意力模块”更有科学性。

| 会议 | 论文 | 可确认的创新主线 |
|---|---|---|
| MICCAI 2024 | Pair Shuffle Consistency for Semi-supervised Medical Image Segmentation | 设计配对/打乱一致性约束，让有标签和无标签样本之间产生更可靠的监督。DOI：[10.1007/978-3-031-72111-3_46](https://doi.org/10.1007/978-3-031-72111-3_46) |
| MICCAI 2024 | FRCNet: Frequency and Region Consistency for Semi-supervised Medical Image Segmentation | 同时利用频域和区域级一致性，针对无标签数据中的结构与外观变化。DOI：[10.1007/978-3-031-72111-3_29](https://doi.org/10.1007/978-3-031-72111-3_29) |
| MICCAI 2024 | Decoupled Training for Semi-supervised Medical Image Segmentation with Worst-Case-Aware Learning | 将半监督训练过程解耦，并显式关注困难或最坏情况样本。DOI：[10.1007/978-3-031-72390-2_5](https://doi.org/10.1007/978-3-031-72390-2_5) |
| ISBI 2024 | ASLSEG: Adapting Sam in the Loop for Semi-Supervised Liver Tumor Segmentation | 把 SAM 引入半监督循环，用基础模型帮助产生或修正伪标签。DOI：[10.1109/ISBI56570.2024.10635501](https://doi.org/10.1109/ISBI56570.2024.10635501) |
| ISBI 2024 | Semi-Supervised Medical Image Segmentation Via Dynamic Pseudo-Label Refinement | 伪标签不是一次生成，而是根据训练状态动态修正。DOI：[10.1109/ISBI56570.2024.10635316](https://doi.org/10.1109/ISBI56570.2024.10635316) |
| ISBI 2025 | MARIO: A Mixed Annotation Framework for Polyp Segmentation | 混合使用不同成本和精度的标注形式，研究标注效率与分割质量的折中。DOI：[10.1109/ISBI60581.2025.10981038](https://doi.org/10.1109/ISBI60581.2025.10981038) |
| ICIP 2024 | Quadruple-Consistency Vision Transformer for Medical Image Segmentation with Limited Number of Sparse Annotations | 在稀疏标注条件下构造多重一致性，而不是假设所有像素标签都可用。DOI：[10.1109/ICIP51287.2024.10647711](https://doi.org/10.1109/ICIP51287.2024.10647711) |

这些论文的共同点不是“用了伪标签”四个字，而是回答了一个更清楚的问题：

> 在标注不完整或不可靠时，什么信息仍然可信，如何把它转化成稳定监督？

### 2.2 域泛化、域适应和跨设备鲁棒性

医学数据经常来自不同医院、设备、协议和人群。跨域问题比单数据集刷分更容易形成科学问题。

| 会议 | 论文 | 可确认的创新主线 |
|---|---|---|
| MICCAI 2024 | LUCIDA: Low-Dose Universal-Tissue CT Image Domain Adaptation for Medical Segmentation | 针对低剂量与组织差异进行 CT 域适应，关注成像分布变化而非单纯网络深度。DOI：[10.1007/978-3-031-72111-3_37](https://doi.org/10.1007/978-3-031-72111-3_37) |
| MICCAI 2024 | Quest for Clone: Test-Time Domain Adaptation for Medical Image Segmentation by Searching the Closest Clone in Latent Space | 在测试阶段寻找潜在空间中更接近的“克隆”表示，减少目标域无标签适应的代价。DOI：[10.1007/978-3-031-72111-3_52](https://doi.org/10.1007/978-3-031-72111-3_52) |
| ISBI 2024 | Improving Test-Time Adaptation For Histopathology Image Segmentation: Gradient-To-Parameter Ratio Guided Feature Alignment | 使用梯度与参数关系指导测试时特征对齐，避免无约束更新。DOI：[10.1109/ISBI56570.2024.10635451](https://doi.org/10.1109/ISBI56570.2024.10635451) |
| ISBI 2024 | Enhanced Structure Preservation and Multi-View Approach in Unsupervised Domain Adaptation for Optic Disc and Cup Segmentation | 结合结构保持和多视角信息处理无监督域适应。DOI：[10.1109/ISBI56570.2024.10635127](https://doi.org/10.1109/ISBI56570.2024.10635127) |
| ICASSP 2024 | MMS: Morphology-Mixup Stylized Data Generation for Single Domain Generalization in Medical Image Segmentation | 通过形态混合和风格变化构造更丰富的训练分布，提升单源域泛化。DOI：[10.1109/ICASSP48485.2024.10448305](https://doi.org/10.1109/ICASSP48485.2024.10448305) |
| ICASSP 2024 | IRLSG: Invariant Representation Learning for Single-Domain Generalization in Medical Image Segmentation | 学习对域变化更不敏感的表示。DOI：[10.1109/ICASSP48485.2024.10446700](https://doi.org/10.1109/ICASSP48485.2024.10446700) |
| ISBI 2025 | Visual Prompting Unsupervised Domain Adaptation for Medical Image Segmentation | 用视觉提示改善无监督域适应中的目标域定位。DOI：[10.1109/ISBI60581.2025.10980974](https://doi.org/10.1109/ISBI60581.2025.10980974) |

这类论文的核心不是“再加一个 domain loss”，而是把域差异具体化：低剂量噪声、染色风格、设备差异、人口差异、模态缺失，分别需要不同机制和验证方式。

### 2.3 SAM、基础模型、语言和扩散模型

基础模型是近两年的明显趋势，但单纯“把 SAM 接进 U-Net”已经不够。审稿人会追问：为什么通用基础模型在这个医学任务上失败？你的适配机制解决了什么？

| 会议 | 论文 | 可确认的创新主线 |
|---|---|---|
| MICCAI 2024 | DeSAM: Decoupled Segment Anything Model for Generalizable Medical Image Segmentation | 将 SAM 的不同功能解耦，目标是提高医学场景中的泛化能力。DOI：[10.1007/978-3-031-72390-2_48](https://doi.org/10.1007/978-3-031-72390-2_48) |
| MICCAI 2024 | SAMU: An Efficient and Promptable Foundation Model for Medical Image Segmentation | 研究更高效、可提示的医学分割基础模型。DOI：[10.1007/978-3-031-73471-7_14](https://doi.org/10.1007/978-3-031-73471-7_14) |
| MICCAI 2024 | Enhancing Label-Efficient Medical Image Segmentation with Text-Guided Diffusion Models | 用文本引导扩散模型帮助少标签分割。DOI：[10.1007/978-3-031-72111-3_24](https://doi.org/10.1007/978-3-031-72111-3_24) |
| MICCAI 2024 | Common Vision-Language Attention for Text-Guided Medical Image Segmentation of Pneumonia | 将视觉语言信息用于肺炎分割中的注意力引导。DOI：[10.1007/978-3-031-72114-4_19](https://doi.org/10.1007/978-3-031-72114-4_19) |
| MICCAI 2024 | CausalCLIPSeg: Unlocking CLIP's Potential in Referring Medical Image Segmentation with Causal Intervention | 用因果干预减少视觉语言提示中的错误关联。DOI：[10.1007/978-3-031-72384-1_8](https://doi.org/10.1007/978-3-031-72384-1_8) |
| ISBI 2024 | Promise: Prompt-Driven 3D Medical Image Segmentation Using Pretrained Image Foundation Models | 研究预训练视觉基础模型在 3D 医学分割中的提示驱动方式。DOI：[10.1109/ISBI56570.2024.10635207](https://doi.org/10.1109/ISBI56570.2024.10635207) |
| ISBI 2025 | SAM3X: Efficient 3D-Aware Network for Medical Image Segmentation Using SAM | 针对 3D 空间信息改造 SAM，而不是直接逐张 2D 推理。DOI：[10.1109/ISBI60581.2025.10981074](https://doi.org/10.1109/ISBI60581.2025.10981074) |
| ICASSP 2025 | Self-Prompting Driven SAM2 for 3D Medical Image Segmentation | 让模型自动生成或更新提示，减少人工点框依赖。DOI：[10.1109/ICASSP49660.2025.10889344](https://doi.org/10.1109/ICASSP49660.2025.10889344) |

基础模型方向的最低科学要求是：

- 明确提示误差、模态差异、2D/3D 不一致或医学边界不确定性中的一个具体问题；
- 有基线比较：原始 SAM/SAM2、医学适配版本、传统 U-Net/nnU-Net；
- 报告不同提示类型、不同器官/病灶和失败案例；
- 不把基础模型的巨大参数量本身当成创新。

### 2.4 边界、拓扑、形状和几何约束

这是与你当前 EMCAD 最接近、也最容易重复的方向。边界注意力、边界损失、多尺度融合已经非常拥挤，必须从“边界为什么错”出发。

| 会议 | 论文 | 可确认的创新主线 |
|---|---|---|
| MICCAI 2024 | Topologically Faithful Multi-class Segmentation in Medical Images | 将拓扑保持作为多类分割目标，避免连接关系断裂或错误连接。DOI：[10.1007/978-3-031-72111-3_68](https://doi.org/10.1007/978-3-031-72111-3_68) |
| MICCAI 2024 | Differentiable Soft Morphological Filters for Medical Image Segmentation | 将形态学滤波做成可微模块，使形状先验能够进入端到端训练。DOI：[10.1007/978-3-031-72111-3_17](https://doi.org/10.1007/978-3-031-72111-3_17) |
| ISBI 2024 | Vascular Topology Rectification Network for Automated Retinal Artery/Vein Segmentation | 对血管拓扑进行显式纠正，针对细长结构断裂问题。DOI：[10.1109/ISBI56570.2024.10635229](https://doi.org/10.1109/ISBI56570.2024.10635229) |
| ISBI 2024 | TopoUT: Enhancing Cell Segmentation Through Efficient Topological Regularization | 用更高效的拓扑正则约束细胞实例结构。DOI：[10.1109/ISBI56570.2024.10635428](https://doi.org/10.1109/ISBI56570.2024.10635428) |
| ISBI 2024 | BCFNET: Boundary-Guided Semantic Cross Fusion for Polyp Segmentation | 以边界引导语义特征交叉融合，解决息肉边界模糊。DOI：[10.1109/ISBI56570.2024.10635517](https://doi.org/10.1109/ISBI56570.2024.10635517) |
| BMVC 2024 | Boundary Contrastive Learning for Label-Efficient Medical Image Segmentation | 在少标签条件下对边界表示进行对比学习。论文页：[BMVC 2024 / 685](https://bmvc2024.org/proceedings/685/) |
| BMVC 2024 | topK dice loss for medical image segmentation | 重新设计 Dice 损失的样本/区域聚合方式，强调困难区域。论文页：[BMVC 2024 / 897](https://bmvc2024.org/proceedings/897/) |
| ACCV 2024 | HDNeXt: Hybrid Dynamic MedNeXt with Level Set Regularization for Medical Image Segmentation | 将动态网络结构与 level-set 几何正则结合。DOI：[10.1007/978-981-96-0963-5_24](https://doi.org/10.1007/978-981-96-0963-5_24) |

边界方向要避免以下弱套路：

```text
Sobel/Canny 边缘图
→ 一个边界分支
→ 一个边界损失
→ Dice 提高 0.x%
```

这通常不足以构成有说服力的科学贡献，除非你进一步证明：

- 边界监督具体减少了哪类错误；
- 对 HD95、ASSD、边界 F-score 或拓扑指标有稳定影响；
- 提升不是来自参数量增加或额外后处理；
- 在多个目标尺度、多个数据集和不同随机种子下仍然成立。

### 2.5 新型骨干、Mamba、MLP、频域和大卷积核

这类工作数量很多，但“换骨干”本身已经很难成为充分创新。真正有价值的工作通常解释了长程依赖、频率信息、显存或 3D 计算瓶颈。

| 会议 | 论文 | 可确认的创新主线 |
|---|---|---|
| MICCAI 2024 | EM-Net: Efficient Channel and Frequency Learning with Mamba for 3D Medical Image Segmentation | 将 Mamba 用于通道与频率学习，针对 3D 分割的效率和全局建模。DOI：[10.1007/978-3-031-72114-4_26](https://doi.org/10.1007/978-3-031-72114-4_26) |
| MICCAI 2024 | EMF-Former: An Efficient and Memory-Friendly Transformer for Medical Image Segmentation | 以显存友好和计算效率为约束设计 Transformer。DOI：[10.1007/978-3-031-72111-3_22](https://doi.org/10.1007/978-3-031-72111-3_22) |
| MICCAI 2024 | MedContext: Learning Contextual Cues for Efficient Volumetric Medical Segmentation | 通过上下文线索改善体数据分割效率。DOI：[10.1007/978-3-031-72390-2_22](https://doi.org/10.1007/978-3-031-72390-2_22) |
| ISBI 2024 | CTranS: A Multi-Resolution Convolution-Transformer Network for Medical Image Segmentation | 结合卷积局部性与 Transformer 多分辨率建模。DOI：[10.1109/ISBI56570.2024.10635192](https://doi.org/10.1109/ISBI56570.2024.10635192) |
| ISBI 2024 | Medical Image Segmentation Using Directional Window Attention | 设计有方向性的窗口注意力，利用医学结构方向性。DOI：[10.1109/ISBI56570.2024.10635414](https://doi.org/10.1109/ISBI56570.2024.10635414) |
| ICASSP 2024 | LK-UNet: Large Kernel Design for 3D Medical Image Segmentation | 以大卷积核扩大 3D 感受野。DOI：[10.1109/ICASSP48485.2024.10446818](https://doi.org/10.1109/ICASSP48485.2024.10446818) |
| ACCV 2024 | LoG-VMamba: Local-Global Vision Mamba for Medical Image Segmentation | 将局部和全局状态空间建模结合。DOI：[10.1007/978-981-96-0901-7_14](https://doi.org/10.1007/978-981-96-0901-7_14) |
| ACCV 2024 | MS-UMLP: Medical Image Segmentation via Multi-Scale U-shape MLP-Mixer | 以多尺度 U 形结构组织 MLP-Mixer 特征。DOI：[10.1007/978-981-96-0972-7_19](https://doi.org/10.1007/978-981-96-0972-7_19) |
| ISBI 2025 | U-Net V2: Rethinking the Skip Connections of U-Net for Medical Image Segmentation | 重新研究 U-Net 跳跃连接，而不是机械堆叠注意力。DOI：[10.1109/ISBI60581.2025.10980742](https://doi.org/10.1109/ISBI60581.2025.10980742) |

这一类论文至少要报告：参数量、FLOPs、显存、推理时间、输入尺寸、2D/3D 设置，以及精度-效率折中曲线。

### 2.6 不确定性、校准、质量评估和失败识别

这是很多纯刷 Dice 的论文没有覆盖、但越来越重要的方向。

| 会议 | 论文 | 可确认的创新主线 |
|---|---|---|
| MICCAI 2024 | Holistic Consistency for Subject-Level Segmentation Quality Assessment in Medical Image Segmentation | 从病例整体层面评估分割质量，而不只看像素 Dice。DOI：[10.1007/978-3-031-73158-7_9](https://doi.org/10.1007/978-3-031-73158-7_9) |
| ISBI 2024 | From Registration Uncertainty to Segmentation Uncertainty | 将配准不确定性传递或关联到分割不确定性。DOI：[10.1109/ISBI56570.2024.10635251](https://doi.org/10.1109/ISBI56570.2024.10635251) |
| ISBI 2024 | Uncertainty Aware Segmentation Quality Assessment in Medical Images | 用不确定性预测分割质量或发现高风险病例。DOI：[10.1109/ISBI56570.2024.10635509](https://doi.org/10.1109/ISBI56570.2024.10635509) |
| ICASSP 2024 | CALSeg: Improving Calibration of Medical Image Segmentation Via Variational Label Smoothing | 研究概率校准，而不只追求分割重叠指标。DOI：[10.1109/ICASSP48485.2024.10446030](https://doi.org/10.1109/ICASSP48485.2024.10446030) |
| MICCAI 2025 | LEXU: Learning from Expert Disagreement for Single-Pass Uncertainty Estimation in Medical Image Segmentation | 利用专家标注分歧学习单次前向的不确定性。DOI：[10.1007/978-3-032-06593-3_11](https://doi.org/10.1007/978-3-032-06593-3_11) |

如果研究不确定性，必须报告校准误差、风险-覆盖率、错误检测能力或病例级质量相关性，不能只在论文中放一张漂亮的热力图。

### 2.7 联邦、多模态、缺失模态和持续学习

这类工作把医学数据现实约束直接变成研究对象。

| 会议 | 论文 | 可确认的创新主线 |
|---|---|---|
| MICCAI 2024 | FedEvi: Improving Federated Medical Image Segmentation via Evidential Weight Aggregation | 用证据/不确定性进行联邦聚合，处理多中心差异。DOI：[10.1007/978-3-031-72117-5_34](https://doi.org/10.1007/978-3-031-72117-5_34) |
| MICCAI 2024 | Low-Rank Mixture-of-Experts for Continual Medical Image Segmentation | 用低秩专家混合处理连续任务和灾难性遗忘。DOI：[10.1007/978-3-031-72111-3_36](https://doi.org/10.1007/978-3-031-72111-3_36) |
| ICASSP 2024 | Patch-Level Knowledge Distillation and Regularization for Missing Modality Medical Image Segmentation | 在模态缺失时进行 patch 级知识迁移。DOI：[10.1109/ICASSP48485.2024.10448218](https://doi.org/10.1109/ICASSP48485.2024.10448218) |
| ICASSP 2025 | Modality Modulation and Dual Consistency for Multi-Modality Semi-Supervised Medical Image Segmentation | 处理多模态半监督中的模态调制与双重一致性。DOI：[10.1109/ICASSP49660.2025.10887861](https://doi.org/10.1109/ICASSP49660.2025.10887861) |
| ICASSP 2025 | PAFedMIS: Personalized Asynchronous Federated Learning for Medical Image Segmentation | 针对异步和个性化联邦分割。DOI：[10.1109/ICASSP49660.2025.10889319](https://doi.org/10.1109/ICASSP49660.2025.10889319) |

## 3. 这些论文的创新单位是什么

把近两年论文抽象后，可以发现一个完整创新通常由四个部分组成：

```text
任务约束 + 失败机制 + 方法机制 + 区分性证据
```

### 例子一：弱标签

```text
任务约束：标注稀疏且昂贵
失败机制：普通伪标签在边界和小目标处不可靠
方法机制：动态伪标签修正或多重一致性
区分性证据：不同标注比例、伪标签质量、边界指标和多次种子
```

### 例子二：跨域泛化

```text
任务约束：训练和测试来自不同医院/设备
失败机制：模型学习了风格或扫描协议，而不是病灶结构
方法机制：形态-风格解耦、频域不变表示或测试时适应
区分性证据：源域、跨域、不同强度变化、无目标标签设置
```

### 例子三：边界和拓扑

```text
任务约束：细长结构、相邻器官或模糊边界
失败机制：像素重叠指标掩盖了断裂、错连和边界偏移
方法机制：可微形态学、拓扑正则、结构纠正或边界对比学习
区分性证据：HD95、ASSD、拓扑指标、错误案例、参数量匹配对照
```

这解释了为什么“一个模块”经常不够：它只是方法机制的一部分，没有任务约束、失败机制和区分性证据就无法形成完整论文。

## 4. 做到什么程度可以发论文

下面是比较现实的层级，不是任何会议的官方录用保证。

### 4.1 探索性工作

具备：

- 一个能运行的新模块；
- 一个数据集上比 baseline 高一点；
- 一两组消融。

这可以作为组内汇报、课程项目或技术报告，但作为正规会议论文通常偏弱。

### 4.2 有机会投 CCF-C 或医学影像相关主会的最低完整度

- 明确的医学分割困难，不是“我要提升 Dice”；
- 一个核心机制，最好只解决一个主问题；
- 强基线，包括 nnU-Net 或该任务公认基线；
- 固定患者级或病例级数据划分；
- 至少两个数据集，或一个数据集加跨域/低标注/噪声等严格设置；
- baseline、单组件、完整方法、参数量/计算量接近对照；
- 至少三次随机种子或明确的稳定性分析；
- Dice/IoU 之外报告 HD95、ASSD、边界或拓扑指标；
- 失败案例和误差类型分析；
- 代码、配置、环境和结果可复现。

### 4.3 更有竞争力的 CCF-B 级别工作

通常需要在上述基础上再增加至少一项真正的外部价值：

- 跨中心、跨设备或跨模态泛化；
- 少标签、弱标签或标注成本显著降低；
- 明确的理论/机制分析；
- 显著的效率、显存或部署优势；
- 多任务、多器官或多数据集一致收益；
- 更可靠的不确定性、校准或失败检测；
- 新数据集、规范化评测或有临床合作的验证。

单个公开数据集上加注意力模块、提升 0.5% Dice，通常不足以支撑强 B 级别投稿。

## 5. 对你当前 EMCAD 项目的直接判断

### 5.1 目前不建议的路线

以下组合容易做很多代码，却难形成清楚论文：

```text
EMCAD + CBAM
EMCAD + SE
EMCAD + 边界分支
EMCAD + 新损失
EMCAD + Mamba
```

如果同时改 backbone、decoder、loss、增强和输入尺寸，最终无法回答“到底什么因素有效”。

此外，EMCAD 本身已经包含多尺度卷积、特征融合、注意力门控和多级解码。再加一个泛化注意力模块，审稿人会自然追问：它与已有 MSDC、MSCB、LGAG、EUCB 的区别是什么？是否只是重复功能？

### 5.2 更适合 EMCAD 的三个候选问题

#### 方向 A：边界/小目标失败机制

研究问题：

> EMCAD 的多尺度解码在小目标和边界复杂病例上为什么仍会出现漏分和边界偏移？一种结构感知的特征交互是否能在参数量受控下改善这些错误？

必须做：

- 按目标面积、边界复杂度或病例难度分层；
- Dice、IoU、HD95、ASSD、边界 F-score；
- baseline、只加结构分支、只加监督、完整方法、参数量匹配对照；
- 可视化边界错误，而不是只展示最好案例。

风险：边界方向非常拥挤，必须有结构机制或拓扑证据，不能只是边缘卷积。

#### 方向 B：跨数据集/跨域泛化

研究问题：

> EMCAD 的多尺度特征是否会过度依赖训练域的纹理和强度分布？频率-形态分离或风格扰动能否提升跨数据集分割稳定性？

必须做：

- 明确源域和目标域；
- 训练只使用源域，目标域不能参与 checkpoint 选择；
- 统计域内和跨域性能差距；
- 使用外观扰动、频域扰动或风格迁移的单变量消融；
- 报告不同随机种子和不同域组合。

这是更像科学问题的方向，但需要你准备多个可比数据集和严格的数据协议。

#### 方向 C：有限标注/半监督 EMCAD

研究问题：

> 在只有 10%、20% 或 50% 标注时，EMCAD 的多尺度解码是否能稳定利用无标签图像？哪些一致性约束能够减少边界和小目标伪标签错误？

必须做：

- 固定标注比例和病例级划分；
- 监督 baseline、半监督 baseline、你的方法；
- 伪标签质量、置信度和错误类型分析；
- 不同标注比例曲线；
- 至少三次随机种子。

这是最容易从“加模块”提升为“真实任务约束”的路线之一。

### 5.3 我的优先级建议

如果你现在只有 EMCAD 代码、公开数据集和一台 GPU，建议优先级是：

```text
第一选择：有限标注/半监督
第二选择：跨数据集泛化
第三选择：边界/小目标/拓扑
```

如果你只有单一数据集且没有额外域，第一选择最现实；如果你已有 Synapse、ACDC、ISIC、Polyp 等多个数据集并能统一协议，第二选择更有研究味道；如果导师或合作方特别关注结构边界，再做第三选择。

## 6. 你现在应该怎样读论文

不要从头到尾逐字翻译几十篇。每篇论文先填一页“六问卡片”：

```text
1. 它观察到了什么失败现象？
2. 它认为失败原因是什么？
3. 它提出的机制具体改变了什么信息流或训练信号？
4. 它用什么实验区分“机制有效”和“参数变多”？
5. 它在哪些设置下失败？
6. 我的 EMCAD 项目能否复现它的核心对照？
```

建议先精读以下 12 篇，覆盖不同创新类型：

1. `EM-Net`：频率与 Mamba。
2. `Topologically Faithful Multi-class Segmentation`：拓扑约束。
3. `LUCIDA`：跨域适应。
4. `FRCNet`：半监督一致性。
5. `DeSAM`：基础模型适配。
6. `Differentiable Soft Morphological Filters`：可微形态学。
7. `CALSeg`：概率校准。
8. `LK-UNet`：效率/大卷积核。
9. `MMS`：单源域泛化。
10. `MARIO`：混合标注。
11. `U-Net V2`：重新思考跳跃连接。
12. `Quest for Clone`：测试时域适应。

阅读顺序应是：摘要 → 引言最后两段 → 方法总图 → 实验表 → 消融 → 失败案例 → 再回头读方法细节。

## 7. 让 Codex 帮你尽快进入正规流程

### 第 1 阶段：三天内完成项目审计

让 Codex 只读检查：

- 当前 Git 分支和 commit；
- EMCADNet 的 encoder、decoder、输出头；
- 训练/验证/测试数据路径；
- 患者级或病例级划分；
- checkpoint 选择逻辑；
- 指标实现；
- 预训练权重是否真正加载；
- 当前能否完成小数据过拟合。

输出应包含 `file:line` 证据，而不是泛泛解释。

### 第 2 阶段：一周内建立可信 baseline

固定：

- 数据划分清单；
- Python、PyTorch、CUDA；
- 输入尺寸和 batch size；
- 随机种子；
- 训练轮数和学习率；
- checkpoint 规则；
- 评估脚本。

每次实验保存：

```text
experiment_id/
├── config.json
├── command.txt
├── environment.txt
├── git_commit.txt
├── train.log
├── best.pth
├── last.pth
├── metrics.csv
├── predictions/
└── notes.md
```

### 第 3 阶段：第二周锁定一个问题

不要问“还能加什么模块”，改问：

```text
当前 baseline 最稳定的失败案例是什么？
这个失败是否能被一个机制解释？
哪个实验可以推翻我的解释？
```

如果你无法写出失败案例和反证实验，就不要开始大规模改代码。

### 第 4 阶段：第三至四周完成最小消融

以一个核心机制为例，至少安排：

```text
E0  原始 EMCAD baseline
E1  只加入机制的结构部分
E2  只加入对应监督/约束
E3  完整方法
E4  参数量或 FLOPs 接近的替代方案
E5  去掉关键子模块的反事实实验
```

如果是半监督或跨域问题，还要加不同标注比例、不同域组合或不同随机种子。

## 8. 你可以把什么叫作“已经步入正规”

当你能做到下面这些，就不再是漫无目的地看论文：

- 能用一句话说清自己的研究问题；
- 能指出 baseline 的具体失败现象；
- 能解释自己的模块改变了哪条信息流；
- 能设计一个结果为负也有意义的实验；
- 所有实验使用固定划分和可追溯配置；
- 能报告平均值、方差、效率和失败案例；
- 能承认方法在哪些条件下无效；
- 论文中的每个数字都能追溯到日志或表格；
- 代码、图表和文字可以由同一套实验记录重新生成。

## 9. 最后的现实判断

你现在不需要先达到“顶会研究者”的水平才开始。你需要先完成一个小而完整的闭环：

```text
一个数据任务
→ 一个可信 baseline
→ 一个稳定失败现象
→ 一个可证伪假设
→ 一个核心机制
→ 六组左右控制实验
→ 一套错误分析
→ 一篇诚实、边界清楚的论文
```

这条路线比同时阅读几百篇论文、尝试十几个模块更可能让你真正产生论文。

对于你当前 EMCAD 项目，下一步最值得做的具体动作是：先让 Codex 审计当前 Synapse/ACDC/Polyp 代码和数据协议，跑出固定 baseline，再从“有限标注”或“跨数据集泛化”中选一个主问题。不要在 baseline 和数据划分尚未完全可信之前继续堆新模块。

## 10. 主要检索入口

- DBLP：用于核对会议、年份、论文题目和书目信息：[https://dblp.org/](https://dblp.org/)
- MICCAI 2024 论文 DOI 示例：[Springer LNCS 978-3-031-72111-3](https://doi.org/10.1007/978-3-031-72111-3)
- IEEE ISBI 2024 论文 DOI 示例：[ISBI 2024](https://doi.org/10.1109/ISBI56570.2024.10635517)
- IEEE ICIP 2024 论文 DOI 示例：[ICIP 2024](https://doi.org/10.1109/ICIP51287.2024.10647750)
- IEEE ICASSP 2024 论文 DOI 示例：[ICASSP 2024](https://doi.org/10.1109/ICASSP48485.2024.10446030)
- BMVC 2024 论文集：[BMVC 2024 Proceedings](https://bmvc2024.org/proceedings/)
- 中国计算机学会推荐国际学术会议和期刊目录入口：[CCF Academic Evaluation](https://www.ccf.org.cn/Academic_Evaluation/By_category/)

CCF 分区、投稿时间、页数限制和是否接受 workshop/challenge 论文，都应在正式投稿前重新核对官方页面和本校规定。
