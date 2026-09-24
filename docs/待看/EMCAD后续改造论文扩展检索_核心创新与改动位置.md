# EMCAD 后续改造论文：核心创新与改动位置

检索/核对日期：2026-09-22  
研究对象：CVPR 2024 的 EMCAD/EMCADNet 及其 MSDC、MSCB、MSCAM、LGAG、EUCB、mutation supervision 等组件。

## 先说结论

确实已经出现多篇“明确在 EMCAD 上改造”或“明确复用 EMCAD 某个模块”的论文，但它们不是同一类工作：

1. **直接改 EMCAD 解码器、在 EMCAD 上验证增强方法，或明确延续其解码/监督方案**：RaEUNet、DeShiftNet、PraNet-V2、XAttnRes、AFF-Net、EDLDNet、Polyp-DiFoM。
2. **只改某个 EMCAD 组件，或采用 EMCAD 的设计思想**：WESCP-Net、PL-UNet、MVCA-UNet、EMCA-UNet、RetiVas-MTNet。
3. **跨任务借用 MSDC/训练策略**：HyPCA-Net、MAIL/Robust-MAIL、Mmrsg-unet、头颈癌剂量预测。它们不是 Synapse 分割上的 EMCAD 改进，不能直接当作 Dice 提升证据。
4. **仅比较、引用或同名缩写**：不计入“基于 EMCAD 改造”的确认名单。

“明确改造”只表示论文正文有相应表述或结构证据，并不表示作者结果已经被独立复现，也不表示一定超过你当前的 83.63 Dice。

以下共整理 17 项相关研究（7 项主要改造/增强工作、5 项组件改造或复用、4 项跨任务/策略延续、1 项应用）；并另列尚不能确认改造关系的候选。**不是 17 篇都改进了 EMCAD 的完整架构。**

## A. 明确改造、增强或延续 EMCAD（7 项，优先阅读）

### 1. RaEUNet

- **题目**：*RaEUNet: a retentive and efficient UNet for medical image segmentation*
- **期刊/年份**：The Visual Computer，2025
- **论文**：[Springer DOI 10.1007/s00371-025-03979-6](https://doi.org/10.1007/s00371-025-03979-6)
- **关系证据**：正文明确写明将 RMT 与“modified EMCAD”结合，并与原 EMCAD 解码器比较。
- **主要创新**：
  - 编码器改为 ResNet50 + RMT（Retentive Transformer）；
  - **EUCB → ECAUB**：在深度卷积后加入 ECAM，加强通道交互；
  - **MSCAM → MRAM**：在 CAB/SAB 主路径旁增加重加权分支；
  - **MSCB → MSCCB**：使用 SCConv 处理空间/通道冗余。
  - **LGAG 保留**，不是全部推倒重来。
- **改动位置**：编码器、EUCB、MSCAM、MSCB；LGAG 保留。
- **对 EMCAD 实验的启发**：适合做“只换 EUCB”“只换 MSCAM”“只换 MSCB”的三组消融，不能把 RMT 与解码器改动混成一个增益。

### 2. DeShiftNet

- **题目**：*DeShiftNet: a deformable-shifted cross-attention network for lightweight and robust organoid image segmentation*
- **期刊/年份**：BMC Bioinformatics，2026
- **论文**：[DOI 10.1186/s12859-026-06454-8](https://doi.org/10.1186/s12859-026-06454-8)
- **关系证据**：正文明确写明沿用 EMCAD 的轻量多阶段解码器，并说明替换/扩展的组件。
- **主要创新**：
  - **LGAG → CAG**（Cross-Attention Gate）：用卷积投影计算编码器—解码器的通道相关性；
  - **MSCAM → D-MSCAM**：增加可变形采样分支，适应不规则器官/类器官形状；
  - 编码器改为 deformable-shifted DSE 结构。
- **改动位置**：LGAG、MSCAM、编码器；EUCB 的逐级上采样骨架仍被采用。
- **注意**：实验对象是 OrganoID，不是 Synapse，不能直接把其 Dice 当作腹部 CT 预期值。

### 3. PraNet-V2：DSRA 插入 EMCAD

- **题目**：*PraNet-V2: Dual-Supervised Reverse Attention for Medical Image Segmentation*
- **状态**：2025 arXiv 预印本；有 Computational Visual Media 的 2026 书目信息。
- **论文**：[arXiv](https://arxiv.org/abs/2504.10986)；[DOI 10.26599/CVM.2025.9450510](https://doi.org/10.26599/CVM.2025.9450510)
- **关系证据**：方法部分明确将 DSRA 集成到 EMCAD。
- **主要创新**：分别学习前景和背景输出，利用类别特定的背景监督和 reverse attention 逐步细化；在原有 Dice + CE 外增加背景 BCE 监督。
- **改动位置**：预测/细化头与损失函数；没有声称重写 MSDC。
- **作者报告**：EMCAD-B2 82.71 → 加 DSRA 后 83.75，HD95 21.74 → 17.77。这里的基线是作者复现实验值，不能直接等同于你当前 83.63 的设置。

### 4. XAttnRes

- **题目**：*XAttnRes: Cross-Stage Attention Residuals for Medical Image Segmentation*
- **状态**：2026 arXiv 预印本。
- **论文**：[arXiv](https://arxiv.org/abs/2604.03297)
- **关系证据**：正文明确写明将 XAttnRes 添加到 EMCAD 解码器现有的 LGAG、MSCAM、EUCB 旁边。
- **主要创新**：维护编码器/解码器的跨阶段历史特征池；用空间对齐、通道投影和可学习聚合生成跨阶段残差。
- **改动位置**：解码器旁路/跨阶段特征融合；原 LGAG、MSCAM、EUCB 保留。
- **作者表格**：Synapse EMCAD 83.63，EMCAD + XAttnRes 83.75（+0.12 个百分点）。增幅较小，必须用相同划分、种子和训练轮数复核。

### 5. AFF-Net

- **题目**：*AFF-net: adaptive feature fusion network for efficient medical image segmentation*
- **期刊/年份**：Scientific Reports，2026
- **论文**：[Nature 页面](https://www.nature.com/articles/s41598-026-69243-6)，[DOI](https://doi.org/10.1038/s41598-026-69243-6)
- **关系证据**：摘要明确称将 LMCAB 集成进 EMCAD；全文给出 LMCAB、D-LGAG、BEA-U 的结构。
- **主要创新**：
  - PVT 改为 UniRepLKNet-S，并引入 MSPA 适配器微调；
  - **LMCAB**：1×1/3×3/5×5 深度卷积分支，经 GAP + 1×1 + softmax 产生分支权重，再 concat + 1×1 融合；
  - 用动态大核门控 **D-LGAG**；
  - 用带边缘注意力的 **BEA-U** 上采样。
- **改动位置**：骨干、MSCAM/CAB-SAB 融合段、LGAG、EUCB/上采样段。
- **重要校正**：作者在 Synapse 报告 Dice 82.76，不高于你当前 83.63；这不能证明方法一定更差，因为训练协议、骨干和划分可能不同。

### 6. EDLDNet

- **题目**：*An efficient dual-line decoder network with multi-scale convolutional attention for multi-organ segmentation*
- **期刊/年份**：Biomedical Signal Processing and Control，Crossref 记录为 2026；另有 2025 arXiv 版本。
- **论文**：[DOI](https://doi.org/10.1016/j.bspc.2025.108611)，[arXiv](https://arxiv.org/abs/2508.17007)
- **关系证据**：正文明确比较 EMCAD，并写明与 EMCAD 不同的是两条解码器输出都参与 mutation。
- **关系边界**：这是方法与训练策略的延续；未核实它是否从原 EMCAD GitHub 仓库直接派生，不能称为“官方 EMCAD 第二版”。
- **主要创新**：无噪声/带噪声双解码路径；训练时两路共同监督，推理时只保留无噪声路径。
- **改动位置**：解码器训练结构、mutation supervision；不是简单替换 MSDC。
- **作者报告**：Synapse 84.00 对比 EMCAD 83.63；需注意不是所有器官都提高。

### 7. Polyp-DiFoM

- **题目**：*From SAM to DINOv2: Towards Distilling Foundation Models to Lightweight Baselines for Generalized Polyp Segmentation*
- **会议/年份**：WACV 2026
- **论文**：[CVF 正式页面](https://openaccess.thecvf.com/content/WACV2026/html/Agnihotri_From_SAM_to_DINOv2_Towards_Distilling_Foundation_Models_to_Lightweight_WACV_2026_paper.html)，[arXiv](https://arxiv.org/abs/2512.09307)
- **关系证据**：实验表明确列出 EMCADNet + Ours。
- **主要创新**：把 SAM、DINOv2、OneFormer、Mask2Former 的语义/结构先验蒸馏给轻量学生网络，并加入频域编码；将蒸馏 latent features 与编码器输出在解码器融合。
- **改动位置**：训练蒸馏目标、解码器特征注入/融合；不是重新设计 EMCAD 的每个卷积块。
- **任务范围**：息肉分割，重点是跨数据集泛化，不是 Synapse。

## B. 只改 EMCAD 某个组件或复用其设计

| 论文 | 改了什么 | 关系与限制 |
|---|---|---|
| *Wavelet-enhanced spatiotemporal connectivity-preserving network for intracranial artery segmentation in DSA sequences*（WESCP-Net），[Scientific Reports 2026](https://www.nature.com/articles/s41598-026-51440-y) | **EUCB 的深度卷积换成 WTConv**，用频域信息辅助细小血管恢复；全模型还加入 Haar 小波下采样和时序增强 | 明确修改 EUCB，但任务是 DSA 时序颅内动脉，不是腹部 CT；原 EMCAD EUCB 已有 channel shuffle，不能把通道重排算作该文首次提出 |
| *PL-UNet: a real-time power line segmentation model for aerial images based on adaptive fusion and cross-stage multi-scale analysis*，[J Real-Time Image Processing 2025](https://doi.org/10.1007/s11554-024-01615-5) | 受 EMCAD 的 LGAG 启发，提出多尺度 MSAG（普通 1×1、3×3 深度卷积、5×5 深度卷积） | 遥感电力线分割，属于 LGAG 思想迁移，不是 EMCAD 医学复现 |
| *MVCA-UNet: A Multi-scale Visual Convolutional Attention Architecture for Skin Lesion Segmentation*，[ICIC 2025](https://doi.org/10.1007/978-981-95-0036-9_18) | VSS 编码器、MSCB 深度卷积解码器、MCSA 多尺度通道-空间跳连模块 | 摘要明确受 EMCAD decoder 启发，但未足以证明逐项替换原 LGAG/MSCAM |
| *EMCA-UNet: A Novel Multi-Scale Convolutional Attention Model for Enhanced Lung Nodule Segmentation*，[ITC 2025](https://doi.org/10.5755/j01.itc.54.4.39602) | RESMAC（ConvNeXt + EMA）编码器，解码器采用 EMCA、LGAG、EUCB | 是“新编码器 + EMCAD 风格解码器”，不能据此宣称改进了原 MSDC |
| *RetiVas-MTNet: Multi-Task Learning Based on Retinal Vasculature*，[CAIBDA 2025](https://doi.org/10.1109/CAIBDA65784.2025.11183230) | 摘要称在 FR-Unet 中引入 EMCAD，并增加血管分割 + 血管面积百分比多任务输出 | 目前是摘要级证据，结构细节不足；任务为视网膜血管 |

## C. 跨任务复用 EMCAD 的 MSDC 或训练策略

| 论文 | 复用内容 | 为什么不能当作 Synapse EMCAD 改进 |
|---|---|---|
| *HyPCA-Net: Advancing Multimodal Fusion in Medical Image Analysis*，[arXiv 2026](https://arxiv.org/abs/2602.16245) | 明确写明用两个 SCALA 模块替换 EMCAD 的 MSDC；含异构多尺度卷积和空间-通道并行融合 | 多模态医学图像分析/预测，不是 Synapse 分割 |
| *Effective and Robust Multimodal Medical Image Analysis*（MAIL/Robust-MAIL），[KDD 2026](https://doi.org/10.1145/3770854.3780331) | MSGDC 受 EMCAD MSDC 启发，再加通道注意力、频域融合和随机投影注意力噪声 | 多模态分类与鲁棒性任务，不是 EMCAD 解码器 |
| *Mmrsg-unet: integrating multi-scale Mamba and reverse semantic gating for medical image segmentation*，[DOI](https://doi.org/10.1007/s00530-026-02439-y) | 采用源自 EMCAD/MERIT 的 combinatorial mutation loss；反向语义门控用浅层特征引导深层特征 | 主要是训练策略复用，网络并非 EMCAD 改版；其 LGAG 同名但定义不同 |
| *Dose prediction for head-and-neck cancer using a fusion of lightweight 3D multi-scale feature enhancement modules*，[Medical Physics 2026](https://doi.org/10.1002/mp.70481)，[PubMed](https://pubmed.ncbi.nlm.nih.gov/42135557/) | PubMed/Europe PMC 收录的作者摘要明确说明：8 层 Transformer + 3D MSCB + EMCAD 特征融合 | 头颈癌剂量回归，不是分割；摘要确认采用关系，但不能由此推断每个 2D 模块的 3D 转换细节 |

MAIL/Robust-MAIL 的[可读预印本](https://arxiv.org/abs/2602.15346)明确写明 MSGDC 受 EMCAD 的 MSDC 启发。其 1×1 分支采用分组点卷积，3×3/5×5 分支采用深度卷积；这是具体的模块设计延续，但整个任务与 Synapse 器官分割不同。

## D. 使用 EMCAD 的应用论文（1 项，不冒充结构创新）

*Advanced Brain Tumor Segmentation Using EMCAD: Efficient Multi-scale Convolutional Attention Decoding*，2025，[arXiv:2509.05431](https://arxiv.org/abs/2509.05431)。

该文明确将 EMCAD 用于脑肿瘤分割，但本轮核实到的材料不足以证明有新的核心模块改造。它适合了解 EMCAD 的任务迁移，不适合作为“作者怎样设计新解码器”的主要参考。

## E. 尚不能确认为 EMCAD 改造的候选与排除项

以下是真实存在但本轮没有足够证据证明其“改造原始 EMCAD”的候选或排除项。列出是为了避免漏掉后续可追查线索，**不代表确认基于 EMCAD，也不代表它们一定没有关联**。

| 完整题目与入口 | 当前判断 |
|---|---|
| [CMFA-Net: A CNN–Mamba Collaborative Feature Alignment Network for Robust Medical Image Segmentation](https://doi.org/10.3390/electronics15112343) | 其 EMCAD 缩写展开不同；不能因缩写相同就确认派生关系 |
| [Resmscam-FFT Net: Residual Multi-Scale Channel Attention Module with Fast Fourier Transform Network for Cardiac Segmentation](https://doi.org/10.1109/ISBI61048.2026.11515747) | PVTv2 + FFT + ResMSCAM，但仅模块名字相似不足以证明改造来源 |
| [VM-UNet++: Vision Mamba with Adaptive Feature Reweighting and Multiscale Fusion for Medical Image Segmentation](https://doi.org/10.1109/ICICML63543.2024.10957962) | 摘要有 LGAG，未核实其出处及完整结构 |
| [ADMC: An Attention-based Dense Multi-scale Convolutional architecture for Medical Image Segmentation](https://doi.org/10.1109/ICICML63543.2024.10958168) | 摘要有 MSDC，但未确认与 EMCAD 的具体派生关系 |
| [GMSRAD: A Global-Local Modulated Sub-Pixel Reconstruction Attention Decoder for Medical Image Segmentation](https://doi.org/10.1007/978-981-92-3513-1_5) | 摘要有与 EMCAD 比较；不足以证明基于其改造 |
| [LWD: A Lightweight Decoder Leveraging Gated Attention and Cross-Group Convolution for Medical Image Segmentation](https://doi.org/10.1007/978-3-032-31673-8_26) | 轻量解码器题材相近，未确认改造关系 |
| [EPHAD: Efficient Phased Hybrid Attention Decoder for Medical Image Segmentation](https://doi.org/10.1007/978-981-95-3312-1_8) | 未取得足够结构证据 |
| [DHR-Net: An ultra-lightweight U-Net based on efficient convolutional attention for medical image segmentation](https://doi.org/10.1016/j.displa.2026.103425) | 全文核实不足；不是另一个同名医学配准 DHR-Net |
| [EDSDF: An Efficient Deep Supervised Distillation Framework for medical image segmentation](https://doi.org/10.1016/j.neucom.2025.130635) | 全文核实不足 |
| [MCAVM-UNet: Enhancing key region focus and feature interaction in Vision Mamba for medical image segmentation](https://doi.org/10.1016/j.bspc.2025.109371) | 全文核实不足 |
| [基于 SwinTransCAD 的高效多尺度注意力解码的甲状腺结节分割方法](https://doi.org/10.3788/lop242250) | 出版社正文获取失败，不能根据题目补写改造内容 |

特别注意：同名缩写不等于同一模型。例如某些论文中的 EMCAD 表示 *Enhanced Multi-scale Context Aggregation Decoder*，并非 CVPR 2024 的 *Efficient Multi-scale Convolutional Attention Decoder*。

## 检索范围与证据边界

- 通过 OpenAlex 进行了 EMCAD 关键词检索（177 条），以及对 EMCAD 会议论文记录的引用检索（493 条）、对预印本记录的引用检索（13 条）。按 OpenAlex ID 合并去重后为 598 条记录。各检索集合有重叠；预印本与正式发表版本还可能拥有不同 ID，所以 **598 也不是严格按论文内容去重的篇数，更不是 598 篇确认改造论文**。
- 候选元数据保存在项目的 `.tmp/emcad_literature/works.json`；抓取/抽取的正文缓存同在该目录。缓存是检索工作底稿，不应把“抓取成功”误解为“全文已逐字审读”。
- 对候选进一步核对了 Springer、Nature、BMC、CVF、arXiv、Crossref 等页面或正文；无法取得全文的论文只保留为低置信度/候选，不补写未见过的结构。
- 本清单截至 2026-09-22，不能保证覆盖尚未被数据库收录、标题不含 EMCAD、或全文不可访问的工作。

## 对你当前 EMCAD/Synapse 项目的实际优先级

建议先做可隔离的改动：

1. **DSRA（PraNet-V2）**：重点改预测/细化头和损失，且已有 EMCAD-B2 的 Synapse 实验，适合优先阅读；实际接入仍需处理逐级前景/背景细化，不是简单加一项 BCE。
2. **XAttnRes**：旁路添加，保留 LGAG/MSCAM/EUCB，便于判断跨阶段历史特征是否有效。
3. **ECAUB 或 MRAM（RaEUNet）**：一次只换一个解码器组件。
4. **D-LGAG/BEA-U 或 LMCAB（AFF-Net）**：组件较多，必须拆成单模块、组合模块和参数匹配对照。
5. **D-MSCAM/CAG（DeShiftNet）**：对边界和形变可能有价值，但先在 224×224 Synapse 上做轻量版本，避免把复杂编码器改动混入结论。

所有实验应固定病例级划分、输入尺寸、预训练权重、随机种子、训练轮数和测试后处理，并同时记录平均 Dice、逐器官 Dice、HD95、参数量、FLOPs 和推理时间。论文中的 82.71、82.76、83.63、83.75、84.00 来自不同实现/协议，不能横向直接当成你的可复现分数。
