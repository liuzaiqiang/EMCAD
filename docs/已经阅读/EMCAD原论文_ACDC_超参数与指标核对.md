# EMCAD 原论文：ACDC 超参数与指标核对

来源：Rahman et al., *EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation*, CVPR 2024。

本文中的页码均指随项目保存的论文 PDF：`EMCAD：Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation.pdf`。

## 1. ACDC 数据设置

- 任务：心脏 MRI 三类器官分割，右心室（RV）、心肌（Myo）、左心室（LV）。
- 数据：100 个 cardiac MRI scans。
- 划分：训练 70 例（1,930 张轴向切片）、验证 10 例、测试 20 例。
- 位置：补充材料第 1 页（该 PDF 的第 12 页），7.1 `Cardiac organ segmentation`。

## 2. ACDC 训练超参数

| 项目 | 原论文设置 | 位置 |
|---|---:|---|
| 输入分辨率 | 224 x 224 | 主论文第 6 页，4.1 |
| 训练轮数 | 400 epochs | 主论文第 6 页，4.1 |
| batch size | 12 | 主论文第 6 页，4.1 |
| 优化器 | AdamW | 主论文第 6 页，4.1 |
| 初始学习率 | 1e-4 | 主论文第 6 页，4.1 |
| weight decay | 1e-4 | 主论文第 6 页，4.1 |
| 数据增强 | 随机旋转、随机翻转 | 主论文第 6 页，4.1 |
| 训练损失 | 0.3 * Cross Entropy + 0.7 * Dice loss | 主论文第 6 页，4.1 |
| 最优模型选择 | 按验证集 Dice 保存 best model | 主论文第 6 页，4.1 |
| 编码器 | ImageNet 预训练的 PVTv2-B0 / PVTv2-B2 | 主论文第 6 页，4.1 |
| EMCAD 多尺度卷积核 | [1, 3, 5] | 主论文第 6 页，4.1 |
| 多尺度卷积组织方式 | parallel depth-wise convolutions | 主论文第 6 页，4.1 |
| 论文运行环境 | PyTorch 1.11.0；单张 NVIDIA RTX A6000 48 GB | 主论文第 6 页，4.1 |

原论文**没有报告** ACDC 的随机种子、学习率调度器、AdamW betas、数据加载 workers、训练切片采样细节。因此这些不能被写成“与原论文完全一致”；只能在自己的复现实验中固定并明确报告。

## 3. ACDC 最终结果

表 3 位于主论文第 7 页，单位均为 Dice（%）。

| 模型 | 平均 Dice | RV | Myo | LV |
|---|---:|---:|---:|---:|
| PVT-EMCAD-B0 | 91.34 +/- 0.2 | 89.37 | 88.99 | 95.65 |
| PVT-EMCAD-B2 | **92.12 +/- 0.2** | **90.65** | **89.68** | **96.02** |

原文声称 B2 的平均 Dice 为 92.12%，并比 Cascaded MERIT 的 91.85% 高约 0.27 个百分点。表 3 中 EMCAD-B2 在 RV、Myo、LV 三个类别上均为该表最高值。

## 4. 指标范围：不要误把 Synapse 指标套到 ACDC

ACDC 在原论文中报告的是 **Dice**，包括平均 Dice 和三类器官 Dice。补充材料第 1 页（PDF 第 12 页）7.2 明确说明：Dice 用于所有数据集；HD95 与 mIoU 是 Synapse 多器官分割的附加指标。

因此，原论文没有给出 ACDC 的 HD95、mIoU、IoU、参数量、FLOPs 或推理时间。若你的 ACDC 创新实验额外报告这些指标，应注明它们是你自己的补充评测，不能与论文表 3 中“原作者已报告的 ACDC 指标”混为一谈。

## 5. 用于 EMCAD 公平对比的最小复现口径

在 ACDC 上比较 `EMCAD` 与 `EMCAD + 单个创新模块` 时，至少保持以下不变：70/10/20 病例划分、224 x 224、400 epochs、batch size 12、AdamW、lr=1e-4、weight decay=1e-4、相同增强、0.3 CE + 0.7 Dice、相同 PVTv2-B2 预训练权重、相同验证集 best-checkpoint 规则。

你自己的 baseline 结果可以不同于论文的 92.12%，但论文中应同时写出：原文报告值（92.12%）和本机复现值（你的实际均值及标准差）。创新模块只应与本机复现 baseline 作主比较。
