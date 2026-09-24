# EMCAD 发表后的后续改进论文：可核实检索清单

> 后续扩展：请优先阅读[《EMCAD 后续改造论文扩展检索：核心创新与改动位置》](EMCAD后续改造论文扩展检索_核心创新与改动位置.md)。新文档补充了 RaEUNet、DeShiftNet、PraNet-V2、Polyp-DiFoM 等，并基于 AFF-Net 全文补充了具体结构及 Synapse Dice 82.76。以下保留首轮检索记录。

检索日期：2026-09-22。目的：查找真正使用、改造或延续 EMCAD 方法的研究，不把仅引用 EMCAD 或与其比较的工作自动当成衍生改进。以下是本轮核实到的清单，不是穷尽全部引用文献的系统综述。

通过 OpenAlex 的 EMCAD 关键词检索寻找候选，再核对出版社页面、arXiv 正文和 Crossref 书目信息。论文真实存在不代表其实验结论已被独立复现；本轮没有运行作者代码。

## 1. AFF-net: adaptive feature fusion network for efficient medical image segmentation

- 发表：Scientific Reports，2026-08-30；出版社页面标记为已接受、提前公开、后续仍可能编辑的版本。
- [出版社页面](https://www.nature.com/articles/s41598-026-69243-6)
- DOI：10.1038/s41598-026-69243-6。
- 作者在论文页面给出的代码地址：https://github.com/wangru1026/AFF-Net 。本轮未核验仓库完整性或复现性。
- 关系判断：**明确改造 EMCAD，优先阅读。**

出版社摘要明确写道：

> we integrate a lightweight multi-scale convolutional attention block (LMCAB) into the efficient multi-scale convolutional attention decoder (EMCAD)

可核实的改进包括：在骨干中使用多尺度感知适配器 MSPA；在 EMCAD 中集成 LMCAB，进行空间与通道信息的自适应融合；结合适配器微调策略。实验涉及 Synapse 和 ACDC。

对你最有用的阅读问题：LMCAB 与原 MSCAM 的计算图到底差在哪里？收益来自解码器还是预训练骨干的微调方式？是否有分别隔离两项改动的消融？本轮主要核实出版社摘要，不据此推断 LMCAB 的精确插入位置和逐层公式。

## 2. XAttnRes: Cross-Stage Attention Residuals for Medical Image Segmentation

- 年份与状态：2026，arXiv 预印本；本轮未核实正式会议或期刊录用。
- [摘要](https://arxiv.org/abs/2604.03297)
- [正文](https://arxiv.org/html/2604.03297)
- 关系判断：**明确在 EMCAD 上添加模块，最贴近“保留基线、添加创新点”的实验。**

正文的“Integration as a versatile plug-in”部分明确写道：

> We add XAttnRes to the EMCAD decoder alongside its existing LGAG, MSCAM and EUCB components.

核心思路是维护跨阶段特征历史池，以可学习的选择性聚合获取前面编码/解码阶段的信息；用空间对齐和通道投影处理跨尺度特征。在 EMCAD 实验中，模块仅应用于解码侧，但历史池包含编码器和解码器输出。

论文表 1 报告 Synapse 上 EMCAD 的 Dice 为 83.63，EMCAD + XAttnRes 为 83.75，即提升 0.12 个百分点。这是作者报告值，不是本项目复现；这种幅度需要检查随机种子、误差范围以及基线是否同设置重新训练，不能直接断言提升稳定显著。

阅读重点：为什么固定 skip 连接不够？历史池如何对齐？额外计算和参数多少？保留 skip、移除 skip、用新机制替代 skip 三种设置如何消融？

## 3. An efficient dual-line decoder network with multi-scale convolutional attention for multi-organ segmentation

- 模型名：EDLDNet。
- 发表：Biomedical Signal Processing and Control；2025 年已有 arXiv 版本，Crossref 记录的期刊出版年月为 2026 年 2 月。不要仅凭 DOI 中的 2025 把卷期年写成 2025。
- [DOI](https://doi.org/10.1016/j.bspc.2025.108611)
- [预印本摘要](https://arxiv.org/abs/2508.17007)
- [可阅读全文](https://arxiv.org/html/2508.17007)
- 论文给出的代码地址：https://github.com/riadhassan/EDLDNet 。本轮直接读取仓库失败，不能声称已核实代码可运行。
- 关系判断：**延续相关多尺度注意力解码设计，并明确扩展 EMCAD 使用的 mutation 策略；不称作作者官方 EMCAD 第二版，也不声称源码完全基于同一仓库。**

正文第 3.3 节写道：

> Unlike EMCAD, outputs from both decoders (noise-free ... and noisy ...) are mutated.

方法采用无噪声和带噪声的双解码路径，训练时使用两路输出参与其组合监督，推理时只使用无噪声分支。这提供了“训练时增加辅助路径，提高鲁棒性；部署时保留单路”的创新路线。

论文报告 Synapse Dice 为 84.00，对比表中的 EMCAD 为 83.63；但不是每个器官都提高。该比较不能直接当作你当前训练环境中的预期增益。

阅读重点：噪声加在哪里、如何选强度？两条解码器是否共享参数？四级输出如何先融合再参与 mutation？不要未经核对就理解为八个输出全部做 255 种组合。训练成本与推理成本要分开比较。

## 4. Advanced Brain Tumor Segmentation Using EMCAD: Efficient Multi-scale Convolutional Attention Decoding

- 时间：2025-09-05，arXiv 预印本；本轮未核实正式录用。
- [原始页面](https://arxiv.org/abs/2509.05431)
- 关系判断：**明确使用 EMCAD，但本轮核实到的摘要不足以证明有新的核心结构创新。**

将 EMCAD 应用于 BraTS2020 脑肿瘤 MRI 分割。摘要称最高 Dice 为 0.31，平均约 0.285±0.015。数值来源于作者摘要，不代表医学分割的一般性能水平，也不代表本项目能获得相同结果。

建议把它作为“应用迁移”的参考，而不是优先模仿的高质量创新范本。标题包含 EMCAD，不代表必然是有充分消融支撑的改进模型。

## 5. 本轮看到但没有列为“已确认基于 EMCAD 改造”的工作

- [MSVM-UNet: Multi-Scale Vision Mamba UNet for Medical Image Segmentation](https://arxiv.org/html/2408.13735)：核实到对比与引用，不能仅凭这些判定其改造 EMCAD。
- [CFII-Net: Explicit Class Embeddings and Feature Maps Through Iterative Interaction for Boosting Medical Image Segmentation](https://www.ijcai.org/proceedings/2025/0283.pdf)：确实是 IJCAI 2025 论文，但本轮提取出的 EMCAD 相关段落主要是相关工作和对比，不列入已确认衍生清单。
- [MedCAGD: Context-Aware Gated Decoder for Efficient Medical Image Segmentation](https://arxiv.org/html/2607.00409)：与 EMCAD 的解码器和门控设计有明确比较，值得扩展阅读，但本轮没有找到足以将它表述为直接改造 EMCAD 的证据。
- [Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](https://arxiv.org/html/2603.12547)：可作为解码器创新的邻近路线阅读，但“指出 EMCAD 的局限”并不自动等于“基于 EMCAD 代码改进”。

## 建议阅读顺序

先看 AFF-Net，了解已发表的直接改造；再看 XAttnRes，学习基线加模块的结构与消融；最后看 EDLDNet，了解从训练策略和辅助分支创新的路线。脑肿瘤应用论文放在后面。

读每篇时记录六项：原方法不足、保留的 EMCAD 组件、新增或替换位置、对应消融、数据划分与模型选择规则、额外训练及推理成本。论文声称的机制、实验支持的事实、你自己的猜测应分栏记录。

本轮检索入口：[OpenAlex EMCAD 检索](https://api.openalex.org/works?search=EMCAD&per-page=200)；EDLDNet 期刊年月核对：[Crossref 记录](https://api.crossref.org/works/10.1016/j.bspc.2025.108611)。关键词搜索可能漏掉未被收录、正文不可检索或不用 EMCAD 缩写的论文，因此不能将本清单解读为全部后续工作。
