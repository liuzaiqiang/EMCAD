# 已核验相关工作与创新边界

| 论文 | 来源 | 已有机制及与本方案的关系 |
|---|---|
| Rahman, Munir, Marculescu, *EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation*, CVPR 2024 | [论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.pdf), [官方代码](https://github.com/SLDGroup/EMCAD) | MSCB 多核、LGAG skip、CAB/SAB 已有。新增机制不得仅重复这些操作。 |
| *CCQ: Cross-Class Query Network for Partially Labeled Organ Segmentation*, AAAI 2023 | [官方论文与 DOI](https://ojs.aaai.org/index.php/AAAI/article/view/25264), [代码](https://github.com/SooLab/CCQ) | 器官类别查询已有先例；其目标是部分标注的类间关系，本方案针对 EMCAD 的逐类别解码尺度选择。 |
| *Enhancing Medical Image Segmentation with TransCeption: A Multi-Scale Feature Fusion Approach*, arXiv 2023 | [预印本](https://arxiv.org/abs/2301.10847) | 多尺度融合及上下文桥已存在；其正式发表 venue 与作者细节尚未在本次检索中核实，不能用来证明新颖性。 |

归类：类别查询与尺度融合均属于已有机制；在 EMCAD 的 d3/d2/d1 上应用属于适配/组合；逐类选择三个解码尺度以及小/大器官差异是待实验验证的具体假设。目前没有完成系统专利/论文查重，不能断言正式新颖性。计算开销主要来自各尺度投影与 logits 上采样；优于任何已有方法的说法目前没有证据。
