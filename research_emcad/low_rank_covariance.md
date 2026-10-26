# 低秩通道协方差交互（LRCI）

## 为什么放在 EMCAD 这里

EMCAD 的 CAB 根据单通道汇聚信息给特征加权；它没有直接描述不同通道在一张图中如何共同变化。LRCI 在 `d4` 和 `d3` 的 MSCB 后，以最多 32 个压缩通道计算图像自身的通道相关矩阵，用正、负相关关系混合空间特征，再投影回原通道。修正后的 `d4/d3` 沿原级联解码路径继续生成 `d2/d1`，所以**仅开启 LRCI 时也能影响最终 p1**。类别条件尺度融合仍在四个输出头之后工作，两者处理的分别是解码特征关系和类别输出尺度。

模块不做语义门控跳连融合、可学习空间对齐或双向一致性约束，也没有修改现有 LGAG、CAB、MSCB、输出头、类别条件尺度融合文件。投影的初始值确定，回投影初始为零；开启时初始预测与对应原模型逐项相同。两个开关都关闭时不创建 LRCI 参数，执行原 EMCAD 路径。PVTv2-B0 默认通道下增加 26,624 个参数；PVTv2-B2 下增加 53,248 个参数。

## 适配与消融

设置 `USE_LOW_RANK_COVARIANCE=0|1`，与现有 `USE_CLASS_CONDITIONAL_SCALE_FUSION=0|1` 独立。六类训练启动器及测试配置覆盖 Synapse、ACDC、Polyp 五集、ISIC 两集、BUSI、Cell 两集，共 12 集；不改变任何数据集的损失或划分。四组命令见 [ablation_commands.md](ablation_commands.md)。必须报告：LRCI 相对基础模型、双开相对基础模型和双开相对仅类别条件尺度融合的 Dice，以及不同 seed 的波动和推理开销。

二阶统计特征并非全新概念；例如 [GSoP](https://openaccess.thecvf.com/content_CVPR_2019/html/Gao_Global_Second-Order_Pooling_Convolutional_Networks_CVPR_2019_paper.html) 已研究通道协方差。这里的待验证研究假设是：在 EMCAD 早期两级解码中使用低秩图像自适应通道交互，能够与现有按类别的尺度融合互补。**代码检查不能证明 Dice 会提高，真实数据实验尚未完成。**
