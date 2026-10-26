# 实验矩阵

| 配置 | 类别条件尺度融合 | 低秩协方差 | 目的 |
|---|---:|---:|---|
| baseline | 0 | 0 | 原模型结构与当前训练协议 |
| low_rank_covariance | 0 | 1 | 验证新模块独立贡献 |
| class_conditional_scale_fusion | 1 | 0 | 验证逐类尺度选择的贡献 |
| scale_fusion_and_covariance | 1 | 1 | 验证联合效果和相互作用 |

每个任务沿用自身划分、分辨率、增强、损失、优化器、学习率与验证频率。相同任务的四组使用相同 seed 队列，至少三个 seed，报告均值和标准差。二分类族的 `USE_PIXEL_RELIABILITY_FUSION`、`USE_MULTI_SCALE_TRAINING` 等已有开关在四组间固定。Synapse/ACDC 报每器官 Dice、HD95 与宏平均；二分类族沿用各自指标。参数量、FLOPs、推理时间与峰值显存的测量条件保持一致。

小/大器官分组按训练集标注面积确定阈值，再固定到验证/测试。检查各类尺度权重、协方差模块的实际修正量、边界及失败样例。用验证集选模，测试集仅用于最终评估。
