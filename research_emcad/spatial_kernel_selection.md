# 空间卷积核选择（Spatial Kernel Selection, SKS）

## 做什么

EMCAD 每级 MSCB 已有 1×1、3×3、5×5 深度卷积分支，但默认把分支响应直接相加。SKS 根据该级局部特征，为每个像素预测这三个分支的 softmax 权重，再以均值为 1 的系数加权原分支。它使小结构、平坦区域、边界可以在同一幅图中使用不同的局部感受野。模块位于四级 MSCB 内；类别条件尺度融合仍在四级输出头之后，两者作用位置和信息粒度不同。

每级只增加 `分支数 × 扩张通道数 + 分支数` 个参数。PVTv2-B0、默认三分支时四级合计增加 3,084 个解码器参数。新参数全零初始化，初始 softmax 均匀，加权系数恰好为 1；因此开启时从既有 EMCAD 或既有类别条件尺度融合模型的相同函数起步。关闭时根本不创建 SKS 参数，并执行原来的分支求和或拼接代码。

## 开关与数据集

`USE_SPATIAL_KERNEL_SELECTION=0|1` 是独立开关；`USE_CLASS_CONDITIONAL_SCALE_FUSION=0|1` 保持原义。六类训练启动器均传入两个开关：Synapse、ACDC、Polyp（BKAI、ClinicDB、ColonDB、ETIS、Kvasir）、ISIC2017/2018、BUSI、Cell（DSB18、EM）。测试入口从训练 `config.json` 恢复开关；Synapse 可使用 `--train_config`。各数据集沿用原损失、增强、类别数和预测逻辑。

四组实验见 [ablation_commands.md](ablation_commands.md)。必须对每个目标数据集在相同划分、seed、预训练权重和训练预算下比较 **基础、仅 SKS、仅类别条件尺度融合、双开**，以验证两个要求：`Dice(SKS)>Dice(基础)` 和 `Dice(双开)>Dice(基础)`。最好还报告 `Dice(双开)` 与 `Dice(仅尺度融合)` 的差值、多 seed 方差和推理开销。代码结构与初始化检查无法证明 Dice 会提高；尚未完成训练验证。

## 论文定位

动态卷积核选择已有前人工作，例如 [Selective Kernel Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html)。本模块的可检验贡献是把**逐像素局部核选择**放到 EMCAD 原多尺度深度卷积分支，与**按类别的输出尺度融合**组合；不能把“动态选择卷积核”本身声称为全新发明。发表前仍需文献检索、跨数据集消融和统计验证。
