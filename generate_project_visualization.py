"""
生成项目可视化图表
"""
# pyplot 负责建立多子图画布、添加文字/箭头并导出 PNG。
import matplotlib.pyplot as plt
# 该模块别名在当前脚本没有后续调用，保留是为了不改变原文件结构。
import matplotlib.patches as mpatches
# FancyBboxPatch 用于所有圆角节点框；FancyArrowPatch 当前未直接调用。
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
# NumPy 当前没有后续调用，属于原脚本保留依赖。
import numpy as np

# 设置中文字体
# 按顺序尝试黑体、微软雅黑和 Arial Unicode MS，实际使用首个已安装字体。
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
# 关闭 Matplotlib 对负号的 Unicode 替换，避免中文字体下负号显示成方框。
plt.rcParams['axes.unicode_minus'] = False

# 创建图表
# 主图采用 20x24 英寸竖向画布，包含六个信息区域。
fig = plt.figure(figsize=(20, 24))
# 四行两列布局；网络图和训练图分别横跨中间整行。
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# ========== 1. 项目结构图 ==========
# 第一行左列展示仓库根目录、关键入口和三个主要子目录。
ax1 = fig.add_subplot(gs[0, 0])
# 设定人工布局所用的横坐标范围。
ax1.set_xlim(0, 10)
# 纵向范围比简化版更高，以容纳根目录和三层节点。
ax1.set_ylim(0, 12)
# 隐藏刻度与坐标轴边框。
ax1.axis('off')
# 设置中文子图标题。
ax1.set_title('项目文件结构', fontsize=14, fontweight='bold', pad=20)

# 根目录
# 用顶部浅蓝色长框表示整个 SLDGroup_EMCAD 仓库。
root = FancyBboxPatch((1, 10), 8, 1.5, boxstyle="round,pad=0.1", 
# 统一使用黑色粗边框。
                     facecolor='lightblue', edgecolor='black', linewidth=2)
# 把根节点框加入 ax1。
ax1.add_patch(root)
# 在根节点中央写仓库目录名。
ax1.text(5, 10.75, 'SLDGroup_EMCAD/', ha='center', va='center', fontsize=12, fontweight='bold')

# 主要文件
# 元组字段依次为显示名、中心 x/y 坐标和填充色。
files = [
# Synapse 训练入口。
    ('train_synapse.py', 2, 8.5, 'lightgreen'),
# Synapse 测试入口。
    ('test_synapse.py', 5, 8.5, 'lightgreen'),
# 训练循环与损失实现。
    ('trainer.py', 8, 8.5, 'lightyellow'),
# 模型与编码/解码结构目录。
    ('lib/', 2, 7, 'lightcoral'),
# 数据、损失和评估工具目录。
    ('utils/', 5, 7, 'lightcoral'),
# 图中假定的 Synapse 数据目录。
    ('data/Synapse/', 8, 7, 'lightcoral'),
]

# 逐项生成一级节点。
for name, x, y, color in files:
# 由中心点换算节点框左下角，并设置固定宽高。
    box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.05",
# 每项使用自身颜色，边框样式保持一致。
                         facecolor=color, edgecolor='black', linewidth=1)
# 添加节点框。
    ax1.add_patch(box)
# 在节点中心显示名称。
    ax1.text(x, y, name, ha='center', va='center', fontsize=9)

# lib子目录
# 展开显示训练真正涉及的三个 lib 文件。
lib_files = ['networks.py', 'decoders.py', 'pvtv2.py']
# i 控制竖向间隔，f 是文件名。
for i, f in enumerate(lib_files):
# 三个框沿 x≈2.25 竖直排列。
    box = FancyBboxPatch((1.5, 5.5-i*0.8), 1.5, 0.5, boxstyle="round,pad=0.05",
# 子项统一使用 wheat 填充。
                         facecolor='wheat', edgecolor='black', linewidth=1)
# 添加 lib 子项框。
    ax1.add_patch(box)
# 写入文件名。
    ax1.text(2.25, 5.75-i*0.8, f, ha='center', va='center', fontsize=8)

# utils子目录
# 展开显示 Synapse Dataset、通用评估和预处理脚本。
utils_files = ['dataset_synapse.py', 'utils.py', 'preprocess_synapse_data.py']
# 逐项绘制 utils 子节点。
for i, f in enumerate(utils_files):
# 节点中心位于 x≈5.25。
    box = FancyBboxPatch((4.5, 5.5-i*0.8), 1.5, 0.5, boxstyle="round,pad=0.05",
# 沿用统一子节点样式。
                         facecolor='wheat', edgecolor='black', linewidth=1)
# 添加节点框。
    ax1.add_patch(box)
# 写入文件名。
    ax1.text(5.25, 5.75-i*0.8, f, ha='center', va='center', fontsize=8)

# data子目录
# 展开训练切片、测试体数据和病例列表三类数据位置。
data_files = ['train_npz/', 'test_vol_h5/', 'lists/']
# 逐项绘制 data 子节点。
for i, f in enumerate(data_files):
# 节点中心位于 x≈8.25。
    box = FancyBboxPatch((7.5, 5.5-i*0.8), 1.5, 0.5, boxstyle="round,pad=0.05",
# 沿用统一子节点样式。
                         facecolor='wheat', edgecolor='black', linewidth=1)
# 添加节点框。
    ax1.add_patch(box)
# 写入目录名。
    ax1.text(8.25, 5.75-i*0.8, f, ha='center', va='center', fontsize=8)

# 箭头
# 三支向下箭头表示 lib、utils、data 与各自展开项的层级关系。
for x in [2, 5, 8]:
# 从一级目录下方向其子节点区域画短箭头。
    ax1.arrow(x, 7.3, 0, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black')

# ========== 2. 数据流程图 ==========
# 第一行右列展示 Synapse 原始体数据到二维模型输入的处理方向。
ax2 = fig.add_subplot(gs[0, 1])
# 横向坐标 0~10。
ax2.set_xlim(0, 10)
# 纵向坐标 0~10。
ax2.set_ylim(0, 10)
# 隐藏坐标轴外观。
ax2.axis('off')
# 设置中文标题。
ax2.set_title('数据流程', fontsize=14, fontweight='bold', pad=20)

# 流程步骤
# 元组依次保存多行节点文字、中心坐标和填充色。
steps = [
# 原始 NIfTI 三维影像。
    ('原始NIfTI\n(.nii.gz)', 1.5, 8.5, 'lightblue'),
# 预处理脚本执行窗宽裁剪、归一化、转轴和格式导出。
    ('预处理\npreprocess_synapse_data.py', 5, 8.5, 'lightgreen'),
# 训练路径使用逐切片 NPZ。
    ('训练数据\n(train_npz/)', 1.5, 6, 'lightyellow'),
# 测试路径保留 H5 体数据。
    ('测试数据\n(test_vol_h5/)', 8.5, 6, 'lightyellow'),
# Dataset/DataLoader 读取训练样本。
    ('数据加载\nSynapse_dataset', 1.5, 3.5, 'lightcoral'),
# RandomGenerator 对图像和标签同步增强。
    ('数据增强\nRandomGenerator', 5, 3.5, 'wheat'),
# 模型接收统一为 224x224 的二维切片。
    ('模型输入\n(224×224)', 8.5, 3.5, 'lightblue'),
]

# 统一绘制所有数据节点。
for text, x, y, color in steps:
# 创建以 (x,y) 为中心的圆角框。
    box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8, boxstyle="round,pad=0.1",
# 颜色来自节点元组，边框统一为黑色。
                         facecolor=color, edgecolor='black', linewidth=1.5)
# 添加节点框。
    ax2.add_patch(box)
# 居中写入节点文字。
    ax2.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 箭头
# 四元组表示每条数据流箭头的起点和终点。
arrows = [
# 原始数据 -> 预处理。
    (1.5, 8.1, 5, 8.9),  # 原始 -> 预处理
# 预处理 -> 训练切片。
    (5, 8.1, 1.5, 6.4),  # 预处理 -> 训练数据
# 预处理 -> 测试体数据。
    (5, 8.1, 8.5, 6.4),  # 预处理 -> 测试数据
# 训练切片 -> Dataset/DataLoader。
    (1.5, 5.6, 1.5, 3.9),  # 训练数据 -> 数据加载
# 数据加载 -> 同步增强。
    (1.5, 3.1, 5, 3.9),  # 数据加载 -> 数据增强
# 数据增强 -> 模型输入。
    (5, 3.1, 8.5, 3.9),  # 数据增强 -> 模型输入
]

# ax.arrow 使用位移而非绝对终点，因此循环中计算 dx、dy。
for x1, y1, x2, y2 in arrows:
# 红色箭头强调处理先后顺序。
    ax2.arrow(x1, y1, x2-x1, y2-y1, head_width=0.2, head_length=0.15, 
# 箭头填充、边缘都设为红色。
             fc='red', ec='red', linewidth=2)

# ========== 3. 网络架构图 ==========
# 第二行横跨两列，人工概括输入适配、PVTv2-B2、EMCAD 和四个预测头。
ax3 = fig.add_subplot(gs[1, :])
# 宽坐标范围允许网络模块从左到右展开。
ax3.set_xlim(0, 20)
# 高度范围用于编码器四级文字、解码模块和输出框。
ax3.set_ylim(0, 8)
# 隐藏坐标轴。
ax3.axis('off')
# 该图对应论文主文 PDF 第4页 Fig.2(a)-(b) 的概念结构，但坐标仅用于展示。
ax3.set_title('EMCAD网络架构', fontsize=16, fontweight='bold', pad=20)

# 输入
# 单通道医学图像输入框。
input_box = FancyBboxPatch((0.5, 6.5), 2, 1, boxstyle="round,pad=0.1",
# 浅蓝色表示网络输入。
                           facecolor='lightblue', edgecolor='black', linewidth=2)
# 添加输入框。
ax3.add_patch(input_box)
# 标注不含 batch 维的输入形状。
ax3.text(1.5, 7, '输入图像\n1×224×224', ha='center', va='center', fontsize=10, fontweight='bold')

# Conv 1->3
# 1x1 卷积把灰度图从 1 通道适配为 ImageNet 编码器需要的 3 通道。
conv_box = FancyBboxPatch((3.5, 6.5), 2, 1, boxstyle="round,pad=0.1",
# 浅绿色表示输入通道变换。
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
# 添加通道转换框。
ax3.add_patch(conv_box)
# 标注 1->3 通道变化；H/W 保持不变。
ax3.text(4.5, 7, 'Conv 1→3\n通道转换', ha='center', va='center', fontsize=10, fontweight='bold')

# 编码器
# 大框汇总 PVTv2-B2 的四级层次化输出。
encoder_box = FancyBboxPatch((6.5, 5), 3, 3, boxstyle="round,pad=0.1",
# 浅黄色区分编码器。
                            facecolor='lightyellow', edgecolor='black', linewidth=2)
# 添加编码器框。
ax3.add_patch(encoder_box)
# 标出默认编码器名称。
ax3.text(8, 7.5, 'PVTv2-B2 编码器', ha='center', va='center', fontsize=11, fontweight='bold')
# x1：1/4 分辨率，64 通道。
ax3.text(8, 6.8, 'x1: 64ch, 56×56', ha='center', va='center', fontsize=9)
# x2：1/8 分辨率，128 通道。
ax3.text(8, 6.3, 'x2: 128ch, 28×28', ha='center', va='center', fontsize=9)
# x3：1/16 分辨率，320 通道。
ax3.text(8, 5.8, 'x3: 320ch, 14×14', ha='center', va='center', fontsize=9)
# x4：1/32 分辨率，512 通道。
ax3.text(8, 5.3, 'x4: 512ch, 7×7', ha='center', va='center', fontsize=9)

# 解码器
# 解码器外框包含四级 MSCAM、三级 EUCB 和三级 LGAG。
decoder_box = FancyBboxPatch((10.5, 1), 6, 5, boxstyle="round,pad=0.1",
# 浅红色突出论文提出的解码区域。
                            facecolor='lightcoral', edgecolor='black', linewidth=2)
# 添加解码器框。
ax3.add_patch(decoder_box)
# 标出 EMCAD 解码器名称。
ax3.text(13.5, 5.5, 'EMCAD 解码器', ha='center', va='center', fontsize=12, fontweight='bold')

# 解码器内部
# 每个元组给出模块标签、示意坐标和颜色；真实顺序以 lib/decoders.py 的 EMCAD.forward 为准。
decoder_steps = [
# 最深层通道/空间注意力和多尺度卷积细化。
    ('MSCAM4\n(CAB+SAB+MSCB)', 11.5, 4, 'wheat'),
# d4 上采样至第三级。
    ('EUCB3\n上采样', 13.5, 4, 'wheat'),
# 用 d4 门控 x3 跳连。
    ('LGAG3\n注意力门控', 15.5, 4, 'wheat'),
# 第三级融合特征细化。
    ('MSCAM3', 11.5, 2.5, 'wheat'),
# 上采样至第二级。
    ('EUCB2', 13.5, 2.5, 'wheat'),
# 门控 x2 跳连。
    ('LGAG2', 15.5, 2.5, 'wheat'),
# 第二级融合特征细化。
    ('MSCAM2', 11.5, 1.5, 'wheat'),
# 上采样至第一级。
    ('EUCB1', 13.5, 1.5, 'wheat'),
# 门控 x1 跳连。
    ('LGAG1', 15.5, 1.5, 'wheat'),
# 最浅层融合特征细化并产生 d1。
    ('MSCAM1', 13.5, 1, 'wheat'),
]

# 逐个绘制解码器内部节点。
for text, x, y, color in decoder_steps:
# 小节点的宽高仅为示意，不代表张量尺寸。
    box = FancyBboxPatch((x-0.5, y-0.3), 1, 0.6, boxstyle="round,pad=0.05",
# 使用统一黑色细边框。
                         facecolor=color, edgecolor='black', linewidth=1)
# 添加节点框。
    ax3.add_patch(box)
# 小字号显示模块缩写。
    ax3.text(x, y, text, ha='center', va='center', fontsize=7)

# 预测头
# 预测头外框概括四个 1x1 分割头及其上采样操作。
head_box = FancyBboxPatch((17.5, 1), 2, 5, boxstyle="round,pad=0.1",
# 浅绿色表示输出投影。
                         facecolor='lightgreen', edgecolor='black', linewidth=2)
# 添加预测头框。
ax3.add_patch(head_box)
# 四个头分别作用于 d4、d3、d2、d1。
ax3.text(18.5, 5.5, '预测头\n(4个1×1 Conv)', ha='center', va='center', fontsize=10, fontweight='bold')
# 返回列表按 [p4,p3,p2,p1] 排列，常规推理取 P[-1] 即 p1。
ax3.text(18.5, 4.5, 'p4, p3, p2, p1', ha='center', va='center', fontsize=9)
# 每个头先映射类别通道，再恢复到输入大小。
ax3.text(18.5, 3.5, '上采样到', ha='center', va='center', fontsize=9)
# 本示意图使用 224x224 输入。
ax3.text(18.5, 3, '224×224', ha='center', va='center', fontsize=9)

# 输出
# 输出框表示 Synapse 的九通道分割结果。
output_box = FancyBboxPatch((17.5, 0.2), 2, 0.6, boxstyle="round,pad=0.1",
# 使用浅蓝色表示网络终点。
                            facecolor='lightblue', edgecolor='black', linewidth=2)
# 添加输出框。
ax3.add_patch(output_box)
# 九类包括背景和八个器官前景类。
ax3.text(18.5, 0.5, '输出\n9类分割', ha='center', va='center', fontsize=10, fontweight='bold')

# 箭头
# 主路径箭头依次连接输入、适配层、编码器、解码器、预测头和输出。
main_arrows = [
# 输入 -> 通道转换。
    (2.5, 7, 3.5, 7),
# 通道转换 -> 编码器。
    (5.5, 7, 6.5, 6.5),
# 编码器 -> 解码器。
    (9.5, 6.5, 10.5, 3.5),
# 解码器 -> 预测头。
    (16.5, 3.5, 17.5, 4),
# 预测头 -> 最终输出。
    (18.5, 1.6, 18.5, 0.8),
]

# 绘制红色主数据流箭头。
for x1, y1, x2, y2 in main_arrows:
# 将绝对终点换算为 dx、dy。
    ax3.arrow(x1, y1, x2-x1, y2-y1, head_width=0.2, head_length=0.15,
# 使用红色粗线。
             fc='red', ec='red', linewidth=2)

# Skip connections
# 蓝色虚线表示编码器四级特征送入解码器的主输入/跳跃连接。
skip_arrows = [
# x4 直接进入最深 MSCAM4。
    (8, 5.3, 11.5, 4.3),  # x4 -> MSCAM4
# x3 进入 LGAG3。
    (8, 5.8, 15.5, 4.3),  # x3 -> LGAG3
# x2 进入 LGAG2。
    (8, 6.3, 15.5, 2.8),  # x2 -> LGAG2
# x1 进入 LGAG1。
    (8, 6.8, 15.5, 1.8),  # x1 -> LGAG1
]

# 绘制每条跳连虚线及末端小箭头。
for x1, y1, x2, y2 in skip_arrows:
# plot 负责虚线主体，alpha 降低视觉权重。
    ax3.plot([x1, x2], [y1, y2], 'b--', linewidth=1.5, alpha=0.6)
# 在终点前添加短箭头，明确跳连方向。
    ax3.arrow(x2-0.3, y2, 0.2, 0, head_width=0.1, head_length=0.1,
# 使用蓝色与虚线匹配。
             fc='blue', ec='blue', linewidth=1)

# ========== 4. 训练流程图 ==========
# 第三行横跨两列，展示模型初始化、训练更新和验证保存关系。
ax4 = fig.add_subplot(gs[2, :])
# 横向排列六个训练节点。
ax4.set_xlim(0, 20)
# 纵向同时容纳主训练链与验证支路。
ax4.set_ylim(0, 6)
# 隐藏坐标轴。
ax4.axis('off')
# 设置训练流程标题。
ax4.set_title('训练流程', fontsize=14, fontweight='bold', pad=20)

# 训练步骤
# 每项定义节点文字、中心坐标和颜色。
train_steps = [
# 创建模型并按配置加载预训练参数。
    ('初始化模型\n加载预训练权重', 2, 5, 'lightblue'),
# DataLoader 读取图像与标签。
    ('加载数据\nDataLoader', 5, 5, 'lightgreen'),
# model(x) 产生四尺度预测。
    ('前向传播\nmodel(x)', 8, 5, 'lightyellow'),
# Synapse 训练使用交叉熵与 Dice 加权损失。
    ('计算损失\nCE + Dice', 11, 5, 'lightcoral'),
# loss.backward() 计算参数梯度。
    ('反向传播\nloss.backward()', 14, 5, 'wheat'),
# optimizer.step() 应用参数更新。
    ('更新参数\noptimizer.step()', 17, 5, 'lightblue'),
# 验证阶段进行无梯度推理。
    ('验证\ninference()', 5, 2.5, 'lightgreen'),
# 根据验证结果保存 best.pth。
    ('保存模型\nbest.pth', 8, 2.5, 'lightyellow'),
# 汇总 Dice、HD95 等指标。
    ('评估指标\nDice, HD95', 11, 2.5, 'lightcoral'),
]

# 统一绘制训练和验证节点。
for text, x, y, color in train_steps:
# 创建宽 1.6、高 0.8 的圆角框。
    box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle="round,pad=0.1",
# 每个节点使用其指定填充色。
                         facecolor=color, edgecolor='black', linewidth=1.5)
# 添加节点框。
    ax4.add_patch(box)
# 居中写入节点文本。
    ax4.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 训练循环箭头
# 顶排六节点之间画五条从左向右的红色箭头。
for i in range(5):
# 当前节点中心。
    x1 = 2 + i * 3
# 下一节点中心。
    x2 = 5 + i * 3
# 箭头长度略短于中心距，避免穿过目标框。
    ax4.arrow(x1, 5, x2-x1-0.2, 0, head_width=0.15, head_length=0.1,
# 红色表示单次训练更新链。
             fc='red', ec='red', linewidth=2)

# epoch循环
# 蓝色虚线从更新端回到数据端，表示跨 epoch 重复。
ax4.arrow(17, 4.6, -12, 0, head_width=0.15, head_length=0.1,
# 虚线样式与主路径区分。
         fc='blue', ec='blue', linewidth=2, linestyle='--')
# 添加循环文字说明。
ax4.text(11, 4.3, '每个epoch循环', ha='center', va='center', fontsize=9, 
# 文字颜色与回路一致。
        color='blue', fontweight='bold')

# 验证箭头
# 从顶排前向节点下接验证分支。
ax4.arrow(8, 4.6, -3, -1.7, head_width=0.15, head_length=0.1,
# 绿色表示验证相关流程。
         fc='green', ec='green', linewidth=2)
# 指标与保存节点的水平关系。
ax4.arrow(11, 2.9, -3, 0, head_width=0.15, head_length=0.1,
# 绿色粗线。
         fc='green', ec='green', linewidth=2)
# 保存节点与指标节点的反向连接保留原图表现。
ax4.arrow(8, 2.9, 3, 0, head_width=0.15, head_length=0.1,
# 绿色粗线。
         fc='green', ec='green', linewidth=2)
# 从验证指标回接主训练链。
ax4.arrow(11, 2.9, 0, 2.1, head_width=0.15, head_length=0.1,
# 绿色粗线。
         fc='green', ec='green', linewidth=2)

# ========== 5. 数据格式说明 ==========
# 第四行左列用卡片说明 Synapse 训练/测试文件组织和类别编号。
ax5 = fig.add_subplot(gs[3, 0])
# 横坐标范围。
ax5.set_xlim(0, 10)
# 纵坐标范围。
ax5.set_ylim(0, 10)
# 隐藏坐标轴。
ax5.axis('off')
# 设置标题。
ax5.set_title('数据格式说明', fontsize=14, fontweight='bold', pad=20)

# 训练数据
# 左上框说明二维训练 NPZ。
train_data_box = FancyBboxPatch((0.5, 7), 4, 2.5, boxstyle="round,pad=0.1",
# 训练数据使用浅黄色。
                               facecolor='lightyellow', edgecolor='black', linewidth=2)
# 添加训练数据框。
ax5.add_patch(train_data_box)
# 写入目录名。
ax5.text(2.5, 8.5, '训练数据 (train_npz/)', ha='center', va='center', 
# 标题使用粗体。
        fontsize=11, fontweight='bold')
# 标明文件扩展名。
ax5.text(2.5, 7.8, '格式: .npz文件', ha='center', va='center', fontsize=9)
# 标明病例与切片编号命名格式。
ax5.text(2.5, 7.3, '命名: caseXXXX_sliceXXX.npz', ha='center', va='center', fontsize=9)
# NPZ 中 image 和 label 均为二维数组。
ax5.text(2.5, 6.8, '内容: {\'image\': 2D数组, \'label\': 2D数组}', ha='center', va='center', fontsize=9)
# 这里的数量是示意说明，不由脚本动态扫描数据得到。
ax5.text(2.5, 6.3, '数量: ~2200+切片', ha='center', va='center', fontsize=9)

# 测试数据
# 右上框说明三维测试 H5。
test_data_box = FancyBboxPatch((5.5, 7), 4, 2.5, boxstyle="round,pad=0.1",
# 测试数据使用浅绿色。
                              facecolor='lightgreen', edgecolor='black', linewidth=2)
# 添加测试数据框。
ax5.add_patch(test_data_box)
# 写入测试目录名。
ax5.text(7.5, 8.5, '测试数据 (test_vol_h5/)', ha='center', va='center',
# 标题使用粗体。
        fontsize=11, fontweight='bold')
# 标明 H5 格式。
ax5.text(7.5, 7.8, '格式: .h5文件', ha='center', va='center', fontsize=9)
# 标明体数据命名约定。
ax5.text(7.5, 7.3, '命名: caseXXXX.npy.h5', ha='center', va='center', fontsize=9)
# H5 中 image 和 label 为三维体数组。
ax5.text(7.5, 6.8, '内容: {\'image\': 3D数组, \'label\': 3D数组}', ha='center', va='center', fontsize=9)
# 12 个测试体是 Synapse 常用划分说明，脚本本身不校验数量。
ax5.text(7.5, 6.3, '数量: 12个体积', ha='center', va='center', fontsize=9)

# 类别信息
# 下方宽框集中说明标签 0~8 的语义。
class_box = FancyBboxPatch((0.5, 3.5), 9, 2.5, boxstyle="round,pad=0.1",
# 浅红色突出标签字典。
                          facecolor='lightcoral', edgecolor='black', linewidth=2)
# 添加类别框。
ax5.add_patch(class_box)
# 标题说明这是九类语义分割。
ax5.text(5, 5.5, 'Synapse数据集 - 9类器官分割', ha='center', va='center',
# 标题使用较大粗体。
        fontsize=12, fontweight='bold')
# 第一行列出背景、脾脏、左右肾、胆囊。
ax5.text(5, 4.8, '0: 背景 | 1: 脾脏 | 2: 右肾 | 3: 左肾 | 4: 胆囊', ha='center', va='center', fontsize=9)
# 第二行列出胰腺、肝脏、胃和主动脉。
ax5.text(5, 4.3, '5: 胰腺 | 6: 肝脏 | 7: 胃 | 8: 主动脉', ha='center', va='center', fontsize=9)

# ========== 6. 关键模块说明 ==========
# 第四行右列列出阅读模型代码时需要识别的模块与训练概念。
ax6 = fig.add_subplot(gs[3, 1])
# 横坐标范围。
ax6.set_xlim(0, 10)
# 纵坐标范围。
ax6.set_ylim(0, 10)
# 隐藏坐标轴。
ax6.axis('off')
# 设置标题。
ax6.set_title('关键模块说明', fontsize=14, fontweight='bold', pad=20)

# 每个元组是模块文字、中心坐标和颜色。
modules = [
# MSCB：扩张、MSDC 多尺度深度卷积、投影和残差。
    ('MSCB\n多尺度卷积块', 2, 8.5, 'lightblue'),
# EUCB：深度卷积、二倍上采样和 1x1 通道投影。
    ('EUCB\n高效上采样', 5, 8.5, 'lightgreen'),
# LGAG：用解码门控信号筛选编码跳连。
    ('LGAG\n大核注意力门控', 8, 8.5, 'lightyellow'),
# CAB：生成通道维权重。
    ('CAB\n通道注意力', 2, 6, 'lightcoral'),
# SAB：生成空间位置权重。
    ('SAB\n空间注意力', 5, 6, 'wheat'),
# MSCAM：代码中由 CAB、SAB、MSCB 顺序组合得到。
    ('MSCAM\n多尺度注意力模块', 8, 6, 'lightblue'),
# Mutation supervision：四个输出的非空组合监督。
    ('Mutation\nSupervision', 2, 3.5, 'lightgreen'),
# Synapse 每项监督采用 CE 与 Dice 加权。
    ('Dice Loss\n+ CE Loss', 5, 3.5, 'lightyellow'),
# 区域与边界指标。
    ('评估指标\nDice, HD95', 8, 3.5, 'lightcoral'),
]

# 统一绘制九个关键概念节点。
for text, x, y, color in modules:
# 每个框宽 1.4、高 0.8。
    box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8, boxstyle="round,pad=0.1",
# 用颜色区分概念类别。
                         facecolor=color, edgecolor='black', linewidth=1.5)
# 添加模块框。
    ax6.add_patch(box)
# 居中写入模块缩写与中文含义。
    ax6.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 保存图片
# 导出完整六区主图；执行脚本时才实际创建 PNG。
plt.savefig('项目可视化图表.png', dpi=300, bbox_inches='tight', facecolor='white')
# 在标准输出报告主图保存路径。
print("✅ 可视化图表已保存为: 项目可视化图表.png")

# 也保存一个简化版本
# 另建 2x2 子图，生成更紧凑的核心流程总览。
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
# 设置整张简化图的总标题；y=0.98 控制标题靠近画布顶部。
fig2.suptitle('EMCAD项目核心流程图', fontsize=16, fontweight='bold', y=0.98)

# 简化版：执行流程图
# 左上子图表示训练脚本从启动到权重保存的主路径。
ax = axes[0, 0]
# 固定人工布局横轴。
ax.set_xlim(0, 10)
# 固定人工布局纵轴。
ax.set_ylim(0, 6)
# 隐藏坐标轴。
ax.axis('off')
# 设置子图标题。
ax.set_title('训练执行流程', fontsize=12, fontweight='bold')

# 元组包含节点文本与中心坐标。
flow = [
# 命令行启动训练入口。
    ('python train_synapse.py', 2, 5),
# 构建 Dataset/DataLoader。
    ('加载数据', 5, 5),
# 执行 epoch/batch 训练。
    ('训练模型', 8, 5),
# 写出 best/last/epoch 检查点。
    ('保存权重', 5, 2.5),
]

# 逐项绘制训练流程节点。
for text, x, y in flow:
# 节点宽 2、高 1。
    box = FancyBboxPatch((x-1, y-0.5), 2, 1, boxstyle="round,pad=0.1",
# 全部训练节点使用浅蓝填充。
                         facecolor='lightblue', edgecolor='black', linewidth=2)
# 添加节点框。
    ax.add_patch(box)
# 居中写入文本。
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

# 根据 flow 顺序连接相邻节点。
for i in range(len(flow)-1):
# 前两段位于顶排，使用水平箭头。
    if i < 2:
# 从当前框右边缘指向下一框附近。
        ax.arrow(flow[i][1]+1, flow[i][2], flow[i+1][1]-flow[i][1]-0.2, 0,
# 红色箭头表示执行顺序。
                head_width=0.2, head_length=0.15, fc='red', ec='red', linewidth=2)
# 最后一段从训练节点斜向下连接保存权重节点。
    else:
# 固定向左下方位移。
        ax.arrow(flow[i][1], flow[i][2]-0.5, -3, -2, head_width=0.2, head_length=0.15,
# 沿用红色主流程样式。
                fc='red', ec='red', linewidth=2)

# 简化版：测试流程
# 右上子图表示测试入口、模型加载、切片推理、指标和结果保存。
ax = axes[0, 1]
# 固定横轴。
ax.set_xlim(0, 10)
# 固定纵轴。
ax.set_ylim(0, 6)
# 隐藏坐标轴。
ax.axis('off')
# 设置标题。
ax.set_title('测试执行流程', fontsize=12, fontweight='bold')

# 测试流程节点及其中心坐标。
test_flow = [
# 启动测试脚本。
    ('python test_synapse.py', 2, 5),
# 从 checkpoint 载入模型参数。
    ('加载模型', 5, 5),
# 对三维体逐二维切片推理并重组。
    ('逐切片推理', 8, 5),
# 计算逐类和平均指标。
    ('计算指标', 5, 2.5),
# 保存预测或日志结果。
    ('保存结果', 8, 2.5),
]

# 绘制所有测试节点。
for text, x, y in test_flow:
# 节点宽 2、高 1。
    box = FancyBboxPatch((x-1, y-0.5), 2, 1, boxstyle="round,pad=0.1",
# 浅绿色区分测试流程。
                         facecolor='lightgreen', edgecolor='black', linewidth=2)
# 添加节点框。
    ax.add_patch(box)
# 写入节点文字。
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

# 简化版：数据流向
# 左下子图只保留原始数据、预处理、训练/测试格式和模型五个节点。
ax = axes[1, 0]
# 固定横轴。
ax.set_xlim(0, 10)
# 固定纵轴。
ax.set_ylim(0, 6)
# 隐藏坐标轴。
ax.axis('off')
# 设置标题。
ax.set_title('数据流向', fontsize=12, fontweight='bold')

# 数据流节点文字与位置。
data_flow = [
# NIfTI 原始数据。
    ('原始数据\n.nii.gz', 1.5, 4.5),
# 预处理阶段。
    ('预处理', 4.5, 4.5),
# 二维 NPZ 训练数据。
    ('训练\n.npz', 1.5, 2),
# H5 测试体数据。
    ('测试\n.h5', 7.5, 2),
# 最终送入模型。
    ('模型', 4.5, 2),
]

# 绘制数据流节点；原代码未在本子图额外绘制连接箭头。
for text, x, y in data_flow:
# 节点宽 1.6、高 0.8。
    box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle="round,pad=0.1",
# 浅黄色表示数据对象或处理步骤。
                         facecolor='lightyellow', edgecolor='black', linewidth=1.5)
# 添加节点框。
    ax.add_patch(box)
# 居中写入文字。
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 简化版：网络输入输出
# 右下子图压缩显示输入、PVTv2-B2、EMCAD 和九类输出。
ax = axes[1, 1]
# 固定横轴。
ax.set_xlim(0, 10)
# 固定纵轴。
ax.set_ylim(0, 6)
# 隐藏坐标轴。
ax.axis('off')
# 设置标题。
ax.set_title('网络输入输出', fontsize=12, fontweight='bold')

# 网络 I/O 节点文字与坐标。
io_flow = [
# 单通道二维输入。
    ('输入\n1×224×224', 2, 4),
# 层次化编码器。
    ('编码器\nPVTv2-B2', 5, 4),
# EMCAD 解码器。
    ('解码器\nEMCAD', 8, 4),
# 九通道全分辨率输出。
    ('输出\n9×224×224', 5, 1.5),
]

# 绘制四个网络 I/O 节点。
for text, x, y in io_flow:
# 节点宽 1.6、高 0.8。
    box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle="round,pad=0.1",
# 浅红色统一表示网络模块。
                         facecolor='lightcoral', edgecolor='black', linewidth=1.5)
# 添加节点框。
    ax.add_patch(box)
# 居中写入形状或模块名。
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 连接输入、编码器、解码器和输出。
for i in range(len(io_flow)-1):
# 前两段是顶排水平箭头。
    if i < 2:
# 从当前框右边缘指向下一模块。
        ax.arrow(io_flow[i][1]+0.8, io_flow[i][2], io_flow[i+1][1]-io_flow[i][1]-0.2, 0,
# 红色箭头表示前向数据流。
                head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
# 最后一段从解码器斜向下指向输出。
    else:
# 固定向左下的位移匹配输出框位置。
        ax.arrow(io_flow[i][1], io_flow[i][2]-0.4, -3, -2.1,
# 沿用红色箭头样式。
                head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)

# 自动调整四个子图间距，降低文字和坐标轴区域重叠风险。
plt.tight_layout()
# 导出简化版核心流程图；执行脚本才会写文件。
plt.savefig('项目核心流程图.png', dpi=300, bbox_inches='tight', facecolor='white')
# 在标准输出提示第二张图的文件名。
print("✅ 核心流程图已保存为: 项目核心流程图.png")




