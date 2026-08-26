"""
生成简化的项目可视化图表（不依赖中文字体）
"""
# pyplot 管理 Figure、Axes、布局以及最终 PNG 导出。
import matplotlib.pyplot as plt
# 该别名在当前脚本中没有被后续语句使用，仍保留原导入以避免改动代码。
import matplotlib.patches as mpatches
# FancyBboxPatch 绘制圆角模块框；FancyArrowPatch 当前未直接使用。
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
# NumPy 当前未被后续语句使用，属于原脚本保留导入。
import numpy as np

# 设置字体
# 简化版所有可见文字均使用英文，DejaVu Sans 可减少运行环境缺少中文字体的问题。
plt.rcParams['font.family'] = 'DejaVu Sans'

# 创建图表
# 建立 18x12 英寸总画布，容纳项目结构、数据流、网络结构和训练流程四部分。
fig = plt.figure(figsize=(18, 12))
# 三行两列网格：前两图各占半行，网络和训练图分别横跨整行。
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# ========== 1. 项目结构图 ==========
# 在第一行左列创建项目结构坐标轴。
ax1 = fig.add_subplot(gs[0, 0])
# 固定横坐标范围，后续所有框的 x 坐标都按 0~10 设计。
ax1.set_xlim(0, 10)
# 固定纵坐标范围，顶部根目录位于 y=8 附近。
ax1.set_ylim(0, 10)
# 隐藏刻度和边框，使坐标轴只作为绘图画布。
ax1.axis('off')
# 设置该子图标题及与绘图区的间距。
ax1.set_title('Project Structure', fontsize=14, fontweight='bold', pad=20)

# 根目录
# 创建代表仓库根目录的浅蓝色圆角框。
root = FancyBboxPatch((1, 8), 8, 1.5, boxstyle="round,pad=0.1",
                      # 指定填充色、边框色和边框宽度。
                      facecolor='lightblue', edgecolor='black', linewidth=2)
# 把根目录框加入项目结构坐标轴。
ax1.add_patch(root)
# 在框中心标出仓库目录名。
ax1.text(5, 8.75, 'SLDGroup_EMCAD/', ha='center', va='center', fontsize=12, fontweight='bold')

# 主要文件
# 每个元组依次保存显示文本、中心 x、中心 y 和填充色。
files = [
    # Synapse 训练入口。
    ('train_synapse.py', 2, 6.5, 'lightgreen'),
    # Synapse 测试入口。
    ('test_synapse.py', 5, 6.5, 'lightgreen'),
    # Synapse 训练循环实现。
    ('trainer.py', 8, 6.5, 'lightyellow'),
    # 模型结构目录。
    ('lib/', 2, 5, 'lightcoral'),
    # 数据、损失和评估工具目录。
    ('utils/', 5, 5, 'lightcoral'),
    # 图中假定的 Synapse 数据目录。
    ('data/Synapse/', 8, 5, 'lightcoral'),
]

# 遍历元组，用统一样式绘制每个一级文件或目录节点。
for name, x, y, color in files:
    # 以给定中心点为基准，创建宽 0.8、高 0.6 的节点框。
    box = FancyBboxPatch((x - 0.4, y - 0.3), 0.8, 0.6, boxstyle="round,pad=0.05",
                         # 元组中的 color 控制填充色，其余边框样式统一。
                         facecolor=color, edgecolor='black', linewidth=1)
    # 将节点框加入 ax1。
    ax1.add_patch(box)
    # 在节点框中心写入文件或目录名。
    ax1.text(x, y, name, ha='center', va='center', fontsize=9)

# lib子目录
# 列出图中要展开显示的三个核心模型文件。
lib_files = ['networks.py', 'decoders.py', 'pvtv2.py']
# enumerate 同时提供从 0 开始的纵向偏移索引和文件名。
for i, f in enumerate(lib_files):
    # 每个后续条目向下移动 0.8，形成竖直列表。
    box = FancyBboxPatch((1.5, 3.5 - i * 0.8), 1.5, 0.5, boxstyle="round,pad=0.05",
                         # 子节点统一使用 wheat 填充。
                         facecolor='wheat', edgecolor='black', linewidth=1)
    # 添加 lib 子文件框。
    ax1.add_patch(box)
    # 文本 y 坐标与框位置使用相同的 0.8 间距。
    ax1.text(2.25, 3.75 - i * 0.8, f, ha='center', va='center', fontsize=8)

# utils子目录
# 选取数据集、通用工具和预处理三个代表性文件显示。
utils_files = ['dataset_synapse.py', 'utils.py', 'preprocess.py']
# 依次绘制 utils 子节点。
for i, f in enumerate(utils_files):
    # utils 列的中心约在 x=5.25。
    box = FancyBboxPatch((4.5, 3.5 - i * 0.8), 1.5, 0.5, boxstyle="round,pad=0.05",
                         # 沿用子节点统一样式。
                         facecolor='wheat', edgecolor='black', linewidth=1)
    # 添加节点框。
    ax1.add_patch(box)
    # 写入文件名。
    ax1.text(5.25, 3.75 - i * 0.8, f, ha='center', va='center', fontsize=8)

# data子目录
# 显示训练切片、测试体数据和病例列表三个数据组织节点。
data_files = ['train_npz/', 'test_vol_h5/', 'lists/']
# 依次绘制 data 子节点。
for i, f in enumerate(data_files):
    # data 列的中心约在 x=8.25。
    box = FancyBboxPatch((7.5, 3.5 - i * 0.8), 1.5, 0.5, boxstyle="round,pad=0.05",
                         # 沿用相同填充和边框。
                         facecolor='wheat', edgecolor='black', linewidth=1)
    # 添加节点框。
    ax1.add_patch(box)
    # 写入目录名。
    ax1.text(8.25, 3.75 - i * 0.8, f, ha='center', va='center', fontsize=8)

# 箭头
# 在 lib、utils、data 三列上方各画一支向下箭头，表示父目录到子项的层级关系。
for x in [2, 5, 8]:
    # 箭头从 y=4.7 向下延伸 0.5，头宽/头长采用固定比例。
    ax1.arrow(x, 4.7, 0, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black')

# ========== 2. 数据流程图 ==========
# 第一行右列用于展示 Synapse 从原始 NIfTI 到模型输入的数据链。
ax2 = fig.add_subplot(gs[0, 1])
# 数据流图同样使用 0~10 横坐标。
ax2.set_xlim(0, 10)
# 纵坐标 0~10 为节点和箭头预留空间。
ax2.set_ylim(0, 10)
# 隐藏坐标轴视觉元素。
ax2.axis('off')
# 使用英文标题以配合不依赖中文字体的简化版目标。
ax2.set_title('Data Flow', fontsize=14, fontweight='bold', pad=20)

# 流程步骤
# 每个元组保存节点文本、中心位置和颜色；文本中的 \n 强制分成两行。
steps = [
    # 原始三维医学影像输入。
    ('Raw NIfTI\n.nii.gz', 1.5, 8.5, 'lightblue'),
    # 预处理脚本负责窗宽裁剪、归一化和轴调整。
    ('Preprocess\npreprocess_synapse_data.py', 5, 8.5, 'lightgreen'),
    # 训练阶段读取二维 NPZ 切片。
    ('Train Data\n(train_npz/)', 1.5, 6, 'lightyellow'),
    # 测试阶段保留 H5 三维体并逐切片推理。
    ('Test Data\n(test_vol_h5/)', 8.5, 6, 'lightyellow'),
    # Dataset/DataLoader 把磁盘样本组织成 batch。
    ('DataLoader\nSynapse_dataset', 1.5, 3.5, 'lightcoral'),
    # RandomGenerator 执行训练时的随机几何变换与缩放。
    ('Augmentation\nRandomGenerator', 5, 3.5, 'wheat'),
    # 图中用 224x224 表示送入模型的统一二维空间尺寸。
    ('Model Input\n(224x224)', 8.5, 3.5, 'lightblue'),
]

# 为每个数据流程节点创建相同大小的圆角框。
for text, x, y, color in steps:
    # 节点宽 1.4、高 0.8，左下角由中心点减去半宽/半高得到。
    box = FancyBboxPatch((x - 0.7, y - 0.4), 1.4, 0.8, boxstyle="round,pad=0.1",
                         # color 区分节点角色，黑色边框保持可读性。
                         facecolor=color, edgecolor='black', linewidth=1.5)
    # 把框加入数据流子图。
    ax2.add_patch(box)
    # 节点文本水平、垂直居中并使用粗体。
    ax2.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 箭头
# 每个四元组是箭头起点 (x1,y1) 和终点 (x2,y2)。
arrows = [
    # 原始数据指向预处理。
    (1.5, 8.1, 5, 8.9),
    # 预处理结果分出训练切片路径。
    (5, 8.1, 1.5, 6.4),
    # 预处理结果同时分出测试体数据路径。
    (5, 8.1, 8.5, 6.4),
    # 训练切片进入 Dataset/DataLoader。
    (1.5, 5.6, 1.5, 3.9),
    # DataLoader 节点指向训练增强。
    (1.5, 3.1, 5, 3.9),
    # 增强后的张量成为统一尺寸模型输入。
    (5, 3.1, 8.5, 3.9),
]

# 把绝对终点换算为 ax.arrow 所需的 dx、dy。
for x1, y1, x2, y2 in arrows:
    # 红色粗箭头强调数据处理顺序。
    ax2.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.2, head_length=0.15,
              # 箭身和箭头均使用红色。
              fc='red', ec='red', linewidth=2)

# ========== 3. 网络架构图 ==========
# 第二行横跨两列，用更宽画布表现输入、编码器、EMCAD 解码器和输出头。
ax3 = fig.add_subplot(gs[1, :])
# 横向范围扩展到 20，以容纳完整网络流水线。
ax3.set_xlim(0, 20)
# 纵向范围 0~8 用于放置四级特征说明和解码模块。
ax3.set_ylim(0, 8)
# 隐藏坐标轴。
ax3.axis('off')
# 该图是脚本中最直接对应论文 Fig.2(a)-(b) 的概览，但这里只是人工示意，不参与模型计算。
ax3.set_title('EMCAD Network Architecture', fontsize=16, fontweight='bold', pad=20)

# 输入
# 输入框表示单通道 224x224 医学图像张量。
input_box = FancyBboxPatch((0.5, 6.5), 2, 1, boxstyle="round,pad=0.1",
                           # 浅蓝填充、黑色粗边框突出流水线起点。
                           facecolor='lightblue', edgecolor='black', linewidth=2)
# 添加输入框。
ax3.add_patch(input_box)
# 标出不含 batch 维的 CxHxW 形状。
ax3.text(1.5, 7, 'Input\n1x224x224', ha='center', va='center', fontsize=10, fontweight='bold')

# Conv 1->3
# 1x1 输入适配层把灰度通道转换为预训练编码器期望的三通道。
conv_box = FancyBboxPatch((3.5, 6.5), 2, 1, boxstyle="round,pad=0.1",
                          # 用浅绿区分通道适配操作。
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
# 添加通道转换框。
ax3.add_patch(conv_box)
# 文本说明通道数由 1 变 3，空间尺寸不变。
ax3.text(4.5, 7, 'Conv 1->3\nChannel Convert', ha='center', va='center', fontsize=10, fontweight='bold')

# 编码器
# 大框概括 PVTv2-B2 的四级层次化特征输出。
encoder_box = FancyBboxPatch((6.5, 5), 3, 3, boxstyle="round,pad=0.1",
                             # 浅黄填充代表编码器区域。
                             facecolor='lightyellow', edgecolor='black', linewidth=2)
# 添加编码器框。
ax3.add_patch(encoder_box)
# 标出默认骨干名称。
ax3.text(8, 7.5, 'PVTv2-B2 Encoder', ha='center', va='center', fontsize=11, fontweight='bold')
# x1 是 1/4 分辨率、64 通道的浅层特征。
ax3.text(8, 6.8, 'x1: 64ch, 56x56', ha='center', va='center', fontsize=9)
# x2 是 1/8 分辨率、128 通道特征。
ax3.text(8, 6.3, 'x2: 128ch, 28x28', ha='center', va='center', fontsize=9)
# x3 是 1/16 分辨率、320 通道特征。
ax3.text(8, 5.8, 'x3: 320ch, 14x14', ha='center', va='center', fontsize=9)
# x4 是 1/32 分辨率、512 通道最深特征。
ax3.text(8, 5.3, 'x4: 512ch, 7x7', ha='center', va='center', fontsize=9)

# 解码器
# 解码器区域包含各级 MSCAM、EUCB 上采样和 LGAG 跳连门控。
decoder_box = FancyBboxPatch((10.5, 1), 6, 5, boxstyle="round,pad=0.1",
                             # 浅红填充把解码器与编码器区分开。
                             facecolor='lightcoral', edgecolor='black', linewidth=2)
# 添加解码器外框。
ax3.add_patch(decoder_box)
# 标出论文提出的 EMCAD 解码器。
ax3.text(13.5, 5.5, 'EMCAD Decoder', ha='center', va='center', fontsize=12, fontweight='bold')

# 解码器内部
# 元组按“模块名、中心坐标、颜色”描述示意节点；真实执行顺序以 lib/decoders.py 为准。
decoder_steps = [
    # 最深层 x4 先经 CAB、SAB、MSCB 组成的 MSCAM4。
    ('MSCAM4', 11.5, 4, 'wheat'),
    # EUCB3 把 d4 上采样并投影到第三级通道。
    ('EUCB3', 13.5, 4, 'wheat'),
    # LGAG3 用上采样门控信号筛选 x3 跳连。
    ('LGAG3', 15.5, 4, 'wheat'),
    # 融合后的第三级特征经 MSCAM3 细化。
    ('MSCAM3', 11.5, 2.5, 'wheat'),
    # EUCB2 继续上采样到第二级。
    ('EUCB2', 13.5, 2.5, 'wheat'),
    # LGAG2 筛选 x2 跳连。
    ('LGAG2', 15.5, 2.5, 'wheat'),
    # 第二级融合特征经 MSCAM2 细化。
    ('MSCAM2', 11.5, 1.5, 'wheat'),
    # EUCB1 上采样到最浅解码级。
    ('EUCB1', 13.5, 1.5, 'wheat'),
    # LGAG1 筛选 x1 跳连。
    ('LGAG1', 15.5, 1.5, 'wheat'),
    # 最终浅层融合结果经 MSCAM1 输出 d1。
    ('MSCAM1', 13.5, 1, 'wheat'),
]

# 逐个绘制解码器内部模块节点。
for text, x, y, color in decoder_steps:
    # 每个内部节点宽 1、高 0.6，属于概念位置而非按真实张量比例绘制。
    box = FancyBboxPatch((x - 0.5, y - 0.3), 1, 0.6, boxstyle="round,pad=0.05",
                         # 所有内部节点采用统一边框。
                         facecolor=color, edgecolor='black', linewidth=1)
    # 添加模块框。
    ax3.add_patch(box)
    # 用较小字号保证缩写适配紧凑节点。
    ax3.text(x, y, text, ha='center', va='center', fontsize=7)

# 预测头
# 右侧大框概括 d4、d3、d2、d1 上的四个 1x1 分割头。
head_box = FancyBboxPatch((17.5, 1), 2, 5, boxstyle="round,pad=0.1",
                          # 浅绿色表示预测阶段。
                          facecolor='lightgreen', edgecolor='black', linewidth=2)
# 添加预测头区域。
ax3.add_patch(head_box)
# 标题说明这是多尺度预测头集合。
ax3.text(18.5, 5.5, 'Prediction\nHeads', ha='center', va='center', fontsize=10, fontweight='bold')
# 代码返回顺序为 [p4,p3,p2,p1]，其中 P[-1] 即 p1 用于常规推理。
ax3.text(18.5, 4.5, 'p4, p3, p2, p1', ha='center', va='center', fontsize=9)
# 四个 logits 都会被恢复到输入空间大小。
ax3.text(18.5, 3.5, 'Upsample to', ha='center', va='center', fontsize=9)
# 当前示意输入对应 224x224 输出。
ax3.text(18.5, 3, '224x224', ha='center', va='center', fontsize=9)

# 输出
# 最下方输出框表示 9 类 Synapse 分割 logits/标签空间。
output_box = FancyBboxPatch((17.5, 0.2), 2, 0.6, boxstyle="round,pad=0.1",
                            # 输出重新使用浅蓝色，与输入形成首尾呼应。
                            facecolor='lightblue', edgecolor='black', linewidth=2)
# 添加输出框。
ax3.add_patch(output_box)
# 9 类包含背景通道和 8 个前景器官类别。
ax3.text(18.5, 0.5, 'Output\n9 classes', ha='center', va='center', fontsize=10, fontweight='bold')

# 箭头
# 五段主箭头依次连接输入、通道适配、编码器、解码器、预测头和输出。
main_arrows = [
    # 输入 -> 1x1 通道适配。
    (2.5, 7, 3.5, 7),
    # 通道适配 -> PVTv2 编码器。
    (5.5, 7, 6.5, 6.5),
    # 编码器最深语义 -> EMCAD 解码器。
    (9.5, 6.5, 10.5, 3.5),
    # 解码器多尺度特征 -> 分割头。
    (16.5, 3.5, 17.5, 4),
    # 分割头 -> 最终类别输出示意。
    (18.5, 1.6, 18.5, 0.8),
]

# 将主路径坐标转为红色箭头。
for x1, y1, x2, y2 in main_arrows:
    # dx=x2-x1、dy=y2-y1 指定箭头方向和长度。
    ax3.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.2, head_length=0.15,
              # 主网络数据流使用红色粗线。
              fc='red', ec='red', linewidth=2)

# ========== 4. 训练流程图 ==========
# 第三行横跨两列，展示训练迭代、验证、指标与检查点之间的关系。
ax4 = fig.add_subplot(gs[2, :])
# 训练步骤横向排列，需要 0~20 的横轴。
ax4.set_xlim(0, 20)
# 纵向 0~6 同时容纳主训练链和下方验证链。
ax4.set_ylim(0, 6)
# 隐藏坐标轴。
ax4.axis('off')
# 设置训练流程标题。
ax4.set_title('Training Flow', fontsize=14, fontweight='bold', pad=20)

# 训练步骤
# 每个元组定义训练/验证节点的文字、中心坐标和颜色。
train_steps = [
    # 初始化 EMCADNet，并按配置加载编码器预训练权重或断点。
    ('Init Model\nLoad Pretrain', 2, 5, 'lightblue'),
    # DataLoader 产生图像和分割标签 batch。
    ('Load Data\nDataLoader', 5, 5, 'lightgreen'),
    # 前向传播得到四个尺度的 logits。
    ('Forward\nmodel(x)', 8, 5, 'lightyellow'),
    # Synapse 监督项使用 CE 与 Dice 的加权组合。
    ('Loss\nCE + Dice', 11, 5, 'lightcoral'),
    # 自动求导把损失梯度传播到可训练参数。
    ('Backward\nloss.backward()', 14, 5, 'wheat'),
    # 优化器根据梯度更新模型权重。
    ('Update\noptimizer.step()', 17, 5, 'lightblue'),
    # 验证阶段关闭梯度并计算病例级预测。
    ('Validation\ninference()', 5, 2.5, 'lightgreen'),
    # 根据验证指标保存 best.pth 等检查点。
    ('Save Model\nbest.pth', 8, 2.5, 'lightyellow'),
    # Dice、HD95 等衡量区域重合与边界距离。
    ('Metrics\nDice, HD95', 11, 2.5, 'lightcoral'),
]

# 统一绘制全部训练与验证节点。
for text, x, y, color in train_steps:
    # 每个节点宽 1.6、高 0.8。
    box = FancyBboxPatch((x - 0.8, y - 0.4), 1.6, 0.8, boxstyle="round,pad=0.1",
                         # 使用元组颜色和统一黑色边框。
                         facecolor=color, edgecolor='black', linewidth=1.5)
    # 添加节点框。
    ax4.add_patch(box)
    # 在框内居中显示两行文字。
    ax4.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# 训练循环箭头
# 前六个顶排节点之间共有五段从左到右的主训练箭头。
for i in range(5):
    # 当前节点中心 x 坐标按 3 的间距排列。
    x1 = 2 + i * 3
    # 下一节点中心 x 坐标。
    x2 = 5 + i * 3
    # 减去 0.2 让箭头尖端不要过度进入下一个框。
    ax4.arrow(x1, 5, x2 - x1 - 0.2, 0, head_width=0.15, head_length=0.1,
              # 红色表示一次 batch 内的正向、损失、反向和更新顺序。
              fc='red', ec='red', linewidth=2)

# epoch循环
# 蓝色虚线从更新节点返回数据加载区域，表示下一个 epoch/迭代周期。
ax4.arrow(17, 4.6, -12, 0, head_width=0.15, head_length=0.1,
          # linestyle='--' 将回路与主红色流程区分。
          fc='blue', ec='blue', linewidth=2, linestyle='--')
# 在回路线下方标注 epoch 循环含义。
ax4.text(11, 4.3, 'Each epoch loop', ha='center', va='center', fontsize=9,
         # 蓝色粗体与虚线箭头匹配。
         color='blue', fontweight='bold')

# 验证箭头
# 从前向/训练区域向下连接验证节点。
ax4.arrow(8, 4.6, -3, -1.7, head_width=0.15, head_length=0.1,
          # 绿色统一表示验证与评估支路。
          fc='green', ec='green', linewidth=2)
# 指标节点与保存节点之间的水平连接示意。
ax4.arrow(11, 2.9, -3, 0, head_width=0.15, head_length=0.1,
          # 沿用绿色验证样式。
          fc='green', ec='green', linewidth=2)
# 保存节点到指标节点的另一方向连接保留原图布局。
ax4.arrow(8, 2.9, 3, 0, head_width=0.15, head_length=0.1,
          # 绿色粗线。
          fc='green', ec='green', linewidth=2)
# 指标节点重新连回顶排损失/训练链，表示按 epoch 重复评估。
ax4.arrow(11, 2.9, 0, 2.1, head_width=0.15, head_length=0.1,
          # 绿色粗线。
          fc='green', ec='green', linewidth=2)

# 保存图片
# 将整个 Figure 导出为 300 DPI PNG；执行本脚本才会在当前目录写入该文件。
plt.savefig('project_visualization.png', dpi=300, bbox_inches='tight', facecolor='white')
# 在终端提示导出文件名。
print("Visualization saved as: project_visualization.png")
