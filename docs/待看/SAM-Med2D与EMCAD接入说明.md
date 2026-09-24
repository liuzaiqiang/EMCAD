# SAM-Med2D 是什么，以及如何接入当前 EMCAD 项目

本文回答两个问题：

1. SAM-Med2D 的定位、结构和输入输出是什么？
2. 它能不能接入当前项目？如果能，哪一种接入方式最稳妥？

本文的代码结论以当前 checkout `D:\files\Medical_Image_Segmentation_Projects\SLDGroup_EMCAD` 为准。当前仓库没有 SAM-Med2D 的源码、权重或专用依赖，所以本文不把“已经接入”与“可以设计接入方案”混为一谈。

## 一、先给结论

可以接入，但不能把 SAM-Med2D 的 checkpoint 直接塞进 `EMCADNet`，也不能只改一行 `num_classes` 就完成接入。

最推荐的顺序是：

1. **标注辅助/伪标签路线**：用 SAM-Med2D 生成或修正 2D 器官掩膜，再把结果转换为当前 Synapse 的 `0..8` 标签，继续训练 EMCAD。改动最小，最适合当前项目。
2. **推理级联路线**：先用 EMCAD 得到粗分割，再将每个器官的粗掩膜转换成框或点提示，交给 SAM-Med2D 细化。需要修改测试流程，适合做增强实验。
3. **知识蒸馏路线**：SAM-Med2D 作为教师，EMCAD 作为学生，在原有 CE+Dice 之外增加教师掩膜约束。研究价值较高，但实现和实验控制更复杂。
4. **直接替换 EMCAD 编码器路线**：理论上可以，但不建议作为第一步。SAM-Med2D 的图像编码器、提示编码器、掩膜解码器与 EMCAD 的 PVT/ResNet + EMCAD 解码器接口不同，现有 EMCAD 权重基本不能直接复用。

一句话概括：**SAM-Med2D 更像一个“需要提示、返回掩膜的医学图像基础模型”；EMCAD 更像一个“输入图像、直接返回固定类别分割 logits 的监督式语义分割网络”。二者能协作，但不是同一种模型接口。**

## 二、SAM-Med2D 是什么

### 2.1 名字的含义

SAM-Med2D 可以理解为：面向医学图像二维切片适配的 Segment Anything Model。它继承了 SAM 的提示式分割思想，并使用医学图像数据进行适配或训练，使模型更适合 CT、MRI、超声、内镜、病理等医学图像中的目标分割。

这里的 `2D` 很重要：它的基本推理单位是二维图像或二维切片，不是直接对一个 Synapse 三维 CT 体数据输出三维标签。若用于 Synapse，通常要逐切片推理，再把切片重新堆叠成三维结果。

### 2.2 它与普通语义分割网络的区别

普通语义分割网络一般是：

```text
图像 -> 网络 -> 每个像素的固定类别 logits -> argmax -> 类别图
```

例如当前 EMCAD 的 Synapse 任务：

```text
单通道 CT 切片 -> EMCADNet -> 9 个类别通道
                                  0 背景
                                  1..8 八个器官
```

SAM-Med2D 更接近：

```text
图像 + 点/框/已有掩膜等提示 -> 模型 -> 一个或多个候选目标掩膜 + 质量分数
```

它通常不会仅凭“这是 Synapse 图像”就自动、稳定地返回固定顺序的 8 个器官类别。你必须告诉它当前要分割哪个目标，或者提供一个能指向目标的提示。

### 2.3 SAM 类模型的三个主要部分

SAM 类模型通常由三部分组成：

1. **图像编码器（Image Encoder）**
   - 把输入图像编码成高维视觉特征。
   - 计算量和显存占用通常较大。
   - 常见实现使用 ViT 类骨干。

2. **提示编码器（Prompt Encoder）**
   - 把点、框、已有掩膜等提示编码成提示特征。
   - 点提示可以表示“目标前景点”或“目标背景点”。
   - 框提示可以表示目标的大致空间范围。

3. **掩膜解码器（Mask Decoder）**
   - 融合图像特征和提示特征。
   - 输出提示所指目标的候选掩膜及相应的质量分数。

不同 SAM-Med2D 发行版本的骨干、权重文件名、预处理函数和 API 可能不同。真正安装时应以所下载版本的 README、配置文件和推理脚本为准，不能只凭模型名称猜参数。

## 三、SAM-Med2D 与 MedSAM、原始 SAM 的关系

这三个名字容易混淆：

| 名称 | 主要定位 | 提示式分割 | 与当前 EMCAD 的关系 |
|---|---|---:|---|
| SAM | 面向通用图像的 Segment Anything 基础模型 | 是 | 原始医学 CT 上通常存在域差异 |
| MedSAM | 医学领域适配的 SAM 系列模型，常见工作流强调框提示 | 是 | 可作为医学提示分割工具，但不是固定 9 类头 |
| SAM-Med2D | 面向医学二维图像的 SAM 适配/训练方案 | 通常支持点、框、掩膜等提示，具体以版本为准 | 可作为标注器、教师模型或后处理模型 |

因此，不能因为它们都包含 `SAM`，就认为它们的 checkpoint 可以互换；也不能认为它们会自动输出相同的类别编号。

## 四、当前 EMCAD 项目的真实接口

### 4.1 EMCAD 的输入和输出

当前 `lib/networks.py` 中的 `EMCADNet` 定义在约第 49 行，构造函数在约第 63 行。它接收形如 `[B, C, H, W]` 的二维图像，当前 Synapse 输入通常是单通道 CT：

```text
[B, 1, H, W]
```

模型内部先用 1x1 卷积把单通道适配到 3 通道，然后经过 PVTv2 或 ResNet 编码器和 EMCAD 解码器。

当前四个分割头在 `lib/networks.py` 约第 211、213、215、217 行：

```python
self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)
```

Synapse 的 `num_classes=9` 意味着每个输出头有 9 个类别通道，而不是 9 个前景器官。模型返回四个多尺度 logits，测试时主要使用最后一个高分辨率输出：

```text
P = model(input)
outputs = P[-1]
prediction = softmax(outputs, dim=1).argmax(dim=1)
```

### 4.2 当前 Synapse 标签的读取方式

`utils/dataset_synapse.py` 的 `Synapse_dataset` 约在第 104 行：

- `split="train"` 时读取 `train_npz/*.npz`；
- 其他分支（包括 `test_vol`）读取 `test_vol_h5/*.npy.h5`；
- 返回的 `label` 是整数标签图。

当前文件中曾经存在的 14 类到 9 类重映射代码，在约第 152--165 行已经全部被注释。也就是说，当前数据加载器本身不会主动执行这些重映射。

`utils/preprocess_synapse_data.py` 约第 81--83 行只对图像和标签做相同的轴转置，约第 99--101 行把标签写入 HDF5，约第 118 行把切片标签写入 NPZ。这个脚本本身没有执行器官名称交换，也没有把 `liver` 和 `pancreas` 按名称重新排序。

因此，SAM-Med2D 接入时必须遵守当前数据文件里已经存在的整数标签约定，不能只根据可视化图例或器官名称重新猜编号。

### 4.3 当前测试指标是按整数标签计算的

`test_synapse.py` 约第 153 行把 `classes` 传给 `test_single_volume()`；约第 172 行按 `i=1..8` 输出类别名称。

真正的指标计算在 `utils/utils.py` 的 `test_single_volume()` 中完成，逻辑相当于：

```python
for i in range(1, classes):
    metric_list.append(
        calculate_metric_percase(prediction == i, label == i)
    )
```

这意味着：

- `prediction == i` 和 `label == i` 决定指标比较的是哪个整数标签；
- `class_names` 主要负责日志名称和可视化语义；
- SAM-Med2D 输出的掩膜必须最终转换成与 EMCAD 相同的整数标签图，才能进入现有指标函数。

## 五、为什么不能直接把 SAM-Med2D 替换 EMCADNet

直接替换至少会遇到以下接口差异：

| 项目 | 当前 EMCAD | SAM-Med2D | 直接替换问题 |
|---|---|---|---|
| 输入 | 单通道 CT 张量 `[B,1,H,W]` | 通常是经过其专用预处理的二维图像 | 通道、归一化、尺寸和色彩处理不同 |
| 条件 | 不需要提示，图像直接决定类别 | 需要点、框或掩膜等提示 | EMCAD 当前没有 prompt 输入 |
| 输出 | 9 类 dense logits | 提示目标的二值/候选掩膜和分数 | 没有天然的 9 类通道对应关系 |
| 训练标签 | 一个整数图 `0..8` | 常见是提示目标的二值掩膜 | 需要逐器官拆分和融合 |
| 训练损失 | Cross Entropy + Dice | 由 SAM-Med2D 实现和微调脚本决定 | 不能直接沿用现有训练循环 |
| 三维处理 | 当前按切片推理后重建病例 | 通常是二维推理 | 需要处理切片连续性和体数据重建 |
| checkpoint | EMCAD 的 state_dict | SAM-Med2D 的独立 state_dict | 参数名和张量形状不兼容 |

尤其要注意：**输出通道数相同不代表类别语义相同。** 即使某个 SAM-Med2D 包装器返回 8 个掩膜，也必须查清每个掩膜是由什么提示得到的，不能把它们直接当作 EMCAD 的标签 1--8。

## 六、推荐路线一：用 SAM-Med2D 辅助标注或生成伪标签

这是当前项目最容易控制、最不容易破坏基线的路线。

### 6.1 工作流

```text
Synapse 原始/现有 CT 切片
        |
        v
转换为 SAM-Med2D 所需的二维输入
        |
        v
给每个器官提供点/框/人工提示
        |
        v
SAM-Med2D 输出器官二值掩膜
        |
        v
人工检查、去除错误区域、处理重叠
        |
        v
融合为当前项目的整数标签图 0..8
        |
        v
保存为 NPZ/H5，继续使用 EMCAD 原训练代码
```

### 6.2 对 Synapse 八个器官如何融合

当前项目需要的目标表示是一个单张整数标签图：

```text
0 background
1 spleen
2 right kidney
3 left kidney
4 gallbladder
5 当前数据约定中的第5类
6 当前数据约定中的第6类
7 stomach
8 aorta
```

对每个器官得到二值掩膜后，按固定 `class_id` 写回：

```python
label[mask_for_class_1] = 1
label[mask_for_class_2] = 2
...
label[mask_for_class_8] = 8
```

其中第 5、6 类的名称必须先与当前训练/测试数据的真实标签约定核对。不能因为 SAM-Med2D 的提示顺序是“先肝脏后胰腺”，就自动把它写成标签 5 或 6。

如果多个器官掩膜发生重叠，必须定义优先级或使用更可靠的提示重新推理。否则融合后的整数标签会出现覆盖，导致训练标签语义不稳定。

### 6.3 图像输入适配

当前 EMCAD 读取的是已经做过 CT 窗宽裁剪和 `[0,1]` 归一化的单通道图像。SAM-Med2D 版本可能要求三通道图像和自己的归一化方式。常见适配方法有：

```python
# 简单复制：仅作为接口适配示例
image_3ch = np.repeat(image_2d[..., None], 3, axis=-1)
```

也可以把三个不同 CT 窗口分别放进三个通道，但这属于新的实验设计，必须记录窗口范围并做消融，不能把它和简单复制通道的结果混为一谈。

输入适配还必须保持：

- 图像和标签同一切片、同一方向；
- resize 使用与模型要求一致的方式；
- 图像可用连续插值，标签只能用最近邻插值；
- 保存回 NPZ/H5 前确认标签仍是整数，不含插值产生的小数类别。

### 6.4 这条路线的优点和风险

优点：

- EMCAD 的模型结构和 CE+Dice 训练循环基本不变；
- 可以先在少量切片上验证标签质量；
- 适合把 SAM-Med2D 当作人工标注工具或半自动标注工具。

风险：

- 伪标签错误会被 EMCAD 学习；
- 用真实标签框生成的掩膜只能作为标注辅助或上界实验，不能冒充完全自动推理；
- 伪标签生成必须只使用训练数据，不能把测试集真实标签用于提示或修正。

## 七、推荐路线二：EMCAD 粗分割 + SAM-Med2D 细化

这是一种推理级联：EMCAD 负责固定 8 类语义识别，SAM-Med2D 负责根据 EMCAD 给出的空间提示修边界。

### 7.1 典型流程

```text
CT 切片
  |
  v
EMCADNet
  |
  v
得到 0..8 的粗标签图
  |
  +--> 对 class_id=1..8 分别提取粗掩膜
          |
          +--> 计算 bounding box 或选择前景点
          |
          +--> 调用 SAM-Med2D
          |
          +--> 得到该器官细化掩膜
  |
  v
按 class_id 融合八个细化掩膜
  |
  v
堆叠切片，计算 Synapse 病例级 Dice/HD95/Jaccard/ASD
```

### 7.2 需要修改哪些地方

建议新增一个独立适配模块，例如：

```text
sam_med2d_refiner.py
```

它只负责：

1. 加载 SAM-Med2D 和其 checkpoint；
2. 把 CT 切片转换为 SAM-Med2D 输入；
3. 接收 EMCAD 的某个器官粗掩膜；
4. 生成框/点提示；
5. 返回经过阈值处理的二值掩膜。

然后在 `test_synapse.py` 或单独的测试脚本中，将 EMCAD 原来的 `prediction` 送入细化器，再把细化后的标签图交给现有指标函数。

不建议把 SAM-Med2D 的实现代码直接塞进 `utils/utils.py` 的 `test_single_volume()`。该函数现在同时承担逐切片推理、指标计算和结果保存，直接塞入大型模型会让基线和级联实验难以区分，也会显著增加显存和调试复杂度。

### 7.3 空掩膜和错误框必须处理

EMCAD 可能对某个器官没有预测结果。如果粗掩膜为空，不能直接计算空框并调用 SAM-Med2D。应明确一种策略：

- 返回空掩膜；或
- 使用检测器/人工提示重新提供框；或
- 跳过该器官并记录日志。

此外，EMCAD 的粗框可能包含相邻器官。SAM-Med2D 能否纠正取决于提示质量，不能假设级联一定提高 Dice。

### 7.4 如何保证实验公平

至少应区分三种实验：

1. **EMCAD baseline**：只用 EMCAD 输出。
2. **SAM-Med2D oracle prompt**：使用真实标签框或真实点，仅作为上界/可行性验证。
3. **EMCAD-generated prompt**：使用 EMCAD 粗掩膜生成提示，这是实际自动系统。

第 2 种不能直接与普通自动分割结果并列宣称性能提升，因为它使用了测试样本的真实空间信息。

## 八、推荐路线三：SAM-Med2D 教师 + EMCAD 学生蒸馏

蒸馏路线的目标不是让 SAM-Med2D 替代 EMCAD，而是让 EMCAD 学习教师模型提供的软掩膜或边界信息。

可以把训练损失设计为：

```text
L_total = L_CE+Dice(EMCAD, ground_truth)
          + lambda * L_distill(EMCAD, SAM-Med2D)
```

但这条路线需要解决：

- SAM-Med2D 对每个器官的提示如何获得；
- 教师二值掩膜如何转换为 9 类学生标签空间；
- 教师错误如何过滤；
- 用 logits、概率图还是二值掩膜蒸馏；
- 教师推理的计算成本和缓存格式；
- 验证集、测试集是否完全禁止使用真实标签提示。

建议在标注辅助路线和级联路线跑通后再做蒸馏，否则很难判断指标变化来自模型能力、伪标签质量还是提示策略。

## 九、不建议的第一步：直接把 SAM-Med2D 当 EMCAD 编码器

理论上可以重写 `EMCADNet`：

```text
SAM-Med2D image encoder
        |
        v
特征维度/分辨率适配层
        |
        v
EMCAD decoder 或新的语义分割头
        |
        v
9 类输出
```

但这不是简单替换：

- SAM-Med2D 的图像编码器输出尺度和通道数未必等于 EMCAD 的四级 skip 特征；
- SAM 的提示编码器和掩膜解码器可能仍然需要保留，不能只拿图像特征就假设行为不变；
- 当前 `out_head1..4` 的权重无法直接加载到新的特征接口；
- 单通道 CT 到 SAM 输入的适配会影响预训练特征；
- 显存、训练速度和 batch size 会发生明显变化；
- 这已经是一个新模型，不应再称为“原始 EMCAD 只接入一个模块”。

如果研究目标是研究新的 backbone，应单独建立新实验目录、重新训练，并把它与 EMCAD baseline 分开记录。

## 十、当前项目推荐的落地顺序

### 第 1 步：冻结 EMCAD baseline

先保存当前实验的：

- 训练配置；
- 数据目录和列表文件；
- `num_classes=9`；
- checkpoint；
- 每类 Dice、HD95、Jaccard、ASD；
- 当前整数标签映射。

不要在 baseline 尚未固定时同时改标签、模型和测试脚本。

### 第 2 步：单独安装 SAM-Med2D

当前 `requirements.txt` 固定了较老版本的 `transformers==4.21.3` 和 `huggingface-hub==0.11.0`，而外部 SAM-Med2D 项目可能要求不同版本。建议使用独立虚拟环境或至少单独记录依赖版本，避免破坏现有 EMCAD 环境。

安装后先只做一个二维切片的 smoke test：

```text
读取一张 CT 切片
-> 完成 SAM-Med2D 预处理
-> 提供一个人工框或点
-> 输出一张掩膜
-> 检查掩膜尺寸、取值范围和方向
```

### 第 3 步：验证标签转换

用一个训练病例做器官级转换，逐项检查：

- SAM 输出是否为二值掩膜；
- 保存后的标签是否只含 `0..8`；
- 第 5、6 类是否按照当前数据契约写入；
- 标签和 CT 是否仍逐像素对齐；
- 多个器官重叠时采用什么规则。

### 第 4 步：先做标注辅助实验

先不要接入测试主流程。用少量训练切片生成伪标签或修正标签，训练一个新的 EMCAD 实验，并与 baseline 比较。这样可以把“数据标签变化”和“模型后处理变化”分开。

### 第 5 步：再做自动级联实验

只有当 EMCAD 粗分割到 SAM-Med2D 的 prompt 转换稳定后，才在测试脚本中启用级联。输出目录、日志和 checkpoint 要与 baseline 分开，日志中记录：

```text
EMCAD checkpoint
SAM-Med2D checkpoint
提示类型：box / positive point / negative point / mask
提示来源：人工 / ground truth / EMCAD prediction
SAM 阈值和候选掩膜选择规则
```

## 十一、接入前必须通过的检查清单

### 数据与标签

- [ ] 训练和测试的标签都使用同一套整数编号。
- [ ] 标签值没有因 resize 出现小数或非法类别。
- [ ] 2D 切片方向与原始 H5/NPZ 一致。
- [ ] 级联后的标签仍是单张 `0..8` 整数图。
- [ ] 第 5、6 类名称不靠猜测，已由数据来源或固定映射证明。

### 模型与接口

- [ ] SAM-Med2D checkpoint 与其代码版本匹配。
- [ ] CT 单通道到 SAM 输入的通道和归一化规则已记录。
- [ ] 空 prompt、空粗掩膜、越界框都有处理。
- [ ] SAM 输出尺寸能恢复到原始切片尺寸。
- [ ] EMCAD checkpoint 没有被 SAM 权重覆盖。

### 实验公平性

- [ ] baseline、oracle prompt、自动 prompt 分开报告。
- [ ] 测试集真实标签没有用于生成 prompt 或修正预测。
- [ ] 使用相同的病例划分、图像预处理和指标实现。
- [ ] 除整体平均指标外，报告每个器官指标。
- [ ] 记录显存、推理时间和失败病例，不只记录 Dice。

## 十二、最终判断

对当前 EMCAD 项目，SAM-Med2D 最合适的定位不是“把 EMCADNet 换掉”，而是以下三种之一：

1. **标注工具**：帮助得到更好的 Synapse 2D 器官掩膜，然后训练 EMCAD；
2. **后处理器**：细化 EMCAD 的粗分割边界；
3. **教师模型**：通过伪标签或蒸馏向 EMCAD 传递额外信息。

如果目标是尽快验证是否有价值，推荐实际执行顺序为：

```text
EMCAD baseline
    -> SAM-Med2D 单切片 smoke test
    -> 少量训练切片标注辅助
    -> 重新训练 EMCAD
    -> EMCAD 粗掩膜生成 SAM prompt
    -> 自动级联测试
    -> 对比每类和整体指标
```

最重要的工程原则是：**SAM-Med2D 的“器官名称”不能直接替代 EMCAD 的“整数标签”；必须先建立从 SAM 输出掩膜到当前 `0..8` 标签图的显式映射，再进入训练或测试。**

## 十三、本文使用的当前仓库证据

- `lib/networks.py:49,63,211-217,227,248-276`：EMCADNet 的构造、四个输出头和多尺度输出。
- `utils/dataset_synapse.py:104-165`：NPZ/H5 标签读取，以及当前被注释的历史重映射代码。
- `utils/preprocess_synapse_data.py:81-83,99-101,118`：预处理中的轴转置和标签保存，没有按器官名称交换标签。
- `trainer.py:121-167,203,255-257`：训练数据读取、9 类输出、CE+Dice 训练。
- `test_synapse.py:118-130,136,153,172,321-338`：类别名称、测试数据读取、模型构造和 checkpoint 加载。
- `utils/utils.py` 的 `test_single_volume()`：按 `prediction == i` 和 `label == i` 对每个整数类别计算指标。

当前 checkout 中未发现 SAM-Med2D 源码或 checkpoint，因此本文没有声称已经完成实际运行集成；真实接入时还应以所下载 SAM-Med2D 版本的官方代码和权重说明为准。
