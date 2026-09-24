# EMCAD 预训练权重升级与可作为创新的方案

## 结论先说

当前项目默认使用：

```text
PVTv2-B2 编码器结构 + ImageNet 预训练参数
```

仓库中实际存在的文件是：

```text
pretrained_pth/pvt/pvt_v2_b2.pth
```

它只初始化 EMCAD 的编码器。解码器、四个分割输出头以及训练得到的完整模型参数，仍然是当前实验自己训练出来的。

“换一个更强的预训练文件”可能提高效果，但通常不能单独构成有说服力的论文创新。原因是：它更像是训练初始化或骨干网络对比，而不是 EMCAD 方法本身的新机制。

更值得作为创新的方向是：

> 在不改变 EMCAD 解码器的前提下，利用医学影像无标签数据对编码器进行领域自适应预训练，再迁移到 Synapse、ACDC、Polyp 或 BUSI 分割任务。

这比“把 ImageNet 权重换成另一个 ImageNet 权重”更有研究问题，也更容易解释为什么医学图像上有效。

---

## 一、当前仓库到底加载了什么

### 1. 训练入口

`train_synapse.py` 中默认配置包括：

```python
--encoder pvt_v2_b2
--pretrained_dir ./pretrained_pth/pvt/
--no_pretrain   # 只有加这个 flag 才关闭预训练
```

实际建模时：

```python
model = EMCADNet(
    num_classes=args.num_classes,
    encoder=args.encoder,
    pretrain=not args.no_pretrain,
    pretrained_dir=args.pretrained_dir
)
```

所以默认 `pretrain=True`。

### 2. 加载位置

`lib/networks.py` 的逻辑可以概括为：

```python
self.backbone = pvt_v2_b2()
path = pretrained_dir + '/pvt_v2_b2.pth'
save_model = torch.load(path)
# 过滤出当前 backbone 中存在的键
# 将匹配的参数写入 self.backbone
```

这意味着：

- `encoder='pvt_v2_b2'` 决定网络结构；
- `pretrain=True/False` 决定是否加载已有参数；
- `pretrained_dir` 决定去哪里找权重；
- 预训练文件不是 `best.pth`，也不是完整 EMCAD checkpoint；
- 预训练参数只进入 backbone，decoder 和 segmentation heads 仍然需要训练。

### 3. ImageNet 预训练的优点与限制

ImageNet 能让编码器先学到通用视觉模式，例如：

- 边缘和方向；
- 局部纹理；
- 形状轮廓；
- 多尺度视觉组合。

但医学图像有明显不同：

- CT、MRI、超声、内窥镜和自然图像的灰度/纹理分布不同；
- 医学目标可能很小、很细、对边界非常敏感；
- Synapse 的 CT 不是 RGB 自然照片；
- ACDC 是心脏 MRI；
- BUSI 是乳腺超声；
- 不同模态不能简单视为同一个数据分布。

因此，ImageNet 是合理基线，但不一定是医学分割最匹配的初始化。

---

## 二、有哪些可能更好的预训练来源

下面按“对当前 EMCAD 的可接入程度”和“论文价值”区分。

### 方案 A：更大的自然图像预训练

例子：

- ImageNet-22K 预训练；
- 更大规模监督视觉数据；
- Swin、ConvNeXt、DINOv2 等模型的通用视觉预训练。

优点：

- 特征通常比小规模 ImageNet-1K 更强；
- 有成熟公开权重；
- 可以做 encoder 对比实验。

问题：

- 许多权重对应的结构不是 PVTv2-B2；
- 换结构会同时改变参数量、速度、特征通道和归纳偏置；
- 如果只换权重但不保持结构，无法判断提升来自预训练还是 backbone；
- 需要改写 `lib/networks.py` 的四级特征接口。

论文定位：

- 可以作为“骨干对比”或“初始化敏感性实验”；
- 单独作为创新通常偏弱。

### 方案 B：医学影像领域预训练

典型思路包括：

- RadImageNet 等医学图像分类预训练；
- 在大量 CT/MRI/超声无标签图像上做自监督预训练；
- 在与目标任务相近的医学数据上进行迁移学习；
- 在目标数据的训练部分做领域自适应预训练。

优点：

- 特征分布更接近医学影像；
- 研究假设清晰：自然图像表征与医学表征存在领域差异；
- 更容易解释小目标、边界、纹理和器官形态的改善。

问题：

- 公开权重不一定是 PVTv2-B2；
- 可能是 2D、3D 或不同模态；
- 通道数、输入尺寸和 stem 结构可能不兼容；
- 3D MedicalNet 权重不能直接塞进当前 2D PVTv2；
- 需要检查许可证和训练数据来源。

论文定位：

- 如果只是下载别人训练好的医学分类权重，创新仍然有限；
- 如果自己设计“医学无标签自监督预训练 + EMCAD 迁移”，研究价值明显更高。

### 方案 C：在目标医学数据上做自监督预训练

这是最推荐的方向。

在分割标签之外，收集或使用可合法使用的无标签医学图像，先训练编码器完成预训练任务，再用标注分割数据训练 EMCAD。

可选预训练任务：

1. **Masked image modeling（掩码重建）**
   - 随机遮挡图像块；
   - 让编码器学习恢复被遮挡区域；
   - 对纹理、局部结构和空间上下文有帮助。

2. **对比学习**
   - 对同一图像做不同增强；
   - 让同一图像的两个视图特征接近；
   - 让不同图像特征区分开。

3. **旋转/几何变换预测**
   - 预测图像被旋转或变换的方式；
   - 实现简单，但医学分割论文中的新颖性较弱。

4. **跨切片一致性**
   - 对同一个 CT/MRI 病例相邻切片提取特征；
   - 约束相邻切片的表示具有连续性；
   - 更贴近三维医学结构，是很有潜力的 EMCAD 适配方向。

5. **边界/器官结构预训练**
   - 从无标签图像构造梯度、边缘或局部结构目标；
   - 让编码器关注器官边界；
   - 需要严谨的消融实验证明不是简单增强带来的效果。

论文定位：

- 这是“训练策略/领域预训练创新”；
- 只要预训练目标、数据隔离、迁移流程和消融设计清楚，能够形成完整方法故事。

---

## 三、最推荐的创新题目方向

可以先把研究问题定义为：

> ImageNet 预训练主要学习自然图像统计规律，是否可以通过医学影像无标签预训练，使 EMCAD 的编码器更好地适应医学结构和边界，从而提升多器官分割性能？

一个可操作的名字可以暂定为：

```text
Medical Domain-Adaptive Pretraining for EMCAD
医学领域自适应预训练 EMCAD
```

不要一开始就声称“提出了全新网络”。先把它定义成一个可验证的训练框架。

### 推荐的最小方法

```text
ImageNet PVTv2-B2
        ↓
医学无标签图像自监督预训练
        ↓
加载 encoder 权重
        ↓
接入原始 EMCAD decoder + segmentation heads
        ↓
在固定分割训练集上微调
```

解码器可以保持不变，这样实验能回答：

```text
提升到底来自医学预训练，而不是 decoder 改动？
```

---

## 四、必须先区分的三个变量

### 变量 1：骨干结构

例如：

```text
PVTv2-B2、ResNet50、Swin-T、ConvNeXt-T
```

改变它会改变网络容量和特征接口。

### 变量 2：预训练来源

例如：

```text
随机初始化、ImageNet、RadImageNet、自监督医学预训练
```

在结构相同的情况下改变它，才是在比较初始化/预训练来源。

### 变量 3：解码器结构

例如：

```text
原始 EMCAD decoder、加入边界模块、加入跨尺度融合
```

这是网络结构创新。

如果同时更换骨干、预训练来源和 decoder，就无法知道提升来自哪里。第一轮实验必须固定其中两个，只改变一个。

---

## 五、建议的实验矩阵

### 第一阶段：确认基线

固定：

- PVTv2-B2；
- 原始 EMCAD decoder；
- 相同输入尺寸；
- 相同数据划分；
- 相同随机种子；
- 相同 `supervision`；
- 相同 batch size、学习率和 epoch。

运行：

```text
B0: PVTv2-B2 + ImageNet 预训练
B1: PVTv2-B2 + 随机初始化
```

目的：确认 ImageNet 预训练在当前仓库确实带来收益。

### 第二阶段：比较外部医学权重

```text
B0: ImageNet PVTv2-B2
B2: 医学领域公开权重 + 兼容 backbone
```

只有在结构、输入维度和预处理能够公平对齐时才比较。若医学权重来自不同 backbone，应明确写成“backbone + pretraining 联合对比”，不能声称只验证了预训练效果。

### 第三阶段：验证自监督医学预训练

```text
B0: ImageNet PVTv2-B2
P1: ImageNet 初始化后，在医学无标签图像上自监督预训练
P2: P1 的 encoder 权重 + 原始 EMCAD decoder 微调
```

核心结果是：

```text
P2 是否比 B0 在 Dice、HD95、边界质量和收敛速度上稳定改善？
```

### 第四阶段：消融预训练策略

至少做：

```text
A0: 不做医学预训练
A1: 医学预训练，但不使用相邻切片一致性
A2: 医学预训练 + 相邻切片一致性
A3: 医学预训练 + 边界辅助目标
```

每次只增加一个因素。

---

## 六、数据泄漏是这个方向最大的风险

如果用目标测试集或验证集图像进行预训练，哪怕没有使用它们的分割标签，也可能造成分布泄漏，论文审稿时会被质疑。

推荐规则：

- 预训练数据只来自训练患者，或来自完全独立的外部无标签数据；
- 不能把验证/测试患者的图像混入预训练池；
- 必须按患者划分，而不是随机按切片划分；
- 预训练和微调的病例清单要单独保存；
- 记录图像来源、患者 ID、模态、预处理和排除规则。

最低记录文件：

```text
pretrain_images.txt
finetune_train.txt
validation.txt
test.txt
data_manifest.csv
```

其中 `data_manifest.csv` 至少记录：

```text
case_id, split, modality, source, used_for_pretrain, used_for_segmentation
```

---

## 七、如何接入当前 EMCAD 代码

### 路线 1：保持 PVTv2-B2 结构，替换其参数

这是最稳妥的路线。

要求自监督预训练最终保存的 state dict 与：

```text
lib/pvtv2.py 中 pvt_v2_b2 的参数名和形状
```

兼容。

然后可以：

1. 建立新的预训练文件目录，例如 `pretrained_pth/medical_pvt/`；
2. 保存 `pvt_v2_b2_medical_ssl.pth`；
3. 在 `lib/networks.py` 增加一个明确的权重路径参数；
4. 只加载 backbone 参数；
5. decoder 和输出头保持随机初始化；
6. 使用同一分割训练入口微调。

不要直接覆盖原始 `pvt_v2_b2.pth`，否则无法复现 ImageNet 基线。

### 路线 2：更换为其他 backbone

如果使用 Swin、ConvNeXt 或其他结构，需要保证它返回 EMCAD decoder 需要的四级特征：

```text
x1, x2, x3, x4
```

并同步修改：

- `lib/networks.py` 的 encoder 构造；
- 四级通道列表；
- 输入 stem；
- decoder 的 channels 参数；
- 可能的上采样比例；
- checkpoint 加载逻辑；
- 参数量和 FLOPs 统计。

这已经不再是“只换预训练文件”，而是 backbone 迁移工程。

---

## 八、效果应该看什么，不要只看 Dice

建议同时记录：

- mean Dice；
- 每个器官 Dice；
- HD95；
- ASD/ASSD；
- Jaccard；
- 最佳 epoch；
- 达到固定 Dice 所需 epoch；
- 单 epoch 时间；
- 显存占用；
- 参数量和 FLOPs；
- 不同随机种子的均值和标准差。

预训练可能不只提升最终 Dice，还可能：

- 更快收敛；
- 减少训练不稳定；
- 改善小器官；
- 改善边界指标；
- 降低不同随机种子之间的波动。

如果只跑一次、只报告一个 Dice，很难证明方法可靠。

---

## 九、这个创新是否足够强

### 仅替换为更大的 ImageNet 权重

评价：

```text
工程价值：中等
效果可能：有可能
论文创新性：偏弱
```

适合写成：

- 预训练敏感性分析；
- backbone 对比；
- 实验设置改进。

不建议单独作为论文主创新。

### 换成公开医学分类权重

评价：

```text
工程价值：中等
医学相关性：较强
论文创新性：取决于权重来源和方法设计
```

如果只是下载并替换，创新仍然有限；如果研究模态匹配、迁移层冻结策略和领域差异，论文价值会增加。

### 自己进行医学无标签自监督预训练

评价：

```text
研究问题：清晰
与医学分割相关性：强
实验工作量：较大
可形成方法故事：较好
```

这是更推荐的主线，但必须处理数据泄漏、预训练目标、对照实验和多种子验证。

---

## 十、给你的实际推荐顺序

### 第一步：先做两个基线

```text
1. PVTv2-B2 + ImageNet
2. PVTv2-B2 + random initialization
```

确认当前训练流程和指标可信。

### 第二步：先不要急着换 decoder

保持 EMCAD 其他代码不动，只研究 encoder 的初始化：

```text
3. PVTv2-B2 + 医学自监督预训练
```

这样创新变量干净。

### 第三步：优先尝试相邻切片一致性

Synapse 和 ACDC 本身具有病例内切片结构。可以让相邻切片的 encoder 表示保持适度一致，同时保留局部变化。这比普通旋转预测更贴近医学影像。

但第一版不要直接改成复杂三维网络。当前 EMCAD 是二维切片训练，先做二维 encoder 的跨切片表示约束，工程风险较小。

### 第四步：做最小可发表证据包

至少包括：

```text
ImageNet baseline
random-init baseline
medical-pretrained model
无一致性损失消融
有一致性损失主模型
多个随机种子
每类指标
HD95/边界指标
收敛曲线
参数量和速度
```

---

## 十一、实验记录模板

每次实验单独建立目录：

```text
model_pth/pretrain_ablation/<experiment_id>/
```

保存：

```text
config.json
command.txt
pretrain_manifest.csv
finetune_manifest.csv
environment.txt
train.log
best.pth
last.pth
metrics.csv
per_case_metrics.csv
notes.md
```

`notes.md` 写清楚：

```text
假设：
唯一改变的变量：
预训练数据来源：
是否包含验证/测试患者：
预训练任务：
预训练 epoch：
微调 epoch：
最佳验证 Dice：
HD95：
失败运行：
下一步：
```

---

## 十二、最终判断

可以研究“比 ImageNet 更适合 EMCAD 的预训练”，但建议把题目从：

```text
寻找一个更好的预训练文件
```

提升为：

```text
面向医学影像结构和跨切片关系的 EMCAD 领域自适应预训练方法
```

前者是模型文件替换，后者才是可验证的研究方法。

当前最稳妥的主线是：

```text
固定 PVTv2-B2 和 EMCAD decoder
→ 保留 ImageNet 作为初始化
→ 使用无标签医学图像进行自监督预训练
→ 只把预训练后的 encoder 迁移给 EMCAD
→ 在患者级固定划分上微调
→ 用 Dice + HD95 + 收敛速度 + 多种子证明收益
```


