# MSCB 扩张倍数与测试配置一致性说明

## 1. 先说结论

代码中的：

```python
parser.add_argument('--expansion_factor', type=int,
                    default=2, help='expansion factor in MSCB block')
```

表示：**MSCB（Multi-Scale Convolution Block，多尺度卷积块）内部先把输入通道数扩大到原来的 `expansion_factor` 倍，再进行多尺度深度卷积，最后投影回输出通道数。**

在本项目中，默认值是 `2`。例如某个 MSCB 输入为 64 个通道时：

```text
输入通道 C_in = 64
expansion_factor = 2
MSCB 内部中间通道 C_ex = int(64 × 2) = 128
最后输出通道 C_out = 64
```

因此它不是学习率、batch size 这一类训练超参数，而是**决定网络层形状的结构参数**。训练时用 `2`，测试时也必须用 `2` 来重新构造模型。

## 2. 参数从命令行到 MSCB 的实际传递路径

以 Synapse 为例，代码路径是：

```text
train_synapse.py / test_synapse.py
    --expansion_factor 2
            ↓
args.expansion_factor
            ↓
EMCADNet(..., expansion_factor=args.expansion_factor, ...)
            ↓
EMCAD(..., expansion_factor=expansion_factor, ...)
            ↓
MSCBLayer(..., expansion_factor=expansion_factor, ...)
            ↓
MSCB(..., expansion_factor=expansion_factor, ...)
```

也就是说，命令行参数最终会影响解码器中四个尺度的 MSCB。当前 EMCAD 的 `mscb4、mscb3、mscb2、mscb1` 每一级通常创建一个 MSCB。

仓库中的核心实现位于 `lib/decoders.py`。MSCB 构造函数中有：

```python
self.ex_channels = int(self.in_channels * self.expansion_factor)
```

随后第一层 `1×1` 卷积把通道从 `in_channels` 映射到 `ex_channels`：

```python
nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False)
```

多尺度深度卷积在 `ex_channels` 个中间通道上工作，之后第二个 `1×1` 卷积把融合后的特征投影到 `out_channels`：

```python
nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False)
```

在 EMCAD 的正常配置中，通常 `in_channels == out_channels` 且 `stride=1`，所以 MSCB 输入和输出的外部形状不变，中间只是“变宽后再压回去”。

## 3. “扩张倍数”到底扩张了什么

这里的“扩张”不是把图片的高和宽放大，也不是扩大卷积核尺寸，而是扩大**特征通道数**。

设输入特征为：

```text
x 的形状 = (B, C_in, H, W)
```

其中：

- `B`：batch size；
- `C_in`：输入通道数；
- `H、W`：特征图空间尺寸。

MSCB 的简化流程是：

```text
(B, C_in, H, W)
        │
        │ 1×1 卷积：C_in → C_ex
        │ C_ex = int(C_in × expansion_factor)
        ▼
(B, C_ex, H, W)
        │
        │ 多尺度深度卷积，例如 1×1、3×3、5×5
        ▼
多尺度特征融合
        │
        │ channel shuffle + 1×1 卷积：C_ex → C_out
        ▼
(B, C_out, H, W)
        │
        │ stride=1 时与捷径分支相加
        ▼
MSCB 输出
```

因此，`expansion_factor` 主要控制 MSCB 内部的“工作宽度”或“隐藏通道宽度”。它类似于某些轻量网络中的 expansion ratio：先在较宽的中间空间里提取和融合特征，再压缩回目标通道数。

## 4. 用具体数字理解 `2、4、6`

假设某一级 MSCB 的输入输出通道都是 64，并且使用默认的 `kernel_sizes=[1, 3, 5]`：

| `expansion_factor` | 中间通道 `C_ex` | 直观含义 |
|---:|---:|---|
| 1 | 64 | 不扩宽，计算较省，表达空间较窄 |
| 2 | 128 | 本项目常用默认值，平衡表达能力与开销 |
| 4 | 256 | 中间特征更宽，通常参数量和计算量更大 |
| 6 | 384 | 更宽，但显存和计算开销明显增加 |

注意：参数量不会只简单增加 `2` 倍或 `4` 倍，因为 MSCB 还包含多尺度深度卷积、第二个 `1×1` 卷积、BatchNorm，以及不同阶段的输入通道数。不过总体趋势是：**扩张倍数越大，中间通道越多，模型通常越重。**

以 `C_in=C_out=64、add=True、kernel_sizes=[1,3,5]` 为例：

- `e=2` 时，`C_ex=128`；三个尺度的结果逐元素相加，融合后仍是 128 通道；第二个 `1×1` 卷积再从 128 投影回 64。
- `e=4` 时，`C_ex=256`；融合后是 256 通道；第二个 `1×1` 卷积从 256 投影回 64。

所以 `e` 增大后，MSCB 中间的特征表示空间变宽了，但模块外部的输入输出接口仍可以保持不变。

## 5. 为什么说它属于“网络结构”

`expansion_factor` 会直接决定卷积层的权重形状。

例如第一层卷积：

```text
Conv2d(C_in, int(C_in × e), 1)
```

若 `C_in=64`：

```text
e=2  → Conv2d(64, 128, 1)
e=4  → Conv2d(64, 256, 1)
```

这两层的权重张量尺寸不同。第二个投影卷积的输入通道也会随之改变。因此 `e=2` 和 `e=4` 不是“同一个网络只换了一个数”，而是两个不同的网络结构。

这与学习率的区别很重要：

- 学习率改变参数更新步长，但不改变卷积层的形状；
- `expansion_factor` 改变卷积层输入或输出通道数，直接改变 `state_dict` 中参数张量的尺寸。

## 6. 为什么测试时必须复用训练配置

测试时通常不是把 checkpoint 直接当成一个完全自描述的模型对象使用，而是先执行以下步骤：

```text
1. 根据命令行参数创建一个空的 EMCADNet
2. 根据 expansion_factor 等参数创建 MSCB
3. 读取 best.pth
4. 将 checkpoint 中的权重加载进刚刚创建的网络
5. 用测试图像前向推理
```

因此，测试脚本中的参数必须能够重建出与训练时相同的网络。

训练时如果是：

```bash
python train_synapse.py --expansion_factor 2
```

测试就应该是：

```bash
python test_synapse.py --expansion_factor 2
```

而不是：

```bash
python test_synapse.py --expansion_factor 4
```

后者会先构造一个 `e=4` 的 MSCB，再尝试加载为 `e=2` 训练得到的权重。常见结果包括：

1. `size mismatch`：权重张量尺寸不一致，程序直接报错；
2. 某些代码使用非严格加载时出现 missing keys 或 unexpected keys；
3. 即使人为绕过部分加载问题，剩余随机初始化层也会让测试结果失去意义；
4. 结构相同但配置语义不同的参数可能不立刻报错，却会导致结果不可与原实验公平比较。

所以“复用训练配置”不只是为了避免报错，也是为了保证测试指标确实对应那次训练得到的模型。

## 7. 不仅 `expansion_factor` 要一致

测试时应当整体复用训练时的结构配置，至少包括：

| 配置 | 影响 |
|---|---|
| `encoder` | 主干网络类型及各级通道数 |
| `expansion_factor` | MSCB 中间通道宽度 |
| `kernel_sizes` | MSDC 的多尺度卷积分支和感受野 |
| `lgag_ks` | LGAG 的空间卷积核大小 |
| `activation_mscb` | MSCB 内部激活函数，如 `relu6` 或 `relu` |
| `no_dw_parallel` | MSDC 分支是并行还是串行 |
| `concatenation` | 多尺度分支是 concat 还是逐元素相加 |
| `num_classes` | 输出头通道数，必须匹配任务标签 |

其中 `no_pretrain` 主要影响编码器初始化方式，不等同于 MSCB 结构；但测试脚本的模型构建和实验目录命名仍应与训练记录保持清楚一致。

## 8. 常见误解

### 误解一：默认值是 2，所以测试永远填 2 就行

不一定。`default=2` 只表示当命令行没有显式传值时使用 2。真正应当使用的是**训练该 checkpoint 时的实际值**。如果某次实验使用了：

```bash
--expansion_factor 4
```

那么测试也应使用 4，即使测试脚本默认值仍然写着 2。

### 误解二：扩张倍数只影响速度，不影响结果

错误。它改变了中间特征宽度、可学习参数数量、计算量和网络表达能力，因此可能影响收敛、显存占用、推理速度以及最终 Dice、HD95 等指标。

### 误解三：只要输入输出通道没变，内部扩张倍数就可以随便改

错误。虽然 MSCB 对外可以保持 `C_in → C_out` 不变，但内部卷积权重形状已经改变，checkpoint 仍然无法自然对应。

### 误解四：`expansion_factor` 是数据集参数

它不是 Synapse、ACDC 或 Polyp 数据集本身的属性，而是模型结构配置。不同数据集可以使用不同值，但实验记录必须明确写出实际值。

## 9. 如何核对某个 checkpoint 的训练配置

建议按以下优先级核对：

1. 查看启动训练的 `.sh` 文件或终端命令，确认是否显式传入 `--expansion_factor`。
2. 查看训练日志开头打印的参数字典或实验目录名，例如目录名中的 `ef2`。
3. 查看训练代码生成的实验配置文件（如果该实验保存了 `config.json` 或类似文件）。
4. 查看测试脚本实际构造模型时传入的参数，确认与训练记录逐项一致。
5. 最后再运行测试，不要只依赖测试脚本的默认值。

一个推荐的实验记录表如下：

```text
实验编号：
数据集：Synapse / ACDC / Polyp / 其他
训练脚本：
checkpoint：
encoder：
expansion_factor：
kernel_sizes：
lgag_ks：
activation_mscb：
dw_parallel：
MSDC 聚合方式：add / concat
num_classes：
测试脚本：
测试命令：
```

## 10. 一句话记忆

可以把 `expansion_factor` 记成：

> **MSCB 内部“先把通道变宽多少倍”的结构开关；它决定卷积层形状，所以训练和测试必须使用同一个值。**

在本项目的常见默认配置中，`expansion_factor=2` 的含义就是：每个 MSCB 先把输入通道扩成 2 倍，在更宽的中间特征空间里做多尺度卷积，再压回目标输出通道。

## 11. 本次仓库核对依据

- `lib/decoders.py`：MSCB 中计算 `self.ex_channels = int(self.in_channels * self.expansion_factor)`，并用两个 `1×1` 卷积完成扩张与投影。
- `networks.py` / `lib/networks.py`：`EMCADNet` 接收 `expansion_factor` 并传入 `EMCAD` 解码器。
- `train_synapse.py`：训练入口解析 `--expansion_factor`，默认值为 2，并传给 `EMCADNet`。
- `test_synapse.py`：测试入口解析同名参数，并使用它重建模型后加载 checkpoint。

