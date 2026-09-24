# Synapse 测试阶段三维体数据逐切片推理流程

## 一句话概括

当前 EMCAD 是二维网络，不直接把整个三维 CT 体送入网络。Synapse 测试时先读取一个完整病例 `[D,H,W]`，再沿第 0 维逐张取出 `[H,W]` 切片，逐张推理，最后把所有切片按原顺序写回，重新组成完整三维预测体。

```text
H5 文件
  image/label: [D,H,W]
       ↓
逐个 ind=0,1,...,D-1 取 image[ind,:,:]
       ↓
缩放到 img_size × img_size
       ↓
EMCAD 二维前向
       ↓
softmax + argmax 得到整数类别图
       ↓
最近邻缩回原始 H×W
       ↓
prediction[ind] = pred
       ↓
完整 [D,H,W] 预测体
```

## 1. 完整体数据从哪里读取

`test_synapse.py` 的 `inference()` 创建数据集：

```python
db_test = args.Dataset(
    base_dir=args.volume_path,
    split="test_vol",
    list_dir=args.list_dir,
    nclass=args.num_classes,
)
```

`utils/dataset_synapse.py` 对非 `train` 的分支读取：

```python
vol_name = self.sample_list[idx].strip('\n')
filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
data = h5py.File(filepath)
image, label = data['image'][:], data['label'][:]
```

当前数据约定是：

```text
image: [D,H,W]，归一化后的 CT 体
label: [D,H,W]，整数类别标签体
```

DataLoader 使用 `batch_size=1`，因此送到 `inference()` 的张量通常是：

```text
image: [1,D,H,W]
label: [1,D,H,W]
```

这里最前面的 `1` 是 batch 维，不是 CT 深度。

## 2. 为什么要先去掉 batch 维

`test_single_volume()` 开始执行：

```python
image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
```

处理结果是：

```text
[1,D,H,W] -> [D,H,W]
```

然后函数可以使用：

```python
for ind in range(image.shape[0]):
```

此时 `image.shape[0]` 就是切片数量 `D`。

注意：当前代码使用的是 `squeeze(0)`，它假定第 0 维确实是大小为 1 的 batch 维。不要直接把一个已经是 `[D,H,W]` 的数组再传入后期待它总是安全；如果 `D=1`，误用 squeeze 可能会把真正的深度维也删掉。调用方应保持 `[1,D,H,W]` 约定。

## 3. 具体如何切片

三维分支的核心代码是：

```python
prediction = np.zeros_like(label)

for ind in range(image.shape[0]):
    slice = image[ind, :, :]
    x, y = slice.shape[0], slice.shape[1]
```

含义：

- `prediction` 先创建一个与真实标签完全相同形状的空数组；
- 初始值通常是 0，也就是背景；
- `ind` 是第几张二维切片；
- `image[ind, :, :]` 取出第 `ind` 张 `[H,W]` 图像；
- `label[ind, :, :]` 是同一空间位置的真实标签切片；
- 循环结束后，`prediction[ind]` 与 `label[ind]` 一一对应。

所以当前实现是沿数组第 0 维切片。它是否对应医学意义上的轴向方向，取决于预处理时 NIfTI 数组轴的约定。当前 `preprocess_synapse_data.py` 曾执行：

```python
ct_array = np.transpose(ct_array, (2, 0, 1))
seg_array = np.transpose(seg_array, (2, 0, 1))
```

因此保存后的第 0 维来自原始数组的第 2 维。不能只看到变量名 `D` 就断言它一定是某个特定解剖方向；需要结合原始 NIfTI header、transpose 规则和可视化检查。

## 4. 切片为什么要缩放

模型训练通常使用固定尺寸，例如 `224×224` 或 `256×256`。原始测试切片的 `H,W` 可能不同，因此代码先检查：

```python
if x != patch_size[0] or y != patch_size[1]:
    slice = zoom(
        slice,
        (patch_size[0] / x, patch_size[1] / y),
        order=3,
    )
```

这里分别按高度和宽度计算缩放比例：

```text
原始 [x,y] -> 模型输入 [patch_size[0], patch_size[1]]
```

`order=3` 是三次插值，适合连续的 CT 灰度图。它不会把图像强行裁成正方形，而是分别缩放两个方向；如果原图是长方形，可能发生几何拉伸。训练阶段和测试阶段必须使用一致的尺寸处理规则，否则模型看到的空间形变分布不同。

## 5. 如何送进二维 EMCAD

当前切片 `[H,W]` 通过两次 `unsqueeze` 变成：

```python
input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
```

形状变化：

```text
[H,W]
  -> 第一次 unsqueeze: [1,H,W]       # batch 维
  -> 第二次 unsqueeze: [1,1,H,W]     # 单通道 CT 维
```

网络接收的是标准二维卷积输入 `[B,C,H,W]`：

```text
B=1：一次只推理一张切片
C=1：CT 灰度单通道
```

代码内部使用 `.cuda()`，因此当前测试实现要求 CUDA；模型和输入必须位于同一张可见 GPU 上。

## 6. 如何从网络输出得到类别图

EMCAD 返回多个输出，测试只取最后一个：

```python
P = net(input)
outputs = P[-1]
out = torch.argmax(
    torch.softmax(outputs, dim=1),
    dim=1,
).squeeze(0)
```

如果 `num_classes=9`，最后输出大致是：

```text
outputs: [1,9,H,W]
```

其中 9 个通道是模型的类别 logits。处理过程是：

1. `softmax(dim=1)`：在类别通道维度把 logits 转成概率；
2. `argmax(dim=1)`：每个像素选概率最大的类别编号；
3. `squeeze(0)`：去掉 batch 维；
4. 得到 `[H,W]` 的整数标签图，像素值应为 `0..8`。

测试阶段不是分别输出 8 张互相独立的二值图，而是输出一张互斥的多类整数标签图。

## 7. 为什么恢复尺寸时必须用最近邻

推理输出 `out` 是离散类别编号，代码使用：

```python
pred = zoom(
    out,
    (x / patch_size[0], y / patch_size[1]),
    order=0,
)
```

`order=0` 是最近邻插值，必须保留。不能对类别图使用三次插值或线性插值，因为可能产生：

```text
1.2、3.5、6.7
```

这些不是合法类别。后续代码通过：

```python
prediction == i
label == i
```

构造二值掩膜计算指标；如果类别图出现小数，像素会匹配不到任何整数类别，Dice 等指标就会失真。

## 8. 如何拼回完整三维体

恢复到原始切片大小后，代码执行：

```python
prediction[ind] = pred
```

这一步把当前二维预测放回原来的深度索引。所有 `ind` 循环结束后：

```text
prediction.shape == label.shape == [D,H,W]
```

然后对每个前景类别计算：

```python
for i in range(1, classes):
    calculate_metric_percase(prediction == i, label == i)
```

背景 `0` 不参与器官平均指标；前景标签 `1..8` 分别计算 Dice、HD95、Jaccard、ASD。

## 9. 测试阶段最需要注意的事项

### 9.1 切片顺序和轴方向

如果 image 做了 transpose，而 label 没有做完全相同的 transpose，图像与标签会错位。当前预处理脚本对二者都使用 `(2,0,1)`，这是必要条件。

不要只检查数组 shape；错误轴置换可能仍然产生相同的 shape。应随机抽取病例，将 CT、label 叠加显示，确认器官在空间上重合。

### 9.2 resize 必须与训练一致

训练用 `224×224`，测试就不能随意改成 `256×256`；如果改动，必须使用匹配的训练配置和 checkpoint。输入尺寸、插值方式、归一化范围都属于模型输入契约。

### 9.3 图像和标签使用不同插值

```text
CT 图像：order=3，连续灰度
预测标签：order=0，离散类别
```

标签绝不能使用高阶插值。

### 9.4 z 方向 spacing 不等于切片索引

`ind` 只是数组索引；`z_spacing` 是保存 NIfTI 时写入的物理间距。当前测试调用传入 `z_spacing=1`，这会影响保存文件的物理坐标和 HD95/ASD 的物理解释，但不会改变 `prediction[ind]` 的数组内容。

当前 `test_single_volume()` 还把面内 spacing 固定为 `(1,1,z_spacing)`。如果原始 Synapse NIfTI 的真实 spacing 没有被保留，保存后的几何距离指标可能不是严格的毫米单位。

### 9.5 测试集每个病例的指标不是切片平均

代码先把所有切片拼成完整三维 `prediction`，再计算每个器官的三维指标。因此它不是：

```text
逐切片 Dice -> 再平均
```

而是：

```text
所有切片组成三维预测体 -> 与三维真实体比较
```

这对 HD95、ASD 尤其重要，因为表面距离需要完整三维结构。

### 9.6 `test_save_path` 要正确设置

三维分支中当前代码会保存每张切片的 PNG 叠加图，并且后面还会保存预测、图像、标签 NIfTI。输出目录必须提前存在，`case` 也不能为 `None`。

如果只想计算指标，应检查当前版本中 PNG 保存语句是否仍然无条件执行；历史代码在 `test_save_path=None` 时可能在保存 PNG 处报错。不要把“关闭 NIfTI 保存”误认为“所有可视化写入都关闭”。

### 9.7 GPU 显存和运行时间

当前一次只处理一张切片，但每个病例要循环 `D` 次。12 个测试病例逐切片推理会比较慢。不能为了提速直接把 `[D,H,W]` 当成 `[B,C,H,W]`，除非明确改写批量推理代码并验证内存与结果一致性。

### 9.8 标签编号必须一致

模型输出的通道编号、H5 中真实标签编号、`classes` 名称列表必须使用同一套映射。修改名称列表不会改变预测整数，也不会改变数值指标，只会改变日志和可视化解释。

## 10. 建议的人工核查方法

随机选择一个病例和几个切片，检查以下内容：

1. `image[ind]` 的形状是否为 `[H,W]`；
2. `label[ind]` 是否与图像空间对齐；
3. `np.unique(label[ind])` 是否只包含合法整数标签；
4. `pred` 缩回后形状是否恢复为原始 `[H,W]`；
5. `np.unique(pred)` 是否只包含 `0..8`；
6. `prediction.shape` 是否等于 `label.shape`；
7. 保存的 NIfTI 在 ITK-SNAP 中叠加后，器官位置是否正确；
8. 预测和真值是否使用相同的 spacing、方向和原点元数据。

## 最终理解

当前 Synapse 测试不是三维卷积推理，而是“二维网络 + 三维病例重组”：

```text
完整 H5 病例
  -> 去 batch
  -> 沿第 0 维切片
  -> 每张切片缩放到训练尺寸
  -> EMCAD 二维推理
  -> softmax/argmax 得离散标签
  -> 最近邻恢复原尺寸
  -> 按 ind 写回完整体
  -> 三维指标计算和结果保存
```

最容易导致结果错误的地方是：轴顺序不一致、图像/标签 resize 插值错误、训练测试输入尺寸不一致、spacing 丢失、标签编号映射不一致，以及把最后一轮或错误名称列表当成了真实指标含义。
