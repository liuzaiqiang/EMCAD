# Synapse `predictions` 文件夹输出文件说明

## 先看文件名规律

截图中的文件名例如：

```text
case0001_140_gt.png
case0001_140_pred.png
case0001_gt.nii.gz
case0001_img.nii.gz
case0001_pred.nii.gz
```

文件名由三部分组成：

```text
病例名 + 切片编号 + 文件类型
```

例如：

```text
case0001_140_pred.png
```

表示：

```text
病例：case0001
切片：第 140 张二维切片
内容：模型预测结果叠加图
```

`case0001_0_*`、`case0001_1_*`、`case0001_2_*` 会依次对应同一个三维病例的第 0、1、2 张切片。截图中的 `140`、`141`、`142` 是切片索引，不是器官编号，也不是 epoch。

## 一、PNG 文件：逐切片可视化检查图

### 1. `case0001_140_gt.png`

其中：

- `case0001`：病例名；
- `140`：第 140 张切片；
- `gt`：ground truth，真实标签；
- `.png`：二维可视化图片。

当前代码先取出：

```python
lbl = label[ind, :, :]
```

再把标签图拆成每个类别的二值掩膜：

```python
for i in range(1, classes):
    masks.append(lbl == i)
```

最后叠加到 CT 灰度切片上：

```python
fig_gt = overlay_masks(
    image[ind, :, :],
    masks,
    ...
)
```

所以 `gt.png` 不是原始标签矩阵的直接黑白导出，而是：

```text
CT 灰度切片 + 真实标签彩色半透明叠加图
```

它表示“真实标注认为器官在哪里”。

### 2. `case0001_140_pred.png`

其中 `pred` 是 prediction，表示模型预测结果。

代码先得到模型预测类别图：

```python
out = torch.argmax(
    torch.softmax(outputs, dim=1),
    dim=1,
).squeeze(0)
```

再构造每个前景类别的预测掩膜：

```python
for i in range(1, classes):
    preds_o.append(pred == i)
```

最后叠加到同一张 CT 切片上：

```python
fig_pred = overlay_masks(
    image[ind, :, :],
    preds_o,
    ...
)
```

所以 `pred.png` 表示：

```text
CT 灰度切片 + 模型预测标签彩色半透明叠加图
```

### 3. `gt.png` 和 `pred.png` 如何对比

对同一个切片，应成对比较：

```text
case0001_140_gt.png
case0001_140_pred.png
```

重点观察：

1. 器官位置是否一致；
2. 预测区域是否比真实区域大或小；
3. 器官边界是否粗糙、断裂或泄漏；
4. 小器官是否完全漏检；
5. 预测颜色是否与真实图中的同一类别对应；
6. 是否出现疑似标签名称或标签编号错位。

注意：当前 PNG 是两个独立的叠加图，不是“绿色表示正确、红色表示错误”的差异图。不能只通过文件大小或肉眼颜色深浅直接得出 Dice。

如果需要严格看错误区域，可以额外生成：

```text
真阳性区域：预测和真实都为该类
假阳性区域：预测有、真实没有
假阴性区域：真实有、预测没有
```

当前项目默认并没有生成这种差异图。

### 4. 为什么 PNG 文件很多

一个三维病例包含 `D` 张切片，代码会循环：

```python
for ind in range(image.shape[0]):
```

每张切片保存两个 PNG：

```text
一个 `_gt.png`
一个 `_pred.png`
```

因此一个包含 150 张切片的病例，大约会产生：

```text
150 × 2 = 300 个 PNG
```

这正是截图中从 `case0001_140`、`case0001_141`、`case0001_142` 连续出现文件的原因。

## 二、NIfTI 文件：完整三维病例结果

### 1. `case0001_img.nii.gz`

这是保存的 CT 图像体：

```python
img_itk = sitk.GetImageFromArray(image.astype(np.float32))
sitk.WriteImage(img_itk, ... + "_img.nii.gz")
```

它是完整三维体，不是单张切片。通常包含：

```text
[D,H,W] 的 CT 灰度数据
```

它用于：

- 在 ITK-SNAP、3D Slicer 等软件中作为背景图像；
- 与 `gt.nii.gz`、`pred.nii.gz` 叠加查看；
- 检查推理前后的空间方向和病例对应关系。

注意：当前代码保存的是预处理后的归一化 CT，不一定是原始 HU 值。代码中使用的是已经加载进来的 `image`，而该 image 通常经过了裁剪和归一化。

### 2. `case0001_gt.nii.gz`

这是完整三维真实标签体：

```python
lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
sitk.WriteImage(lab_itk, ... + "_gt.nii.gz")
```

它的每个体素是整数类别编号，例如：

```text
0：背景编号
1..8：前景类别编号
```

这里的数字是标签编号，不应直接从文件名推断器官名称。标签编号与器官的对应关系必须由数据集映射证据确认。

虽然保存时转换为了 `float32`，但语义仍然是离散标签，不是连续灰度图。查看时应该使用离散颜色表或 label overlay，不要把它当普通 CT 灰度图显示。

### 3. `case0001_pred.nii.gz`

这是完整三维模型预测标签体：

```python
prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
sitk.WriteImage(prd_itk, ... + "_pred.nii.gz")
```

`prediction` 是把每张切片预测结果拼回去后的三维数组：

```python
prediction[ind] = pred
```

所以它与 `gt.nii.gz` 的空间形状应一致：

```text
pred.shape == gt.shape == [D,H,W]
```

每个体素保存的是模型选择出的整数类别编号，而不是 softmax 概率。模型内部的概率经过 `argmax` 后已经被转成离散标签。

## 三、三种 NIfTI 文件如何一起查看

推荐用 ITK-SNAP 或 3D Slicer：

```text
1. 打开 case0001_img.nii.gz 作为主图像
2. 加载 case0001_gt.nii.gz 作为真实标签层
3. 加载 case0001_pred.nii.gz 作为预测标签层
4. 调整标签层透明度
5. 逐层滚动检查同一切片
```

最重要的是三者必须是同一个病例：

```text
case0001_img.nii.gz
case0001_gt.nii.gz
case0001_pred.nii.gz
```

不要把 `case0001_img` 和 `case0002_pred` 混合比较。

## 四、如何用 Python 检查 NIfTI 文件

```python
import nibabel as nib
import numpy as np

path = r"D:\path\case0001_pred.nii.gz"
nii = nib.load(path)
array = nii.get_fdata()

print("shape:", array.shape)
print("dtype:", array.dtype)
print("spacing:", nii.header.get_zooms())
print("labels:", np.unique(array))
```

对 `gt` 和 `pred` 都运行一次，应该重点检查：

```text
shape 是否一致
spacing 是否一致
标签唯一值是否合理
病例空间是否一致
```

## 五、当前代码中的 spacing 注意事项

当前保存代码使用：

```python
img_itk.SetSpacing((1, 1, z_spacing))
prd_itk.SetSpacing((1, 1, z_spacing))
lab_itk.SetSpacing((1, 1, z_spacing))
```

而 `test_synapse.py` 当前传入：

```python
z_spacing=1
```

因此这批保存的 NIfTI 被写成近似 `(1,1,1)` 的 spacing。文件可以正常打开，但这不一定等于原始 CT 的真实体素间距。

这会影响：

- 软件中的空间测量；
- 真实物理距离解释；
- 依赖 spacing 的 HD95、ASD 报告。

另外，当前 `gt.nii.gz` 和 `pred.nii.gz` 是用数组重新创建的 SimpleITK 图像，代码只设置了 spacing，没有完整复制原始 NIfTI 的 direction、origin 等元数据。做严格空间配准分析时需要额外核对这些信息。

## 六、从文件名不能得到什么

以下信息不能仅靠文件名确定：

- `label 5` 一定是哪个器官；
- 某个预测颜色对应哪个真实器官名称；
- PNG 文件大小是否代表指标好坏；
- `pred.nii.gz` 中的数值是否已经是概率；
- NIfTI 的 spacing 是否等于原始 CT spacing。

这些需要分别查看：

```text
标签映射代码/原始数据
颜色表和可视化代码
argmax 前后的模型输出逻辑
NIfTI header 和原始 spacing
```

## 七、最简文件对照表

| 文件模式 | 范围 | 内容 | 主要用途 |
|---|---|---|---|
| `caseXXXX_N_gt.png` | 单张切片 | CT + 真实标签叠加图 | 人工检查真实标注 |
| `caseXXXX_N_pred.png` | 单张切片 | CT + 模型预测叠加图 | 人工检查预测 |
| `caseXXXX_img.nii.gz` | 完整三维病例 | 预处理后的 CT 体 | 作为查看背景 |
| `caseXXXX_gt.nii.gz` | 完整三维病例 | 三维真实整数标签 | 与预测比较 |
| `caseXXXX_pred.nii.gz` | 完整三维病例 | 三维预测整数标签 | 计算/查看预测结果 |

## 最后总结

你截图中的 `case0001_140_gt.png` 和 `case0001_140_pred.png` 是同一病例、同一张切片的真实标签叠加图和预测标签叠加图；`case0001_img.nii.gz`、`case0001_gt.nii.gz`、`case0001_pred.nii.gz` 则是同一病例的完整三维 CT、真实标签和预测标签。

最稳妥的检查方式是：先用 PNG 快速看切片级对齐，再用 ITK-SNAP/3D Slicer 同时加载三个 NIfTI 查看三维结构，最后用 Python 检查三者的 shape、spacing 和唯一标签值。
