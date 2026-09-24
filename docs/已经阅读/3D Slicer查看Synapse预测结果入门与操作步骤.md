# 3D Slicer 查看 Synapse 预测结果：入门与操作步骤

## 先看懂你当前的界面

你截图中已经加载了：

```text
case0001_gt.nii.gz
```

画面中三个二维窗口左下角都显示：

```text
B: case0001_gt
```

这里的 `B` 是 Background（背景图像）的意思。也就是说，你目前把真实标签文件当成了普通背景体数据加载，而不是作为彩色标签覆盖层显示。

因此当前画面看到的是黑色背景上的灰色/白色区域，3D 窗口里也可能只有一个框而没有清晰的器官表面。这不一定表示文件错误，主要是显示方式还没有设置好。

## 一、三个文件各自应该扮演什么角色

对于一个病例，建议这样使用：

```text
case0001_img.nii.gz   CT 图像，作为 Background
case0001_gt.nii.gz    真实标签，作为标签覆盖层
case0001_pred.nii.gz  模型预测标签，作为另一个标签覆盖层
```

不要把 `gt` 或 `pred` 当作普通 CT 灰度背景来单独查看；它们本质上是离散整数标签图。

## 二、最简单的查看流程：先把 CT 作为背景

### 第 1 步：打开 `case0001_img.nii.gz`

可以使用任意一种方法：

1. 点击左上角 `File`；
2. 选择 `Add Data...`；
3. 选择：

   ```text
   case0001_img.nii.gz
   ```

4. 点击 `OK` 或 `Load`。

也可以把文件直接拖入 Slicer 窗口。

加载后，`case0001_img` 应该作为 CT 背景显示。二维窗口中的 `B:` 后面应当出现 `case0001_img`，而不是 `case0001_gt`。

如果图像太黑或太亮，可以在左侧模块下拉框中选择 `Volumes`，调整 Window/Level；也可以在切片窗口中使用鼠标左键拖动进行亮度/对比度调整。

## 三、临时叠加查看：把 GT 或预测放到前景层

### 方法 A：用切片窗口顶部的前景/背景选择器

每个 Red、Green、Yellow 切片窗口顶部都有自己的控制条。不同 Slicer 主题下图标可能略有差异，但通常可以找到：

```text
Background volume selector
Foreground volume selector
Foreground opacity slider
```

设置为：

```text
Background = case0001_img
Foreground = case0001_gt
```

然后把前景透明度调到大约 `0.3~0.6`，就能看到：

```text
CT 灰度图 + GT 标签覆盖层
```

再把前景切换为：

```text
Foreground = case0001_pred
```

就能查看：

```text
CT 灰度图 + 预测标签覆盖层
```

### 方法 B：使用 Data 模块选择可见层

1. 在顶部 `Modules` 下拉框中选择 `Data`；
2. 在数据树中找到 `case0001_img`、`case0001_gt`、`case0001_pred`；
3. 确认三个文件都已加载；
4. 通过切片窗口控制条选择背景层和前景层；
5. 使用前景透明度滑块比较真实标签和预测结果。

注意：仅仅在 Data 树中点击眼睛图标，不一定能得到正确的前景/背景关系；关键是要把 CT 指定为背景，把标签指定为前景。

## 四、正确显示离散标签：推荐转换为 Segmentation

直接把 `gt.nii.gz` 当 Volume 打开时，Slicer 可能把它当作普通标量体显示。更适合的做法是将标签图转换为 Segmentation。

### 导入 GT 标签

不同 Slicer 版本按钮名称可能略有不同，常见路径是：

1. 打开 `Segmentations` 模块；
2. 创建一个新的 segmentation；
3. 找到 `Import/Export`、`Import labelmap` 或类似按钮；
4. 选择已经加载的 `case0001_gt` 作为 labelmap；
5. 确认导入。

导入后，Slicer 会把整数标签 `1..8` 作为不同的 segment；背景 `0` 不会作为器官 segment 导入。

导入后可以在 `Segmentations` 模块中：

- 打开或关闭 2D 可见性；
- 打开或关闭 3D 可见性；
- 调整整体透明度；
- 单独隐藏某个标签编号；
- 查看每个 segment 的体积。

### 导入预测标签

对 `case0001_pred.nii.gz` 重复同样操作，但建议创建第二个 segmentation 节点，例如：

```text
GT_Segmentation
Pred_Segmentation
```

这样可以分别控制真实标签和预测标签的可见性。

不要急着把 `Segment_1` 改名为 spleen、liver 等器官。当前项目正在核查标签顺序，Slicer 导入时只应先保留数字编号，避免把未经证明的名称写进去。

## 五、如何打开三维显示

右上角紫色窗口是 3D 视图。直接加载一个普通 `gt` Volume 时，3D 视图不一定自动显示器官表面。

正确做法：

1. 把 `gt` 或 `pred` 导入为 Segmentation；
2. 在 `Segmentations` 模块或数据树中打开 3D 可见性；
3. 右上角 3D 窗口中应出现彩色器官表面；
4. 拖动鼠标旋转模型，滚轮缩放，按住中键平移。

如果只想看 CT 的三维体渲染，需要使用 `Volume Rendering` 模块；但对于标签结果检查，Segmentation 的 3D 显示通常更直接。

## 六、如何逐层查看切片

Slicer 有三个正交二维视图：

```text
Red    一个方向的切面
Green  第二个方向的切面
Yellow 第三个方向的切面
```

每个视图顶部都有切片位置滑块。你可以：

- 拖动滑块浏览切片；
- 鼠标悬停在窗口内滚动滚轮；
- 打开十字光标，让三个视图联动；
- 点击某个器官位置，让其他两个视图跳到对应坐标。

你的文件名中的：

```text
case0001_140_gt.png
```

表示当前 Python 推理循环中的 `ind=140`，但 Slicer 的滑块显示可能使用不同的空间索引或物理坐标。不要仅凭顶部显示的毫米数字，直接把它当成 Python 的 `ind`。

## 七、GT 与 Pred 的实际比较方法

建议按以下顺序操作：

1. 加载 `case0001_img.nii.gz`；
2. 导入 `case0001_gt.nii.gz` 为 `GT_Segmentation`；
3. 导入 `case0001_pred.nii.gz` 为 `Pred_Segmentation`；
4. 同时打开 GT 和 Pred 的 2D 可见性；
5. 将两个 segmentation 设置为不同透明度或不同颜色；
6. 在 Red/Green/Yellow 三个视图逐层检查；
7. 再打开 3D 可见性查看整体形状。

重点观察：

- GT 与 Pred 是否在同一空间位置；
- 是否整体平移或翻转；
- 某个器官是否漏检；
- 预测边界是否比 GT 大很多；
- 小器官是否完全没有预测；
- 预测中是否出现 GT 没有的区域。

如果所有器官都整体错位，优先怀疑 spacing、origin、direction 或轴顺序；如果只有某些标签错位，才进一步怀疑类别编号或模型预测问题。

## 八、截图中出现灰色、白色而不是彩色，怎么办

这通常表示 `case0001_gt` 被当作普通 Scalar Volume 显示，而不是离散 Labelmap/Segmentation。

可以依次尝试：

1. 将 `case0001_img` 设置为 Background；
2. 将 `case0001_gt` 设置为 Foreground；
3. 调低 Foreground opacity；
4. 在 `Volumes` 模块关闭标签图的插值；
5. 使用 `Segmentations` 模块导入 labelmap；
6. 打开 segmentation 的 2D/3D 可见性。

标签图是离散类别，不应使用平滑插值；否则边界会被显示得模糊，甚至看起来出现不存在的中间值。

## 九、如何检查文件是否真的成功对应

在 Slicer 中，三者必须是同一病例：

```text
case0001_img.nii.gz
case0001_gt.nii.gz
case0001_pred.nii.gz
```

使用 Python 也可以检查：

```python
import nibabel as nib
import numpy as np

for suffix in ["img", "gt", "pred"]:
    path = fr"D:\\your\\predictions\\case0001_{suffix}.nii.gz"
    nii = nib.load(path)
    print(suffix)
    print("  shape:", nii.shape)
    print("  zooms:", nii.header.get_zooms())
    print("  values:", np.unique(nii.get_fdata())[:20])
```

至少确认：

```text
img、gt、pred 的 shape 一致
gt 和 pred 的唯一值是离散标签
gt 和 pred 的 spacing 一致
三者的空间位置没有明显偏移
```

## 十、关于当前项目输出文件的现实限制

当前项目保存 NIfTI 时使用类似：

```python
img_itk.SetSpacing((1, 1, z_spacing))
prd_itk.SetSpacing((1, 1, z_spacing))
lab_itk.SetSpacing((1, 1, z_spacing))
```

测试调用中 `z_spacing=1`，所以保存结果的 spacing 近似是 `(1,1,1)`。这通常仍然可以让 `img`、`gt`、`pred` 三者彼此叠加查看，因为三者使用同一套保存规则；但它不一定等于原始 Synapse CT 的真实物理 spacing。

因此：

- 用 Slicer 检查相对位置和形状通常没问题；
- 用 Slicer 测量真实毫米距离时要谨慎；
- 用 spacing 解释 HD95、ASD 时要先核对原始 spacing；
- 不要把截图中的画面直接当作标签编号与器官名称的证明。

## 最短操作路线

如果只想快速完成一次正确查看：

```text
1. File -> Add Data -> case0001_img.nii.gz
2. File -> Add Data -> case0001_gt.nii.gz
3. File -> Add Data -> case0001_pred.nii.gz
4. 将 img 设置为 Background
5. 将 gt 设置为 Foreground，调整透明度
6. 逐层检查
7. 再切换 pred 为 Foreground
8. 需要三维显示时，把 gt/pred 导入 Segmentations 并打开 3D 可见性
```

你当前截图的下一步，就是先加载 `case0001_img.nii.gz`，把它设为 Background，再把 `case0001_gt` 从普通背景体改为前景标签或 Segmentation。
