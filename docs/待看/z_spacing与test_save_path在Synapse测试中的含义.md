# `z_spacing` 与 `test_save_path` 在 Synapse 测试中的含义

## 一、`z_spacing=1` 是什么意思

当前 `test_synapse.py` 调用：

```python
metric_i = test_single_volume(
    image, label, model,
    classes=args.num_classes,
    patch_size=[args.img_size, args.img_size],
    test_save_path=test_save_path,
    case=case_name,
    z_spacing=1,
    class_names=classes,
)
```

这里的 `z_spacing=1` 表示：保存 NIfTI 时，把相邻两张切片之间的物理间距写成 1 个单位，通常可理解为 1 mm。

在 `utils/utils.py` 中：

```python
img_itk.SetSpacing((1, 1, z_spacing))
prd_itk.SetSpacing((1, 1, z_spacing))
lab_itk.SetSpacing((1, 1, z_spacing))
```

因此当前保存结果的 spacing 是：

```text
x 方向：1
y 方向：1
z 方向：1
```

## 二、为什么 spacing 会影响 HD95 和 ASD

Dice 和 Jaccard 主要比较体素重叠数量，对 spacing 不敏感；但 HD95 和 ASD 是表面距离指标，距离单位取决于体素间距。

例如，两个表面相差 5 个 z 方向切片：

```text
真实 z spacing = 5 mm  -> 物理距离约 25 mm
代码 z_spacing = 1     -> 物理距离被解释为约 5 mm
```

所以如果真实 NIfTI 的 z 间距不是 1，而代码保存时强行写成 1：

- 保存出来的 NIfTI 空间尺度不真实；
- 在支持 spacing 的软件中测量距离会不准确；
- HD95、ASD 的“毫米”解释可能偏小或偏大。

## 三、当前代码中更重要的细节

当前 `test_single_volume()` 的指标计算调用是：

```python
calculate_metric_percase(prediction == i, label == i)
```

这个函数内部调用 MedPy 的二值指标，但当前调用没有把 spacing 传给 MedPy。因此当前代码实际计算的 HD95/ASD 很可能按数组坐标（体素单位）计算，而不是严格按真实毫米计算。

也就是说，`z_spacing=1` 至少会影响保存 NIfTI 的空间元数据；对于当前这条指标计算路径，不能简单认为设置它就已经让 HD95/ASD 变成了真实毫米值。若要报告物理距离，需要在指标函数中显式传入真实 spacing，并确认 x、y、z 顺序正确。

## 四、真实 spacing 应从哪里来

理想流程是：

```text
原始 NIfTI header
    -> 读取真实 spacing
    -> 预处理时保存到病例元数据
    -> 测试时按病例传给评估和 NIfTI 保存
```

不能把所有 Synapse 病例都默认当成 `(1,1,1)`，也不能把一个固定值当作所有病例的真实 spacing，除非已经核对过原始数据确实如此。

还要注意坐标顺序：SimpleITK 的 `SetSpacing` 使用 `(x, y, z)`，而 NumPy 数组访问通常写成 `[z, y, x]`。数组轴和 spacing 顺序必须对应。

## 五、`test_save_path` 是什么意思

`test_save_path` 是测试结果输出目录。例如：

```python
test_save_path = "model_pth/.../predictions"
```

当它不为 `None` 时，函数末尾会保存：

```text
caseXXXX_pred.nii.gz  预测标签体
caseXXXX_img.nii.gz   CT 图像体
caseXXXX_gt.nii.gz    真实标签体
```

三维循环中还会保存切片可视化 PNG：

```text
caseXXXX_0_gt.png
caseXXXX_0_pred.png
caseXXXX_1_gt.png
caseXXXX_1_pred.png
...
```

这些 PNG 是把每张 CT 切片与真实/预测掩膜叠加后的检查图，用于人工查看空间对齐和分割质量。

## 六、为什么说目录必须正确设置

当前保存语句使用字符串拼接，例如：

```python
fig_gt.savefig(test_save_path + '/' + case + '_' + str(ind) + '_gt.png')
```

因此需要同时满足：

1. `test_save_path` 不是 `None`；
2. 该目录已经存在；
3. `case` 是有效病例名；
4. 当前进程对目录有写权限；
5. 磁盘空间足够。

否则可能出现：

```text
TypeError：test_save_path 为 None，无法拼接字符串
FileNotFoundError：目录不存在
PermissionError：没有写权限
```

## 七、为什么会感觉“我没有要求保存 PNG，怎么还报错”

当前代码的三维分支中，PNG 保存语句位于切片循环内；历史实现没有始终用：

```python
if test_save_path is not None:
```

包住 PNG 保存代码，而 NIfTI 保存部分有条件判断。因此可能出现：

```text
你以为 test_save_path=None 就是不保存结果
实际上循环中先执行了 PNG 保存
然后因为 None 拼接路径而报错
```

所以当前实际测试应传入一个已创建的输出目录。若确实不想保存 PNG，需要修改保存逻辑或增加独立的保存开关，不能只依赖 `None` 猜测行为。

## 八、实际检查清单

测试前检查：

```python
from pathlib import Path

save_dir = Path(test_save_path)
save_dir.mkdir(parents=True, exist_ok=True)
print(save_dir.resolve())
```

检查 spacing 时记录：

```text
原始病例 spacing
传入的 z_spacing
保存后 NIfTI spacing
HD95/ASD 的计算是否显式使用 spacing
```

## 最后总结

第一句话的意思是：当前代码把保存文件的体素间距写成 `(1,1,1)`，如果真实 CT 的层厚不是 1，保存后的空间距离解释就不准确；而且当前指标函数还需要进一步确认是否真正使用了 spacing。

第二句话的意思是：测试过程中会写入 PNG 和 NIfTI，`test_save_path` 必须是一个存在且可写的目录，否则三维病例推理可能在保存阶段报错，即使模型前向本身完全正常。
