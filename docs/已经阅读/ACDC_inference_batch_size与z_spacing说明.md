# ACDC 测试参数：`inferance_batch_size` 与 `z_spacing`

## 结论

```bash
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
Z_SPACING="${Z_SPACING:-10.0}"
```

这两个参数作用完全不同：

| 参数 | 作用对象 | 影响预测类别 | 影响 Dice/Jaccard | 影响 HD95/ASD | 影响显存和速度 |
|---|---|---:|---:|---:|---:|
| `INFERENCE_BATCH_SIZE` | 每病例并行推理的切片数 | 理论上不影响 | 理论上不影响 | 理论上不影响 | 会影响 |
| `Z_SPACING` | 三维 z 方向体素物理间距 | 不影响 | 不影响 | 会影响 | 不影响 |

## 1. `INFERENCE_BATCH_SIZE`

当前 `start_test_acdc.sh` 把它传给：

```bash
--inference_batch_size "${INFERENCE_BATCH_SIZE}"
```

随后 `test_acdc.py` 传给：

```python
prediction = predict_volume(
    ...,
    batch_size=args.inference_batch_size,
)
```

`predict_volume()` 对一个病例的 `[D,H,W]` 体数据沿深度方向分批：

```python
for start in range(0, depth, batch_size):
    batch = image[start:start + batch_size]
```

例如一个病例有 24 张切片：

```text
INFERENCE_BATCH_SIZE=8  ->  3 次模型前向：8 + 8 + 8
INFERENCE_BATCH_SIZE=4  ->  6 次模型前向：4 + 4 + 4 + 4 + 4 + 4
```

它不是训练阶段的 `batch_size`，不会改变每次参数更新，也不会改变训练样本数量。

### 对结果的影响

当前 `predict_volume()` 中模型处于 `model.eval()` 和 `torch.no_grad()`：

- 不进行反向传播；
- 不更新参数；
- BatchNorm 使用已有统计量；
- Dropout 关闭。

因此，在显存足够且没有数值溢出的情况下，改变它不应改变预测结果和指标。理论上只会改变：

- GPU 峰值显存；
- 推理速度；
- GPU 利用率；
- 是否发生 CUDA out-of-memory。

建议：

```text
显存充足：8 或 16，通常更快
显存不足：4、2 或 1
需要严格比较模型指标：所有实验固定同一个值
```

不要为了提高 Dice 调大这个参数；它不是模型性能超参数。

## 2. `Z_SPACING`

当前测试代码构造：

```python
voxelspacing = (float(args.z_spacing), 1.0, 1.0)
```

当前预测数组形状是 `[D,H,W]`，所以代码将其解释为：

```text
(深度方向 z 间距, 高度方向 y 间距, 宽度方向 x 间距)
= (Z_SPACING, 1.0, 1.0)
```

该间距传给：

```python
metric.binary.hd95(prediction, target, voxelspacing=voxelspacing)
metric.binary.assd(prediction, target, voxelspacing=voxelspacing)
```

因此，`HD95` 和 `ASD` 会按物理距离计算，而不是单纯按体素索引距离计算。

### `Z_SPACING=10.0` 的含义

它表示代码假设相邻两张 ACDC 切片之间相距 10.0 个物理单位，通常按毫米解释：

```text
z 方向相邻切片距离 = 10.0 mm
y 方向相邻像素距离 = 1.0
x 方向相邻像素距离 = 1.0
```

如果真实 ACDC NIfTI 文件的 z 间距不是 10.0，那么：

- Dice 不变；
- Jaccard 不变；
- HD95 可能不准确；
- ASD 可能不准确；
- 保存的 NIfTI 文件的 z 方向空间信息也会被写成 10.0。

当前 `save_nifti_triplet()` 同样用这个值写入：

```python
itk_image.SetSpacing((1.0, 1.0, float(z_spacing)))
```

## 3. 这两个参数是否影响你当前模型对比

如果 EMCAD baseline 和创新模型使用：

```text
相同 INFERENCE_BATCH_SIZE
相同 Z_SPACING
相同测试病例列表
相同 img_size
相同 checkpoint 选择规则
```

那么 Dice/Jaccard 的模型对比是公平的，HD95/ASD 也具有相同的计算口径。

论文中建议记录：

```text
img_size=224 或 256
inference_batch_size=8
z_spacing=实际采用的值
```

其中 `INFERENCE_BATCH_SIZE` 主要属于推理资源配置，`Z_SPACING` 属于指标和输出空间定义配置。

## 4. 当前项目的特别注意点

`Z_SPACING=10.0` 是脚本默认值，不等于每个 ACDC 病例真实的 z 间距。若论文需要严谨报告 HD95/ASD，应从每个病例的 NIfTI header 读取真实 spacing，并在指标计算时逐病例使用；不能把固定 10.0 直接称为真实物理间距，除非你确认数据预处理已统一到该 spacing。

