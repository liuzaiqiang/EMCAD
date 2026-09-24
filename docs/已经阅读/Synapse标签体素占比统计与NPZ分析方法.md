# Synapse 标签体素占比统计与 `.npz` 分析方法

## 直接运行

项目根目录已经提供脚本：

```text
analyze_synapse_labels.py
```

默认读取：

```text
D:\files\Medical_Image_Segmentation_Projects\data\Synapse
├─ train_npz
└─ test_vol_h5
```

运行：

```powershell
python analyze_synapse_labels.py
```

如果当前 Python 没有安装依赖：

```powershell
python -m pip install numpy h5py
```

也可以指定目录：

```powershell
python analyze_synapse_labels.py --root "D:\files\Medical_Image_Segmentation_Projects\data\Synapse"
```

脚本会：

- 读取训练集全部 `.npz` 文件；
- 读取测试集全部 `.h5` 文件；
- 统计标签 `0..8` 的体素数；
- 计算占全部体素的比例；
- 计算占前景体素的比例；
- 在终端逐文件打印标签形状和各标签计数；
- 生成汇总 CSV 和逐文件 CSV。

生成的文件包括：

```text
synapse_label_stats_train_npz_aggregate.csv
synapse_label_stats_train_npz_per_file.csv
synapse_label_stats_test_h5_aggregate.csv
synapse_label_stats_test_h5_per_file.csv
```

## 重要提醒：体素占比只能作为初步旁证

你给出的名称列表是：

```python
[
    'spleen', 'right kidney', 'left kidney', 'gallbladder',
    'pancreas', 'liver', 'stomach', 'aorta'
]
```

脚本会按这个列表显示名称，但它不会修改标签，也不会自动证明名称列表正确。体素总量可以帮助发现明显错位，例如：

- 某个标签体素量非常大，比较像 liver；
- 某个标签体素量很小，比较像 pancreas 或 gallbladder；
- 左右肾体积通常接近；
- spleen、stomach、aorta 处于不同的体积量级。

但不同病例、扫描范围、裁剪方式会影响器官体积。仅凭“谁大谁小”不能作为最终证据。最终证据仍应是：当前标签与原始 NIfTI 标签在同一病例、同一空间位置上的逐体素映射。

## 如何打开一个 `.npz` 文件

`.npz` 是 NumPy 的压缩容器，不是普通文本文件，也不是 NIfTI。它通常包含多个键；本项目每个训练文件一般包含：

```text
image：二维 CT 切片
label：二维整数标签图
```

### 方法一：用 Python 查看（推荐）

```python
import numpy as np

path = r"D:\files\Medical_Image_Segmentation_Projects\data\Synapse\train_npz\case0005_slice000.npz"

with np.load(path, allow_pickle=False) as data:
    print("文件中的键：", data.files)
    image = data["image"]
    label = data["label"]

    print("image shape:", image.shape)
    print("image dtype:", image.dtype)
    print("label shape:", label.shape)
    print("label dtype:", label.dtype)
    print("label values:", np.unique(label))

    values, counts = np.unique(label, return_counts=True)
    for value, count in zip(values, counts):
        print(f"label={int(value)}: {int(count):,} voxels")
```

### 方法二：用 VS Code 运行

1. 用 VS Code 打开项目目录。
2. 新建 `inspect_one_npz.py`。
3. 粘贴上面的代码。
4. 把 `path` 改成你要检查的文件。
5. 在 VS Code 终端运行：

   ```powershell
   python inspect_one_npz.py
   ```

### 方法三：查看标签图像

如果还想把 `label` 画出来：

```python
import matplotlib.pyplot as plt
import numpy as np

path = r"D:\files\Medical_Image_Segmentation_Projects\data\Synapse\train_npz\case0005_slice000.npz"
with np.load(path, allow_pickle=False) as data:
    image = data["image"]
    label = data["label"]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(label, cmap="tab10", vmin=0, vmax=8)
plt.title("label")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image, cmap="gray")
plt.imshow(label, cmap="tab10", vmin=0, vmax=8, alpha=0.45)
plt.title("overlay")
plt.axis("off")

plt.tight_layout()
plt.show()
```

安装绘图依赖：

```powershell
python -m pip install matplotlib
```

### 方法四：不要用记事本直接打开

记事本打开 `.npz` 只会看到二进制乱码；这不表示文件损坏。`.npz` 应通过 NumPy、Python、VS Code 的 Python 环境或 Jupyter Notebook 读取。

## 如何理解一个 `.npz` 的结果

假设输出：

```text
image shape: (224, 224)
label shape: (224, 224)
label values: [0 1 4 6]
label=0: 42,000 voxels
label=1:  3,000 voxels
label=4:  1,500 voxels
label=6:  3,724 voxels
```

含义是：

- 这是一张 `224×224` 的训练切片；
- 当前切片只出现背景、标签 1、标签 4、标签 6；
- 没出现的标签不代表数据错误，只表示这些器官不在这一张切片中；
- 标签值是离散整数，不是灰度强度；
- 不能把 `label=5` 直接当成 pancreas，除非你已经确认当前数据的标签映射。

## CSV 应该怎么看

`*_aggregate.csv` 用来比较整个训练集或测试集的总体分布；`*_per_file.csv` 用来定位异常文件。

重点看三类列：

```text
label_5_pancreas_voxels
label_5_pancreas_pct_all
label_5_pancreas_pct_foreground
```

其中：

- `pct_all`：该标签占背景加全部前景的比例；
- `pct_foreground`：该标签占所有前景器官体素的比例；
- 对比器官大小时，`pct_foreground` 通常更直观；
- 判断背景是否异常时，`pct_all` 更重要。

如果你发现某个文件出现 `label_9`、`label_10` 等超出 `0..8` 的标签，脚本会把它显示成 `unknown_label_9` 等，这说明该文件仍含 14 类编码或存在其他标签问题，需要单独排查。

## 推荐的分析顺序

```text
先看 aggregate.csv 的总体体素分布
    ↓
再看 per_file.csv 是否有异常病例/切片
    ↓
用 inspect_one_npz.py 查看具体切片
    ↓
最后与原始 NIfTI 做逐体素映射
```

体素占比适合做第一轮筛查；它不能单独证明器官标签顺序。尤其不要仅根据某个器官“通常比较大”就修改 `classes`。
