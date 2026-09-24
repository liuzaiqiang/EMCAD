# `test_synapse.py` 第 125–130 行：`liver` 与 `pancreas` 顺序核对

## 1. 最终结论

当前代码是：

```python
classes = ['spleen', 'right kidney', 'left kidney',
           'gallbladder', 'liver', 'pancreas', 'stomach', 'aorta']
```

而被注释掉的原始代码是：

```python
classes = ['spleen', 'right kidney', 'left kidney', 'gallbladder',
           'pancreas', 'liver', 'stomach', 'aorta']
```

**如果你的训练标签遵循 Synapse 常见的 9 类映射，那么原始顺序 `pancreas → liver` 才是正确的；你当前把它改成 `liver → pancreas`，从标签编号对应关系看是不正确的。**

但还要区分两件事：

1. 这几行传入的是 `class_names`，主要用于可视化图例和器官名称显示；
2. 当前 `test_single_volume()` 的 Dice、HD95、Jaccard、ASD 是按整数标签 `1..8` 直接计算的，不读取器官字符串名称。

因此，**仅修改第 125–130 行通常不会改变数值指标，但可能把可视化图例中的肝脏和胰腺名称标错。**

## 2. Synapse 标准标签编号

原始 Synapse 多器官标签通常采用如下编号：

| 原始标签编号 | 器官 |
|---:|---|
| 0 | 背景 |
| 1 | spleen |
| 2 | right kidney |
| 3 | left kidney |
| 4 | gallbladder |
| 5 | esophagus |
| 6 | liver |
| 7 | stomach |
| 8 | aorta |
| 9 | inferior vena cava |
| 10 | portal vein and splenic vein |
| 11 | pancreas |
| 12 | right adrenal gland |
| 13 | left adrenal gland |

如果做 9 类任务，即：

```text
背景 + 8 个前景器官
```

常见处理是保留：

```text
1 spleen
2 right kidney
3 left kidney
4 gallbladder
6 liver
7 stomach
8 aorta
11 pancreas
```

并把不参与任务的器官设为背景：

```text
5、9、10、12、13 → 0
```

为了让 8 个前景类别连续成为 `1..8`，还需要把保留下来的原始标签重新映射。例如常见的 9 类顺序是：

| 训练/测试类别编号 | 器官 | 对应原始 Synapse 标签 |
|---:|---|---:|
| 1 | spleen | 1 |
| 2 | right kidney | 2 |
| 3 | left kidney | 3 |
| 4 | gallbladder | 4 |
| 5 | pancreas | 11 |
| 6 | liver | 6 |
| 7 | stomach | 7 |
| 8 | aorta | 8 |

因此，若你的 9 类标签已经按照这种顺序重映射，那么 `class_names` 必须写成：

```python
[
    'spleen',
    'right kidney',
    'left kidney',
    'gallbladder',
    'pancreas',
    'liver',
    'stomach',
    'aorta',
]
```

也就是说，在常见 9 类映射中：

```text
类别 5 = pancreas
类别 6 = liver
```

所以原始代码中的 `pancreas, liver` 并没有写反。

## 3. 你当前修改为什么看起来可能“有道理”

从自然语言或原始器官编号看，`liver` 的编号 6 小于 `pancreas` 的编号 11，容易直觉上写成：

```text
liver, pancreas
```

但 9 类训练任务不是简单把原始器官按原编号全部保留。为了去掉 esophagus、血管和肾上腺，通常会重新压缩类别编号。压缩后，胰腺被放到第 5 个前景类别，肝脏被放到第 6 个前景类别：

```text
原始 11（pancreas） → 训练类别 5
原始 6（liver）    → 训练类别 6
```

因此不能只根据原始标签编号大小来排列 `class_names`，必须根据**实际训练标签中的类别编号**排列。

## 4. 仓库代码给出的直接证据

### 4.1 `test_synapse.py` 当前只把名称传给可视化

当前调用类似：

```python
metric_i = test_single_volume(
    image,
    label,
    model,
    classes=args.num_classes,
    patch_size=[args.img_size, args.img_size],
    test_save_path=test_save_path,
    case=case_name,
    z_spacing=1,
    class_names=classes,
)
```

### 4.2 `utils/utils.py` 明确说明 `class_names` 只影响显示名称

在 `test_single_volume()` 中：

```python
if class_names == None:
    mask_labels = np.arange(1, classes)
else:
    mask_labels = class_names
```

之后 `mask_labels` 被用于 `overlay_masks()` 的图例和叠加图标签。

代码注释也明确写出：

```text
class_names 只影响可视化图例，不影响 prediction 的类别编号和指标计算。
```

### 4.3 指标计算使用整数 `i`，不使用字符串名称

当前指标循环是：

```python
for i in range(1, classes):
    metric_list.append(
        calculate_metric_percase(prediction == i, label == i)
    )
```

这里的 `i=1..8` 直接对应预测和标签中的整数类别。它不会检查：

```python
classes[4] == 'pancreas'
```

还是：

```python
classes[4] == 'liver'
```

所以只改 `classes` 名称，不会把模型的第 5 个输出通道改成另一个器官，也不会交换 Dice 数组的列。

## 5. 更严重的问题：当前数据预处理中的重映射实际上被注释了

`utils/dataset_synapse.py` 中曾经保留过类似逻辑：

```python
label[label == 5] = 0
label[label == 9] = 0
label[label == 10] = 0
label[label == 12] = 0
label[label == 13] = 0
label[label == 11] = 5
```

这段逻辑的含义是：

```text
esophagus、IVC、portal vein、双侧肾上腺 → 背景 0
pancreas（原始 11）                → 训练类别 5
```

但是当前文件中这些行是注释状态，并没有执行。

`utils/preprocess_synapse_data.py` 当前也只是读取和转置标签，然后直接保存：

```python
seg_array = seg.get_fdata()
seg_array = np.transpose(seg_array, (2, 0, 1))
hf.create_dataset('label', data=seg_array)
```

从当前仓库代码本身看，不能仅凭 `test_synapse.py` 判断本地 H5/NPZ 文件到底是：

1. 原始 0–13 标签；还是
2. 已经在外部脚本中重映射成 0–8 标签。

这是必须实际检查数据文件唯一值才能确认的地方。

## 6. 因此，“原始代码到底有没有写反”的准确回答

### 情况 A：使用标准 9 类重映射标签

如果你的训练标签满足：

```text
label=5 表示 pancreas
label=6 表示 liver
```

那么：

```python
'pancreas', 'liver'
```

是正确顺序，原始代码没有写反；你当前的：

```python
'liver', 'pancreas'
```

是显示名称反了。

### 情况 B：你的数据被自定义映射成 liver=5、pancreas=6

如果你明确验证过数据文件满足：

```text
label=5 表示 liver
label=6 表示 pancreas
```

那么你当前的名称顺序才是正确的。但这不是标准 9 类映射的默认结论，必须有数据唯一值、生成脚本或标签映射表作为证据。

### 情况 C：数据仍然是原始 0–13 标签，但模型是 9 类输出

这是另一种配置错误，不是简单交换 `liver` 和 `pancreas` 能解决的。因为模型只有类别 `0..8`，而标签还可能出现 `9、10、11、12、13`。训练 loss 或指标计算会产生类别范围不匹配，或者把类别解释错。

## 7. 建议你马上做的检查

不要先根据可视化图片猜名称，直接检查训练和测试标签的唯一值。

### 检查 H5 文件中的唯一标签

```python
import h5py
import numpy as np

path = r'你的 test_vol_h5 文件\case0001.npy.h5'
with h5py.File(path, 'r') as f:
    labels = f['label'][:]

print(np.unique(labels, return_counts=True))
```

### 如何解释结果

```text
只出现 0..8：很可能已经做过 9 类重映射，此时通常使用 pancreas, liver
出现 0..13：仍是原始标签，不能只修改 class_names，必须先统一标签映射
出现 0..6 且没有 7、8：需要检查数据生成或类别裁剪是否有额外规则
```

但是，仅看唯一值还不能区分“5 是 liver 还是 pancreas”。最可靠的方法是回到生成该 H5/NPZ 的脚本，查看是否存在：

```python
label[label == 11] = 5
```

或其他明确映射语句。

## 8. 建议的修正方向

在尚未检查实际 label 映射前，不建议继续保留“上面原始代码，liver 和 pancreas 的位置反了”这句确定性注释，因为当前仓库证据更支持相反结论。

如果确认采用标准 9 类映射，建议使用：

```python
classes = [
    'spleen', 'right kidney', 'left kidney', 'gallbladder',
    'pancreas', 'liver', 'stomach', 'aorta'
]
```

如果确认采用自定义 `liver=5、pancreas=6` 映射，则保留你当前顺序，但应在数据生成代码旁写出明确的映射表，并在训练、验证、测试和可视化中统一使用。

## 9. 一句话记忆

> `class_names` 的顺序必须对应实际标签整数 `1..8`，不是对应器官中文常识或原始 Synapse 编号。标准 9 类重映射通常是第 5 类 pancreas、第 6 类 liver，所以原始 `pancreas, liver` 通常没有写反；但最终仍应以实际 H5/NPZ 的标签映射代码为准。

## 10. 本次仓库核对依据

- `test_synapse.py:120–130`：14 类名称列表、原始 9 类列表和当前修改。
- `test_synapse.py:153–156`：`classes` 作为 `class_names` 传入 `test_single_volume()`。
- `utils/utils.py:494–502`：`class_names` 被转成 `mask_labels`，用于可视化标签。
- `utils/utils.py:640–646`：指标按 `prediction == i`、`label == i` 计算，不读取器官名称。
- `utils/dataset_synapse.py:152–165`：曾保留但当前注释掉的 9 类标签重映射逻辑，其中原始 pancreas `11 → 5`。
- `utils/preprocess_synapse_data.py:67、83、101`：当前预处理读取、转置并保存标签，但没有在此处执行活动状态的重映射。
- Git 提交 `2e6cf51`：2026-09-03 对 `test_synapse.py` 的这次 `liver/pancreas` 顺序修改。

