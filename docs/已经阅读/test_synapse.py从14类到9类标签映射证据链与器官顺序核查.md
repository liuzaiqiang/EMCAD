# `test_synapse.py` 从 14 类到 9 类：标签映射证据链与器官顺序核查

## 先给结论

本项目中有两套必须区分的“顺序”：

1. **原始 Synapse 数据集的标准标签编号**，来自 `dataset.json`。
2. **当前工作区已经生成的 `train_npz`/`test_vol_h5` 中实际保存的标签编号**。

结论如下：

- 原始 Synapse 14 类定义是：`1=spleen`、`2=right kidney`、`3=left kidney`、`4=gallbladder`、`5=esophagus`、`6=liver`、`7=stomach`、`8=aorta`、`9=inferior vena cava`、`10=portal vein and splenic vein`、`11=pancreas`、`12=right adrenal gland`、`13=left adrenal gland`。
- 历史代码中把 14 类压缩为 9 类时，删除原始标签 `5、9、10、12、13`，把原始标签 `11`（pancreas）改成新标签 `5`。因此标准 9 类前景顺序是：`spleen, right kidney, left kidney, gallbladder, pancreas, liver, stomach, aorta`。**按这套标准映射，pancreas 和 liver 没有写反。**
- 但是，当前磁盘上的训练 NPZ 与原始 `labelsTr` 做同病例、同像素位置的逐体素核对后，得到的是另一套完整置换：

  ```text
  当前标签 1 = 原始标签 8  = aorta
  当前标签 2 = 原始标签 4  = gallbladder
  当前标签 3 = 原始标签 3  = left kidney
  当前标签 4 = 原始标签 2  = right kidney
  当前标签 5 = 原始标签 6  = liver
  当前标签 6 = 原始标签 11 = pancreas
  当前标签 7 = 原始标签 1  = spleen
  当前标签 8 = 原始标签 7  = stomach
  ```

- 因此，**当前 `test_synapse.py` 中的标准顺序列表与当前磁盘数据不匹配**。最直接的两个证据是：当前数据中 `label 5` 实际是 liver，`label 6` 实际是 pancreas；而代码把它们打印成 `label 5 pancreas`、`label 6 liver`。这会造成指标名称错配。
- 这不只是 pancreas/liver 两项互换。若继续使用当前数据，正确的 8 类名称列表应是：

  ```python
  ['aorta', 'gallbladder', 'left kidney', 'right kidney',
   'liver', 'pancreas', 'spleen', 'stomach']
  ```

下文按代码执行顺序给出证据。

## 1. 原始 14 类标签是什么

证据文件：

```text
D:\files\Medical_Image_Segmentation_Projects\unetr_plus_plus_synapse_dataset\DATASET_Synapse\DATASET_Synapse\unetr_pp_raw\unetr_pp_raw_data\Task002_Synapse\dataset.json
```

该文件的 `labels` 字段是原始整数标签到器官名称的权威映射：

| 原始整数标签 | 原始名称 |
|---:|---|
| 0 | background |
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

“14 类”包括背景 `0`，所以前景器官数是 13 个。模型若设置 `num_classes=14`，输出通道通常对应整数 `0..13`。

## 2. 历史代码如何从 14 类压缩成 9 类

历史版本的 `utils/dataset_synapse.py` 中曾出现如下逻辑（当前文件中的对应代码已被注释）：

```python
if self.nclass == 9:
    label[label == 5] = 0
    label[label == 9] = 0
    label[label == 10] = 0
    label[label == 12] = 0
    label[label == 13] = 0
    label[label == 11] = 5
```

逐行解释：

| 代码 | 含义 |
|---|---|
| `label[label == 5] = 0` | 删除 esophagus，改成背景 |
| `label[label == 9] = 0` | 删除 inferior vena cava，改成背景 |
| `label[label == 10] = 0` | 删除 portal vein and splenic vein，改成背景 |
| `label[label == 12] = 0` | 删除 right adrenal gland，改成背景 |
| `label[label == 13] = 0` | 删除 left adrenal gland，改成背景 |
| `label[label == 11] = 5` | 保留 pancreas，但把原始编号 11 重编码成新编号 5 |

没有被删除、也没有被重编号的原始标签 `1、2、3、4、6、7、8` 会保持原整数值。因此得到的新 9 类编码是：

| 新标签 | 来源原始标签 | 器官 |
|---:|---:|---|
| 0 | 0，以及 5、9、10、12、13 | background |
| 1 | 1 | spleen |
| 2 | 2 | right kidney |
| 3 | 3 | left kidney |
| 4 | 4 | gallbladder |
| 5 | 11 | pancreas |
| 6 | 6 | liver |
| 7 | 7 | stomach |
| 8 | 8 | aorta |

所以，**如果数据确实经过了这段映射，`classes` 的标准写法就是：**

```python
['spleen', 'right kidney', 'left kidney', 'gallbladder',
 'pancreas', 'liver', 'stomach', 'aorta']
```

特别注意：这段映射不是“把原始标签按 1、2、3、4、5、6、7、8 简单截断”。它是“删除五类，并将 pancreas 从 11 改为 5”。

## 3. 当前预处理脚本有没有执行这段映射

当前文件：

```text
utils/preprocess_synapse_data.py
```

该脚本对标签的关键操作是：

```python
seg_array = seg.get_fdata()
seg_array = np.transpose(seg_array, (2, 0, 1))
```

然后直接写入：

```python
hf.create_dataset('label', data=seg_array)       # 测试 H5
np.savez(..., image=ct_array_s, label=seg_array_s)  # 训练 NPZ
```

当前脚本没有执行 `label[label == 5] = 0` 等重编码语句。`utils/dataset_synapse.py` 中的 14→9 代码也全部处于注释状态。因此，必须检查已经生成的数据本身，不能只看 `classes` 列表推断标签含义。

当前工作区实际数据目录检查结果：

```text
data\Synapse\train_npz       2211 个 NPZ 切片
data\Synapse\test_vol_h5      12 个 H5 病例
```

## 4. 训练 NPZ 给出的决定性证据

### 4.1 核对方法

将当前训练数据与原始 Synapse `labelsTr` 按病例名配对：

1. 读取原始 `labelXXXX.nii.gz`。
2. 读取当前 `caseXXXX_sliceYYY.npz`。
3. 对原始标签执行与预处理脚本相同的轴变换 `np.transpose(..., (2, 0, 1))`。
4. 由于文件导出过程中存在 x 轴方向差异，同时对当前切片做对应的水平翻转。
5. 对每个像素比较“当前标签值”和“原始标签值”。
6. 对全部 18 个训练病例逐体素统计映射关系。

这不是通过器官大小、指标高低或肉眼猜测顺序，而是直接比较同一个病例、同一个像素位置的整数标签。

### 4.2 全部 18 个病例得到同一套置换

```text
当前标签 0 = 原始标签 0、5、9、10、12、13
当前标签 1 = 原始标签 8  = aorta
当前标签 2 = 原始标签 4  = gallbladder
当前标签 3 = 原始标签 3  = left kidney
当前标签 4 = 原始标签 2  = right kidney
当前标签 5 = 原始标签 6  = liver
当前标签 6 = 原始标签 11 = pancreas
当前标签 7 = 原始标签 1  = spleen
当前标签 8 = 原始标签 7  = stomach
```

“18 个病例全部通过”意味着这是一套稳定的数据编码方式，而不是单个病例异常或统计误差。于是，当前 `train_npz` 的真实名称顺序应为：

```python
['aorta', 'gallbladder', 'left kidney', 'right kidney',
 'liver', 'pancreas', 'spleen', 'stomach']
```

### 4.3 这份证据能证明什么

它能直接证明：

- 当前训练数据的 `label 5` 是 liver，不是 pancreas。
- 当前训练数据的 `label 6` 是 pancreas，不是 liver。
- 当前数据中的 `label 1..8` 整体都不是 `test_synapse.py` 当前 `classes` 列表所假定的顺序。
- 因而，若训练和测试使用的是同一批预处理产物，当前测试名称列表会对多个器官造成错名，而不仅是 liver/pancreas 两个名字。

## 5. `test_synapse.py` 怎样把标签变成指标名称

当前 9 类分支是：

```python
classes = ['spleen', 'right kidney', 'left kidney',
           'gallbladder', 'pancreas', 'liver', 'stomach', 'aorta']
```

测试调用链如下：

```text
test_synapse.py::inference
    -> test_single_volume(..., classes=args.num_classes, class_names=classes)
    -> utils/utils.py::test_single_volume
    -> prediction = argmax(softmax(model_output), axis=...)  # 得到整数标签图
    -> for i in range(1, classes):
           calculate_metric_percase(prediction == i, label == i)
```

`utils/utils.py` 中指标计算的本质是：

```python
prediction == i
label == i
```

也就是说，指标数值先按整数标签 `i` 计算；名称只在输出日志时通过 `classes[i - 1]` 查表：

```python
for i in range(1, args.num_classes):
    logging.info('Mean class (%d) %s ...' % (i, classes[i - 1], ...))
```

因此必须满足这个不变量：

```text
classes[0] 对应真实 label 1
classes[1] 对应真实 label 2
...
classes[7] 对应真实 label 8
```

如果名称列表错了：

- 模型输出张量不会改变；
- `prediction` 不会改变；
- `label` 不会改变；
- Dice、HD95、Jaccard、ASD 的原始数值不会改变；
- 但日志中每一行器官名称会被贴错，论文表格中的器官解释也会错。

## 6. 用户日志中的 liver/pancreas 如何重新解释

用户曾得到：

```text
Mean class (5) pancreas mean_dice 0.672110
Mean class (6) liver    mean_dice 0.951506
```

当前数据逐体素映射显示：

```text
label 5 = liver
label 6 = pancreas
```

所以这两行的正确器官解释应是：

```text
label 5 / liver    Dice 0.672110
label 6 / pancreas Dice 0.951506
```

数值没有因修改 `classes` 而交换；只是名称从错误标签改为正确标签。类似地，当前代码输出的 `Mean class (1) spleen` 实际应按当前数据解释为 aorta，`Mean class (7) stomach` 实际应解释为 spleen，等等。

## 7. 测试 H5 证据的强度与限制

当前 `test_vol_h5` 的标签只出现 `0..8`，这说明测试数据已经是 9 类编码。累计体素量中：

```text
label 5: 9,762,041
label 6:   504,539
```

这种体积关系符合 `label 5=liver`、`label 6=pancreas` 的解释，因为肝脏通常远大于胰腺。但这属于**强旁证**，不是训练 NPZ 那种直接的原始 NIfTI 逐体素对照：当前工作区没有找到对应测试 `labelsTs/TestSet` NIfTI 可供同样核对。

因此严谨表述是：

- 训练 NPZ：已有全部 18 个病例的直接逐体素证据。
- 测试 H5：与训练数据同目录、同一预处理体系生成，且标签范围和体积关系一致，因此高度推断采用同一置换；若要达到完全闭环，仍应拿原始测试标签做逐体素核对。

## 8. 两种修复路线，不要混用

### 路线 A：恢复标准 9 类编码

目标是让数据标签回到历史代码所定义的标准顺序：

```text
1 spleen, 2 right kidney, 3 left kidney, 4 gallbladder,
5 pancreas, 6 liver, 7 stomach, 8 aorta
```

做法是重新从原始 NIfTI 生成训练 NPZ 和测试 H5，并在保存前明确执行 14→9 映射。完成后继续使用：

```python
['spleen', 'right kidney', 'left kidney', 'gallbladder',
 'pancreas', 'liver', 'stomach', 'aorta']
```

必须删除或移走旧的预处理产物，避免训练/测试混用两套编码；同时记录新数据生成时间、源目录、映射代码和病例清单。

### 路线 B：保留当前数据编码

如果不重新生成数据，就必须让测试和统计名称匹配当前磁盘标签：

```python
classes = [
    'aorta', 'gallbladder', 'left kidney', 'right kidney',
    'liver', 'pancreas', 'spleen', 'stomach'
]
```

训练代码的 `num_classes=9` 仍然可以保持；改变的是名称解释，而不是模型通道数。建议在实验配置中额外保存一个显式字典，例如：

```python
label_to_organ = {
    0: 'background',
    1: 'aorta',
    2: 'gallbladder',
    3: 'left kidney',
    4: 'right kidney',
    5: 'liver',
    6: 'pancreas',
    7: 'spleen',
    8: 'stomach',
}
```

不要只交换 `pancreas` 和 `liver` 后就宣布全部修复，因为当前数据还存在其余六个位置的置换。

## 9. 建议的最终审计动作

在正式论文或最终实验前，建议保存一份机器可复核的审计输出，至少包括：

1. 原始 `dataset.json` 的完整 `labels` 字段。
2. 14→9 映射函数的代码版本或 Git 提交号。
3. 训练 NPZ 与原始 `labelsTr` 的逐体素混淆矩阵。
4. 测试 H5 与原始测试标签的逐体素混淆矩阵（若原始测试标签可取得）。
5. `label_id -> organ_name` 字典，并将其写入实验目录的 `config.json` 或 `log.txt`。
6. 重新计算后的每器官指标表，注明指标是按哪个标签编码解释的。

可以用下面的逻辑检查映射是否为一对一，而不是只检查 Dice 高低：

```python
for current_id in range(1, 9):
    # 统计当前标签与原始标签在同一空间位置的共现次数
    # 最大共现的原始标签就是 current_id 的候选真实器官
    pass
```

## 最终判断

如果问题是“原始 Synapse 的标准 14→9 代码是否把 pancreas 和 liver 写反”：**没有写反**，标准映射明确规定 pancreas 为新标签 5、liver 为新标签 6。

如果问题是“当前工作区训练/测试数据是否与 `test_synapse.py` 的标准 `classes` 列表一致”：**不一致**。当前训练 NPZ 的直接逐体素证据表明 `label 5=liver`、`label 6=pancreas`，并且 1..8 存在完整器官顺序置换。因此当前测试日志中的器官名称解释确实错了；修改名称列表会修正指标的器官归属，但不会改变已经计算出的数值。
