# `test_synapse.py` 类别名称改动对测试输出和指标的影响

## 结论先说

你之前把 9 类 Synapse 配置中的：

```python
# 原来处于激活状态
['spleen', 'right kidney', 'left kidney',
 'gallbladder', 'liver', 'pancreas', 'stomach', 'aorta']

# 当前工作区的激活状态
['spleen', 'right kidney', 'left kidney',
 'gallbladder', 'pancreas', 'liver', 'stomach', 'aorta']
```

改动的直接效果是：第 5、6 类在日志和叠加可视化中的器官名称互换。它不会改变模型前向预测，也不会改变 Dice、HD95、Jaccard、ASD 的数值，更不会改变最后一行总平均指标。

这里讨论的是 `test_synapse.py` 中 `classes` 这个“名称列表”的改动，不是修改 `num_classes=9`、标签数组或网络输出通道。

## 代码链路

### 1. 名称列表传入的位置

`test_synapse.py` 的测试调用大致是：

```python
metric_i = test_single_volume(
    image, label, model,
    classes=args.num_classes,
    patch_size=[args.img_size, args.img_size],
    class_names=classes,
)
```

注意这里有两个不同的参数：

- `classes=args.num_classes`：仍然是整数 `9`，决定输出有 9 个标签通道、指标循环计算标签 `1..8`。
- `class_names=classes`：才是 `['spleen', ..., 'pancreas', 'liver', ...]` 这个文字列表。

所以你改的是名称列表，并没有改指标循环的类别编号。

### 2. 指标如何计算

`utils/utils.py` 中的测试指标循环相当于：

```python
for i in range(1, classes):
    metric_list.append(
        calculate_metric_percase(prediction == i, label == i)
    )
```

指标依据的是整数标签 `i`：

- 第 1 行指标对应标签 1；
- 第 5 行指标对应标签 5；
- 第 6 行指标对应标签 6；
- 依此类推。

这里没有读取 `class_names[i - 1]`，因此名称顺序不会进入 Dice、HD95、Jaccard、ASD 的计算。

### 3. 最终总指标如何计算

测试脚本先把所有病例的 `metric_i` 累加并除以病例数，再执行：

```python
performance = np.mean(metric_list, axis=0)[0]
mean_hd95 = np.mean(metric_list, axis=0)[1]
mean_jacard = np.mean(metric_list, axis=0)[2]
mean_asd = np.mean(metric_list, axis=0)[3]
```

这个过程同样只处理数值矩阵，不读取器官名称。因此最终的总均值不变。

## 你的日志应该怎样解释

你贴出的日志中，第 5、6 类是：

```text
Mean class (5) liver    mean_dice 0.951506 ...
Mean class (6) pancreas mean_dice 0.672110 ...
```

这说明该次运行使用的是旧的名称顺序，或者日志是在你修改代码之前生成的。

如果按照当前工作区的名称顺序重新解释同一批数值，则应读成：

```text
Mean class (5) pancreas mean_dice 0.951506 mean_hd95 17.679956,
                 mean_jacard 0.908009 mean_asd 2.384199

Mean class (6) liver    mean_dice 0.672110 mean_hd95 9.133429,
                 mean_jacard 0.516374 mean_asd 2.086993
```

这里是“标签编号对应的数值不动，只重新贴正确名称”。不能把第 5 类和第 6 类的数值重新平均或丢弃。

你日志中的最终均值可以直接由 8 个类别数值验证：

```text
mean_dice   = 0.838059
mean_hd95   = 16.981820
mean_jacard = 0.747131
mean_asd    = 2.865512
```

例如 8 个 `mean_dice` 的平均值是 `0.83805925`，按日志的 6 位小数显示为 `0.838059`。交换第 5、6 类的名字不会改变这 8 个数字的集合，所以这些最终结果保持不变。

## 逐项说明：哪些会变，哪些不会变

| 输出或行为 | 是否受名称顺序改动影响 | 说明 |
|---|---:|---|
| `prediction` 预测标签图 | 否 | 网络仍按 `argmax` 产生同样的整数类别图 |
| 每个病例的 `idx ... mean_dice ...` | 否 | 这一行只打印病例平均数，没有器官名称 |
| `Mean class (5)/(6)` 的数值 | 否 | 仍然是标签 5、标签 6 的指标 |
| `Mean class (5)/(6)` 后面的器官名 | 是 | `classes[4]` 与 `classes[5]` 互换 |
| `Testing performance in best val model` | 否 | 对 `metric_list` 数值矩阵求平均 |
| Dice、HD95、Jaccard、ASD | 否 | 计算使用 `prediction == i` 和 `label == i` |
| PNG 叠加图的器官图例/标签 | 是 | 图像掩膜不变，但文字名称随列表改变 |
| 保存的 `_pred.nii.gz`、`_gt.nii.gz`、`_img.nii.gz` | 否 | 保存的是数组和体数据，不保存这份 Python 名称列表 |
| 模型结构、checkpoint 加载、训练权重 | 否 | 本次改动位于测试脚本的名称列表 |

## 为什么这次改动仍然重要

虽然数值不变，但名称错位会导致科研结论错位。例如：

- 把标签 5 的指标写成 liver，会把 pancreas 的性能误报为 liver；
- 把标签 6 的指标写成 pancreas，会把 liver 的性能误报为 pancreas；
- 论文表格、实验记录、失败病例分析和可视化解读都会因此出错。

所以这不是“指标数值错误”，而是“指标归属的器官名称错误”。修正后，建议重新运行一次或至少在实验记录中明确标注：旧日志的第 5、6 类名称需要交换解释。

## 仍需注意的验证边界

当前仓库的测试预处理脚本会把原始 Synapse 标签直接写入 H5；`utils/dataset_synapse.py` 中关于标签重映射的代码目前是注释状态。因此，“第 5 类究竟是 pancreas 还是 liver”最终应以实际生成 H5 时采用的标签映射为准。

标准 Synapse 9 类重映射通常是：原始标签 11（pancreas）映射为连续类别 5，原始标签 6（liver）保留为类别 6，因此常见顺序是：

```text
类别 5 = pancreas
类别 6 = liver
```

若要对当前数据做最终确认，应读取一个实际测试 H5 的 `label` 唯一值，并检查生成该 H5 的脚本是否执行过类似 `label[label == 11] = 5` 的映射。名称列表修正本身不会替换 H5 中的标签。

## 一句话记忆

`classes` 改名只改变“这行数字叫什么”，不改变“数字怎么算”；你贴出的最终四项总指标仍然有效，但第 5、6 类的器官名需要按实际标签映射重新核对。
