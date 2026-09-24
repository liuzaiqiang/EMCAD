# `test_synapse.py` 中 `liver` 与 `pancreas` 改动对测试指标的实际影响

## 1. 直接回答

你之前把：

```python
'pancreas', 'liver'
```

改成了：

```python
'liver', 'pancreas'
```

**这次改动没有改变模型预测、没有改变标签、没有改变 Dice/HD95/Jaccard/ASD 的计算，也没有改变最终总均值。**

它改变的是：

1. 日志中 `Mean class (5)` 和 `Mean class (6)` 后面显示的器官名称；
2. 保存的预测/真值叠加图中的器官图例名称；
3. 任何人根据日志名称手工整理“liver 指标”和“pancreas 指标”时的语义解释。

换句话说：**数值没有被交换，名称被交换了。**

## 2. 你的日志中哪些内容不会改变

例如：

```text
idx 0 case case0008 mean_dice 0.697247 mean_hd95 11.297472, mean_jacard 0.602939 mean_asd 2.758877
```

这类 `idx ... case ... mean_*` 是对当前病例中 8 个整数类别的指标求平均：

```python
np.mean(metric_i, axis=0)
```

这里不使用 `classes` 中的字符串名称。因此，无论写：

```python
['pancreas', 'liver']
```

还是：

```python
['liver', 'pancreas']
```

这类病例均值都完全相同。

你日志中的以下部分也不会改变：

```text
idx 0 ... mean_dice 0.697247 ...
idx 1 ... mean_dice 0.905954 ...
...
idx 11 ... mean_dice 0.875652 ...
```

此外，推理速度、每个病例耗时、预测 mask、模型权重和 checkpoint 加载过程也不会因这次名称调整而改变。

## 3. 你的日志中哪些内容会改变

你当前的日志是：

```text
Mean class (5) liver   mean_dice 0.951506 mean_hd95 17.679956 mean_jacard 0.908009 mean_asd 2.384199
Mean class (6) pancreas mean_dice 0.672110 mean_hd95  9.133429 mean_jacard 0.516374 mean_asd 2.086993
```

如果恢复原始名称顺序，则同样的两行数值会被标记为：

```text
Mean class (5) pancreas mean_dice 0.951506 mean_hd95 17.679956 mean_jacard 0.908009 mean_asd 2.384199
Mean class (6) liver   mean_dice 0.672110 mean_hd95  9.133429 mean_jacard 0.516374 mean_asd 2.086993
```

请注意：这里是**名称跟着类别位置变化**，不是把数值重新计算或交换。

代码的对应关系是：

```python
for i in range(1, args.num_classes):
    logging.info(
        'Mean class (%d) %s ...',
        i,
        classes[i - 1],
        metric_list[i - 1][0],
        ...,
    )
```

所以：

```text
metric_list[4] → 整数类别 5 的真实指标
classes[4]     → 这一行显示的器官名字
metric_list[5] → 整数类别 6 的真实指标
classes[5]     → 这一行显示的器官名字
```

修改 `classes[4]` 和 `classes[5]` 只会改第二列名称，不会改 `metric_list[4]` 和 `metric_list[5]`。

## 4. 最终总均值为什么也不会改变

你的最后一行是：

```text
Testing performance in best val model: mean_dice : 0.838059 mean_hd95 : 16.981820, mean_jacard : 0.747131 mean_asd : 2.865512
```

代码计算方式是：

```python
performance = np.mean(metric_list, axis=0)[0]
mean_hd95 = np.mean(metric_list, axis=0)[1]
mean_jacard = np.mean(metric_list, axis=0)[2]
mean_asd = np.mean(metric_list, axis=0)[3]
```

这里对 8 个整数类别的指标行求平均，不读取 `classes` 字符串。

即使只考虑类别 5 和 6，它们的平均值满足：

```text
(0.951506 + 0.672110) / 2 = 0.811808
```

交换这两个类别的名称后，集合中的数值仍然是同样的两个数，平均值自然不变。HD95、Jaccard、ASD 也是同理。

因此，你这次修改不会解释 `0.838059` 与另一次测试之间的差异，也不会导致整个测试集重新得到一套不同的总指标。

## 5. 这次改动真正可能造成的错误

### 5.1 逐器官表格名称错位

如果你把日志直接抄到论文表格：

```text
liver    Dice=0.951506
pancreas Dice=0.672110
```

那么这组结果是否正确，取决于整数类别 5、6 在你的数据中的真实含义。如果实际是标准 9 类映射（类别 5=pancreas、类别 6=liver），那么这样抄表会把两列器官名称写反。

### 5.2 可视化图例错位

`class_names` 会传到 `overlay_masks()`，用于叠加图的标签名称。若名称顺序错，图片上的颜色/图例可能把类别 5 标为肝脏、类别 6 标为胰腺，导致人工观察时误判模型到底预测了哪个器官。

### 5.3 后续人工分析方向错误

如果你看到“liver Dice 0.95、pancreas Dice 0.67”后，进一步决定创新点、损失函数或边界优化策略，那么一旦名称和整数标签不一致，后续研究判断也会跟着错位。

## 6. 你的这份数值能不能反过来证明名称顺序是对的

不能仅凭这份日志下定论。

从医学任务的常见经验看，肝脏通常比胰腺大，肝脏 Dice 较高、胰腺 Dice 较低是合理现象。因此你看到：

```text
0.951506 和 0.672110
```

很容易觉得当前的 `liver, pancreas` 顺序“看起来正确”。但这只能说明数值形态符合一种可能性，不能证明整数标签 5 确实是肝脏。

真正决定名称的，是生成训练/测试标签时的映射：

```text
训练标签整数 5 到底代表什么？
训练标签整数 6 到底代表什么？
```

如果采用常见 Synapse 9 类重映射：

```text
原始 pancreas=11 → 训练类别 5
原始 liver=6    → 训练类别 6
```

那么正确名称仍然是：

```python
'pancreas', 'liver'
```

如果你的数据生成脚本明确采用了另一套映射：

```text
训练类别 5 = liver
训练类别 6 = pancreas
```

那么当前名称顺序才是正确的。

## 7. 需要重新测试吗

### 只关心数值指标

如果你确认这次只改了 `class_names` 列表，没有改模型、标签、推理和指标代码，那么**不需要重新跑 80 分钟左右的完整测试**。已有日志中的数值仍然有效；你只需要根据正确的标签映射重新解释第 5、6 类名称。

### 需要正确的可视化图例

如果已经保存了带错误图例的 PNG/NIfTI 旁路说明，建议用正确的 `classes` 顺序重新生成可视化，或者在整理图片时明确标注“类别 5/6 名称需按映射修正”。NIfTI 中的整数预测 mask 本身不会因为字符串修改而改变。

### 需要论文逐器官结果

不建议直接使用当前日志中的器官名称。先确认 label 映射，再决定是否只交换报告表中的第 5、6 列名称，还是重新生成日志。

## 8. 推荐的核对方法

优先检查生成 H5/NPZ 的代码，而不是只看 `test_synapse.py`：

1. 查找是否存在 `label[11] = 5`、`label[6] = 5`、`label[11] = 6` 等重映射语句。
2. 确认训练和测试使用的是同一套标签映射。
3. 确认模型输出通道 5、6 与 loss 中的标签整数一致。
4. 再决定 `classes[4]` 和 `classes[5]` 的名称。

当前仓库中曾保留过：

```python
label[label == 11] = 5
```

这支持“标准 9 类映射中类别 5 是 pancreas”的解释；但这些语句当前处于注释状态，而且当前预处理脚本直接保存 `seg_array`，所以本地实际 H5/NPZ 是否已经在别处完成映射，仍需要检查数据文件或实际生成脚本。

## 9. 最简短的结论

对于你贴出的日志：

```text
0.951506、0.672110、16.981820、0.838059 等数值没有因为改名而改变。
```

改变的是：

```text
“类别 5 的这一行叫 liver 还是 pancreas”
“类别 6 的这一行叫 pancreas 还是 liver”
```

所以你之前的改动不是“改变测试指标”，而是“改变指标对应的器官名称解释”。

## 10. 本次仓库核对依据

- `test_synapse.py:153–156`：`classes` 作为 `class_names` 传入 `test_single_volume()`。
- `test_synapse.py:163–190`：病例均值由 `metric_i` 直接求平均，未使用器官名称。
- `test_synapse.py:176–190`：逐类别日志使用 `classes[i-1]` 显示名称，但数值来自 `metric_list[i-1]`。
- `test_synapse.py:183–190`：最终总均值直接对 `metric_list` 求平均，未使用 `classes`。
- `utils/utils.py:494–502`：`class_names` 转为 `mask_labels`，用于叠加图标签。
- `utils/utils.py:640–646`：指标按 `prediction == i`、`label == i` 计算。
- `utils/dataset_synapse.py:152–165`：曾保留的标签重映射示例包含原始 pancreas `11 → 5`，但当前为注释代码。

