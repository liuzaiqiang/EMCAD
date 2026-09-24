# Synapse 训练期 `best_dice` 与测试 Dice 一致性排查

## 结论

本次结果没有矛盾，也不是加载错模型：

```text
训练日志：mean_dice = 0.845107，best_dice = 0.843304
测试日志：mean_dice = 0.845107
```

训练日志这一行中的 `best_dice=0.843304` 是**本轮评估开始前保存的历史最好值**；
`mean_dice=0.845107` 是**当前轮模型刚刚在评估集上得到的结果**。

由于 `0.845107 > 0.843304`，训练代码随后才会：

1. 把内存中的 `best_performance` 更新为 `0.845107`；
2. 把当前模型权重覆盖保存到 `best.pth`。

因此，随后用该 `best.pth` 执行测试，重新得到 `0.845107` 是正确且预期的结果。

正确对应关系是：

```text
训练阶段本轮 mean_dice 0.845107
              ==
best.pth 的测试 mean_dice 0.845107
```

而不是：

```text
训练日志中打印的旧 best_dice 0.843304
              ==
best.pth 的测试 mean_dice
```

## 代码依据

### 1. 训练期先打印，再更新历史最佳值

`trainer.py` 的 `inference()` 计算当前模型 `performance` 后，直接打印：

```python
logging.info(
    'Testing performance in val model: mean_dice : %f, best_dice : %f'
    % (performance, best_performance)
)
return performance
```

此时传入的 `best_performance` 仍是上一次历史最佳值，即 `0.843304`。

训练主循环在 `inference()` 返回之后才执行：

```python
performance = inference(args, model, best_performance)

if best_performance <= performance:
    best_performance = performance
    torch.save(model.state_dict(), save_mode_path)  # save_mode_path 是 best.pth
```

所以本例真实执行顺序为：

```text
进入本轮评估前：best_performance = 0.843304
当前权重评估结果：performance = 0.845107
日志先打印：mean_dice=0.845107, best_dice=0.843304
比较成立：0.843304 <= 0.845107
更新历史最好值：best_performance = 0.845107
保存当前权重：best.pth
```

### 2. 训练期与独立测试期的 Dice 计算一致

训练期的 `inference()` 和 `test_synapse.py` 都：

- 读取 `split="test_vol"`；
- 使用 `batch_size=1`；
- 传入相同形式的 `patch_size=[args.img_size, args.img_size]`；
- 对每个类别计算 Dice 后，按病例和类别取平均。

`val_single_volume()` 用 `calculate_dice_percase()` 计算 Dice；
`test_single_volume()` 用 `calculate_metric_percase()` 计算 Dice、HD95、Jaccard、ASD。
两者的 Dice 都调用 `medpy.metric.binary.dc`，且空预测/空标签的处理规则相同。因此，额外计算 HD95、Jaccard、ASD 不会改变 Dice。

## 为什么以前有时看起来严格对应

有两种常见情况：

1. 有些项目会在更新 `best_dice` 后才打印日志，因此同一行显示的是更新后的最佳值。
2. 若当前轮没有超过历史最佳值，日志会自然表现为：`mean_dice < best_dice`。此时最终测试的 `best.pth` 会对应更早的历史最佳轮次。

当前 EMCAD 代码属于“先记录当前分数与旧最佳值，再更新保存”的写法，因此不能把同一行的两个数当作同一个 checkpoint 的最终记录。

## 本次结果代表什么

`0.845107` 在训练期和重新测试时精确一致到六位小数，反而说明以下条件大概率一致：

- 加载的是本轮保存的 `best.pth`；
- 测试模型结构与保存权重一致；
- `img_size`、数据路径、类别数和 `test_vol` 列表一致；
- 两处的推理与 Dice 聚合逻辑一致。

## `best.pth` 是否存在保存逻辑 Bug

没有。当前 `trainer.py` 的主循环已经正确处理“最后一轮恰好成为历史最高 Dice”的情形：

```python
performance = inference(args, model, best_performance)

if best_performance <= performance:
    best_performance = performance
    save_mode_path = os.path.join(snapshot_path, 'best.pth')
    torch.save(model.state_dict(), save_mode_path)
```

它在每一个 epoch 末尾执行，包括最后一个 epoch。因此，只要最后一轮 `performance` 大于或等于历史最高值，当前轮的模型参数就会覆盖写入 `best.pth`。

本例中实际执行为：

```text
旧 best = 0.843304
最后一轮当前 performance = 0.845107
判断 0.843304 <= 0.845107 为真
best.pth 被当前最后一轮权重覆盖
```

需要澄清的是：每个 epoch 结束时，代码也会先保存一次 `last.pth`。这是“最新一轮权重”，与 `best.pth` 是两个不同文件：

```text
last.pth : 每轮都覆盖，始终是最后训练轮的权重。
best.pth : 仅在当前指标达到历史最高时覆盖，是验证指标最优的权重。
```

## 建议的最小日志修改（可选）

权重保存逻辑不需要改。若希望日志不再引起误解，只替换 `trainer.py` 中从

```python
performance = inference(args, model, best_performance)
```

开始，到原有 `logging.info("save model to ...")` 结束的整段代码。

将其替换为：

```python
# 当前 epoch 在评估集上的 Dice。
# 注意：inference() 内部目前会打印“旧的 best_performance”，该输出仅作过程记录。
performance = inference(args, model, best_performance)

# 先保存比较结果，避免 best_performance 更新后无法判断本轮是否写入了 best.pth。
# 使用 >= 保持与原代码的 <= 等价：当前 Dice 与历史最好值相等时也覆盖 best.pth。
is_best = performance >= best_performance

if is_best:
    # 更新为包含当前轮在内的历史最佳 Dice。
    best_performance = performance

    # 保存的正是当前 epoch 对应的模型全部参数和 BatchNorm 缓冲区。
    save_mode_path = os.path.join(snapshot_path, 'best.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))

# 这一行在更新之后输出，因此 current_dice 与 best_dice_after_update 的语义明确。
logging.info(
    "validation summary | current_dice: %.6f | best_dice_after_update: %.6f | best_pth_updated: %s",
    performance,
    best_performance,
    is_best,
)
```

替换位置：`trainer.py` 的训练 epoch 循环内，紧接 `last.pth` 保存之后的评估和 `best.pth` 保存区域。当前工作区约为第 270 至 287 行。

### 是否改为严格大于

若你希望 Dice 相等时保留首次达到最高分的 checkpoint，而不是用后一次相同分数覆盖它，将：

```python
is_best = performance >= best_performance
```

改成：

```python
is_best = performance > best_performance
```

这只影响 Dice 完全相等时保存哪一个 checkpoint，不影响本例 `0.845107 > 0.843304` 的保存结果。

## 论文实验中的重要提醒

当前训练代码每个 epoch 都在 `split="test_vol"` 上评估，并据此选择 `best.pth`。因此日志中的 “val model” 实际上是代码命名，不自动意味着存在独立验证集。

若 `test_vol.txt` 被你作为论文最终测试集使用，那么它已经参与了 checkpoint 选择，不能再称为完全独立的最终测试集。正式论文应固定训练集、验证集和最终测试集的职责；至少需要明确说明你沿用了原项目的评估协议，或另划分验证集用于选择 checkpoint。
