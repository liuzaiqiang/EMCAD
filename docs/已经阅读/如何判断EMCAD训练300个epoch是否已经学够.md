# 如何判断 EMCAD 训练 300 个 epoch 是否已经学够

## 先记住一句话

`max_epochs=300` 只表示“最多允许训练 300 轮”，不表示模型在第 300 轮一定学够，也不表示第 300 轮一定是最好模型。

判断模型是否学够，核心不是看 epoch 数字，而是同时观察：

```text
训练集损失是否还在下降
验证集 Dice 是否还在稳定上升
验证集是否已经平台期
训练集与验证集之间是否出现明显差距
各个器官的指标是否仍在改善
```

对当前 EMCAD 项目而言，最重要的判断指标是**独立验证集上的 mean Dice**，而不是训练 loss，也不是“已经跑到 300”这个事实。

## 1. epoch 到底是什么

一个 epoch 表示：训练集中的每个样本通常都被模型看过一次。

当前 `trainer.py` 的主循环是：

```python
for epoch_num in range(max_epoch):
    for i_batch, sampled_batch in enumerate(trainloader):
        ...
        loss.backward()
        optimizer.step()
```

因此：

- `max_epochs=300` 时，外层循环最多执行 300 轮；
- 每轮包含 `len(trainloader)` 个 batch 更新；
- 总参数更新次数约为：

  ```text
  300 × 每轮 batch 数
  ```

- 当前代码真正控制循环的是 `args.max_epochs`；`max_iterations` 只用于记录最大迭代数和目录命名，不负责提前停止。

如果训练集有 1800 张切片、有效 batch size 为 6，则每个 epoch 大约有 300 次参数更新，300 epoch 约有 90,000 次更新。epoch 数必须结合数据量、batch size、学习率、增强方式和任务难度理解。

## 2. “学够”与“训练完成”不是一回事

### 2.1 训练完成

训练完成只意味着：

```text
程序走完了 max_epochs 指定的循环
```

例如 300 epoch 跑完，说明程序完成了 300 轮训练，但模型可能：

- 仍然处于欠拟合状态；
- 已经在 100 轮左右达到最佳，后面发生过拟合；
- 150 轮后基本不变，后面的训练只是浪费时间；
- 验证指标上下波动，尚未稳定；
- 训练数据或标签存在问题，继续增加 epoch 也不会解决。

### 2.2 模型学够

通常需要满足下面的组合条件：

1. 验证集 mean Dice 在连续多个 epoch 内没有实质性提升；
2. 验证集曲线已经进入平台期，而不是刚开始上升；
3. 训练 loss 继续下降时，验证 Dice 不再上升，或者开始下降；
4. 各主要器官的 Dice 没有持续改善，HD95 等几何指标也没有明显改善；
5. 最优 epoch 在不同随机种子或重复实验中大致稳定，而不是偶然的单点峰值。

## 3. 当前 EMCAD 代码实际上记录了什么

### 3.1 每轮评估的是 mean Dice

当前 `trainer.py` 每轮训练结束后调用：

```python
performance = inference(args, model, best_performance)
```

`inference()` 对 `test_vol` 列表中的病例进行推理，计算每个前景器官 Dice，然后再求宏平均：

```text
每个病例、每个器官 Dice
    -> 按病例平均
    -> 按器官再平均
    -> mean Dice
```

这个 `performance` 才是当前代码选择 `best.pth` 的主要依据：

```python
if best_performance <= performance:
    best_performance = performance
    torch.save(model.state_dict(), snapshot_path + '/best.pth')
```

所以最终测试时，通常应优先使用 `best.pth`，而不是机械使用最后一轮的 `last.pth` 或 `epoch_299.pth`。

### 3.2 当前日志中的 epoch loss 不是平均 loss

当前代码在一个 epoch 结束时写入：

```python
logging.info('iteration %d, epoch %d : loss : %f, lr: %f' %
             (iter_num, epoch_num, loss.item(), lr_))
```

但 `loss` 在 batch 循环中不断被覆盖；循环结束时，它代表的是**最后一个 batch 的 loss**，不是整轮所有 batch 的平均值。

因此不能根据如下现象直接下结论：

```text
最后一个 batch loss 变小了，所以模型已经学够
```

更可靠的方法是记录每轮平均训练 loss：

```text
epoch_loss = 所有 batch loss 之和 / batch 数
```

如果暂时不改代码，也可以用 TensorBoard 的逐 step `info/total_loss` 曲线观察总体趋势，但要注意当前 `mutation` 模式会把多个监督组合的损失相加，loss 的绝对数值不能直接与 `last_layer` 或 `deep_supervision` 实验横向比较。

### 3.3 当前代码每轮评估的 `test_vol` 必须确认身份

当前函数名叫 `inference`，但读取的是：

```python
Synapse_dataset(..., split="test_vol", ...)
```

如果 `test_vol.txt` 实际上是验证集，那么可以把它称为验证 Dice；如果它是官方测试集，那么每轮都用它选择 `best.pth` 会造成测试集参与模型选择，最终测试结果会偏乐观。

正确的科研流程应是：

```text
训练集：更新参数
验证集：每轮监控、选择 best.pth、决定是否早停
测试集：训练全部结束后只评估一次
```

在判断“300 epoch 是否学够”之前，先确认 `test_vol.txt` 的病例是否与训练病例隔离、是否真正承担验证集角色。

## 4. 判断标准一：验证 Dice 是否进入平台期

这是最重要的标准。

假设每轮得到如下 mean Dice：

```text
epoch 220: 0.8341
epoch 230: 0.8360
epoch 240: 0.8372
epoch 250: 0.8374
epoch 260: 0.8375
epoch 270: 0.8374
epoch 280: 0.8376
epoch 290: 0.8375
epoch 300: 0.8374
```

虽然最后跑到了 300，但 250 epoch 左右已经进入平台期。第 280 轮的微小变化可能只是病例组成、边界误差或随机因素造成的波动，不足以证明模型又学到了实质性新知识。

可以采用一个明确的“改善阈值”：

```text
新 Dice > 历史最好 Dice + 0.001
```

才认为有实质改善；也可以根据任务噪声使用 0.0005 或 0.002，但必须在实验前固定规则。

同时设置 patience，例如：

```text
连续 30～50 个 epoch 没有超过历史最好 Dice + 0.001
    -> 认为进入平台期
```

这叫 early stopping（早停）。当前项目没有实现自动早停，但可以通过日志离线判断。

## 5. 判断标准二：训练 loss 与验证 Dice 的组合关系

不要单独看一条曲线，应看二者组合。

### 情况 A：训练 loss 下降，验证 Dice 上升

```text
train loss: 下降
val Dice:   上升
```

说明模型仍在有效学习，通常还没有学够，可以继续训练。

如果 Dice 的提升已经非常小，则说明接近平台期，而不是一定要继续到 300。

### 情况 B：训练 loss 下降，验证 Dice 横盘

```text
train loss: 继续下降
val Dice:   基本不变
```

这是“可能已经学够，或者开始过拟合”的典型表现。模型在训练切片上越来越会做，但泛化能力没有继续改善。此时应优先保存并使用历史最佳 `best.pth`，而不是默认最后一轮。

### 情况 C：训练 loss 下降，验证 Dice 下降

```text
train loss: 下降
val Dice:   下降
```

通常是过拟合或训练/验证分布不一致。增加 epoch 往往会更糟，应检查：

- 是否应该减小训练轮数；
- 数据增强是否不足；
- 学习率是否过大或长期不衰减；
- 训练和验证病例是否存在预处理差异；
- 标签编码是否一致；
- 验证集是否被错误地当成测试集使用。

### 情况 D：训练 loss 和验证 Dice 都没有改善

可能是欠拟合，也可能是实现问题。应检查：

- 学习率是否过小；
- 输入、标签、类别数是否匹配；
- 模型是否真的在更新参数；
- loss 是否含有错误的多次累加或尺度异常；
- 标签是否全背景或类别顺序错位；
- 是否误用了冻结参数或错误 checkpoint。

### 情况 E：验证 Dice 大幅上下波动

不要根据单个峰值判断学够。应查看：

- 验证病例是否太少；
- 每个器官的 Dice 是否被少数病例主导；
- HD95 是否存在极端值；
- 是否使用了随机验证增强；
- BatchNorm、评估模式或数据预处理是否不稳定。

## 6. 判断标准三：训练集与验证集的差距

如果可以计算训练集 Dice，应比较：

```text
训练 Dice - 验证 Dice
```

示例：

| 现象 | 训练 Dice | 验证 Dice | 解释 |
|---|---:|---:|---|
| 两者都低且接近 | 0.70 | 0.67 | 可能欠拟合或实现有问题 |
| 两者都较高且接近 | 0.88 | 0.86 | 泛化较好，接近学够 |
| 训练很高、验证明显低 | 0.97 | 0.78 | 过拟合或训练/验证分布差异 |
| 验证突然高后下降 | 0.95 | 0.84→0.80 | 可能在峰值后过拟合 |

医学图像分割中，不应只看训练集 Dice。模型把训练切片记住，并不代表能正确分割新的患者。

## 7. 判断标准四：看每个器官，而不是只看总 mean Dice

总 mean Dice 可能掩盖某个器官持续变差。

例如：

```text
spleen      0.93 -> 0.94
kidney      0.88 -> 0.89
liver       0.95 -> 0.95
pancreas    0.60 -> 0.62
gallbladder 0.70 -> 0.65
```

总平均可能仍然略有上升，但 gallbladder 已经退化。应同时记录：

- 每个器官 Dice；
- HD95；
- Jaccard；
- ASD；
- 每个病例的指标分布。

尤其是 Synapse 这类多器官任务，小器官如 pancreas、gallbladder 的指标波动通常比 liver、spleen 大。总 Dice 平台期后，某个困难器官仍可能有实际改善，因此不能只凭一个总平均数机械早停。

## 8. 300 epoch 的实用判定流程

建议按以下流程判断，不要凭感觉。

### 第一步：确认实验记录完整

至少保存：

```text
epoch
平均 train loss
验证 mean Dice
每器官 Dice
HD95/Jaccard/ASD
learning rate
best epoch
best Dice
```

同时保存 `config.json` 或日志中的：

```text
数据划分、随机种子、batch size、学习率、num_classes、img_size、supervision、模型配置
```

### 第二步：画三条核心曲线

至少画：

1. epoch–平均 train loss；
2. epoch–验证 mean Dice；
3. epoch–每器官 Dice。

如果使用 TensorBoard，可观察当前代码写入的：

```text
info/total_loss
info/lr
```

但要注意 `info/total_loss` 是每个 step 写入的 mutation 总损失，不是每轮平均损失。

### 第三步：找历史最佳 epoch

令：

```text
best_epoch = 验证 mean Dice 达到最大值的 epoch
```

例如：

```text
best_epoch = 236
best Dice  = 0.8427
epoch 300 Dice = 0.8398
```

这说明 300 轮并没有优于 236 轮。最终应使用 `best.pth`，而不是 `epoch_299.pth`。

### 第四步：检查最佳点之后是否长期没有改善

例如设置：

```text
patience = 40
min_delta = 0.001
```

若从 epoch 236 到 276 都没有超过 `0.8427 + 0.001`，可认为 236 左右已经基本学够。若 290 又明显提升到 0.8450，则说明之前只是平台期或波动，不能过早停止。

### 第五步：检查是否存在过拟合

如果最佳 Dice 后训练 loss 继续下降，但验证 Dice 连续下降，说明继续训练没有价值；若最佳 epoch 接近 300 且 Dice 仍在稳定上升，则 300 可能不够，应增加到 400 或 500 做受控实验，而不是直接断言“论文用了 300，所以 300 足够”。

## 9. 四种典型曲线与结论

### 曲线 1：300 轮仍在上升

```text
val Dice: 0.78 -> 0.81 -> 0.83 -> 0.84 -> 0.845（第300轮仍上升）
```

结论：300 可能没有学够。可以增加训练上限，例如 400/500，并保持其他配置不变，验证是否继续提升。

### 曲线 2：100 轮后平台，300 轮无改善

```text
val Dice: 0.80 -> 0.83 -> 0.84（约100轮后长期稳定）
```

结论：模型大概率在 100 轮左右已经学够；继续到 300 主要是重复训练。可以考虑早停或缩短默认训练时间。

### 曲线 3：180 轮达到峰值，之后下降

```text
val Dice: 0.80 -> 0.84（epoch 180）-> 0.82（epoch 300）
```

结论：发生过拟合或优化后期退化。使用 epoch 180 附近保存的 `best.pth`，不要使用最后一轮权重。

### 曲线 4：从头到尾都低且平

```text
val Dice: 0.45 -> 0.46 -> 0.46
```

结论：不能简单说“300 还没学够”。先排查标签、类别数、输入归一化、学习率、checkpoint、数据划分和代码执行路径。增加 epoch 可能只是把错误训练更久。

## 10. 当前 EMCAD 配置下的特殊注意事项

### 10.1 当前学习率是常数

当前代码将：

```python
lr_ = base_lr
```

并在每个 step 写回优化器，因此默认是恒定学习率，而不是随 epoch 自动衰减的学习率。

这意味着：

- 后期验证 Dice 平台，不一定只代表模型容量不足；
- 可能需要学习率衰减才能在平台期附近继续优化；
- 如果学习率过大，后期可能在最优点附近震荡；
- 如果学习率过小，300 epoch 可能仍未充分学习。

因此“增加 epoch”与“调整学习率策略”是两个不同实验，不能混为一谈。

### 10.2 mutation 模式的 loss 绝对值不能直接横比

当前 `mutation` 会对多个输出组合分别计算 CE+Dice，然后累加。于是 mutation 的 loss 数值天然可能比 `last_layer` 大很多。

所以判断是否学够时，应重点看：

```text
同一个 supervision 配置内部的 loss 趋势
验证 mean Dice
每器官指标
```

不要因为 mutation 的 loss 比另一实验大，就断言 mutation 没学好。

### 10.3 训练后要确认使用的是 best 而不是 last

当前代码分别保存：

```text
last.pth       每轮覆盖，代表最后一轮
best.pth       验证 mean Dice 达到历史最好时保存
epoch_*.pth    阶段性或最终轮快照
```

三者含义不同。判断 300 是否学够时，应比较 `best epoch` 与 `epoch_299` 的验证指标，而不是只检查文件是否存在。

### 10.4 评估模式必须稳定

验证时应使用：

```python
model.eval()
with torch.no_grad():
    ...
```

训练下一轮前应恢复：

```python
model.train()
```

当前代码已经在每轮开头调用 `model.train()`，这是判断训练曲线可信度的重要条件。若未来改动代码，不能让模型在验证后一直停留在 eval 模式。

## 11. 一个可以直接采用的判断规则

对于当前 300 epoch 实验，可以先采用以下固定规则：

```text
主指标：验证集 mean Dice
min_delta：0.001
patience：40 个 epoch
最佳模型：验证 mean Dice 最高的 best.pth
辅助指标：每器官 Dice、HD95、Jaccard、ASD
```

解释为：

1. 每个 epoch 结束后计算验证 mean Dice；
2. 若新 Dice 比历史最好高至少 0.001，则更新 best；
3. 若连续 40 个 epoch 没有达到这个改善幅度，认为进入平台期；
4. 若训练 loss 继续下降而验证 Dice 下降，判定有过拟合倾向；
5. 若第 300 轮仍稳定上升，不认为已经学够，下一次实验把上限提高，同时保持其他配置不变；
6. 若第 100～200 轮后稳定平台，则 300 已经足够，甚至可以缩短训练。

这不是医学分割的永恒标准，而是一套可复现的实验判据。真正重要的是：同一项目所有实验都用同样的 `min_delta`、`patience` 和验证划分。

## 12. 实验记录模板

每次训练结束后建议填写：

```text
实验编号：
模型/encoder：
数据划分：
随机种子：
max_epochs：300
batch size：
学习率：
supervision：
num_classes：

最佳 epoch：
最佳验证 mean Dice：
第 300 轮验证 mean Dice：
最佳 epoch 与最后一轮差值：
连续无改善 epoch 数：

训练 loss 是否仍下降：
验证 Dice 是否仍上升：
是否出现训练/验证差距：
是否有器官指标持续恶化：
最终测试使用的文件：best.pth / last.pth / 其他

结论：
[ ] 300 轮仍未学够
[ ] 约某个 epoch 后已平台
[ ] 最佳点后发生过拟合
[ ] 需要先排查实现/数据问题
```

## 最终判断口诀

```text
看验证，不看轮数；
看趋势，不看单点；
看 best，不迷信 last；
看每类，不只看平均；
先排查数据和代码，再决定是否加 epoch。
```

对当前 EMCAD 项目，最可靠的回答方式不是“300 轮够”或“300 轮不够”，而是报告：

```text
best epoch = ?
best validation mean Dice = ?
epoch 300 validation mean Dice = ?
最后 40 个 epoch 是否有超过 0.001 的改善？
训练 loss 与验证 Dice 是否出现分叉？
```

拿到这五个答案，才能有证据地判断模型是在继续学习、已经学够，还是已经过拟合。
