# 单次运行日志如何判断 EMCAD 创新涨点是否可信

## 先给结论

一次运行日志不能在统计意义上证明某个创新点“稳定有效”。它最多能证明：这次运行中的提升是否具有完整证据链，是否可能由训练异常、指标选择、实现错误或偶然的单个 epoch 造成。

因此，单次日志的目标应当是排除以下几类假提升：

1. 训练没有按预期执行；
2. 验证指标只在一个尖峰 epoch 上上涨；
3. 比较的 checkpoint、数据划分或评估流程不一致；
4. 提升来自测试集反复选模或日志统计错误；
5. 总体 Dice 上涨，但某个关键器官明显退化；
6. 代码实际运行的配置不是你以为的创新配置。

## 一、先确认日志对应的确实是目标实验

检查启动日志中的以下字段：

```text
DATASET
IMG_SIZE
BATCH_SIZE
MAX_EPOCHS
BASE_LR
SEED
SUPERVISION
FUSION_MODE
FUSION_LOSS_WEIGHT
RELIABILITY_LOSS_WEIGHT
```

当前 `sh/start_train_synapse.sh` 会把这些参数传给 `train_synapse.py`。必须确认 baseline 与创新实验只有目标创新因素发生变化。例如要验证 fusion 模块，不能同时改变：

- supervision；
- batch size；
- 学习率；
- 数据划分；
- encoder；
- 预训练权重；
- max epochs；
- checkpoint 选择规则。

日志中的 `seed` 只能标识这次随机轨迹，不能替代配置审计。

## 二、确认训练过程真的有效

### 1. 检查训练是否完整

当前训练器记录：

```text
iterations per epoch
max iterations
epoch
loss
lr
```

应确认：

- 实际 epoch 到达 `MAX_EPOCHS`；
- 每个 epoch 的 iteration 数一致；
- 中间没有 CUDA OOM、NaN、inf、DataLoader 异常或进程重启；
- 学习率等于预期值；
- 训练没有提前退出后误把已有 checkpoint 当成新结果。

### 2. 不要把日志中的 epoch loss 当作 epoch 平均损失

当前 `trainer.py` 在 epoch 结束时记录的是该 epoch 最后一个 batch 的 `loss.item()`，不是所有 batch 的平均值。因此只能用它观察大致趋势，不能据此声称训练损失下降了多少。

如果最后一个 batch 恰好很容易或很困难，日志曲线会产生假波动。单次日志分析时，应优先查看 TensorBoard 中的 `info/total_loss` 整体趋势，或重新统计每个 epoch 的 batch 平均损失。

## 三、判断验证提升是不是“稳定形态”

当前每个 epoch 都会输出：

```text
Testing performance in val model: mean_dice : ..., best_dice : ...
```

单次运行中，创新点更可信的形态是：

1. 验证 Dice 在训练中后期持续改善，或在一个连续平台上稳定高于 baseline；
2. 最佳值不是只出现一次的孤立尖峰；
3. 最佳 epoch 之后仍保持接近该水平，而不是立即大幅回落；
4. baseline 与创新方法的提升幅度明显大于日志本身的短期抖动；
5. 提升不是只发生在训练早期，随后被过拟合反超。

可以把验证曲线分成三个区间：

```text
早期：前 20% epoch
中期：20%～80% epoch
后期：最后 20% epoch
```

若创新方法只在早期高一点，后期与 baseline 持平或下降，不能称为稳定涨点。若中后期大多数 epoch 都高于 baseline，证据更强。

## 四、检查当前代码的 checkpoint 选择风险

当前 `trainer.py` 每个 epoch 都调用 `inference()`，并在性能达到历史最好值时保存：

```text
best.pth
```

但 `inference()` 使用的是：

```python
split="test_vol"
```

因此必须确认 `test_vol` 实际上是验证集，而不是官方测试集。

如果 `test_vol` 是测试集，那么当前流程会用测试集每个 epoch 选 `best.pth`。这会把测试集参与模型选择，导致单次日志中的“最佳 Dice 涨点”偏乐观。此时日志即使曲线很漂亮，也不能作为严格测试集结论。

单次运行至少要记录：

```text
模型选择集：validation
最终报告集：test
best epoch：...
best checkpoint：...
```

若项目暂时只能使用现有 `test_vol`，报告中应明确写成“按当前 test_vol 选择的最佳结果”，不要把它表述为完全独立测试结果。

## 五、判断涨点是否来自目标创新，而不是输出或损失实现问题

当前 EMCAD 的 `mutation` 监督会对 4 个输出的 15 个非空子集累加损失。它改变的是梯度路径，不只是多打印几个 loss。

因此单次日志必须核对：

1. 日志打印的监督模式确实是目标模式；
2. 四个输出数量符合预期；
3. 监督组合数量符合模式：
   - `mutation`：15 个非空组合；
   - `deep_supervision`：4 个单输出组合；
   - 其他模式：最后一个输出；
4. loss 数值不能直接拿 mutation 与 deep supervision 横向比较，因为 mutation 天然累加更多项；
5. 比较创新点时应主要比较同一评估流程下的 Dice、HD95、每器官指标和收敛行为。

## 六、必须看每个器官，而不是只看平均 Dice

总体 Dice 上涨可能来自一个大器官明显改善，同时小器官退化。单次日志若只有宏平均 Dice，证据不完整。

应至少整理：

| 指标 | Baseline | Innovation | 差值 |
|---|---:|---:|---:|
| mean Dice | | | |
| HD95 | | | |
| organ 1 Dice | | | |
| organ 2 Dice | | | |
| … | | | |

更可信的单次涨点通常表现为：

- mean Dice 上升；
- 关键器官没有明显退化；
- HD95 没有同步恶化；
- 改善集中在创新点理论上应该影响的器官或边界区域。

如果 Dice 上升但 HD95 大幅变差，可能只是区域重叠增加，边界质量变差，不能简单称为全面提升。

## 七、用“时间段稳定性”替代只看最高点

单次运行可以做一个不需要重复训练的检查：

1. 取 baseline 和 innovation 的验证曲线；
2. 对每 10 个 epoch 或每 20 个 epoch 做窗口平均；
3. 比较中后期窗口均值，而不是只比较最高点；
4. 记录最高点与中后期窗口均值的差距。

例如：

```text
innovation best Dice = 0.842
innovation last-20-epoch mean = 0.836
baseline best Dice = 0.835
baseline last-20-epoch mean = 0.832
```

这种结果比“innovation 某一个 epoch 达到 0.842，其他时候都在 0.82 左右”更可信。

## 八、一次日志可以得出的结论等级

### A 级：本次运行的提升证据完整

满足：

- 配置核对无误；
- 训练完整且无异常；
- 验证/测试划分角色清楚；
- 中后期曲线持续领先；
- best epoch 不是孤立尖峰；
- mean Dice、关键器官 Dice、HD95 方向一致；
- 提升符合创新机制预期。

可以写：

> 在本次固定划分、固定训练配置和指定 seed 下，创新方法表现出持续的验证性能优势，结果与预期机制一致。

### B 级：本次运行有提升，但证据有限

例如：

- mean Dice 上升；
- 但只领先少数 epoch；
- 或 HD95、器官级指标不一致；
- 或 checkpoint 选择集角色不够清楚。

可以写：

> 本次运行观察到初步提升，但单次曲线和指标结构不足以确认其稳定性。

### C 级：不能判定为有效涨点

例如：

- 只比较 best Dice；
- 训练日志不完整；
- 配置发生多处变化；
- 测试集被反复用于选 best；
- 指标提升伴随明显边界恶化；
- 日志中存在 NaN、异常退出或 checkpoint 来源不明。

## 九、对你当前 EMCAD 日志最实用的单次判定模板

每次实验结束后填写：

```text
实验 ID：
Baseline / Innovation：
目标创新因素：
数据划分文件：
Seed：
Deterministic：
最佳 epoch：
最佳 checkpoint：
模型选择集：
最终评估集：

Baseline best mean Dice：
Innovation best mean Dice：
Baseline 中后期窗口均值：
Innovation 中后期窗口均值：

mean Dice 差值：
HD95 差值：
各器官 Dice 差值：
是否存在孤立尖峰：
是否存在 NaN/异常退出：
是否只改变了目标创新因素：

单次证据等级：A / B / C
结论：
```

## 最终判断

你要求“一次日志就确定稳定涨点”，严格来说做不到，因为稳定性是关于重复随机过程的命题，单次观测无法证明其概率性质。

但一次日志可以完成很有价值的工作：判断这次提升是否真实执行、是否持续、是否覆盖多个指标、是否符合机制、是否被测试集选模或日志统计方式夸大。

对当前 EMCAD，最先应检查的不是 seed，而是：

1. `test_vol` 是否被当作验证集使用；
2. `best.pth` 的选择是否依赖该集合；
3. 日志中的 loss 是否被误读为 epoch 平均值；
4. mean Dice 的提升是否伴随器官级和 HD95 的支持；
5. baseline 与创新实验是否只改变目标创新因素。
