# 基于 EMCAD 快速形成 CCF-C 论文的现实路线与 Batch 训练机制

## 先把话说透

没有任何方法能保证“几天改一个模块就发 CCF-C”。但在你当前仓库里，最快的可行路线不是继续堆两个、三个创新点，而是：

> 只保留一个能解释、能做消融、能跨数据集验证的机制，把实验和论文证据做完整。

当前最适合保留的主线是：

> **分歧引导的自适应多尺度路由（Disagreement-Guided Adaptive Multi-Scale Routing）**：让 EMCAD 的 1×1、3×3、5×5 深度卷积分支根据当前位置的不确定性和尺度间预测分歧动态加权，而不是所有位置固定相加。

这个方向适合当前仓库的原因是：

- `DG_EMCAD_reference/lib/dg_emcad.py` 已经有 `DisagreementGuidedMSCB`、`DisagreementGuidedEMCAD` 和 `DGEMCADNet` 的参考实现；
- 代码已经内置 `equal`、`global`、`feature`、`disagreement` 四种路由模式，天然可以做消融；
- 论文叙事简单：固定多尺度融合无法适应边界、细小器官和均匀区域，分歧图提供位置相关的尺度选择信号；
- 不需要同时发明新 backbone、新 loss、新数据增强和新 decoder，实验变量可控；
- 即使 Dice 提升不大，也可以同时报告边界指标、参数量、FLOPs、推理时间和小器官结果。

但必须注意一个现实问题：当前 `train_synapse.py` 仍然导入原始 `EMCADNet`，而当前 `start_train_synapse.sh` 传入了 `--adaptive_msdc` 和 `--dg_router_mode disagreement`。现有 Python 参数解析器没有这两个参数，直接运行会报 `unrecognized arguments`；即使暂时没有报错，也不能把 shell 中写了参数当成模块已经生效。先把集成链路做通，再谈论文结果。

## 一、你真正要做的论文，而不是“发明两个点”

### 建议的论文题目

英文题目可以先用工作标题：

```text
Disagreement-Guided Adaptive Multi-Scale Decoding for Medical Image Segmentation
```

中文工作标题：

```text
分歧引导的自适应多尺度解码医学图像分割方法
```

不要在标题里写“通用”“智能”“高效精准”等无法由实验直接证明的词。

### 唯一主贡献

整篇论文只围绕一个问题：

> EMCAD 的 MSDC 在每个像素位置都把多个尺度分支固定相加；但器官内部、器官边界、细小结构和纹理复杂区域需要的感受野不一样。能否用多尺度输出之间的预测分歧，生成每个位置的尺度权重？

方法链条保持短：

```text
EMCAD 的每个 MSCB
    -> 1/3/5 深度卷积分支
    -> 每个分支产生 routing logits
    -> 熵 + 相邻尺度 JS 分歧得到 uncertainty map
    -> 1x1 router 预测各尺度权重
    -> 加权混合多尺度响应
    -> 原有 point-wise 投影和残差连接
```

不要在第一版论文里再加入：

- 第二个注意力模块；
- 额外教师网络；
- 复杂边界 loss；
- 新 backbone；
- 多个数据增强策略；
- 同时改变 supervision、optimizer、batch size。

这些内容会让审稿人无法判断提升来自哪里，也会让你自己的实验无法收敛到一个结论。

## 二、当前仓库必须先处理的三个硬问题

### 1. 先确认自适应模块真的被训练入口调用

当前入口文件：

```text
train_synapse.py
```

当前是：

```python
from lib.networks import EMCADNet
from trainer import trainer_synapse
```

并且模型构造是：

```python
model = EMCADNet(...)
```

而参考自适应模型在：

```text
DG_EMCAD_reference/lib/dg_emcad.py
```

类名是：

```python
DGEMCADNet
```

因此必须完成以下二选一：

1. 把参考实现稳定地迁移到当前 `lib/` 和训练入口；
2. 直接让 `train_synapse.py` 显式导入参考模型，并确保依赖路径、输出格式和 trainer 完全兼容。

不要只在 `.sh` 中增加 `--adaptive_msdc`、`--dg_router_mode`，因为参数是否存在由 Python 的 `argparse` 决定，shell 不会自动给 Python 增加功能。

### 2. 处理输出格式

原始 `EMCADNet` 返回：

```python
[p4, p3, p2, p1]
```

每个元素是 logits，典型形状为：

```text
[B, 9, 224, 224]
```

参考 `DGEMCADNet` 在 `return_aux=True` 时返回：

```python
{
    'logits': [...],
    'adaptive_aux': auxiliary,
}
```

当前 `trainer.py` 直接把返回值当作列表处理，所以必须明确选择一种接口：

- 训练主损失只使用 `outputs['logits']`；
- `adaptive_aux` 只在确实设计并验证了辅助损失时使用；
- 第一版建议不加辅助 loss，先证明路由机制本身有效。

如果 trainer 收到字典却仍执行 `len(P)`、`P[s[idx]]`，那不是“效果不好”，而是接口错误。

### 3. 禁止测试集参与每轮选模

当前 Synapse 的 `trainer.py` 在每个 epoch 后调用：

```python
Synapse_dataset(..., split="test_vol", ...)
```

然后用这个结果更新 `best.pth`。如果 `test_vol` 是最终测试病例，这就是测试集参与模型选择。这样的结果不能作为可信论文主结果。

最快的修复不是换评价指标，而是建立固定的病例级划分：

```text
train cases：只训练
validation cases：每轮选 best、调参
test cases：模型和超参数冻结后只评估一次
```

必须按病例划分，不能把同一病例的不同切片随机分到 train 和 validation，否则会产生切片级泄漏。

## 三、最小实验矩阵：只做这些，先不要扩张

### 阶段 A：开发实验，先用一个 seed 筛掉无效方向

所有实验固定：

- 同一数据划分；
- 同一 `img_size`；
- 同一 `batch_size`；
- 同一 `base_lr`；
- 同一 `max_epochs`；
- 同一预训练权重；
- 同一 supervision；
- 同一 seed；
- 同一验证和测试代码。

只改变 `router_mode`：

| 实验 | 路由模式 | 作用 |
|---|---|---|
| E0 | 原始 EMCAD | 可信 baseline |
| E1 | `equal` | 应接近原始固定相加，验证实现是否公平 |
| E2 | `global` | 只有全局固定尺度权重 |
| E3 | `feature` | 只根据特征预测空间权重 |
| E4 | `disagreement` | 特征 + 熵/JS 分歧，核心方法 |

这五个实验回答四个问题：

1. 自适应代码是否真的能复现固定融合；
2. 提升是否仅来自多了一个 router 参数；
3. 仅看 feature 是否已经足够；
4. 分歧信号是否在 `feature` 之上提供额外价值。

### 阶段 B：只保留最有希望的两个模型

如果 E4 相对 E0 或 E1 没有提升，不要继续加模块。先检查：

- `equal` 是否与 E0 使用相同分支和相同初始化；
- router 权重是否真的参与了 `mixed`；
- 训练输出是否被 trainer 正确解析；
- 验证是否发生测试泄漏；
- 参数量和学习率是否被悄悄改变。

如果实现正确但 E4 仍不提升，核心假设在当前任务上没有证据，应该止损。

### 阶段 C：第二个数据集只验证泛化，不重新搜索

建议使用仓库已有训练入口支持的 ACDC，或你已经准备好的一个 polyp 数据集。不要在第二个数据集上重新搜索几十个超参数。

只跑：

- 原始 EMCAD baseline；
- 最终 disagreement 模型。

最终论文至少报告：

- Synapse：多器官 Dice、HD95、逐器官 Dice；
- ACDC 或 polyp：主指标、边界/表面指标；
- 参数量、FLOPs、单张推理时间或显存；
- 三个不同 seed 的均值和标准差。

## 四、什么结果才值得写进论文

不要只盯着一个最高 Dice。建议把停止线预先写死：

### 可以继续写的结果

- 核心方法在同一 seed 下相对 baseline 有稳定提升，且不是只有一次偶然波动；
- 至少两个 seed 方向一致；
- 第二个数据集没有明显崩溃；
- 小器官或边界指标有可解释改善；
- 消融表能显示 `disagreement` 优于 `feature` 和 `global`；
- 没有测试集泄漏。

### 建议的最低证据门槛

这不是 CCF-C 的官方录用标准，而是你的项目止损标准：

- 验证 Dice 平均提升至少约 `0.3` 个百分点；
- 三个 seed 的平均提升方向为正；
- 关键小器官/边界指标至少一项稳定改善；
- 参数量和 FLOPs 增幅可解释，最好不超过约 `10%`；
- 任何一个主要器官不能出现不可接受的大幅退化。

如果只有 `0.05` 个百分点提升、只在一个 seed 有效、第二个数据集下降，不要把它包装成强方法。可以把它作为失败实验或负结果，继续换问题，而不是继续堆复杂度。

## 五、快速执行时间表

### 第 1 天：冻结实验协议

完成：

- 固定 commit；
- 固定病例级 train/validation/test 清单；
- 记录 Python、PyTorch、CUDA、GPU；
- 跑一次原始 EMCAD baseline；
- 保存命令、配置、日志、checkpoint。

### 第 2 天：打通自适应模型

完成：

- `DGEMCADNet` 能被训练入口构造；
- trainer 能正确解析 logits；
- `equal` 模式能正常前向、反向；
- 参数量、输出 shape、梯度非零检查通过。

### 第 3—5 天：跑 E0—E4

每个实验先用一个 seed。每个实验都记录：

```text
commit
command
seed
dataset split hash
train loss
validation Dice
HD95
per-organ Dice
best epoch
params/FLOPs
runtime
failure reason
```

### 第 6—8 天：跑第二数据集和多个 seed

只有当 E4 通过阶段 A 的门槛，才跑第二数据集和三个 seed。不要让一个没有提升的模块吞掉一周时间。

### 第 9—12 天：写论文和补图表

论文最少需要：

1. 方法总图；
2. 路由机制公式；
3. E0—E4 消融表；
4. 两个数据集主结果表；
5. 参数量/FLOPs/速度表；
6. 失败案例可视化；
7. 局限性和数据划分说明。

## 六、论文中可以直接使用的机制描述

设第 `l` 个解码层的多尺度深度卷积分支为：

```text
U_l^k = DWConv_k(PWConv(F_l)),   k ∈ {1, 3, 5}
```

原始 EMCAD 采用固定求和：

```text
U_l = U_l^1 + U_l^3 + U_l^5
```

改进模型预测每个位置的类别 logits `q_l`，计算归一化熵 `H(q_l)`；对于非最深层，再把相邻深层 logits 上采样后计算 Jensen–Shannon 分歧：

```text
u_l = [H(q_l) + λ JS(q_l, up(q_{l+1}))] / (1 + λ)
```

router 使用特征和 `u_l` 产生尺度权重：

```text
α_l = softmax(R([P_l(F_l), u_l]) / τ)
```

多尺度分支按位置混合：

```text
U_l = K · Σ_k α_l^k U_l^k
```

其中 `K` 是分支数。这样当权重相等时，`K·α_l^k=1`，可以恢复原始分支求和，便于构造公平的 `equal` 对照。

论文里不要声称“分歧一定等于真实不确定性”。更准确的写法是：分歧是一个可计算的 uncertainty proxy，实验检验它是否能帮助尺度选择。

## 七、`batch_size=16` 到底是怎么训练的

### 1. 当前脚本中的真实 batch 大小

`start_train_synapse.sh` 设置：

```bash
BATCH_SIZE=16
```

随后传给：

```bash
--batch_size "${BATCH_SIZE}"
```

在 `trainer.py` 中又计算：

```python
batch_size = args.batch_size * args.n_gpu
```

当前默认 `n_gpu=1`，并且脚本只暴露 GPU 0，因此 DataLoader 的实际 batch size 是 16。

### 2. 一个样本如何读取

`utils/dataset_synapse.py` 的训练分支每次读取一个：

```text
<root_path>/<slice_name>.npz
```

里面有：

```text
image
label
```

经过 `RandomGenerator` 后，单个样本通常是：

```text
image: [1, 224, 224]，float32
label: [224, 224]，int64
```

这里的 `1` 是 CT 单通道，不是 batch 维。

### 3. DataLoader 一次拿 16 个样本

代码是：

```python
trainloader = DataLoader(
    db_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    worker_init_fn=worker_init_fn,
)
```

DataLoader 把 16 个单样本沿最前面增加并堆叠成：

```text
image_batch: [16, 1, 224, 224]
label_batch: [16, 224, 224]
```

如果训练切片总数不是 16 的整数倍，最后一个 batch 默认可以小于 16，因为当前没有设置 `drop_last=True`。

每个 epoch 的 batch 数是：

```python
len(trainloader) = ceil(len(train.txt) / 16)
```

`len(train.txt)` 这里指清单里的切片行数，不是患者数。

### 4. 16 张图如何经过 EMCAD

代码先把 batch 移到 GPU：

```python
image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
```

然后前向：

```python
P = model(image_batch, mode='train')
```

默认 PVTv2-B2 + EMCAD 的典型形状是：

```text
输入                 [16, 1, 224, 224]
单通道适配为 3 通道   [16, 3, 224, 224]
encoder x1           [16,  64, 56, 56]
encoder x2           [16, 128, 28, 28]
encoder x3           [16, 320, 14, 14]
encoder x4           [16, 512,  7,  7]
四个分割 logits       4 个 [16, 9, 224, 224]
```

这里的 `9` 是 Synapse 的总类别数：背景 1 类 + 8 个器官类。

### 5. loss 如何计算

当前默认 `supervision='mutation'`。四个输出记为：

```text
P = [p4, p3, p2, p1]
```

训练器生成四个输出的所有非空子集，总共 15 组。例如：

```text
[p4]
[p3]
[p4,p3]
...
[p4,p3,p2,p1]
```

每一组先把 logits 逐元素相加，再计算：

```python
loss_group = 0.3 * CrossEntropyLoss(iout, label_batch) \
           + 0.7 * DiceLoss(iout, label_batch)
```

然后把 15 组损失相加得到当前 batch 的一个标量 `loss`。注意：当前代码没有除以 15，所以 mutation 的 loss 数值尺度会比只监督一个输出大；这不影响同一 supervision 内部训练，但不能把不同 supervision 的原始 loss 直接横向比较。

### 6. 一个 batch 只更新一次模型

当前代码顺序是：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

含义是：

1. 清空上一个 batch 的梯度；
2. 用这 16 张图的总 loss 反向传播；
3. 把 16 张图的梯度汇总到同一套模型参数；
4. AdamW 对每个参数执行一次更新。

所以：

> `batch_size=16` 不是一次更新 16 次，而是一次用 16 个样本共同估计梯度，然后更新 1 次。

可以把它写成：

```text
16 个样本
    -> 16 个预测
    -> 汇总成一个 batch loss
    -> backward 一次
    -> AdamW.step 一次
```

### 7. 一个 epoch 做多少次参数更新

假设 `train.txt` 有 `N` 行，且 `batch_size=16`：

```text
每个 epoch 的更新次数 = ceil(N / 16)
300 epoch 的总更新次数 = 300 × ceil(N / 16)
```

`iter_num` 每处理完一个 batch 加 1，不是每读一张图片加 1。

例如如果训练清单有 3200 张切片：

```text
每 epoch = 3200 / 16 = 200 次 optimizer.step()
300 epoch = 60000 次 optimizer.step()
```

如果有 3205 张切片：

```text
每 epoch = ceil(3205 / 16) = 201 次更新
最后一个 batch 只有 5 张图
```

实际数量不要猜，直接查看日志中的：

```text
iterations per epoch
```

或者在 Python 中打印：

```python
print('train slices:', len(db_train))
print('batches per epoch:', len(trainloader))
```

### 8. `batch_size` 和多 GPU 的关系

当前代码先计算：

```python
batch_size = args.batch_size * args.n_gpu
```

如果 `args.batch_size=16`、`args.n_gpu=1`：

```text
DataLoader 取 16 张，单卡处理 16 张
```

如果错误地设置 `args.n_gpu=2`，则 DataLoader 可能取 32 张，再由 `DataParallel` 分到两张卡，每卡约 16 张。它和“总 batch size=16”不是一回事。

你当前 `.sh` 中有：

```bash
export CUDA_VISIBLE_DEVICES=0
```

所以现在讨论的实际情况就是：GPU 0 一次处理 16 张切片。

### 9. 当前没有梯度累积

代码每个 batch 都立即调用 `optimizer.step()`，所以没有梯度累积。若想让显存仍按 16 张图运行，但得到近似有效 batch size 64，需要明确实现：

```text
4 个 micro-batch × 16 张
累积梯度
最后再 optimizer.step()
```

不能只把 `BATCH_SIZE=16` 改成 64 就声称做了梯度累积；那会直接增加显存占用。

## 八、哪些东西不算论文创新

下面这些可以作为训练设置，但不能单独包装成方法贡献：

- 把 batch size 从 6 改成 16；
- 把 epoch 从 300 改成 400；
- 只换学习率；
- 只换随机增强；
- 只把 `mutation` 改成 `deep_supervision`；
- 只把 PVTv2-B2 换成 B0/B3；
- 只把输入从 224 改成 256；
- 只报告一次最好的 seed。

这些变量必须在 baseline 和候选方法中保持一致，除非论文明确研究的是训练策略本身。

## 九、最小可交付论文包

在投稿前至少准备以下文件：

```text
paper/
├── main_results.csv
├── ablation.csv
├── per_case_metrics.csv
├── commands/
│   ├── baseline_synapse.txt
│   ├── disagreement_synapse.txt
│   └── disagreement_acdc.txt
├── configs/
├── logs/
├── checkpoints_manifest.csv
├── split_manifest.csv
├── environment.txt
├── failure_log.csv
└── figures/
```

每个数字都要能追溯到：

```text
代码 commit + 命令 + seed + 数据清单 + checkpoint + 日志
```

## 十、最后的执行决策

你的行动顺序应当是：

1. 先修复并验证 `DGEMCADNet` 与当前 trainer 的接口；
2. 建立无测试泄漏的固定病例划分；
3. 跑原始 EMCAD 和 `equal/global/feature/disagreement` 五组；
4. 如果 disagreement 没有稳定提升，停止这个方向，不再添加第二个创新点；
5. 如果有效，只保留 disagreement 一个主创新，去第二个数据集和三个 seed；
6. 用消融、边界/小器官结果、效率和失败案例支撑论文；
7. 最后再选择具体 CCF-C 会议或期刊，并根据其最新征稿范围调整格式。

最重要的一句话是：

> 你现在缺的不是第四个模块，而是一条被固定协议、对照实验和可追溯记录证明过的单一机制。

