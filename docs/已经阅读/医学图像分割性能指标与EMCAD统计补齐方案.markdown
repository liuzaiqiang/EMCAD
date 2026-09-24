# 医学图像分割性能指标与 EMCAD 统计补齐方案

本文基于当前分支的 Synapse/ACDC 代码，只新增说明，不直接修改 Python 或 Bash 文件。

## 一、先给结论

你当前代码不是“完全没有指标”，而是指标分散在训练和测试不同文件中，而且工程指标没有形成可用于论文的结构化记录。

当前状态大致如下：

| 指标 | 当前代码状态 | 主要位置 |
|---|---|---|
| 训练 CE/Dice 混合 loss | 有，按 batch 写 TensorBoard，文本日志记录最后一个 batch | `trainer.py` |
| 验证 Dice | 有，每轮验证 | `trainer.py`、`utils/utils.py` |
| 测试 Dice | 有，按器官和总体计算 | `test_synapse.py` |
| HD95 | 有，最终测试计算 | `test_synapse.py`、`utils/utils.py` |
| Jaccard/IoU | 有，最终测试计算 | `test_synapse.py`、`utils/utils.py` |
| ASD/ASSD | 有，最终测试计算 | `test_synapse.py`、`utils/utils.py` |
| Params | 有辅助函数，但训练入口没有统一调用和保存 | `utils/utils.py` |
| FLOPs/MACs | 有 THOP/PTFLOPS 辅助函数，但训练入口没有统一调用和保存 | `utils/utils.py` |
| 每轮训练时间 | 没有正式记录 | `trainer.py` 已导入 `time`，但当前路径未使用 |
| 每轮验证时间 | 没有正式记录 | `trainer.py` |
| 最终测试时间 | 没有正式记录 | `test_synapse.py` |
| 峰值显存 | 没有正式记录 | 需要新增 |
| 参数量、FLOPs、时间、显存 CSV/JSON | 没有 | 需要新增 |

论文中建议至少报告：

```text
Dice、HD95、IoU/Jaccard、ASSD/ASD、逐类别结果、Params、FLOPs 或 MACs、
训练总时间、每 epoch 训练时间、验证时间、最终推理时间、单病例推理时间、峰值显存。
```

不要把所有指标都放在一个“accuracy”里。医学分割通常重点关注区域重叠和边界误差，而不是普通分类准确率。

## 二、质量指标应该计算哪些

### 1. Dice

Dice 衡量预测区域和真实区域的重叠：

```text
Dice = 2 |P ∩ G| / (|P| + |G|)
```

建议报告：

- 每个器官/病灶的 Dice；
- 前景类别 macro mean Dice；
- 多个 seed 的均值和标准差；
- 最好同时报告中位数或病例级分布，而不只报告最高值。

当前 Synapse 的最终测试路径会跳过背景，只平均 8 个前景器官，这是合理的论文口径，但必须在论文中写清楚。

### 2. IoU/Jaccard

```text
IoU = |P ∩ G| / |P ∪ G|
```

Dice 和 IoU 高度相关，但 IoU 对区域错误的惩罚方式不同。可以作为辅助指标，不能把 Dice 和 IoU 当成两个互相独立的创新证据。

### 3. HD95

HD95 是预测表面和真实表面之间的 95% Hausdorff 距离，对边界离群点比最大 Hausdorff 距离更稳健。

- 越低越好；
- 单位必须写清楚是像素还是毫米；
- 必须说明体素 spacing；
- 空预测、空真值和单侧空掩膜必须有明确处理规则。

### 4. ASD/ASSD

ASSD 是预测表面和真实表面的平均对称距离，也属于边界质量指标。

- 越低越好；
- 对 spacing 同样敏感；
- 不能只看 Dice，不看边界偏移。

### 5. Sensitivity、Specificity、Precision

病灶分割、医学筛查任务有时还会报告：

```text
Sensitivity/Recall = TP / (TP + FN)
Specificity = TN / (TN + FP)
Precision = TP / (TP + FP)
```

Synapse 多器官分割通常以 Dice、HD95、ASSD 为主；如果你的方法主张减少漏检或减少假阳性，再补 Sensitivity/Specificity，否则不要为了“指标数量多”而堆表格。

### 6. 训练 loss 不是论文最终质量指标

当前训练器使用：

```python
0.3 * CrossEntropyLoss + 0.7 * DiceLoss
```

loss 用于优化，不等于最终报告的 Dice。尤其当前 `mutation` supervision 会把 15 组损失相加，loss 尺度不能直接和 `deep_supervision` 或 `last_layer` 横向比较。

## 三、工程指标应该计算哪些

### 1. Params

参数量是模型中参数标量的总数。建议同时报告：

- Total Params：全部参数；
- Trainable Params：`requires_grad=True` 的参数；
- 单位统一为 M；
- 是否包含 encoder、decoder、输入适配层和预测头。

当前 `utils/utils.py` 的 `CalParams` 可以通过 `model.parameters()` 统计，但它没有返回原始数值，也没有被 Synapse 训练入口统一调用。论文最好保存原始整数和格式化后的 M 值。

### 2. FLOPs 与 MACs

不同工具和论文会把一次乘加记为 1 个 MAC 或 2 个 FLOPs，因此必须写明：

- 工具，例如 THOP；
- 输入尺寸，例如 `[1, 1, 224, 224]`；
- batch size，复杂度统计一般用 batch=1；
- 输出是 FLOPs 还是 MACs；
- 是否包含激活、插值、BN 和自定义算子。

当前 `requirements.txt` 已包含 `thop`、`ptflops` 和 `torchprofile`。不需要同时用三种工具作为三个不同结果；建议固定 THOP 为主口径，另一种工具只做交叉检查。

### 3. 训练时间

建议记录：

- 每个 epoch 的纯训练时间；
- 每个 epoch 的验证时间；
- 总训练 wall-clock 时间；
- 是否包含首次启动、数据预取和验证；
- GPU 型号、PyTorch/CUDA 版本。

### 4. 验证时间和最终推理时间

必须区分：

- `validation time`：训练过程中每轮完整验证的耗时；
- `test inference time`：冻结模型后对最终测试集的耗时；
- `per-case time`：一个三维病例从读取到预测和指标计算的耗时；
- `per-slice time`：单张二维切片的纯网络前向耗时。

当前 Synapse 测试是按三维病例读取，再逐切片调用二维网络；如果只报告单张 224×224 的前向时间，不能把它等同于完整病例推理时间。

### 5. 峰值显存

训练和推理分别记录：

```python
torch.cuda.max_memory_allocated() / 1024**3
```

记录前要调用：

```python
torch.cuda.reset_peak_memory_stats()
```

显存和 batch size 强相关，所以论文中必须同时写 batch size、输入尺寸和 AMP 是否开启。

### 6. 吞吐量

可选报告：

```text
images_per_second = 样本数 / 纯前向或训练耗时
slices_per_second = 切片数 / 推理耗时
```

医学三维测试更建议报告 `cases/min` 或 `slices/s`，不要只报告 GPU 理论算力。

## 四、当前代码中必须先说明的指标风险

### 1. 训练阶段只算 Dice，HD95/ASSD 在最终测试算

`trainer.py` 调用 `val_single_volume()` 时只计算 Dice，避免每个 epoch 都计算昂贵的表面距离。`test_synapse.py` 的最终测试才计算 Dice、HD95、Jaccard、ASD。

这是可以接受的效率设计，但论文中不要写成“每个 epoch 都用 HD95 选择最佳模型”。当前实际是用验证 Dice 选择 `best.pth`，最终冻结后再测试 HD95/ASSD。

### 2. 当前 spacing 不是可靠的物理毫米口径

`utils/utils.py` 的 `calculate_metric_percase()` 调用 MedPy 的 `hd95` 和 `assd` 时没有传入 `voxelspacing`。因此当前数值实际更接近像素距离，而不是可靠的毫米距离。

如果论文写“HD95=xx mm”，必须先把真实 spacing 从数据读取出来并传给 MedPy。否则只能写“HD95（pixel）”，并明确这是当前评价实现的单位。

### 3. 当前空掩膜策略必须审计

当前实现对“预测有前景、真实为空”的分支返回 Dice=1、Jaccard=1、距离=0。这个规则通常会把假阳性当成完美结果，不能直接当作标准评价。

建议在论文实验前建立一个版本化的评价函数：

```text
预测空、真值空：Dice=1，距离=0
预测非空、真值空：Dice=0，距离按约定处理
预测空、真值非空：Dice=0，距离按约定处理
双方非空：调用 MedPy 正常计算
```

不要悄悄修改历史结果。旧结果和修正后的结果必须标记为不同评价版本，并重新跑 baseline 和候选模型。

## 五、在 `trainer.py` 中补齐训练/验证时间和显存

以下方案只说明插入位置。

### 1. 在 import 区确认 `time`

当前 `trainer.py` 已有：

```python
import time
```

所以无需重复导入。

### 2. 在文件顶部增加 CUDA 同步辅助函数

放在 import 区之后、`inference()` 之前：

```python
def sync_cuda():
    """等待 CUDA 队列完成，避免异步执行导致耗时偏小。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
```

CUDA kernel 默认可能异步执行。如果不调用 `torch.cuda.synchronize()`，`time.perf_counter()` 测到的可能只是 CPU 发起任务的时间。

### 3. 在 `trainer_synapse()` 中记录总训练起点

找到：

```python
def trainer_synapse(args, model, snapshot_path):
```

紧接着函数体第一行插入：

```python
    total_train_start = time.perf_counter()
```

### 4. 在 epoch 循环中记录纯训练时间

找到：

```python
for epoch_num in iterator:
    model.train()
```

改成：

```python
for epoch_num in iterator:
    sync_cuda()
    epoch_train_start = time.perf_counter()
    model.train()
```

找到内层 batch 循环结束后、原来的：

```python
logging.info('iteration %d, epoch %d : loss : %f, lr: %f' %
             (iter_num, epoch_num, loss.item(), lr_))
```

之后插入：

```python
sync_cuda()
train_seconds = time.perf_counter() - epoch_train_start
```

这里统计的是从 epoch 开始进入训练模式，到最后一个 batch 完成 GPU 计算的时间。

### 5. 在验证调用前后记录验证时间

当前代码是：

```python
performance = inference(args, model, best_performance)
```

替换为：

```python
sync_cuda()
validation_start = time.perf_counter()
performance = inference(args, model, best_performance)
sync_cuda()
validation_seconds = time.perf_counter() - validation_start

logging.info(
    'epoch=%d train_seconds=%.3f validation_seconds=%.3f',
    epoch_num + 1,
    train_seconds,
    validation_seconds,
)
```

### 6. 记录峰值显存

在训练开始前、`model.to(device)` 之后插入：

```python
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
```

在每个 epoch 验证结束后插入：

```python
peak_memory_gb = 0.0
if device.type == 'cuda':
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

logging.info(
    'epoch=%d peak_memory_allocated_gb=%.3f',
    epoch_num + 1,
    peak_memory_gb,
)
```

如果要获得“每 epoch 峰值”，在每个 epoch 开始处再次调用 `reset_peak_memory_stats()`；如果要获得“整个运行峰值”，只在训练开始前调用一次。

### 7. 训练结束时记录总时间

找到函数结尾的：

```python
writer.close()
return "Training Finished!"
```

在 `writer.close()` 之前插入：

```python
sync_cuda()
total_train_seconds = time.perf_counter() - total_train_start
logging.info(
    'training_total_seconds=%.3f training_total_hours=%.3f',
    total_train_seconds,
    total_train_seconds / 3600.0,
)
```

## 六、把每个 epoch 写入 CSV

仅写文本日志不利于画图和统计均值。建议在 `trainer.py` 的 `trainer_synapse()` 开始处创建：

```python
epoch_metrics_path = os.path.join(snapshot_path, 'epoch_metrics.csv')
if not os.path.exists(epoch_metrics_path):
    with open(epoch_metrics_path, 'w', newline='', encoding='utf-8') as stream:
        stream.write(
            'epoch,iter_num,train_seconds,validation_seconds,'
            'validation_mean_dice,peak_memory_gb,learning_rate\n'
        )
```

然后在每个 epoch 验证和显存统计完成后插入：

```python
with open(epoch_metrics_path, 'a', newline='', encoding='utf-8') as stream:
    stream.write(
        '{},{},{:.6f},{:.6f},{:.8f},{:.6f},{:.10g}\n'.format(
            epoch_num + 1,
            iter_num,
            train_seconds,
            validation_seconds,
            float(performance),
            peak_memory_gb,
            float(optimizer.param_groups[0]['lr']),
        )
    )
```

如果你的文件使用 `csv.writer`，也可以替代字符串拼接；关键是保留固定列名和 UTF-8 编码。

最终可以用 pandas 读取：

```python
import pandas as pd

table = pd.read_csv('model_pth/run_seed3333/epoch_metrics.csv')
print(table.tail())
print('total train hours:', table['train_seconds'].sum() / 3600.0)
print('mean epoch train seconds:', table['train_seconds'].mean())
```

## 七、在训练入口中统计 Params 和 FLOPs

### 1. 推荐放置位置

在 `trainer.py` 中，模型完成：

```python
model.to(device)
```

之后、创建 loss 和 optimizer 之前，插入复杂度统计。此时模型结构已经确定，输入尺寸也由 `args.img_size` 确定。

### 2. 可直接复制的统计代码

```python
    profile_model = model.module if isinstance(model, nn.DataParallel) else model

    total_params = sum(param.numel() for param in profile_model.parameters())
    trainable_params = sum(
        param.numel()
        for param in profile_model.parameters()
        if param.requires_grad
    )

    complexity = {
        'input_shape': [1, 1, args.img_size, args.img_size],
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'total_params_m': float(total_params / 1e6),
        'trainable_params_m': float(trainable_params / 1e6),
        'flops': None,
        'macs': None,
        'profile_tool': 'thop',
    }

    try:
        from thop import profile

        was_training = profile_model.training
        profile_model.eval()
        dummy = torch.zeros(
            1, 1, args.img_size, args.img_size,
            device=device,
        )
        sync_cuda()
        with torch.no_grad():
            macs, params_from_thop = profile(
                profile_model,
                inputs=(dummy,),
                verbose=False,
            )
        sync_cuda()
        if was_training:
            profile_model.train()

        complexity['macs'] = int(macs)
        # THOP 通常把一次乘加按 MAC 统计；若论文报告 FLOPs，明确采用 2*MACs。
        complexity['flops'] = int(2 * macs)
        complexity['thop_params'] = int(params_from_thop)
    except Exception as error:
        # 复杂模型或第三方算子可能无法被 THOP 完整识别；不能因此阻止训练。
        logging.warning('THOP profile failed: %s', error)
        complexity['profile_error'] = repr(error)

    logging.info(
        'complexity total_params=%d trainable_params=%d macs=%s flops=%s',
        complexity['total_params'],
        complexity['trainable_params'],
        complexity['macs'],
        complexity['flops'],
    )

    with open(
        os.path.join(snapshot_path, 'complexity.json'),
        'w',
        encoding='utf-8',
    ) as stream:
        json.dump(complexity, stream, ensure_ascii=False, indent=2)
```

### 3. 需要增加的 import

当前 `trainer.py` 需要在 import 区增加：

```python
import json
```

`torch`、`nn`、`os`、`logging`、`time` 已经存在，不要重复导入。

### 4. THOP 统计失败怎么办

不要把失败时的 `0 FLOPs` 写进论文。正确处理是：

1. 保留 `profile_error`；
2. 使用 `ptflops` 或 `torchprofile` 交叉统计；
3. 对自定义模块补 hook；
4. 统一输入尺寸和工具后重新统计。

可以先单独验证环境：

```bash
python -c "import thop, ptflops; print('complexity tools available')"
```

## 八、在 `test_synapse.py` 中统计最终测试时间

### 1. 增加 import

在 `test_synapse.py` 的 import 区确认：

```python
import time
```

### 2. 增加 CUDA 同步函数

放在 `inference()` 之前：

```python
def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
```

### 3. 记录整个测试集耗时

找到文件末尾：

```python
inference(args, model, test_save_path)
```

替换为：

```python
sync_cuda()
test_start = time.perf_counter()
inference(args, model, test_save_path)
sync_cuda()
test_seconds = time.perf_counter() - test_start

logging.info(
    'test_total_seconds=%.3f test_total_minutes=%.3f',
    test_seconds,
    test_seconds / 60.0,
)
print('TEST_TOTAL_SECONDS={:.3f}'.format(test_seconds))
```

### 4. 记录每个病例耗时

在 `inference()` 的病例循环中，找到：

```python
for i_batch, sampled_batch in tqdm(enumerate(testloader)):
```

循环体第一行插入：

```python
        sync_cuda()
        case_start = time.perf_counter()
```

找到调用：

```python
metric_i = test_single_volume(...)
```

调用结束后插入：

```python
        sync_cuda()
        case_seconds = time.perf_counter() - case_start
        logging.info(
            'case=%s inference_seconds=%.6f',
            case_name,
            case_seconds,
        )
```

这段统计包含 H5 读取、逐切片推理、预测重组和指标计算。论文如果需要“纯网络前向时间”，应另写一个不包含磁盘读取和 MedPy 指标的 benchmark，不能混用。

## 九、纯前向 benchmark：不要用第一次运行直接计时

FLOPs 和纯前向时间建议单独做，不要混在训练循环里。可在一个单独脚本中使用下面代码：

```python
import time
import torch


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


model.eval()
device = next(model.parameters()).device
dummy = torch.randn(1, 1, 224, 224, device=device)

with torch.no_grad():
    for _ in range(20):
        model(dummy, mode='test')
sync_cuda()

if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)

repeat = 100
start = time.perf_counter()
with torch.no_grad():
    for _ in range(repeat):
        model(dummy, mode='test')
sync_cuda()
elapsed = time.perf_counter() - start

print('mean_forward_ms={:.4f}'.format(elapsed / repeat * 1000.0))
if device.type == 'cuda':
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print('peak_inference_memory_gb={:.4f}'.format(peak_gb))
```

前面的 20 次是 warm-up，避免首次 CUDA 初始化、内核选择和缓存分配污染结果。报告时固定：

```text
batch=1、输入=1×224×224、eval、no_grad、warm-up 次数、重复次数、GPU 型号。
```

## 十、建议的最终结果表

### 1. 主结果表

| Method | Dataset | Dice ↑ | HD95 ↓ | ASSD ↓ | IoU ↑ |
|---|---|---:|---:|---:|---:|
| EMCAD baseline | Synapse |  |  |  |  |
| Proposed | Synapse |  |  |  |  |
| EMCAD baseline | ACDC/Polyp |  |  |  |  |
| Proposed | ACDC/Polyp |  |  |  |  |

主表必须使用同一个测试划分、同一个评价实现和同一个 spacing 口径。

### 2. 逐类别表

| Method | Spleen | R Kidney | L Kidney | Gallbladder | Liver | Pancreas | Stomach | Aorta | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline |  |  |  |  |  |  |  |  |  |
| Proposed |  |  |  |  |  |  |  |  |  |

Synapse 的小器官通常比大器官更能体现边界和多尺度机制是否有用，不能只报告总体平均值。

### 3. 效率表

| Method | Total Params (M) | Trainable Params (M) | MACs (G) | FLOPs (G) | Peak Train VRAM (GB) | Mean Forward (ms) | Test Time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| EMCAD baseline |  |  |  |  |  |  |  |
| Proposed |  |  |  |  |  |  |  |

如果使用 THOP，建议主表写 `MACs (G)`，另在脚注说明 `FLOPs=2×MACs` 的换算口径，避免读者误以为所有工具中的 FLOPs 完全同义。

### 4. 训练成本表

| Method | Epochs | Batch | Train h | Validation h | Total h | Seed |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 300 | 16 |  |  |  | 2222 |
| Proposed | 300 | 16 |  |  |  | 2222 |

训练总时间必须在同一 GPU、同一数据读取设置和同一 epoch 数下比较。

## 十一、当前 batch size=16 对时间和 FLOPs 的影响

复杂度通常按 batch=1 统计，训练时间则受 batch size 影响：

- batch 从 6 改成 16 会增加显存；
- 每个 epoch 的 optimizer 更新次数会减少；
- 总样本数不变时，每 epoch 仍然看一遍训练集；
- batch 内梯度是 16 张切片共同估计的结果；
- batch size 变化可能改变收敛轨迹，不能把它当成纯工程小改动；
- 如果比较模型效率，必须固定 batch size 和输入尺寸。

你当前 Synapse 训练入口在单 GPU 下的实际形状是：

```text
训练输入： [16, 1, 224, 224]
训练标签： [16, 224, 224]
模型输出： 4 个 [16, 9, 224, 224]
一次 optimizer.step()：使用这 16 张切片的联合梯度更新一次
```

FLOPs 统计不要把 `[16, 1, 224, 224]` 当成论文模型复杂度输入，否则结果会变成 batch=16 的总计算量，无法和常见 batch=1 报告直接比较。

## 十二、最小实施顺序

推荐按下面顺序修改，避免一上来改十个文件：

1. 在 `trainer.py` 增加 epoch 训练时间、验证时间和总时间；
2. 在 `trainer.py` 增加 Params/FLOPs 的 `complexity.json`；
3. 在 `trainer.py` 增加 `epoch_metrics.csv`；
4. 在 `test_synapse.py` 增加最终测试总时间和病例级时间；
5. 审计 spacing 和空掩膜指标政策；
6. 用同一 baseline 和候选模型重新跑全部统计；
7. 将 CSV、JSON、日志、配置和 checkpoint 一起归档。

不要先修改指标函数再拿新结果和旧日志直接比较。指标政策一旦变化，baseline 和候选方法必须在同一评价版本上重新计算。

## 十三、最终论文前的检查清单

- [ ] Dice 是否按前景类别分别报告；
- [ ] 是否报告 HD95 和 ASSD 的方向、单位和 spacing；
- [ ] 是否报告 IoU/Jaccard 的计算口径；
- [ ] 是否说明空预测/空真值处理；
- [ ] 是否避免使用测试集选择 `best.pth`；
- [ ] Params 是否包含 encoder、decoder 和输出头；
- [ ] FLOPs/MACs 是否写明工具、输入尺寸和换算规则；
- [ ] 训练时间是否包含验证，是否使用 CUDA 同步；
- [ ] 测试时间是否区分病例级、切片级和纯前向；
- [ ] 峰值显存是否说明 batch size、输入尺寸和 AMP；
- [ ] 是否报告 seed 均值和标准差；
- [ ] 是否保存每病例结果，而不是只保存一个总体均值；
- [ ] 每个数字是否能追溯到 commit、命令、配置、日志和 checkpoint。
