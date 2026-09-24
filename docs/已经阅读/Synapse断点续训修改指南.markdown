# Synapse 训练断点续训修改指南

本文基于当前分支的 Synapse 训练代码编写。目标是：训练在第 204 轮异常退出后，可以从最近一次完整保存的第 200 轮继续训练到第 300 轮，而不是重新从第 0 轮开始。

本文只提供修改方法，不直接修改仓库中的 Python 或 Bash 文件。

## 一、先说结论

当前代码可以每 50 轮生成 `.pth`，但这些文件目前不能直接实现严格的断点续训。

当前 `trainer.py` 的保存语句是：

```python
torch.save(model.state_dict(), save_mode_path)
```

它只保存模型参数，不保存：

- 当前完成到第几轮；
- AdamW 的内部状态，例如动量和参数步数；
- `best_performance`；
- `iter_num`；
- Python、NumPy、PyTorch 和 CUDA 的随机数状态。

因此，现有的 `epoch_199.pth` 可以用于“从第 200 轮的模型参数重新开始优化”，但不是完整意义上的无缝恢复。

推荐分两步做：

1. 先增加 `--resume` 参数，并支持读取现有的裸 `state_dict`。这样已有的 `epoch_199.pth` 可以救回当前实验。
2. 同时把以后保存的文件改成完整 checkpoint，至少保存 `model`、`optimizer`、`epoch`、`iter_num` 和 `best_performance`。之后再中断时，就能严格从最近完成的轮次恢复。

## 二、当前代码中最容易误判的地方

### 1. 文件名不是人类轮次

`trainer.py` 的外层循环是：

```python
for epoch_num in iterator:
```

其中 `epoch_num` 从 `0` 开始。当前定期保存条件是：

```python
if (epoch_num + 1) % save_interval == 0:
```

但文件名使用的是 `epoch_num`：

```python
'epoch_' + str(epoch_num) + '.pth'
```

所以当前文件对应关系是：

| 文件名 | 实际含义 |
|---|---|
| `epoch_49.pth` | 第 50 轮训练完成后的参数 |
| `epoch_99.pth` | 第 100 轮训练完成后的参数 |
| `epoch_149.pth` | 第 150 轮训练完成后的参数 |
| `epoch_199.pth` | 第 200 轮训练完成后的参数 |
| `epoch_249.pth` | 第 250 轮训练完成后的参数 |
| `epoch_299.pth` | 第 300 轮训练完成后的参数 |

因此，如果程序在第 204 轮停止，最近的完整阶段文件通常是 `epoch_199.pth`，恢复时应该从人类意义上的第 201 轮开始，而不是再次训练第 200 轮。

### 2. `last.pth` 和 `best.pth` 不能直接续训

当前代码每轮都会覆盖：

```python
last.pth
```

但它同样只有模型参数。

`best.pth` 是验证 Dice 达到历史最好时保存的参数，不代表固定轮次。例如它可能来自第 137 轮或第 186 轮。因此：

- 要继续最近训练进度，优先使用完整的 `checkpoint_latest.pth`；
- 如果只有旧文件，使用 `epoch_199.pth`；
- 不要把 `best.pth` 当作“第 200 轮断点”，除非日志明确证明它是在第 200 轮保存的。

## 三、第一部分：在 `train_synapse.py` 增加命令行参数

打开项目根目录下的：

```text
train_synapse.py
```

找到大约第 90 行、这一行之后：

```python
args = parser.parse_args()
```

把下面代码插入到 `args = parser.parse_args()` 之前：

```python
parser.add_argument(
    '--resume',
    type=str,
    default='',
    help='path to a checkpoint used to resume training; empty means start from scratch'
)
```

插入后应类似：

```python
parser.add_argument('--seed', type=int, default=2222, help='random seed')

parser.add_argument(
    '--resume',
    type=str,
    default='',
    help='path to a checkpoint used to resume training; empty means start from scratch'
)

args = parser.parse_args()
```

### 为什么参数放在这里

`train_synapse.py` 负责解析命令行参数，然后创建模型并调用：

```python
trainer[dataset_name](args, model, snapshot_path)
```

所以 `args.resume` 必须在调用 `trainer_synapse` 前存在，`trainer.py` 才能读取它。

## 四、第二部分：修改 `trainer.py` 的训练函数初始化区

打开：

```text
trainer.py
```

在 `trainer_synapse()` 中找到现有代码，大约在第 151 至 165 行：

```python
optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
writer = SummaryWriter(snapshot_path + '/log')
iter_num = 0
max_epoch = args.max_epochs
max_iterations = args.max_epochs * len(trainloader)
logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
best_performance = 0.0
iterator = tqdm(range(max_epoch), ncols=70)
```

用下面代码替换这一整段。注意：`optimizer` 必须先创建，之后才能把 checkpoint 中的优化器状态加载进去。

```python
optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
writer = SummaryWriter(snapshot_path + '/log')

max_epoch = args.max_epochs
max_iterations = args.max_epochs * len(trainloader)
logging.info("{} iterations per epoch. {} max iterations ".format(
    len(trainloader), max_iterations))

# 默认从第 0 轮开始；resume 成功后会覆盖这三个变量。
start_epoch = 0
iter_num = 0
best_performance = 0.0

if args.resume:
    resume_path = os.path.abspath(args.resume)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError("Resume checkpoint not found: {}".format(resume_path))

    logging.info("Loading resume checkpoint: %s", resume_path)
    checkpoint = torch.load(resume_path, map_location=device)

    # 新格式是字典；旧的 epoch_199.pth 只有模型 state_dict，也是字典。
    is_full_checkpoint = (
        isinstance(checkpoint, dict)
        and 'model_state_dict' in checkpoint
    )

    if is_full_checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # checkpoint 的 epoch 使用“已完成的轮数”，例如第 200 轮保存为 200。
        start_epoch = int(checkpoint.get('epoch', 0))
        iter_num = int(checkpoint.get('iter_num', start_epoch * len(trainloader)))
        best_performance = float(checkpoint.get('best_performance', 0.0))
    else:
        # 兼容当前仓库已经生成的 epoch_199.pth、last.pth、best.pth。
        model.load_state_dict(checkpoint, strict=True)

        # 旧文件名 epoch_199.pth 表示第 200 轮已经完成。
        checkpoint_name = os.path.basename(resume_path)
        if checkpoint_name.startswith('epoch_') and checkpoint_name.endswith('.pth'):
            epoch_text = checkpoint_name[len('epoch_'):-len('.pth')]
            if epoch_text.isdigit():
                start_epoch = int(epoch_text) + 1
        else:
            raise ValueError(
                'A legacy model-only checkpoint has no reliable epoch number. '
                'Use a file such as epoch_199.pth or convert it manually.'
            )

        iter_num = start_epoch * len(trainloader)
        logging.warning(
            'Legacy model-only checkpoint loaded: optimizer and RNG states were not restored.'
        )

    if start_epoch >= max_epoch:
        raise ValueError(
            'Resume epoch {} is not less than max_epochs {}.'.format(
                start_epoch, max_epoch))

    logging.info(
        'Resume successful: start_epoch=%d, iter_num=%d, best_performance=%.6f',
        start_epoch, iter_num, best_performance)

iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
```

### 这段代码的轮次行为

如果传入旧文件：

```text
epoch_199.pth
```

程序会计算：

```python
start_epoch = 199 + 1
```

随后执行：

```python
range(200, 300)
```

这意味着程序从第 201 轮开始训练，正好接在第 200 轮完成之后。

## 五、第三部分：修改循环内对 `epoch_num` 的使用

找到原来的循环：

```python
for epoch_num in iterator:
```

保持这一行不变，因为上面的 `iterator` 已经改成了：

```python
iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
```

但是要注意原代码中这段初始化逻辑：

```python
if epoch_num == 0 and i_batch == 0:
    n_outs = len(P)
    out_idxs = list(np.arange(n_outs))
    ...
    print(ss)
```

续训时 `epoch_num` 从 200 开始，这个条件不会执行，`ss` 就不会被初始化。必须把条件改为只初始化一次，而不是只在第 0 轮初始化。

### 推荐改法

在 `for epoch_num in iterator` 之前插入：

```python
ss = None
```

然后找到：

```python
if epoch_num == 0 and i_batch == 0:
```

替换为：

```python
if ss is None:
```

这样无论从第 0 轮开始还是从第 200 轮续训，第一次读取 batch 时都会正确计算监督组合。

对应片段应类似：

```python
iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
ss = None

for epoch_num in iterator:
    model.train()
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
        P = model(image_batch, mode='train')

        if not isinstance(P, list):
            P = [P]

        if ss is None:
            n_outs = len(P)
            out_idxs = list(np.arange(n_outs))
            if args.supervision == 'mutation':
                ss = [x for x in powerset(out_idxs)]
            elif args.supervision == 'deep_supervision':
                ss = [[x] for x in out_idxs]
            else:
                ss = [[-1]]
            print(ss)
```

这一步非常重要。只把 `range(max_epoch)` 改成 `range(start_epoch, max_epoch)`，会导致旧代码在续训时出现 `UnboundLocalError: local variable 'ss' referenced before assignment`。

## 六、第四部分：把每轮保存改成完整 checkpoint

当前每轮保存 `last.pth` 的代码大约在第 263 至 267 行：

```python
save_mode_path = os.path.join(snapshot_path, 'last.pth')
torch.save(model.state_dict(), save_mode_path)
```

把它替换为：

```python
def build_checkpoint(epoch_num, best_value):
    # epoch_num 是从 0 开始的内部编号，保存给 checkpoint 时转成人类轮次。
    return {
        'epoch': int(epoch_num + 1),
        'iter_num': int(iter_num),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_performance': float(best_value),
        'args': vars(args),
    }


save_mode_path = os.path.join(snapshot_path, 'checkpoint_latest.pth')
torch.save(build_checkpoint(epoch_num, best_performance), save_mode_path)

# 可选：继续保留只用于测试/推理的裸模型文件。
torch.save(model.state_dict(), os.path.join(snapshot_path, 'last.pth'))
```

### 为什么 `epoch` 要保存为 `epoch_num + 1`

内部循环的 `epoch_num=199` 代表第 200 轮。checkpoint 中保存：

```python
'epoch': 200
```

恢复时直接使用：

```python
start_epoch = int(checkpoint.get('epoch', 0))
```

于是 `range(200, max_epoch)` 就从第 201 轮开始，不会重复第 200 轮。

## 七、第五部分：修改定期保存和最终保存

当前定期保存区块大约在第 289 至 297 行：

```python
if (epoch_num + 1) % save_interval == 0:
    save_mode_path = os.path.join(
        snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
```

建议替换为：

```python
if (epoch_num + 1) % save_interval == 0:
    human_epoch = epoch_num + 1

    # 完整断点：可用于继续训练。
    checkpoint_path = os.path.join(
        snapshot_path, 'checkpoint_epoch_{}.pth'.format(human_epoch))
    torch.save(build_checkpoint(epoch_num, best_performance), checkpoint_path)

    # 裸模型：可继续给测试脚本加载，不包含优化器状态。
    model_path = os.path.join(
        snapshot_path, 'epoch_{}.pth'.format(epoch_num))
    torch.save(model.state_dict(), model_path)

    logging.info("save full checkpoint to %s", checkpoint_path)
    logging.info("save model weights to %s", model_path)
```

这样第 200 轮结束后会同时得到：

```text
checkpoint_epoch_200.pth   # 推荐用于 resume
epoch_199.pth              # 裸模型，仅兼容旧测试/推理流程
```

最终保存区块也要改。找到原来的：

```python
if epoch_num >= max_epoch - 1:
    save_mode_path = os.path.join(
        snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    iterator.close()
    break
```

替换为：

```python
if epoch_num >= max_epoch - 1:
    human_epoch = epoch_num + 1

    final_checkpoint_path = os.path.join(
        snapshot_path, 'checkpoint_epoch_{}.pth'.format(human_epoch))
    torch.save(build_checkpoint(epoch_num, best_performance), final_checkpoint_path)

    final_model_path = os.path.join(
        snapshot_path, 'epoch_{}.pth'.format(epoch_num))
    torch.save(model.state_dict(), final_model_path)

    logging.info("save final full checkpoint to %s", final_checkpoint_path)
    logging.info("save final model weights to %s", final_model_path)
    iterator.close()
    break
```

## 八、`best.pth` 应该如何保存

当前 `best.pth` 用于测试时加载裸模型，建议继续保持裸模型格式：

```python
torch.save(model.state_dict(), save_mode_path)
```

如果希望以后也能从最佳模型继续训练，可以额外保存一个完整文件，但不要覆盖原来的 `best.pth`：

```python
best_resume_path = os.path.join(snapshot_path, 'checkpoint_best.pth')
torch.save(build_checkpoint(epoch_num, best_performance), best_resume_path)
```

不过，恢复“最近训练进度”时仍然应该使用 `checkpoint_latest.pth` 或明确的 `checkpoint_epoch_200.pth`，而不是 `checkpoint_best.pth`。

## 九、`.sh` 启动脚本如何修改

打开：

```text
start_train_synapse.sh
```

### 1. 第一次从头训练

不添加 `--resume`，保持原来的启动命令即可。程序会从第 0 轮开始，并在每轮更新：

```text
checkpoint_latest.pth
```

每 50 轮额外生成：

```text
checkpoint_epoch_50.pth
checkpoint_epoch_100.pth
checkpoint_epoch_150.pth
checkpoint_epoch_200.pth
```

### 2. 使用完整第 200 轮断点继续

在 Python 命令参数末尾加入：

```bash
  --resume "${PROJECT_DIR}/model_pth/run_seed3333/checkpoint_epoch_200.pth" \
```

完整示例：

```bash
nohup env RUN_ID="${RUN_ID}" python train_synapse.py \
  --root_path "${ROOT_PATH}" \
  --list_dir "${LIST_DIR}" \
  --volume_path "${VOLUME_PATH}" \
  --dataset "${DATASET}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --base_lr "${BASE_LR}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --resume "${PROJECT_DIR}/model_pth/run_seed3333/checkpoint_epoch_200.pth" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &
```

### 3. 只有旧的 `epoch_199.pth` 时如何恢复

如果当前只有旧格式文件：

```bash
  --resume "${PROJECT_DIR}/model_pth/run_seed3333/epoch_199.pth" \
```

程序会从第 201 轮开始，但日志中会明确警告：

```text
Legacy model-only checkpoint loaded: optimizer and RNG states were not restored.
```

这表示模型参数接上了第 200 轮，但 AdamW 会重新初始化。对于“尽量救回训练进度”可以接受；对于正式可复现实验，应从头训练一次以产生新的完整 checkpoint，或提前采用本指南的完整保存方案。

## 十、续训时必须保持一致的参数

续训不是普通测试，以下参数不能随意改变：

- `encoder`；
- `num_classes`；
- `kernel_sizes`；
- `expansion_factor`；
- `lgag_ks`；
- `activation_mscb`；
- `supervision`；
- `batch_size`；
- `img_size`；
- 优化器类型和 `weight_decay`；
- 数据集划分和 `train.txt` 内容。

尤其不能把 `batch_size=16` 的旧训练改成 `batch_size=6` 后，直接声称是同一个严格续训。虽然模型参数可能可以加载，但每个 epoch 的 batch 数和 AdamW 的更新轨迹都会改变。

`max_epochs` 可以从原来的 300 保持为 300，表示最终仍训练到第 300 轮。不要把它改成 100 后期待程序自动训练到 300；它是总目标轮数，不是“再训练多少轮”。

## 十一、验证是否真的从第 200 轮恢复

启动后立即查看日志，应该出现类似：

```text
Loading resume checkpoint: /.../checkpoint_epoch_200.pth
Resume successful: start_epoch=200, iter_num=...
```

随后 tqdm 或日志的第一轮应该是：

```text
epoch 200/300
```

这里的内部编号 `200` 对应人类意义上的第 201 轮。若你想让日志显示“第 201 轮”，可以把进度描述从：

```python
desc="epoch {}/{}".format(epoch_num, max_epoch)
```

改成：

```python
desc="epoch {}/{}".format(epoch_num + 1, max_epoch)
```

当前 Synapse 训练代码主要通过 `logging.info` 记录 epoch，建议额外加入：

```python
logging.info(
    'epoch_start human_epoch=%d internal_epoch=%d',
    epoch_num + 1,
    epoch_num,
)
```

放在 `for epoch_num in iterator:` 循环体第一行附近，便于确认续训边界。

## 十二、推荐的文件布局

一个完整实验目录最终可以类似：

```text
model_pth/run_seed3333/
├── checkpoint_latest.pth
├── checkpoint_epoch_50.pth
├── checkpoint_epoch_100.pth
├── checkpoint_epoch_150.pth
├── checkpoint_epoch_200.pth
├── checkpoint_epoch_250.pth
├── checkpoint_epoch_300.pth
├── epoch_49.pth
├── epoch_99.pth
├── epoch_149.pth
├── epoch_199.pth
├── epoch_249.pth
├── epoch_299.pth
├── last.pth
├── best.pth
└── log.txt
```

含义是：

- `checkpoint_latest.pth`：每轮覆盖，训练中断时优先恢复它；
- `checkpoint_epoch_200.pth`：固定保留第 200 轮完整状态；
- `epoch_199.pth`：第 200 轮的裸模型参数，供旧测试脚本兼容；
- `last.pth`：最新裸模型参数；
- `best.pth`：验证 Dice 最好的裸模型参数。

## 十三、保存安全性建议

直接 `torch.save(path)` 在进程被强制杀死或磁盘异常时，可能留下半截文件。更稳妥的做法是先写临时文件，再原子替换目标文件。

在 `trainer.py` 的 import 区增加：

```python
import tempfile
```

然后在 `trainer_synapse()` 前或文件顶部定义：

```python
def atomic_torch_save(obj, path):
    directory = os.path.dirname(os.path.abspath(path))
    fd, temp_path = tempfile.mkstemp(
        prefix='.checkpoint_',
        suffix='.tmp',
        dir=directory,
    )
    os.close(fd)
    try:
        torch.save(obj, temp_path)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
```

将本指南中的：

```python
torch.save(build_checkpoint(epoch_num, best_performance), checkpoint_path)
```

替换为：

```python
atomic_torch_save(
    build_checkpoint(epoch_num, best_performance),
    checkpoint_path,
)
```

裸模型保存也可以使用 `atomic_torch_save`。这样即使保存时进程退出，旧的完整 checkpoint 不会被半截的新文件直接覆盖。

## 十四、一次性测试方法

不要直接拿 300 轮正式实验测试。建议先把 `.sh` 中：

```bash
MAX_EPOCHS=300
```

临时改成：

```bash
MAX_EPOCHS=3
```

先启动一次，确认生成：

```text
checkpoint_epoch_3.pth
```

再把 `MAX_EPOCHS` 改成 `5`，加入：

```bash
--resume ".../checkpoint_epoch_3.pth"
```

确认日志显示：

```text
Resume successful: start_epoch=3
```

并且只训练第 4、5 轮。测试完成后，再恢复正式的 `MAX_EPOCHS=300`。

## 十五、最简执行方案

如果现在已经在第 204 轮停掉，且只有旧文件，先按以下顺序处理：

1. 在 `train_synapse.py` 增加 `--resume` 参数。
2. 在 `trainer.py` 增加 `start_epoch` 和旧裸模型兼容加载逻辑。
3. 把 `if epoch_num == 0 and i_batch == 0` 改成 `if ss is None`，并在循环前设置 `ss = None`。
4. 传入：

   ```bash
   --resume ".../epoch_199.pth"
   ```

5. 保持 `--max_epochs 300`。
6. 确认日志从 `start_epoch=200` 开始。
7. 从这次续训开始采用完整 checkpoint 保存，以后优先使用 `checkpoint_latest.pth` 或 `checkpoint_epoch_200.pth`。

如果还没有开始改代码，最理想的顺序是先完成完整 checkpoint 保存，再重新启动实验。这样以后第 204 轮中断时，直接读取第 200 轮的 `checkpoint_epoch_200.pth`，模型参数、AdamW 状态、轮数和最佳指标都能一起恢复。

## 十六、重要限制

完整 checkpoint 也不能保证在所有硬件、PyTorch、CUDA、数据加载器 worker 状态下逐 batch 完全复现。它能恢复训练状态和轮次，但如果程序是在某个 epoch 中间崩溃，最近一个“已成功写盘”的 checkpoint 仍然是上一个完整保存点。

本项目当前每 50 轮保存一次阶段 checkpoint，因此最坏情况下会重做不到 49 轮。若希望最多只重做 1 轮，建议保留每轮 `checkpoint_latest.pth`，并使用原子保存；如果希望进程在单个 epoch 内中断后也能恢复，则需要把 checkpoint 保存频率进一步改为每若干 batch 保存，并额外保存 DataLoader、batch 位置和随机数状态。

