# 启动输出拆解：`start_train_synapse.sh` 的逐行分析

## 1. 先给结论

截图显示训练已经正常启动，没有出现导致退出的错误。程序已经完成：

1. 加载 EMCAD 所需的 PVTv2-B2 编码器预训练权重；
2. 创建 PVTv2-B2 编码器和 EMCAD 解码器；
3. 读取 Synapse 训练切片，共 2211 张；
4. 创建 batch size 为 20 的 DataLoader；
5. 进入第 0 个 epoch，并完成至少 100 次参数更新。

截图中的 `FutureWarning` 是安全性兼容警告，不是训练失败原因。当前真正需要核对的是：截图里的运行参数和当前服务器上的启动脚本是否是同一版本，因为截图显示 `seed=3333`，而当前项目中的 `start_train_synapse.sh` 写的是 `SEED=2222`。

## 2. 输出顺序总览

截图大致对应下面这条执行链：

```text
Shell 脚本
  -> 加载 Conda 环境
  -> 设置 GPU 和日志
  -> python train_synapse.py
  -> torch.load 读取 PVT 预训练权重
  -> 创建 PVTv2-B2
  -> 创建 EMCAD 解码器
  -> 打印 Namespace 配置
  -> 读取训练数据
  -> 计算每个 epoch 的 batch 数
  -> 前向传播、mutation supervision、反向传播
  -> 每 50 次迭代打印一次 loss
```

## 3. `ready to train`

```text
---------------------------ready to train---------------------------------
```

这行来自 `start_train_synapse.sh`，只是 Shell 脚本在执行 Python 训练命令前打印的分隔标记。

它只说明：

- 项目目录已经切换成功；
- 日志目录已经创建或存在；
- Conda 初始化和环境激活没有在此之前失败；
- 脚本准备执行 Python 训练进程。

它不代表模型已经完成初始化，也不代表训练已经收敛。后面的 Python 输出才是模型和训练状态的依据。

## 4. `torch.load` 的 `FutureWarning`

截图中的核心内容是：

```text
FutureWarning: You are using `torch.load` with `weights_only=False`
```

对应代码是 `lib/networks.py` 中加载 PVT 预训练权重的语句：

```python
save_model = torch.load(path)
```

### 4.1 这条警告在说什么

旧版 PyTorch 的 `torch.load()` 默认允许通过 Python pickle 反序列化完整对象。pickle 文件如果来自不可信来源，加载时可能执行恶意代码。因此 PyTorch 提醒：未来版本可能把默认行为改成更严格的：

```python
torch.load(path, weights_only=True)
```

`weights_only=True` 更适合只加载模型参数的场景。

### 4.2 这是不是报错

不是。判断依据是警告后面继续出现了：

```text
Model pvt_v2_b2 backbone: created, param count: 24849856
Model EMCAD decoder: created, param count: 1913515
Model successfully created.
```

如果是致命错误，Python 通常会出现 `Traceback`，随后进程退出，不会继续创建模型并进入训练循环。

### 4.3 对当前运行的实际影响

如果 `./pretrained_pth/pvt/pvt_v2_b2.pth` 是你自己下载、来源可信的权重，这条警告不会影响本次训练结果。它主要提醒未来升级 PyTorch 或加载不可信 checkpoint 时要改用更安全的加载方式。

不要因为这条警告就删除预训练权重、重装环境或中断当前训练。

## 5. PVTv2-B2 编码器参数量

```text
Model pvt_v2_b2 backbone: created, param count: 24849856
```

含义是：

- 编码器名称：`pvt_v2_b2`；
- 编码器创建成功；
- 编码器包含 `24,849,856` 个可训练参数，约 2485 万个；
- 这个数字是参数个数，不是显存占用，也不是 FLOPs。

在当前 EMCAD 代码中，PVTv2-B2 提供四级特征，通道规模大致为：

```text
[64, 128, 320, 512]
```

解码器会利用这些不同分辨率、不同语义层次的特征恢复分割图。

参数量不能单独决定训练速度。实际速度还受输入尺寸、batch size、GPU 型号、显存带宽、DataLoader 和 CUDA/cuDNN 版本影响。

## 6. EMCAD 解码器参数量

```text
Model EMCAD decoder: created, param count: 1913515
```

含义是：

- EMCAD 解码器创建成功；
- 解码器本身包含 `1,913,515` 个参数，约 191 万个；
- 这行只统计 `self.decoder`，不包含 PVT 编码器；
- 也不包含后面四个 `1×1` 分割输出头的参数。

因此，截图中模型主体参数量约为：

```text
24,849,856 + 1,913,515 ≈ 26,763,371
```

四个输出头还会增加少量参数。对于 Synapse 的 9 类输出，它们的参数量相对整个模型很小。

## 7. `Model successfully created.`

```text
Model successfully created.
```

这行来自 `train_synapse.py`，表示：

1. `EMCADNet(...)` 已成功实例化；
2. PVTv2-B2 编码器和 EMCAD 解码器均已创建；
3. 模型已经执行 `model.cuda()`，被放到 CUDA 设备上；
4. 接下来训练器可以开始创建 DataLoader、优化器并执行前向传播。

它仍然不等于“训练效果好”，只表示模型初始化阶段通过了。

## 8. `Namespace(...)`：本次训练的完整配置快照

截图中这一大段：

```text
Namespace(root_path='/data/Synapse/train_npz', ...)
```

是 Python `argparse` 解析后的全部参数。可以把它理解为“本次训练实际拿到的配置清单”。下面按功能拆开。

### 8.1 数据路径

```text
root_path='/data/Synapse/train_npz'
volume_path='../data/Synapse/test_vol_h5'
list_dir='/root/shared-nvme/lzq_MedicalImageSegmentation/SLDGroup_EMCAD/../data/Synapse/lists/lists_Synapse'
dataset='Synapse'
```

含义：

- `root_path`：二维训练切片目录，通常包含 `.npz` 文件；
- `volume_path`：完整体数据目录，用于病例级验证/测试；
- `list_dir`：数据划分清单目录，通常需要 `train.txt`、`test_vol.txt` 等文件；
- `dataset='Synapse'`：选择 Synapse 数据集配置。

这里有一个值得注意的路径现象：`root_path` 是绝对路径 `/data/Synapse/train_npz`，而 `volume_path` 是相对路径 `../data/Synapse/test_vol_h5`。相对路径会按照 Python 进程启动时的当前工作目录解析。当前脚本先 `cd` 到项目目录，所以它通常会解析到项目上一级的 `data/Synapse/test_vol_h5`。

只要训练集已经成功读取到 2211 张切片，`root_path` 当前就是可用的。验证能否正常运行，还要在第一个 epoch 结束时观察是否能成功读取 `test_vol_h5`。

### 8.2 类别数

```text
num_classes=9
```

Synapse 当前是多类器官分割：

- 类别 0：背景；
- 类别 1 到 8：8 个目标器官；
- 网络每个输出头产生 9 个通道；
- 使用多类 Cross Entropy 和多类 Dice loss；
- 训练时输出是 logits，通常不在模型内部直接做 softmax。

因此这里的 9 不是 9 个前景器官，而是“背景加 8 个前景类别”。

### 8.3 编码器与 EMCAD 结构参数

```text
encoder='pvt_v2_b2'
expansion_factor=2
kernel_sizes=[1, 3, 5]
lgag_ks=3
activation_mscb='relu6'
no_dw_parallel=False
concatenation=False
```

对应关系如下：

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `encoder` | `pvt_v2_b2` | 选择 PVTv2-B2 编码器 |
| `expansion_factor` | `2` | 控制 MSCB 中间通道扩张宽度 |
| `kernel_sizes` | `[1,3,5]` | MSDC 使用的多尺度深度卷积核 |
| `lgag_ks` | `3` | LGAG 局部门控卷积核大小 |
| `activation_mscb` | `relu6` | MSCB 使用 ReLU6 激活 |
| `no_dw_parallel=False` | 并行开启 | 多尺度深度卷积分支并行执行 |
| `concatenation=False` | 加法聚合 | MSDC 分支采用逐元素相加，而不是通道拼接 |

这些参数决定的是 EMCAD 的结构配置和消融变量，不是数据路径或训练是否启动的开关。

### 8.4 预训练权重

```text
no_pretrain=False
pretrained_dir='./pretrained_pth/pvt/'
```

含义：

- `no_pretrain=False`：不关闭预训练，因此会尝试加载 PVTv2-B2 权重；
- `pretrained_dir`：预训练权重目录；
- 这正是前面 `torch.load` 警告出现的原因。

如果预训练权重路径错误，通常会出现 `FileNotFoundError`，而截图中模型已成功创建，说明当前加载过程没有导致失败。

### 8.5 监督方式：`mutation`

```text
supervision='mutation'
```

EMCAD 在当前实现中会产生四个分割输出。`mutation` 监督会枚举四个输出的所有非空组合，并将组合中的 logits 逐元素相加后计算损失。

四个输出共有：

```text
2^4 = 16 个子集
```

去掉空集后，实际参与损失计算的是 15 组。截图紧接着打印的列表：

```text
[[], [0], [1], [2], [3], [0, 1], ... , [0, 1, 2, 3]]
```

正好证明当前使用的是 mutation supervision，而不是只监督最后一个输出。

一个重要后果是：当前 `loss` 是 15 组损失直接相加，没有除以 15。因此 `mutation` 下的 loss 数值天然可能比 `deep_supervision` 或 `last_layer` 大，不能直接拿不同监督策略的原始 loss 数字横向比较。

### 8.6 训练轮数和迭代参数

```text
max_iterations=50000
max_epochs=300
batch_size=20
base_lr=0.0001
img_size=256
```

各参数含义：

- `max_epochs=300`：外层训练循环运行 300 个 epoch；
- `batch_size=20`：每次从训练集取 20 张二维切片；
- `base_lr=0.0001`：AdamW 学习率为 `1e-4`；
- `img_size=256`：切片被调整到 256×256；
- `max_iterations=50000`：在当前 `trainer.py` 中主要用于实验目录命名，不控制真实训练循环的终止。

当前训练器真正使用的是：

```python
max_iterations = args.max_epochs * len(trainloader)
```

所以判断实际训练总步数，应看后面打印的 `33300 max iterations`，而不是只看 `max_iterations=50000`。

### 8.7 GPU、确定性和随机种子

```text
n_gpu=1
deterministic=1
seed=3333
z_spacing=1
```

- `n_gpu=1`：按单 GPU 训练配置；
- `deterministic=1`：关闭 cuDNN benchmark，优先保证可复现性；
- `seed=3333`：固定 Python、NumPy、PyTorch 等随机数种子；
- `z_spacing=1`：当前配置中用于体数据处理/保存的 z 方向间距。

### 8.8 截图与当前启动脚本的参数不一致

当前项目工作区中的 `start_train_synapse.sh` 写的是：

```bash
SEED=2222
```

但截图显示：

```text
seed=3333
```

同时，当前脚本定义的训练集路径是：

```bash
ROOT_PATH="../data/Synapse/train_npz"
```

截图却显示：

```text
root_path='/data/Synapse/train_npz'
```

这说明至少存在一种情况：

- 服务器上的脚本不是当前本地文件的同一版本；
- 实际执行的是 `start_train_synapse_AIStation.sh` 或其他副本；
- 服务器脚本被手动修改过；
- 启动时额外传入了不同参数；
- 当前截图来自另一次运行。

如果你要复现实验，必须以服务器实际运行时打印出的 `Namespace` 为准，并把它保存进实验记录。不要仅凭本地脚本推断服务器实际配置。

## 9. 训练集长度：2211

```text
The length of train set is: 2211
```

这表示训练 DataLoader 的底层数据集包含 2211 个二维训练样本，通常对应 2211 个 `.npz` 切片。

它不是：

- 2211 个患者；
- 2211 个 3D CT 体积；
- 2211 个 batch。

Synapse 的训练通常是“二维切片训练、三维病例级验证”。因此这里的 2211 应理解为切片数量。

## 10. 每个 epoch 的迭代数：111

```text
111 iterations per epoch. 33300 max iterations
```

训练集有 2211 张切片，batch size 为 20：

```text
ceil(2211 / 20) = ceil(110.55) = 111
```

最后一个 batch 可能只有 11 张切片，因为当前 DataLoader 没有设置 `drop_last=True`。

真实总更新次数为：

```text
111 batches/epoch × 300 epochs = 33300 updates
```

因此这行是合理的，也说明训练器确实读到了 2211 个样本，并采用了 batch size 20。

## 11. tqdm 进度条：`0/300`

```text
0%|          | 0/300 [00:00<?, ?it/s]
```

它表示外层 epoch 进度：

- 总共 300 个 epoch；
- 当前显示在第 0 个 epoch；
- 不是 0 个 batch；
- 不是模型没有工作。

因为 epoch 编号在代码中从 0 开始，所以日志里的 `epoch 0` 实际是第 1 轮训练。

## 12. mutation 输出组合列表

截图中的列表是：

```text
[[], [0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]]
```

四个输出编号为 0、1、2、3。列表中的每个元素代表一次监督组合，例如：

- `[0]`：只使用第 0 个输出；
- `[0, 1]`：将第 0、1 个输出的 logits 相加后计算损失；
- `[0, 1, 2, 3]`：将四个输出全部相加后计算损失；
- `[]`：空组合，训练代码会跳过，不产生损失。

因此这行是正常的初始化信息，不是异常堆栈。

## 13. 第 50 次和第 100 次迭代日志

截图中出现：

```text
iteration 50, epoch 0 : loss : 8.232925, lr: 0.000100
iteration 100, epoch 0 : loss : 5.578654, lr: 0.000100
```

### 13.1 `iteration 50`

表示已经完成第 50 次 batch 更新。由于每个 epoch 有 111 个 batch，此时仍在第 1 个 epoch 内。

### 13.2 `loss`

这是当前 batch 的 mutation 总损失，即最多 15 组监督损失的加权和：

```text
单组损失 = 0.3 × CrossEntropyLoss + 0.7 × DiceLoss
总损失 = 所有非空输出组合的单组损失相加
```

因此 `8.232925` 或 `5.578654` 不能直接当成最终 Dice，也不能单独证明模型已经达到论文水平。

从第 50 次到第 100 次下降，说明当前训练批次上的优化至少在起作用，但这只是非常早期的训练信号。真正需要关注的是：

- 每个 epoch 结束后的验证 Dice；
- best Dice 是否持续提高；
- 各器官的 Dice 是否极不平衡；
- HD95、ASSD 等边界指标；
- 训练集和验证集之间是否出现明显过拟合。

### 13.3 `lr: 0.000100`

表示当前学习率为 `1e-4`。当前 `trainer.py` 中多项式衰减公式被注释掉，学习率会保持为常数 `0.0001`，不会按照迭代次数自动下降。

## 14. 当前运行是否正常：逐项判定

| 检查项 | 截图结果 | 判断 |
|---|---|---|
| Shell 到 Python | 已进入 Python 输出 | 正常 |
| 预训练权重加载 | 出现 warning 后继续 | 正常，只有兼容性警告 |
| PVT 编码器 | 24849856 参数并创建成功 | 正常 |
| EMCAD 解码器 | 1913515 参数并创建成功 | 正常 |
| CUDA 模型初始化 | `Model successfully created` | 正常 |
| 训练数据读取 | 2211 张切片 | 正常 |
| DataLoader | 111 batch/epoch | 与 2211、batch 20 一致 |
| mutation 监督 | 打印 16 个子集 | 正常，空集会被跳过 |
| 反向更新 | 已打印 iteration 50、100 | 正常，至少完成 100 次更新 |

结论：截图对应的是“已经启动并正在训练”，不是“启动报错”。

## 15. 你现在应该观察什么

训练继续运行时，重点看以下输出：

1. 第一个 epoch 结束时是否进入验证阶段；
2. 是否出现 `save model to .../best.pth`；
3. 是否能读取 `test_vol_h5` 中的完整病例；
4. 是否出现 `CUDA out of memory`、`FileNotFoundError`、`KeyError` 或 `Traceback`；
5. 后续 epoch 的验证 Dice 是否有数值，而不是全部为 0 或 `nan`；
6. `last.pth`、`best.pth` 和 TensorBoard 日志是否写入预期目录。

当前代码的保存目录是：

```text
model_pth/run_seed3333/
```

如果使用当前本地脚本中的 seed 2222，则目录通常会是：

```text
model_pth/run_seed2222/
```

最终以服务器日志中实际创建的路径为准。

## 16. 服务器上建议执行的只读核对命令

为了确认截图对应的到底是哪份脚本和哪组参数，可以在服务器项目目录执行：

```bash
pwd
grep -nE 'SEED=|ROOT_PATH=|VOLUME_PATH=|BATCH_SIZE=|MAX_EPOCHS=|--seed|--root_path|--volume_path' start_train_synapse.sh
grep -nE 'SEED=|ROOT_PATH=|VOLUME_PATH=|BATCH_SIZE=|MAX_EPOCHS=|--seed|--root_path|--volume_path' start_train_synapse_AIStation.sh
ps -ef | grep '[t]rain_synapse.py'
```

其中：

- `pwd` 确认项目根目录；
- `grep` 对比两个启动脚本的真实参数；
- `ps` 查看当前运行进程的命令行；
- 这些命令只读，不会中断训练，也不会修改项目文件。

## 17. 最终判断

这次 `./start_train_synapse.sh` 的输出表明：

```text
环境加载成功
模型创建成功
数据读取成功
训练循环开始
反向传播已经执行
```

唯一明显的“提示”是 `torch.load(..., weights_only=False)` 的未来兼容性警告，它不是当前失败原因。实验记录中应特别保存截图里的完整 `Namespace`，因为它和当前本地脚本的 `seed`、训练路径存在差异；复现实验时，实际运行参数比脚本文件名更可靠。
