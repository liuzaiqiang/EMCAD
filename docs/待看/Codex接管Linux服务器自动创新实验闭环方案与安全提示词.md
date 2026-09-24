# Codex 接管 Linux 服务器自动创新实验闭环：可行性、安全方案与完整提示词

> 适用项目：当前 EMCAD 医学图像分割项目  
> 主要任务：Synapse 多器官分割、自动生成候选改进、修改 PyTorch 代码、训练、验证、筛选与最终测试  
> 文档日期：2026-08-28  
> 重要说明：本文是在设计方案和提示词。本次并没有连接你的 Linux 服务器，也没有修改服务器代码、数据或训练任务。

---

## 1. 先给结论：可行，但不能按“无限自动试到测试集过线”为目标

可以做，而且这类工作非常适合用 Codex 加速，但正确的定义应该是：

> **在固定的数据划分、实验预算和服务器权限内，由 Codex 执行一个有边界、可审计、可停止、可恢复的自动实验闭环。**

它可以完成以下工作：

1. 通过已经安全配置好的 SSH 主机别名连接 Linux 服务器；
2. 只读审计项目、环境、数据划分、GPU 状态和现有结果；
3. 修复影响实验可信度的训练基础设施问题；
4. 在独立 Git 分支或 worktree 中提出并实现候选改进；
5. 执行语法检查、前向/反向冒烟测试、短程代理实验和完整训练；
6. 解析日志和指标，按事先写死的规则决定“淘汰、晋级、停止”；
7. 保存每次实验的代码差异、命令、配置、环境、日志、指标、checkpoint 和失败原因；
8. 达到**验证集**停止条件后冻结代码，再执行预先登记的最终测试；
9. 在 GPU 小时、候选数量、磁盘空间或连续失败次数达到上限时自动停止。

但不能诚实地承诺以下事情：

- 不能保证每一轮都能产生真正达到论文水平的创新；
- 不能保证一定超过某个公开论文指标；
- 不能只靠一句提示词，在 Codex 会话结束后永久、无限地继续“思考下一轮”；
- 不能把不断查看最终测试集并继续改模型包装成正规科研；
- 不能只凭 Dice 上升就证明方法具有新颖性、普适性或论文价值；
- 不能在没有系统文献检索时声称“首次提出”“从未有人做过”。

训练程序可以借助 Slurm、`tmux` 或 `nohup` 在 Codex 会话结束后继续运行，但这只表示**训练进程仍在运行**，不表示 Codex 仍在后台持续阅读结果、发明新结构并提交下一轮任务。若要跨会话自动开展多轮决策，还需要服务器端实验控制器、任务调度器，或你当前 Codex 客户端确实支持并已经配置好的定时/持续任务能力。这个能力必须在实际客户端中验证，不能靠提示词凭空获得。

---

## 2. 你提出的 `Dice > 83.63，连续两次就收手` 有什么问题

### 2.1 `83.63` 不是一个脱离实验协议就能使用的通用门槛

本项目附带的 EMCAD 论文给出了以下结果：

| 项目 | 论文设置或结果 |
|---|---:|
| 模型 | PVT-EMCAD-B2 |
| 数据集 | Synapse Multi-organ |
| 主表平均 Dice | 83.63% |
| 输入尺寸 | 224 x 224 |
| 训练轮数 | 300 epochs |
| batch size | 6 |
| 优化器 | AdamW |
| 学习率 | 1e-4 |
| weight decay | 1e-4 |
| 损失 | 0.3 x Cross-Entropy + 0.7 x Dice |
| 重复次数 | 论文说明 EMCAD 结果取 5 次运行平均 |

论文附录的输入分辨率消融进一步报告：

| PVT-EMCAD-B2 输入尺寸 | 平均 Dice |
|---:|---:|
| 224 x 224 | 83.63% |
| 256 x 256 | 84.47% |
| 384 x 384 | 85.78% |
| 512 x 512 | 86.53% |

而当前仓库的 `start_train_synapse.sh` 使用的是：

```text
IMG_SIZE=256
BATCH_SIZE=20
MAX_EPOCHS=300
BASE_LR=1e-4
DETERMINISTIC=1
SEED=2222
```

因此，当前脚本与论文主结果至少在输入尺寸和 batch size 上不一致。若继续使用 256 x 256，却把 224 x 224 的 `83.63%` 当作创新停止线，原始 EMCAD 基线本来就可能超过这条线，代理会把“复现了基线”误判成“创新成功”。

正确做法只能二选一：

1. **论文复现轨道**：严格采用 224 x 224、batch size 6、300 epochs，以及相同的数据划分、预处理、预训练权重、损失和评价实现，先复现 83.63% 附近的五次运行均值；
2. **本地探索轨道**：继续采用 256 x 256，但重新运行当前代码的本地基线，以同一服务器、同一数据划分和同一组 seed 的结果作为比较基准，不能再把 83.63% 当作等价基线。

### 2.2 代码中的 Dice 和论文百分数可能相差 100 倍

当前训练验证路径通过 MedPy 返回 0 到 1 范围的 Dice。例如：

```text
代码日志中的 0.8363 = 论文表格中的 83.63%
```

自动停止器必须显式保存 `dice_fraction` 和 `dice_percent` 两个字段，禁止一处使用 `0.8363`、另一处使用 `83.63`，否则很容易永远不停止或一开始就错误停止。

### 2.3 最终测试集不能充当自动搜索反馈

如果流程是：

```text
改模型 -> 训练 -> 看最终测试 Dice -> 没超过 83.63 -> 再改模型 -> 再看最终测试 Dice
```

那么最终测试集已经参与了模型选择。即使没有对测试集做反向传播，反复根据测试结果决定下一次结构，也是在人工或自动地对测试集过拟合。这样得到的最终测试成绩会偏乐观，不能再被视为独立泛化证据。

正确的数据流是：

```text
训练集：更新模型参数
验证集：选择 checkpoint、筛选候选结构、决定是否停止开发
最终测试集：代码、配置、seed 和分析方案冻结后才运行
```

### 2.4 “连续两次”应理解为两个预注册 seed，而不是同一 seed 重跑两遍

推荐定义：

> 同一个冻结的 Git commit、同一份配置、同一数据划分，在两个事先写入配置且互不相同的 seed 上，均达到验证门槛。

不推荐以下定义：

- 同一个 deterministic seed 重复两遍；
- 两个不同的候选模型各超过一次；
- 第一次没过后临时挑一个“更容易过”的 seed；
- 两次之间改学习率、数据增强、输入尺寸或代码；
- 在最终测试集上连续试到两次超过阈值。

两次通过只能作为**开发停止门槛**，不能替代论文中的多 seed 统计。正式结果仍建议按照原论文口径至少运行 5 个预注册 seed，报告 `mean +/- std`、HD95、各器官 Dice、逐病例结果、参数量和 FLOPs。

---

## 3. 当前 EMCAD 代码在自动搜索前必须解决的问题

以下结论来自当前工作区代码，不是泛泛而谈。

### 3.1 每个 epoch 都读取 `test_vol` 并据此保存 `best.pth`

`trainer.py` 中的 `inference()` 使用：

```python
Synapse_dataset(base_dir=args.volume_path, split="test_vol", ...)
```

训练循环每个 epoch 调用它，并根据返回的 mean Dice 覆盖 `best.pth`。如果 `test_vol.txt` 和 `volume_path` 对应的就是最终测试病例，那么现有代码已经让测试集参与了 checkpoint 选择。

自动实验开始前必须明确：

- `test_vol` 到底是开发验证集，还是论文最终测试集；
- `train.txt`、`val_vol.txt`、`test_vol.txt` 分别包含哪些病例；
- 切片级训练文件能否追溯到病例，是否存在同一病例切片跨集合泄漏；
- 最终测试病例列表在开发阶段是否能被实验控制器读取。

推荐新增真正独立的验证列表和参数，例如 `--val_volume_path`、`--val_list`，并让最终测试入口只在冻结后调用。

### 3.2 第一次验证后没有切回 `model.train()`

当前流程只在进入 epoch 循环前执行一次 `model.train()`。每轮验证中的 `inference()` 会执行 `model.eval()`，但下一 epoch 开始前没有再次执行 `model.train()`。

这意味着从第 2 个 epoch 开始，训练可能一直处于 eval 模式：

- BatchNorm 不再更新运行统计；
- Dropout 若存在则不再按训练模式工作；
- 得到的训练行为可能与论文、常规 PyTorch 训练和预期基线不一致。

必须在每个训练 epoch 开始前显式调用 `model.train()`，并用测试确认训练/验证模式切换正确。

### 3.3 checkpoint 不能完整恢复训练

当前 `last.pth`、`best.pth` 和 epoch checkpoint 只保存：

```python
model.state_dict()
```

没有保存：

- 当前 epoch 和全局 iteration；
- optimizer 状态；
- scheduler 状态（若后续引入）；
- AMP GradScaler 状态（若后续引入）；
- 历史 best metric；
- Python、NumPy、PyTorch CPU/CUDA 随机数状态；
- 完整实验配置和数据清单哈希。

服务器断电、任务超时或进程中断后，即使重新加载模型权重，也不等于从原训练状态无缝继续。自动闭环必须先实现完整 checkpoint 和 `--resume`，并做一次“训练若干步 -> 保存 -> 重启进程 -> 恢复”的等价性验证。

### 3.4 当前评价代码的空类别约定需要核对

`utils/utils.py` 当前把“预测有该类、真值没有该类”的情况返回为 `Dice=1`，而“预测和真值都没有该类”会落入返回 `Dice=0` 的分支。这与许多常见评价约定相反。

如果 Synapse 的某些病例确实可能缺少某个器官，该规则可能明显影响平均值。不能直接悄悄改掉，因为改评价规则后将无法与旧日志直接比较；正确做法是：

1. 核对每个测试病例是否都包含全部 8 个前景器官；
2. 明确论文或原始实现采用的空类别约定；
3. 把评价协议写入配置并增加单元测试；
4. 若修正规则，重新跑基线，旧指标标记为不可直接比较。

### 3.5 先建立可信 baseline，再谈自动创新

如果直接让代理在当前流程上不断改结构，它可能把以下现象误判为“创新提升”：

- `eval()` 模式训练造成的偶然差异；
- 测试集参与选模造成的乐观偏差；
- 输入尺寸、batch size 或 seed 改变；
- 空类别评价规则的变化；
- 预训练权重是否正确加载；
- 数据列表变化或病例泄漏；
- 中断后错误恢复造成的训练轨迹变化。

所以第一项“创新”不应是模型模块，而应是把实验基础设施修到可复现、可恢复、无测试泄漏。该修复属于科研基础设施，不应包装成论文创新点。

---

## 4. 推荐的可信实验协议

### 4.1 固定两个实验轨道，不要混用

#### 轨道 A：论文复现轨道（推荐先做）

```text
encoder: PVTv2-B2
input_size: 224
batch_size: 6
epochs: 300
optimizer: AdamW
lr: 1e-4
weight_decay: 1e-4
loss: 0.3 CE + 0.7 Dice
pretrain: ImageNet PVTv2-B2
```

目标是确认本地数据、代码和环境能够得到与论文合理接近的分布。它不是创新实验。

#### 轨道 B：本地 256 探索轨道

可以使用当前脚本的 256 x 256，但必须先在同一协议下重新获得本地 baseline。所有候选与该 baseline 使用相同数据划分、seed、训练轮数、batch size、预训练权重、损失和评价实现。

轨道 A 的 83.63 与轨道 B 的结果不能无条件横向比较。

### 4.2 固定病例级划分并生成不可变清单

至少保存：

```text
splits/train_cases.txt
splits/val_cases.txt
splits/test_cases.txt
splits/manifest.json
```

`manifest.json` 应记录：

- 数据集版本和来源；
- 每个病例 ID；
- 各集合病例数；
- 每份列表文件的 SHA-256；
- 预处理脚本 Git commit；
- 原始体数据到训练切片的映射；
- 生成日期和固定 seed；
- 是否包含所有 8 个前景类别。

最终测试列表在开发阶段最好由独立测试脚本持有，自动候选搜索器只获得训练和验证入口。

### 4.3 建立配对 baseline

候选和 baseline 应使用相同 seed 做配对比较。例如预先登记：

```text
开发 seed: 2222, 3407
最终报告 seed: 2222, 3407, 6666, 7777, 8888
```

seed 本身没有魔法，关键是**在看到结果前写死，之后不挑 seed**。

对每个 seed，保存：

```text
delta_dice_pp(seed) = candidate_val_dice_percent(seed) - baseline_val_dice_percent(seed)
```

这样可以减少“候选刚好遇到一个幸运 seed”的误判。

### 4.4 推荐的分级实验预算

| 阶段 | 目的 | 建议做法 | 是否允许看最终测试集 |
|---|---|---|---|
| 冒烟测试 | 排除代码错误 | 小张量前向、反向、shape、NaN、显存 | 否 |
| 代理实验 | 快速淘汰明显无效候选 | 固定训练子集/轮数，所有候选预算相同 | 否 |
| 单 seed 完整实验 | 判断是否值得晋级 | 完整训练，固定开发 seed | 否 |
| 双 seed 确认 | 判断是否停止搜索 | 同 commit、同配置、两个预注册 seed | 否 |
| 五 seed 最终实验 | 论文报告 | 冻结后运行并全部报告 | 是，仅此阶段 |

代理实验只能用于淘汰，不应把短程排名直接当作最终结论。某些结构早期收敛快但最终不一定更好，反之亦然。

### 4.5 推荐停止条件

建议把停止条件拆成三个层次。

#### A. 开发成功门槛

设：

```text
B_s = baseline 在验证集、seed=s 的 Dice 百分数
C_s = 候选在验证集、seed=s 的 Dice 百分数
delta_s = C_s - B_s
```

推荐门槛：

```text
同一个候选 Git commit；
两个预注册开发 seed 均完成；
两个 seed 的 delta_s 均 >= 0.30 个 Dice 百分点；
两个 seed 的验证 Dice 均达到预先登记的绝对门槛（如果设置了）；
平均 HD95 不显著恶化；
没有 NaN、空预测、病例失败或数据泄漏；
参数量、FLOPs、显存和训练时间没有超过事先预算。
```

`0.30` 只是一个较实用的初始示例。更严谨的取值应根据本地 baseline 的 seed 波动来定，例如：

```text
MIN_DELTA_PP = max(0.30, baseline_seed_std_pp)
```

#### B. 自动循环停止门槛

满足任一条件立即停止产生新候选：

- 某候选通过开发成功门槛；
- 达到最大候选数量；
- 达到最大 GPU 小时；
- 达到最大磁盘占用；
- 连续 3 个候选在冒烟或训练中因同一基础问题失败；
- 发现数据泄漏、评价口径不明、baseline 不可复现；
- GPU 出现硬件错误、文件系统异常或 checkpoint 持续损坏；
- 需要 sudo、删除数据、修改系统服务或处理不属于本项目的进程。

#### C. 论文最终确认

开发成功后：

1. 冻结 Git commit、配置、数据清单和 5 个 seed；
2. 生成不可变的 `frozen_manifest.json`；
3. 才允许调用最终测试入口；
4. 运行全部 5 个 seed，不能只保留过线的运行；
5. 报告 mean +/- std、每个 seed、每个器官和每个病例；
6. 最终测试结果无论好坏，都不得再反馈给自动结构搜索；
7. 如果最终测试不理想，只能如实报告，或重新开始一个新的、明确标记的研究阶段并使用新的独立测试方案。

### 4.6 如果你坚持使用 `83.63` 和“两次过线”

最低限度应改写为：

> 在 224 x 224 论文复现协议下，对同一个冻结候选 commit，使用预先登记的两个不同 seed，在固定开发验证集上均达到 `Dice > 83.63%`，并且相对同 seed baseline 均提升至少 `0.30` 个百分点，则停止产生新候选。之后冻结全部配置，使用预注册的 5 个 seed执行最终测试并报告全部结果。最终测试结果不再用于继续改模型。

但仍要明白：论文的 83.63 是其测试协议下五次运行的平均值，不是你本地验证集天然应达到的阈值。最科学的门槛依然是“相对本地同协议 baseline 的配对提升”。

---

## 5. 自动实验闭环应该怎样工作

推荐使用以下状态机，而不是让代理无规则地改代码：

```text
只读审计
  -> 修复实验基础设施
  -> 冻结可信 baseline commit
  -> 多 seed 复现 baseline
  -> 生成候选创新说明
  -> 候选质量门控
  -> 独立 worktree 实现
  -> 静态检查与冒烟测试
  -> 固定预算代理实验
  -> 单 seed 完整实验
  -> 双 seed 配对确认
  -> 达标：冻结候选
  -> 最终 5 seed 测试
  -> 生成论文证据包并停止
```

如果某一步不通过，只能返回到允许的前一步。例如，候选冒烟失败可以修候选实现；但最终测试不理想不能返回“继续根据测试集发明模块”。

### 5.1 每轮只改变一个核心变量

“一个核心变量”不是说只能改一行代码，而是同一轮只能检验一个清晰假设。例如：

```text
可以：为了减少浅层 skip 中的背景噪声，新增一个具有明确输入/输出的门控机制。
不可以：同时换 backbone、输入尺寸、loss、增强、优化器和解码器，再把提升都归因于新模块。
```

所有非核心设置必须与 baseline 相同。若实现一个模块必须配套改动若干文件，应在报告中说明这些改动都属于同一机制。

### 5.2 创新点必须先通过质量门控

每个候选至少回答：

1. EMCAD 当前究竟存在什么可观察问题？
2. 这个问题由失败病例、特征统计、梯度、边界误差还是文献空白支持？
3. 新机制如何针对该问题，而不是随机堆模块？
4. 输入、输出、张量形状和插入位置是什么？
5. 与已有方法最相似的工作有哪些？实质差异是什么？
6. EMCAD 原论文是否已经做过相同消融？
7. 预计增加多少参数、FLOPs、显存和延迟？
8. 最可能失败的原因是什么？
9. 哪个消融可以证明提升确实来自该机制？
10. 如果 Dice 不升，这个实验仍能回答什么科研问题？

EMCAD 原论文已经讨论或消融过：

- decoder 组件组合；
- 多尺度 kernel 组合；
- 深度卷积的串行/并行排列；
- LGAG 与传统 AG；
- 预训练权重；
- deep supervision；
- 输入分辨率。

仅仅再试一次这些已有维度，通常不能直接声称是新创新。

### 5.3 建议的候选评分表

| 维度 | 分值 | 判断标准 |
|---|---:|---|
| 问题证据 | 20 | 有失败病例、统计或明确结构局限支持 |
| 机制合理性 | 20 | 机制与问题之间存在可解释因果链 |
| 与既有工作的区别 | 20 | 已检索相近方法，并能说明实质差异 |
| 可证伪性和消融设计 | 15 | 能用控制实验验证，而不是只看总分 |
| 工程可行性 | 10 | 能在当前代码和预算中可靠实现 |
| 效率价值 | 10 | 参数、FLOPs、显存、延迟代价合理 |
| 医学分割意义 | 5 | 对边界、小器官、尺度变化等有明确意义 |

建议总分低于 75/100 的候选不进入训练。没有联网文献检索或用户提供文献库时，“与既有工作的区别”只能标记为待核实，不能写成“首次提出”。

---

## 6. 你需要提供哪些信息

### 6.1 绝对不要提供的内容

不要在聊天中粘贴：

- SSH 私钥全文；
- Linux 登录密码；
- root 密码；
- API Key、GitHub Token、云厂商 Token；
- 数据库密码；
- 含患者隐私的数据样本；
- `~/.ssh` 目录的完整打包文件。

更安全的做法是：你在本机自行配置 SSH key 和 `~/.ssh/config`，确认下面的命令能够登录：

```bash
ssh <主机别名>
```

然后只告诉 Codex **SSH 主机别名**，而不是私钥和密码。

### 6.2 连接与身份信息

- SSH 主机别名；
- IP/域名和端口（若别名尚未配置）；
- 普通用户名；
- 由服务器管理员或控制台确认的主机公钥指纹；
- 是否需要跳板机/堡垒机；
- 是否只能通过校园网或 VPN；
- 允许登录的时间窗口；
- 是否使用 Slurm、PBS，还是直接使用单机 GPU。

注意：主机指纹应通过管理员、云控制台或其他可信通道核对。`ssh-keyscan` 只能获取网络端返回的指纹，不能单独证明它没有被中间人替换。

### 6.3 项目与 Git 信息

- 服务器上的项目绝对路径；
- 基础模型的 Git 仓库地址；
- baseline 分支和精确 commit；
- 是否存在未提交修改；
- 哪些修改属于你的重要工作，不能覆盖；
- 是否允许创建本地分支和 Git worktree；
- 是否允许 commit；
- 是否禁止 push（推荐默认禁止自动 push）；
- 预训练权重的绝对路径和 SHA-256。

### 6.4 Python、PyTorch 和 GPU 信息

- 操作系统版本；
- Conda/Mamba 可执行文件路径；
- 环境名；
- Python、PyTorch、torchvision、CUDA runtime、cuDNN 版本；
- GPU 型号、数量和显存；
- 允许使用的 GPU 编号；
- 同一 GPU 是否有其他人的任务；
- 最大并发任务数；
- 单任务最大运行时长；
- 是否允许混合精度；
- 是否允许 `torch.compile`；
- 是否有已知驱动、NCCL 或显存问题。

### 6.5 数据和评价协议

- Synapse 原始数据路径；
- 训练 `.npz` 路径；
- 验证/测试 `.npy.h5` 路径；
- `train.txt`、验证列表、`test_vol.txt` 的路径；
- 病例级 train/val/test 清单；
- 预处理脚本和生成参数；
- 类别编号与 8 个器官名称；
- Dice、HD95 的准确实现和空类别规则；
- 指标是 `[0,1]` 还是百分数；
- 当前可信 baseline 的命令、日志、checkpoint 和结果；
- 是否已有论文协议的 5 seed 结果。

### 6.6 资源预算和停止预算

- 最大候选数，例如 6 个；
- 最大 GPU 小时，例如 72 GPU-hours；
- 最大并发数，建议初始为 1；
- 最大实验磁盘，例如 200 GB；
- checkpoint 保留策略；
- 代理实验轮数；
- 完整训练轮数；
- OOM 最多允许自动重试几次；
- 连续失败多少次后停机；
- 服务器维护时间；
- 达标后是否立即停止所有尚未启动的候选。

### 6.7 文献和论文目标

- 目标会议/期刊及截止日期；
- 是否允许联网检索论文；
- 你已经阅读或必须比较的方法；
- 允许使用的代码许可证；
- 论文更看重精度、效率、小器官、边界还是部署；
- 可接受的参数量、FLOPs 和推理延迟增幅；
- 是否需要第二数据集验证泛化。

仅在 Synapse 上搜索并得到提升，通常不足以证明方法具有广泛泛化性。最终论文至少应考虑第二个适合该机制的数据集，且同样保持独立测试协议。

---

## 7. 推荐的最小权限服务器方案

### 7.1 账号权限

推荐为自动实验使用专用普通账号：

- 无 root；
- 无 sudo，或 sudo 权限为空；
- 只对项目 worktree、实验输出和指定缓存目录有写权限；
- 数据集原始目录只读；
- 无权修改 `/etc`、系统服务、其他用户目录和其他人的任务；
- 设置磁盘 quota、GPU/CPU/内存和运行时长限制；
- 若有 Slurm，优先通过队列申请资源，而不是绕过调度器直接占卡。

### 7.2 SSH 建议

本机 SSH 配置可以类似：

```sshconfig
Host emcad-exp
    HostName <服务器IP或域名>
    User <普通用户名>
    Port 22
    IdentityFile <本机专用私钥路径>
    IdentitiesOnly yes
    StrictHostKeyChecking yes
```

私钥应保留在本机，由 SSH agent 或系统安全存储管理。不要让提示词读取或打印私钥。若出现 host key changed，代理必须停止并让你通过可信渠道核对，不能自动删除 `known_hosts` 记录。

### 7.3 文件隔离

推荐目录：

```text
<PROJECT_ROOT>/                 # 只作为受保护 baseline 工作区
<WORKTREE_ROOT>/baseline-fixed/ # 可信基础设施版本
<WORKTREE_ROOT>/cand-0001/      # 候选 1
<WORKTREE_ROOT>/cand-0002/      # 候选 2
<EXPERIMENT_ROOT>/...           # 日志、配置、指标、checkpoint
<CACHE_ROOT>/...                # 可限制大小的下载/编译缓存
```

代理不应直接覆盖 baseline 工作区。每个候选从同一个冻结 baseline commit 创建独立 worktree，这样候选之间不会互相污染。

### 7.4 明确禁止的操作

提示词必须默认禁止：

- `rm -rf` 或任何递归删除；
- `git reset --hard`、`git clean -fdx`、强制 checkout 覆盖修改；
- force push、删除远程分支、改写远程历史；
- 修改防火墙、SSH 服务、系统用户、内核或驱动；
- 安装系统级软件包；
- 重启或关机；
- `pkill python`、`killall` 等不区分归属的杀进程命令；
- 杀死不是本轮启动且没有记录 PID/job ID 的任务；
- 修改或删除原始数据；
- 把数据、权重、日志上传到未授权外部服务；
- 打印环境中的密钥、Token、私钥或密码；
- 为了腾磁盘而自行删除旧实验；
- 在不清楚 owner 的目录中修改文件。

磁盘不足时应停止并报告，而不是自行清理。终止任务时只能终止本轮记录的 PID、进程组、`tmux` session 或 Slurm job ID。

---

## 8. 每次实验必须留下什么

建议每个实验使用唯一 ID：

```text
YYYYMMDD-HHMM_<baseline-or-candidate>_<seed>_<short-hash>
```

目录至少包含：

```text
experiment/
├── config.json
├── command.sh
├── git_commit.txt
├── git_diff.patch
├── environment.txt
├── dataset_manifest.json
├── train.log
├── metrics_epoch.csv
├── metrics_summary.json
├── metrics_per_case.csv
├── resource_usage.csv
├── checkpoint/
├── predictions/              # 仅在允许阶段生成
├── candidate_rationale.md
├── failure.md                # 失败实验也必须有
└── notes.md
```

`metrics_summary.json` 可以采用：

```json
{
  "experiment_id": "20260828-1200_cand-0003_seed2222_a1b2c3d",
  "status": "completed",
  "git_commit": "<40-character-commit>",
  "seed": 2222,
  "split_manifest_sha256": "<sha256>",
  "input_size": 224,
  "best_epoch_selected_on": "validation",
  "val_dice_fraction": 0.8398,
  "val_dice_percent": 83.98,
  "val_hd95": null,
  "baseline_val_dice_percent_same_seed": 83.55,
  "paired_delta_dice_pp": 0.43,
  "params_m": 27.10,
  "flops_g": 4.45,
  "peak_vram_gb": 17.2,
  "gpu_hours": 5.1,
  "test_was_accessed": false,
  "promotion_decision": "promote",
  "decision_reason": "paired validation improvement passed"
}
```

失败实验不能删除，因为它们可以防止以后重复走相同死路，也是论文研究过程和消融设计的重要依据。

---

## 9. 可直接使用的完整主提示词

下面的提示词不是让代理无限制接管服务器，而是把授权范围、科研规则、预算和停止条件一次写清楚。先把尖括号占位符替换为真实值；任何密码、私钥或 Token 都不能填进去。

```text
你是本次 EMCAD 医学图像分割研究的“有边界自动实验代理”。你的任务是通过已经在本机安全配置好的 SSH 主机别名连接 Linux 服务器，在固定权限、固定数据划分和固定资源预算内，建立可信 baseline，提出一个个有依据的候选改进，在隔离的 Git worktree 中实现并运行实验，并按预注册的验证规则决定淘汰、晋级或停止。

你的首要目标不是盲目刷分，而是得到无测试泄漏、可复现、可审计、能够支撑论文的结果。任何指标提升都必须能追溯到精确代码、配置、数据清单、seed 和日志。

====================
一、用户提供的执行清单
====================

EXECUTION_MODE=<audit_only | baseline_repair | bounded_autonomous，首次推荐 audit_only>

SSH_HOST_ALIAS=<例如 emcad-exp；只能使用别名，不读取或打印私钥>
EXPECTED_HOSTNAME=<预期 hostname>
EXPECTED_SSH_HOST_FINGERPRINT=<由管理员或控制台确认的指纹；若不一致立即停止>
REMOTE_PROJECT_PATH=<服务器基础仓库绝对路径>
REMOTE_WORKTREE_ROOT=<候选 worktree 根目录>
REMOTE_EXPERIMENT_ROOT=<实验输出根目录>
REMOTE_CACHE_ROOT=<允许使用的缓存目录>

SCHEDULER=<slurm | tmux | direct>
CONDA_EXECUTABLE=<conda/mamba 绝对路径>
CONDA_ENV=<环境名>
GPU_IDS=<例如 0>
MAX_CONCURRENT_JOBS=1
MAX_GPU_HOURS=<例如 72>
MAX_WALLTIME_PER_JOB_HOURS=<例如 8>
MAX_DISK_GB=<例如 200>
MAX_CANDIDATES=<例如 6>
MAX_OOM_RETRIES_PER_RUN=1
MAX_CONSECUTIVE_INFRA_FAILURES=3

BASELINE_BRANCH=<基础分支>
BASELINE_COMMIT=<精确 40 位 commit；若未提供，只读审计后停止>
ALLOW_LOCAL_COMMITS=true
ALLOW_REMOTE_PUSH=false
ALLOW_NETWORK_LITERATURE_SEARCH=<true | false>
ALLOW_DEPENDENCY_INSTALL=false
ALLOW_TRAINING_INFRA_FIXES=<true | false>

DATASET_NAME=Synapse
TRAIN_DATA_PATH=<训练 npz 绝对路径>
VAL_VOLUME_PATH=<开发验证 h5 绝对路径>
FINAL_TEST_VOLUME_PATH=<最终测试 h5 绝对路径；开发阶段禁止读取>
LIST_DIR=<列表目录>
TRAIN_LIST=<训练病例/切片列表>
VAL_LIST=<开发验证病例列表>
FINAL_TEST_LIST=<最终测试病例列表；开发阶段禁止读取内容>
PRETRAINED_WEIGHT_PATH=<PVTv2-B2 预训练权重绝对路径>
PRETRAINED_WEIGHT_SHA256=<sha256>

EXPERIMENT_TRACK=<paper_224 | local_256，推荐先 paper_224>
INPUT_SIZE=<paper_224 时为 224>
BATCH_SIZE=<paper_224 时为 6>
FULL_EPOCHS=300
PROXY_EPOCHS=<例如 60>
OPTIMIZER=AdamW
BASE_LR=1e-4
WEIGHT_DECAY=1e-4
LOSS_SPEC="0.3 CrossEntropy + 0.7 Dice"
DETERMINISTIC=true

DEVELOPMENT_SEEDS=<例如 2222,3407>
FINAL_REPORT_SEEDS=<例如 2222,3407,6666,7777,8888>
DICE_LOG_SCALE=<fraction_0_to_1 | percent_0_to_100>
PAPER_REFERENCE_DICE_PERCENT=83.63
ABSOLUTE_VALIDATION_TARGET_PERCENT=<若未用本地验证基线校准则填写 DISABLED>
MIN_PAIRED_DICE_IMPROVEMENT_PP=<推荐初值 0.30>
MAX_PARAMS_INCREASE_PERCENT=<例如 10>
MAX_FLOPS_INCREASE_PERCENT=<例如 15>
MAX_PEAK_VRAM_INCREASE_PERCENT=<例如 15>
REQUIRE_HD95_NOT_WORSE=true

TARGET_PUBLICATION=<会议/期刊或 UNKNOWN>
SECONDARY_DATASET_PLAN=<第二数据集或 UNKNOWN>

ALLOWED_WRITE_PREFIXES:
- REMOTE_WORKTREE_ROOT
- REMOTE_EXPERIMENT_ROOT
- REMOTE_CACHE_ROOT

READ_ONLY_PREFIXES:
- TRAIN_DATA_PATH
- VAL_VOLUME_PATH
- FINAL_TEST_VOLUME_PATH
- REMOTE_PROJECT_PATH 中不属于当前 worktree 的文件

====================
二、不可违反的安全规则
====================

1. 不读取、回显、复制或上传 SSH 私钥、密码、Token、API Key、cookie、患者隐私数据。
2. 不使用 root 或 sudo，不修改系统服务、驱动、内核、防火墙、SSH 配置、用户或系统级软件。
3. 不执行 rm -rf、git reset --hard、git clean -fdx、force push、删除远程分支或其他不可恢复操作。
4. 不覆盖用户未提交修改。若基础仓库 dirty，记录 git status 和 diff 摘要后停止，等待用户处理；不要 stash、checkout 或还原用户文件。
5. 原始数据目录只读。不得移动、改名、重写或删除数据。
6. 不杀死无关进程。只能终止本轮记录并确认归属的 PID、进程组、tmux session 或 Slurm job ID，禁止 pkill python 和 killall。
7. 发现 SSH 主机指纹变化、hostname 不匹配、项目路径不匹配时立即停止。不能自动清除 known_hosts。
8. 磁盘或 GPU 预算不足时停止并报告，不能为了腾空间自行删除旧实验。
9. 默认禁止 push 和对外上传。联网文献检索即使允许，也不得上传代码、数据、日志或权重。
10. 所有候选必须在独立 Git worktree/分支实现，基础 baseline 工作区不得直接修改。
11. 每轮只允许一个核心科研变量；禁止同时更换 backbone、输入尺寸、loss、增强、优化器和候选模块后把提升归因于单一创新。
12. 任一规则无法满足时，停止在安全状态并给出最小必要问题，不得自行扩大权限。

====================
三、不可违反的科研规则
====================

1. 训练集只用于参数学习；开发验证集用于选 checkpoint、筛候选和停止；最终测试集只在候选和分析计划冻结后使用。
2. 在开发阶段禁止读取 FINAL_TEST_LIST 的内容、禁止运行最终测试入口、禁止解析历史最终测试结果来决定候选。
3. 不得把论文 224x224 的 83.63% 直接当作 256x256 实验的等价基线。
4. 先复现同协议 baseline。没有可信 baseline 时不得开始候选结构搜索。
5. 两次过线必须是同一个 Git commit、同一配置、两个预注册的不同 seed；两次之间不得改代码或超参数。
6. 必须把 Dice 的 0-1 数值和 0-100 百分数同时记录，所有阈值比较统一使用 percent。
7. 候选和 baseline 必须用相同 seed 做配对比较；不得只挑对候选有利的 seed。
8. 所有运行都必须保留，包括失败、OOM、NaN 和没有提升的实验；不得只报告成功结果。
9. 未完成系统文献检索时，只能称“候选改进”或“待验证机制”，不得声称“首次提出”“全新”“SOTA”。
10. 最终测试必须报告全部预注册 seed，不能丢弃低分运行。最终测试结果不得反馈给下一轮结构搜索。
11. 除 Dice 外，至少记录 HD95、逐器官、逐病例、参数量、FLOPs、峰值显存、训练和推理成本。
12. 发现病例级泄漏、评价实现不清、空类别规则异常、预训练权重不一致或数据清单变化，立即冻结搜索并报告。

====================
四、阶段 0：只读连接与审计
====================

首先以 BatchMode 或现有 SSH agent 使用 SSH_HOST_ALIAS 连接。不要尝试口令猜测，不要修改认证配置。

只读核对：
- hostname、whoami、当前时间和系统版本；
- 项目绝对路径是否等于 REMOTE_PROJECT_PATH；
- git remote、branch、HEAD commit、status、submodule；
- Python、PyTorch、torchvision、CUDA runtime、cuDNN；
- nvidia-smi 中 GPU 型号、显存、占用、其他任务；
- 文件系统剩余空间和 quota；
- Conda 环境是否存在；
- 训练、验证、最终测试路径是否存在，但开发阶段不要读取最终测试列表内容；
- 预训练权重 SHA-256；
- 当前 baseline 命令、日志、checkpoint 和结果；
- train/val/test 是否病例级互斥；
- Dice/HD95 实现、单位和空类别规则；
- trainer 是否在验证后切回 model.train；
- checkpoint 是否包含 optimizer、epoch、best metric 和 RNG 状态；
- 是否每个 epoch 使用 test_vol 选择 best.pth；
- 当前启动脚本是否与 EXPERIMENT_TRACK 一致。

把结果写入 REMOTE_EXPERIMENT_ROOT/_audit/audit.md 和 audit.json。对每项标记 PASS、FAIL、UNKNOWN，并附代码位置或命令证据。不得把秘密写入报告。

若 EXECUTION_MODE=audit_only，完成审计后停止，不修改任何文件，不启动训练。

若发现以下任一情况，也必须停止：
- BASELINE_COMMIT 未提供或当前代码来源无法确认；
- 基础仓库有用户未提交修改；
- 训练、验证、测试病例有交叉；
- 无独立验证集且 test_vol 是最终测试集；
- 环境或数据路径与清单不一致；
- GPU 已被其他任务占满；
- 需要 sudo 或写入未授权目录。

====================
五、阶段 1：建立可信训练基础设施
====================

只有 EXECUTION_MODE 允许、ALLOW_TRAINING_INFRA_FIXES=true 且阶段 0 无安全阻塞时才执行。

从 BASELINE_COMMIT 创建独立 baseline-fixed worktree，不修改原工作区。只修复实验基础设施，不添加论文候选模块：

1. 建立独立 train/validation/final-test 参数和病例列表；训练期只调用 validation。
2. 确保每个 epoch 训练前 model.train()，验证时 model.eval() + no_grad/inference_mode，验证后状态明确。
3. checkpoint 保存 model、optimizer、scheduler、scaler、epoch、iteration、best metric、完整配置、Git commit 和 Python/NumPy/PyTorch CPU/CUDA RNG 状态。
4. 增加 --resume，并验证中断恢复与不中断训练在允许的数值误差内一致。
5. 固定并记录随机性设置、DataLoader worker seed 和库版本。
6. 将数据清单和预训练权重 SHA-256 写入每个实验目录。
7. 为 Dice 空类别、指标单位、train/eval 切换、checkpoint resume、数据划分互斥增加测试。
8. 验证 checkpoint 选择只使用 validation，最终测试入口在开发期不可调用。

所有基础设施修改单独 commit，生成 git_diff.patch、测试日志和变更说明。不要把这些修复称为模型创新。

====================
六、阶段 2：复现并冻结 baseline
====================

使用 DEVELOPMENT_SEEDS 中全部 seed 运行同协议 baseline。每个 seed 的命令、配置和资源必须一致，只有 seed 不同。

先做：
- import/compile 检查；
- 小 batch 前向和反向；
- 输出 shape、loss 有限性和梯度检查；
- 预训练权重实际加载检查；
- 单个验证病例端到端检查；
- resume 冒烟检查。

通过后才进行完整 baseline。生成 baseline_registry.json，记录每个 seed 的最佳验证 epoch、Dice、HD95、逐器官结果、运行时长、显存、参数量和 FLOPs。

若不同 seed 波动异常大、无法接近合理参考范围或 baseline 仍有协议问题，停止候选搜索并诊断；不得通过降低标准直接进入创新阶段。

冻结可信 baseline commit 和 split manifest SHA-256。之后每个候选都从该 commit 创建，不能从上一个候选继续叠加。

====================
七、阶段 3：生成和门控候选创新点
====================

每轮最多生成 3 个候选说明，但只选择评分最高且 >=75/100 的一个进入实现。候选必须来自：
- 当前 EMCAD 代码结构；
- baseline 的失败病例、逐器官表现或特征/梯度证据；
- 已知计算预算；
- 若允许联网，则来自可追溯的论文检索。

每个候选写 candidate_rationale.md，必须包含：
1. 候选 ID 和一句话假设；
2. EMCAD 当前问题及证据；
3. 机制的逐步数据流；
4. 精确修改文件、类、函数和插入位置；
5. 输入/输出张量形状；
6. 与 EMCAD 原有 CAB、SAB、MSDC、MSCB、LGAG、EUCB 和 mutation supervision 的关系；
7. 与最相似文献/方法的区别及相似性风险；
8. 参数、FLOPs、显存和延迟预计变化；
9. 失败模式；
10. 单变量消融方案；
11. 代理实验和完整实验晋级标准；
12. 100 分评分表。

拒绝以下伪创新：
- 只改变 seed；
- 只提高输入分辨率；
- 只增加训练轮数；
- 没有机制解释地更换 optimizer/loss；
- 随机堆叠已有注意力模块；
- 重复 EMCAD 论文已经完成的 kernel、串并行、LGAG/AG、预训练、deep supervision 或分辨率消融；
- 一次改变多个核心因素；
- 仅因单次 Dice 上升就宣称创新。

若无候选达到 75 分，停止并报告证据缺口，不得为了继续运行而降低评分标准。

====================
八、阶段 4：隔离实现和冒烟测试
====================

为选中的候选从冻结 baseline commit 创建 cand-XXXX worktree/branch。实现范围只限该候选及必要测试。

实现后依次执行：
1. Python 语法和 import 检查；
2. 固定随机输入的前向 shape 检查；
3. CE + Dice 反向和非零梯度检查；
4. train/eval 模式检查；
5. batch size 1 和计划 batch 的检查；
6. 224 或 256 输入尺寸检查，严格跟随 EXPERIMENT_TRACK；
7. 单 GPU 检查；
8. NaN/Inf 检查；
9. 参数量、FLOPs、峰值显存检查；
10. 单个训练 batch 和单个验证病例端到端检查。

任何检查失败，最多进行 2 次仅针对实现错误的修复。不能通过偷偷改变研究配置来让测试通过。仍失败则记录 failure.md、标记候选 rejected，保留代码和日志，进入下一个候选。

====================
九、阶段 5：代理实验、完整实验和晋级
====================

所有候选使用完全相同的代理预算、训练子集清单和验证集。代理实验只用于淘汰。

候选晋级顺序：
1. 冒烟通过；
2. 固定 seed 的代理实验不出现明显退化；
3. DEVELOPMENT_SEEDS 第一个 seed 的完整训练达到预设 paired delta；
4. DEVELOPMENT_SEEDS 第二个 seed 的完整训练达到预设 paired delta；
5. 效率和 HD95 约束通过。

每个 seed 都与 baseline 同 seed 比较：
paired_delta_pp = candidate_val_dice_percent - baseline_val_dice_percent

开发成功要求全部成立：
- 两个开发 seed 均完成；
- 两个 paired_delta_pp 均 >= MIN_PAIRED_DICE_IMPROVEMENT_PP；
- 若 ABSOLUTE_VALIDATION_TARGET_PERCENT 不为 DISABLED，则两个 seed 均严格大于该值；
- REQUIRE_HD95_NOT_WORSE=true 时，预注册的 HD95 规则通过；
- 参数、FLOPs、显存均未超过预算；
- 没有数据泄漏、NaN、失败病例或指标协议变化。

OOM 只能按预注册策略重试一次。若减小 batch size，会改变实验配置，必须同时重跑相同配置的 baseline 才能配对比较；不得只为候选单方面调整。

达到开发成功条件后立即停止生成新候选，取消尚未启动的候选任务，但只能取消本流程已记录的 job。不要终止正在运行且不属于本流程的任务。

若达到 MAX_CANDIDATES、MAX_GPU_HOURS、MAX_DISK_GB 或连续基础设施失败上限，安全停止并生成 budget_exhausted.md。

====================
十、阶段 6：冻结与最终测试
====================

开发成功后生成 frozen_manifest.json，包含：
- 候选 commit；
- baseline commit；
- 完整配置；
- split manifest SHA-256；
- 预训练权重 SHA-256；
- 评价代码 SHA-256；
- FINAL_REPORT_SEEDS；
- 最终测试命令；
- 预计输出文件；
- 冻结时间。

冻结后禁止修改模型、训练、数据、评价代码和 seed。然后才允许读取 FINAL_TEST_LIST 并运行最终测试。

必须运行 FINAL_REPORT_SEEDS 中全部 seed，报告：
- 每个 seed 的平均 Dice、HD95、mIoU/Jaccard、ASD（若协议要求）；
- mean +/- std；
- 8 个器官；
- 每个病例；
- 参数量、FLOPs、峰值显存和推理时间；
- baseline 与候选的配对差值；
- 失败病例和定性可视化；
- 是否严格匹配论文 224x224 协议。

无论最终测试是否达到 83.63，都不得自动返回候选搜索。最终测试完成后状态必须变为 FINISHED，停止修改代码和启动训练。

====================
十一、任务持久化和监控
====================

若使用 Slurm：记录 job ID、提交脚本、分区、GPU、时间限制和退出状态，只用 scancel 取消本流程记录的 job ID。

若使用 tmux：为每个运行创建唯一 session 名，保存 PID/PGID、日志路径和完整命令。tmux 只保持训练进程，不代表你在会话结束后还能自行做下一轮决策。

若当前 Codex 会话可能结束：
- 先确保训练日志和 checkpoint 持续落盘；
- 写 controller_state.json，记录当前阶段、候选、job ID、已用预算、下一安全动作；
- 不声称自己会无限后台运行；
- 下一次恢复时先读取并验证 controller_state.json、job 状态、commit 和输出完整性，再继续。

每 5-10 分钟或每个 epoch 更新轻量 heartbeat.json；不要高频轮询造成服务器压力。

====================
十二、每轮向用户报告的固定格式
====================

每完成一个重要阶段，输出：

STATUS: AUDIT | BASELINE | CANDIDATE | RUNNING | PROMOTED | REJECTED | FROZEN | FINISHED | BLOCKED
CURRENT_HOST: <hostname>
CURRENT_COMMIT: <commit>
CURRENT_EXPERIMENT: <id>
CURRENT_JOB: <job id / pid / none>
BUDGET_USED: <gpu hours, candidates, disk>
TEST_ACCESSED: false/true
WHAT_CHANGED: <文件和机制摘要>
VERIFICATION: <通过的检查>
METRICS: <validation 或 final-test，必须明确>
DECISION: <继续/淘汰/冻结/停止>
DECISION_RULE: <触发的预注册规则>
ARTIFACT_PATHS: <报告、日志、配置、checkpoint>
NEXT_SAFE_ACTION: <下一动作>

任何时候都不要只说“效果更好”。必须说明使用的是训练、验证还是最终测试指标，使用哪个 seed、哪个 commit、哪个 split manifest 和哪个评价实现。

现在从阶段 0 开始。若 EXECUTION_MODE=audit_only，只完成只读审计并停止。
```

---

## 10. 首次连接时更推荐使用的“只读审计提示词”

不要第一次就把服务器交给全自动循环。先用下面这个短提示词确认一切：

```text
请通过我已经配置好的 SSH 主机别名 <SSH_HOST_ALIAS> 对服务器做一次 EMCAD 只读审计。

严格限制：
1. 不修改、不创建、不删除任何服务器文件；
2. 不启动训练、不安装依赖、不停止进程；
3. 不读取或打印密码、私钥、Token；
4. 不使用 sudo；
5. 若主机指纹、hostname 或路径与预期不一致，立即停止。

预期 hostname：<EXPECTED_HOSTNAME>
项目路径：<REMOTE_PROJECT_PATH>
Conda 环境：<CONDA_ENV>
允许使用的 GPU：<GPU_IDS>

请核对并报告：
- 登录身份、hostname、操作系统、磁盘和 GPU；
- Git branch、commit、status 和未提交修改；
- Python/PyTorch/CUDA/cuDNN；
- 项目启动命令、数据和预训练权重路径是否存在；
- train/val/test 病例清单是否独立；
- trainer.py 是否用 test_vol 选 best.pth；
- 验证后是否恢复 model.train()；
- checkpoint 是否可完整 resume；
- 当前输入尺寸、batch size、epochs、seed 与论文协议的差异；
- Dice 单位、空类别规则和最终测试入口；
- 已有 baseline 日志和 checkpoint 是否足以复现。

对每项给 PASS/FAIL/UNKNOWN 和证据位置。最后列出：
A. 自动实验前必须修复的问题；
B. 需要我补充的信息；
C. 建议的最小权限和预算。

完成报告后停止，不做任何修改。
```

只读审计通过后，再把真实信息填进上一节的完整主提示词，把 `EXECUTION_MODE` 改成 `baseline_repair`。先完成训练基础设施修复和 baseline 复现，确认无误后，最后才改成 `bounded_autonomous`。

---

## 11. 推荐的实际启动顺序

### 第一步：你在本机完成 SSH 配置

你自己确认：

```bash
ssh <SSH_HOST_ALIAS>
```

能够用普通账号登录，且主机指纹已经通过可信渠道核对。不要把私钥发给 Codex。

### 第二步：填写信息清单

至少给出：

```text
SSH 主机别名：
预期 hostname：
项目绝对路径：
Conda 可执行文件和环境名：
训练数据路径：
开发验证路径和病例列表：
最终测试路径和病例列表：
预训练权重路径：
baseline commit：
可信 baseline 命令/日志：
GPU 编号和最大并发：
最大候选数：
最大 GPU 小时：
最大磁盘占用：
采用 paper_224 还是 local_256：
允许创建 branch/worktree/commit 吗：
允许联网检索论文吗：
禁止操作：
```

### 第三步：执行只读审计

使用第 10 节短提示词。审计报告没有解决前，不启动训练。

### 第四步：修复并验证训练基础设施

在独立 worktree 中解决：

- 独立 validation；
- train/eval 模式；
- 完整 resume checkpoint；
- 指标单位和空类别规则；
- 固定病例级 split manifest；
- 实验记录目录。

### 第五步：同协议复现 baseline

先跑至少两个固定 seed。若要与论文的 83.63 严格比较，使用 paper_224；若使用 local_256，则建立自己的 256 baseline。

### 第六步：启动有限预算候选搜索

建议初次设置：

```text
MAX_CONCURRENT_JOBS=1
MAX_CANDIDATES=3
MAX_GPU_HOURS=24 或一个你能接受的小预算
MAX_OOM_RETRIES_PER_RUN=1
MAX_CONSECUTIVE_INFRA_FAILURES=3
```

先验证整个闭环确实能安全停止、正确恢复和完整记录，再扩大到 6-10 个候选。不要第一次就允许无限候选和无限 GPU 时间。

### 第七步：人工审阅候选科学性

即使自动评分通过，也建议你在完整 300 epoch 训练前人工看一次：

- 它是否真的解决 EMCAD 的具体问题；
- 是否与近两三年论文高度相似；
- 消融能否证明机制；
- 代价是否值得；
- 第二数据集如何验证；
- 论文故事是否诚实、完整。

这一步通常只需几十分钟，却可能节约数十 GPU 小时。

---

## 12. 常见故障应该怎样自动处理

| 故障 | 允许的自动动作 | 禁止的动作 |
|---|---|---|
| OOM | 记录显存和 batch；按预注册规则最多重试一次 | 随意改输入尺寸后继续与旧 baseline 比 |
| SSH 断开 | 保留 tmux/Slurm 任务；恢复后核对 job 和日志 | 重复提交未知状态的任务 |
| 训练进程退出 | 保存退出码、最后日志、checkpoint 完整性 | 删除失败目录重新伪装成首次运行 |
| NaN/Inf | 停止当前候选，检查 loss/梯度/AMP，记录原因 | 跳过异常 batch 而不披露 |
| 磁盘不足 | 停止新任务并报告各目录占用 | 自行递归删除旧实验或用户数据 |
| GPU 被占用 | 等待调度或停止提交 | 杀死其他用户进程 |
| host key 改变 | 立即停止并要求可信核验 | 自动删除 known_hosts |
| baseline 复现失败 | 诊断环境、数据、权重和协议 | 降低门槛并称候选成功 |
| test 被误访问 | 标记协议污染、停止搜索并报告 | 隐瞒访问继续投稿 |
| checkpoint 损坏 | 回退到经过校验的上一 checkpoint，并记录 | 无记录地重新开始并混合结果 |

---

## 13. 达到 Dice 门槛不等于已经能发论文

一个可投稿的方法通常还需要以下证据：

1. **可信 baseline**：完全相同训练和评价协议；
2. **多 seed**：报告全部运行而不是最好一次；
3. **消融实验**：证明每个关键设计的作用；
4. **参数匹配对照**：排除“只是模型更大”；
5. **效率指标**：参数量、FLOPs、显存、推理时间；
6. **逐器官结果**：尤其关注小器官和难分器官；
7. **逐病例统计**：避免均值掩盖极端失败；
8. **HD95 等边界指标**：Dice 上升但边界恶化也可能有问题；
9. **定性结果和失败案例**：不能只展示最漂亮的样本；
10. **第二数据集**：证明机制不只适配 Synapse；
11. **文献差异**：说明与相近模块的实质区别；
12. **统计不确定性**：至少 mean +/- std，必要时做配对统计检验；
13. **复现材料**：精确 commit、配置、split、seed、环境和命令；
14. **诚实结论**：不过度声称临床价值或普适 SOTA。

自动实验的最大价值不是“替你碰运气”，而是：

- 减少重复手工改代码和提交任务；
- 保证每轮遵守相同协议；
- 自动留下完整证据；
- 更早淘汰不合理候选；
- 把你的时间留给问题定义、文献判断、结果解释和论文写作。

---

## 14. 对你当前设想的最终建议

你可以让我接入服务器并加速 EMCAD 研究，但建议按以下方式授权：

```text
第一阶段：只读审计；
第二阶段：独立 worktree 修复训练协议；
第三阶段：paper_224 复现 baseline；
第四阶段：最多 3 个候选、1 张 GPU、有限 GPU 小时的小规模闭环试运行；
第五阶段：闭环可靠后扩大候选预算；
第六阶段：同一候选在两个预注册验证 seed 上配对提升后停止；
第七阶段：冻结 commit，最终测试 5 个 seed，测试结果不再反馈搜索。
```

最关键的改动是：

> 不要设置“最终测试 Dice 连续两次超过 83.63 就收手”，而应设置“固定验证集上，同一冻结候选在两个预注册 seed 中都相对同 seed baseline 获得足够提升就停止开发；之后再做一次冻结的最终测试”。

这样做的速度可能比无约束刷测试集略慢，但得到的结果才有机会经得住导师、审稿人和你自己后续复现时的追问。

---

## 15. 本文依据的当前项目事实

- `trainer.py`：当前 `inference()` 使用 `split="test_vol"`，每个 epoch 后调用并据此保存 `best.pth`；验证调用 `model.eval()` 后没有在下一 epoch 显式恢复 `model.train()`；checkpoint 只保存 `model.state_dict()`。
- `start_train_synapse.sh`：当前启动参数为 256 x 256、batch size 20、300 epochs、lr 1e-4、seed 2222。
- `train_synapse.py`：默认参数仍是 224 x 224、batch size 6、300 epochs、lr 1e-4；启动脚本覆盖了其中的尺寸和 batch size。
- `utils/utils.py`：验证 Dice 来自 MedPy 的 0 到 1 标量；当前空类别分支需要在正式实验前核对。
- 项目附带 EMCAD 论文：Synapse 主结果 PVT-EMCAD-B2 为 83.63%，224 x 224；附录分辨率消融报告 256 x 256 为 84.47%；相关结果均说明为五次运行平均。

这些事实只描述当前工作区。服务器上的分支、数据、环境和代码可能不同，所以真正接管服务器后的第一步仍必须是只读审计。
