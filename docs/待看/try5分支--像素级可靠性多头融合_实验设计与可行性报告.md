# 像素级可靠性多头融合：候选方法实验设计与可行性报告

日期：2026-09-23

## 结论边界

本报告把“像素级可靠性多头融合”称为候选方法或待验证方法，不预先声称它能提高 Dice。当前没有在 Synapse 或 ACDC 上完成远程正式训练，因此目前没有任何 Dice 提升证据。只有在固定病例划分、相同 seed 配对、相同 checkpoint 选择规则和相同患者级评估口径下，多个 seed 的总体 Dice 稳定改善，才可以把改善归因于该模块。

## 现有 EMCAD baseline 审计

`lib/networks.py::EMCADNet` 保留 PVT/ResNet 编码器、EMCAD 解码器、四个输出头和原有四输出监督。默认 `--fusion_mode p1 --fusion_loss_weight 0` 时，前向仍返回 `[p4,p3,p2,p1]`，训练仍使用原来的 `mutation/deep_supervision/last_layer`，测试仍等价于原来的 `P[-1]`。

输入为 `[B,1,H,W]` 时，EMCAD 内部先变为 3 通道；以输入 `224x224` 为例，四个输出在上采样后均为 `[B,K,224,224]`，其中 `K=9`（Synapse）或 `K=4`（ACDC）。输出顺序是 `[p4,p3,p2,p1]`，`p1` 是最高分辨率头。

本次还修复了一个必须统一应用于所有未来实验的协议错误：`train_synapse.py` 原来把实验目录片段写成 tuple，`os.path.join` 会在启动时抛 `TypeError`。因此修复前未成功启动的结果不能作为正式结果；若已有结果来自另一份路径修复代码，需保留其原始配置并单独标记。

## 候选方法定义

四个 EMCAD 分割头各自产生类别 logits：

`p_i in R^(B x K x H x W), i in {4,3,2,1}`。

候选方法只新增输出端融合，不改变编码器、解码器、四个分割头的主结构：

1. `p1`：直接使用 `p1`，这是 EMCAD baseline。
2. `fixed_sum`：`z = p4 + p3 + p2 + p1`。不引入可学习融合参数。
3. `global_scalar`：学习四个标量 `a_i`，`w=softmax(a)`，`z=sum_i(w_i p_i)`。权重对整张图和所有患者相同。
4. `pixel_reliability`：每个全分辨率 EMCAD logits 增加一个 `1x1` 可靠性头，输出 `r_i in R^(B x 1 x H x W)`；`w_i(x)=softmax_i(r_i(x))`，再计算 `z(x)=sum_i(w_i(x)p_i(x))`。可靠性头直接读取对应 logits，因此多 GPU 下不依赖 replica 间传递临时特征。

像素可靠性模式的可靠性目标是每个分割头的 detached 像素级 argmax 正确性，使用 BCE 校准可靠性头；融合 logits 另使用与当前 ACDC/Synapse 相同的 `0.3*CE+0.7*Dice` 辅助项。这个辅助项是候选方法训练协议的一部分，不能被解释为“只改推理、完全零代价”。

## 必做四组比较

| 组 | `fusion_mode` | `fusion_loss_weight` | 结果可归因 | 不能归因 |
|---|---|---:|---|---|
| G0 baseline | `p1` | `0` | 当前 EMCAD 原行为 | 不能把旧 checkpoint 与新训练协议混比 |
| G1 固定求和 | `fixed_sum` | `1` | 固定四头求和读出及其明确辅助训练项 | 不能声称是像素可靠性收益 |
| G2 全局标量 | `global_scalar` | `1` | 四个全局可学习标量的贡献 | 不能把全局权重解释为像素级机制 |
| G3 像素可靠性 | `pixel_reliability` | `1` | 像素可靠性头、像素 softmax 融合及其校准项的整体贡献 | 不能把可靠性头和辅助损失拆开归因，除非增加预注册的额外消融 |

为回答“提升来自哪个模块”，至少应在每个数据集上用相同 seed 配对运行 G0/G1/G2/G3；建议 seed 为 `2222, 3333, 4444`，但必须在开始前固定，不得根据测试集挑 seed。每组只改变 `fusion_mode` 以及该模式所必需的融合训练项；编码器、输入尺寸、batch size、epoch、优化器、学习率、预处理、病例清单、checkpoint 选择和评估代码保持一致。

Synapse 当前仓库的验证链路实际使用 `test_vol_h5` 病例级评估；按本项目既定规则，这里不另造验证划分，正式报告应明确这是当前 Synapse 的 checkpoint 选择集合。ACDC 使用固定 `valid` 选择 checkpoint，最终只在固定 `test` 上报告。

## 必须检查的证据

每个 checkpoint 保存 `config.json` 或等价命令日志，确认四个组的参数。测试日志会记录 `fusion_weight_statistics`：

- `global_scalar`：四个 softmax 标量，检查是否长期接近 `[1,0,0,0]` 或 `[0,0,0,1]`。
- `pixel_reliability`：四个权重的均值、标准差、最小值、最大值；若均值长期接近单头独占且空间标准差接近 0，说明出现权重塌缩。
- 同时保存每病例 Dice，而不是只保存一个全局均值；用患者级均值、标准差和 G3-G0 的配对差值报告结果。

不能把 smoke test、参数量、训练 loss 下降、单个器官改善或单个 seed 改善写成总体 Dice 提升证据。

## 远程运行参数

以下参数必须加入对应 `.sh` 的 Python 命令行；本次没有直接修改任何 `.sh`，以遵守“不修改启动脚本”的约束。四组训练只替换 `FUSION_MODE`，候选组把 `FUSION_LOSS_WEIGHT=1`，baseline 保持 0。

按现有文件名对应修改位置如下（只复制到相应脚本，不要把四组参数混在同一次运行中）：

- `start_train_synapse.sh`：在现有 `SUPERVISION=...` 附近增加 `FUSION_MODE`、`FUSION_LOSS_WEIGHT`、`RELIABILITY_LOSS_WEIGHT` 三个变量；在 `python train_synapse.py` 命令的 `--supervision` 后增加三项 `--fusion_mode`、`--fusion_loss_weight`、`--reliability_loss_weight`。
- `start_test_synapse.sh`：增加 `FUSION_MODE` 和 `FUSION_LOSS_WEIGHT`；在 `python test_synapse.py` 命令增加 `--fusion_mode "${FUSION_MODE}" --fusion_loss_weight "${FUSION_LOSS_WEIGHT}"`。
- `start_train_acdc.sh`：增加同样三个变量；在 `python -u train_acdc.py` 命令的 `--supervision` 后增加三项参数。
- `start_test_acdc.sh`：增加 `FUSION_MODE`；将命令中的 `test_ACDC.py` 改为仓库实际文件名 `test_acdc.py`，并增加 `--fusion_mode "${FUSION_MODE}"`。

Synapse 训练命令追加：

```bash
FUSION_MODE="${FUSION_MODE:-p1}"
FUSION_LOSS_WEIGHT="${FUSION_LOSS_WEIGHT:-0}"
RELIABILITY_LOSS_WEIGHT="${RELIABILITY_LOSS_WEIGHT:-1}"

python train_synapse.py ... \
  --fusion_mode "${FUSION_MODE}" \
  --fusion_loss_weight "${FUSION_LOSS_WEIGHT}" \
  --reliability_loss_weight "${RELIABILITY_LOSS_WEIGHT}"
```

Synapse 测试命令追加：

```bash
python test_synapse.py ... --fusion_mode "${FUSION_MODE}"
```

ACDC 训练命令追加：

```bash
python train_acdc.py ... \
  --fusion_mode "${FUSION_MODE}" \
  --fusion_loss_weight "${FUSION_LOSS_WEIGHT}" \
  --reliability_loss_weight "${RELIABILITY_LOSS_WEIGHT}"
```

ACDC 测试命令追加：

```bash
python test_acdc.py ... --fusion_mode "${FUSION_MODE}"
```

启动前建议在 `.sh` 中显式设置：

```bash
export FUSION_MODE=pixel_reliability
export FUSION_LOSS_WEIGHT=1
export RELIABILITY_LOSS_WEIGHT=1
```

四组运行必须使用不同的 `RUN_ID` 和独立输出目录，不覆盖已有 checkpoint、CSV 或实验记录。Synapse 候选模式的训练代码会自动在旧路径后追加 `_fusion_<mode>_fw<weight>`；`p1 + 0` 保留旧 baseline 路径，候选组测试时必须传入相同的 `--fusion_mode` 和 `--fusion_loss_weight` 才能自动找到对应目录。

## 现有 `.sh` 必须人工核对的启动问题

这不是本次方法代码偷偷修复的范围，远程运行前必须处理：

1. `start_test_synapse.sh` 在 `set -u` 下引用了未定义的 `SUPERVISION`、`BATCH_SIZE`、`seed` 和 `MAX_EPOCHS`，需要先定义，或从 `RUN_ID` 中移除这些未定义变量。
2. `start_train_acdc.sh` 在 `set -u` 下使用了未定义的 `n_gpu`，需要定义 `n_gpu=${n_gpu:-1}` 或改为已定义的 `N_GPU`。
3. `start_test_acdc.sh` 当前调用 `test_ACDC.py`；Linux 大小写敏感，而仓库文件是 `test_acdc.py`，命令必须与实际文件名一致。
4. 这些启动脚本还要把上面的三个训练参数和一个测试参数显式传给 Python；仅设置环境变量而不在命令行传参不会改变模型。

## 变更文件清单

方法代码：

- `lib/networks.py`：四种融合算子、全局标量、像素可靠性头、校准损失和塌缩统计。

实验协议代码：

- `train_synapse.py`、`trainer.py`：Synapse 参数传递与候选融合辅助损失。
- `train_acdc.py`、`utils/acdc_utils.py`：ACDC 参数传递、辅助损失和体推理融合。
- `train_synapse.py`：修复 tuple 实验路径导致的启动错误。

测试代码：

- `test_synapse.py`、`test_acdc.py`：重建候选模型时传入 `fusion_mode`。
- `utils/utils.py`：Synapse 测试/验证使用统一融合读出，并保持 `p1` 兼容。

文档：

- `docs/像素级可靠性多头融合_实验设计与可行性报告.md`。

## 当前验证状态

已完成 Python 语法编译检查；本地环境没有 PyTorch，未在本机运行 GPU forward、训练或正式评估。远程服务器上必须先做单 batch forward、单 batch backward、一个病例验证和 checkpoint 严格加载，再开始完整实验。当前没有经过 Synapse/ACDC 正式训练的 Dice 结果，效果仍是“尚未验证”。
