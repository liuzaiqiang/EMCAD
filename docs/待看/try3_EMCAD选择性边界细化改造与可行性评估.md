# try3：EMCAD 选择性边界细化改造与可行性评估

日期：2026-09-23  
分支：`try3`

## 结论

本次实现是保留 EMCAD 编码器、解码器和原有多头监督的增量模块：在 EMCAD 最终 logits 上增加一个零初始化残差细化器，并仅在预测不确定、分支不一致或预测边界附近选择图块进行细化。新模块零初始化，因此加载相同 EMCAD 权重且不训练细化器时，初始预测与 baseline 相同。

实现和两数据集的测试入口均可运行；目前**没有证据证明 Synapse 或 ACDC 的总体 Dice 得到提高**。已完成的训练仅为管线 smoke，指标接近零，不能作为结果。合成输入基准还显示当前稀疏实现有额外采样开销，不能宣称推理加速。科研上可作为待验证的 EMCAD 增量方法，但现阶段尚不构成已验证的论文创新或可投稿结果。

## 改了什么

- `lib/selective_refinement.py`：新增选择性细化模块。路由依据 EMCAD 两路输出的归一化熵、预测分歧和局部类别边界；sparse 按分数阈值及每图预算选块；每块附带 halo 上下文，中心区域残差写回；未选区域残差严格为零。`dense`、`tiled_dense` 用于消融对照。
- `lib/networks.py`：将细化器作为 EMCAD 最终输出后的可选分支；默认 `off` 保持 baseline 路径。
- `train_synapse.py`、`train_acdc.py`：接入统一细化参数与 checkpoint 配置记录；Synapse 验证按患者级固定拆分，验证集选 best，不使用测试集选权重。
- `test_synapse.py`、`test_acdc.py`：使用 checkpoint 配置核对细化结构/路由设置，输出逐病例、逐类别指标 CSV，并记录实际细化块比例。
- `trainer.py`：接入细化训练、验证及实验配置持久化。
- `utils/refinement_options.py`：统一 CLI 参数、checkpoint 配置核对和细化工作量统计。
- `utils/research_protocol.py`：训练/验证/测试划分的审计与防患者泄漏检查。
- `utils/segmentation_metrics.py`、`utils/utils.py`、`utils/acdc_utils.py`：整理 Dice、HD95、Jaccard、ASSD、Boundary F1 的评估口径，并修复空掩膜距离指标处理；Synapse 的 `--skip_save` 现在可跳过 PNG/NIfTI 写入，修正不兼容的 overlay 参数。
- `prepare_synapse_refinement_split.py`：生成固定 Synapse 患者级验证拆分。当前留出 `case0007`、`case0023`、`case0026`，共 390 张验证切片；按病例拆分避免切片级泄漏。
- `benchmark_refinement.py`、`tests/test_selective_refinement.py`：提供计算基准脚本及核心契约测试。
- `docs/refinement_synapse_benchmark.json`：记录 RTX 3060 Laptop GPU 上合成输入基准结果。

新增/改造的创新方法定义上方均用“创新点”和 `20260923` 标记；创新模块按语句添加中文注释，说明张量含义、形状、路由、写回和实验约束。

## 验证记录

- 标准库 `unittest`：10 项通过，覆盖初始等价、稀疏未选位置不变、零预算跳过细化卷积、dense 预算、梯度、指标空类口径、患者泄漏检查和 checkpoint 配置保护。
- Synapse 测试 smoke：成功读取官方测试病例 `case0008`，执行 `sparse`，生成指标 CSV；统计激活比例 25%。由于 smoke checkpoint 未充分训练，mean Dice 为 `0.000175`，这仅说明测试入口和文件输出打通。
- ACDC 测试 smoke：成功读取测试病例 `case_002_volume_ED`，执行 `sparse`，生成指标 CSV；统计激活比例 25%。mean Dice 为 `0`，同样只说明管线可运行。
- 计算基准使用随机合成输入、无训练权重，224×224，warmup 5 次、计时 20 次，包含 AMP/反向检查。测得 off `19.73 ms`、sparse `23.27 ms`、dense `21.81 ms`、tiled_dense `22.76 ms`。因此目前实现有可测额外开销；需优化采样/索引或以精度收益证明开销合理。
- 两个测试命令均使用 PyTorch 环境，并临时复用另一环境的 `medpy`/`timm`；正常直接启动时当前 `pytorch` 环境缺少 `medpy`。没有安装依赖或更改环境。

## 可行性判断与下一步

代码层面可行：模块保留 EMCAD 主干和监督结构，初始化严格回退到 baseline，路由不读取标签，稀疏位置未写回时 logits 不变，训练梯度和 AMP smoke 已验证。方法有效性仍未知。熵/分歧/边界驱动局部细化属于合理实验假设，但并非天然能提升 Dice；错误的基线边界和预算过低会漏掉整块器官或较大结构错误。

发表前至少完成同一数据划分和训练设置下的多 seed 对照：EMCAD baseline (`off`)、稀疏细化 (`sparse`)、全图同一细化头 (`dense`)、同覆盖率图块对照 (`tiled_dense`)；报告总体及逐类别 Dice、HD95/ASSD、Boundary F1、推理时间/显存、逐病例结果和均值/标准差。只用验证集选 checkpoint/阈值，冻结后在官方测试集评估一次。Synapse 当前固定验证集已与测试病例区分；ACDC 需确认 `lists_ACDC` 是患者级固定划分，并在报告中记录清单哈希。若总体 Dice 没有稳定提升，或只提升边界指标，就不能以“提高总体 Dice”作为论文主结论。

## 启动脚本需调整的参数（本次没有修改任何 `.sh`）

训练和测试 Python 入口默认 `--refinement off`，因此想训练创新组时必须在 `.sh` 中调用对应 Python 命令的位置增加同一组参数：

```text
--refinement sparse --refine_tile 16 --refine_ratio 0.25 --refine_hidden 32 --refine_threshold 0.1
```

Synapse 训练还需使用新固定划分对应的参数：

```text
--list_dir data/synapse_refinement_seed2222/lists --volume_path data/synapse_refinement_seed2222/valid_vol_h5
```

Synapse 测试继续使用同一个 `--list_dir`（其中 `test_vol.txt` 保留官方测试病例），但测试体数据目录必须指向官方测试集，例如 `--volume_path ../data/Synapse/test_vol_h5`。训练和测试都要传相同的细化配置，测试入口会拒绝与 checkpoint 的 `config.json` 不一致的设置。ACDC 的训练和测试脚本也都需要增加同一组 `--refinement ...` 参数，且必须使用完全相同的细化参数；其数据路径继续按现有 `--root_path`、`--list_dir` 设置。

三种消融的模式名为 `off`、`sparse`、`dense`、`tiled_dense`。每种模式应独立训练 checkpoint；不能把 `off` 权重直接当作已训练的 sparse/dense 结果。
