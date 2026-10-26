# CAA 与面积自适应 Tversky：运行前审查（2026-10-26）

## 结论与边界

- 当前的两个改动是 **CAA 上采样结构** 与 **面积自适应 Tversky 训练辅助损失**。后者不是推理网络模块。`CAA=0, AAT=0` 时，解码器结构和训练损失路径回到本仓库的 EMCAD 基线；这不等于其余 11 个任务的完整训练协议逐项等同官方实现。
- 用本仓库历史基线提交 `0ecb282` 的 `lib/decoders.py` 做严格对照：关闭 CAA 后，EUCB 和完整 EMCAD decoder 的 `state_dict` 键顺序一致，加载历史权重 `strict=True` 成功，同一输入的输出逐位相同。`lib/networks.py` 相对该提交只增加 CAA 参数向 decoder 传递，未改原有 encoder 和四个输出头。[官方 EUCB 源码](https://github.com/SLDGroup/EMCAD/blob/main/lib/decoders.py)也使用相同的最近邻上采样、逐通道卷积、BN、激活和逐点卷积顺序。
- AAT 只在 `--use_area_adaptive_tversky=1` 时追加到原监督损失。单/多类张量的前向、反向、空前景返回值已做 CPU 检查；不代表 Dice 收益可从 Synapse 推断到其他任务。
- 所有训练启动器现在默认两项都开；`USE_CONTENT_AWARE_ANTIALIAS=0/1` 唯一映射 `off/caa`，`USE_AREA_ADAPTIVE_TVERSKY=0/1` 唯一映射关闭/追加损失。Python CLI 的默认值仍是 `off/0`，便于直接运行基础模型。测试启动器从权重旁 `config.json` 重建 CAA 结构；AAT 是训练损失，推理时不需要开关。

## 12 个数据集的接线与本机预检

| 数据集 | 训练入口 | 测试入口 | CAA/AAT | 本机准备数据 |
| --- | --- | --- | --- | --- |
| Synapse | `train_synapse.py` | `test_synapse.py` | 直连/直连 | 用户已在服务器验证；本次未重跑 |
| ACDC | `train_acdc.py` | `test_acdc.py` | 直连/直连 | 列表存在，train/valid/test 病例 ID 无交叉 |
| BKAI、ClinicDB、ColonDB、ETIS、Kvasir | `train_polyp.py` | `test_polyp.py` | 直连/直连 | 各自 train/val/test 目录存在；同一数据集三个划分的文件 stem 无交叉 |
| ISIC2017、ISIC2018 | `train_isic.py` 包装二分类入口 | `test_isic.py` 包装二分类入口 | 继承/继承 | 本机准备后目录缺失；代码检查了划分 manifest 与摘要 |
| BUSI | `train_busi.py` 包装二分类入口 | `test_busi.py` 包装二分类入口 | 继承/继承 | 本机准备后目录缺失；代码检查了划分 manifest、哈希与摘要 |
| DSB18、EM | `start_train_cell.sh` 复用 `train_polyp.py` | `start_test_cell.sh` 复用 `test_polyp.py` | 继承/继承 | 本机准备后目录缺失；DSB18 使用实例掩膜合并选项 |

“接线”仅表示 CLI、模型构造、损失条件分支及权重恢复连通，不表示该数据集训练能提高 Dice。服务器上的实际数据格式、mask 编码、预训练权重、GPU 内存及最终指标仍待实测。

## 本次修正

1. ACDC、Polyp、ISIC、BUSI、Cell 启动器的 AAT 默认值统一为 1；ACDC 的 CAA 默认值也统一为 1。所有启动器都保留两个独立开关。`run_seed_queue.sh` 不再错误显示 AAT 的默认值为 0。
2. 删除启动脚本中没有任何 Python 实现对应的“像素可靠性融合”变量与日志，避免把不存在的第三个模块误记进消融表。二值 CAA 总开关设为 1 时固定传 `caa`，设为 0 时固定传 `off`。
3. ACDC train/test 启动器支持 `ROOT_PATH` 覆盖，训练器另支持 `LIST_DIR` 覆盖。现有默认路径保持项目上一级的 `data`；若服务器把 `data` 放在上两级，必须显式设置数据根（其他任务用 `DATA_ROOT`）。
4. 修正 ACDC 的假阳性指标：GT 为空、预测有前景时 Dice/IoU=0；仅单侧有前景时 HD95/ASD 写 `NaN`，因为双向表面距离无定义；双方都空时 Dice/IoU=1、距离=0。旧 ACDC 结果与修正后结果不可直接拼表，须用同一评估代码重评四组。
5. ACDC、Polyp、BUSI 训练启动器现在可通过环境变量覆盖 Conda 位置；ISIC 原本已有此能力。各数据集仍需在目标服务器核对环境和预训练权重。

## 论文实验必须先固定的事项

- **Synapse 的 `test_vol`**：当前 `trainer.py` 每轮用 `test_vol.txt` 选择 `best.pth`，`test_synapse.py` 也评估同一列表。如果该列表是论文最终测试集，就是测试集参与模型选择。应另设独立验证集并对四组重新训练，然后只在冻结模型后评估最终测试集。本次保持已验证的 Synapse 训练代码原样，不能把现有流程声称为独立测试。
- **官方代码范围**：官方 EMCAD 仓库现已公开 Polyp 训练和测试代码（2026-01-12），ACDC README 指向 CASCADE；ISIC/BUSI/Cell 在本仓库仍是补充实现。官方 Polyp 源码的默认 batch size、学习率、监督参数、逐图概率归一化与当前脚本并不完全一致。因此论文中的“原始 EMCAD”必须指**同一任务、同一数据划分与训练评估协议下，两个开关都为 0 的本仓库基线**；如要引用作者公开分数，需先严格对齐协议。来源：[官方 README](https://github.com/SLDGroup/EMCAD)、[官方 Polyp 训练代码](https://github.com/SLDGroup/EMCAD/blob/main/train_polyp.py)。
- **划分单位**：本机检查的是 ACDC 病例 ID、息肉文件名 stem。息肉视频近重复帧、BUSI 病人归属、ISIC 官方划分、DSB18/EM 数据来源与 mask 语义，仍需结合原始数据元信息核对。文件名无交叉不等于患者或视频级无泄漏。
- **统一四组**：对每个数据集、每个种子固定同一清单、预处理、预训练权重、epoch、batch、优化器、验证规则、推理脚本和阈值；只改变两个开关。四组分别为 `0/0`、`1/0`、`0/1`、`1/1`。报告逐病例/逐类 Dice 与分布、平均值、多种子方差；不能仅报告最优单次运行。测试集不参与调参或 checkpoint 选择。
- **路径**：项目上两级数据目录不是当前脚本默认值。在服务器使用绝对路径设置：ACDC `ROOT_PATH`、`LIST_DIR`；Polyp/ISIC/BUSI/Cell `DATA_ROOT`；Synapse 使用其训练脚本现有的数据路径变量。确认路径后再提交长训练队列。

## 消融命令示例

```bash
# 以 ClinicDB 为例，其余任务换启动器和 DATASET_NAME。
DATASET_NAME=ClinicDB USE_CONTENT_AWARE_ANTIALIAS=0 USE_AREA_ADAPTIVE_TVERSKY=0 bash sh/start_train_polyp.sh
DATASET_NAME=ClinicDB USE_CONTENT_AWARE_ANTIALIAS=1 USE_AREA_ADAPTIVE_TVERSKY=0 bash sh/start_train_polyp.sh
DATASET_NAME=ClinicDB USE_CONTENT_AWARE_ANTIALIAS=0 USE_AREA_ADAPTIVE_TVERSKY=1 bash sh/start_train_polyp.sh
DATASET_NAME=ClinicDB USE_CONTENT_AWARE_ANTIALIAS=1 USE_AREA_ADAPTIVE_TVERSKY=1 bash sh/start_train_polyp.sh
```

## 已执行的验证

- `bash -n` 检查 `sh/` 下全部脚本，通过。
- `compileall` 检查全部训练、测试、模型、工具代码，通过。
- 历史 EUCB/EMCAD decoder 与当前 `caa_mode='off'` 的严格权重键和逐位输出对照，通过。
- AAT 在二值和四分类 logits 上反向梯度有限、空前景为零，通过。
- PVTv2-B0 全网用 64×64 输入分别检查二值/四分类、CAA 开/关；四个输出的形状均为 `[1,C,64,64]`。解码器参数数目分别为关闭 506507、开启 534881；AAT 不改变这两个数目。
- ACDC 指标空 GT 假阳性回归测试见 `tests/test_acdc_metric_policy.py`。
