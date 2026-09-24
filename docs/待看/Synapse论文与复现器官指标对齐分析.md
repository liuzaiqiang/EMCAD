# Synapse 论文与复现器官指标对齐分析

## 1. 八个器官缩写

论文表格的列顺序是：`Aorta, GB, KL, KR, Liver, PC, SP, SM`。

| 缩写 | 英文全称 | 中文 | 说明 |
|---|---|---|---|
| Aorta | Aorta | 主动脉 | 大血管 |
| GB | Gallbladder | 胆囊 |  |
| KL | Kidney Left | 左肾 | 按人体解剖学左右，不按图像显示左右 |
| KR | Kidney Right | 右肾 | 按人体解剖学左右 |
| Liver | Liver | 肝脏 |  |
| PC | Pancreas | 胰腺 |  |
| SP | Spleen | 脾脏 |  |
| SM | Stomach | 胃 |  |

## 2. 论文列顺序与当前日志顺序不同

`test_synapse.py` 中的 `classes` 顺序是：

```text
1 spleen
2 right kidney
3 left kidney
4 gallbladder
5 pancreas
6 liver
7 stomach
8 aorta
```

论文表格顺序是：

```text
Aorta, GB, KL, KR, Liver, PC, SP, SM
```

因此不能按表格从左到右直接和日志逐列比较。论文中的 PVT-EMCAD-B2 行为：

```text
Aorta 88.14, GB 68.87, KL 88.08, KR 84.10,
Liver 95.26, PC 68.51, SP 92.17, SM 83.92
```

你的第二次测试日志（小数乘以 100 后）为：

```text
spleen 88.84, right kidney 74.31, left kidney 86.08,
gallbladder 82.85, pancreas 95.41, liver 69.49,
stomach 90.72, aorta 85.44
```

按器官名称对齐后：

| 器官 | 论文 Dice (%) | 你的日志 Dice (%) | 直接差值（百分点） |
|---|---:|---:|---:|
| Aorta | 88.14 | 85.44 | -2.70 |
| GB | 68.87 | 82.85 | +13.98 |
| KL | 88.08 | 86.08 | -2.00 |
| KR | 84.10 | 74.31 | -9.79 |
| Liver | 95.26 | 69.49 | -25.77 |
| PC | 68.51 | 95.41 | +26.90 |
| SP | 92.17 | 88.84 | -3.33 |
| SM | 83.92 | 90.72 | +6.80 |

论文均值约为 83.63%，你的第二次均值为 84.1425%，总体反而高约 0.51 个百分点。因此“整体差很多”并不成立，主要异常是 Liver/PC 的语义或标签对应关系。

## 3. 已确认：Liver/PC 标签语义错位

我对项目实际的 `../data/Synapse/test_vol_h5` 和 `../data/Synapse/train_npz` 做了只读统计。测试集 12 个 H5 病例的标签只有 `0..8`，其中第 5 类累计约 9,762,041 个体素，第 6 类累计约 504,539 个体素；训练 NPZ 中第 5 类约 14,192,919 个体素，第 6 类约 651,861 个体素。第 5 类体素量远大于第 6 类，结合 Synapse 的器官大小，确定第 5 类是肝脏、第 6 类是胰腺。

你的日志第 5 类为 95.41%、第 6 类为 69.49%，与论文 Liver/PC 数值几乎正好交换：

```text
若第 5 类实际是 Liver，第 6 类实际是 Pancreas：
Liver   95.41 vs 95.26（+0.15）
Pancreas 69.49 vs 68.51（+0.98）
```

因此，当前代码的显示名称在 `test_synapse.py` 第 123-124 行确实写反：它把第 5 类写成 `pancreas`、第 6 类写成 `liver`，而实际数据是第 5 类 Liver、第 6 类 Pancreas。`utils/utils.py` 第 644-648 行按整数标签直接计算指标，不会自动交换名称，所以日志中的两行名称就是反的。

另外，`utils/preprocess_synapse_data.py` 和 `utils/preprocess_synapse_data_3d.py` 只是读取并保存标签，没有执行重映射；`utils/dataset_synapse.py` 中原本用于把原始标签 11（胰腺）映射到 5 的代码目前全部被注释。你当前使用的数据已经是外部生成的 `0..8` 标签，实际采用的是 `5=Liver, 6=Pancreas` 映射。因此不能再套用注释中“11→5”的旧映射，否则会再次改变语义。

### 只读核验标签

在服务器上检查测试 H5 的实际标签值：

```bash
python - <<'PY'
import glob, h5py, numpy as np
p = glob.glob('../data/Synapse/test_vol_h5/*.npy.h5')[0]
with h5py.File(p, 'r') as f:
    y = f['label'][:]
u, c = np.unique(y, return_counts=True)
print('file:', p)
print('labels:', dict(zip(u.astype(int).tolist(), c.tolist())))
PY
```

判断规则：

* 如果仍出现原始标签 `11`，说明预处理没有完成 8 器官映射；当前 9 通道训练/测试协议不正确。
* 当前文件已经确认是 `0..8`，且实际语义为 `5=Liver, 6=Pancreas`；应同步修正 `classes` 显示顺序，不能依据已注释的旧映射推断。

## 4. 更关键的实验协议问题：测试的不是第二次训练权重

第二次训练日志（2026-08-16）保存的目录名包含：

```text
..._pretrain_bs32_256_s2222/best.pth
```

第二次测试日志（2026-08-17）实际加载的目录名包含：

```text
..._pretrain_30k_bs6_256_s2222/best.pth
```

这两个 checkpoint 不是同一个训练过程：一个是 batch size 32，另一个是 30k 迭代、batch size 6。因而第二次测试结果不能作为“第二次 bs32 训练”的测试结果。必须让测试日志中的 `snapshot` 与训练日志最后一次 `save model to .../best.pth` 完全一致。

## 5. 224×224 与 256×256 不是唯一变量

论文设置和你的运行至少有以下差异：

| 项目 | 论文表格对应实验 | 你的日志 |
|---|---|---|
| 输入尺寸 | 224×224 | 256×256 |
| batch size | 6 | 32（训练日志）；测试加载的是 bs6 权重 |
| 随机性 | 论文报告多次运行平均 | 你的结果是单个 seed=2222 |
| checkpoint | 论文对应的训练协议 | 测试日志加载了另一套 30k/bs6 权重 |
| 训练/验证模式 | 需确保每轮验证后回到 train | 8 月运行时的旧代码曾缺少恢复；当前代码第 170 行已在每个 epoch 开头调用 `model.train()` |
| 表面距离指标 | 需要真实 voxel spacing | 当前测试调用传入 `z_spacing=1`，HD95/ASD 不是物理毫米值 |

256×256 只改变输入重采样分辨率，不能抵消 batch size、训练步数、随机种子、权重文件和标签映射的差异。对胆囊、胰腺等小器官，单次运行出现 5-15 个百分点波动并不罕见；但 Liver/PC 这种近似互换首先应按标签问题排查。

## 6. 建议的重跑顺序

1. 先用上面的命令核验 H5/NPZ 标签值和生成脚本，确认第 5、6 类的真实语义。
2. 测试时只加载第二次训练实际生成的 `..._pretrain_bs32_256_s2222/best.pth`，并把完整 snapshot 路径记录到日志。
3. 确认训练集和测试集来自同一套预处理、同一套类别映射，目录不要混用 `Synapse` 与 `synapse`、`test_vol_h5` 与 `test_vol_h5_new`。
4. 使用当前包含每轮开头 `model.train()` 的代码重新训练；8 月 16 日的旧结果不能证明该修复后的行为。
5. 先比较 Dice/Jaccard 和每病例结果，再单独处理真实 spacing 下的 HD95/ASD；不要用 HD95/ASD 的数值差异反推模型 Dice 复现失败。
6. 若要与论文严格对比，固定 224×224、batch size=6、相同预训练权重和监督方式，并至少用多个 seed 重复，而不是把 256×256、bs32 的单次结果直接对照论文均值。

结论：当前结果的总体水平已经接近论文；代码确实把 Liver/PC 名称写反，必须先修正报告名称，再比较模型差异。另一个独立问题仍是 checkpoint 对应关系，而不是先怀疑 256×256 本身。
