# EMCAD 除 Synapse 外数据集融合与 CAA 接入说明

## 结论与范围

本次把“像素级可靠性多头融合”和“内容感知抗混叠上采样（CAA）”接入 Synapse 以外的数据入口。Synapse 的 Python 文件和启动脚本均未修改，共享 EMCAD 编码器、解码器和输出头也未改动；接入工作只复用现有 `fusion_mode` / `caa_mode` 接口，在二分类训练、测试、数据适配和 `.sh` 参数处补齐缺少的连接。

覆盖入口如下：

| 数据集 | 入口和本次处理 | 关键约束 |
|---|---|---|
| ACDC | 保留既有融合/CAA训练接线；测试按 checkpoint 配置恢复结构，并修正清单目录、测试脚本大小写、GPU 数量变量和 Python/Conda 初始化 | 当前启动器默认 256，而论文描述 224；若按论文尺寸复现，训练和测试都设置 `IMG_SIZE=224`。服务器实测仍待完成 |
| BKAI、ClinicDB、ColonDB、ETIS、Kvasir | 接入通用二分类 EMCAD 训练、融合推理与 CAA；原始 Polyp 监督损失、增强、阈值和指标流程不变 | 默认输入 352，多尺度倍率 0.75、1.0、1.25 |
| ISIC2017、ISIC2018 | 通过 ISIC 包装器复用通用二分类融合和 CAA | 保留既有 split manifest；不重做 ISIC2017 官方划分 |
| BUSI | 按 EMCAD 论文的数据集范围一并接入。虽然本次需求括号中没有列出 BUSI，但本地论文也包含 BUSI，现有专用加载器和 manifest 校验继续使用 | 保留现有 BUSI split summary、manifest 和固定 256 输入 |
| DSB18、EM | 通过通用二分类入口接入；DSB18 可把单图多个实例掩膜合并为一个语义前景 | 论文报告 256 输入及 80:10:10 划分；代码要求用户预先准备对应 train/val/test，不生成或猜测划分 |

EMCAD 论文来源是 Rahman、Munir、Marculescu 的 *EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation*（CVPR 2024）。本地论文列出的数据还包括 BUSI，因此这里按“论文中所有数据集”一并接入；若本课题的明确范围不含 BUSI，可在实验报告中说明，本次代码已为 BUSI 开放这两个模块。

论文正文还明确区分输入协议：五个 Polyp 数据集和 ISIC17/18 使用 352 输入及 0.75/1.0/1.25 多尺度训练；BUSI、EM、DSB18 使用 256；ACDC 使用 224。当前既有 ACDC 启动器默认 256，本次只保证训练/测试尺寸一致，没有擅自更改此前 ACDC 实验协议。需要对齐论文时，应在训练和测试两边都设置 `IMG_SIZE=224`，并将实际值记录在实验配置中。

## 两项模块的实际接线

### 像素级可靠性多头融合

EMCAD 仍返回 `[p4, p3, p2, p1]` 四个同尺寸 logits。`fusion_mode=pixel_reliability` 时，四个可靠性头输出逐像素权重，沿四个预测头归一化后得到融合 logits。训练中的原二分类监督仍按原有 `supervised_structure_loss` 计算；只有 `fusion_loss_weight > 0` 且融合模式不是 `p1` 时，才额外加入融合辅助项。

多分类的 `EMCADNet.fusion_auxiliary_loss` 当前采用 `0.3*CE + 0.7*Dice`，并在像素可靠性模式加入正确性 BCE。单通道二分类不能直接复用多分类 CE、softmax 和 `argmax` 正确性标签，因此二分类扩展对应使用 `BCEWithLogits + sigmoid soft Dice`，Dice 沿用平方和分母和平滑常数 `1e-5`；可靠性标签按每头 `sigmoid(logit) >= 0.5` 与二值真值比较生成。该辅助项再乘 `fusion_loss_weight`。这是将当前多分类融合损失适配到一个前景通道的实现定义，不是 EMCAD 论文原始损失的新主张。

`fusion_mode=p1` 时模型不创建可靠性头，测试仍取 `outputs[-1]`，训练不增加融合辅助项，因此关闭融合后回到原单头推理和原二分类监督路径。

### 内容感知抗混叠上采样

`caa_mode` 由 `build_model` 传入现有 EMCAD EUCB，作用于三级解码上采样，不改输出头、编码器、MSCB、LGAG 或原数据处理。可用模式为：

| `caa_mode` | EUCB 上采样行为 |
|---|---|
| `off` | 原 EMCAD 上采样路径 |
| `aa_only` | 固定低通抗混叠路径 |
| `content_only` | 最近邻基底加内容门控残差 |
| `caa` | 抗混叠基底加内容门控残差 |

主消融建议保持 `CAA_RESIDUAL_SCALE=0.1` 不变，只切换 `USE_CONTENT_AWARE_ANTIALIAS`。这与 `docs/像素可靠性融合与内容感知抗混叠上采样组合候选方法_严格实验报告_20260928.md` 中原有消融口径一致。

## 启动脚本参数

非 Synapse 训练启动脚本默认打开两项模块。需要关闭其中一项时，在启动前覆盖对应变量：

```bash
USE_PIXEL_RELIABILITY_FUSION=0 bash sh/start_train_polyp.sh
USE_CONTENT_AWARE_ANTIALIAS=0 bash sh/start_train_polyp.sh
USE_PIXEL_RELIABILITY_FUSION=0 USE_CONTENT_AWARE_ANTIALIAS=0 bash sh/start_train_polyp.sh
```

相同开关也已加入 ACDC、ISIC、BUSI 的训练启动器。关闭融合会把 `fusion_mode` 设为 `p1` 并把 `fusion_loss_weight` 设为 `0`；关闭 CAA 会把 `caa_mode` 设为 `off`。开启融合时可覆盖 `FUSION_LOSS_WEIGHT` 和 `RELIABILITY_LOSS_WEIGHT`；CAA 组件消融可设置 `CAA_MODE=aa_only` 或 `content_only`。每次训练会把有效参数记入日志和 checkpoint 同目录的 `config.json`。

细胞数据现有专用 Cell 包装启动器；训练循环仍复用 `train_polyp.py` 的通用二分类实现。论文对 DSB18 和 EM 使用 256 输入、200 轮、batch size 16，启动器以这些值为默认设置，单卡数 `N_GPU` 默认是 1；仍可通过同名环境变量显式覆盖。`DATA_ROOT` 应指向包含 `DSB18` 和 `EM` 子目录的已准备数据根：

```bash
DATA_ROOT=/path/to/cell/target DATASET_NAME=DSB18 bash sh/start_train_cell.sh
```

DSB18 默认打开实例掩膜并集合并；如果已提前把每张图的实例掩膜合并为同名语义掩膜，可设置 `MERGE_INSTANCE_MASKS=0`。EM 默认按一图一掩膜读取；若图像应按灰度单通道读取，设置 `INPUT_CHANNELS=1`。

队列连续训练 DSB18 的多个 seed：

```bash
DATA_ROOT=/path/to/cell/target DATASET_NAME=DSB18 \
bash sh/run_seed_queue.sh cell 2222 3407 5678
```

把 `DATASET_NAME` 换成 `EM` 可连续运行 EM。队列每轮都会等待当前训练进程退出后再启动下一个 seed。SSH 断开后继续运行可执行：

```bash
DATA_ROOT=/server/data/cell/target DATASET_NAME=DSB18 \
nohup bash sh/run_seed_queue.sh cell 2222 3407 5678 >/dev/null 2>&1 < /dev/null &
echo "QUEUE_PID=$!"
```

队列会自行创建 `logs/seed_queue_cell_*.log` 记录 seed 开始、完成和失败状态；`>/dev/null` 避免再额外复制一份相同的队列终端输出。用 `ls -t logs/seed_queue_cell_*.log | head -n 1` 找到最新队列日志，再用 `tail -f <日志路径>` 跟踪。训练 PID 可通过 `bash sh/stop_train_cell.sh <RUN_ID>` 停止；队列 PID 可通过 `kill -TERM <QUEUE_PID>` 停止，队列会向正在运行的训练进程转发终止信号。两类标识均可从启动输出或对应日志中查到。

测试时将 `CKPT` 指向训练实验目录中的 `best.pth`；目录内的 `config.json` 会恢复编码器、输入通道和 DSB18 掩膜解释设置：

```bash
DATA_ROOT=/path/to/cell/target DATASET_NAME=DSB18 \
CKPT=/path/to/Cell/experiment/best.pth bash sh/start_test_cell.sh
```

停止测试可使用 `bash sh/stop_test_cell.sh <RUN_ID>`。训练和测试脚本均保持通用二分类入口已有的数据目录与掩膜语义要求，不自动下载原始数据、不生成 80:10:10 划分，也不将 DSB18 的实例标签转换成实例分割指标。

Polyp、ISIC 和 BUSI 测试启动器默认从 checkpoint 同目录 `config.json` 恢复训练时的融合、CAA、输入尺寸和掩膜解码方式。若显式设置测试开关，指定值必须与 checkpoint 一致；否则会报配置冲突，避免用不匹配的网络解释权重。ACDC 测试同样校验训练配置，并默认采用与当前训练启动器一致的 256 输入。测试结果不用于重新选 checkpoint。

## 数据目录要求与歧义

通用 Polyp/细胞入口要求每个数据集使用以下固定布局，文件主体名必须一一对应：

```text
<DATA_ROOT>/<DATASET_NAME>/train/images/<stem>.<ext>
<DATA_ROOT>/<DATASET_NAME>/train/masks/<stem>.<ext>
<DATA_ROOT>/<DATASET_NAME>/val/images/<stem>.<ext>
<DATA_ROOT>/<DATASET_NAME>/val/masks/<stem>.<ext>
<DATA_ROOT>/<DATASET_NAME>/test/images/<stem>.<ext>
<DATA_ROOT>/<DATASET_NAME>/test/masks/<stem>.<ext>
```

DSB18 开启 `MERGE_INSTANCE_MASKS=1` 后，也接受 `masks/<stem>/<instance-mask>.<ext>`：同一目录内所有非零实例像素取并集。该目标是“任意细胞前景”的语义二分类，不保留实例编号或实例边界。**如果要严格评估实例分割指标，当前单通道二分类指标流程不适用。**

EM 的具体本地文件结构和掩膜编码没有出现在当前 checkout，因此本次按同 stem 的一图一掩膜（灰度 0/1、0/255 或常见 TIFF/PNG）接入；代码不会把未知原始目录自动转换成上述结构，也不会创建划分。DSB18 原始数据也不在本机，代码只支持已经整理成 `images/<stem>.<ext>` 与 `masks/<stem>/<instance>.<ext>`（或合并后同 stem 文件）的形式。论文采用 DSB18、EM 的 80:10:10 划分；服务器上的实际目录、预处理、图像/患者分组及清单必须核实与论文协议的对应关系，不能仅凭目录名宣称复现。当前 checkout 的 `../data/polyp/target` 存在五个息肉数据集目录，但没有本地 cell 数据。

ACDC 的本地清单目录已核实为 `../data/ACDC/lists_ACDC`，含 `train.txt`、`valid.txt`、`test.txt`；训练/测试脚本默认统一到该目录。ISIC 与 BUSI 继续由已有专用加载器校验 manifest。所有新入口均要求复用已经确定的数据划分，不会调用 split 生成器。

## 验证状态和待知情事项

验证方面，`train_polyp.py` 与 `test_polyp.py` 的 Python 语法编译、`git diff --check` 已通过；此前一次检查覆盖过仓库内 25 个 `.sh` 文件。随后新增了 Cell 的 `N_GPU=1` 默认值并调整了通用验证进度标题，但当前 Windows 会话无法启动 Bash/WSL，因此未能在这两处小改动后重新运行 Bash 语法检查或队列 mock。当前 Python 环境未安装 PyTorch、OpenCV 或 Albumentations，没有执行真实 EMCAD 前向/反向或图像加载；服务器上的 DSB18/EM 数据仍待核验。ACDC 代码路径已完成静态检查，但训练/测试尚未在服务器上实测。

当前没有 Dice/HD95 结果。本次改动只表明入口和参数可以按代码契约接线，**效果尚未验证**；不能据此宣称任一数据集提升。正式比较仍需按同一固定清单、相同 seed 配对，报告验证集 checkpoint 选择依据和逐病例结果。

需要特别知情的两点：

1. EMCAD 论文还包含 BUSI，故本次虽未在需求枚举中看到 BUSI，仍按“所有论文数据集”扩展了 BUSI 启动参数。
2. 旧说明 `docs/像素可靠性融合与内容感知抗混叠上采样组合候选方法_严格实验报告_20260928.md` 写有融合 Dice 直接吃 logits、未加 softmax；但当前 `lib/networks.py` 多分类实现实际调用 `dice_loss(..., softmax=True)`。本次按当前可执行代码实现多分类口径，并为单通道任务采用 sigmoid soft Dice；旧说明文件未被覆盖。
