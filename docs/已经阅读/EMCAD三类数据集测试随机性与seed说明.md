# EMCAD 三类数据集测试随机性与 seed 说明

## 结论先看

在当前 checkout 中，Synapse、ACDC 以及 Polyp 系列数据集的正式测试路径都属于“固定输入 + 固定 checkpoint + eval 推理”。测试时没有启用随机旋转、随机翻转、随机裁剪、随机采样或 TTA，因此同一环境下重复执行，预测结果和指标通常应保持一致。

执行测试时：

- 对一个已经完整加载的固定 checkpoint，seed 不是“模型能否正常预测”的硬性条件。
- 但建议显式设置并记录 seed=2222，同时打开 deterministic 模式。这能固定模型构造阶段、DataLoader worker 以及将来可能加入的随机处理，也让实验记录完整。
- 真正需要多个 seed 重复运行来估计方差的主要是训练阶段；固定 checkpoint 的纯测试一般不需要用多个 seed 重复。

这里的“结果一致”默认指相同代码、相同 checkpoint、相同数据文件和划分、相同输入尺寸、相同 Python/PyTorch/CUDA/cuDNN 环境以及相同 GPU。换 GPU 或软件栈后，浮点计算和底层并行算法仍可能带来极小差异。

## 当前三条测试路径

### 1. Synapse

实际测试入口是 [test_synapse.py](../test_synapse.py)：

- DataLoader 使用 batch_size=1、shuffle=False、num_workers=1，病例顺序来自固定的 test_vol.txt。
- 测试集调用 Synapse_dataset(..., split="test_vol")，没有传入训练用的 RandomGenerator。RandomGenerator 中的随机旋转和翻转只会在训练数据路径中使用。
- model.eval() 关闭 Dropout 的随机失活，并让 BatchNorm 使用已保存的运行统计量。
- test_single_volume() 按深度方向固定顺序逐切片推理，使用 torch.no_grad()，取网络输出 P[-1]，执行 softmax + argmax。
- 主程序默认 --seed 2222、--deterministic 1，并设置 Python、NumPy、PyTorch、CUDA seed 以及 cudnn.deterministic=True、cudnn.benchmark=False。

需要特别注意 [start_test_synapse.sh](../start_test_synapse.sh) 的一个配置细节：

~~~bash
SEED=2222
...
nohup env RUN_ID="${RUN_ID}" python test_synapse.py \
  --volume_path "${VOLUME_PATH}" \
  --dataset "${DATASET}" \
  --img_size "${IMG_SIZE}" \
  --list_dir "${LIST_PATH}" \
  >> "${LOG_FILE}" 2>&1 &
~~~

这个 shell 变量目前只用于 RUN_ID 的命名，没有传给 Python 的 --seed 和 --deterministic。由于 Python 默认值恰好也是 2222 和 1，默认运行时没有实际差异；但是如果只修改 shell 中的 SEED，Python 实际仍会使用 test_synapse.py 的默认 2222。需要改变 Synapse 测试 seed 时，应直接显式运行 Python，或在启动脚本命令中补充：

~~~bash
  --seed "${SEED}" \
  --deterministic 1 \
~~~

脚本中的 RAND="$(head -c 6 /dev/urandom ...)" 只用于生成唯一日志名和 RUN_ID，不参与模型计算，不会让预测结果随机。

### 2. ACDC

实际测试入口是 [test_acdc.py](../test_acdc.py) 和 [utils/acdc_utils.py](../utils/acdc_utils.py)：

- 主函数先调用 seed_everything(args.seed, deterministic=True)。
- 测试集使用 split="test"，DataLoader 为 batch_size=1、shuffle=False；启动脚本默认 num_workers=0。
- predict_volume() 使用 model.eval() 和 torch.no_grad()，按固定深度顺序分批处理切片。
- 没有测试时随机增强，也没有 TTA；输出为最终 logits 的 argmax 类别图。

因此，ACDC 测试结果不依赖每次重新抽取随机数。--seed 主要是为了统一脚本行为、固定模型构造和 worker 初始化，并方便复现实验。

注意当前 [`start_test_acdc.sh`](../start_test_acdc.sh) 写的是 `test_ACDC.py`，但仓库中的文件实际名为 `test_acdc.py`。Linux 区分文件名大小写，因此该启动脚本在 Linux 上可能找不到测试文件；这与随机性无关，但会阻止测试启动。直接运行时请使用实际文件名 `python test_acdc.py ...`。ACDC 的 Python 代码固定调用 `seed_everything(args.seed, deterministic=True)`，命令行只接受 `--seed`，不接受额外的 `--deterministic` 参数。

### 3. Polyp 系列五个数据集

当前通用加载器明确覆盖的名称包括 ClinicDB、Kvasir、ColonDB、ETIS 和 BKAI（具体以 data/polyp/target/<DATASET_NAME>/ 下实际目录为准）。测试入口为 [test_polyp.py](../test_polyp.py)，加载器为 [utils/dataloader_polyp.py](../utils/dataloader_polyp.py)：

- 测试/验证使用 split=val 或 split=test，并且 shuffle=False。
- 文件按 stem 排序，图像和 mask 按 stem 严格配对，样本顺序固定。
- augmentation=False；加载器中的 Rotate、VerticalFlip、HorizontalFlip 只在 split="train" and augmentation=True 时启用。
- 默认 num_workers=0；即使提高 worker 数，当前测试也只执行固定的 Resize、Normalize 和张量转换。
- 测试阶段调用 model.eval() 和 torch.no_grad()，取 outputs[-1]，经过 sigmoid、尺寸恢复、逐图 min-max 归一化和固定阈值（默认 0.5）得到二值 mask。
- 仓库中虽然保留了一个旧的 tta_model() 辅助函数，但当前 test_polyp.py 正式评估路径没有调用它。

test_polyp.py 会显式调用 seed_everything(args.seed, deterministic=args.deterministic)，并把同一个 seed 传给 Polyp DataLoader 的 torch.Generator。这是良好的复现设计，但在当前 shuffle=False 且关闭增强的测试路径中，seed 通常不会改变预测。

## 为什么测试时通常没有随机性

需要区分两件事：

1. 代码是否调用了随机数。当前测试路径没有主动随机增强或随机采样。
2. GPU 算子是否严格位级确定。即便没有随机数，某些 CUDA/cuDNN 并行归约或原子操作也可能存在非位级确定性。

当前代码已经设置：

~~~python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
~~~

但没有统一调用 torch.use_deterministic_algorithms(True)。因此应表述为“同一服务器和同一软件环境下通常可稳定复现”，不要承诺跨不同 GPU、驱动、CUDA、cuDNN 或 PyTorch 版本逐位完全相同。若研究需要更强的确定性，可以在确认所有算子支持后额外启用该开关，但这可能报不支持确定性算法的错误，也可能降低速度。

## seed 到底是否必须

### 可以不设置的情况

如果满足以下条件，固定 checkpoint 的测试通常可以不依赖 seed：

- checkpoint 完整且严格加载成功；
- 模型已经切换到 eval()；
- 测试没有随机增强、随机裁剪、随机 TTA 或随机后处理；
- 数据顺序、预处理、输入尺寸和阈值固定。

此时模型构造时的随机初始化参数会被 checkpoint 覆盖，随机状态不会进入正常前向计算。

### 仍然建议设置的原因

建议保留 --seed 2222 --deterministic 1，原因是：

- 防止 checkpoint 不完整、加载不严格或未来代码变更后留下随机初始化参数；
- 固定 DataLoader generator 和 worker 的初始状态；
- 防止以后加入 TTA、随机采样或其他随机后处理时实验悄悄失去可复现性；
- 让日志、配置和论文实验记录可以准确复现。

因此，推荐说法是：**测试推理不以 seed 为功能前提，但实验复现应显式设置并记录 seed。**

## 推荐运行方式

### Synapse

不要只改 start_test_synapse.sh 里的 SEED 后就认为 Python 收到了新 seed。需要改变或明确记录参数时，直接在 Python 命令中写出：

~~~bash
python test_synapse.py \
  --volume_path ../data/Synapse/test_vol_h5 \
  --list_dir ../data/Synapse/lists/lists_Synapse \
  --img_size 256 \
  --seed 2222 \
  --deterministic 1
~~~

实际 checkpoint 路径仍按当前 test_synapse.py 中的实验命名规则准备。

### ACDC

~~~bash
SEED=2222 \
CKPT=/absolute/path/to/best.pth \
bash start_test_acdc.sh
~~~

该启动脚本的命令会传递 SEED，但应先将其中的 `test_ACDC.py` 更正为实际文件名 `test_acdc.py`，否则 Linux 会启动失败。直接运行 Python 时可显式写 `--seed 2222`；ACDC 的确定性模式已在代码中固定开启。

### Polyp

~~~bash
SEED=2222 \
DATASET_NAME=ClinicDB \
SPLIT=test \
CKPT=/absolute/path/to/best.pth \
bash start_test_polyp.sh
~~~

更换数据集时只改变 DATASET_NAME，例如 ColonDB、ETIS、Kvasir 或 BKAI，同时确认对应目录和 checkpoint 的训练配置一致。

## 建议做一次的复现验证

可以对同一个 checkpoint 连续运行两次，并比较：

1. 最终 test_metrics.csv 中的逐病例指标和均值；
2. 预测 mask 文件的 SHA-256；
3. 日志中的 checkpoint、数据根目录、split、输入尺寸、阈值和实际设备。

若同一环境下两次 mask 哈希一致，说明当前测试链路达到了很强的工程复现性。若只改变 seed（例如 2222 与 7）而其他配置完全相同，固定 checkpoint 的结果也应通常一致；若不一致，应优先排查：是否误开启增强/TTA、是否修改了输入尺寸或阈值、是否使用了不同 checkpoint、是否存在未严格加载的参数，以及 GPU/软件环境是否发生变化。

## 训练阶段和测试阶段的区别

| 阶段 | 随机性的主要来源 | 是否建议多个 seed |
| --- | --- | --- |
| 训练 | 参数初始化、数据 shuffle、随机增强、dropout、CUDA 算子 | 建议在最终实验中使用多个 seed 估计均值和方差；至少固定一个 seed 做基线复现 |
| 固定 checkpoint 测试 | 当前路径基本没有主动随机操作；主要剩余风险是底层 GPU 非确定性 | 通常不需要多个 seed；固定并记录一个 seed 即可 |

论文或实验报告中建议记录：代码 commit、checkpoint、数据划分清单、输入尺寸、预处理、阈值、GPU、Python/PyTorch/CUDA/cuDNN 版本、测试命令以及 seed。测试集应只用于最终报告，checkpoint 选择仍应在验证集完成。
