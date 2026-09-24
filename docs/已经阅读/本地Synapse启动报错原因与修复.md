# 本地 Synapse 启动报错原因与修复

## 结论先说

截图里有两个问题，但真正让程序退出的是第二个：

1. **NumPy 被用户目录中的 2.2.6 抢先加载，与当前 PyTorch 1.11.0 的 NumPy 1.x 二进制接口不兼容。** 这在截图中先表现为 `UserWarning: Failed to initialize NumPy: _ARRAY_API not found`。它暂时是警告，不是最后那条致命异常，但后续调用 `torch.from_numpy()` 或 `tensor.numpy()` 很可能失败，必须修复。
2. **训练脚本拼出的 `snapshot_path` 太长。** 当前完整路径长度为 339 个字符，`os.makedirs(snapshot_path)` 在 Windows 上报 `FileNotFoundError: [WinError 3] 系统找不到指定的路径`。这才是本次进程退出的直接原因。

当前不是 CUDA 不可用，也不是模型导入失败。已经实际确认：

```text
Python       3.10.19
PyTorch      1.11.0+cu113
CUDA         11.3（PyTorch 内置运行时）
CUDA 可用    True
环境内 NumPy 1.22.4
用户目录 NumPy 2.2.6（被优先加载）
```

用隔离用户目录的方式测试后，下面这条链路正常：

```text
NumPy 1.22.4 -> torch 1.11.0 -> CUDA True -> torch.from_numpy 正常
```

---

## 一、第一段报错：NumPy 版本不是当前环境真正使用的版本

截图中的核心文字是：

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
...
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

### 实际加载路径

这个环境目录中已经有：

```text
D:\install\miniconda3\envs\SLDGroup_EMCAD_env\lib\site-packages\numpy
版本：1.22.4
```

但是当前 Python 默认开启用户级 site-packages，并把下面路径放在环境 site-packages 前面：

```text
C:\Users\lzq\AppData\Roaming\Python\Python310\site-packages
版本：2.2.6
```

所以虽然命令提示符显示已经激活 `SLDGroup_EMCAD_env`，真正执行 `import numpy` 时却加载了用户目录的 2.2.6。

这解释了为什么“环境里明明有 NumPy 1.22.4，仍然出现 NumPy 2.2.6 报错”。**激活 Conda 环境并不自动屏蔽用户级 Python 包。**

### 为什么 PyTorch 1.11 不适合 NumPy 2

当前 `torch==1.11.0+cu113` 发布时，PyTorch 的 NumPy 互操作代码是按 NumPy 1.x 的 C/API 编译和测试的。NumPy 2 改变了相关 API，因而出现 `_ARRAY_API not found`。

这不等于 `import torch` 立刻失败。当前日志就是先发出警告，再继续执行。但一旦数据集代码执行：

```python
torch.from_numpy(array)
```

或者：

```python
tensor.numpy()
```

可能出现 `RuntimeError: Numpy is not available` 或类似错误。因此不能把这条警告当作可以永久忽略的小提示。

---

## 二、第二段报错：实验目录超过 Windows 路径限制

截图最后的异常是：

```text
FileNotFoundError: [WinError 3] 系统找不到指定的路径。
```

触发位置是当前 [`train_synapse.py`](../train_synapse.py) 的：

```python
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
```

当前代码为了记录所有超参数，把编码器、卷积核、并行模式、聚合方式、LGAG、扩张倍数、激活、监督方式、数据集、输入尺寸、预训练、batch、seed 全部塞进目录名。最终路径大致是：

```text
model_pth/
  pvt_v2_b2_EMCAD_kernel_sizes_1_3_5_dw_parallel_add_lgag_ks_3_ef2_act_mscb_relu6_loss_mutation_output_final_layer_Run1_Synapse224/
    pvt_v2_b2_EMCAD_kernel_sizes_1_3_5_dw_parallel_add_lgag_ks_3_ef2_act_mscb_relu6_loss_mutation_output_final_layer_Run1_pretrain_bs6_224_s2222
```

从当前工作区测得：

```text
项目绝对路径 + 完整 snapshot_path = 339 个字符
外层实验目录已经存在
内层目录尚未创建
```

Windows 传统 Win32 路径限制通常约为 260 个字符。Python 在创建这个深层目录时，常见表现就是 `WinError 3`，即使 `model_pth` 本身和外层目录都存在。

因此，这里的“系统找不到路径”不是说你的项目根目录真的不存在，而是 Windows 无法按当前路径长度解析或创建最后的目录。

---

## 三、按这个顺序修复

### 第 1 步：本次 PowerShell 会话屏蔽用户级包

在 VS Code/PyCharm 的集成终端执行：

```powershell
$env:PYTHONNOUSERSITE = "1"
```

这个变量只影响当前终端及其子进程，不会删除你用户目录里的其他包，也不会破坏别的项目。

然后用**实际执行训练的解释器**检查：

```powershell
$PY = "D:\install\miniconda3\envs\SLDGroup_EMCAD_env\python.exe"

& $PY -s -c "import sys, numpy, torch; print('python =', sys.executable); print('numpy =', numpy.__version__, numpy.__file__); print('torch =', torch.__version__); print('cuda =', torch.cuda.is_available()); print(torch.from_numpy(numpy.zeros((2,2), dtype=numpy.float32)).shape)"
```

预期应看到：

```text
numpy = 1.22.4 D:\install\miniconda3\envs\SLDGroup_EMCAD_env\lib\site-packages\numpy\__init__.py
torch = 1.11.0+cu113
cuda = True
torch.Size([2, 2])
```

其中 `-s` 也会临时禁用用户级 site-packages；环境变量和 `-s` 同时使用，是为了让“终端启动”和“单条验证命令”都明确。

### 第 2 步：训练启动时也使用同一规则

若从当前 PowerShell 直接启动：

```powershell
$env:PYTHONNOUSERSITE = "1"
& $PY -s .\train_synapse.py
```

如果你点击 IDE 的运行按钮，确认 IDE 运行配置使用的是：

```text
D:\install\miniconda3\envs\SLDGroup_EMCAD_env\python.exe
```

并在该运行配置的环境变量中加入：

```text
PYTHONNOUSERSITE=1
```

否则 IDE 仍可能再次加载 `C:\Users\lzq\AppData\Roaming\Python\Python310\site-packages\numpy`。

当前环境已经有可用的 NumPy 1.22.4，**不需要因为这张截图立刻重装 PyTorch，也不建议先动 CUDA。** 若另一个环境没有兼容 NumPy，再按对应环境单独安装 `numpy<2`；不要把用户级 NumPy 2.2.6 当成所有项目都必须删除的对象。

### 第 3 步：缩短训练输出目录名

这一步是解决本次实际退出的必要步骤。推荐把实验目录改成短 ID，把完整超参数放到日志或 `config.json`，不要全部放进 Windows 文件夹名。

可采用类似下面的短路径逻辑：

```python
run = 1
args.exp = f"emcad_synapse_{args.encoder}_run{run}"
snapshot_path = os.path.join(
    "model_pth",
    args.exp,
    f"img{args.img_size}_bs{args.batch_size}_seed{args.seed}",
)
os.makedirs(snapshot_path, exist_ok=True)
```

这会把目录控制在很短的范围内；完整配置仍应写入实验记录。注意：[`test_synapse.py`](../test_synapse.py) 当前会按旧命名规则重新拼接 checkpoint 路径。如果只修改训练目录而不同步修改测试脚本，训练可能能启动，但测试找不到 `best.pth`。因此更稳妥的长期改法是让训练和测试都显式接收 `--snapshot_path` 或 `--checkpoint`。

如果暂时不想改代码，也可以在 `train_synapse.py` 中把当前长目录名逻辑临时替换为：

```python
snapshot_path = os.path.join("model_pth", f"run_seed{args.seed}")
os.makedirs(snapshot_path, exist_ok=True)
```

但要记住同步记录本次配置，否则不同实验会覆盖同一目录。

### 第 4 步：重新做最小启动测试

不要一上来就跑 300 epoch。先确认目录、导入、前向和一个极短训练流程：

```powershell
$env:PYTHONNOUSERSITE = "1"
& $PY -s -c "import torch, numpy; from lib.networks import EMCADNet; m=EMCADNet(num_classes=9, encoder='pvt_v2_b2', pretrain=False); m.cuda().eval(); x=torch.randn(1,1,224,224,device='cuda'); y=m(x); print(type(y), len(y) if isinstance(y,list) else 'single', [tuple(t.shape) for t in y] if isinstance(y,list) else tuple(y.shape))"
```

如果这一步通过，再启动训练。这样可以把“环境错误”“模型导入错误”“目录错误”“数据加载错误”分开，不会让一个长日志混在一起。

---

## 四、不要采用的误判和处理方式

### 误判 1：看到 NumPy 警告就认定 PyTorch 没装好

不对。当前 PyTorch 已成功导入，CUDA 也可用。问题是 NumPy 的**加载来源**错了，不是 PyTorch 安装包一定损坏。

### 误判 2：看到 `WinError 3` 就去重新下载数据集

不对。本次异常发生在 `os.makedirs(snapshot_path)`，还没有进入 DataLoader 读取训练数据的阶段。先缩短输出路径。

### 误判 3：只执行 `conda activate` 就认为用户包不会干扰

不对。当前实际证据正好表明，激活环境后仍加载了用户目录 NumPy 2.2.6。项目级 Python 环境应显式控制 `PYTHONNOUSERSITE`，并记录 `numpy.__file__`。

### 误判 4：把两个问题混成一个版本冲突

两者独立：

```text
NumPy 问题：影响 NumPy/PyTorch 互操作，先是警告，后续可能在数据转换时报错
路径问题：影响 Windows 创建实验输出目录，本次直接导致程序退出
```

即使修好 NumPy，长路径仍会报错；即使缩短路径，NumPy 2.2.6 仍不应与 torch 1.11 长期混用。两个问题都要处理。

---

## 五、最终判断

本次启动失败的因果顺序是：

```text
IDE 使用 SLDGroup_EMCAD_env 的 Python
    -> 用户级 site-packages 抢先加载 NumPy 2.2.6
    -> torch 导入时发出 _ARRAY_API 警告
    -> train_synapse.py 继续运行
    -> 生成 339 字符的 snapshot_path
    -> os.makedirs 在 Windows 上报 WinError 3
    -> 进程退出
```

最短修复路线是：

```text
1. 设置 PYTHONNOUSERSITE=1，确认加载环境内 NumPy 1.22.4；
2. 把 snapshot_path 改成短实验 ID；
3. 同步让 test_synapse.py 使用新的 checkpoint 路径；
4. 先做 import/CUDA/forward/短训练，再跑完整训练。
```

相关代码位置：[`train_synapse.py`](../train_synapse.py) 的实验目录生成与 `os.makedirs`，以及 [`test_synapse.py`](../test_synapse.py) 的 checkpoint 路径重建。
