# EMCAD Synapse 模型输出目录改造方案

## 目标

把当前很长的实验输出路径改成下面这种结构：

```text
项目根目录/
├─ model_pth/
│  └─ synapse/
│     └─ run1_seed2222/
│        ├─ best.pth
│        ├─ last.pth
│        ├─ epoch_49.pth
│        ├─ epoch_99.pth
│        ├─ epoch_149.pth
│        ├─ epoch_199.pth
│        ├─ epoch_249.pth
│        ├─ epoch_299.pth
│        ├─ log.txt
│        ├─ config.json             （可选，建议保留）
│        ├─ log/                    （TensorBoard 事件文件）
│        ├─ predictions/            （测试产生的 PNG/NIfTI）
│        └─ test_logs/              （测试日志）
```

也就是说，所有 Synapse 训练和测试产物都位于：

```text
model_pth/synapse/<实验短名称>/
```

这里保留一层 `<实验短名称>` 很重要。如果把所有运行都直接写到 `model_pth/synapse/` 根目录，多次训练会互相覆盖 `best.pth`、`last.pth`、`log.txt` 和 TensorBoard 文件。推荐目录短名使用：

```text
run1_seed2222
run2_seed3333
baseline_224_s2222
cab_ablation_s2222
```

不要再把所有超参数塞入文件夹名；完整超参数写入 `config.json` 和 `log.txt`。

---

## 一、当前代码为什么会生成长路径

当前 [`train_synapse.py`](../train_synapse.py) 先构造很长的 `args.exp`，再构造一层长目录和一层更长目录：

```text
model_pth/<很长的 args.exp>/<再次重复很多超参数的目录名>
```

例如编码器、kernel_sizes、dw、add、LGAG、expansion_factor、activation、supervision、Run、数据集、输入尺寸、pretrain、batch、seed 都被写入路径。当前 [`trainer.py`](../trainer.py) 接收到这个路径后，会在其中写入：

- `log.txt`；
- `log/` TensorBoard 事件目录；
- `best.pth`；
- `last.pth`；
- 每 50 epoch 的 `epoch_*.pth`。

因此真正需要改的是入口中的 `snapshot_path`，不是逐个修改每个 `.pth` 保存语句。

当前 [`test_synapse.py`](../test_synapse.py) 又复制了一份旧的长路径拼接逻辑。若只修改训练入口而不修改测试入口，训练会把权重写到短目录，但测试仍会去旧长目录找 `best.pth`，最后报 checkpoint 不存在。下面的方案会同时处理这两个入口。

---

## 二、修改 `train_synapse.py`

### 2.1 增加两个命令行参数

在现有 `parser.add_argument(...)` 区域中，加入下面两项。建议放在 `--list_dir` 或训练超参数附近：

```python
parser.add_argument(
    '--snapshot_root',
    type=str,
    default='./model_pth/synapse',
    help='root directory for Synapse training outputs',
)
parser.add_argument(
    '--run_name',
    type=str,
    default='run1_seed2222',
    help='short and unique experiment directory name',
)
```

含义是：

```text
snapshot_root = 所有 Synapse 实验的根目录
run_name      = 当前这一轮实验的短目录名
```

### 2.2 替换旧的长目录生成代码

在当前 `train_synapse.py` 中，找到从下面这行开始的整段代码：

```python
run = 1
```

直到下面这段结束：

```python
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
```

这一整段旧逻辑都不要保留。它包括：

- 拼接长 `args.exp` 的代码；
- `snapshot_path = "model_pth/{}/{}"...`；
- `replace('[', ...)`；
- `_pretrain`、`_bs6`、`_224`、`_s2222` 等后缀拼接；
- 旧的 `os.makedirs`。

用下面代码完整替换：

```python
# 项目根目录：以 train_synapse.py 文件所在位置为准，避免从其他工作目录启动时路径漂移。
project_root = os.path.dirname(os.path.abspath(__file__))

# 把用户传入的输出根目录解析为绝对路径。
snapshot_root = args.snapshot_root
if not os.path.isabs(snapshot_root):
    snapshot_root = os.path.join(project_root, snapshot_root)
snapshot_root = os.path.abspath(snapshot_root)

# 只允许短目录名，避免再次把超参数全部写进 Windows 路径。
args.exp = args.run_name

# 当前实验的全部训练产物都写入 model_pth/synapse/<run_name>/。
snapshot_path = os.path.join(snapshot_root, args.run_name)

# 短路径仍建议每次使用新的 run_name，避免旧 best.pth 残留造成误判。
if os.path.exists(snapshot_path) and os.listdir(snapshot_path):
    raise FileExistsError(
        'The experiment directory is not empty: {}'.format(snapshot_path)
    )
os.makedirs(snapshot_path, exist_ok=True)

# 把解析后的绝对路径挂到 args，便于 trainer 日志和后续调试查看。
args.snapshot_path = snapshot_path

# 可选但建议：保存一份机器可读的配置快照。
# 注意此处需要在文件顶部增加 import json。
with open(os.path.join(snapshot_path, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

print('Snapshot path:', snapshot_path)
```

如果采用上面的 `config.json` 代码，必须在 `train_synapse.py` 顶部导入：

```python
import json
```

如果你暂时不想保存 `config.json`，可以删掉 `with open(...)` 到 `json.dump(...)` 的整个代码块，也不影响权重路径。

### 2.3 训练入口后半部分不需要改

下面这几行可以原样保留：

```python
model = EMCADNet(
    num_classes=args.num_classes,
    kernel_sizes=args.kernel_sizes,
    expansion_factor=args.expansion_factor,
    dw_parallel=not args.no_dw_parallel,
    add=not args.concatenation,
    lgag_ks=args.lgag_ks,
    activation=args.activation_mscb,
    encoder=args.encoder,
    pretrain=not args.no_pretrain,
    pretrained_dir=args.pretrained_dir,
)

model.cuda()
print('Model successfully created.')

trainer = {'Synapse': trainer_synapse}
trainer[dataset_name](args, model, snapshot_path)
```

因为 `trainer_synapse` 已经接收 `snapshot_path`，所以改完后它会自动把训练文件写到：

```text
model_pth/synapse/<run_name>/
```

---

## 三、`trainer.py` 是否需要改

### 3.1 结论：保存位置逻辑不需要改

当前 [`trainer.py`](../trainer.py) 中这些代码都使用传入的 `snapshot_path`：

```python
logging.basicConfig(filename=snapshot_path + '/log.txt', ...)
writer = SummaryWriter(snapshot_path + '/log')
torch.save(model.state_dict(), os.path.join(snapshot_path, 'last.pth'))
torch.save(model.state_dict(), os.path.join(snapshot_path, 'best.pth'))
torch.save(model.state_dict(), os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth'))
```

所以只要 `train_synapse.py` 传入短的 `snapshot_path`，这些文件会自动落在短目录中。

### 3.2 可选的跨平台写法

为了避免 Linux/Windows 混用正斜杠，建议把 `trainer.py` 的前两处字符串拼接替换为：

```python
log_file = os.path.join(snapshot_path, 'log.txt')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(message)s',
    datefmt='%H:%M:%S',
)

writer = SummaryWriter(log_dir=os.path.join(snapshot_path, 'log'))
```

这不是解决长路径的必要条件，只是让路径表达更稳健。`trainer.py` 已经导入 `os`，不需要额外导入。

---

## 四、修改 `test_synapse.py`

测试入口必须和训练入口使用同一套短路径规则。更推荐显式传入 checkpoint，而不是测试脚本再次猜目录。

### 4.1 增加测试参数

在 `test_synapse.py` 的参数区加入：

```python
parser.add_argument(
    '--snapshot_root',
    type=str,
    default='./model_pth/synapse',
    help='root directory for Synapse outputs',
)
parser.add_argument(
    '--run_name',
    type=str,
    default='run1_seed2222',
    help='training run directory name',
)
parser.add_argument(
    '--checkpoint',
    type=str,
    default='',
    help='explicit checkpoint path; defaults to <run_dir>/best.pth',
)
```

### 4.2 删除测试入口中的旧长路径重建代码

找到 `test_synapse.py` 中从：

```python
run = 1
```

开始，到最后一个：

```python
snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
```

结束的整段旧逻辑。包括 `args.exp = ...` 的超长字符串拼接都删除。

替换为：

```python
# 以 test_synapse.py 所在位置为项目根，避免从其他目录启动时相对路径错误。
project_root = os.path.dirname(os.path.abspath(__file__))

# 解析 Synapse 输出根目录。
snapshot_root = args.snapshot_root
if not os.path.isabs(snapshot_root):
    snapshot_root = os.path.join(project_root, snapshot_root)
snapshot_root = os.path.abspath(snapshot_root)

# 训练入口和测试入口必须使用同一个短实验名。
args.exp = args.run_name
snapshot_path = os.path.join(snapshot_root, args.run_name)

if not os.path.isdir(snapshot_path):
    raise FileNotFoundError(
        'Experiment directory not found: {}'.format(snapshot_path)
    )

# 优先使用显式 checkpoint；不传时默认加载该实验目录中的 best.pth。
if args.checkpoint:
    snapshot = args.checkpoint
    if not os.path.isabs(snapshot):
        snapshot = os.path.join(project_root, snapshot)
    snapshot = os.path.abspath(snapshot)
else:
    snapshot = os.path.join(snapshot_path, 'best.pth')

# 保留旧行为：best.pth 不存在时回退到最后一个 epoch 文件。
if not os.path.isfile(snapshot):
    fallback = os.path.join(snapshot_path, 'epoch_{}.pth'.format(args.max_epochs - 1))
    if os.path.isfile(fallback):
        snapshot = fallback
    else:
        raise FileNotFoundError(
            'Checkpoint not found. Checked: {} and {}'.format(snapshot, fallback)
        )

snapshot_name = os.path.basename(snapshot)
print('Checkpoint:', snapshot)
```

### 4.3 删除原来的重复 `snapshot` 代码

替换后，原来下面这几行不能继续保留，否则会把你新计算的路径覆盖掉：

```python
snapshot = os.path.join(snapshot_path, 'best.pth')
if not os.path.exists(snapshot):
    snapshot = snapshot.replace('best', 'epoch_' + str(args.max_epochs - 1))
snapshot_name = snapshot_path.split('/')[-1]
```

要确保整个文件中只保留一套 checkpoint 选择逻辑，并使用：

```python
model.load_state_dict(torch.load(snapshot))
```

### 4.4 把测试日志和预测结果也收进当前实验目录

当前测试脚本会把日志写到项目根目录的 `test_log/test_log_<args.exp>`，并且会额外套几层 `args.exp` 和 `snapshot_name`。如果希望所有相关文件都在 `model_pth/synapse` 下，将原来的测试日志和保存目录代码替换为：

```python
# 测试日志放在当前训练实验目录中。
test_log_dir = os.path.join(snapshot_path, 'test_logs')
os.makedirs(test_log_dir, exist_ok=True)

test_log_file = os.path.join(
    test_log_dir,
    '{}_test.txt'.format(os.path.splitext(snapshot_name)[0]),
)
logging.basicConfig(
    filename=test_log_file,
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(message)s',
    datefmt='%H:%M:%S',
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(str(args))
logging.info('checkpoint: {}'.format(snapshot))

# PNG/NIfTI 直接写入当前实验目录下的 predictions。
if args.is_savenii:
    test_save_path = os.path.join(snapshot_path, 'predictions')
    os.makedirs(test_save_path, exist_ok=True)
else:
    test_save_path = None
```

同时删除旧的：

```python
log_folder = 'test_log/test_log_' + args.exp
...
args.test_save_dir = os.path.join(snapshot_path, "predictions")
test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name + '2')
```

这样测试产生的文件会变成：

```text
model_pth/synapse/run1_seed2222/
├─ best.pth
├─ log.txt
├─ test_logs/
│  └─ best_test.txt
└─ predictions/
   ├─ case0001_pred.nii.gz
   ├─ case0001_img.nii.gz
   ├─ case0001_gt.nii.gz
   └─ ...
```

### 4.5 关于 `test_save_path` 和 `utils.py`

当前 `utils/utils.py` 的 `test_single_volume()` 会直接拼接：

```python
test_save_path + '/' + case + ...
```

因此这里应继续传入普通字符串路径，不要传 `Path` 对象。上面的 `os.path.join(...)` 返回的就是字符串，能够兼容当前实现。

---

## 五、修改启动脚本

### 5.1 `start_train_synapse.sh`

在训练启动脚本的超参数区域增加：

```bash
RUN_NAME="run1_seed2222"
SNAPSHOT_ROOT="${PROJECT_DIR}/model_pth/synapse"
```

在 Python 命令中增加：

```bash
  --snapshot_root "${SNAPSHOT_ROOT}" \
  --run_name "${RUN_NAME}" \
```

完整的命令尾部应类似：

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
  --snapshot_root "${SNAPSHOT_ROOT}" \
  --run_name "${RUN_NAME}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &
```

训练完成后，模型文件就在：

```text
${PROJECT_DIR}/model_pth/synapse/run1_seed2222/
```

### 5.2 `start_test_synapse.sh`

把测试脚本中的实验名和 checkpoint 设置为同一个训练目录：

```bash
RUN_NAME="run1_seed2222"
SNAPSHOT_ROOT="${PROJECT_DIR}/model_pth/synapse"
CKPT="${SNAPSHOT_ROOT}/${RUN_NAME}/best.pth"
```

然后把测试命令改为：

```bash
nohup env RUN_ID="${RUN_ID}" python test_synapse.py \
  --volume_path "${VOLUME_PATH}" \
  --dataset "${DATASET}" \
  --img_size "${IMG_SIZE}" \
  --list_dir "${LIST_PATH}" \
  --snapshot_root "${SNAPSHOT_ROOT}" \
  --run_name "${RUN_NAME}" \
  --checkpoint "${CKPT}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &
```

这样测试不会再依赖 `test_synapse.py` 自己猜测编码器、kernel、batch、seed 等目录后缀。

### 5.3 停止脚本不需要因模型路径改变而修改

`stop_train_synapse.sh` 和 `stop_test_synapse.sh` 主要通过 `RUN_ID`、PID 文件和进程环境停止后台进程，与 `snapshot_path` 无关。只要训练/测试启动脚本仍然正确生成 PID 文件，模型目录改短不会影响停止逻辑。

---

## 六、本机直接运行命令

如果不经过 shell 脚本，在项目根目录的 PowerShell 中可以这样运行：

### 训练

```powershell
$env:PYTHONNOUSERSITE = "1"
$PY = "D:\install\miniconda3\envs\SLDGroup_EMCAD_env\python.exe"

& $PY -s .\train_synapse.py `
  --run_name "run1_seed2222" `
  --snapshot_root ".\model_pth\synapse"
```

### 测试

```powershell
$env:PYTHONNOUSERSITE = "1"
$PY = "D:\install\miniconda3\envs\SLDGroup_EMCAD_env\python.exe"

& $PY -s .\test_synapse.py `
  --run_name "run1_seed2222" `
  --snapshot_root ".\model_pth\synapse" `
  --checkpoint ".\model_pth\synapse\run1_seed2222\best.pth"
```

如果你的训练使用的是 `img_size=256`、`batch_size=20`，请保证测试命令的 `--img_size` 与训练一致。它不影响目录定位，但会影响推理结果。

---

## 七、修改后的检查方法

### 7.1 先只检查路径，不启动训练

可以在 `train_synapse.py` 临时保留：

```python
print('Snapshot path:', snapshot_path)
print('Snapshot path length:', len(snapshot_path))
```

预期类似：

```text
Snapshot path: D:\files\Medical_Image_Segmentation_Projects\SLDGroup_EMCAD\model_pth\synapse\run1_seed2222
Snapshot path length: < 180
```

具体数值会随你的项目位置变化，但不应再接近原来的 339。

### 7.2 模型创建后做目录检查

训练启动成功后，在 PowerShell 执行：

```powershell
Get-ChildItem -LiteralPath ".\model_pth\synapse\run1_seed2222" -Force
```

训练开始后至少应逐步看到：

```text
config.json
log.txt
log\
last.pth
```

第一次验证性能刷新最好值后才会看到：

```text
best.pth
```

每 50 个 epoch 才会出现对应的 `epoch_*.pth`。

### 7.3 测试前检查 checkpoint

```powershell
Test-Path -LiteralPath ".\model_pth\synapse\run1_seed2222\best.pth"
Get-Item -LiteralPath ".\model_pth\synapse\run1_seed2222\best.pth" | Select-Object FullName,Length,LastWriteTime
```

第一条应输出 `True`。如果输出 `False`，先查看训练日志，不要直接启动测试。

---

## 八、几个必须注意的细节

### 1. `run_name` 不能随意重复

上面的推荐代码对非空目录直接抛出 `FileExistsError`。这是为了避免旧的 `best.pth` 被误当成新实验结果。每次新实验使用新的名称，例如：

```text
baseline_224_s2222
baseline_256_s2222
cab_224_s2222
cab_224_s3333
```

### 2. `args.exp` 仍然保留，但只保存短名

当前代码和测试日志会使用 `args.exp`。把它设成 `args.run_name`，可以保留原有接口，同时不再生成长字符串。

### 3. 完整超参数不要丢掉

缩短目录名不等于不记录配置。建议至少保留：

- `config.json`；
- `log.txt` 开头的 `str(args)`；
- 启动命令；
- Git commit 或代码变更说明；
- Python、PyTorch、CUDA 和 GPU 信息。

目录负责定位，配置文件负责解释实验。

### 4. `best.pth` 只在验证指标刷新时生成

当前 `trainer.py` 的逻辑是每轮保存 `last.pth`，用 `inference()` 的性能决定是否保存 `best.pth`。因此训练刚启动时没有 `best.pth` 是正常的，不能据此判断路径方案失败。

### 5. 这次改法不改变模型和训练算法

它只改变：

```text
实验输出路径
checkpoint 选择方式
测试日志和预测文件的归档位置
```

不会改变 EMCAD 的网络结构、损失函数、数据增强、学习率或数据划分。

---

## 九、最终实施清单

按下面顺序人工修改即可：

```text
[ ] train_synapse.py 顶部增加 import json（若保存 config.json）
[ ] train_synapse.py 增加 --snapshot_root 和 --run_name
[ ] 删除旧的长 args.exp/snapshot_path 拼接段
[ ] 写入 model_pth/synapse/<run_name>/ 的短路径代码
[ ] 保留 trainer_synapse(args, model, snapshot_path) 调用
[ ] test_synapse.py 增加 --snapshot_root、--run_name、--checkpoint
[ ] 删除测试入口中的旧长路径重建代码
[ ] 测试入口用显式 checkpoint 加载 best.pth
[ ] 测试日志改到 <run_dir>/test_logs/
[ ] 测试预测改到 <run_dir>/predictions/
[ ] start_train_synapse.sh 传 --run_name 和 --snapshot_root
[ ] start_test_synapse.sh 传同一个 --run_name 和 --checkpoint
[ ] 新实验使用新 run_name，避免覆盖旧实验
[ ] 先做目录检查和短训练，再跑完整 300 epoch
```

实施完成后的核心结果应是：

```text
所有 Synapse 训练权重、TensorBoard、训练日志、测试日志和预测结果
都能从 model_pth/synapse/<同一个实验短名>/ 找到。
```
