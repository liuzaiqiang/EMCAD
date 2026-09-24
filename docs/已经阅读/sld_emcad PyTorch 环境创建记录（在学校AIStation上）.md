# sld_emcad PyTorch 环境创建记录

服务器当前 Conda：`/opt/conda`，目标环境：`sld_emcad`。

## 第 1 步：检查 GPU、驱动和 CUDA

```bash
nvidia-smi
```

把输出发回后，再根据驱动支持的 CUDA 版本选择 PyTorch 安装命令。不要在未确认 GPU/驱动前直接安装固定版本。

已确认：NVIDIA A30，驱动 `550.144.03`，驱动报告 CUDA `12.4`。

## 第 2 步：创建环境

```bash
conda create -n sld_emcad python=3.10 -y
```

环境创建成功。

## 第 3 步：激活环境

```bash
conda activate sld_emcad
```

若出现 `CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'`，先执行：

```bash
source /opt/conda/etc/profile.d/conda.sh
```

然后再次执行 `conda activate sld_emcad`。

## 第 4 步：安装 PyTorch

A30 驱动 `550.144.03` 支持 CUDA 12.4，使用 CUDA 12.1 的 PyTorch 官方 wheel：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

PyTorch 已成功安装：`torch-2.5.1+cu121`、`torchvision-0.20.1+cu121`。

## 第 5 步：验证 PyTorch 和 GPU

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda runtime:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## 第 6 步：安装 EMCAD 项目依赖

使用 PyPI 官方源并忽略服务器中错误配置的额外 NVIDIA 源：

```bash
python -m pip install --isolated -r requirements.txt --index-url https://pypi.org/simple
```

服务器没有 `requirements.txt` 时，可直接执行等价命令：

```bash
python -m pip install --isolated --index-url https://pypi.org/simple numpy==1.22.4 loguru tqdm pyyaml pandas matplotlib scikit-learn scikit-image scipy opencv-python seaborn albumentations==1.1.0 tabulate warmup-scheduler transformers==4.21.3 torchprofile torchmetrics einops ptflops torchsummary torchsummaryx segmentation-mask-overlay==0.3.4 timm==0.6.12 tifffile pillow thop simpleitk nibabel h5py huggingface-hub==0.11.0 ml_collections tensorboardx medpy
```

安装过程中若看到 `pypi.ngc.nvidia.com` 的 `NameResolutionError`，但同时仍在从 `pypi.org` 下载包，表示安装仍在进行，只是额外源配置导致重试和变慢。先等待命令结束；最终以 `Successfully installed` 或 `ERROR` 为准。

## 第 7 步：EMCAD 导入测试

在服务器 EMCAD 项目根目录 `/107552503563` 执行：

```bash
python -c "import torch; from lib.networks import EMCADNet; print('import ok'); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## 第 8 步：EMCAD GPU 前向测试

使用 `pretrain=False`，只测试模型结构和 GPU，不依赖服务器上的 PVT 预训练文件：

```bash
python -c "import torch; from lib.networks import EMCADNet; torch.set_grad_enabled(False); model=EMCADNet(num_classes=1, encoder='pvt_v2_b2', pretrain=False).cuda().eval(); x=torch.randn(1,1,352,352,device='cuda'); y=model(x); print('forward ok'); print('input:', tuple(x.shape)); print('outputs:', [tuple(t.shape) for t in y]); print('device:', y[-1].device)"
```

## 新服务器启动脚本路径配置

当前服务器的 `conda env list` 显示：

```text
base       /opt/conda
sld_emcad  /opt/conda/envs/sld_emcad
```

因此以下配置正确：

```bash
CONDA_BASE="/opt/conda"
CONDA_ENV_PREFIX="/opt/conda/envs/sld_emcad"
```

启动脚本中的激活方式也正确：

```bash
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"
```

可用下面的命令验证路径和 Python：

```bash
test -x /opt/conda/envs/sld_emcad/bin/python && echo "env path ok"; conda run -p /opt/conda/envs/sld_emcad python -c "import sys; print(sys.executable)"
```

## 在新服务器上确定 Conda 三个路径值

### 1. 确定 `CONDA_BASE`

```bash
conda info --base
```

输出的目录就是 `CONDA_BASE`。当前服务器输出应为：

```text
/opt/conda
```

也可以检查 Conda 可执行文件：

```bash
command -v conda
```

若输出 `/opt/conda/bin/conda`，则 `CONDA_BASE` 就是去掉 `/bin/conda` 后的：

```text
/opt/conda
```

### 2. 确定 `CONDA_ENV_PREFIX`

```bash
conda env list
```

找到环境名 `sld_emcad` 所在行，最右侧的目录就是 `CONDA_ENV_PREFIX`。当前服务器应为：

```text
/opt/conda/envs/sld_emcad
```

也可以用不依赖激活的方式确认：

```bash
conda run -n sld_emcad python -c "import sys; print(sys.prefix)"
```

### 3. 确定 `source` 路径

先设置 Conda 根目录：

```bash
CONDA_BASE="$(conda info --base)"
```

检查初始化脚本是否存在：

```bash
test -f "${CONDA_BASE}/etc/profile.d/conda.sh" && echo "conda.sh ok" || echo "conda.sh missing"
```

存在时，启动脚本应写：

```bash
source "${CONDA_BASE}/etc/profile.d/conda.sh"
```

### 4. 推荐写法

```bash
CONDA_BASE="$(conda info --base)"
CONDA_ENV_NAME="sld_emcad"
CONDA_ENV_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV_NAME}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"
```

注意：`CONDA_BASE` 是 `/opt/conda`，`CONDA_ENV_PREFIX` 才是 `/opt/conda/envs/sld_emcad`，两者不能互换。

## OpenCV 导入错误：`libGL.so.1`

若训练日志出现：

```text
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

这表示服务器缺少 OpenCV 需要的系统动态库，不是 PyTorch 或 CUDA 错误。训练型服务器可先执行：

```bash
apt-get update
```

更新完成后执行：

```bash
apt-get install -y libgl1 libglib2.0-0
```

然后验证：

```bash
python -c "import cv2; print('cv2 ok:', cv2.__version__)"
```

## 确认 `libGL.so.1` 是否缺失

当前使用 `opencv-python` 时，`libGL.so.1` 是 `cv2` 导入所需的运行时库；它不是 PyTorch 或 CUDA 的必需库。先执行：

```bash
ldconfig -p | grep -F 'libGL.so.1'
```

无输出通常表示动态链接器缓存中没有该库。再执行文件系统检查：

```bash
find /usr /lib /lib64 -name 'libGL.so.1*' 2>/dev/null
```

如果仍无输出，基本可以确认系统缺少该库。

也可以直接检查 OpenCV 的依赖：

```bash
CV2_SO="$(find "${CONDA_PREFIX}/lib/python3.10/site-packages/cv2" -type f -name '*.so' | head -n 1)"; echo "${CV2_SO}"; ldd "${CV2_SO}" | grep 'not found'
```

如果输出：

```text
libGL.so.1 => not found
```

就能直接证明 `cv2` 的动态依赖缺失。安装 `libgl1` 后重新运行同一条 `ldd` 命令，不应再出现 `not found`。
