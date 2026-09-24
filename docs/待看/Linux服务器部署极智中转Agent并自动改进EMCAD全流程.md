# Linux 服务器部署极智中转 Agent 并自动改进 EMCAD 全流程

本文针对一台具有公网 IP、RTX 3090 24 GB 显存的 Linux 服务器，以及本仓库 `SLDGroup_EMCAD`。目标是让服务器上的编码 Agent 使用极智中转平台的 API，在受控范围内完成：代码阅读、论文资料检索、模块缝合、单元测试、训练/验证、实验记录和失败回滚。

## 先说结论

极智 API Key 本身不会获得服务器文件权限。必须在服务器上运行一个 Agent 客户端，由客户端读取和修改本地代码、执行命令，再把必要的代码和结果发送给模型。

本方案建议分两阶段：

1. 先用 Aider 做稳定的终端代码修改和审阅；
2. 再增加论文检索脚本、实验门禁和自动化循环。不要第一天就让 Agent 以 root 身份无限制执行命令。

RTX 3090 主要用于 EMCAD 训练和测试；如果模型通过极智中转调用，Agent 推理通常消耗的是远端服务资源，不会占用本地显存。

## 一、服务器基础准备

以下命令以 Ubuntu 22.04/24.04 为例。若服务器不是 Ubuntu，请将 `apt` 替换为对应发行版的包管理器。

```bash
sudo apt update
sudo apt install -y git curl jq ripgrep tmux htop build-essential \
  python3 python3-venv python3-pip python3-dev
```

创建专用普通用户（已有普通用户可以跳过）：

```bash
sudo adduser emcad
sudo usermod -aG sudo emcad
su - emcad
```

检查 GPU、驱动和 CUDA：

```bash
nvidia-smi
python3 --version
git --version
```

`nvidia-smi` 能正常显示 RTX 3090 后再安装 PyTorch。不要仅凭 CUDA Toolkit 版本猜测 PyTorch 版本，应按 PyTorch 官方安装命令选择匹配的 CUDA wheel。

## 二、部署仓库和 Python 环境

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone <你的仓库地址> SLDGroup_EMCAD
cd SLDGroup_EMCAD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

先安装与驱动匹配的 PyTorch。例如官方提供 CUDA 12.1 wheel 时，使用官方给出的命令；示意形式如下：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

然后安装本仓库依赖：

```bash
pip install -r requirements.txt
```

仓库的 `requirements.txt` 中包含较旧的 `numpy==1.22.4`、`timm==0.6.12`、`transformers==4.21.3` 等固定版本。若新版本 Python 安装失败，应优先使用 Python 3.10 虚拟环境，不要让 Agent 随意升级全部依赖。

验证 GPU：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')
PY
```

## 三、配置极智中转平台

不要把真实 Key 写入 Git、聊天记录或 `AGENTS.md`。创建只对当前用户可读的环境文件：

```bash
mkdir -p ~/.config/emcad-agent
chmod 700 ~/.config/emcad-agent
nano ~/.config/emcad-agent/env
chmod 600 ~/.config/emcad-agent/env
```

内容使用平台实际给出的地址和模型名：

```bash
export OPENAI_API_KEY='替换为极智APIKey'
export OPENAI_BASE_URL='https://极智平台提供的地址/v1'
export EMCAD_MODEL='平台显示的模型名'
```

加载：

```bash
source ~/.config/emcad-agent/env
```

测试接口（不会把 Key 打印出来）：

```bash
curl -sS "$OPENAI_BASE_URL/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY" | jq '.data[]?.id' | head
```

若平台要求的鉴权头、Base URL 或模型字段不同，以平台文档为准。先确认 `/models` 或一次最小聊天请求成功，再安装 Agent。

## 四、安装终端编码 Agent

推荐先使用 Aider，原因是它可以在现有 Git 仓库中理解多个文件、生成补丁、显示 diff，并且对 OpenAI 兼容接口支持较成熟：

```bash
source ~/workspace/SLDGroup_EMCAD/.venv/bin/activate
pip install aider-chat
aider --version
```

加载中转配置后启动：

```bash
cd ~/workspace/SLDGroup_EMCAD
source ~/.config/emcad-agent/env
aider --model "openai/$EMCAD_MODEL" \
  --openai-api-base "$OPENAI_BASE_URL" \
  networks.py lib/networks.py trainer.py train_polyp.py test_polyp.py
```

Aider 的具体参数可能随版本变化；如果 `--openai-api-base` 不被识别，使用环境变量 `OPENAI_API_BASE="$OPENAI_BASE_URL"` 后再启动。先不要使用全自动确认参数。确认修改合理后再让它写文件。

## 五、让 Agent 理解本项目的规则

在仓库根目录建立 `AGENTS.md`，把实验边界写成机器可读规则：

```markdown
# EMCAD Agent Rules

- 只能修改当前 Git 工作树中的代码；不要修改数据集、预训练权重和系统目录。
- 禁止删除文件、覆盖原始数据、执行 git reset --hard、修改远程仓库或上传密钥。
- 每次修改前先创建分支：git switch -c agent/<experiment-id>。
- 先阅读 networks.py、lib/networks.py、trainer.py 和对应 train/test 入口，再提出补丁。
- 二分类分割保持输出通道、标签编码、loss 和 sigmoid/threshold 逻辑一致。
- 数据划分必须固定，不能用测试集选模型或调参。
- 每次实验必须保存 config.json、训练日志、验证指标、测试指标、Git commit 和失败原因。
- 先运行 Python 语法检查和一个最小 batch，再启动完整训练。
- 没有用户确认时，不得启动长时间训练或安装新的系统级软件包。
```

然后要求 Agent 先做只读分析：

```text
阅读 AGENTS.md、README、networks.py、lib/networks.py、trainer.py、train_polyp.py 和 test_polyp.py。
先画出模型构建、数据加载、loss、checkpoint、验证和测试的调用链；不要修改文件。
指出模块缝合最可能影响的张量尺寸、输出头、监督分支和显存位置。
```

## 六、论文检索能力的可靠实现

不要假设模型天然拥有实时论文搜索能力。给 Agent 配一个明确的检索工具，至少记录标题、作者、年份、DOI/arXiv ID、链接和与你代码的对应关系。

可先使用 OpenAlex 或 Semantic Scholar 的公开 API。示例：

```bash
curl -sS 'https://api.openalex.org/works?search=medical%20image%20segmentation%20attention&per-page=5' \
  | jq '.results[] | {title,year,doi,id}'
```

要求 Agent：

- 不把摘要中的方法描述当成已经验证的代码事实；
- 给出论文中的模块输入/输出、损失、数据集和消融设置；
- 明确哪些是论文原文，哪些是针对 EMCAD 的推断；
- 保存检索结果和访问日期，避免重复搜索和引用错误论文；
- 只有在代码接口、张量尺寸和训练目标都对齐后，才提出缝合补丁。

## 七、模块缝合的标准闭环

每一个实验使用唯一 ID，例如 `20260910_cbam_decoder_v01`，并执行以下流程：

```text
建立分支
→ 读取基线代码和配置
→ 论文检索与模块规格表
→ 只改一个主要变量
→ 语法检查/导入检查
→ 单 batch 前向与反向
→ 短跑 1～3 个 epoch
→ 验证集选择 checkpoint
→ 完整训练
→ 独立测试集评估
→ 保存 diff、配置、日志、指标、显存和失败原因
```

在仓库中可先执行低成本检查：

```bash
python -m compileall networks.py lib utils trainer.py train_polyp.py test_polyp.py
python train_polyp.py --help
```

完整训练应由 Agent 生成明确命令，而不是凭记忆猜参数。例如：

```bash
python train_polyp.py \
  --data_root /data/polyp/target \
  --dataset_name ClinicDB \
  --output_dir ./model_pth/Polyp \
  --run_name 20260910_cbam_decoder_v01
```

长任务放入 `tmux`：

```bash
tmux new -s emcad-train
source .venv/bin/activate
source ~/.config/emcad-agent/env
python train_polyp.py ... 2>&1 | tee logs/20260910_cbam_decoder_v01.log
```

按 `Ctrl-b`、再按 `d` 可退出会话而不停止训练；重新连接：

```bash
tmux attach -t emcad-train
```

## 八、如何逐步提高自动化程度

第一阶段只允许 Agent 分析和生成 diff；你审阅后执行命令。

第二阶段允许它自动执行以下非破坏性命令：`python -m compileall`、单元测试、单 batch smoke test、指标汇总脚本。

第三阶段才允许它启动短跑训练。完整训练、删除文件、改变数据划分、改变评估协议和安装依赖仍应要求确认。

不要一开始使用“自动接受所有修改并无限循环”的模式。模型可能在训练失败后反复改动，造成数据泄漏、指标不可比或显存耗尽。

## 九、建议增加的实验门禁

可以让 Agent 在每次提交前检查：

```bash
git diff --check
python -m compileall .
git status --short
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv
```

每次实验至少保存：

- Git commit ID 和完整 diff；
- `config.json`、随机种子、数据划分清单；
- 训练/验证曲线；
- 最佳验证 checkpoint；
- 测试集 Dice、IoU、HD95 等指标；
- 每病例结果和异常病例；
- GPU 显存、运行时间、失败日志；
- 论文来源和模块版本。

## 十、公网服务器安全要求

- SSH 只允许密钥登录，关闭密码登录和 root 远程登录；
- 防火墙只开放 SSH 端口，Agent 不要直接暴露到公网；
- 如果必须使用 Web Agent，放在反向代理后并启用强认证、HTTPS 和 IP 白名单；
- Agent 使用普通用户和项目目录权限，不授予 `/`、`/etc`、`~/.ssh` 的写权限；
- 数据集和医学图像可能包含敏感信息，确认中转平台的数据保留和训练政策；
- 将 API Key 放在权限为 600 的文件或密钥管理器中，并定期轮换；
- 训练输出和日志中不要记录完整 Key、患者信息或数据库密码。

## 十一、3090 24 GB 的实际建议

- 优先使用 AMP/mixed precision；仓库训练代码已经导入 `GradScaler` 和 `autocast`，但仍需实测；
- 从较小 batch size 开始，遇到 CUDA out of memory 再降低 batch 或输入尺寸；
- 模块缝合先做单 batch，不要一上来跑数小时训练；
- 记录峰值显存和训练速度，不能只看 Dice；
- Agent 通过中转调用远端模型时，本地 3090 不会自动承担大模型推理；
- 若想本地部署代码模型，需要额外安装推理框架和量化模型，24 GB 显存通常只能运行量化后的中小模型，而且会与训练争抢显存，不建议作为第一步。

## 十二、最终推荐的日常操作模板

```bash
cd ~/workspace/SLDGroup_EMCAD
source .venv/bin/activate
source ~/.config/emcad-agent/env
git switch main
git pull --ff-only
git switch -c agent/20260910_experiment_name
tmux new -s emcad-agent
aider --model "openai/$EMCAD_MODEL" \
  --openai-api-base "$OPENAI_BASE_URL" \
  AGENTS.md networks.py lib/networks.py trainer.py train_polyp.py test_polyp.py
```

给 Agent 的第一条任务建议：

```text
先不要改代码。请基于当前仓库完成：
1. 画出 EMCADNet 的构建、前向、损失、checkpoint 和评估调用链；
2. 检查目标模块与现有张量尺寸、输出通道和二分类监督是否兼容；
3. 检索并列出 3 篇相关论文及 DOI/arXiv 链接；
4. 给出一个只改一个主要变量的实验方案；
5. 列出将修改的文件、风险、验证命令和回滚方式；
6. 未经确认不要写文件、安装依赖或启动完整训练。
```

完成分析后，再让它实施一个小补丁，并要求它先运行 `compileall`、单 batch 和短跑训练。只有基线和改进方案使用完全相同的数据划分、随机种子和 checkpoint 选择规则时，指标比较才有意义。

## 常见失败判断

| 现象 | 优先检查 |
|---|---|
| 401/403 | Key、鉴权头、Base URL、模型权限 |
| 404 | Base URL 是否重复了 `/v1`，模型路径是否正确 |
| 能聊天但 Agent 不能改代码 | 客户端没有本地工具权限，或没有在仓库目录启动 |
| Agent 频繁改错 | 没有 `AGENTS.md`、没有先做只读分析、上下文文件过多 |
| CUDA out of memory | batch、输入尺寸、AMP、并发训练进程 |
| 指标突然大幅提升 | 数据泄漏、测试集调参、划分变化、标签处理变化 |
| 论文引用不可靠 | 未保存 DOI/arXiv ID，模型把推断写成原文结论 |

这套流程的核心不是让 Agent 获得“无限自动权力”，而是让它拥有足够的代码、论文和实验工具，同时把数据、评估协议、系统权限和长任务执行置于可审计的门禁之下。
