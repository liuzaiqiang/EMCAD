# `stop_train_acdc.sh` 无法停止 ACDC 训练：排查与手动修复

## 根因

`start_train_acdc.sh` 的启动关系是：

```text
nohup env RUN_ID=... python -u train_acdc.py ... &
```

它把 Python 进程 PID 写入：

```text
${RUN_ID}.pid
```

`stop_train_acdc.sh` 读取该 PID 后，用下面的命令从 `/proc/<PID>/environ` 验证 `RUN_ID`：

```bash
ENV_RUN_ID="$(tr '\0' '\n' < "/proc/${PID}/environ" | sed -n 's/^RUN_ID=//p' | head -n 1)"
```

但脚本开头启用了：

```bash
set -euo pipefail
```

`head -n 1` 找到第一项后会提前关闭管道；`tr` 或 `sed` 可能收到 `SIGPIPE`，使整个管道返回非零状态。此时脚本会因 `set -e` 直接退出，还没有执行后面的 `kill -TERM`。

## 手动修改一处

打开 `stop_train_acdc.sh`，找到原来的环境读取行：

```bash
ENV_RUN_ID="$(tr '\0' '\n' < "/proc/${PID}/environ" | sed -n 's/^RUN_ID=//p' | head -n 1)"
```

替换为：

```bash
# 在严格模式下，head 提前关闭管道可能让上游收到 SIGPIPE；允许该读取管道正常结束。
ENV_RUN_ID="$(tr '\0' '\n' < "/proc/${PID}/environ" | sed -n 's/^RUN_ID=//p' | head -n 1 || true)"
```

其余 `RUN_ID` 比较和 `kill` 逻辑先不要改。

## 正确停止方式

启动脚本输出的完整值例如：

```text
[INFO] RUN_ID=train_ACDC_2026-09-09_214215_gpu_0_SEED_2222_RAND_f2b8308c6558
```

在项目根目录执行：

```bash
bash stop_train_acdc.sh 'train_ACDC_2026-09-09_214215_gpu_0_SEED_2222_RAND_f2b8308c6558'
```

必须使用完整 `RUN_ID`，不能只传 `ACDC`、时间戳或日志文件名。

## 停止前核对

```bash
RUN_ID='这里替换为完整RUN_ID'
cat "${RUN_ID}.pid"
PID="$(tr -d ' \n' < "${RUN_ID}.pid")"
ps -p "${PID}" -o pid,ppid,stat,cmd
tr '\0' '\n' < "/proc/${PID}/environ" | grep '^RUN_ID='
```

最后一条应显示与传入值完全一致的 `RUN_ID`。如果 PID 不存在，说明训练已经结束或 PID 文件过期；如果显示的 RUN_ID 不同，不要执行强制 kill，应传入对应 PID 文件的 RUN_ID。

## 关于子进程

当前停止脚本只向训练主 Python PID 发送 `SIGTERM`，然后等待，超时再向该 PID 发送 `SIGKILL`。`num_workers=0` 时通常没有 DataLoader 子进程；若 `num_workers>0`，可能存在 worker 子进程，主进程正常退出时一般会清理它们，但脚本不保证清理整个进程组。

