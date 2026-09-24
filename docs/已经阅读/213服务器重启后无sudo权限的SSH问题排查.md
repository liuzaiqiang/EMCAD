# 213服务器重启后无sudo权限的SSH问题排查

> 之前的关键证据是：210到213的ping为0%丢包，TCP/22可以建立，但213返回 Exceeded MaxStartups。
> 这更像是213上的sshd认证前连接达到上限，而不是210到213网络不通。

## 一、重启后能确认和不能确认的事

重启通常会清掉当时占用MaxStartups的未认证连接和sshd子进程，所以现场连接列表已经无法恢复。

现在要查的是：

- 故障发生时间和持续时间；
- 主要来源是210、其他内部任务还是陌生IP；
- 是否存在脚本重试、备份并发或暴力扫描；
- 当前服务是否恢复且没有复发。

不要为了复现而再次重启，也不要同时开多个Xshell自动重连窗口。

## 二、你现在可以在213执行的检查

### 1. 保存当前基线

~~~bash
date -Is
hostname -f 2>/dev/null || hostname
whoami
id
uptime
echo "SSH_CONNECTION=$SSH_CONNECTION"
echo "SSH_CLIENT=$SSH_CLIENT"
echo "SSH_TTY=$SSH_TTY"
~~~

建议保存到个人目录：

~~~bash
OUT="$HOME/ssh-incident-$(date +%F-%H%M%S)"
mkdir -p "$OUT"
~~~

### 2. 查看当前连接概况

~~~bash
ss -s
ss -nt state established '( sport = :22 or dport = :22 )'
ss -nt state syn-recv '( sport = :22 or dport = :22 )'
ss -ntp '( sport = :22 or dport = :22 )' 2>&1
~~~

普通用户可能看不到PID，这不代表命令失败。重点看是否有大量来自210或陌生网段的22端口连接。

### 3. 查看登录历史

~~~bash
who
w
last -Fai | head -n 40
~~~

`last`只记录已经完成登录的会话，不能替代sshd认证日志；认证前被拒绝的连接通常不会出现在这里。

### 4. 检查自己名下的任务

~~~bash
crontab -l 2>&1
ps -u "$USER" -f
pgrep -a -u "$USER" 'ssh|scp|sftp|rsync' 2>/dev/null
~~~

如果210上使用的是你的账号，也在210执行同样检查。重点排查循环重试、rsync/scp并发、备份、训练和发布脚本。

### 5. 尝试读取日志和生效配置

~~~bash
journalctl --since "2026-08-31 00:00:00" --no-pager 2>&1 | head -n 80
test -r /var/log/auth.log && tail -n 100 /var/log/auth.log
test -r /var/log/secure && tail -n 100 /var/log/secure
sshd -T 2>&1 | grep -Ei 'maxstartups|logingracetime|maxsessions|maxauthtries'
~~~

出现Permission denied是正常的权限限制，不要尝试绕过。记录错误文本，交给管理员处理。

## 三、在210上做一次低频验证

~~~bash
ssh -vvv -o ConnectTimeout=5 -o ConnectionAttempts=1 <用户名>@10.109.119.213
nc -vz -w 5 10.109.119.213 22
~~~

如果现在能走到密码/密钥认证，说明重启后服务可接受连接；这不等于根因已经解决。

## 四、需要管理员提供的最小证据

你没有sudo时，无法可靠读取认证日志、查看带PID的sshd连接或确认最终配置。可以把下面内容发给管理员：

~~~bash
sshd -T | grep -Ei '^(maxstartups|logingracetime|maxsessions|maxauthtries)'
ss -ntp '( sport = :22 or dport = :22 )'
ps -eo pid,ppid,lstart,stat,cmd | grep '[s]shd'
journalctl -u ssh --since '故障开始时间' --until '故障结束时间' --no-pager
journalctl -u sshd --since '故障开始时间' --until '故障结束时间' --no-pager
grep -Ei 'MaxStartups|preauth|Failed password|Invalid user|kex_exchange|Connection closed' /var/log/auth.log /var/log/secure 2>/dev/null
uptime
free -h
df -h
last reboot | head
~~~

请管理员至少反馈：

- MaxStartups和LoginGraceTime生效值；
- 故障时间段是否出现MaxStartups；
- 主要来源IP和连接数量；
- 是否有Failed password或Invalid user；
- 是否有210侧脚本、备份、监控或发布任务并发连接；
- 重启前CPU、内存、磁盘和文件描述符状态。

## 五、根据证据判断根因

| 证据 | 判断 |
| --- | --- |
| 大量来自210，时间与脚本一致 | 210侧重试或并发失控 |
| 大量陌生IP，伴随Failed password | 扫描或暴力破解 |
| 来源正常但认证/DNS很慢 | LoginGraceTime过长或认证后端异常 |
| 没有异常来源但上限很小 | 正常并发超过MaxStartups配置 |
| CPU、内存、磁盘或句柄接近上限 | 系统资源拥塞 |
| 只剩Permission denied | 前置连接问题缓解，转为账号/密钥问题 |

## 六、没有sudo时不要做的事

- 不要反复ssh、telnet或Xshell自动重连。
- 不要再次重启服务器来清现场。
- 不要读取其他用户的密钥、日志或受限进程信息。
- 不要修改/etc/ssh/sshd_config。
- 不要执行pkill sshd或批量杀进程。
- 不要把当前能登录当成根因已解决。

## 七、可直接发给管理员

~~~text
213已重启，目前SSH/XShell可以正常连接。重启前从210 ping 213为0%丢包，
telnet 213:22可以建立TCP连接，但随后收到“Exceeded MaxStartups”，
初步判断为213的sshd认证前连接达到MaxStartups上限。
我没有sudo权限，烦请按故障时间窗口提供MaxStartups/LoginGraceTime、
sshd连接与进程、auth.log或journalctl中的MaxStartups/preauth来源，
并确认是否有210侧脚本、备份、监控或发布任务产生高频SSH重试。
~~~

最终结论：你现在能完成的是保存213重启后的基线、检查自己负责的任务，并向管理员索取故障时间段证据。真正确认MaxStartups触发原因，必须依赖213的SSH认证日志、连接来源和生效配置。
