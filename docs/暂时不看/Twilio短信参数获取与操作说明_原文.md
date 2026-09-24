这三个参数都来自 Twilio 控制台。

## 1. 创建并登录 Twilio

打开：

[https://console.twilio.com/](https://console.twilio.com/)

注册账户并完成：

1. 邮箱验证；
2. 手机号验证；
3. 账户登录；
4. 绑定付款方式或充值。

试用账户通常有以下限制：

- 只能向已经验证的目标手机号发送；
- 短信正文可能自动带有试用账户提示；
- 可用国家和地区受限制；
- 账户余额不足时无法发送。

## 2. 获取 `TWILIO_ACCOUNT_SID`

登录控制台后进入首页 Dashboard，在账户信息区域可以看到类似：

```text
Account SID
ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

它通常以 `AC` 开头。

复制后填写：

```python
TWILIO_ACCOUNT_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Account SID 主要是账户标识，不是短信发送号码，也不是密码。

## 3. 获取 `TWILIO_AUTH_TOKEN`

在 Twilio Dashboard 的账户信息区域找到：

```text
Auth Token
```

点击显示或查看按钮，可能需要重新输入账户密码。

然后填写：

```python
TWILIO_AUTH_TOKEN = "你的真实AuthToken"
```

注意：

- Auth Token 等同于账户 API 密钥；
- 不要发给别人；
- 不要提交到 GitHub；
- 不要放入公开截图；
- 不要把它写进论文或聊天记录；
- 如果怀疑泄露，应立即在 Twilio 控制台重新生成或轮换。

更安全的方式是使用 Windows 环境变量：

```powershell
$env:TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:TWILIO_AUTH_TOKEN="你的AuthToken"
```

程序中仍然可以保留：

```python
import os

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
```

## 4. 获取 `TWILIO_FROM_NUMBER`

在控制台进入：

```text
Phone Numbers
```

然后选择：

```text
Buy a number
```

选择一个支持 SMS 的号码，重点查看它是否具有：

```text
SMS capable
```

购买后，在号码详情中可以看到类似：

```text
+15017122661
```

然后填写：

```python
TWILIO_FROM_NUMBER = "+15017122661"
```

必须使用国际 E.164 格式：

```text
+国家代码手机号
```

例如中国大陆手机号格式一般是：

```text
+8613812345678
```

但 `TWILIO_FROM_NUMBER` 不能随便填写普通手机号，必须是 Twilio 账户中已经购买、验证或配置好的发送号码。

## 5. 三者的关系

```text
TWILIO_ACCOUNT_SID
    表示哪个 Twilio 账户

TWILIO_AUTH_TOKEN
    证明你有权操作这个账户

TWILIO_FROM_NUMBER
    表示短信从哪个 Twilio 号码发出
```

发送时，程序会向 Twilio API 提交：

```text
账户身份：Account SID + Auth Token
发送方：From
接收方：To
短信内容：Body
```

## 6. 最小配置示例

```python
TWILIO_ACCOUNT_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

TWILIO_AUTH_TOKEN = "你的AuthToken"

TWILIO_FROM_NUMBER = "+15017122661"

TO_PHONE_NUMBER = "+8613812345678"

MESSAGE_BODY = "这是一条经过授权的测试短信。"

DRY_RUN = False
```

实际发送前，建议先设置：

```python
DRY_RUN = True
```

确认号码和内容打印正确后，再改成：

```python
DRY_RUN = False
```

## 7. 如何确认发送结果

程序成功调用后，Twilio 通常会返回：

```json
{
  "sid": "SMxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "status": "queued"
}
```

常见状态包括：

- `queued`：已进入发送队列；
- `accepted`：请求已接受；
- `sending`：正在发送；
- `sent`：已发送；
- `delivered`：已送达；
- `failed`：发送失败；
- `undelivered`：未送达。

在 Twilio 控制台的消息日志中，也可以查看发送记录、错误代码、费用和状态变化。

特别注意：如果你要向中国大陆手机号发送，必须先在 Twilio 控制台确认该目的地当前支持短信、账户已满足当地合规要求，并查看具体国家/地区的发送限制。不要假设“有 Twilio 号码就一定能发到任何国家”。
