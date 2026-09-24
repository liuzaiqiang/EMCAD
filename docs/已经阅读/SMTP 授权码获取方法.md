# SMTP 授权码获取方法

SMTP 授权码是邮件客户端或 Python 程序登录邮箱 SMTP 服务器时使用的专用密码。通常不能使用邮箱网页登录密码。

## QQ 邮箱

1. 登录 QQ 邮箱网页版：`https://mail.qq.com`
2. 点击右上角齿轮，进入“设置”。
3. 打开“账户”或“账户与安全”。
4. 找到“POP3/IMAP/SMTP/Exchange/CardDAV/CalDAV 服务”一栏。
5. 开启“SMTP 服务”或“POP3/SMTP 服务”。
6. 按页面要求完成短信或二次验证。
7. 点击“生成授权码”或“管理服务”，生成新的授权码。
8. 将生成的授权码填入 Python 代码中的 `SMTP_PASSWORD`。

QQ 邮箱配置示例：

```python
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465
USE_SSL = True
SMTP_USER = "你的QQ邮箱@qq.com"
SMTP_PASSWORD = "QQ邮箱生成的授权码"
```

## 163 邮箱

1. 登录 163 邮箱网页版：`https://mail.163.com`
2. 进入“设置”。
3. 找到“POP3/SMTP/IMAP”或“客户端授权密码”设置。
4. 开启 SMTP 服务。
5. 按要求完成短信验证。
6. 生成客户端授权密码。
7. 将授权密码填入 `SMTP_PASSWORD`。

163 邮箱配置示例：

```python
SMTP_HOST = "smtp.163.com"
SMTP_PORT = 465
USE_SSL = True
SMTP_USER = "你的163邮箱@163.com"
SMTP_PASSWORD = "163邮箱客户端授权密码"
```

## Gmail

Gmail 通常不叫“SMTP 授权码”，而是使用“应用专用密码”。

1. 打开 Google 账号安全页面：`https://myaccount.google.com/security`
2. 开启“两步验证”。
3. 在安全设置中找到“应用专用密码”。
4. 新建一个应用专用密码，例如命名为 `Python SMTP`。
5. 复制生成的 16 位密码。输入 Python 时可以去掉显示出来的空格。

Gmail 配置示例：

```python
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
USE_SSL = True
SMTP_USER = "你的Gmail地址@gmail.com"
SMTP_PASSWORD = "Gmail应用专用密码"
```

## Outlook / Microsoft 365

Outlook.com 或 Microsoft 365 是否允许“用户名+密码”登录 SMTP，取决于账户和组织策略。若管理员关闭了基本认证，当前 `smtplib.login()` 代码可能无法使用，需要 OAuth2 或 Microsoft Graph API。

常见服务器配置为：

```python
SMTP_HOST = "smtp-mail.outlook.com"
SMTP_PORT = 587
USE_SSL = False
SMTP_USER = "你的Outlook邮箱"
SMTP_PASSWORD = "应用专用密码或组织允许的凭据"
```

## 安全注意事项

- 不要把 SMTP 授权码提交到 GitHub 或公开代码仓库。
- 不要把授权码发到群聊或截图中。
- 代码中的 `SMTP_PASSWORD` 泄露后，应立即在邮箱设置中撤销并重新生成。
- 授权码只用于 SMTP/客户端登录，不能替代邮箱网页登录密码。
- 如果出现 `535 Authentication failed`，优先检查 SMTP 服务是否开启、账号是否完整、授权码是否复制正确，以及是否误填了网页登录密码。
