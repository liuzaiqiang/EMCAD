# Python 标准库发送短信：Twilio HTTPS API 示例

> 这是一份不依赖任何第三方框架、只使用 Python 标准库的示例。
>
> 代码通过 Twilio 的 HTTPS REST API 发送短信。你必须拥有 Twilio 账户、已验证的发送号码或 Messaging Service，并且只能向已经同意接收短信的号码发送内容。不要用于群发骚扰、绕过验证码、欺诈或其他未经授权的用途。
>
> 程序默认 `DRY_RUN = True`，只打印将要发送的请求信息而不会真正发送。确认账号、号码、内容和资费无误后，再把它改成 `False`。

## 完整程序

将下面内容保存为 `send_sms_twilio_stdlib.py`。保存后只需要修改“全局参数区”的值即可。

```python
# 导入 base64 模块，用于按照 HTTP Basic Authentication 规范编码“账号 SID:认证令牌”。
import base64

# 导入 json 模块，用于把 Twilio API 返回的 JSON 文本转换成 Python 字典，并把请求参数编码成 JSON 无关的表单数据之前进行调试输出。
import json

# 导入 os 模块，用于可选地从操作系统环境变量读取凭据，避免把真实认证令牌提交到代码仓库。
import os

# 导入 re 模块，用于检查手机号码是否基本符合 E.164 国际号码格式。
import re

# 导入 time 模块，用于实现发送前的最小时间间隔保护，防止程序循环调用时产生意外高频发送。
import time

# 从 urllib.error 导入 HTTPError 和 URLError，用于分别处理服务商返回的 HTTP 错误和网络连接错误。
from urllib.error import HTTPError, URLError

# 从 urllib.parse 导入 urlencode，用于把 POST 请求参数编码成 application/x-www-form-urlencoded 格式。
from urllib.parse import urlencode

# 从 urllib.request 导入 Request 和 urlopen，用于在不安装 requests 等第三方库的情况下发送 HTTPS 请求。
from urllib.request import Request, urlopen


# ============================= 全局参数区 =============================

# 设置 Twilio 账户 SID；建议优先通过环境变量提供，代码中的占位文本只用于提醒你需要配置该值。
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# 设置 Twilio Auth Token；真实令牌属于敏感凭据，不要写入 Git、截图、论文或发给其他人。
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "请在这里填写你的TwilioAuthToken")

# 设置 Twilio 发送号码；该号码必须已经在 Twilio 账户中购买、验证或由 Messaging Service 合法管理。
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "+15017122661")

# 设置接收短信的目标号码；必须使用 E.164 格式，例如中国大陆手机号通常写成 +8613812345678。
TO_PHONE_NUMBER = os.getenv("TO_PHONE_NUMBER", "+8613812345678")

# 设置短信正文；请确保内容合法、准确，并且接收者已经明确同意接收这类消息。
MESSAGE_BODY = os.getenv("MESSAGE_BODY", "这是一条经过授权的测试短信。")

# 设置 Twilio REST API 的版本号；v1 是当前 Messages API 的常用版本路径。
TWILIO_API_VERSION = "2010-04-01"

# 设置网络请求超时时间，单位为秒；网络异常时程序最多等待这个时间后抛出超时错误。
REQUEST_TIMEOUT_SECONDS = 20

# 设置两次发送之间的最短等待时间，单位为秒；单条测试建议保持较大间隔，不要连续轰炸号码。
MIN_INTERVAL_SECONDS = 5

# 设置是否只进行模拟运行；True 表示不联网发送，False 才会调用 Twilio 接口产生实际短信费用。
DRY_RUN = True

# 设置是否允许程序在短信内容为空时直接报错；这里固定为不允许空短信，避免产生无意义请求。
REQUIRE_NON_EMPTY_MESSAGE = True


# 定义一个函数，用于检查字符串是否是基本合规的 E.164 电话号码格式。
def validate_e164_phone_number(phone_number):
    # 使用正则表达式要求号码以加号开头，后面是 8 到 15 位数字；这只是格式检查，不代表号码真实存在。
    pattern = r"^\+[1-9]\d{7,14}$"

    # 调用 re.fullmatch 确保整个字符串都符合格式，而不是只检查其中一部分。
    return re.fullmatch(pattern, phone_number) is not None


# 定义一个函数，用于在真正发送前集中检查所有必需配置，尽早发现占位符或格式错误。
def validate_configuration():
    # 检查账户 SID 是否仍然是示例占位符；如果是，说明用户尚未填写真实账户信息。
    if TWILIO_ACCOUNT_SID.startswith("ACxxxxxxxx"):
        raise ValueError("TWILIO_ACCOUNT_SID 仍是占位符，请填写真实 Twilio Account SID。")

    # 检查认证令牌是否仍然是中文提示文本；真实令牌不应出现在公开代码中。
    if TWILIO_AUTH_TOKEN.startswith("请在这里填写"):
        raise ValueError("TWILIO_AUTH_TOKEN 仍是占位符，请填写真实 Auth Token 或设置同名环境变量。")

    # 检查发送号码是否符合国际号码格式，避免把本地号码或多余空格发送给服务商。
    if not validate_e164_phone_number(TWILIO_FROM_NUMBER):
        raise ValueError("TWILIO_FROM_NUMBER 不是有效的 E.164 格式，例如 +15017122661。")

    # 检查目标号码是否符合国际号码格式，避免因号码格式错误导致请求失败。
    if not validate_e164_phone_number(TO_PHONE_NUMBER):
        raise ValueError("TO_PHONE_NUMBER 不是有效的 E.164 格式，例如 +8613812345678。")

    # 在配置要求非空短信时，去除首尾空白后检查正文是否仍有实际内容。
    if REQUIRE_NON_EMPTY_MESSAGE and not MESSAGE_BODY.strip():
        raise ValueError("MESSAGE_BODY 不能为空。")

    # 检查短信正文长度；Twilio 可能把超长内容拆成多条短信并按条计费，因此这里给出明确限制提醒。
    if len(MESSAGE_BODY) > 1600:
        raise ValueError("MESSAGE_BODY 超过 1600 个字符，请拆分内容并明确控制短信条数。")


# 定义一个函数，把账户 SID 和认证令牌编码成 HTTP Basic Authentication 请求头。
def build_basic_auth_header(account_sid, auth_token):
    # 按 Basic Auth 规范拼接用户名和密码，中间使用英文冒号分隔。
    credential_text = f"{account_sid}:{auth_token}"

    # 将凭据转换为 UTF-8 字节后进行 Base64 编码，再转换回 ASCII 文本以便放入 HTTP 请求头。
    credential_base64 = base64.b64encode(credential_text.encode("utf-8")).decode("ascii")

    # 返回完整的 Authorization 请求头值，调用者无需接触 Base64 的内部细节。
    return f"Basic {credential_base64}"


# 定义发送短信的核心函数，返回 Twilio 创建的消息资源信息。
def send_sms():
    # 在所有网络操作开始前执行配置检查，避免使用明显错误的号码或占位凭据调用服务商。
    validate_configuration()

    # 如果处于模拟模式，只打印不包含认证令牌的必要信息，并返回模拟结果。
    if DRY_RUN:
        print("[DRY_RUN] 当前为模拟模式，不会真正发送短信。")
        print(f"[DRY_RUN] 发送号码: {TWILIO_FROM_NUMBER}")
        print(f"[DRY_RUN] 接收号码: {TO_PHONE_NUMBER}")
        print(f"[DRY_RUN] 短信内容: {MESSAGE_BODY}")
        return {"status": "dry_run"}

    # 组合 Twilio Messages API 的完整 HTTPS 地址，其中账户 SID 属于路径的一部分。
    api_url = (
        f"https://api.twilio.com/{TWILIO_API_VERSION}/Accounts/"
        f"{TWILIO_ACCOUNT_SID}/Messages.json"
    )

    # 组装 Twilio 要求的表单字段；From、To 和 Body 是发送普通短信时的核心参数。
    form_data = {
        "From": TWILIO_FROM_NUMBER,
        "To": TO_PHONE_NUMBER,
        "Body": MESSAGE_BODY,
    }

    # 使用 urlencode 将字典编码为 application/x-www-form-urlencoded 格式，并转换为 UTF-8 字节。
    encoded_form_data = urlencode(form_data).encode("utf-8")

    # 创建 POST 请求对象，并声明服务商需要的内容类型以及认证信息。
    request = Request(
        api_url,
        data=encoded_form_data,
        method="POST",
        headers={
            "Authorization": build_basic_auth_header(
                TWILIO_ACCOUNT_SID,
                TWILIO_AUTH_TOKEN,
            ),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": "stdlib-sms-example/1.0",
        },
    )

    # 调用 urlopen 发送 HTTPS 请求；timeout 防止网络不可用时程序无限等待。
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        # 读取响应字节并按照 UTF-8 解码为 JSON 文本。
        response_text = response.read().decode("utf-8")

        # 把 JSON 文本解析为字典，便于后续打印消息 SID、状态和服务商错误字段。
        response_json = json.loads(response_text)

        # 返回服务商的结构化结果，让 main 函数统一负责显示。
        return response_json


# 定义程序入口函数，负责打印结果、控制发送间隔以及处理常见异常。
def main():
    # 在发送前显示当前配置摘要，但绝不打印 Auth Token 等敏感凭据。
    print("准备执行短信发送程序。")
    print(f"发送号码: {TWILIO_FROM_NUMBER}")
    print(f"接收号码: {TO_PHONE_NUMBER}")
    print(f"模拟模式: {DRY_RUN}")

    # 在执行真实网络请求前等待最小间隔，给手动终止程序留下时间，也降低误触发风险。
    if MIN_INTERVAL_SECONDS > 0:
        print(f"将在 {MIN_INTERVAL_SECONDS} 秒后继续；如发现号码或内容错误，请立即按 Ctrl+C 终止。")
        time.sleep(MIN_INTERVAL_SECONDS)

    # 调用核心函数执行模拟发送或真实发送。
    result = send_sms()

    # 以缩进 JSON 打印结果，便于查看 Twilio 返回的 message SID、status 或错误信息。
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 如果是模拟模式，给出下一步提示；真实发送时则打印成功消息的常用字段。
    if result.get("status") == "dry_run":
        print("模拟完成；确认配置和授权后，把 DRY_RUN 改为 False 才会实际发送。")
    else:
        print(f"短信请求已提交，消息 SID: {result.get('sid', '服务商未返回 SID')}")


# 当该文件被直接运行时执行 main；被其他模块导入时不会自动发送短信。
if __name__ == "__main__":
    try:
        # 调用程序入口函数。
        main()
    except KeyboardInterrupt:
        # 用户按下 Ctrl+C 时给出简洁提示，不打印冗长堆栈。
        print("\n程序已被用户手动终止，未继续发送。")
    except HTTPError as error:
        # 读取服务商返回的错误正文，以便定位认证、余额、号码或内容问题。
        error_body = error.read().decode("utf-8", errors="replace")

        # 打印 HTTP 状态码和脱敏后的错误正文；正文中若包含敏感信息，仍应避免向他人转发。
        print(f"Twilio 返回 HTTP 错误 {error.code}: {error_body}")
    except URLError as error:
        # 处理 DNS、代理、TLS 握手或网络不可达等连接层错误。
        print(f"网络连接失败: {error.reason}")
    except TimeoutError:
        # 处理 Python 标准库抛出的超时异常。
        print("网络请求超时，请检查网络后重试。")
    except (ValueError, json.JSONDecodeError) as error:
        # 处理参数校验失败或服务商返回非预期 JSON 的情况。
        print(f"参数或响应解析失败: {error}")

