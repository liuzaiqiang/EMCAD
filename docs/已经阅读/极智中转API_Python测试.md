# 极智中转 API：Python 控制台提问测试

本示例依据本机 Codex 配置中已确认的中转地址 `https://jizhiapi.site/v1`、模型 `gpt-5.6-sol` 和 `responses` 协议编写。它不读取 Codex 配置文件，也不将 API Key 写入代码。

## 1. 在 PowerShell 设置本次会话的 Key

将下面命令中的占位内容替换为你从极智平台获取的 Key。该变量仅在当前 PowerShell 窗口有效；关闭窗口后失效。

```powershell
$env:JIZHI_API_KEY = '你的极智API-Key'
```

## 2. 保存并运行脚本

将下列代码保存为 `test_jizhi_responses.py`，再在同一 PowerShell 窗口执行：

```powershell
python .\test_jizhi_responses.py
```

```python
# 导入 json 模块，用于把 Python 字典转换为接口请求所需的 JSON 文本。
import json
# 导入 os 模块，用于从环境变量读取 API Key，而不是把 Key 写进代码。
import os
# 导入 urllib.request 中的 Request 和 urlopen，用标准库发送 HTTPS 请求，无需安装第三方包。
from urllib.request import Request, urlopen
# 导入 HTTPError 和 URLError，用于输出服务端错误或网络错误的具体原因。
from urllib.error import HTTPError, URLError

# 从环境变量读取极智 API Key；没有设置时返回空字符串。
api_key = os.getenv("JIZHI_API_KEY", "")
# 设置极智 OpenAI 兼容接口的基础地址；末尾不保留斜杠以便后面拼接路径。
base_url = "https://jizhiapi.site/v1".rstrip("/")
# 使用本机 Codex 配置中当前使用的模型；若你的平台后台显示其他模型名，在此处替换。
model_name = "gpt-5.6-sol"
# 设置本次要问模型的问题；可直接改成你的任意问题。
question = "请用三句话说明医学图像分割中 Dice 系数的含义。"

# 在没有读取到 Key 时立即停止，避免发送无效请求。
if not api_key:
    # 提示用户先在当前 PowerShell 会话设置环境变量。
    raise SystemExit("未检测到 JIZHI_API_KEY。请先执行：$env:JIZHI_API_KEY = '你的极智API-Key'")

# 按 Responses API 格式构造请求数据；input 可直接传入一段文本。
payload = {
    # 指定要调用的模型。
    "model": model_name,
    # 指定用户问题。
    "input": question,
}
# 将请求字典编码为 UTF-8 JSON 字节串，ensure_ascii=False 保证中文正常传输。
body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
# 创建 HTTP POST 请求，并加入鉴权与 JSON 内容类型请求头。
request = Request(
    # 将基础地址与 Responses 路径组合成完整请求地址。
    url=f"{base_url}/responses",
    # 指定 HTTP 方法为 POST。
    method="POST",
    # 放入已编码的 JSON 请求体。
    data=body,
    # 加入 Bearer API Key、JSON 请求头和常见客户端标识；部分 WAF 会拦截默认的 Python-urllib 标识。
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
    },
)

# 定义函数，从标准 Responses API 返回结构中提取模型生成的文本。
def get_output_text(response_data):
    # 创建列表，用于收集可能被分成多段返回的文本。
    text_parts = []
    # 遍历 output 数组；缺失 output 时按空数组处理。
    for output_item in response_data.get("output", []):
        # 只处理模型消息，跳过工具调用等非文本输出。
        if output_item.get("type") != "message":
            # 当前项不是消息时进入下一项。
            continue
        # 遍历该消息内的内容块；缺失 content 时按空数组处理。
        for content_item in output_item.get("content", []):
            # 只收集输出文本内容块。
            if content_item.get("type") == "output_text":
                # 读取文本并追加；不存在 text 字段时追加空字符串。
                text_parts.append(content_item.get("text", ""))
    # 将所有文本块拼接后返回。
    return "".join(text_parts)

# 开始处理网络请求和服务端响应。
try:
    # 发送请求，timeout=120 表示最多等待 120 秒。
    with urlopen(request, timeout=120) as response:
        # 读取响应字节并按 UTF-8 解码，再转换为 Python 字典。
        response_data = json.loads(response.read().decode("utf-8"))
    # 提取最终文本回答。
    answer = get_output_text(response_data)
    # 有文本回答时直接打印到控制台。
    if answer:
        # 输出模型回答。
        print(answer)
    # 无法按标准结构提取文本时保留完整 JSON，便于判断中转平台的返回差异。
    else:
        # 以格式化 JSON 输出响应，但不要将其中内容提交到公开位置。
        print(json.dumps(response_data, ensure_ascii=False, indent=2))
# 捕获 HTTP 状态码错误，例如 401 Key 无效、403 无权限、404 路径错误或 429 限流。
except HTTPError as error:
    # 读取服务端错误内容；读取失败时以空字节串代替。
    error_body = error.read().decode("utf-8", errors="replace")
    # 输出 HTTP 状态码和服务端错误信息，且不输出 API Key。
    print(f"HTTP {error.code}: {error_body}")
# 捕获网络、DNS 或 TLS 等非 HTTP 错误。
except URLError as error:
    # 输出网络层错误原因。
    print(f"网络请求失败: {error.reason}")
```

## 3. 常见结果

- 正常：控制台直接输出模型回答。
- `HTTP 401`：Key 无效、已过期，或复制时多了空格/引号。
- `HTTP 403` 且错误码 `1010`：Key 已被读取，但平台前置 WAF/Cloudflare 按请求特征、IP、地区或访问策略拒绝了 Python 请求；这不是环境变量错误。可先尝试上面脚本中的浏览器 User-Agent。若仍是 1010，需要联系极智客服将你的出口 IP/API 访问方式加入允许范围，或使用平台明确提供的 API 域名；不要反复更换 Key。
- `HTTP 404`：中转平台未提供 `responses` 路径；先核对平台文档，或确认 Codex 当前仍使用 `wire_api = "responses"`。
- `HTTP 429`：账户余额、速率或并发限制触发。
- 返回完整 JSON 而不是正文：平台的返回结构与标准 Responses API 有差异。保留报文中的 `id`、`error` 和字段结构即可，不要发送或截图 API Key。

## 安全说明

不要把 Key 放进 `.py` 文件、Git 仓库、聊天记录或截图。脚本只从 `JIZHI_API_KEY` 环境变量读取 Key；如需长期保存，应使用系统环境变量或密钥管理工具，而不是明文代码。
