"""
ai_client.py —— OpenAI 兼容大模型客户端
=======================================
用原生 `requests` 实现流式对话，兼容 DeepSeek / OpenAI / LM Studio 等
所有“OpenAI API 规范”的服务商，无需额外 SDK。

支持两种工具调用风格：
  auto —— 走标准的 tools 参数（原生 function calling）
  text —— 不走 tools，而是在系统提示中约定 JSON 文本格式，由本地解析
           （适用于不支持 function calling 的旧模型）
"""

from __future__ import annotations

import json
import re
from typing import Callable, Iterator

import requests

EVENT_CHAT = object()  # 用于类型占位，无实际作用


class AIError(Exception):
    """AI 服务端 / 网络错误。"""


class AIClient:
    def __init__(self, base_url: str, api_key: str = "", model: str = "",
                 temperature: float = 0.2, max_tokens: int = 4096,
                 tool_style: str = "auto", timeout: int = 600):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or "deepseek-chat"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_style = tool_style if tool_style in ("auto", "text") else "auto"
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _endpoint(self) -> str:
        url = self.base_url
        if not url:
            raise AIError("未配置 API 地址 (base_url)。请在「AI 设置」中填写。")
        return f"{url}/chat/completions"

    # ------------------------------------------------------------------ #
    def system_text_tool_hint(self) -> str:
        """当使用 text 工具风格时附加的系统提示。"""
        return (
            "\n\n【工具调用约定 - 重要】当你需要执行工具时，不要插入正文，而是输出如下格式：\n"
            '```\n<tool_call>{"name": "工具名", "arguments": {...}}</tool_call>\n```\n'
            "工具会以 <tool_result> 形式返回。多条请换行连续输出多个 <tool_call> 块。\n"
        )

    def parse_text_tool_calls(self, content: str) -> list[dict]:
        """从文本中解析 <tool_call>...</tool_call> 或 ```json 代码块形式的调用。"""
        calls: list[dict] = []
        for m in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.S):
            try:
                obj = json.loads(m.group(1).strip())
                calls.append({"id": "call-text", "type": "function",
                              "function": {"name": obj.get("name", ""),
                                           "arguments": json.dumps(obj.get("arguments", {}), ensure_ascii=False)}})
            except json.JSONDecodeError:
                continue
        if not calls:  # 兼容 fenced json 输出
            for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S):
                try:
                    obj = json.loads(m.group(1))
                    calls.append({"id": "call-text", "type": "function",
                                  "function": {"name": obj.get("name", ""),
                                               "arguments": json.dumps(obj.get("arguments", {}), ensure_ascii=False)}})
                except json.JSONDecodeError:
                    continue
        return calls

    # ------------------------------------------------------------------ #
    def stream_chat(self, messages: list[dict],
                    tools: list[dict] | None = None,
                    on_text: Callable[[str], None] | None = None,
                    on_reasoning: Callable[[str], None] | None = None,
                    ) -> dict:
        """
        发起流式对话。

        :param on_text:     收到正文字符增量时回调
        :param on_reasoning:收到思考内容增量时回调（reasoner 类模型）
        :return: 组装好的完整 assistant 消息
            {"role": "assistant", "content": str, "tool_calls": [...] or None,
             "reasoning": str}
        """
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        use_tools = bool(tools) and self.tool_style == "auto"
        if use_tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            resp = self._session.post(
                self._endpoint(), headers=self._headers(),
                json=payload, stream=True, timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise AIError(f"请求失败：{exc}") from exc

        if resp.status_code >= 400:
            body = resp.text[:500]
            raise AIError(f"HTTP {resp.status_code}: {body}")

        return self._consume_stream(resp, on_text, on_reasoning, use_tools)

    # ------------------------------------------------------------------ #
    def _consume_stream(self, resp: requests.Response,
                        on_text, on_reasoning, use_tools: bool) -> dict:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict] = {}   # index -> 累积中的 tool call
        finish_reason: str | None = None

        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                if not line.startswith("{"):
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                finish_reason = choice.get("finish_reason") or finish_reason

                # 1) 工具调用增量（原生 function calling）
                if use_tools and delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        entry = tool_calls.setdefault(idx, {
                            "id": "", "type": "function",
                            "function": {"name": "", "arguments": ""}})
                        entry["id"] += tc.get("id") or ""
                        fn = tc.get("function") or {}
                        entry["function"]["name"] += fn.get("name") or ""
                        entry["function"]["arguments"] += fn.get("arguments") or ""

                # 2) 思考内容（deepseek-reasoner 等）
                r = delta.get("reasoning_content")
                if r:
                    reasoning_parts.append(r)
                    if on_reasoning:
                        on_reasoning(r)

                # 3) 正文
                c = delta.get("content")
                if c:
                    content_parts.append(c)
                    if on_text:
                        on_text(c)
        finally:
            resp.close()

        content = "".join(content_parts)
        ordered = [tool_calls[i] for i in sorted(tool_calls)]

        # text 风格：从正文中解析工具调用（并把正文里的标签剥离）
        if not use_tools:
            parsed = self.parse_text_tool_calls(content)
            if parsed:
                content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.S).strip()
                content = re.sub(r"```(?:json)?\s*\{.*?\}\s*```", "", content, flags=re.S).strip()
                ordered = parsed

        return {
            "role": "assistant",
            "content": content or None,
            "tool_calls": ordered or None,
            "reasoning": "".join(reasoning_parts),
            "finish_reason": finish_reason,
        }

    # ------------------------------------------------------------------ #
    def simple_chat(self, messages: list[dict]) -> str:
        """非工具、非流式的快速对话（用于连接测试）。"""
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "max_tokens": 64}
        try:
            resp = self._session.post(self._endpoint(), headers=self._headers(),
                                      json=payload, timeout=(15, 60))
        except requests.RequestException as exc:
            raise AIError(f"请求失败：{exc}") from exc
        if resp.status_code >= 400:
            raise AIError(f"HTTP {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or "(空响应)"
