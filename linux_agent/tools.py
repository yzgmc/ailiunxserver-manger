"""
tools.py —— Agent 工具定义与注册表
==================================
工具以 dict(schema) + handler 形式注册，供 AI 客户端 / Agent 循环调用。

当前工具：
  ssh_exec    —— 在指定服务器上执行命令（经风险分级 + 用户确认）
  web_search  —— 联网搜索，辅助排查运维问题
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .risk import classify
from .search import format_results, search_web, SearchError


@dataclass
class ToolContext:
    """工具执行期上下文（由 Agent 每次运行注入）。"""
    config: Any                                        # ConfigManager
    ssh: Any                                           # SSHManager
    audit: Any                                         # AuditLogger
    # 确认回调：入参请求 dict，返回 {"allow": bool, "source": "user"/"auto"}
    permission: Callable[[dict], dict]
    memory: Any = None                                 # GlobalMemory
    search_engine: str = "duckduckgo"
    search_max: int = 6
    stop_flag: Callable[[], bool] = lambda: False      # 是否被要求停止


# --------------------------------------------------------------------------- #
# SSH 执行工具
# --------------------------------------------------------------------------- #
def _resolve_server(ctx: ToolContext, name: str) -> dict:
    """按名称/ID 解析服务器；缺省使用当前激活服务器。"""
    cfg = ctx.config
    if name:
        plain = cfg.server_plain(cfg.server_by_name(name))
        if plain is None:
            names = "、".join(s["name"] for s in cfg.servers())
            raise ValueError(f"找不到服务器 {name!r}，可用的服务器：{names or '（无）'}")
        return plain
    plain = cfg.active_server_plain()
    if plain is None:
        raise ValueError("未指定服务器，且没有默认激活的服务器。请先通过工具参数 server 指定，或要求用户选择。")
    return plain


def tool_ssh_exec(ctx: ToolContext, args: dict) -> str:
    """在服务器上执行 shell 命令并返回输出。"""
    command = str(args.get("command", "")).strip()
    server_name = str(args.get("server", "")).strip() or None
    timeout = int(args.get("timeout", 0) or 0)

    if not command:
        return "错误：command 参数为空。"

    server = _resolve_server(ctx, server_name)
    display = server["name"]

    # --- 风险分级 ---
    level, reason = classify(command)
    request = {
        "kind": "ssh",
        "server": display,
        "host_user": f"{server['username']}@{server['host']}",
        "command": command,
        "level": int(level),
        "level_label": level.label,
        "reason": reason,
    }
    decision = ctx.permission(request)  # {"allow": bool, "source": "user"/"auto"}
    approved = decision["allow"]
    if ctx.stop_flag():
        return "已停止."

    if not approved:
        ctx.audit.log_ssh(server=display, command=command,
                          decision="denied_by_user", summary=reason)
        return "该命令已被用户拒绝执行。请向用户说明你原本的计划，并给出替代方案或等待进一步指示。"

    # --- 执行 ---
    try:
        session = ctx.ssh.get(server)
        result = session.execute(command, timeout=timeout or None)
    except Exception as exc:  # noqa: BLE001
        ctx.audit.log_ssh(server=display, command=command, decision="error",
                          summary=str(exc))
        return f"执行失败：{exc}"

    audit_decision = "approved_by_user" if decision["source"] == "user" else "auto_approved"
    ctx.audit.log_ssh(server=display, command=command,
                      decision=audit_decision, exit_code=result.exit_code,
                      summary=result.summary(500))
    if result.timed_out:
        return f"命令执行超时。\n{result.detail()}"
    head = f"（{display}）退出码 {result.exit_code}\n"
    return head + result.summary(4000)


# --------------------------------------------------------------------------- #
# 网络搜索工具
# --------------------------------------------------------------------------- #
def tool_web_search(ctx: ToolContext, args: dict) -> str:
    """联网搜索并返回结构化结果文本。"""
    query = str(args.get("query", "")).strip()
    max_results = int(args.get("max_results", 0) or ctx.search_max)
    if not query:
        return "错误：query 参数为空。"
    try:
        results = search_web(query, engine=ctx.search_engine, max_results=max_results)
        ctx.audit.log("web_search", query=query[:300], count=len(results))
        return format_results(results, query)
    except SearchError as exc:
        return f"搜索失败：{exc}"


# --------------------------------------------------------------------------- #
# 长期记忆工具
# --------------------------------------------------------------------------- #
def tool_memory_remember(ctx: ToolContext, args: dict) -> str:
    """把值得跨会话长期记住的事实写入全局记忆。"""
    if ctx.memory is None:
        return "错误：全局记忆不可用。"
    text = str(args.get("text", "")).strip()
    if not text:
        return "错误：text 参数为空。"
    topic = str(args.get("topic", "")).strip()
    note = ctx.memory.add(text, topic=topic, source="ai")
    ctx.audit.log("memory_add", text=text[:300], note_id=note["id"])
    return (f"已写入全局记忆（当前共 {len(ctx.memory.notes())} 条）。"
            f"内容：{text}。后续会话我会持续参考它。")


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
@dataclass
class ToolSpec:
    name: str
    description: str
    handler: Callable[[ToolContext, dict], str]
    parameters: dict = field(default_factory=dict)

    def schema(self) -> dict:
        """转成 OpenAI 函数调用格式的 schema。"""
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="ssh_exec",
        description=("通过 SSH 在 Linux 服务器上执行一条 shell 命令，返回其输出。"
                     "管理服务器（查看状态、改配置、装软件、重启服务等）都必须使用本工具，禁止凭空编造结果。"
                     "复杂操作请拆成多步执行；写/危险命令会弹窗请求用户批准。"
                     "输出很长时应阅读并总结关键信息。"),
        handler=tool_ssh_exec,
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string",
                            "description": "要执行的 shell 命令，例如 `df -h`、`systemctl status nginx`"},
                "server": {"type": "string",
                           "description": "目标服务器名称（可选）。省略时使用当前激活的服务器。"},
                "timeout": {"type": "integer",
                            "description": "超时秒数（可选，默认 30）。"},
            },
            "required": ["command"],
        },
    ),
    ToolSpec(
        name="web_search",
        description=("联网搜索互联网获取最新资料。当你需要查 Linux 报错解决方案、软件用法、"
                     "已知问题或手册时使用；也可用于对比最佳实践。"),
        handler=tool_web_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词（建议用英文更准）"},
                "max_results": {"type": "integer", "description": "返回条数（可选，默认按设置）"},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="memory_remember",
        description=("把值得跨会话长期记住的内容写入全局记忆（例如用户偏好、约定、"
                     "环境约束、重要结论）。当用户明确说“记住……”或你判断某约定应长期保留时调用；"
                     "不要保存一次性任务的琐碎细节。"),
        handler=tool_memory_remember,
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string",
                         "description": "要记住的事实，用一句完整的中文描述。"},
                "topic": {"type": "string",
                          "description": "可选主题，例如「部署约定」「用户偏好」「排查结论」"},
            },
            "required": ["text"],
        },
    ),
]

TOOLS_BY_NAME: dict[str, ToolSpec] = {t.name: t for t in TOOL_SPECS}


def tool_schemas() -> list[dict]:
    """返回所有工具的 OpenAI 格式 schema。"""
    return [t.schema() for t in TOOL_SPECS]


def dispatch(name: str, ctx: ToolContext, args: dict) -> str:
    """按名称分发工具调用。"""
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        return f"错误：未知工具 {name!r}，可用工具：{', '.join(TOOLS_BY_NAME)}"
    try:
        return spec.handler(ctx, args or {})
    except Exception as exc:  # noqa: BLE001 —— 工具异常转可读文本回给 AI
        return f"工具 {name} 执行出错：{exc}"
