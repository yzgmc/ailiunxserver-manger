"""
agent.py —— AI 代理主循环（Agent Loop）
=======================================
职责：
  1. 依据用户话术与历史上下文调用模型；
  2. 收到工具调用后：风险分级 -> 请求用户确认 -> 执行 -> 把结果回填给模型；
  3. 循环直到模型给出最终答复或无工具调用；
  4. 全程把命令与结果写入审计日志，并支持随时中止。

本模块不依赖 Qt，可在任意线程中使用（GUI 线程、QThread 均可用）。
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Callable

from .ai_client import AIClient, AIError
from .config import ConfigManager
from .audit import AuditLogger
from .ssh import SSHManager
from .risk import RiskLevel, safety_threshold
from .memory import GlobalMemory
from .tools import ToolContext, dispatch, tool_schemas

MAX_HISTORY_TURNS = 12  # 多轮对话保留的用户/助手轮次上限

# 回调类型
TextCb = Callable[[str], None]
ToolEventCb = Callable[[dict], None]
PermissionCb = Callable[[dict], bool]


def build_system_prompt(cfg: ConfigManager, tool_style: str,
                        memory_notes: str = "") -> str:
    """动态构造系统提示：包含可用服务器清单与当前激活服务器。"""
    servers = cfg.servers()
    if servers:
        lines = ["可用服务器："]
        for s in servers:
            marker = " ★(当前激活)" if s.get("name") == cfg.get_active_server_name() else ""
            lines.append(f"- {s.get('name')}（{s.get('username')}@{s.get('host')}:{s.get('port')}）{marker}")
        server_hint = "\n".join(lines)
    else:
        server_hint = "当前没有任何已配置服务器，请先提醒用户去「服务器」里添加。"

    prompt = f"""你是「Linux 服务器管理 AI 代理」，运行在用户的桌面客户端中，用于代替用户管理其 Linux 服务器。

## 你的能力
1. 通过 ssh_exec 工具在服务器上执行 shell 命令（读取状态、修改配置、安装软件、管理服务等）。
2. 通过 web_search 工具联网搜索资料，用于排查报错、查询用法与最佳实践。
3. 通过 memory_remember 工具保存值得长期记住的用户偏好/环境约定（跨会话生效）。

## 必须遵守的规则
1. 涉及服务器的操作一律通过 ssh_exec 执行，禁止凭空编造命令输出或假装已执行。
2. 动手修改前，先用只读命令了解现状（如 uptime / free -h / df -h / systemctl status xxx / 查看配置文件），做到先诊断后操作。
3. 复杂任务拆成多步执行，每一步看结果再决定下一步；命令太长或含多段逻辑时应拆开。
4. 修改类/高危命令会弹出确认框等用户批准；发起前请用一两句中文说明“你要执行什么、为什么”，拒绝后不要强行重试同一命令。
5. 命令输出过长时总结要点；出现错误时结合 stderr 排查，必要时 web_search 查询解决方案后再修复。
6. 全程使用中文与用户交流，保持简洁专业。

{server_hint}

当前激活服务器：{cfg.get_active_server_name() or '未选择'}

## 长期记忆（跨会话，来自用户之前的指示）
{memory_notes if memory_notes else "（暂无）"}

长期记忆仅供参考，若与用户当前要求冲突，以用户当前要求为准；必要时用 memory_remember 更新记忆。
"""
    if tool_style == "text":
        prompt += "\n（注意：当前模型以文本方式调用工具，请遵守下方工具调用格式约定。）"
    return prompt


class ServerAgent:
    """
    单次“运行”即与模型完成一轮多步推理。
    连续消息记忆保存在 self.history（紧凑形式），可跨轮使用。
    """

    def __init__(self, config: ConfigManager, audit: AuditLogger | None = None,
                 ssh: SSHManager | None = None,
                 on_text: TextCb | None = None,
                 on_reasoning: TextCb | None = None,
                 on_tool_event: ToolEventCb | None = None,
                 permission: PermissionCb | None = None):
        self.config = config
        self.audit = audit or AuditLogger(config.audit_dir)
        self.ssh = ssh or SSHManager()
        self.on_text = on_text
        self.on_reasoning = on_reasoning
        self.on_tool_event = on_tool_event or (lambda ev: None)
        self.permission = permission or (lambda req: True)

        ai = config.get_ai_config()
        self.client = AIClient(base_url=ai["base_url"], api_key=ai["api_key"],
                               model=ai["model"], temperature=ai["temperature"],
                               max_tokens=ai["max_tokens"], tool_style=ai["tool_style"])
        self.max_iterations = ai["max_iterations"]
        self.context_budget = int(ai.get("context_budget", 32000) or 32000)
        self._memory = GlobalMemory(config.base_dir)
        self.history: list[dict] = []
        self._approved_commands: set[str] = set()  # 会话内已批准的命令
        self._denied_commands: set[str] = set()    # 会话内已被用户拒绝的命令
        self._stopped = False

    # ------------------------------------------------------------------ #
    # 上下文预算管理：宁可压缩旧内容，也不因超限中断任务
    # ------------------------------------------------------------------ #
    @staticmethod
    def _est_tokens(text: str) -> int:
        """粗略估算 token 数（中英文混合场景 3 字符≈1 token）。"""
        text = text or ""
        return max(1, (len(text) + 2) // 3)

    def _total_est(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            total += self._est_tokens(m.get("content", ""))
            for call in m.get("tool_calls") or []:
                fn = (call.get("function") or {})
                total += self._est_tokens(fn.get("arguments", ""))
        return total

    def _compact_messages(self, messages: list[dict]) -> None:
        """
        若累计内容超出 context_budget，则从“最早的非关键消息”开始压缩：
        先压旧的命令输出(tool)，再压旧的 assistant 长文，仍不够则截断最老的 user。
        系统提示与最新一条消息永远保留完整。
        """
        budget = self.context_budget
        last_idx = len(messages) - 1

        # 预压缩：把每条旧 tool/assistant 压到合理上限，避免单条就爆
        # 注意：阈值需把后缀长度算进去，否则压缩后仍超阈值、内容不再变短
        pre_suffix = "\n[旧内容过长，已被自动压缩]"
        for i, m in enumerate(messages):
            if i == 0 or i == last_idx:
                continue
            content = m.get("content") or ""
            cap = 1000 if m.get("role") == "tool" else 6000
            if len(content) > cap + len(pre_suffix):
                m["content"] = content[:cap] + pre_suffix

        # 动态压缩：直到总预算达标（带保险次数，杜绝任何死循环）
        # 每级阈值 = 保留长度 + 后缀长度，保证压缩一次后必然低于阈值、必有进展
        suf_tool = "\n[…(较早命令输出已截断，不再影响执行)]"
        suf_asst = "\n[…(较早分析已截断)]"
        suf_user = "\n[…(较早提问已截断)]"
        guard = 0
        while self._total_est(messages) > budget and guard < 1000:
            guard += 1
            changed = False
            # 1) 压缩最早的工具结果
            for i in range(1, last_idx):
                if messages[i].get("role") == "tool":
                    content = messages[i].get("content") or ""
                    if len(content) > 400 + len(suf_tool):
                        messages[i]["content"] = content[:400] + suf_tool
                        changed = True
                        break
            if changed:
                continue
            # 2) 压缩最早的 assistant 长文
            for i in range(1, last_idx):
                if messages[i].get("role") == "assistant":
                    content = messages[i].get("content") or ""
                    if len(content) > 800 + len(suf_asst):
                        messages[i]["content"] = content[:800] + suf_asst
                        changed = True
                        break
            if changed:
                continue
            # 3) 最后手段：压缩最早的历史 user（保留最近一轮用户输入）
            for i in range(1, last_idx):
                if messages[i].get("role") == "user":
                    content = messages[i].get("content") or ""
                    if len(content) > 300 + len(suf_user):
                        messages[i]["content"] = content[:300] + suf_user
                        changed = True
                        break
            if not changed:
                break  # 已无可压缩内容，交给服务端（理论上不会再超限）

    def _emergency_compact(self, messages: list[dict]) -> None:
        """把较早消息内容整体砍半，用于服务端仍报超限时的二次兜底。"""
        suffix = "\n[…(上下文紧张，已再次压缩)]"
        for i in range(1, len(messages) - 1):
            content = messages[i].get("content") or ""
            # 阈值保证砍半 + 后缀后一定变短，避免越压越长
            if len(content) > 600 + len(suffix) * 2:
                messages[i]["content"] = content[:max(300, len(content) // 2)] + suffix

    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        """请求中止（由其它线程调用）。"""
        self._stopped = True

    @property
    def stopped(self) -> bool:
        return self._stopped

    # ------------------------------------------------------------------ #
    def _permission_gate(self, request: dict) -> dict:
        """
        确认门：决定某条命令是自动放行、询问用户、还是按记忆放行。

        :return: {"allow": bool, "source": "user"/"auto"}
        """
        # 按安全模式决定是否需要询问
        safety = self.config.get_safety_config()
        threshold = safety_threshold(safety["mode"])
        level = int(request.get("level", 0))
        need_ask = level >= threshold
        if not need_ask and safety["force_danger_confirm"] and level == int(RiskLevel.DANGEROUS):
            need_ask = True

        key = hashlib.md5(f"{request.get('server')}|{request.get('command')}"
                          .encode("utf-8")).hexdigest()
        if key in self._approved_commands:
            return {"allow": True, "source": "auto"}  # 本会话已批准过，直接放行
        if key in self._denied_commands:
            return {"allow": False, "source": "auto"}  # 本会话已拒绝过，不再重复打扰

        if not need_ask:
            return {"allow": True, "source": "auto"}

        approved = bool(self.permission(request))
        if approved:
            self._approved_commands.add(key)
        else:
            self._denied_commands.add(key)
        return {"allow": approved, "source": "user"}

    # ------------------------------------------------------------------ #
    def _tool_context(self) -> ToolContext:
        return ToolContext(
            config=self.config, ssh=self.ssh, audit=self.audit,
            permission=self._permission_gate,
            memory=self._memory,
            search_engine=self.config.get_search_config()["engine"],
            search_max=self.config.get_search_config()["max_results"],
            stop_flag=lambda: self._stopped,
        )

    # ------------------------------------------------------------------ #
    def _call_model(self, messages: list[dict], tools: list[dict] | None,
                    style: str) -> dict:
        """
        带自动重试的模型调用，保证长任务不因临时故障中断：
          1. 上下文超限：逐级加大压缩力度后立即重试（最多 6 次），
             压缩完继续任务，绝不因超限而停止；
          2. 网络/限流/服务端瞬断：指数退避重试（最多 4 次）；
          3. 其余错误原样抛出，交由上层展示。
        """
        ctx_round = 0
        net_round = 0
        emitted = {"text": 0, "reason": 0}   # 已流式输出的字符数（重试去重依据）

        def on_text(t: str) -> None:
            emitted["text"] += len(t or "")
            if self.on_text:
                self.on_text(t)

        def on_reason(t: str) -> None:
            emitted["reason"] += len(t or "")
            if self.on_reasoning:
                self.on_reasoning(t)

        while True:
            try:
                return self.client.stream_chat(
                    messages=messages,
                    tools=tools if style == "auto" else None,
                    on_text=on_text,
                    on_reasoning=on_reason,
                )
            except AIError as exc:
                low = str(exc).lower()
                # ---- 上下文超限：压缩后继续（多轮逐级加压） ----
                if ctx_round < 6 and (
                        "context" in low or "too long" in low
                        or "reduce the length" in low or "length" in low
                        and "token" in low):
                    ctx_round += 1
                    self.on_tool_event({
                        "event": "retry",
                        "message": f"上下文接近模型上限，正在压缩较早内容（第 {ctx_round}/6 次）后继续…"})
                    saved = self.context_budget
                    # 每轮把临时压缩目标砍到上轮的 60%，确保一定压到限额以内
                    self.context_budget = max(8000, int(saved * (0.6 ** ctx_round)))
                    try:
                        self._emergency_compact(messages)
                        self._compact_messages(messages)
                    finally:
                        self.context_budget = saved
                    continue
                # ---- 瞬时故障：退避重试（仅在尚未输出内容时，避免重复文本） ----
                transient = any(k in low for k in (
                    "timeout", "timed out", "connection", "temporarily",
                    "rate limit", "429", " 500", " 502", " 503", " 504",
                    "server error", "bad gateway", "overloaded", "eof",
                    "incomplete read", "remote disconnected"))
                if transient and emitted["text"] == 0 and emitted["reason"] == 0 \
                        and net_round < 4:
                    net_round += 1
                    delay = (2, 5, 10, 20)[net_round - 1]
                    self.on_tool_event({
                        "event": "retry",
                        "message": f"网络/服务暂不可用，{delay} 秒后自动重试（第 {net_round}/4 次）…"})
                    time.sleep(delay)
                    continue
                raise

    # ------------------------------------------------------------------ #
    def run(self, user_text: str) -> str:
        """
        处理一条用户消息，返回最终答复文本。
        长任务支持：步数跑满一轮后自动“轻推”模型继续，而不是停止；
        只有达到硬上限才强制收束总结，且必定给出可见的结束说明。
        """
        self._stopped = False
        audit = self.audit
        audit.log("ai_run_start", server=self.config.get_active_server_name(),
                  user_text=user_text[:500])

        ai = self.config.get_ai_config()
        style = ai["tool_style"]
        system = build_system_prompt(self.config, style,
                                     memory_notes=self._memory.summary())
        if style == "text":
            system += self.client.system_text_tool_hint()

        # 组装本轮消息
        messages: list[dict] = [{"role": "system", "content": system}]
        for msg in self.history[-MAX_HISTORY_TURNS:]:
            messages.append(dict(msg))     # 拷贝，避免压缩时污染已保存的历史
        messages.append({"role": "user", "content": user_text})

        tool_schemas_list = tool_schemas()
        final_content = ""
        max_iter = max(int(self.max_iterations), 1)
        hard_cap = max(max_iter * 10, 60)      # 总步数硬上限（兜底防失控）
        iteration = 0
        wrapup_at: int | None = None

        try:
            while not self._stopped:
                iteration += 1

                # 长任务自动续跑：每跑满一轮步数就轻推模型从中断处继续
                if (wrapup_at is None and iteration > 1
                        and (iteration - 1) % max_iter == 0):
                    messages.append({"role": "user", "content":
                        "（系统提示）这是一个较长的任务，以上进度正常。"
                        "请从中断处继续执行未完成的步骤，不要重复已完成的操作；"
                        "全部完成后输出最终答复。"})
                    audit.log("ai_auto_continue", iteration=iteration)

                # 硬上限兜底：强制收束总结，绝不无声停止
                if wrapup_at is None and iteration > hard_cap:
                    wrapup_at = iteration
                    messages.append({"role": "user", "content":
                        "（系统提示）任务步数已达硬上限。请立即停止调用工具，"
                        "直接输出：1) 已完成的操作与结果；2) 未完成的部分与建议的下一步。"})
                elif wrapup_at is not None and iteration > wrapup_at + 3:
                    final_content = final_content or (
                        "（任务步数达到硬上限，已自动收束。"
                        "以上为已完成内容；发送“继续”可让 AI 接着执行未完成部分。）")
                    break

                self._compact_messages(messages)   # 超预算则压缩旧内容
                assistant = self._call_model(messages, tool_schemas_list, style)
                messages.append(assistant)

                calls = assistant.get("tool_calls")
                if not calls:
                    final_content = assistant.get("content") or ""
                    break  # 模型已给出最终答复

                # 逐个执行工具调用
                for call in calls:
                    if self._stopped:
                        break
                    name = (call.get("function") or {}).get("name", "")
                    try:
                        args = json.loads((call.get("function") or {}).get("arguments", "{}"))
                        if not isinstance(args, dict):
                            args = {}
                    except json.JSONDecodeError:
                        args = {"_parse_error": "参数解析失败"}
                    call_id = call.get("id") or f"call-{name}-{iteration}"

                    if name == "ssh_exec":
                        audit.log("ai_tool_call", tool=name,
                                  command=args.get("command", "")[:500],
                                  server=args.get("server", "") or "（激活）")

                    self.on_tool_event({"event": "start", "name": name, "args": args})
                    result = dispatch(name, self._tool_context(), args)
                    if self._stopped:
                        self.on_tool_event({"event": "stopped", "name": name})
                        result += "\n（已按用户要求中止）"
                    self.on_tool_event({"event": "end", "name": name, "result": result,
                                        "args": args})

                    messages.append({"role": "tool", "tool_call_id": call_id,
                                     "content": result or "(无输出)"})

            # 保留紧凑记忆（user + 最终 assistant），丢弃内部工具消息
            self.history.append({"role": "user", "content": user_text})
            if final_content:
                self.history.append({"role": "assistant", "content": final_content})
        except AIError as exc:
            audit.log("ai_error", error=str(exc)[:500])
            # 出错时撤销本次追加的 user 消息，避免下轮出现“连续 user”的不合法序列
            if self.history and self.history[-1].get("role") == "user":
                self.history.pop()
            raise
        finally:
            audit.log("ai_run_end", server=self.config.get_active_server_name())
        return final_content
