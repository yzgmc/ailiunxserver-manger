"""
workers.py —— 后台工作线程
==========================
把 Agent 循环放进 QThread，避免阻塞 GUI；通过信号与界面通信。
确认弹窗走 “信号发出 -> 主线程弹窗 -> 线程继续等待” 的协作模式。
"""

from __future__ import annotations

import threading

from PyQt5.QtCore import QThread, pyqtSignal

from ..agent import ServerAgent
from ..audit import AuditLogger
from ..config import ConfigManager
from ..ssh import SSHManager


class AgentWorker(QThread):
    """执行一轮 Agent 任务的工作线程。"""

    sig_text = pyqtSignal(str)       # 助手正文增量
    sig_reasoning = pyqtSignal(str)  # 推理过程增量
    sig_tool = pyqtSignal(dict)      # 工具事件 {"event":start/end/stopped,...}
    sig_confirm = pyqtSignal(dict)   # 请求用户确认（线程会等待结果）
    sig_failed = pyqtSignal(str)     # 出错
    sig_finished = pyqtSignal()      # 正常结束

    def __init__(self, config: ConfigManager, user_text: str,
                 seed_history: list[dict] | None = None, parent=None):
        super().__init__(parent)
        self.config = config
        self.user_text = user_text
        self.seed_history = list(seed_history or [])
        self.memory: list[dict] = list(self.seed_history)  # 运行后更新的会话记忆
        self.agent: ServerAgent | None = None
        # 确认机制同步原语
        self._confirm_event = threading.Event()
        self._confirm_result = False

    # ------------------------------------------------------------------ #
    # GUI 线程调用：提交确认结果，唤醒等待中的 run()
    # ------------------------------------------------------------------ #
    def submit_confirm(self, approved: bool) -> None:
        self._confirm_result = bool(approved)
        self._confirm_event.set()

    def _ask_permission(self, request: dict) -> bool:
        self._confirm_event.clear()
        self.sig_confirm.emit(request)      # 排队通知主线程弹窗
        self._confirm_event.wait()          # 等待用户决定
        return self._confirm_result

    def stop(self) -> None:
        if self.agent is not None:
            self.agent.stop()

    # ------------------------------------------------------------------ #
    def run(self) -> None:  # noqa: D102 —— QThread 入口
        audit = AuditLogger(self.config.audit_dir)
        self.agent = ServerAgent(
            self.config,
            audit=audit,
            ssh=SSHManager(),
            on_text=lambda t: self.sig_text.emit(t),
            on_reasoning=lambda t: self.sig_reasoning.emit(t),
            on_tool_event=lambda ev: self.sig_tool.emit(ev),
            permission=self._ask_permission,
        )
        self.agent.history = list(self.seed_history)
        try:
            self.agent.run(self.user_text)
        except Exception as exc:  # noqa: BLE001 —— 交给界面展示错误
            self.sig_failed.emit(str(exc))
        finally:
            self.memory = list(self.agent.history)
            self.sig_finished.emit()
            self.agent.ssh.close_all()
