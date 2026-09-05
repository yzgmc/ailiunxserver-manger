"""
main_window.py —— 主窗口
========================
左侧：服务器列表管理；右侧：AI 对话（含流式输出、工具卡片、执行确认）。
通过 AgentWorker(QThread) 在后台执行 Agent 循环，保持界面不卡顿。
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFontDatabase, QFontMetrics, QTextBlockFormat, QTextCharFormat, QTextCursor, QTextOption
from PyQt5.QtWidgets import (
    QAbstractItemView, QDialog, QHBoxLayout, QInputDialog, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QSizePolicy, QSplitter, QToolButton, QVBoxLayout,
    QWidget,
)

from ..audit import AuditLogger
from ..config import ConfigManager
from ..memory import GlobalMemory
from .dialogs import AiConfigDialog, AuditDialog, MemoryDialog, ServerDialog, ask_confirm
from .workers import AgentWorker

_MONO = QFontDatabase.systemFont(QFontDatabase.FixedFont).family()
_COLORS = {
    "user": "#0b5394",
    "ai": "#0e7c5c",
    "system": "#767676",
    "error": "#cf222e",
    "tool": "#9a6700",
    "reason": "#9aa0a6",
    "ok": "#1a7f37",
}
_NAME = {"user": "你", "ai": "AI 助手", "system": "系统", "error": "错误",
         "tool": "工具", "reason": "思考"}


def _short(text: str, n: int = 26) -> str:
    """把长文本压成单行短摘要，供折叠行标题使用。"""
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


# --------------------------------------------------------------------------- #
# 聊天气泡控件
# --------------------------------------------------------------------------- #
class _Pane(QPlainTextEdit):
    """只读、整块随内容伸缩的文本区（不使用内部滚动条，避免出现拖动条）。"""

    def __init__(self, color="#1f2328", italic=False, family=None,
                 size=15, cap=None, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFrameStyle(QPlainTextEdit.NoFrame)
        self.setStyleSheet("background:transparent; border:none;")
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._fmt = QTextCharFormat()
        self._fmt.setForeground(QColor(color))
        self._fmt.setFontItalic(italic)
        if family:
            self._fmt.setFontFamily(family)
        if size:
            self._fmt.setFontPointSize(size)
        self.document().contentsChanged.connect(self._fit_height)
        self._fit_height()

    def _fit_height(self) -> None:
        """
        内容/宽度变化时整块长高，不设上限、不使用内部滚动条。
        注意：QPlainTextDocumentLayout 的 document().size().height() 单位是
        “行数”而非像素，必须乘以行距换算，否则所有消息都会被压扁。
        """
        fm = self.fontMetrics()
        fmt_size = self._fmt.fontPointSize() if hasattr(self, "_fmt") else 0
        if fmt_size > 0:                    # 正文用了独立字号，按该字号算行距
            f = self._fmt.font()
            f.setPointSizeF(fmt_size)
            fm = QFontMetrics(f)
        lines = self.document().size().height()          # 行数（含自动折行）
        h = int(lines * fm.lineSpacing()) \
            + 2 * int(self.document().documentMargin()) + 8
        h = max(h, 20)
        if getattr(self, "_last_h", None) != h:
            self._last_h = h
            self.setFixedHeight(h)

    def resizeEvent(self, event) -> None:  # noqa: N802
        # 宽度变化后按新宽度重新排版高度，避免出现空白
        super().resizeEvent(event)
        self._fit_height()

    def append_text(self, text: str) -> None:
        if not text:
            return
        cur = self.textCursor()
        cur.movePosition(QTextCursor.End)
        cur.insertText(text, self._fmt)

    def clear_all(self) -> None:
        self.clear()


class _Bubble(QWidget):
    """一条聊天气泡：头部（角色名）+ 内容区。"""

    def __init__(self, chat: "ChatLog", name: str, color: str,
                 pane_kw: dict | None = None, bg: str | None = None):
        super().__init__(chat.viewport())
        self._chat = chat
        pane_kw = pane_kw or {}
        if bg:
            self.setAttribute(Qt.WA_StyledBackground, True)  # 让 QSS 背景生效
            self.setStyleSheet(f"background:{bg}; border-radius:10px;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(4)

        self.header = QLabel(
            f"{name} · {time.strftime('%H:%M:%S')}", self)
        self.header.setStyleSheet(f"color:{color}; font-weight:bold; font-size:13px;")
        lay.addWidget(self.header)

        self.pane = _Pane(parent=self, **pane_kw)
        lay.addWidget(self.pane)
        chat._append_widget(self)


class _AssistantBubble(_Bubble):
    """
    AI 消息气泡：默认只显示正文；若模型输出思考过程，
    会先出现在“思考过程”折叠区内（默认收起），点击标题可展开。
    """

    def __init__(self, chat: "ChatLog"):
        super().__init__(chat, _NAME["ai"], _COLORS["ai"],
                         pane_kw={"color": "#1f2328", "size": 15, "cap": 640})
        self._reason_ready = False      # 是否已有思考内容
        self._reason_visible = False
        self._reason_area: _Pane | None = None
        self._toggle: QToolButton | None = None

    # ---- 思考折叠区 ----
    def _ensure_reason(self) -> _Pane:
        if self._reason_area is None:
            self._reason_area = _Pane(color="#8a8f98", italic=True,
                                      size=10.5, parent=self)
            self._reason_area.setVisible(False)          # 默认折叠
            self.layout().insertWidget(1, self._reason_area)
        return self._reason_area

    def _ensure_toggle(self) -> QToolButton:
        if self._toggle is None:
            self._toggle = QToolButton(self)
            self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            self._toggle.setText("思考过程（点击展开）")
            self._toggle.setStyleSheet(
                "QToolButton{color:#8a8f98; font-size:12px; border:none;"
                " background:transparent; padding:2px 6px;}")
            self._toggle.clicked.connect(self._toggle_reason)
            self.layout().insertWidget(1, self._toggle)
        return self._toggle

    def add_reasoning(self, text: str) -> None:
        self._ensure_reason().append_text(text)
        if not self._reason_ready:
            self._reason_ready = True
            self._ensure_toggle()          # 有思考内容才出现折叠开关

    def _toggle_reason(self) -> None:
        self._reason_visible = not self._reason_visible
        if self._reason_area:
            self._reason_area.setVisible(self._reason_visible)
        if self._toggle:
            self._toggle.setText("思考过程（点击收起）" if self._reason_visible
                                 else "思考过程（点击展开）")
        QTimer.singleShot(0, self._chat._scroll_bottom)

    def add_answer(self, text: str) -> None:
        self.pane.append_text(text)


class _ToolCard(QWidget):
    """
    仿命令行 Agent（Claude Code）风格的工具调用卡片：
        ● SSH 执行   uptime && free -h
        └ 输出（12 行，点击展开）
    状态点颜色：琥珀=执行中，绿=完成，红=中止/失败。
    """

    def __init__(self, chat: "ChatLog", title: str, arg: str):
        super().__init__(chat.viewport())
        self._chat = chat
        self._expanded = False
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background:#f6f8fa; border-radius:8px;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 5, 10, 5)
        lay.setSpacing(2)

        # 第一行：状态点 + 工具名 + 参数摘要
        head = QHBoxLayout()
        head.setSpacing(7)
        self.dot = QLabel("●", self)
        self.dot.setStyleSheet("color:#9a6700; font-size:12px;")
        head.addWidget(self.dot)
        self.title = QLabel(title, self)
        self.title.setStyleSheet("font-weight:bold; font-size:13px; color:#24292f;")
        head.addWidget(self.title)
        self.arg = QLabel(arg, self)
        self.arg.setStyleSheet(f"font-family:'{_MONO}'; font-size:12px; color:#57606a;")
        self.arg.setWordWrap(True)
        self.arg.setTextInteractionFlags(Qt.TextSelectableByMouse)
        head.addWidget(self.arg, 1)
        lay.addLayout(head)

        # 第二行：执行中提示 / 可展开的输出
        self.btn = QToolButton(self)
        self.btn.setStyleSheet(
            "QToolButton{color:#8a8f98; font-size:12px; border:none;"
            " background:transparent; text-align:left; padding:1px 0;}")
        self.btn.clicked.connect(self._toggle)
        self.btn.setText("└ 执行中…")
        self.btn.setVisible(True)
        lay.addWidget(self.btn)

        self.body = _Pane(color="#24292f", size=11, family=_MONO, parent=self)
        self.body.setVisible(False)
        lay.addWidget(self.body)
        chat._append_widget(self)

    def set_result(self, body: str, error: bool = False) -> None:
        """填入执行结果，默认收起，点击“└ 输出”可展开。"""
        self.dot.setStyleSheet("color:#cf222e; font-size:12px;" if error
                               else "color:#1a7f37; font-size:12px;")
        lines = len(str(body).splitlines()) or 1
        self._expanded = False
        self.body.setVisible(False)
        self.btn.setText(f"└ 输出（{lines} 行，点击展开）")
        self.btn.setVisible(True)
        text = str(body)
        if len(text) > 20000:
            text = text[:20000] + f"\n……（内容过长，仅显示前 20000 字符，共 {len(body)} 字符）"
        self.body.append_text(text)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self.body.setVisible(self._expanded)
        self.btn.setText(self.btn.text().replace("点击展开", "点击收起")
                         if self._expanded else
                         self.btn.text().replace("点击收起", "点击展开"))
        QTimer.singleShot(0, self._chat._scroll_bottom)


class ChatLog(QScrollArea):
    """
    聊天区：消息以“气泡”为单位竖向排列，并自动滚到底部。
    所有可见内容同步记录到 self.log（可序列化），供会话切换/持久化回放。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameStyle(QScrollArea.NoFrame)
        self.setStyleSheet("QScrollArea { background:#ffffff; border:none; }")
        self._container = QWidget(self)
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(12, 10, 12, 10)
        self._layout.setSpacing(10)
        self._layout.addStretch(1)
        self.setWidget(self._container)
        self.log: list[dict] = []
        self._active: _AssistantBubble | None = None
        self._active_log_idx: int | None = None
        self._pending_tool: dict | None = None
        self._pending_card: _ToolCard | None = None
        self._sticky_bottom = True
        self.verticalScrollBar().valueChanged.connect(self._on_scroll_value)
        self.verticalScrollBar().rangeChanged.connect(self._on_range_changed)

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _append_widget(self, w: QWidget) -> None:
        self._layout.insertWidget(self._layout.count() - 1, w)
        self._scroll_bottom()

    def _scroll_bottom(self) -> None:
        """滚到底部并进入“吸底”模式（内容长高时自动跟随）。"""
        self._sticky_bottom = True
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_scroll_value(self, value: int) -> None:
        # 用户手动往上翻就停止吸底，翻回底部附近则恢复吸底
        self._sticky_bottom = value >= self.verticalScrollBar().maximum() - 60

    def _on_range_changed(self, _mn: int, mx: int) -> None:
        # 消息撑高会使滚动范围异步变大，吸底时跟随到新的底部
        if getattr(self, "_sticky_bottom", True):
            self.verticalScrollBar().setValue(mx)

    def _clear_widgets(self) -> None:
        self._active = None
        self._active_log_idx = None
        self._pending_tool = None
        self._pending_card = None
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    # ------------------------------------------------------------------ #
    # 序列化：会话切换时保存/回放
    # ------------------------------------------------------------------ #
    def snapshot(self) -> list[dict]:
        """返回当前对话日志（供外部持久化/切换使用）。"""
        return list(self.log)

    def set_log(self, entries: list[dict]) -> None:
        """清空并用既有日志重建界面。"""
        self._clear_widgets()
        self.log = list(entries or [])
        for e in self.log:
            self._render_entry(e)

    def _render_entry(self, e: dict) -> None:
        kind = e.get("kind", "user")
        if kind == "assistant":
            bubble = _AssistantBubble(self)
            if e.get("reasoning"):
                bubble.add_reasoning(e["reasoning"])
            if e.get("text"):
                bubble.pane.append_text(e["text"])
        elif kind == "tool":
            self._render_tool(e)
        elif kind == "code":
            self._render_code(e.get("title", ""), e.get("text", ""),
                              e.get("accent", "tool"))
        else:
            self._render_message(kind, e.get("text", ""))

    # ------------------------------------------------------------------ #
    # 纯渲染（不入日志，供 set_log 回放使用）
    # ------------------------------------------------------------------ #
    def _render_message(self, role: str, text: str) -> None:
        if not text:
            return
        cfg = {
            "user": (_NAME["user"], _COLORS["user"], {"color": "#1f2328", "size": 15},
                     "#e8f1fb"),
            "system": (_NAME["system"], _COLORS["system"],
                       {"color": "#5a5a5a", "italic": True, "size": 13}, None),
            "error": (_NAME["error"], _COLORS["error"],
                      {"color": "#cf222e", "size": 14}, "#fdecec"),
        }.get(role)
        if cfg:
            name, color, pane_kw, bg = cfg
            bubble = _Bubble(self, name, color, pane_kw, bg)
            bubble.pane.append_text(text)

    def _render_code(self, title: str, body: str, accent: str) -> None:
        color = _COLORS.get(accent, "#9a6700")
        bubble = _Bubble(self, title or "输出", color,
                         pane_kw={"family": _MONO, "color": "#24292f",
                                  "size": 11, "cap": 600},
                         bg="#f6f8fa")
        bubble.header.setStyleSheet(
            f"color:{color}; font-weight:bold; font-size:12px;")
        bubble.pane.append_text(body)

    def _render_tool(self, e: dict) -> None:
        """按工具事件日志渲染卡片（兼容旧格式 {"label","body"}）。"""
        name = e.get("name")
        arg = e.get("arg", "")
        body = e.get("body", "")
        if name is None:                     # 旧会话日志：从 label 里恢复标题
            label = e.get("label", "") or "工具调用"
            name = label.split("：", 1)[0] if "：" in label else label
            arg = label.split("：", 1)[1] if "：" in label else ""
        title, _ = self._tool_title_arg(name, {})
        card = _ToolCard(self, title, arg or "")
        card.set_result(body)

    @staticmethod
    def _tool_title_arg(name: str, args: dict) -> tuple[str, str]:
        """工具名 -> 展示标题；参数 -> 单行摘要。"""
        if name == "ssh_exec":
            return "SSH 执行", _short(str(args.get("command", "")), 100)
        if name == "web_search":
            return "网络搜索", _short(str(args.get("query", "")), 60)
        if name == "memory_remember":
            return "全局记忆", _short(str(args.get("text", "")), 60)
        return f"工具 {name}", _short(json.dumps(args, ensure_ascii=False), 60)

    def tool_event(self, ev: dict) -> None:
        """
        接收 Agent 工具事件，渲染为 Claude Code 风格卡片：
        start 时先显示“● 工具名 参数（执行中）”，end 时就地填入结果。
        """
        event = ev.get("event")
        if event == "start":
            name = ev.get("name", "")
            args = ev.get("args") or {}
            self._pending_tool = {"name": name, "args": args}
            title, arg = self._tool_title_arg(name, args)
            self._pending_card = _ToolCard(self, title, arg)
        elif event == "end" and self._pending_tool:
            info = self._pending_tool
            self._pending_tool = None
            name = info["name"]
            args = info["args"]
            result = str(ev.get("result", "") or "")
            if name == "ssh_exec":
                command = str(args.get("command", ""))
                body = f"执行的命令：\n{command}\n\n执行结果：\n{result}"
            elif name == "web_search":
                query = str(args.get("query", ""))
                body = f"搜索关键词：{query}\n\n搜索结果：\n{result}"
            elif name == "memory_remember":
                body = f"保存内容：\n{str(args.get('text', ''))}"
            else:
                body = result
            card = self._pending_card
            self._pending_card = None
            if card is not None:
                card.set_result(body)
            else:                            # 兜底：没有 start 事件时补一张卡
                title, arg = self._tool_title_arg(name, args)
                c = _ToolCard(self, title, arg)
                c.set_result(body)
            self.log.append({"kind": "tool", "name": name,
                             "arg": self._tool_title_arg(name, args)[1],
                             "body": body})
            self._scroll_bottom()
        elif event == "stopped":
            card = self._pending_card
            self._pending_tool = None
            self._pending_card = None
            if card is not None:
                card.set_result("（已按用户要求中止）", error=True)

    # ------------------------------------------------------------------ #
    # 公共 API（渲染 + 记日志）
    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        self._clear_widgets()
        self.log = []
        self._scroll_bottom()

    def add_message(self, role: str, text: str, margin: int = 4) -> None:
        if text:
            self.log.append({"kind": role, "text": text})
            self._render_message(role, text)

    def add_code(self, title: str, body: str, accent: str = "tool",
                 max_len: int = 3000) -> None:
        body = (body or "").strip()
        if len(body) > max_len:
            body = body[:max_len] + f"\n……（内容过长，已截断，共 {len(body)} 字符）"
        self.log.append({"kind": "code", "title": title or "输出",
                         "accent": accent, "text": body})
        self._render_code(title, body, accent)

    def start_ai_stream(self) -> None:
        if self._active is not None:
            return
        entry = {"kind": "assistant", "text": "", "reasoning": ""}
        self.log.append(entry)
        self._active_log_idx = len(self.log) - 1
        self._active = _AssistantBubble(self)

    def stream_ai(self, text: str) -> None:
        if not text:
            return
        if self._active is None:
            self.start_ai_stream()
        self._active.add_answer(text)
        if self._active_log_idx is not None and 0 <= self._active_log_idx < len(self.log):
            self.log[self._active_log_idx]["text"] += text
        self._scroll_bottom()

    def stream_reasoning(self, text: str) -> None:
        if not text:
            return
        if self._active is None:
            self.start_ai_stream()
        self._active.add_reasoning(text)
        if self._active_log_idx is not None and 0 <= self._active_log_idx < len(self.log):
            self.log[self._active_log_idx]["reasoning"] += text
        self._scroll_bottom()

    # ------------------------------------------------------------------ #
    def history_for_api(self) -> list[dict]:
        """从可见日志推导多轮对话用的精简历史（仅 user/assistant 正文）。"""
        history: list[dict] = []
        for e in self.log:
            if e.get("kind") == "user":
                history.append({"role": "user", "content": e.get("text", "")})
            elif e.get("kind") == "assistant":
                text = e.get("text", "")
                if text and (not history or history[-1]["role"] != "assistant"):
                    history.append({"role": "assistant", "content": text})
        return history


# --------------------------------------------------------------------------- #
# 输入框（回车发送，Ctrl+回车换行）
# --------------------------------------------------------------------------- #
class InputEdit(QPlainTextEdit):
    submitted = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("font-size:15px; padding:6px;")
        self.setPlaceholderText("描述你想让 AI 对服务器做的事，例如：检查 nginx 为什么启动失败并修复……（Enter 发送，Ctrl+Enter 换行）")
        self.setMinimumHeight(52)
        self.textChanged.connect(self._grow)
        self._grow()

    def _grow(self) -> None:
        """随内容自动增高（1~10 行），长描述也不用憋在小框里。"""
        h = int(self.document().size().height()) + 16
        self.setFixedHeight(max(52, min(h, 220)))

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and \
                not (event.modifiers() & Qt.ControlModifier):
            self.submitted.emit()
            event.accept()
            return
        super().keyPressEvent(event)


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    # 全局字号放大（会级联到本窗口及其子对话框）
    _QSS = """
    QWidget { font-size: 13px; }
    QPushButton { font-size: 13px; min-height: 24px; }
    QListWidget { font-size: 14px; }
    QListWidget::item { padding: 4px 2px; }
    QLabel { font-size: 13px; }
    QGroupBox, QTabWidget, QTabBar::tab { font-size: 13px; }
    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { font-size: 13px; }
    """

    def __init__(self, cfg: ConfigManager):
        super().__init__()
        self.cfg = cfg
        self.audit = AuditLogger(cfg.audit_dir)
        self.global_mem = GlobalMemory(cfg.base_dir)
        self.sessions_file = Path(cfg.base_dir) / "sessions.json"
        self._sessions: list[dict] = []     # [{id,title,log,created,updated}]
        self._current_id: str | None = None
        self._worker: AgentWorker | None = None
        self._ai_stream_open = False

        self.setStyleSheet(self._QSS)
        self.setWindowTitle("Linux 服务器 AI 管理代理")
        self.resize(1360, 860)
        self._build_ui()
        self._refresh_servers()
        self._init_conversations()
        self._sync_auto_exec_button()

    # ------------------------------------------------------------------ #
    # UI 构建
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)

        # 左栏：会话 + 服务器
        left = QWidget(root)
        left.setFixedWidth(300)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)

        # ---- 会话列表（豆包式切换） ----
        lv.addWidget(self._mk_label("会话（可切换/保存）", "system"))
        self.conv_list = QListWidget(left)
        self.conv_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.conv_list.currentRowChanged.connect(self._on_conv_selected)
        self.conv_list.setMaximumHeight(230)
        lv.addWidget(self.conv_list)
        conv_btns = QHBoxLayout()
        self.btn_conv_new = QPushButton("＋ 新会话", left)
        self.btn_conv_rename = QPushButton("✎ 重命名", left)
        self.btn_conv_del = QPushButton("🗑 删除", left)
        for b in (self.btn_conv_new, self.btn_conv_rename, self.btn_conv_del):
            b.setMinimumHeight(30)
            conv_btns.addWidget(b)
        self.btn_conv_new.clicked.connect(self._new_conversation)
        self.btn_conv_rename.clicked.connect(self._rename_conversation)
        self.btn_conv_del.clicked.connect(self._delete_conversation)
        lv.addLayout(conv_btns)

        # ---- 服务器 ----
        lv.addWidget(self._mk_label("服务器（点击切换激活）", "system"))
        self.server_list = QListWidget(left)
        self.server_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.server_list.currentRowChanged.connect(self._on_server_selected)
        lv.addWidget(self.server_list, 1)

        self.btn_add_srv = QPushButton("＋ 添加服务器", left)
        self.btn_edit_srv = QPushButton("✎ 编辑选中", left)
        self.btn_del_srv = QPushButton("－ 删除选中", left)
        for b in (self.btn_add_srv, self.btn_edit_srv, self.btn_del_srv):
            b.setMinimumHeight(32)
            lv.addWidget(b)
        self.btn_add_srv.clicked.connect(lambda: self._open_server(None))
        self.btn_edit_srv.clicked.connect(self._edit_server)
        self.btn_del_srv.clicked.connect(self._delete_server)

        self.active_label = self._mk_label("", "tool")
        self.active_label.setWordWrap(True)
        lv.addWidget(self.active_label)

        # 右栏：对话
        right = QWidget(root)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 0, 0, 0)

        toolbar = QHBoxLayout()
        for text, slot, accent in [
            ("⚙ AI 设置", self._open_ai_cfg, False),
            ("🖥 服务器", self._open_server_mgr, False),
            ("🧠 全局记忆", self._open_memory, False),
            ("📋 审计日志", self._open_audit, False),
        ]:
            btn = QPushButton(text, right)
            btn.setMinimumHeight(30)
            if accent:
                btn.setStyleSheet("background:#0e7c5c;color:white;")
            btn.clicked.connect(slot)
            toolbar.addWidget(btn)
        toolbar.addStretch(1)
        rv.addLayout(toolbar)

        self.chat = ChatLog(right)
        rv.addWidget(self.chat, 1)

        bottom = QVBoxLayout()
        self.input = InputEdit(right)
        self.input.submitted.connect(self._send)
        bottom.addWidget(self.input)

        row = QHBoxLayout()
        self.status_label = self._mk_label("", "system")
        self.btn_send = QPushButton("发送", right)
        self.btn_send.setMinimumSize(96, 36)
        self.btn_send.setStyleSheet("background:#0b5394;color:white;font-weight:bold;font-size:15px;")
        self.btn_stop = QPushButton("■ 停止", right)
        self.btn_stop.setEnabled(False)
        self.btn_auto_exec = QPushButton("⚡ 自动执行", right)
        self.btn_auto_exec.setCheckable(True)
        self.btn_auto_exec.setToolTip(
            "开启后：AI 的命令（含修改/高危）不再逐个弹窗确认，直接自动执行。\n"
            "强烈建议仅在你有把握的服务器上使用。")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_send.clicked.connect(self._send)
        self.btn_auto_exec.toggled.connect(self._on_auto_exec_toggled)
        row.addWidget(self.status_label, 1)
        row.addWidget(self.btn_auto_exec)
        row.addWidget(self.btn_stop)
        row.addWidget(self.btn_send)
        bottom.addLayout(row)
        rv.addLayout(bottom)

        splitter = QSplitter(Qt.Horizontal, root)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 1060])   # 右侧聊天区启动即占满剩余宽度
        splitter.setCollapsible(1, False)  # 禁止聊天区被拖到不可见
        layout.addWidget(splitter)

        self.statusBar().showMessage("就绪")

    @staticmethod
    def _mk_label(text: str, kind: str = "system") -> QLabel:
        lbl = QLabel(text)
        color = {"system": "#767676", "tool": "#9a6700",
                 "info": "#0b5394", "ok": "#1a7f37"}.get(kind, "#333")
        lbl.setStyleSheet(f"color:{color};padding:2px;")
        return lbl

    # ------------------------------------------------------------------ #
    # 服务器侧栏
    # ------------------------------------------------------------------ #
    def _refresh_servers(self) -> None:
        self.server_list.blockSignals(True)
        self.server_list.clear()
        active = self.cfg.get_active_server_name()
        for s in self.cfg.servers():
            item = QListWidgetItem(f"{s['name']}\n{s['username']}@{s['host']}:{s['port']}")
            item.setData(Qt.UserRole, s["name"])
            if s["name"] == active:
                item.setSelected(True)
            self.server_list.addItem(item)
        self.server_list.blockSignals(False)
        self._update_active_label()

    def _update_active_label(self) -> None:
        plain = self.cfg.active_server_plain()
        if plain:
            self.active_label.setText(
                f"当前激活：\n{plain['name']}  ({plain['username']}@{plain['host']})\n"
                f"认证：{'私钥' if plain['auth_type']=='key' else '密码'}"
                + ("\nsudo：已启用" if plain["sudo_enabled"] else ""))
        else:
            self.active_label.setText("当前未激活任何服务器。")

    def _on_server_selected(self, row: int) -> None:
        """按行号激活服务器（名称存在 item 的 UserRole 数据里）。"""
        item = self.server_list.item(row)
        if item is None:
            return
        name = item.data(Qt.UserRole)
        if name and name != self.cfg.get_active_server_name():
            self.cfg.set_active_server(name)
            self._update_active_label()

    def _current_server(self) -> str | None:
        item = self.server_list.currentItem()
        if item:
            name = item.data(Qt.UserRole)
            if name:
                return name
        return None

    def _open_server(self, server: dict | None) -> None:
        dlg = ServerDialog(self.cfg, server, self)
        if dlg.exec_() == QDialog.Accepted:
            self._refresh_servers()
            self._system_msg("服务器配置已更新。")

    def _edit_server(self) -> None:
        name = self._current_server()
        if not name:
            QMessageBox.information(self, "提示", "请先在左侧选择一个服务器。")
            return
        self._open_server(self.cfg.server_by_name(name))

    def _delete_server(self) -> None:
        name = self._current_server()
        if not name:
            QMessageBox.information(self, "提示", "请先在左侧选择一个服务器。")
            return
        ret = QMessageBox.question(self, "删除服务器",
                                   f"确定删除服务器“{name}”吗？此操作不可恢复。",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        srv = self.cfg.server_by_name(name)
        if srv:
            self.cfg.delete_server(srv["id"])
        self._refresh_servers()
        self._system_msg(f"已删除服务器 {name}。")

    def _open_server_mgr(self) -> None:
        """集中管理：若无选中则引导添加。"""
        if not self.cfg.servers():
            self._open_server(None)
            return
        items = [s["name"] for s in self.cfg.servers()]
        choice, ok = QInputDialog.getItem(self, "服务器管理", "选择要编辑的服务器：",
                                          items, 0, False)
        if ok and choice:
            self._open_server(self.cfg.server_by_name(choice))

    # ------------------------------------------------------------------ #
    # 其它工具入口
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # 会话管理（豆包式多会话切换）
    # ------------------------------------------------------------------ #
    def _init_conversations(self) -> None:
        self._load_sessions()
        if not self._sessions:
            self._create_conversation(show_tip=True)
        else:
            self._current_id = self._sessions[0]["id"]
            self._reload_conv_list()
            self._open_current_session()

    def _load_sessions(self) -> None:
        try:
            if self.sessions_file.exists():
                data = json.loads(self.sessions_file.read_text(encoding="utf-8"))
                self._sessions = [s for s in data if isinstance(s, dict)
                                  and s.get("id") and s.get("title")]
        except (json.JSONDecodeError, OSError):
            self._sessions = []

    def _save_sessions(self) -> None:
        try:
            self.sessions_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.sessions_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._sessions, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(self.sessions_file)
        except OSError:
            pass

    def _current_session(self) -> dict | None:
        for s in self._sessions:
            if s["id"] == self._current_id:
                return s
        return None

    def _snapshot_current(self) -> None:
        """把当前聊天区日志同步回当前会话并标记更新时间。"""
        cur = self._current_session()
        if cur is None:
            return
        cur["log"] = self.chat.snapshot()
        cur["updated"] = datetime.now().isoformat(timespec="seconds")

    def _reload_conv_list(self) -> None:
        self.conv_list.blockSignals(True)
        self.conv_list.clear()
        for s in self._sessions:
            item = QListWidgetItem(s.get("title") or "未命名")
            item.setData(Qt.UserRole, s["id"])
            item.setToolTip(f"最近更新：{s.get('updated', '')}")
            self.conv_list.addItem(item)
        idx = next((i for i, s in enumerate(self._sessions)
                    if s["id"] == self._current_id), -1)
        if idx >= 0:
            self.conv_list.setCurrentRow(idx)
        self.conv_list.blockSignals(False)

    def _open_current_session(self) -> None:
        cur = self._current_session()
        if cur is None:
            self._current_id = self._sessions[0]["id"] if self._sessions else None
            cur = self._current_session()
        self.chat.set_log((cur or {}).get("log", []))
        self.statusBar().showMessage(f"会话：{(cur or {}).get('title', '')}")

    def _on_conv_selected(self, row: int) -> None:
        item = self.conv_list.item(row)
        if item is None:
            return
        sid = item.data(Qt.UserRole)
        if sid == self._current_id:
            return
        if self._worker is not None and self._worker.isRunning():
            # AI 运行中不允许切换，还原选中
            self._reload_conv_list()
            QMessageBox.information(self, "提示", "AI 正在处理，结束后再切换会话。")
            return
        self._snapshot_current()
        self._current_id = sid
        self._open_current_session()

    def _create_conversation(self, show_tip: bool = False) -> dict:
        now = datetime.now().isoformat(timespec="seconds")
        session = {"id": uuid.uuid4().hex[:12], "title": "新对话",
                   "log": [], "created": now, "updated": now}
        self._sessions.insert(0, session)
        if len(self._sessions) > 40:      # 保留最近 40 个会话
            self._sessions = self._sessions[:40]
        self._current_id = session["id"]
        self._reload_conv_list()
        self.chat.set_log([])
        self._save_sessions()
        if show_tip:
            self._welcome_tip()
        return session

    def _new_conversation(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "提示", "AI 正在处理，结束后再新建会话。")
            return
        self._snapshot_current()
        self._create_conversation()

    def _rename_conversation(self) -> None:
        cur = self._current_session()
        if cur is None:
            return
        new_title, ok = QInputDialog.getText(self, "重命名会话",
                                             "会话名称：", text=cur.get("title", ""))
        if ok and new_title.strip():
            cur["title"] = new_title.strip()
            self._save_sessions()
            self._reload_conv_list()

    def _delete_conversation(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "提示", "AI 正在处理，结束后再删除会话。")
            return
        cur = self._current_session()
        if cur is None:
            return
        ret = QMessageBox.question(self, "删除会话",
                                   f"确定删除会话“{cur.get('title')}”吗？对话内容将被清除。",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        idx = self._sessions.index(cur)
        self._sessions.remove(cur)
        self._current_id = None
        if self._sessions:
            self._current_id = self._sessions[min(idx, len(self._sessions) - 1)]["id"]
        else:
            self._create_conversation()          # 至少保留一个会话
        self._save_sessions()
        self._reload_conv_list()
        self._open_current_session()

    def _maybe_auto_title(self) -> None:
        """会话还没有有意义的名字时，用第一条用户消息自动命名。"""
        cur = self._current_session()
        if cur is None:
            return
        title = cur.get("title", "")
        if title and not title.startswith("新对话"):
            return
        for e in cur.get("log", []):
            if e.get("kind") == "user":
                base = " ".join(str(e.get("text", "")).split())
                if base:
                    base = base[:18] + "…" if len(base) > 18 else base
                    cur["title"] = base
                break
        self._reload_conv_list()

    # ------------------------------------------------------------------ #
    # 其它工具入口
    # ------------------------------------------------------------------ #
    def _open_ai_cfg(self) -> None:
        dlg = AiConfigDialog(self.cfg, self)
        dlg.exec_()
        self._system_msg("AI 设置已保存。" if dlg.result() == QDialog.Accepted else "")

    def _open_audit(self) -> None:
        dlg = AuditDialog(self.audit, self)
        dlg.exec_()

    def _open_memory(self) -> None:
        dlg = MemoryDialog(self.global_mem, self)
        dlg.exec_()

    # ------------------------------------------------------------------ #
    # 自动执行开关（免二次确认）
    # ------------------------------------------------------------------ #
    def _sync_auto_exec_button(self) -> None:
        s = self.cfg.get_safety_config()
        on = s["mode"] == "auto" and not s["force_danger_confirm"]
        self.btn_auto_exec.blockSignals(True)
        self.btn_auto_exec.setChecked(on)
        self.btn_auto_exec.blockSignals(False)
        self._paint_auto_exec(on)

    def _paint_auto_exec(self, on: bool) -> None:
        self.btn_auto_exec.setStyleSheet(
            "background:#1a7f37;color:white;font-weight:bold;" if on else "")

    def _on_auto_exec_toggled(self, on: bool) -> None:
        # 开启=完全自动执行（含高危，不再弹窗）；关闭=回到标准模式（修改/高危需确认）
        if on:
            self.cfg.set_safety_config("auto", False)
            self.status_label.setText("已开启：AI 自动执行，不再逐个确认")
        else:
            self.cfg.set_safety_config("standard", True)
            self.status_label.setText("已关闭：修改/高危命令需确认")
        self._paint_auto_exec(on)

    def _welcome_tip(self) -> None:
        self.chat.add_message(
            "system",
            "欢迎使用 Linux 服务器 AI 管理代理。使用步骤：\n"
            "1. 点击「⚙ AI 设置」配置模型（DeepSeek / LM Studio 等），并保存；\n"
            "2. 点击「＋ 添加服务器」录入 SSH 信息（密码或私钥），可先“测试连接”；\n"
            "3. 选择左侧服务器，即可指挥 AI 执行/排查服务器问题。\n"
            "4. 左侧“会话”列表可新建/切换多段对话，全部自动保存。")
        if not self.cfg.get_ai_config().get("base_url"):
            self.chat.add_message("system", "⚠ 尚未配置 AI 模型，请先打开「⚙ AI 设置」。")
        self._snapshot_current()
        self._save_sessions()

    # ------------------------------------------------------------------ #
    # 消息显示辅助
    # ------------------------------------------------------------------ #
    def _system_msg(self, text: str) -> None:
        if text:
            self.chat.add_message("system", text)

    def _ensure_ai_open(self) -> None:
        if not self._ai_stream_open:
            self.chat.start_ai_stream()
            self._ai_stream_open = True

    # ------------------------------------------------------------------ #
    # 发送与运行
    # ------------------------------------------------------------------ #
    def _send(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        cur = self._current_session()
        if cur is None:
            self._create_conversation()
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()

        if not self.cfg.get_active_server_name():
            self.chat.add_message("system",
                                  "提示：尚未选择服务器，AI 将无法执行 SSH。请先在左侧选择或添加服务器。")

        seed_history = self.chat.history_for_api()   # 精简的多轮上下文
        self.chat.add_message("user", text)
        self._ai_stream_open = False
        self._set_busy(True)

        worker = AgentWorker(self.cfg, text, seed_history, self)
        worker.sig_text.connect(self._on_ai_text)
        worker.sig_reasoning.connect(self._on_ai_reason)
        worker.sig_tool.connect(self._on_tool_event)
        worker.sig_confirm.connect(self._on_confirm)
        worker.sig_failed.connect(self._on_failed)
        worker.sig_finished.connect(self._on_finished)
        self._worker = worker
        self.status_label.setText("AI 处理中…")
        worker.start()

    def _set_busy(self, busy: bool) -> None:
        self.btn_send.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)
        self.input.setEnabled(not busy)
        # AI 运行时锁定会话/服务器操作，避免状态错乱
        for w in (self.conv_list, self.server_list, self.btn_conv_new,
                  self.btn_conv_rename, self.btn_conv_del,
                  self.btn_add_srv, self.btn_edit_srv, self.btn_del_srv):
            w.setEnabled(not busy)

    def _stop(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self.status_label.setText("正在停止…")

    # ------------------------------------------------------------------ #
    # AgentWorker 信号槽
    # ------------------------------------------------------------------ #
    def _on_ai_text(self, chunk: str) -> None:
        self._ensure_ai_open()
        self.chat.stream_ai(chunk)

    def _on_ai_reason(self, chunk: str) -> None:
        self._ensure_ai_open()
        self.chat.stream_reasoning(chunk)

    def _on_tool_event(self, ev: dict) -> None:
        """工具调用以“命令行 Agent 风格卡片”加入对话；状态栏同步提示。"""
        self.chat.tool_event(ev)
        event = ev.get("event")
        if event == "start":
            name = ev.get("name", "")
            hint = {"ssh_exec": "AI 正在服务器上执行操作…",
                    "web_search": "AI 正在联网搜索…"}.get(name, "AI 正在调用工具…")
            self.status_label.setText(hint)
        elif event == "retry":
            self.status_label.setText(ev.get("message", "正在自动重试…"))
        elif event == "stopped":
            self.status_label.setText("工具调用已中止")
        else:  # end
            self.status_label.setText("AI 处理中…")

    def _on_confirm(self, request: dict) -> None:
        approved, _remember = ask_confirm(self, request)
        if self._worker is not None:
            self._worker.submit_confirm(approved)

    def _on_failed(self, msg: str) -> None:
        self.chat.add_message("error", "执行出错：" + msg)
        self.status_label.setText("出错")
        self._snapshot_current()
        self._save_sessions()
        self._set_busy(False)
        self.statusBar().showMessage("就绪")

    def _on_finished(self) -> None:
        # 助手内容已随流式写入聊天区；同步日志到会话并自动命名
        self._snapshot_current()
        self._maybe_auto_title()
        self._save_sessions()
        self._set_busy(False)
        self.status_label.setText("完成")
        self.statusBar().showMessage("就绪")

    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:  # noqa: N802 —— Qt 保留命名
        if self._worker is not None and self._worker.isRunning():
            # 若正卡在命令确认等待，先置“拒绝”以唤醒线程，再请求停止
            self._worker.submit_confirm(False)
            self._worker.stop()
            self._worker.wait(3000)
        self._snapshot_current()
        self._save_sessions()
        event.accept()
