"""
dialogs.py —— 各类对话框
========================
  AiConfigDialog   模型服务商设置 + 搜索/安全策略设置 + 连接测试
  ServerDialog     服务器新增/编辑 + 连接测试
  ask_confirm      命令执行二次确认弹窗
  AuditDialog      审计日志查看器
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPlainTextEdit, QPushButton, QRadioButton, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..ai_client import AIClient
from ..audit import AuditLogger
from ..config import PROVIDER_PRESETS, ConfigManager
from ..ssh import SSHSession


# --------------------------------------------------------------------------- #
# 通用异步任务线程
# --------------------------------------------------------------------------- #
class _JobThread(QThread):
    """在后台执行 fn()，结束后发 done(success, message)。"""
    done = pyqtSignal(bool, str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            result = self._fn()
            self.done.emit(True, str(result))
        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, str(exc))


def _busy(button: QPushButton | None, running: bool, text: str = "测试中…"):
    """测试期间禁用按钮并改文案。"""
    if button:
        button.setEnabled(not running)
        if running:
            button._orig_text = button.text()
            button.setText(text)
        else:
            button.setText(getattr(button, "_orig_text", button.text()))


# --------------------------------------------------------------------------- #
# AI 模型服务商设置
# --------------------------------------------------------------------------- #
class AiConfigDialog(QDialog):
    def __init__(self, cfg: ConfigManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.cfg = cfg
        ai = cfg.get_ai_config()
        search = cfg.get_search_config()
        safety = cfg.get_safety_config()

        self._orig_key = ai["api_key"]
        self._orig_base = ai["base_url"]
        self._orig_model = ai["model"]
        self._test_thread: _JobThread | None = None
        self._search_test_thread: _JobThread | None = None

        self.setWindowTitle("AI 设置")
        self.setMinimumWidth(560)
        root = QVBoxLayout(self)
        tabs = QTabWidget(self)

        # ------------------------- 连接设置 ------------------------- #
        conn = QWidget(self)
        form = QFormLayout(conn)

        self.provider = QComboBox(conn)
        for key, p in PROVIDER_PRESETS.items():
            self.provider.addItem(p["label"], key)
        idx = self.provider.findData(ai.get("provider", ""))
        self.provider.setCurrentIndex(idx if idx >= 0 else 0)

        self.base_url = QLineEdit(ai["base_url"], conn)
        self.base_url.setPlaceholderText("https://api.deepseek.com/v1")
        self.base_url.textEdited.connect(self._on_base_edited)

        self.model = QLineEdit(ai["model"], conn)
        self.model.setPlaceholderText("deepseek-chat")
        self.model.setMinimumWidth(240)

        self.api_key = QLineEdit(conn)
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("sk-…  （留空则沿用已保存的 Key）")
        btn_show = QCheckBox("显示", conn)
        btn_show.toggled.connect(
            lambda on: self.api_key.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        key_row = QHBoxLayout()
        key_row.addWidget(self.api_key, 1)
        key_row.addWidget(btn_show)

        self.temperature = QDoubleSpinBox(conn)
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(ai["temperature"])
        self.max_tokens = QSpinBox(conn)
        self.max_tokens.setRange(256, 131072)
        self.max_tokens.setSingleStep(1024)
        self.max_tokens.setValue(ai["max_tokens"])
        self.max_iters = QSpinBox(conn)
        self.max_iters.setRange(3, 200)
        self.max_iters.setValue(ai["max_iterations"])
        self.max_iters.setToolTip("一次任务中 AI 可执行的“推理-工具-再推理”循环步数上限；越长任务越不会中途停")
        self.context_budget = QSpinBox(conn)
        self.context_budget.setRange(8000, 1048576)
        self.context_budget.setSingleStep(8000)
        self.context_budget.setValue(ai.get("context_budget", 200000))
        self.context_budget.setToolTip(
            "发送给模型的上下文预算(Tokens)。接近上限时会自动压缩较早的命令输出/对话，"
            "压缩后继续任务，而不是报错中断，从而能支撑特别长的任务。\n"
            "按所用模型的实际上下文窗口设置：DeepSeek V4-Flash 支持 100 万，"
            "可设 400000；deepseek-chat（V3.2）为 128K，可设 110000。")
        self.tool_style = QComboBox(conn)
        self.tool_style.addItem("原生 function calling（推荐）", "auto")
        self.tool_style.addItem("文本格式调用（兼容旧模型）", "text")
        self.tool_style.setCurrentIndex(0 if ai["tool_style"] == "auto" else 1)

        form.addRow("服务商", self.provider)
        form.addRow("API 地址", self.base_url)
        form.addRow("模型名称", self.model)
        form.addRow("API Key", key_row)
        form.addRow("温度", self.temperature)
        form.addRow("最大输出 Tokens", self.max_tokens)
        form.addRow("上下文预算 Tokens", self.context_budget)
        form.addRow("单轮最多迭代步数", self.max_iters)
        form.addRow("工具调用方式", self.tool_style)

        self.hint = QLabel("DeepSeek 等 OpenAI 兼容服务均可使用；LM Studio 本地模型无需 Key。", conn)
        self.hint.setStyleSheet("color:#888;")
        form.addRow(self.hint)
        tabs.addTab(conn, "模型连接")

        # ------------------------- 策略设置 ------------------------- #
        pol = QWidget(self)
        pform = QFormLayout(pol)

        self.search_engine = QComboBox(pol)
        self.search_engine.addItem("DuckDuckGo（默认）", "duckduckgo")
        self.search_engine.addItem("Bing", "bing")
        si = self.search_engine.findData(search["engine"])
        self.search_engine.setCurrentIndex(max(si, 0))
        self.search_max = QSpinBox(pol)
        self.search_max.setRange(3, 10)
        self.search_max.setValue(search["max_results"])

        self.btn_search_test = QPushButton("测试此搜索源", pol)
        self.btn_search_test.setToolTip("用上面选中的搜索引擎实际搜索一次，检查网络是否可达")
        engine_row = QHBoxLayout()
        engine_row.addWidget(self.search_engine, 1)
        engine_row.addWidget(self.btn_search_test)

        self.safety_mode = QComboBox(pol)
        self.safety_mode.addItem("严格：每条命令都询问", "strict")
        self.safety_mode.addItem("标准：修改/高危命令询问（推荐）", "standard")
        self.safety_mode.addItem("自动：不询问直接执行", "auto")
        mi = self.safety_mode.findData(safety["mode"])
        self.safety_mode.setCurrentIndex(max(mi, 0))
        self.force_danger = QCheckBox("高危命令即使其它策略允许也强制二次确认", pol)
        self.force_danger.setChecked(safety["force_danger_confirm"])

        pform.addRow("搜索引擎", engine_row)
        pform.addRow("每次返回条数", self.search_max)
        pform.addRow("命令确认策略", self.safety_mode)
        pform.addRow("", self.force_danger)
        tabs.addTab(pol, "搜索与安全")

        root.addWidget(tabs)

        # ------------------------- 底部按钮 ------------------------- #
        btns = QDialogButtonBox(self)
        self.btn_test = btns.addButton("测试模型连接", QDialogButtonBox.ActionRole)
        self.btn_test.setToolTip("用「模型连接」页填写的服务商地址/Key 发起一次真实请求")
        btns.addButton("保存", QDialogButtonBox.AcceptRole)
        btns.addButton("取消", QDialogButtonBox.RejectRole)
        root.addWidget(btns)

        self.btn_search_test.clicked.connect(self._test_search)

        self.provider.currentIndexChanged.connect(self._on_provider_change)
        self.btn_test.clicked.connect(self._test)
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)

    # ------------------------------------------------------------------ #
    def _on_base_edited(self, _text: str) -> None:
        self._base_touched = True

    def _on_provider_change(self) -> None:
        key = self.provider.currentData()
        preset = PROVIDER_PRESETS[key]
        if not getattr(self, "_base_touched", False):
            self.base_url.setText(preset["base_url"])
        if preset["default_model"] and (not self.model.text() or
                                       self.model.text() == self._orig_model and key != "custom"):
            self.model.setText(preset["default_model"])
        self.hint.setText("DeepSeek / OpenAI 需 API Key；LM Studio 无需 Key。" if preset["needs_key"]
                          else "本地服务（如 LM Studio）通常无需 Key，可直接连接。")

    def _collect(self) -> dict:
        """收集字段并返回可写配置（Key 为空时沿用原值）。"""
        key_text = self.api_key.text().strip()
        return {
            "provider": self.provider.currentData(),
            "base_url": self.base_url.text().strip(),
            "api_key": key_text or self._orig_key,   # 留空=不修改
            "model": self.model.text().strip() or "deepseek-chat",
            "temperature": self.temperature.value(),
            "max_tokens": self.max_tokens.value(),
            "context_budget": self.context_budget.value(),
            "tool_style": self.tool_style.currentData(),
            "max_iterations": self.max_iters.value(),
            "engine": self.search_engine.currentData(),
            "search_max": self.search_max.value(),
            "safety_mode": self.safety_mode.currentData(),
            "force_danger": self.force_danger.isChecked(),
        }

    def _test(self) -> None:
        if self._test_thread and self._test_thread.isRunning():
            return
        data = self._collect()
        _busy(self.btn_test, True)

        def job():
            client = AIClient(base_url=data["base_url"], api_key=data["api_key"],
                              model=data["model"], tool_style="auto")
            return client.simple_chat([{"role": "user", "content": "请回复：连接成功"}])

        self._test_thread = _JobThread(job, self)
        self._test_thread.done.connect(self._on_test_done)
        self._test_thread.start()

    def _on_test_done(self, ok: bool, msg: str) -> None:
        _busy(self.btn_test, False)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information if ok else QMessageBox.Critical)
        box.setWindowTitle("模型连接测试")
        box.setText(("连接成功，模型返回：\n" + msg[:400]) if ok else ("连接失败：\n" + msg))
        box.exec_()

    # ------------------ 搜索源独立测试 ------------------ #
    def _test_search(self) -> None:
        if self._search_test_thread and self._search_test_thread.isRunning():
            return
        _busy(self.btn_search_test, True)
        engine = self.search_engine.currentData()

        def job():
            # 直接在后台导入，避免模块加载影响对话框
            from ..search import search_web
            results = search_web("Linux nginx 502 排查", engine=engine, max_results=3)
            return f"搜索成功，返回 {len(results)} 条结果：\n" + \
                   "\n".join(f"- {r.title[:60]}\n  {r.url}" for r in results[:3])

        self._search_test_thread = _JobThread(job, self)
        self._search_test_thread.done.connect(self._on_search_test_done)
        self._search_test_thread.start()

    def _on_search_test_done(self, ok: bool, msg: str) -> None:
        _busy(self.btn_search_test, False)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information if ok else QMessageBox.Critical)
        box.setWindowTitle("搜索源测试")
        box.setText(msg[:1200])
        box.exec_()

    def _save(self) -> None:
        data = self._collect()
        self.cfg.set_ai_config(provider=data["provider"], base_url=data["base_url"],
                               api_key=data["api_key"], model=data["model"],
                               temperature=data["temperature"], max_tokens=data["max_tokens"],
                               context_budget=data["context_budget"],
                               tool_style=data["tool_style"], max_iterations=data["max_iterations"])
        self.cfg.set_search_config(data["engine"], data["search_max"])
        self.cfg.set_safety_config(data["safety_mode"], data["force_danger"])
        self.accept()


# --------------------------------------------------------------------------- #
# 服务器新增 / 编辑
# --------------------------------------------------------------------------- #
class ServerDialog(QDialog):
    def __init__(self, cfg: ConfigManager, server: dict | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.server = server                     # None=新增；否则为原始配置
        plain = cfg.server_plain(server) if server else None
        self._orig_password = (plain or {}).get("password", "")
        self._orig_passphrase = (plain or {}).get("passphrase", "")
        self._orig_sudo = (plain or {}).get("sudo_password", "")
        self._test_thread: _JobThread | None = None

        self.setWindowTitle("编辑服务器" if server else "添加服务器")
        self.setMinimumWidth(480)
        root = QVBoxLayout(self)
        form = QFormLayout()

        self.name = QLineEdit(plain["name"] if plain else "", self)
        self.name.setPlaceholderText("例如：生产服务器-1（用于 AI 识别，唯一）")
        self.host = QLineEdit(plain["host"] if plain else "", self)
        self.host.setPlaceholderText("192.168.1.10 或 my-server.example.com")
        self.port = QSpinBox(self)
        self.port.setRange(1, 65535)
        self.port.setValue(plain["port"] if plain else 22)
        self.username = QLineEdit(plain["username"] if plain else "root", self)

        # 认证方式
        auth_box = QGroupBox("认证方式", self)
        auth = QHBoxLayout(auth_box)
        self.rb_password = QRadioButton("密码", auth_box)
        self.rb_key = QRadioButton("私钥", auth_box)
        auth.addWidget(self.rb_password)
        auth.addWidget(self.rb_key)
        auth.addStretch(1)

        self.password = QLineEdit(self)
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setPlaceholderText("SSH 密码（编辑时留空=不修改）")
        self.key_path = QLineEdit(self)
        self.key_path.setPlaceholderText("~/.ssh/id_ed25519")
        btn_browse = QPushButton("浏览", self)
        btn_browse.clicked.connect(self._browse_key)
        key_row = QHBoxLayout()
        key_row.addWidget(self.key_path, 1)
        key_row.addWidget(btn_browse)
        self.passphrase = QLineEdit(self)
        self.passphrase.setEchoMode(QLineEdit.Password)
        self.passphrase.setPlaceholderText("私钥口令（可选）")

        # sudo
        sudo_group = QGroupBox("sudo（提权，可选）", self)
        sudo_lay = QFormLayout(sudo_group)
        self.chk_sudo = QCheckBox("允许 AI 执行 sudo 命令（自动注入 sudo 密码）", sudo_group)
        self.sudo_password = QLineEdit(sudo_group)
        self.sudo_password.setEchoMode(QLineEdit.Password)
        self.sudo_password.setPlaceholderText("当前用户的 sudo 密码")
        sudo_lay.addRow(self.chk_sudo, self.sudo_password)

        form.addRow("名称", self.name)
        form.addRow("主机", self.host)
        form.addRow("端口", self.port)
        form.addRow("用户名", self.username)
        form.addRow(auth_box)
        form.addRow("密码", self.password)
        form.addRow("私钥文件", key_row)
        form.addRow("私钥口令", self.passphrase)
        form.addRow(sudo_group)

        if plain:
            self.rb_key.setChecked(plain["auth_type"] == "key")
            self.rb_password.setChecked(plain["auth_type"] != "key")
            self.key_path.setText(plain["key_path"] or "")
            self.chk_sudo.setChecked(plain["sudo_enabled"])
        else:
            self.rb_password.setChecked(True)
        self._sync_auth_enabled()
        self.rb_password.toggled.connect(lambda _: self._sync_auth_enabled())
        root.addLayout(form)

        btns = QDialogButtonBox(self)
        self.btn_test = btns.addButton("测试连接", QDialogButtonBox.ActionRole)
        btns.addButton("保存", QDialogButtonBox.AcceptRole)
        btns.addButton("取消", QDialogButtonBox.RejectRole)
        root.addWidget(btns)

        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        self.btn_test.clicked.connect(self._test)

    # ------------------------------------------------------------------ #
    def _sync_auth_enabled(self) -> None:
        use_key = self.rb_key.isChecked()
        self.password.setEnabled(not use_key)
        self.key_path.setEnabled(use_key)
        self.passphrase.setEnabled(use_key)

    def _browse_key(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择私钥文件", str(Path.home() / ".ssh"))
        if path:
            self.key_path.setText(path)

    def _collect(self) -> dict:
        """收集表单值；留空的密码/口令沿用原值。"""
        pwd = self.password.text().strip()
        pp = self.passphrase.text().strip()
        sp = self.sudo_password.text().strip()
        return {
            "name": self.name.text().strip(),
            "host": self.host.text().strip(),
            "port": self.port.value(),
            "username": self.username.text().strip(),
            "auth_type": "key" if self.rb_key.isChecked() else "password",
            "password": pwd or self._orig_password,
            "key_path": self.key_path.text().strip(),
            "passphrase": pp or self._orig_passphrase,
            "sudo_enabled": self.chk_sudo.isChecked(),
            "sudo_password": sp or (self._orig_sudo if self.chk_sudo.isChecked() else ""),
        }

    def _build_plain(self) -> dict:
        d = self._collect()
        return {"id": (self.server or {}).get("id", "test"),
                "name": d["name"], "host": d["host"], "port": d["port"],
                "username": d["username"], "auth_type": d["auth_type"],
                "password": d["password"], "key_path": d["key_path"],
                "passphrase": d["passphrase"], "sudo_enabled": d["sudo_enabled"],
                "sudo_password": d["sudo_password"], "timeout": 20}

    def _test(self) -> None:
        if self._test_thread and self._test_thread.isRunning():
            return
        if not (self.host.text().strip() and self.username.text().strip()):
            QMessageBox.warning(self, "提示", "请先填写主机和用户名。")
            return
        _busy(self.btn_test, True)
        plain = self._build_plain()

        def job():
            sess = SSHSession(plain)
            res = sess.execute("echo __SSH_OK__ && hostname && uname -sr")
            sess.close()
            if res.exit_code != 0:
                raise RuntimeError(res.summary(500))
            return f"连接成功：{res.stdout.strip()}"

        self._test_thread = _JobThread(job, self)
        self._test_thread.done.connect(self._on_test_done)
        self._test_thread.start()

    def _on_test_done(self, ok: bool, msg: str) -> None:
        _busy(self.btn_test, False)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information if ok else QMessageBox.Critical)
        box.setWindowTitle("连接测试")
        box.setText(msg[:600])
        box.exec_()

    def _save(self) -> None:
        d = self._collect()
        if not (d["name"] and d["host"] and d["username"]):
            QMessageBox.warning(self, "提示", "名称、主机、用户名不能为空。")
            return
        if d["auth_type"] == "key" and not d["key_path"]:
            QMessageBox.warning(self, "提示", "请选择私钥文件或改回密码认证。")
            return
        if self.server is not None:
            self.cfg.update_server(self.server["id"], **d)
        else:
            self.cfg.add_server(name=d["name"], host=d["host"], port=d["port"],
                                username=d["username"], auth_type=d["auth_type"],
                                password=d["password"], key_path=d["key_path"],
                                passphrase=d["passphrase"], sudo_enabled=d["sudo_enabled"],
                                sudo_password=d["sudo_password"])
        self.accept()


# --------------------------------------------------------------------------- #
# 命令执行二次确认
# --------------------------------------------------------------------------- #
def ask_confirm(parent: QWidget, request: dict) -> tuple[bool, bool]:
    """
    弹出命令确认框。

    :return: (是否放行, 是否“本会话记住”——由 Agent 的会话记忆统一处理，
             此处恒为 False 仅保留接口)
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("命令执行确认")
    dlg.setMinimumWidth(560)
    lay = QVBoxLayout(dlg)

    level = int(request.get("level", 0))
    level_label = request.get("level_label", "未知")
    colors = {0: "#1a7f37", 1: "#9a6700", 2: "#cf222e"}
    level_html = f"<span style='color:{colors.get(level,'#333')};font-weight:bold'>{level_label}</span>"
    head = QLabel(
        f"AI 请求在服务器 <b>{request.get('server','?')}</b> "
        f"（{request.get('host_user','')}）上执行命令，风险级别：{level_html}", dlg)
    head.setWordWrap(True)
    lay.addWidget(head)

    if request.get("reason"):
        reason = QLabel(f"判定依据：{request['reason']}", dlg)
        reason.setWordWrap(True)
        reason.setStyleSheet("color:#888;")
        lay.addWidget(reason)

    lay.addWidget(QLabel("将要执行的命令：", dlg))
    cmd_view = QPlainTextEdit(dlg)
    cmd_view.setPlainText(request.get("command", ""))
    cmd_view.setReadOnly(True)
    cmd_view.setMaximumHeight(160)
    cmd_view.setStyleSheet(
        "font-family:Consolas,'Courier New',monospace;"
        "background:#f6f8fa;border:1px solid #d0d7de;")
    lay.addWidget(cmd_view)

    tip = QLabel("高危命令可能对系统造成不可逆影响，请核对无误后再放行。", dlg)
    tip.setStyleSheet("color:#cf222e;" if level >= 2 else "color:#9a6700;")
    lay.addWidget(tip)

    remember = QLabel("提示：被放行的命令，本会话内再次出现相同命令时将不再重复询问。", dlg)
    remember.setStyleSheet("color:#888;")
    lay.addWidget(remember)

    btns = QDialogButtonBox(dlg)
    allow = btns.addButton("允许执行", QDialogButtonBox.AcceptRole)
    deny = btns.addButton("拒绝", QDialogButtonBox.RejectRole)
    lay.addWidget(btns)

    # 关键：必须显式接线，否则点击按钮后对话框不会关闭、操作永远无法继续
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)

    allow.setStyleSheet("background:#cf222e;color:white;font-weight:bold;" if level >= 2
                        else "background:#1a7f37;color:white;font-weight:bold;")
    deny.setDefault(True)

    ok = dlg.exec_() == QDialog.Accepted
    return ok, False


# --------------------------------------------------------------------------- #
# 审计日志查看
# --------------------------------------------------------------------------- #
class AuditDialog(QDialog):
    def __init__(self, logger: AuditLogger, parent=None):
        super().__init__(parent)
        self.logger = logger
        self.setWindowTitle("操作审计日志")
        self.resize(980, 560)
        lay = QVBoxLayout(self)

        self.table = QTableWidget(self)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["时间", "事件", "服务器", "命令/内容", "决定/退出码", "结果摘要"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        lay.addWidget(self.table)

        btns = QDialogButtonBox(self)
        refresh = btns.addButton("刷新", QDialogButtonBox.ActionRole)
        btns.addButton("关闭", QDialogButtonBox.RejectRole)
        lay.addWidget(btns)
        refresh.clicked.connect(self.reload)
        btns.rejected.connect(self.reject)
        self.reload()

    def reload(self) -> None:
        rows = self.logger.recent(800)
        self.table.setRowCount(0)
        labels = {"ssh_command": "SSH 命令", "ssh_denied": "SSH 拒绝",
                  "ai_run_start": "AI 开始", "ai_run_end": "AI 结束",
                  "ai_tool_call": "AI 调用工具", "ai_error": "AI 错误",
                  "web_search": "网络搜索"}
        for r in rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [
                r.get("time", ""),
                labels.get(r.get("event", ""), r.get("event", "")),
                r.get("server", "") or "-",
                (r.get("command") or r.get("user_text") or r.get("query") or r.get("tool", ""))[:200],
                self._decision_text(r),
                (r.get("summary") or r.get("error") or "")[:180],
            ]
            for col, v in enumerate(values):
                item = QTableWidgetItem(str(v))
                if r.get("event") == "ssh_command" and r.get("decision") == "denied_by_user":
                    item.setForeground(Qt.red)
                self.table.setItem(row, col, item)
        self.table.scrollToBottom()

    @staticmethod
    def _decision_text(r: dict) -> str:
        if r.get("decision"):
            return {"approved_by_user": "用户批准", "denied_by_user": "用户拒绝",
                    "auto_approved": "自动放行", "error": "出错"}.get(
                r["decision"], r["decision"]) + (f"  码{r['exit_code']}" if r.get("exit_code") is not None else "")
        if r.get("exit_code") is not None:
            return f"退出码 {r['exit_code']}"
        return "-"


# --------------------------------------------------------------------------- #
# 全局记忆管理
# --------------------------------------------------------------------------- #
class MemoryDialog(QDialog):
    """查看 / 新增 / 删除全局长期记忆条目。"""

    def __init__(self, mem, parent: QWidget | None = None):
        super().__init__(parent)
        self.mem = mem
        self.setWindowTitle("🧠 全局记忆")
        self.resize(560, 480)
        lay = QVBoxLayout(self)

        intro = QLabel(
            "这些内容会在每次对话开始时注入给 AI（跨会话生效）。\n"
            "也可以直接对 AI 说“记住：……”，由它自动写入。", self)
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#767676;")
        lay.addWidget(intro)

        self.list_view = QListWidget(self)
        lay.addWidget(self.list_view, 1)

        add_row = QHBoxLayout()
        self.input = QLineEdit(self)
        self.input.setPlaceholderText("输入要长期记住的内容，例如：生产环境禁止直接重启数据库…")
        btn_add = QPushButton("＋ 添加", self)
        add_row.addWidget(self.input, 1)
        add_row.addWidget(btn_add)
        lay.addLayout(add_row)

        btn_row = QHBoxLayout()
        btn_del = QPushButton("🗑 删除选中", self)
        btn_clear = QPushButton("清空全部", self)
        btn_close = QPushButton("关闭", self)
        btn_row.addWidget(btn_del)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

        self.input.returnPressed.connect(self._add)
        btn_add.clicked.connect(self._add)
        btn_del.clicked.connect(self._delete_selected)
        btn_clear.clicked.connect(self._clear_all)
        btn_close.clicked.connect(self.accept)
        self.refresh()

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        self.list_view.clear()
        for n in self.mem.notes():
            topic = n.get("topic", "")
            time_str = (n.get("time") or "")[:16]
            title = n.get("text", "")
            topic_tag = f"·{topic}  " if topic else ""
            item = QListWidgetItem(f"{title}\n{topic_tag}{time_str}")
            item.setData(Qt.UserRole, n.get("id"))
            self.list_view.addItem(item)
        if self.list_view.count() == 0:
            self.list_view.addItem("（暂无记忆）")

    def _add(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.mem.add(text, source="manual")
        self.input.clear()
        self.refresh()

    def _delete_selected(self) -> None:
        item = self.list_view.currentItem()
        if item is None:
            return
        note_id = item.data(Qt.UserRole)
        if note_id and self.mem.remove(note_id):
            self.refresh()

    def _clear_all(self) -> None:
        count = self.mem.clear()
        if count:
            self.refresh()
