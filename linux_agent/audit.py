"""
audit.py —— 审计日志
====================
把 AI 每次实际执行的 SSH 命令、批准/拒绝决定等以 JSONL 格式追加写入
~/.linux-server-agent/audit/ 目录，便于事后追查。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from .config import ConfigManager


class AuditLogger:
    """线程安全的 JSONL 审计记录器。"""

    def __init__(self, base_dir: Path | None = None):
        self._lock = threading.Lock()
        self.base_dir = Path(base_dir) if base_dir else Path.home() / ".linux-server-agent" / "audit"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _today() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _file_for(self, day: str | None = None) -> Path:
        return self.base_dir / f"audit-{day or self._today()}.jsonl"

    def log(self, event: str, **fields) -> None:
        """
        记录一条审计事件。

        :param event: 事件类型，如 ssh_command / ssh_denied / ai_start / ai_error
        :param fields: 其它字段（server、command、decision、exit_code 等）
        """
        record = {"time": datetime.now().isoformat(timespec="seconds"),
                  "event": event}
        record.update(fields)
        with self._lock:
            with open(self._file_for(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 500) -> list[dict]:
        """读取最近若干条记录（跨多天文件，新的在前）。"""
        files = sorted(self.base_dir.glob("audit-*.jsonl"), reverse=True)
        rows: list[dict] = []
        for f in files:
            try:
                lines = f.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(rows) >= limit:
                    return rows
        return rows

    def log_ssh(self, server: str, command: str, decision: str,
                exit_code: int | None = None, summary: str = "") -> None:
        """SSH 命令执行审计的快捷入口。"""
        self.log("ssh_command", server=server, command=command[:2000],
                 decision=decision, exit_code=exit_code, summary=summary[:500])


def from_config(cfg: ConfigManager) -> AuditLogger:
    """基于 ConfigManager 构造审计器。"""
    return AuditLogger(cfg.audit_dir)
