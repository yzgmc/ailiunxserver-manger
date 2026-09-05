"""
memory.py —— 全局长期记忆
=========================
以 JSON 文件保存一组跨会话有效的“备忘条目”（例如用户偏好、服务器约定、
已知问题结论等），供 AI 在每次对话开始时读取，实现跨会话记忆。

文件位置：~/.linux-server-agent/memory.json
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path


class GlobalMemory:
    """线程安全的长期记忆存储。"""

    def __init__(self, base_dir: str | Path, file_name: str = "memory.json"):
        self._lock = threading.Lock()
        self.file: Path = Path(base_dir) / file_name
        self._notes: list[dict] = []
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if self.file.exists():
            try:
                data = json.loads(self.file.read_text(encoding="utf-8"))
                self._notes = list(data) if isinstance(data, list) else []
            except (json.JSONDecodeError, OSError):
                self._notes = []
        self._normalize()

    def _normalize(self) -> None:
        """保证条目结构完整并排序（新的在前）。"""
        cleaned = []
        for i, n in enumerate(self._notes):
            if isinstance(n, dict) and str(n.get("text", "")).strip():
                cleaned.append({
                    "id": str(n.get("id") or f"m{i}"),
                    "text": str(n["text"]).strip(),
                    "topic": str(n.get("topic", "")).strip(),
                    "time": str(n.get("time") or datetime.now().isoformat(timespec="seconds")),
                })
        self._notes = cleaned

    def _save(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._notes, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.file)

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #
    def notes(self) -> list[dict]:
        """返回所有备忘条目（副本）。"""
        with self._lock:
            return list(self._notes)

    def add(self, text: str, topic: str = "", source: str = "manual") -> dict:
        """新增一条记忆，返回创建的条目。"""
        text = (text or "").strip()
        if not text:
            raise ValueError("记忆内容不能为空")
        note = {
            "id": f"m{datetime.now().strftime('%H%M%S%f')}",
            "text": text,
            "topic": (topic or "").strip(),
            "time": datetime.now().isoformat(timespec="seconds"),
            "source": source,
        }
        with self._lock:
            self._notes.insert(0, note)
            self._save()
        return note

    def remove(self, note_id: str) -> bool:
        """按 id 删除一条记忆。"""
        with self._lock:
            before = len(self._notes)
            self._notes = [n for n in self._notes if n.get("id") != note_id]
            removed = len(self._notes) < before
            if removed:
                self._save()
            return removed

    def clear(self) -> int:
        """清空全部记忆，返回删除条数。"""
        with self._lock:
            count = len(self._notes)
            self._notes = []
            if count:
                self._save()
            return count

    # ------------------------------------------------------------------ #
    def summary(self, limit: int = 30) -> str:
        """生成注入系统提示词的多行摘要文本（空则返回空串）。"""
        with self._lock:
            notes = list(self._notes[:limit])
        if not notes:
            return ""
        lines = []
        for n in notes:
            topic = f"（{n['topic']}）" if n.get("topic") else ""
            lines.append(f"- {n['text']}{topic}")
        return "\n".join(lines)
