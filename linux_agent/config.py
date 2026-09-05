"""
config.py —— 配置管理器
========================
负责统一的 JSON 配置文件读写（含密钥字段的透明加密），并提供模型服务商
预设与服务器列表 CRUD。配置文件默认位于用户主目录：
    ~/.linux-server-agent/config.json
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from pathlib import Path

from .vault import Vault

# --------------------------------------------------------------------------- #
# 模型服务商预设（OpenAI 兼容 /v1 API）
# --------------------------------------------------------------------------- #
PROVIDER_PRESETS: dict = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro",
                   "deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-v4-flash",
        "needs_key": True,
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini"],
        "default_model": "gpt-4o-mini",
        "needs_key": True,
    },
    "lmstudio": {
        "label": "LM Studio (本地)",
        "base_url": "http://localhost:1234/v1",
        "models": [],
        "default_model": "",
        "needs_key": False,
    },
    "custom": {
        "label": "自定义(OpenAI 兼容)",
        "base_url": "",
        "models": [],
        "default_model": "",
        "needs_key": False,
    },
}

DEFAULT_CONFIG: dict = {
    "ai": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "",          # enc:xxxxx
        "model": "deepseek-v4-flash",   # V4-Flash：1M 上下文，适合超长任务
        "temperature": 0.2,
        "max_tokens": 8192,
        "context_budget": 200000,  # 上下文预算(Tokens)：超出后自动压缩旧内容而非中断
        "stream": True,
        "tool_style": "auto",   # auto=原生工具调用 ; text=文本格式(旧模型兜底)
        "max_iterations": 60,
    },
    "search": {"engine": "duckduckgo", "max_results": 6},
    "safety": {"mode": "standard", "force_danger_confirm": True},  # auto/standard/strict
    "active_server": "",
    "servers": [],
}


def _deep_update(base: dict, patch: dict) -> dict:
    """递归合并 patch 到 base 的副本中，用于兼容旧版本缺省字段。"""
    out = deepcopy(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


class ConfigManager:
    """配置管理单例式对象（建议在主线程创建后向下传递）。"""

    def __init__(self, base_dir: str | Path | None = None):
        base_dir = Path(base_dir) if base_dir else Path.home() / ".linux-server-agent"
        self.base_dir: Path = Path(base_dir)
        self.config_file: Path = self.base_dir / "config.json"
        self.audit_dir: Path = self.base_dir / "audit"
        self.vault = Vault(self.base_dir)
        self.cfg: dict = DEFAULT_CONFIG
        self.load()

    # ------------------------------------------------------------------ #
    # 读写
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """加载配置；文件不存在时写入默认配置。"""
        if self.config_file.exists():
            try:
                raw = json.loads(self.config_file.read_text(encoding="utf-8"))
                self.cfg = _deep_update(DEFAULT_CONFIG, raw)
                return
            except (json.JSONDecodeError, OSError) as exc:
                # 配置损坏时备份并重建，避免程序无法启动
                backup = self.config_file.with_suffix(".broken.json")
                try:
                    self.config_file.replace(backup)
                except OSError:
                    pass
                raise RuntimeError(f"配置文件损坏，已备份到 {backup}: {exc}") from exc
        self.save()

    def save(self) -> None:
        """把配置写回磁盘。"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.config_file.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.config_file)

    # ------------------------------------------------------------------ #
    # AI 配置
    # ------------------------------------------------------------------ #
    def get_ai_config(self) -> dict:
        """返回可直接用于客户端连接的 AI 配置（api_key 已解密）。"""
        ai = self.cfg["ai"]
        return {
            "provider": ai.get("provider", ""),
            "base_url": (ai.get("base_url") or "").rstrip("/"),
            "api_key": self.vault.open(ai.get("api_key", "")),
            "model": ai.get("model", ""),
            "temperature": float(ai.get("temperature", 0.2)),
            "max_tokens": int(ai.get("max_tokens", 8192)),
            "context_budget": int(ai.get("context_budget", 200000) or 200000),
            "stream": bool(ai.get("stream", True)),
            "tool_style": ai.get("tool_style", "auto"),
            "max_iterations": int(ai.get("max_iterations", 20)),
        }

    def set_ai_config(self, provider, base_url, api_key, model,
                      temperature=0.2, max_tokens=8192, context_budget=200000,
                      tool_style="auto", max_iterations=60) -> None:
        """保存 AI 服务商配置（api_key 加密存储）。"""
        ai = self.cfg["ai"]
        ai["provider"] = provider
        ai["base_url"] = (base_url or "").rstrip("/")
        ai["api_key"] = self.vault.seal(api_key)
        ai["model"] = model
        ai["temperature"] = float(temperature)
        ai["max_tokens"] = int(max_tokens)
        ai["context_budget"] = int(context_budget or 200000)
        ai["tool_style"] = tool_style
        ai["max_iterations"] = int(max_iterations)
        self.save()

    # ------------------------------------------------------------------ #
    # 服务器配置
    # ------------------------------------------------------------------ #
    def servers(self) -> list[dict]:
        return list(self.cfg.get("servers", []))

    def server_by_name(self, name: str) -> dict | None:
        """按名称查找服务器原始配置（含加密字段）。"""
        for s in self.servers():
            if s.get("name") == name or s.get("id") == name:
                return s
        return None

    def server_plain(self, server: dict | None) -> dict | None:
        """
        把服务器配置中的加密字段解密，得到可直接用于 SSH 连接的字典。
        """
        if not server:
            return None
        return {
            "id": server.get("id", ""),
            "name": server.get("name", ""),
            "host": server.get("host", ""),
            "port": int(server.get("port", 22) or 22),
            "username": server.get("username", ""),
            "auth_type": server.get("auth_type", "password"),
            "password": self.vault.open(server.get("password", "")),
            "key_path": server.get("key_path", ""),
            "passphrase": self.vault.open(server.get("passphrase", "")),
            "sudo_password": self.vault.open(server.get("sudo_password", "")),
            "sudo_enabled": bool(server.get("sudo_enabled", False)),
            # 早期版本默认写死 30s，现统一视为“宽松缺省”：<=60 秒的旧值放宽为 600s
            "timeout": 600 if int(server.get("timeout", 600) or 600) <= 60
            else int(server.get("timeout")),
        }

    def add_server(self, name, host, port, username, auth_type="password",
                   password="", key_path="", passphrase="",
                   sudo_enabled=False, sudo_password="") -> dict:
        """新增服务器，返回其 id。"""
        server = {
            "id": uuid.uuid4().hex[:12],
            "name": name,
            "host": host,
            "port": int(port or 22),
            "username": username,
            "auth_type": auth_type,
            "password": self.vault.seal(password),
            "key_path": key_path or "",
            "passphrase": self.vault.seal(passphrase),
            "sudo_password": self.vault.seal(sudo_password),
            "sudo_enabled": bool(sudo_enabled),
            "timeout": 600,
        }
        self.cfg["servers"].append(server)
        if not self.cfg.get("active_server"):
            self.cfg["active_server"] = name
        self.save()
        return server

    def update_server(self, server_id: str, **fields) -> bool:
        """按 id 更新服务器字段。secret 字段传明文，函数内部加密。"""
        for s in self.cfg["servers"]:
            if s.get("id") != server_id:
                continue
            for k, v in fields.items():
                if k in ("password", "passphrase", "sudo_password"):
                    s[k] = self.vault.seal(v)
                else:
                    s[k] = v
            self.save()
            return True
        return False

    def delete_server(self, server_id: str) -> bool:
        """按 id 删除服务器。"""
        before = len(self.cfg["servers"])
        self.cfg["servers"] = [s for s in self.cfg["servers"] if s.get("id") != server_id]
        if self.cfg.get("active_server") == server_id:
            self.cfg["active_server"] = ""
        removed = len(self.cfg["servers"]) < before
        if removed:
            self.save()
        return removed

    # ------------------------------------------------------------------ #
    # 当前选中服务器
    # ------------------------------------------------------------------ #
    def get_active_server_name(self) -> str:
        return self.cfg.get("active_server", "") or ""

    def set_active_server(self, name: str) -> None:
        self.cfg["active_server"] = name
        self.save()

    def active_server_plain(self) -> dict | None:
        return self.server_plain(self.server_by_name(self.get_active_server_name()))

    # ------------------------------------------------------------------ #
    # 搜索 / 安全
    # ------------------------------------------------------------------ #
    def get_search_config(self) -> dict:
        s = self.cfg.get("search", {})
        return {"engine": s.get("engine", "duckduckgo"),
                "max_results": int(s.get("max_results", 6) or 6)}

    def set_search_config(self, engine: str, max_results: int) -> None:
        self.cfg["search"] = {"engine": engine, "max_results": int(max_results)}
        self.save()

    def get_safety_config(self) -> dict:
        s = self.cfg.get("safety", {})
        return {"mode": s.get("mode", "standard"),
                "force_danger_confirm": bool(s.get("force_danger_confirm", True))}

    def set_safety_config(self, mode: str, force_danger: bool) -> None:
        self.cfg["safety"] = {"mode": mode, "force_danger_confirm": bool(force_danger)}
        self.save()
