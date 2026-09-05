"""
ssh.py —— SSH 连接池与执行器
============================
基于 paramiko 封装，支持：
  - 密码认证 / 私钥认证（可选口令）
  - 懒连接 + 断线自动重连
  - sudo 密码通过 stdin 安全注入（不回显到命令行）
  - 统一返回结构化执行结果
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import paramiko


@dataclass
class SSHResult:
    """一次命令执行的结果。"""
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def summary(self, limit: int = 4000) -> str:
        """格式化为适合回填给 AI 的文本（含截断）。"""
        parts = []
        if self.stdout:
            parts.append("[stdout]\n" + self.stdout.strip())
        if self.stderr:
            parts.append("[stderr]\n" + self.stderr.strip())
        text = "\n".join(parts) if parts else "(无输出)"
        if len(text) > limit:
            text = text[:limit] + f"\n……(输出过长已截断,共{len(text)}字符)"
        return text

    def detail(self, limit: int = 6000) -> str:
        head = f"退出码: {self.exit_code}   耗时: {self.duration:.1f}s\n"
        return head + self.summary(limit)


class SSHConnectionError(Exception):
    """SSH 连接/执行失败的异常基类。"""


class SSHSession:
    """对单台服务器的 SSH 会话封装。"""

    def __init__(self, cfg: dict):
        # cfg 必须来自 ConfigManager.server_plain()（已解密）
        self.cfg = cfg
        self._client: paramiko.SSHClient | None = None
        self.last_used = 0.0

    # ------------------------------------------------------------------ #
    def connect(self, force: bool = False) -> None:
        """建立（或复用）连接。"""
        if self._client and not force:
            return
        self.close()
        cfg = self.cfg
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            if cfg.get("auth_type") == "key":
                client.connect(
                    hostname=cfg["host"],
                    port=int(cfg.get("port", 22)),
                    username=cfg["username"],
                    key_filename=cfg.get("key_path") or None,
                    passphrase=cfg.get("passphrase") or None,
                    timeout=15,
                    banner_timeout=15,
                    auth_timeout=15,
                    look_for_keys=False,
                    allow_agent=False,
                )
            else:
                client.connect(
                    hostname=cfg["host"],
                    port=int(cfg.get("port", 22)),
                    username=cfg["username"],
                    password=cfg.get("password") or None,
                    timeout=15,
                    banner_timeout=15,
                    auth_timeout=15,
                    look_for_keys=False,
                    allow_agent=False,
                )
        except paramiko.AuthenticationException as exc:
            client.close()
            raise SSHConnectionError(f"认证失败: {exc}") from exc
        except (paramiko.SSHException, OSError) as exc:
            client.close()
            raise SSHConnectionError(f"无法连接 {cfg['host']}:{cfg.get('port', 22)}: {exc}") from exc
        # 保活：防止长任务执行期间被防火墙/服务端因空闲而断开
        try:
            transport = client.get_transport()
            if transport is not None:
                transport.set_keepalive(30)
        except Exception:  # noqa: BLE001
            pass
        self._client = client

    def _ensure(self) -> paramiko.SSHClient:
        if self._client is None:
            self.connect()
        return self._client

    # ------------------------------------------------------------------ #
    def execute(self, command: str, timeout: int | None = None) -> SSHResult:
        """执行命令，支持长时间运行任务。

        - timeout 语义为“无任何输出到达的最大空闲秒数”，而非总时长上限。
          命令若持续产生输出可一直跑下去；单次空闲超过 timeout 才判定超时。
        - 若命令以 sudo 开头且服务器启用了 sudo_enabled，则改为
          `sudo -S -p '' ...` 并把 sudo 密码写入 stdin，避免密码出现在
          进程命令行参数中。
        """
        timeout = timeout or int(self.cfg.get("timeout") or 600)
        start = time.time()
        result = SSHResult()
        try:
            client = self._ensure()
        except SSHConnectionError:
            # 连接失败时重试一次（自动重连）
            self.connect(force=True)
            client = self._client

        effective = command
        stdin_data = None
        if command.lstrip().startswith("sudo") and self.cfg.get("sudo_enabled") \
                and self.cfg.get("sudo_password"):
            rest = command.lstrip()[4:].lstrip()
            effective = f"sudo -S -p '' {rest}"
            stdin_data = self.cfg["sudo_password"] + "\n"

        import socket as _socket
        channel = None
        try:
            stdin, stdout, stderr = client.exec_command(
                effective, get_pty=stdin_data is not None
            )
            channel = stdout.channel
            channel.settimeout(timeout)          # 单次读取空闲上限
            try:
                channel.set_combine_stderr(True)  # stdout/stderr 合并，避免读满阻塞
            except Exception:  # noqa: BLE001
                pass                              # pty 等场景不允许合并，忽略
            if stdin_data is not None:
                stdin.write(stdin_data)
                stdin.flush()
                stdin.close()
            parts = []
            while True:
                chunk = stdout.read(32768)
                if not chunk:
                    break
                parts.append(chunk)
            result.exit_code = channel.recv_exit_status()
            result.stdout = b"".join(parts).decode("utf-8", errors="replace")
        except _socket.timeout:
            result.timed_out = True
            result.exit_code = -1
            result.stderr = f"执行超时：命令已连续 {timeout}s 无输出，已中止该命令。"
            if channel is not None:
                try:
                    channel.close()
                except Exception:  # noqa: BLE001
                    pass
        except (paramiko.SSHException, OSError) as exc:
            result.stderr = f"SSH 执行错误: {exc}"
            result.exit_code = -1
        except Exception as exc:  # noqa: BLE001 —— 兜底并转成可读文本
            result.stderr = f"执行异常: {exc}"
            result.exit_code = -1
        finally:
            result.duration = time.time() - start
            self.last_used = time.time()
        return result

    def is_connected(self) -> bool:
        if not self._client:
            return False
        try:
            return bool(self._client.get_transport() and self._client.get_transport().is_active())
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None


def _read_all(stream) -> str:
    """读取 channel 输出；stream 在 paramiko >=3 中是 BufferedFile，支持 read。"""
    try:
        data = stream.read().decode("utf-8", errors="replace")
        return data
    except Exception:  # noqa: BLE001
        try:
            data = stream.channel.recv(65535).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""
        chunk = data
        while True:
            try:
                more = stream.channel.recv(65535)
            except Exception:  # noqa: BLE001
                break
            if not more:
                break
            chunk += more.decode("utf-8", errors="replace")
        return chunk


class SSHManager:
    """连接池：按服务器名/ID 管理多个 SSHSession，程序退出时统一关闭。"""

    def __init__(self):
        self._sessions: dict[str, SSHSession] = {}

    def get(self, cfg_plain: dict) -> SSHSession:
        """获取（必要时新建）某服务器的会话。cfg 须为解密后的字典。"""
        key = cfg_plain.get("id") or cfg_plain.get("name")
        session = self._sessions.get(key)
        if session is None or session.cfg.get("host") != cfg_plain.get("host") \
                or session.cfg.get("username") != cfg_plain.get("username"):
            session = SSHSession(cfg_plain)
            self._sessions[key] = session
        return session

    def close_all(self) -> None:
        for s in self._sessions.values():
            s.close()
        self._sessions.clear()
