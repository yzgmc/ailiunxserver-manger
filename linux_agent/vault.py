"""
vault.py —— 凭据加密存储
========================
使用 `cryptography` 的 Fernet 对称加密为 API Key、SSH 密码等敏感信息提供
加密保护。密钥保存在数据目录下的 secret.key 文件中（不随配置一起保存），
因此即使配置文件泄露，凭据也无法被直接读取。
"""

from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet

ENC_PREFIX = "enc:"          # 配置文件中加密值的标记前缀
MARKER = "\\x00"             # 占位，无实际用途（保留扩展位）


class Vault:
    """提供加解密能力的本地凭据保险箱。"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        """加载已有密钥，或首次运行时生成并保存。"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        key_file = self.data_dir / "secret.key"
        if key_file.exists():
            return key_file.read_bytes()
        key = Fernet.generate_key()
        key_file.write_bytes(key)
        try:
            os.chmod(key_file, 0o600)  # 限制为当前用户可读写（Unix 下有效）
        except OSError:
            pass  # Windows 无 POSIX 权限位，忽略
        return key

    def encrypt_text(self, plain: str) -> str:
        """加密明文，返回加密后的字符串。"""
        return self._fernet.encrypt(plain.encode("utf-8")).decode("ascii")

    def decrypt_text(self, token: str) -> str:
        """解密字符串；解密失败时抛出 ValueError。"""
        return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")

    # ---------- 配合 ConfigManager 使用的便捷方法 ----------

    def seal(self, plain: str | None) -> str:
        """把明文包装为带 enc: 前缀的加密字段值；None/空 原样返回。"""
        if not plain:
            return ""
        return ENC_PREFIX + self.encrypt_text(plain)

    def open(self, value: str) -> str:
        """读取字段值：带 enc: 前缀则解密，否则视为旧版明文直接返回。"""
        if isinstance(value, str) and value.startswith(ENC_PREFIX):
            return self.decrypt_text(value[len(ENC_PREFIX):])
        return value or ""
