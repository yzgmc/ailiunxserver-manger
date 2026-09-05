"""
Linux 服务器管理 AI 代理 (Linux Server AI Agent)
==================================================

一个基于 PyQt5 的桌面程序：
  - 可配置 DeepSeek 或其它任何兼容 OpenAI API 的模型服务商（如 LM Studio）
  - 由 AI 作为 Agent，通过 SSH 管理你的 Linux 服务器
  - 集成网络搜索，辅助解决服务器运维问题
  - 具备命令风险分级、执行前二次确认、凭据加密与操作审计
"""

__version__ = "1.0.0"
APP_NAME = "Linux Server AI Agent"
