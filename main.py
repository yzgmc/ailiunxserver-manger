"""
main.py —— Linux 服务器 AI 管理代理 程序入口
=============================================
用法：先安装依赖，再运行

    pip install -r requirements.txt
    python main.py
"""

import sys
import traceback


def main() -> int:
    # 高 DPI 支持（必须在 QApplication 创建前设置）
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Linux Server AI Agent")
    # 基础字号整体放大一档
    _font = app.font()
    _font.setPointSize(10)
    app.setFont(_font)

    try:
        from linux_agent.config import ConfigManager
        from linux_agent.gui.main_window import MainWindow

        cfg = ConfigManager()
    except Exception as exc:  # noqa: BLE001 —— 启动期错误用消息框提示
        traceback.print_exc()
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(None, "启动失败", f"初始化配置出错：\n{exc}")
        return 1

    window = MainWindow(cfg)
    window.showMaximized()   # 最大化启动：聊天区始终占满屏幕，不再被挤压
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
