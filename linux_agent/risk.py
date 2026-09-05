"""
risk.py —— 命令风险评估
========================
在执行 AI/用户命令前做静态风险分级：

  RiskLevel.SAFE       只读信息类命令（ls/cat/df/ps/journalctl …）
  RiskLevel.WRITE      修改系统/数据类命令（装软件、重启服务、写文件…）
  RiskLevel.DANGEROUS  高风险破坏性命令（删除、格式化、关机重启…）

分级是静态启发式，最终安全性依赖二次确认机制与用户判断。
"""

from __future__ import annotations

import re
from enum import IntEnum

# --------------------------------------------------------------------------- #
# 动词表
# --------------------------------------------------------------------------- #
# 纯只读命令
READ_VERBS = {
    "ls", "cat", "less", "more", "head", "tail", "grep", "find", "df", "du",
    "free", "uptime", "who", "w", "last", "history", "ps", "top", "htop",
    "netstat", "ss", "ifconfig", "ping", "dig", "nslookup", "getent",
    "journalctl", "dmesg", "dstat", "vmstat", "iostat", "sar", "uname",
    "hostname", "hostnamectl", "date", "timedatectl", "stat", "file", "which",
    "whereis", "id", "groups", "env", "printenv", "pwd", "lsof", "mount",
    "blkid", "lsblk", "lscpu", "lsmem", "lsusb", "lspci", "locale",
    "localectl", "loginctl", "systemd-analyze", "type", "help", "sysctl",
}

# 纯写操作命令
WRITE_VERBS = {
    "apt", "apt-get", "yum", "dnf", "zypper", "apk", "dpkg", "snap", "flatpak",
    "pip", "pip3", "npm", "yarn", "pnpm", "cnpm", "gem", "cargo", "composer",
    "iptables", "nft", "ufw", "firewall-cmd", "setenforce",
    "useradd", "usermod", "passwd", "chsh", "chage", "groupadd", "groupmod",
    "chmod", "chown", "chattr", "mv", "cp", "touch", "mkdir", "rmdir", "ln",
    "kill", "pkill", "killall", "tee", "sed", "crontab", "at", "batch",
    "make", "cmake", "install", "certbot", "update-alternatives",
    "ssh-keygen", "openssl", "wget", "unzip", "zip", "gzip", "gunzip", "bzip2", "xz",
    "ddrescue", "rsync", "scp", "sftp",
}

# 按参数决定读/写的“上下文动词”：(默认是否写, 只读正则, 写正则)
CONTEXT_VERBS: dict[str, tuple[bool, str, str]] = {
    "systemctl": (True,
                  r"\b(status|list-units|list-timers|list-sockets|list-dependencies|is-active|is-enabled|show|cat|help)\b",
                  r"\b(start|stop|restart|reload|enable|disable|mask|unmask|daemon-reload|set-default|reset-failed|kill|edit)\b"),
    "service": (True, r"\b(status)\b", r"\b(start|stop|restart|reload|force-reload)\b"),
    "docker": (True,
               r"\b(ps|images|inspect|logs|stats|info|version|search|history|port|top|network\s+ls|volume\s+ls|image\s+ls)\b",
               r"\b(run|start|stop|restart|rm|rmi|kill|exec|build|pull|push|commit|tag|save|load|network\s+\w+|volume\s+\w+|compose|update)\b"),
    "git": (True,
            r"\b(status|log|diff|show|branch|remote\s+-v|config\s+(-l|--list)|\s-l$|tag|-l)\b",
            r"\b(push|commit|reset|checkout|clean|revert|rebase|merge|branch\s+-D|tag\s+-d|fetch|pull|clone|init|add|rm|mv|submodule)\b"),
    "ip": (True,
           r"\b(addr|address|link\s+show|route\s+(show|get|list)|neigh|maddr|monitor|stats)\b",
           r"\b(link\s+(set|add|del)|route\s+(add|del|replace|change|flush)|addr\s+(add|del|flush)|rule|tunnel|neigh\s+(add|del|replace))\b"),
    "curl": (False,
             r".*",  # 默认读远端
             r"(\s-o\s|\s--output\s|>\s|\s-d\s|\s--data\b|\s-X\s|\s--request\b|\s-F\b|\s--form\b)"),
    "echo": (False, r".*", r">>|>"),
    "printf": (False, r".*", r">>|>"),
    "tar": (True,
            r"(\s-t\b|--list\b|\s-v\s+t)|\s-t[f-vz]*\s|--list",
            r"(\s-x\b|--extract|\s-x[f-vz]*|--create|\s-c[f-vz]*\s|--append|\s-r[f-vz]*\s|--delete|\s-d\b|--update|\s-u\b)"),
    "nmcli": (True, r"\b(general|device\s+status|connection\s+show)\b", r"\b(connection\s+(up|down|modify|add|delete)|device\s+(connect|disconnect|reapply|wifi)|radio)\b"),
}

# 高危命令动词：一旦出现即 DANGEROUS
DANGEROUS_VERBS = {
    "rm", "mkfs", "mkfs.ext4", "mkfs.xfs", "mkswap", "shutdown", "reboot",
    "halt", "poweroff", "dd", "init", "fdisk", "parted", "wipefs",
    "pvcreate", "vgremove", "lvremove", "lvcreate", "vgcreate", "userdel",
    "groupdel", "deluser", "delgroup", "resize2fs", "debugfs", "badblocks", "format",
}

# 高危特征正则：任何命令命中即 DANGEROUS
DANGEROUS_PATTERNS = [
    r"\brm\b\s+-[a-z]*r",                                # 任何 rm -r / rm -f
    r"(^|[;&|]\s*)rm\b",                                 # 任何 rm（含单文件删除也要确认）
    r"\bchmod\s+(-R\s+)?\d{4}\s+/\s*$",                  # chmod 777 / 根目录
    r"\bchown\s+(-R\s+)?[^\s]+\s+/\s*$",
    r"\b(>|>>)\s*/dev/sd\w+",                            # 直接写块设备
    r"(^|[;&|]\s*)mkfs", r"(^|[;&|]\s*)wipefs",
    r"\binit\s+[06]\b",
    r"(^|[;&|]\s*)(:\(\)\s*\{|shutdown|reboot|halt|poweroff)",
    r"\bdd\s+if=.*of=/dev/",
    r"\brm\s+(-rf?\s+)?(/\*|/)\s*$",
]

# 命令分隔符（多命令拼接往往需要人工确认）
META_SEPARATORS = re.compile(r"[;&|]|\$\(|`|\n")


class RiskLevel(IntEnum):
    """命令风险级别，数值越高风险越大。"""
    SAFE = 0
    WRITE = 1
    DANGEROUS = 2

    @property
    def label(self) -> str:
        return {0: "安全（只读）", 1: "修改操作", 2: "高危操作"}[self.value]


def classify(command: str) -> tuple[RiskLevel, str]:
    """
    对单条命令进行静态风险分级。

    :return: (RiskLevel, 判定说明)
    """
    cmd = (command or "").strip()
    if not cmd:
        return RiskLevel.SAFE, "空命令"

    low = cmd.lower()

    # 1) 高危特征直接命中 -> DANGEROUS
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, low):
            return RiskLevel.DANGEROUS, f"命中高危特征：{pat}"

    # 2) 提取命令动词（去掉 sudo / env / nohup 前缀）
    verb, rest = _first_verb(low)
    if verb in DANGEROUS_VERBS:
        return RiskLevel.DANGEROUS, f"危险命令动词：{verb}"

    # 3) 上下文动词（按参数区分）
    if verb in CONTEXT_VERBS:
        write_default, safe_re, write_re = CONTEXT_VERBS[verb]
        if write_re and re.search(write_re, rest):
            return RiskLevel.WRITE, f"{verb} 写操作"
        if safe_re and re.search(safe_re, rest):
            return RiskLevel.SAFE, f"{verb} 只读操作"
        return RiskLevel.WRITE if write_default else RiskLevel.SAFE, \
            f"{verb} 按{'写' if write_default else '只读'}处理"

    # 4) 纯只读 / 纯写动词
    if verb in READ_VERBS:
        if verb == "find" and re.search(r"-(delete|exec|ok)\b", rest):
            return RiskLevel.DANGEROUS, "find -delete/-exec 可能删除或执行"
        if _WRITE_META.search(rest):
            return RiskLevel.WRITE, f"{verb} 重定向写文件"
        return RiskLevel.SAFE, f"只读命令：{verb}"

    if verb in WRITE_VERBS:
        return RiskLevel.WRITE, f"写操作命令：{verb}"

    # 5) 未知动词：存在重定向/多命令则按写处理
    if _WRITE_META.search(low) or META_SEPARATORS.search(low):
        return RiskLevel.WRITE, "包含重定向或多命令分隔符"
    return RiskLevel.SAFE, "未识别动词，按只读处理"


_WRITE_META = re.compile(r">>|>|\btee\b|\bdd\b")


def _first_verb(low: str) -> tuple[str, str]:
    """去掉前缀后返回 (首动词, 剩余内容)。"""
    rest = low
    while re.match(r"^\s*(sudo|env|time|command|nohup|setsid)\s+", rest):
        rest = re.sub(r"^\s*(sudo|env|time|command|nohup|setsid)\s+", "", rest, count=1)
    m = re.match(r"^([^\s;&|]+)", rest)
    verb = m.group(1) if m else (rest or "")
    return verb, rest[len(verb):]


def safety_threshold(mode: str) -> int:
    """把安全配置模式映射为“需要确认的最低风险级别”。

    auto → 全部自动执行(阈值 3，永不确认)
    standard → 修改与高危需确认
    strict → 所有命令都确认
    """
    return {"auto": 3, "standard": RiskLevel.WRITE, "strict": RiskLevel.SAFE}.get(mode, RiskLevel.WRITE)
