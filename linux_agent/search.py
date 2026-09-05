"""
search.py —— 网络搜索
=====================
集成两个免 API Key 的 HTML 搜索源（DuckDuckGo / Bing），
互为兜底，结果统一为结构化列表，方便回填给 AI。

说明：免 Key 抓取可能被反爬或网络环境限制，失败时会抛出 SearchError，
AI 会得到可读的错误提示（例如建议在“AI 设置”中确认网络可达）。
"""

from __future__ import annotations

import html as html_lib
import re
import urllib.parse
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""

    def to_text(self, idx: int) -> str:
        snip = re.sub(r"\s+", " ", self.snippet).strip()
        return f"[{idx}] {self.title}\n    链接: {self.url}\n    摘要: {snip or '（无摘要）'}"


class SearchError(Exception):
    """搜索失败。"""


def _request(url: str, **kw) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=12, **kw)
    resp.raise_for_status()
    return resp


# --------------------------------------------------------------------------- #
# DuckDuckGo
# --------------------------------------------------------------------------- #
def _duckduckgo(query: str, limit: int) -> list[SearchResult]:
    resp = _request(
        "https://html.duckduckgo.com/html/",
        params={"q": query, "kl": "cn-zh", "ia": "web"},
    )
    soup = BeautifulSoup(resp.text, "lxml")
    results: list[SearchResult] = []
    for tag in soup.select("div.result__body")[:limit]:
        a = tag.select_one("a.result__a")
        if not a:
            continue
        url = a.get("href", "")
        if "uddg=" in url:
            url = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("uddg", [url])[0]
        snippet_tag = tag.select_one("a.result__snippet")
        snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""
        results.append(SearchResult(title=a.get_text(" ", strip=True),
                                    url=url,
                                    snippet=html_lib.unescape(snippet)))
    if not results:
        raise SearchError("DuckDuckGo 未返回结果（可能被网络限制）")
    return results


# --------------------------------------------------------------------------- #
# Bing
# --------------------------------------------------------------------------- #
def _bing(query: str, limit: int) -> list[SearchResult]:
    resp = _request("https://www.bing.com/search", params={"q": query, "setlang": "zh-hans"})
    soup = BeautifulSoup(resp.text, "lxml")
    results: list[SearchResult] = []
    for li in soup.select("li.b_algo")[:limit]:
        a = li.select_one("h2 a")
        if not a:
            continue
        url = a.get("href", "")
        snippet = " ".join(p.get_text(" ", strip=True) for p in li.select("div.b_caption p"))
        results.append(SearchResult(title=a.get_text(" ", strip=True),
                                    url=url,
                                    snippet=html_lib.unescape(snippet)))
    if not results:
        raise SearchError("Bing 未返回结果")
    return results


_ENGINES = {"duckduckgo": _duckduckgo, "bing": _bing}


def search_web(query: str, engine: str = "duckduckgo", max_results: int = 6) -> list[SearchResult]:
    """执行搜索，主引擎失败时自动尝试备选引擎。"""
    order = [engine] if engine in _ENGINES else []
    order += [e for e in _ENGINES if e not in order]

    last_err: Exception | None = None
    for eng in order:
        try:
            return _ENGINES[eng](query, max(max_results, 1))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise SearchError(f"所有搜索源均失败（{last_err}）")


def format_results(results: list[SearchResult], query: str) -> str:
    """把搜索结果格式化为给 AI 的文本。"""
    if not results:
        return f"搜索 “{query}” 无结果。"
    lines = [f"关于 “{query}” 的搜索结果："]
    lines += [r.to_text(i + 1) for i, r in enumerate(results)]
    return "\n".join(lines)
