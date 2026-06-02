"""
Built-in web search tool using DuckDuckGo Instant Answer API.
Zero API key required. Works globally.

Used for: looking up facts, checking current info, research tasks.
Not used for: streaming news (use a news MCP server for that).

YAML:
    mcp_servers:
      - name: web_search
        builtin: true

LLM call:
    TOOL_CALL: web_search({"query": "BMI healthy range for adults", "limit": 3})
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any, Dict, List

from truenorth.mcp.builtin import register

logger = logging.getLogger(__name__)

_DDG_URL  = "https://api.duckduckgo.com/"
_USER_AGENT = "TrueNorth/0.1 (AI Agent Framework; +https://github.com/truenorth-ai/truenorth)"


@register("web_search")
async def web_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """
    Search the web and return top results.
    Uses DuckDuckGo — zero API key required.
    Returns a dict with 'results' (list) and 'abstract' (str).
    """
    limit = max(1, min(limit, 10))
    
    try:
        ddg_result = await _ddg_instant(query)
        if ddg_result["abstract"] or ddg_result["results"]:
            ddg_result["results"] = ddg_result["results"][:limit]
            return ddg_result
    except Exception as e:
        logger.debug("web_search: DDG instant failed: %s", e)

    try:
        scrape_result = await _ddg_scrape(query, limit)
        if scrape_result:
            return {"query": query, "results": scrape_result, "abstract": ""}
    except Exception as e:
        logger.debug("web_search: DDG scrape failed: %s", e)

    # Final fallback: return empty (don't raise — tool errors break flow)
    logger.warning("web_search: all methods failed for query=%r", query[:80])
    return {
        "query":    query,
        "results":  [],
        "abstract": "",
        "error":    "Search unavailable — check network connectivity",
    }


async def _ddg_instant(query: str) -> Dict[str, Any]:
    """DuckDuckGo Instant Answer API — returns structured data."""
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx required: pip install httpx")

    params = {
        "q":      query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
    }
    url = f"{_DDG_URL}?{urllib.parse.urlencode(params)}"

    async with httpx.AsyncClient(
        headers = {"User-Agent": _USER_AGENT},
        follow_redirects = True,
        timeout = 8.0,
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    abstract = data.get("Abstract", "") or data.get("Answer", "")
    source   = data.get("AbstractSource", "") or data.get("AnswerType", "")

    related = []
    for topic in data.get("RelatedTopics", []):
        if isinstance(topic, dict) and topic.get("Text"):
            related.append({
                "title": topic.get("Text", "")[:120],
                "url":   topic.get("FirstURL", ""),
            })

    return {
        "query":    query,
        "abstract": abstract[:500] if abstract else "",
        "source":   source,
        "results":  related,
    }


async def _ddg_scrape(query: str, limit: int) -> List[Dict[str, str]]:
    """
    Lightweight DuckDuckGo HTML scrape.
    Only extracts titles and URLs — no full page content.
    """
    try:
        import httpx
    except ImportError:
        return []

    params = urllib.parse.urlencode({"q": query, "kl": "us-en"})
    url    = f"https://html.duckduckgo.com/html/?{params}"

    async with httpx.AsyncClient(
        headers         = {"User-Agent": _USER_AGENT},
        follow_redirects = True,
        timeout         = 8.0,
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    title_re   = re.compile(r'class="result__a"[^>]*>([^<]+)</a>', re.IGNORECASE)
    snippet_re = re.compile(r'class="result__snippet"[^>]*>([^<]+)</[a-z]+>', re.IGNORECASE)
    url_re     = re.compile(r'class="result__url"[^>]*>([^<]+)<', re.IGNORECASE)

    titles   = title_re.findall(html)
    snippets = snippet_re.findall(html)
    urls     = url_re.findall(html)

    results = []
    for i in range(min(limit, len(titles))):
        results.append({
            "title":   _clean(titles[i] if i < len(titles) else ""),
            "snippet": _clean(snippets[i] if i < len(snippets) else ""),
            "url":     _clean(urls[i] if i < len(urls) else ""),
        })

    return results


def _clean(text: str) -> str:
    """Strip HTML entities and extra whitespace."""
    text = re.sub(r"&[a-z]+;|&#\d+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()