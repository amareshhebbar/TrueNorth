"""truenorth/mcp/builtin — built-in MCP tools."""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional

_BUILTINS: Dict[str, Callable] = {}

def register(name: str):
    """Decorator: register a built-in tool function by name."""
    def decorator(fn: Callable) -> Callable:
        _BUILTINS[name] = fn
        return fn
    return decorator

def get_builtin(name: str) -> Optional[Callable]:
    """Return a built-in tool callable, or None if not registered."""
    _ensure_loaded()
    return _BUILTINS.get(name)

def list_builtins() -> list:
    _ensure_loaded()
    return list(_BUILTINS.keys())

_loaded = False

def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from truenorth.mcp.builtin import calculator    # noqa
    except Exception:
        pass
    try:
        from truenorth.mcp.builtin import datetime_tool # noqa
    except Exception:
        pass
    try:
        from truenorth.mcp.builtin import web_search    # noqa
    except Exception:
        pass
