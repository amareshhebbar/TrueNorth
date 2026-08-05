"""
Built-in datetime tool — returns current date/time in any timezone.

Use cases:
  - Goal needs to know today's date for scheduling
  - Age validation (born 1996 → today's year - 1996 = 28)
  - Appointment booking agents
  - Reminder/scheduler goals

YAML:
    mcp_servers:
      - name: datetime_tool
        builtin: true

LLM call:
    TOOL_CALL: datetime_tool({})
    TOOL_CALL: datetime_tool({"timezone": "Asia/Kolkata"})
    TOOL_CALL: datetime_tool({"timezone": "America/New_York", "format": "%B %d, %Y"})
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Dict

from truenorth.mcp.builtin import register

logger = logging.getLogger(__name__)

_ALIASES: Dict[str, str] = {

    "ist":              "Asia/Kolkata",
    "india":            "Asia/Kolkata",
    "mumbai":           "Asia/Kolkata",
    "delhi":            "Asia/Kolkata",
    "bangalore":        "Asia/Kolkata",

    "est":              "America/New_York",
    "edt":              "America/New_York",
    "cst":              "America/Chicago",
    "mst":              "America/Denver",
    "pst":              "America/Los_Angeles",
    "pdt":              "America/Los_Angeles",
    "eastern":          "America/New_York",
    "central":          "America/Chicago",
    "pacific":          "America/Los_Angeles",

    "gmt":              "UTC",
    "bst":              "Europe/London",
    "london":           "Europe/London",
    "uk":               "Europe/London",
    "cet":              "Europe/Paris",
    "paris":            "Europe/Paris",
    "berlin":           "Europe/Berlin",

    "jst":              "Asia/Tokyo",
    "japan":            "Asia/Tokyo",
    "tokyo":            "Asia/Tokyo",
    "cst china":        "Asia/Shanghai",
    "beijing":          "Asia/Shanghai",
    "shanghai":         "Asia/Shanghai",
    "singapore":        "Asia/Singapore",
    "sgt":              "Asia/Singapore",
    "dubai":            "Asia/Dubai",

    "aest":             "Australia/Sydney",
    "sydney":           "Australia/Sydney",
    "melbourne":        "Australia/Melbourne",
    "perth":            "Australia/Perth",

    "utc":              "UTC",
    "universal":        "UTC",
}

@register("datetime_tool")
async def datetime_tool(
    timezone: str = "UTC",
    format:   str = "%Y-%m-%d %H:%M:%S",
) -> Dict[str, Any]:
    """
    Return the current date and time in the specified timezone.
    Supports IANA timezone names (e.g. Asia/Kolkata) and common aliases.
    """
    tz_name = _resolve_timezone(timezone)

    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            tz = ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, KeyError):
            logger.warning("datetime_tool: unknown timezone %r, falling back to UTC", tz_name)
            tz      = ZoneInfo("UTC")
            tz_name = "UTC"
    except ImportError:
        tz      = _dt.timezone.utc
        tz_name = "UTC"

    now = _dt.datetime.now(tz)

    try:
        formatted = now.strftime(format)
    except ValueError:
        formatted = now.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "datetime":   formatted,
        "iso":        now.isoformat(),
        "timezone":   tz_name,
        "date":       now.strftime("%Y-%m-%d"),
        "time":       now.strftime("%H:%M:%S"),
        "year":       now.year,
        "month":      now.month,
        "day":        now.day,
        "hour":       now.hour,
        "minute":     now.minute,
        "weekday":    now.strftime("%A"),
        "unix":       int(now.timestamp()),
    }

def _resolve_timezone(tz: str) -> str:
    """Resolve a timezone string — handles aliases and loose names."""
    if not tz:
        return "UTC"
    key = tz.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    return tz.strip()
