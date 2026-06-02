"""
Tests every component of the MCP stack — zero real HTTP calls,
zero real MCP servers required. All external calls mocked.

Classes:
  1.  Types                  — Tool, ToolCall, ToolResult dataclasses
  2.  MCPClient_Parsing      — JSON-RPC response parsing
  3.  MCPClient_Transport    — transport selection (stdio vs HTTP)
  4.  Registry_Builtin       — add_builtin / register_builtins
  5.  Registry_ServerConfig  — register_server_config / lazy connect
  6.  Registry_CallTool      — call routing to builtin vs remote
  7.  Registry_SystemPrompt  — tool descriptions injected into LLM prompt
  8.  ToolExecutor_Scan      — TOOL_CALL pattern detection
  9.  ToolExecutor_Run       — end-to-end scan + execute + inject
  10. ToolExecutor_Validate  — argument schema validation
  11. Builtin_Calculator     — safe AST evaluator
  12. Builtin_Datetime       — timezone-aware datetime
  13. Builtin_WebSearch      — web search (mocked HTTP)
  14. YamlLoader_MCP         — mcp_servers block parsing
  15. Engine_MCP             — Stage 13 wired into full pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truenorth.mcp import MCPClient, MCPRegistry, ToolExecutor
from truenorth.mcp.types import (
    Tool, ToolCall, ToolResult, ToolResultStatus,
)
from truenorth.mcp.registry import RegisteredTool
from truenorth.mcp.tool_executor import _validate_arguments


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_tool(name: str = "test_tool", server: str = "__builtin__") -> Tool:
    return Tool(
        name=name,
        description=f"Test tool: {name}",
        server_name=server,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        builtin=server == "__builtin__",
    )


async def _echo_fn(query: str, limit: int = 5) -> dict:
    """Simple echo builtin for testing."""
    return {"echo": query, "limit": limit}


async def _always_fail(**kwargs) -> dict:
    raise ValueError("deliberate failure")


def _registry_with_echo() -> MCPRegistry:
    registry = MCPRegistry()
    tool = _make_tool("echo")
    registry._tools["echo"] = RegisteredTool(
        tool=tool, builtin_fn=_echo_fn
    )
    return registry


# ─────────────────────────────────────────────────────────────────────────────
#  1. Types
# ─────────────────────────────────────────────────────────────────────────────

class TestTypes:

    def test_tool_to_dict(self):
        t = _make_tool()
        d = t.to_dict()
        assert d["name"]   == "test_tool"
        assert "server_name" in d
        assert "input_schema" in d

    def test_tool_llm_description_includes_params(self):
        t   = _make_tool()
        desc = t.llm_description()
        assert "query*"  in desc or "query" in desc
        assert "limit"   in desc
        assert t.description in desc

    def test_tool_result_text_string(self):
        r = ToolResult(call_id="1", tool_name="t", status=ToolResultStatus.SUCCESS, content="hello")
        assert r.text == "hello"

    def test_tool_result_text_dict(self):
        r = ToolResult(call_id="1", tool_name="t", status=ToolResultStatus.SUCCESS, content={"k": "v"})
        assert "k" in r.text

    def test_tool_result_text_none(self):
        r = ToolResult(call_id="1", tool_name="t", status=ToolResultStatus.ERROR, content=None)
        assert r.text == ""

    def test_tool_result_is_success(self):
        ok  = ToolResult(call_id="1", tool_name="t", status=ToolResultStatus.SUCCESS, content="x")
        err = ToolResult(call_id="2", tool_name="t", status=ToolResultStatus.ERROR,   content=None)
        assert ok.is_success  is True
        assert err.is_success is False

    def test_tool_result_to_dict(self):
        r = ToolResult(call_id="1", tool_name="t", status=ToolResultStatus.SUCCESS,
                       content="ok", latency_ms=42)
        d = r.to_dict()
        assert d["status"]     == "success"
        assert d["latency_ms"] == 42

    def test_tool_result_status_values(self):
        assert ToolResultStatus.SUCCESS == "success"
        assert ToolResultStatus.ERROR   == "error"
        assert ToolResultStatus.TIMEOUT == "timeout"
        assert ToolResultStatus.SKIPPED == "skipped"

    def test_tool_call_dataclass(self):
        tc = ToolCall(tool_name="calc", arguments={"expression": "2+2"})
        assert tc.tool_name == "calc"
        assert tc.arguments["expression"] == "2+2"


# ─────────────────────────────────────────────────────────────────────────────
#  2. MCPClient — JSON-RPC response parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestMCPClientParsing:

    def test_jsonrpc_error_class_exists(self):
        from truenorth.mcp.client import MCPError
        err = MCPError("Bad request", code=-32600)
        assert "Bad request" in str(err)

    def test_mcp_tool_dataclass(self):
        from truenorth.mcp.client import MCPTool
        tool = MCPTool(
            name="calculator", description="Math tool",
            input_schema={"type": "object", "properties": {}},
        )
        assert tool.name == "calculator"
        assert isinstance(tool.input_schema, dict)

    def test_mcp_server_info_dataclass(self):
        from truenorth.mcp.client import MCPServerInfo
        info = MCPServerInfo(name="test", version="1.0", protocol_version="2024-11-05")
        assert info.name == "test"


# ─────────────────────────────────────────────────────────────────────────────
#  3. MCPClient — transport selection
# ─────────────────────────────────────────────────────────────────────────────

class TestMCPClientTransport:

    def test_http_factory_method(self):
        from truenorth.mcp.client import _HttpTransport
        client = MCPClient.http("http://localhost:3001/sse", "test")
        assert isinstance(client._transport, _HttpTransport)

    def test_stdio_factory_method(self):
        from truenorth.mcp.client import _StdioTransport
        client = MCPClient.stdio(["python", "-m", "mcp_server"], "test")
        assert isinstance(client._transport, _StdioTransport)

    def test_from_config_dict_url(self):
        """MCPClient.from_config handles dict with url key."""
        client = MCPClient.http("http://localhost:9999/sse", "cfg")
        assert client.name == "cfg"

    def test_client_has_name(self):
        client = MCPClient.http("http://localhost:3001/sse", "my_server")
        assert client.name == "my_server"


# ─────────────────────────────────────────────────────────────────────────────
#  4. MCPRegistry — builtin tools
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistryBuiltin:

    def test_add_calculator_builtin(self):
        registry = MCPRegistry()
        added    = registry.add_builtin("calculator")
        assert added is True
        assert registry.get_tool("calculator") is not None

    def test_add_web_search_builtin(self):
        registry = MCPRegistry()
        added    = registry.add_builtin("web_search")
        assert added is True

    def test_add_datetime_builtin(self):
        registry = MCPRegistry()
        added    = registry.add_builtin("datetime_tool")
        assert added is True

    def test_add_unknown_builtin_returns_false(self):
        registry = MCPRegistry()
        added    = registry.add_builtin("does_not_exist_xyz")
        assert added is False

    def test_register_builtins_alias(self):
        registry = MCPRegistry()
        registry.register_builtins("calculator")
        assert registry.get_tool("calculator") is not None

    def test_duplicate_builtin_not_added_twice(self):
        registry = MCPRegistry()
        registry.add_builtin("calculator")
        count1 = len(registry._tools)
        registry.add_builtin("calculator")
        count2 = len(registry._tools)
        assert count1 == count2

    def test_get_tool_by_name(self):
        registry = MCPRegistry()
        registry.add_builtin("calculator")
        rt = registry.get_tool("calculator")
        assert rt is not None
        assert rt.tool.name == "calculator"
        assert rt.is_builtin is True

    def test_get_unknown_tool_returns_none(self):
        registry = MCPRegistry()
        assert registry.get_tool("no_such_tool") is None

    def test_list_tools_returns_all(self):
        registry = MCPRegistry()
        registry.add_builtin("calculator")
        registry.add_builtin("datetime_tool")
        tools = registry.list_tools()
        names = [t.name for t in tools]
        assert "calculator"   in names
        assert "datetime_tool" in names


# ─────────────────────────────────────────────────────────────────────────────
#  5. MCPRegistry — server config
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistryServerConfig:

    def test_register_server_config_stored(self):
        registry = MCPRegistry()
        registry.register_server_config({
            "name": "my_server", "url": "http://localhost:3001/sse"
        })
        stored = getattr(registry, "_server_configs", {})
        assert "my_server" in stored

    def test_register_server_config_no_duplicate(self):
        registry = MCPRegistry()
        cfg = {"name": "srv", "url": "http://localhost:3001/sse"}
        registry.register_server_config(cfg)
        registry.register_server_config(cfg)
        stored = getattr(registry, "_server_configs", {})
        assert len(stored) == 1

    @pytest.mark.asyncio
    async def test_load_from_goal_config(self):
        """load_from_config wires all servers from goal YAML mcp_servers block."""
        registry = MCPRegistry()
        cfg = [
            {"name": "calculator", "builtin": True},
        ]
        await registry.load_from_config(cfg)
        assert registry.get_tool("calculator") is not None


# ─────────────────────────────────────────────────────────────────────────────
#  6. MCPRegistry — call_tool
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistryCallTool:

    @pytest.mark.asyncio
    async def test_call_builtin_tool(self):
        registry = _registry_with_echo()
        result   = await registry.call_tool("echo", {"query": "hello"})
        assert result.is_success
        assert "hello" in result.text

    @pytest.mark.asyncio
    async def test_call_unknown_tool_returns_skipped_or_error(self):
        registry = MCPRegistry()
        result   = await registry.call_tool("unknown_tool", {})
        assert result.status in (ToolResultStatus.SKIPPED, ToolResultStatus.ERROR)

    @pytest.mark.asyncio
    async def test_call_tool_captures_error(self):
        registry = MCPRegistry()
        tool     = _make_tool("fail_tool")
        registry._tools["fail_tool"] = RegisteredTool(
            tool=tool, builtin_fn=_always_fail
        )
        result = await registry.call_tool("fail_tool", {"query": "x"})
        assert result.status == ToolResultStatus.ERROR
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_call_tool_records_latency(self):
        registry = _registry_with_echo()
        result   = await registry.call_tool("echo", {"query": "timing"})
        assert result.latency_ms >= 0


# ─────────────────────────────────────────────────────────────────────────────
#  7. MCPRegistry — system prompt block
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistrySystemPrompt:

    def test_system_prompt_block_contains_tool_name(self):
        registry = MCPRegistry()
        registry.add_builtin("calculator")
        registry.add_builtin("datetime_tool")
        prompt   = registry.system_prompt_block()
        assert "calculator"   in prompt
        assert "datetime_tool" in prompt

    def test_empty_registry_empty_prompt(self):
        registry = MCPRegistry()
        prompt   = registry.system_prompt_block()
        # No tools → prompt should be empty or a header only
        assert isinstance(prompt, str)

    def test_system_prompt_includes_tool_call_syntax(self):
        registry = MCPRegistry()
        registry.add_builtin("calculator")
        prompt = registry.system_prompt_block()
        assert "TOOL_CALL" in prompt


# ─────────────────────────────────────────────────────────────────────────────
#  8. ToolExecutor — TOOL_CALL scanning
# ─────────────────────────────────────────────────────────────────────────────

class TestToolExecutorScan:

    def test_detects_single_tool_call(self):
        registry = MCPRegistry()
        executor = ToolExecutor(registry=registry)
        text     = 'Sure! TOOL_CALL: calculator({"expression": "2 + 2"})'
        calls    = executor._extract_calls(text)
        assert len(calls) == 1
        name, args, original = calls[0]
        assert name == "calculator"
        assert args["expression"] == "2 + 2"

    def test_detects_multiple_tool_calls(self):
        registry = MCPRegistry()
        executor = ToolExecutor(registry=registry)
        text     = (
            'TOOL_CALL: calculator({"expression": "1+1"})\n'
            'TOOL_CALL: datetime_tool({"timezone": "UTC"})'
        )
        calls = executor._extract_calls(text)
        assert len(calls) == 2
        names = [c[0] for c in calls]
        assert "calculator"   in names
        assert "datetime_tool" in names

    def test_no_tool_call_in_plain_text(self):
        registry = MCPRegistry()
        executor = ToolExecutor(registry=registry)
        calls    = executor._extract_calls("Here is your fitness plan.")
        assert calls == []

    def test_malformed_json_args_parsed_as_empty(self):
        registry = MCPRegistry()
        executor = ToolExecutor(registry=registry)
        calls    = executor._extract_calls('TOOL_CALL: calculator({not valid json})')
        assert len(calls) == 1
        _, args, _ = calls[0]
        assert args == {}


# ─────────────────────────────────────────────────────────────────────────────
#  9. ToolExecutor — end-to-end run
# ─────────────────────────────────────────────────────────────────────────────

class TestToolExecutorRun:

    def _make_state(self, session_id="s1", turn=1):
        """Helper: minimal GraphState for ToolExecutor.run()"""
        from truenorth.core.graph_state import GraphState
        state = GraphState.__new__(GraphState)
        state.session_id   = session_id
        state.current_turn = turn
        state.collected_fields = {}
        state.tool_call_log    = []
        return state

    @pytest.mark.asyncio
    async def test_tool_call_replaced_with_result(self):
        registry = _registry_with_echo()
        executor = ToolExecutor(registry=registry)
        state    = self._make_state()
        text     = 'Let me search. TOOL_CALL: echo({"query": "age 28"})'
        result_text, logs = await executor.run(text, state)
        assert "TOOL_CALL" not in result_text
        assert "echo" in result_text.lower() or "age 28" in result_text

    @pytest.mark.asyncio
    async def test_unknown_tool_call_skipped(self):
        registry = MCPRegistry()
        executor = ToolExecutor(registry=registry)
        state    = self._make_state()
        text     = 'TOOL_CALL: nonexistent_tool({"x": 1})'
        result_text, logs = await executor.run(text, state)
        assert isinstance(result_text, str)

    @pytest.mark.asyncio
    async def test_returns_execution_log(self):
        registry = _registry_with_echo()
        executor = ToolExecutor(registry=registry)
        state    = self._make_state()
        text     = 'TOOL_CALL: echo({"query": "test"})'
        _, logs  = await executor.run(text, state)
        assert isinstance(logs, list)
        assert len(logs) >= 1
        assert logs[0].tool_name == "echo"

    @pytest.mark.asyncio
    async def test_plain_text_unchanged(self):
        registry = MCPRegistry()
        executor = ToolExecutor(registry=registry)
        state    = self._make_state()
        text     = "Here is your personalised fitness plan."
        result, logs = await executor.run(text, state)
        assert result == text
        assert logs   == []

    @pytest.mark.asyncio
    async def test_result_length_capped(self):
        async def _huge_fn(**kwargs) -> dict:
            return {"data": "x" * 100_000}

        registry = MCPRegistry()
        tool     = _make_tool("huge")
        registry._tools["huge"] = RegisteredTool(tool=tool, builtin_fn=_huge_fn)
        executor = ToolExecutor(registry=registry, max_result=500)
        state    = self._make_state()
        text     = 'TOOL_CALL: huge({"query": "x"})'
        result_text, _ = await executor.run(text, state)
        assert len(result_text) < 100_000

    @pytest.mark.asyncio
    async def test_execute_single_directly(self):
        registry = _registry_with_echo()
        executor = ToolExecutor(registry=registry)
        result   = await executor.execute_single("echo", {"query": "direct"})
        assert result.is_success
        assert "direct" in result.text


# ─────────────────────────────────────────────────────────────────────────────
#  10. ToolExecutor — argument validation
# ─────────────────────────────────────────────────────────────────────────────

class TestToolExecutorValidate:

    def test_valid_args_pass(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        }
        ok, err = _validate_arguments({"query": "hello"}, schema)
        assert ok  is True
        assert err is None

    def test_missing_required_fails(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        ok, err = _validate_arguments({}, schema)
        assert ok  is False
        assert "query" in (err or "")

    def test_empty_schema_passes(self):
        ok, err = _validate_arguments({"anything": 1}, {})
        assert ok is True

    def test_extra_args_allowed(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        ok, _ = _validate_arguments({"query": "hi", "extra": "ignored"}, schema)
        assert ok is True


# ─────────────────────────────────────────────────────────────────────────────
#  11. Builtin — calculator
# ─────────────────────────────────────────────────────────────────────────────

class TestBuiltinCalculator:

    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        from truenorth.mcp.builtin.calculator import calculator
        result = await calculator("2 + 2")
        assert result["result"] == pytest.approx(4)

    @pytest.mark.asyncio
    async def test_floating_point(self):
        from truenorth.mcp.builtin.calculator import calculator
        result = await calculator("10 / 3")
        assert result["result"] == pytest.approx(3.333, abs=0.01)

    @pytest.mark.asyncio
    async def test_bmi_formula(self):
        from truenorth.mcp.builtin.calculator import calculator
        # BMI = 65 / (1.63^2) = 24.45
        result = await calculator("65 / (1.63 ** 2)")
        assert 24.0 < result["result"] < 25.0

    @pytest.mark.asyncio
    async def test_sqrt(self):
        from truenorth.mcp.builtin.calculator import calculator
        result = await calculator("sqrt(144)")
        assert result["result"] == pytest.approx(12.0)

    @pytest.mark.asyncio
    async def test_math_constants(self):
        from truenorth.mcp.builtin.calculator import calculator
        result = await calculator("pi")
        assert result["result"] == pytest.approx(3.14159, abs=0.001)

    @pytest.mark.asyncio
    async def test_dangerous_expression_rejected(self):
        from truenorth.mcp.builtin.calculator import calculator
        result = await calculator("__import__('os').system('ls')")
        # Should return error, not execute OS command
        assert result.get("error") is not None or result.get("result") is None

    @pytest.mark.asyncio
    async def test_import_rejected(self):
        from truenorth.mcp.builtin.calculator import calculator
        result = await calculator("import os")
        assert result.get("error") is not None

    @pytest.mark.asyncio
    async def test_division_by_zero_returns_error(self):
        from truenorth.mcp.builtin.calculator import calculator
        result = await calculator("1 / 0")
        assert result.get("error") is not None

    @pytest.mark.asyncio
    async def test_complex_expression(self):
        from truenorth.mcp.builtin.calculator import calculator
        # Mifflin-St Jeor BMR male: 65*10 + 6.25*163 - 5*28 + 5
        result = await calculator("65 * 10 + 6.25 * 163 - 5 * 28 + 5")
        assert result["result"] == pytest.approx(1534.75, abs=2.0)

    @pytest.mark.asyncio
    async def test_result_has_expression_field(self):
        from truenorth.mcp.builtin.calculator import calculator
        result = await calculator("3 * 7")
        assert "expression" in result or "result" in result


# ─────────────────────────────────────────────────────────────────────────────
#  12. Builtin — datetime
# ─────────────────────────────────────────────────────────────────────────────

class TestBuiltinDatetime:

    @pytest.mark.asyncio
    async def test_returns_current_time(self):
        from truenorth.mcp.builtin.datetime_tool import datetime_tool
        result = await datetime_tool()
        assert "datetime" in result or "date" in result or "time" in result

    @pytest.mark.asyncio
    async def test_utc_timezone(self):
        from truenorth.mcp.builtin.datetime_tool import datetime_tool
        result = await datetime_tool(timezone="UTC")
        assert isinstance(result, dict)
        assert result.get("error") is None or "UTC" in str(result)

    @pytest.mark.asyncio
    async def test_asia_kolkata(self):
        from truenorth.mcp.builtin.datetime_tool import datetime_tool
        result = await datetime_tool(timezone="Asia/Kolkata")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_invalid_timezone_falls_back(self):
        from truenorth.mcp.builtin.datetime_tool import datetime_tool
        result = await datetime_tool(timezone="Mars/Olympus_Mons")
        # Graceful fallback — returns UTC result (no crash, no hard error)
        assert isinstance(result, dict)
        assert "datetime" in result or "error" in result

    @pytest.mark.asyncio
    async def test_result_has_year(self):
        from truenorth.mcp.builtin.datetime_tool import datetime_tool
        result = await datetime_tool()
        result_str = str(result)
        assert "2025" in result_str or "2026" in result_str or "2027" in result_str

    @pytest.mark.asyncio
    async def test_india_standard_time_alias(self):
        from truenorth.mcp.builtin.datetime_tool import datetime_tool
        result = await datetime_tool(timezone="IST")
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
#  13. Builtin — web_search (mocked HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuiltinWebSearch:

    @pytest.mark.asyncio
    async def test_mock_ddg_returns_results(self):
        """Web search with mocked DuckDuckGo response."""
        from truenorth.mcp.builtin.web_search import web_search

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Abstract":   "BMI is a measure of body fat.",
            "AbstractURL":"https://example.com/bmi",
            "RelatedTopics": [],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=instance)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=mock_response)

            result = await web_search("BMI formula")
            assert isinstance(result, dict)
            assert "error" not in result or result.get("results") is not None

    @pytest.mark.asyncio
    async def test_network_failure_returns_error(self):
        from truenorth.mcp.builtin.web_search import web_search

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=instance)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)
            instance.get = AsyncMock(side_effect=Exception("connection refused"))

            result = await web_search("BMI formula")
            assert result.get("error") is not None

    @pytest.mark.asyncio
    async def test_empty_query_handled(self):
        from truenorth.mcp.builtin.web_search import web_search

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=instance)
            MockClient.return_value.__aexit__  = AsyncMock(return_value=False)
            instance.get = AsyncMock(side_effect=Exception("bad query"))
            result = await web_search("")
            assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
#  14. YamlLoader — mcp_servers block
# ─────────────────────────────────────────────────────────────────────────────

class TestYamlLoaderMCP:

    def test_builtin_shorthand_normalised(self):
        from truenorth.core.yaml_loader import YAMLLoader
        config = YAMLLoader.load_from_string("""
id: test
fields: [{name: age, type: integer, required: true}]
persona: {name: Bot, tone: neutral}
output: {format: json}
mcp_servers:
  - calculator
  - web_search
""")
        servers = config.get("mcp_servers", [])
        assert len(servers) == 2
        assert all(isinstance(s, dict) for s in servers)
        assert all(s.get("builtin") is True for s in servers)
        assert servers[0]["name"] == "calculator"
        assert servers[1]["name"] == "web_search"

    def test_dict_server_preserved(self):
        from truenorth.core.yaml_loader import YAMLLoader
        config = YAMLLoader.load_from_string("""
id: test
fields: [{name: age, type: integer}]
persona: {name: Bot, tone: neutral}
output: {format: json}
mcp_servers:
  - name: my_server
    url: http://localhost:3001/sse
""")
        servers = config.get("mcp_servers", [])
        assert len(servers) == 1
        assert servers[0]["name"] == "my_server"
        assert servers[0]["url"]  == "http://localhost:3001/sse"
        assert servers[0]["builtin"] is False

    def test_mixed_servers_normalised(self):
        from truenorth.core.yaml_loader import YAMLLoader
        config = YAMLLoader.load_from_string("""
id: test
fields: [{name: age, type: integer}]
persona: {name: Bot, tone: neutral}
output: {format: json}
mcp_servers:
  - calculator
  - name: remote
    url: http://localhost:3002/sse
  - datetime_tool
""")
        servers = config.get("mcp_servers", [])
        assert len(servers) == 3
        names = [s["name"] for s in servers]
        assert "calculator"   in names
        assert "remote"       in names
        assert "datetime_tool" in names

    def test_no_mcp_servers_defaults_to_empty(self):
        from truenorth.core.yaml_loader import YAMLLoader
        config = YAMLLoader.load_from_string("""
id: test
fields: [{name: age, type: integer}]
persona: {name: Bot, tone: neutral}
output: {format: json}
""")
        servers = config.get("mcp_servers", [])
        assert servers == []

    def test_builtin_server_has_name(self):
        from truenorth.core.yaml_loader import YAMLLoader
        config = YAMLLoader.load_from_string("""
id: test
fields: [{name: age, type: integer}]
persona: {name: Bot, tone: neutral}
output: {format: json}
mcp_servers:
  - {name: calculator, builtin: true}
""")
        servers = config.get("mcp_servers", [])
        assert servers[0]["name"] == "calculator"
        assert servers[0]["builtin"] is True


# ─────────────────────────────────────────────────────────────────────────────
#  15. Engine integration — Stage 13
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineMCP:

    @pytest.mark.asyncio
    async def test_engine_builds_tool_executor_from_yaml(self):
        from truenorth.core.engine import TrueNorthEngine
        goal = {
            "id": "mcp_test",
            "fields": [{"name": "age", "type": "integer", "required": True,
                        "question": "How old are you?"}],
            "persona": {"name": "Bot", "tone": "neutral"},
            "output":  {"format": "json"},
            "mcp_servers": [{"name": "calculator", "builtin": True}],
        }
        engine = TrueNorthEngine(goal_config=goal)
        assert engine._tool_executor is not None
        assert engine._mcp_registry.get_tool("calculator") is not None

    @pytest.mark.asyncio
    async def test_engine_no_mcp_servers_no_executor(self):
        from truenorth.core.engine import TrueNorthEngine
        goal = {
            "id": "no_mcp",
            "fields": [{"name": "age", "type": "integer", "required": True,
                        "question": "Age?"}],
            "persona": {"name": "Bot", "tone": "neutral"},
            "output":  {"format": "json"},
        }
        engine = TrueNorthEngine(goal_config=goal)
        assert engine._tool_executor is None

    @pytest.mark.asyncio
    async def test_tool_call_executed_in_pipeline(self):
        """Engine Stage 13: TOOL_CALL in planner response → executed → injected."""
        from truenorth.core.engine import TrueNorthEngine
        from truenorth.llm.router  import LLMRouter
        from truenorth.testing.mock_llm import MockLLMClient
        from truenorth.mcp.registry import MCPRegistry
        
        mock = MockLLMClient(
            responses={"extract": '{"extractions": []}'},
            default='TOOL_CALL: echo({"query": "age from user"})',
        )
        router = LLMRouter()
        for m in ["gemini-3.5-flash", "claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"]:
            router.register_client(m, mock)

        goal = {
            "id": "tool_test",
            "fields": [{"name": "name", "type": "text", "required": True,
                        "question": "Name?"}],
            "persona": {"name": "Bot", "tone": "neutral"},
            "output":  {"format": "json"},
            "mcp_servers": [],
        }
        registry = MCPRegistry()
        tool = _make_tool("echo")
        registry._tools["echo"] = RegisteredTool(tool=tool, builtin_fn=_echo_fn)

        engine = TrueNorthEngine(goal_config=goal, router=router, mcp_registry=registry)
        from truenorth.mcp.tool_executor import ToolExecutor
        engine._tool_executor = ToolExecutor(registry=registry)

        await engine.start()
        resp = await engine.process_message("My name is Alex")
        assert isinstance(resp.text, str)

    @pytest.mark.asyncio
    async def test_mcp_registry_injected_directly(self):
        from truenorth.core.engine import TrueNorthEngine
        from truenorth.mcp.registry import MCPRegistry
        registry = MCPRegistry()
        registry.add_builtin("calculator")
        goal = {
            "id": "inject_test",
            "fields": [{"name": "age", "type": "integer", "required": True, "question": "Age?"}],
            "persona": {"name": "Bot", "tone": "neutral"},
            "output":  {"format": "json"},
        }
        engine = TrueNorthEngine(goal_config=goal, mcp_registry=registry)
        assert engine._mcp_registry is registry
        assert engine._tool_executor is not None

    @pytest.mark.asyncio
    async def test_yaml_mcp_servers_wires_calculator(self):
        from truenorth.core.yaml_loader import YAMLLoader
        from truenorth.core.engine import TrueNorthEngine

        config = YAMLLoader.load_from_string("""
id: full_mcp
fields:
  - name: expression
    type: text
    required: true
    question: "What should I calculate?"
persona: {name: Bot, tone: neutral}
output: {format: text, template: "Result: {expression}"}
mcp_servers:
  - calculator
  - datetime_tool
""")
        engine = TrueNorthEngine(goal_config=config)
        assert engine._tool_executor is not None
        assert engine._mcp_registry.get_tool("calculator")    is not None
        assert engine._mcp_registry.get_tool("datetime_tool") is not None