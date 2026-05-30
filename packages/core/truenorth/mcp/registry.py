"""
MCPRegistry — manages multiple MCP server connections for one session.

Responsibilities:
  - Connect/disconnect all configured MCP servers on session start/end
  - Merge tool lists from all servers into one unified namespace
  - Resolve tool names to the correct server
  - Inject tool descriptions into LLM system prompts
  - Cache tool schemas for validation

Goal YAML usage:
    mcp_servers:
      - name: calculator
        builtin: true

      - name: web_search
        builtin: true

      - name: my_db
        url: http://localhost:3001/sse
        auth: ${MY_DB_TOKEN}

      - name: filesystem
        command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/data"]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from truenorth.mcp.client import MCPClient
from truenorth.mcp.types  import Tool, ToolCall, ToolResult, ToolResultStatus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  RegisteredTool — one tool in the registry
# ─────────────────────────────────────────────────────────────────────────────

class RegisteredTool:
    """A tool known to the registry — wraps an MCP tool or a builtin."""

    def __init__(
        self,
        tool:       Tool,
        client:     Optional[MCPClient] = None,
        builtin_fn: Optional[Any]       = None,
    ):
        self.tool      = tool
        self._client   = client
        self._builtin  = builtin_fn

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def server(self) -> str:
        return self.tool.server_name

    @property
    def is_builtin(self) -> bool:
        return self.tool.builtin

    def description_for_llm(self) -> str:
        return self.tool.llm_description()


# ─────────────────────────────────────────────────────────────────────────────
#  MCPRegistry
# ─────────────────────────────────────────────────────────────────────────────

class MCPRegistry:
    """
    Manages all MCP server connections for one TrueNorth session.

    Usage:
        registry = MCPRegistry()

        # Load from YAML goal config
        await registry.load_from_config(goal_config["mcp_servers"])

        # Or add servers manually
        registry.add_builtin("web_search")
        registry.add_builtin("calculator")
        registry.add_server(MCPClient.http("http://localhost:3001/sse", name="db"))

        # Connect all servers
        await registry.connect_all()

        # Get tools for LLM
        tools     = registry.list_tools()
        prompt    = registry.system_prompt_block()

        # Execute a tool
        result    = await registry.call_tool("search", {"query": "..."})

        # Shutdown
        await registry.disconnect_all()
    """

    def __init__(self):
        self._clients:  List[MCPClient] = []
        self._tools:          Dict[str, RegisteredTool] = {}  # tool_name → RegisteredTool
        self._server_configs: Dict[str, dict]          = {}  # name → server cfg
        self._connected = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    async def load_from_config(self, servers_config: List[dict]) -> None:
        """
        Load MCP servers from a goal YAML `mcp_servers:` list.
        Connects all non-builtin servers.
        """
        for server_cfg in servers_config:
            name     = server_cfg.get("name", "")
            builtin  = server_cfg.get("builtin", False)

            if builtin:
                self.add_builtin(name)
                logger.info("registry: loaded builtin tool=%s", name)
            else:
                try:
                    client = MCPClient.from_config(server_cfg)
                    self._clients.append(client)
                    logger.info("registry: queued server=%s", name)
                except Exception as e:
                    logger.error("registry: failed to create client for '%s': %s", name, e)

        await self.connect_all()

    def add_server(self, client: MCPClient) -> None:
        """Add an MCP server client. Call connect_all() afterwards."""
        self._clients.append(client)

    def add_builtin(self, tool_name: str) -> bool:
        """
        Register one of TrueNorth's built-in tools by name.
        Built-ins don't need an external server.

        Available built-ins:
          web_search    — search the web
          calculator    — evaluate mathematical expressions
          datetime_tool — current date/time in any timezone
        """
        from truenorth.mcp.builtin import get_builtin
        fn = get_builtin(tool_name)
        if fn is None:
            logger.warning("registry: unknown builtin tool '%s'", tool_name)
            return False

        tool = Tool(
            name         = tool_name,
            description  = fn.__doc__ or tool_name,
            server_name  = "__builtin__",
            input_schema = _builtin_schema(tool_name),
            builtin      = True,
        )
        self._tools[tool_name] = RegisteredTool(tool=tool, builtin_fn=fn)
        logger.debug("registry: registered builtin=%s", tool_name)
        return True

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect_all(self) -> Dict[str, bool]:
        """
        Connect all registered (non-builtin) MCP servers.
        Returns {server_name: success}.
        """
        results: Dict[str, bool] = {}
        for client in self._clients:
            try:
                await client.connect()
                tools = await client.list_tools()
                for mcp_tool in tools:
                    tool = mcp_tool.to_tool()
                    self._tools[tool.name] = RegisteredTool(tool=tool, client=client)
                results[client.name] = True
                logger.info(
                    "registry: server=%s connected with %d tools",
                    client.name, len(tools),
                )
            except Exception as e:
                results[client.name] = False
                logger.error(
                    "registry: server=%s failed to connect: %s", client.name, e
                )
        self._connected = True
        return results

    async def disconnect_all(self) -> None:
        """Disconnect all MCP servers."""
        for client in self._clients:
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("registry: disconnect error for %s: %s", client.name, e)
        self._connected = False

    async def __aenter__(self) -> "MCPRegistry":
        await self.connect_all()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect_all()

    # ------------------------------------------------------------------
    # Tool access
    # ------------------------------------------------------------------

    def list_tools(self) -> List[Tool]:
        """Return all registered tools (builtin + external)."""
        return [rt.tool for rt in self._tools.values()]

    # ------------------------------------------------------------------
    # Convenience aliases for engine integration
    # ------------------------------------------------------------------

    def register_builtins(self, server_name: str = "builtin") -> None:
        """Register a specific builtin tool by its name."""
        self.add_builtin(server_name)

    def register_server_config(self, server_cfg: dict) -> None:
        """Register a remote MCP server config for lazy connection."""
        name = server_cfg.get("name", "unnamed")
        if name not in self._server_configs:
            self._server_configs[name] = server_cfg
            logger.info("registry: registered server config name=%s url=%s",
                        name, server_cfg.get("url", ""))

    def get_tool(self, name: str) -> Optional[RegisteredTool]:
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def builtin_names(self) -> List[str]:
        return [n for n, rt in self._tools.items() if rt.is_builtin]

    def external_names(self) -> List[str]:
        return [n for n, rt in self._tools.items() if not rt.is_builtin]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: str = "",
        turn:       int = 0,
        timeout:    float = 30.0,
    ) -> ToolResult:
        """
        Execute a tool by name.
        Routes to builtin handler or MCP server as appropriate.
        """
        rt = self._tools.get(tool_name)
        if rt is None:
            logger.warning("registry: tool not found: %s", tool_name)
            return ToolResult(
                call_id   = "",
                tool_name = tool_name,
                status    = ToolResultStatus.ERROR,
                content   = None,
                error     = f"Tool '{tool_name}' not registered",
            )

        if rt.is_builtin:
            return await self._call_builtin(rt, tool_name, arguments, session_id, turn)

        if rt._client is None:
            return ToolResult(
                call_id   = "",
                tool_name = tool_name,
                status    = ToolResultStatus.ERROR,
                content   = None,
                error     = f"No MCP client for tool '{tool_name}'",
            )

        return await rt._client.call_tool(tool_name, arguments, timeout=timeout)

    async def _call_builtin(
        self,
        rt:         RegisteredTool,
        tool_name:  str,
        arguments:  Dict[str, Any],
        session_id: str,
        turn:       int,
    ) -> ToolResult:
        """Execute a built-in tool function."""
        import time, uuid
        call_id = str(uuid.uuid4())[:8]
        t0      = time.perf_counter()
        try:
            result = await rt._builtin(**arguments)
            return ToolResult(
                call_id    = call_id,
                tool_name  = tool_name,
                status     = ToolResultStatus.SUCCESS,
                content    = result,
                latency_ms = int((time.perf_counter() - t0) * 1000),
            )
        except Exception as e:
            logger.error("registry: builtin=%s error: %s", tool_name, e)
            return ToolResult(
                call_id    = call_id,
                tool_name  = tool_name,
                status     = ToolResultStatus.ERROR,
                content    = None,
                error      = str(e)[:300],
                latency_ms = int((time.perf_counter() - t0) * 1000),
            )

    # ------------------------------------------------------------------
    # LLM prompt integration
    # ------------------------------------------------------------------

    def system_prompt_block(self) -> str:
        """
        Generate a system prompt block describing all available tools.
        Injected into the LLM's system prompt so it knows what tools to call.

        Format:
          You have access to the following tools. To call a tool, respond with:
          TOOL_CALL: tool_name({"param": "value"})
          <tool descriptions>
        """
        if not self._tools:
            return ""

        lines = [
            "## Available Tools",
            "",
            "You have access to the following tools. When a user's request requires",
            "real-world data or computation, call the appropriate tool using:",
            "  TOOL_CALL: tool_name({\"param\": \"value\"})",
            "",
        ]
        for rt in self._tools.values():
            lines.append(f"- {rt.description_for_llm()}")

        return "\n".join(lines)

    def inject_tool_result(self, message: str, result: ToolResult) -> str:
        """
        Append a tool result to a message so the LLM can see the output.
        Used after tool execution — the result becomes part of the conversation context.
        """
        if not result.is_success:
            return f"{message}\n\n[Tool '{result.tool_name}' failed: {result.error}]"
        return f"{message}\n\n[Tool '{result.tool_name}' result: {result.text}]"

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "tool_count":    len(self._tools),
            "builtin_count": len(self.builtin_names()),
            "server_count":  len(self._clients),
            "tools":         [t.to_dict() for t in self.list_tools()],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Built-in tool schema lookup
# ─────────────────────────────────────────────────────────────────────────────

def _builtin_schema(tool_name: str) -> Dict[str, Any]:
    schemas = {
        "web_search": {
            "type":     "object",
            "properties": {
                "query":    {"type": "string",  "description": "Search query"},
                "limit":    {"type": "integer", "description": "Max results", "default": 5},
            },
            "required": ["query"],
        },
        "calculator": {
            "type":     "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"},
            },
            "required": ["expression"],
        },
        "datetime_tool": {
            "type":     "object",
            "properties": {
                "timezone": {"type": "string", "description": "Timezone name (e.g. Asia/Kolkata)", "default": "UTC"},
                "format":   {"type": "string", "description": "strftime format", "default": "%Y-%m-%d %H:%M:%S"},
            },
            "required": [],
        },
    }
    return schemas.get(tool_name, {"type": "object", "properties": {}})