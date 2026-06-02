"""
MCP protocol client.  Implements the Model Context Protocol (MCP) spec
over two transports:

  stdio   — spawn a local MCP server process, communicate over stdin/stdout
  http    — connect to a remote MCP server via HTTP + SSE

MCP uses JSON-RPC 2.0.  Key methods used by TrueNorth:
  initialize          — handshake, get server capabilities
  tools/list          — discover available tools
  tools/call          — execute a tool, get result
  resources/list      — (optional) list data resources
  prompts/list        — (optional) list server-provided prompts

Reference: https://spec.modelcontextprotocol.io/
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from truenorth.mcp.types import Tool, ToolCall, ToolResult, ToolResultStatus

logger = logging.getLogger(__name__)

# JSON-RPC 2.0 version
_JSONRPC = "2.0"

_CONNECT_TIMEOUT   = 10.0   # seconds
_CALL_TIMEOUT      = 30.0   # seconds for tool execution
_INIT_TIMEOUT      = 15.0   # seconds for handshake

_CLIENT_INFO = {
    "name":    "truenorth",
    "version": "0.1.1",
}
_CLIENT_CAPABILITIES = {
    "roots":   {"listChanged": False},
    "sampling": {},
}


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MCPTool:
    """Tool definition as returned by the MCP server."""
    name:         str
    description:  str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    server_name:  str = ""

    def to_tool(self) -> Tool:
        return Tool(
            name         = self.name,
            description  = self.description,
            server_name  = self.server_name,
            input_schema = self.input_schema,
        )


@dataclass
class MCPServerInfo:
    """Server info returned during initialize handshake."""
    name:            str
    version:         str
    protocol_version: str
    capabilities:    Dict[str, Any] = field(default_factory=dict)


class MCPError(Exception):
    """MCP protocol or transport error."""
    def __init__(self, message: str, code: int = -1):
        self.code = code
        super().__init__(f"MCP error {code}: {message}")


# ─────────────────────────────────────────────────────────────────────────────
#  Transport base
# ─────────────────────────────────────────────────────────────────────────────

class _Transport:
    """Abstract transport layer for MCP communication."""

    async def send(self, message: dict) -> None:
        raise NotImplementedError

    async def receive(self) -> dict:
        raise NotImplementedError

    async def connect(self) -> None:
        raise NotImplementedError

    async def disconnect(self) -> None:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
#  stdio transport
# ─────────────────────────────────────────────────────────────────────────────

class _StdioTransport(_Transport):
    """
    Spawns a local MCP server process and communicates over stdin/stdout.
    Used for: filesystem servers, database servers, local CLI tools.
    """

    def __init__(self, command: List[str], env: Optional[Dict[str, str]] = None):
        self._command = command
        self._env     = env
        self._process: Optional[asyncio.subprocess.Process] = None

    async def connect(self) -> None:
        import os
        proc_env = {**os.environ, **(self._env or {})}
        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            stdin  = asyncio.subprocess.PIPE,
            stdout = asyncio.subprocess.PIPE,
            stderr = asyncio.subprocess.PIPE,
            env    = proc_env,
        )
        logger.debug("mcp_stdio: spawned pid=%s cmd=%s", self._process.pid, self._command)

    async def disconnect(self) -> None:
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except Exception:
                self._process.kill()
            self._process = None

    async def send(self, message: dict) -> None:
        if not self._process or not self._process.stdin:
            raise MCPError("Not connected")
        data = json.dumps(message) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()

    async def receive(self) -> dict:
        if not self._process or not self._process.stdout:
            raise MCPError("Not connected")
        line = await asyncio.wait_for(
            self._process.stdout.readline(), timeout=_CALL_TIMEOUT
        )
        if not line:
            raise MCPError("Server closed connection")
        return json.loads(line.decode().strip())


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP + SSE transport
# ─────────────────────────────────────────────────────────────────────────────

class _HttpTransport(_Transport):
    """
    Connects to a remote MCP server via HTTP.
    Supports two modes:
      SSE endpoint  — server pushes messages via Server-Sent Events
      REST endpoint — plain HTTP POST for each request (simpler)
    """

    def __init__(
        self,
        url:     str,
        headers: Optional[Dict[str, str]] = None,
        mode:    str = "auto",   # "auto" | "sse" | "rest"
    ):
        self._url     = url.rstrip("/")
        self._headers = headers or {}
        self._mode    = mode
        self._client  = None
        self._pending: Dict[str, asyncio.Future] = {}
        self._sse_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        try:
            import httpx
            self._client = httpx.AsyncClient(
                headers = {
                    "Content-Type":     "application/json",
                    "Accept":           "application/json, text/event-stream",
                    "X-TrueNorth":      "mcp-client/0.1",
                    **self._headers,
                },
                timeout = httpx.Timeout(
                    connect = _CONNECT_TIMEOUT,
                    read    = _CALL_TIMEOUT,
                    write   = 10.0,
                    pool    = 5.0,
                ),
            )
        except ImportError:
            raise MCPError("httpx required for HTTP transport: pip install httpx")

        # Detect mode
        if self._mode == "auto":
            self._mode = await self._detect_mode()

        logger.debug("mcp_http: connected url=%s mode=%s", self._url, self._mode)

    async def disconnect(self) -> None:
        if self._sse_task:
            self._sse_task.cancel()
            self._sse_task = None
        if self._client:
            await self._client.aclose()
            self._client = None

    async def send(self, message: dict) -> None:
        """POST a JSON-RPC message. For SSE mode, also start listener if needed."""
        resp = await self._client.post(self._url, json=message)
        resp.raise_for_status()
        # For REST mode, parse and store the response for receive()
        if self._mode == "rest":
            msg_id = message.get("id")
            if msg_id and msg_id in self._pending:
                try:
                    data = resp.json()
                    self._pending[msg_id].set_result(data)
                except Exception as e:
                    self._pending[msg_id].set_exception(e)

    async def receive(self) -> dict:
        raise NotImplementedError("Use rpc() for HTTP transport")

    async def rpc(self, message: dict) -> dict:
        """
        Send a JSON-RPC message and await the response.
        Abstracts over REST and SSE modes.
        """
        resp = await self._client.post(self._url, json=message)
        resp.raise_for_status()
        return resp.json()

    async def _detect_mode(self) -> str:
        """Probe the endpoint to detect SSE vs REST."""
        try:
            resp = await self._client.get(
                f"{self._url}/info", timeout=_CONNECT_TIMEOUT
            )
            if resp.status_code == 200:
                caps = resp.json().get("capabilities", {})
                if "streaming" in caps or "sse" in caps:
                    return "sse"
        except Exception:
            pass
        return "rest"


# ─────────────────────────────────────────────────────────────────────────────
#  MCPClient — main interface
# ─────────────────────────────────────────────────────────────────────────────

class MCPClient:
    """
    MCP protocol client.  Manages one connection to one MCP server.

    Use the factory methods:
      MCPClient.http(url, name)             — HTTP/SSE transport
      MCPClient.stdio(command, name)        — stdio transport
      MCPClient.from_config(server_config)  — from YAML config dict

    Then either use as an async context manager, or call connect()/disconnect() manually.
    """

    def __init__(
        self,
        transport:   _Transport,
        name:        str = "unnamed",
        config:      Optional[dict] = None,
    ):
        self._transport  = transport
        self.name        = name
        self._config     = config or {}
        self._server_info: Optional[MCPServerInfo] = None
        self._tools:      List[MCPTool] = []
        self._connected:  bool = False
        self._req_counter = 0

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def http(
        cls,
        url:     str,
        name:    str = "",
        headers: Optional[Dict[str, str]] = None,
        auth:    Optional[str] = None,
    ) -> "MCPClient":
        """Create an HTTP transport client."""
        hdrs = dict(headers or {})
        if auth:
            hdrs["Authorization"] = f"Bearer {auth}"
        transport = _HttpTransport(url, headers=hdrs)
        return cls(transport=transport, name=name or _name_from_url(url))

    @classmethod
    def stdio(
        cls,
        command: List[str],
        name:    str = "",
        env:     Optional[Dict[str, str]] = None,
    ) -> "MCPClient":
        """Create a stdio transport client (spawns local MCP server process)."""
        transport = _StdioTransport(command, env=env)
        return cls(transport=transport, name=name or command[0])

    @classmethod
    def from_config(cls, server_config: dict) -> "MCPClient":
        """
        Create from a YAML goal `mcp_servers:` entry.

        Config formats:
          {name: "fs", command: ["npx", "-y", "@mcp/server-filesystem", "/tmp"]}
          {name: "search", url: "http://localhost:3001/sse"}
          {name: "db", url: "http://...", auth: "my-token"}
        """
        name = server_config.get("name", "")

        if "command" in server_config:
            cmd = server_config["command"]
            if isinstance(cmd, str):
                cmd = cmd.split()
            return cls.stdio(
                command = cmd,
                name    = name,
                env     = server_config.get("env"),
            )

        if "url" in server_config:
            return cls.http(
                url     = server_config["url"],
                name    = name,
                headers = server_config.get("headers"),
                auth    = server_config.get("auth"),
            )

        raise MCPError(f"MCP server config for '{name}' needs 'url' or 'command'")

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> MCPServerInfo:
        """Connect and perform MCP initialize handshake."""
        await self._transport.connect()
        info = await self._initialize()
        self._connected = True
        logger.info(
            "mcp: connected server=%s version=%s protocol=%s tools_supported=%s",
            self.name,
            info.version,
            info.protocol_version,
            bool(info.capabilities.get("tools")),
        )
        return info

    async def disconnect(self) -> None:
        """Clean shutdown."""
        if self._connected:
            try:
                await self._rpc("shutdown", {})
            except Exception:
                pass
        await self._transport.disconnect()
        self._connected = False
        logger.debug("mcp: disconnected server=%s", self.name)

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Tool operations
    # ------------------------------------------------------------------

    async def list_tools(self) -> List[MCPTool]:
        """
        List all tools available on this server.
        Caches results; call refresh_tools() to re-fetch.
        """
        if self._tools:
            return self._tools
        return await self.refresh_tools()

    async def refresh_tools(self) -> List[MCPTool]:
        """Force-refresh the tool list from the server."""
        resp  = await self._rpc("tools/list", {})
        tools_raw = resp.get("tools", [])
        self._tools = [
            MCPTool(
                name         = t["name"],
                description  = t.get("description", ""),
                input_schema = t.get("inputSchema", {}),
                server_name  = self.name,
            )
            for t in tools_raw
        ]
        logger.info(
            "mcp: server=%s discovered %d tools: %s",
            self.name,
            len(self._tools),
            [t.name for t in self._tools],
        )
        return self._tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout:   float = _CALL_TIMEOUT,
    ) -> ToolResult:
        """
        Execute a tool by name with the given arguments.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool input parameters (validated against input_schema).
            timeout:   Seconds to wait for the tool to complete.

        Returns:
            ToolResult with status and content.
        """
        call_id = str(uuid.uuid4())[:8]
        t0      = time.perf_counter()

        logger.info(
            "mcp: calling tool=%s server=%s args=%s",
            tool_name, self.name, list(arguments.keys()),
        )

        try:
            resp = await asyncio.wait_for(
                self._rpc(
                    "tools/call",
                    {"name": tool_name, "arguments": arguments},
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            logger.warning(
                "mcp: tool=%s timed out after %.1fs", tool_name, timeout
            )
            return ToolResult(
                call_id    = call_id,
                tool_name  = tool_name,
                status     = ToolResultStatus.TIMEOUT,
                content    = None,
                error      = f"Tool timed out after {timeout:.1f}s",
                latency_ms = latency_ms,
            )
        except Exception as e:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            logger.error("mcp: tool=%s error: %s", tool_name, e)
            return ToolResult(
                call_id    = call_id,
                tool_name  = tool_name,
                status     = ToolResultStatus.ERROR,
                content    = None,
                error      = str(e)[:300],
                latency_ms = latency_ms,
            )

        latency_ms = int((time.perf_counter() - t0) * 1000)

        # Parse MCP tool result format
        content   = self._extract_content(resp)
        is_error  = resp.get("isError", False)

        logger.info(
            "mcp: tool=%s latency=%dms is_error=%s",
            tool_name, latency_ms, is_error,
        )

        return ToolResult(
            call_id    = call_id,
            tool_name  = tool_name,
            status     = ToolResultStatus.ERROR if is_error else ToolResultStatus.SUCCESS,
            content    = content,
            error      = content if is_error and isinstance(content, str) else None,
            latency_ms = latency_ms,
        )

    # ------------------------------------------------------------------
    # Resource and prompt operations (optional MCP features)
    # ------------------------------------------------------------------

    async def list_resources(self) -> List[dict]:
        """List data resources if the server supports them."""
        try:
            resp = await self._rpc("resources/list", {})
            return resp.get("resources", [])
        except MCPError:
            return []

    async def read_resource(self, uri: str) -> Optional[str]:
        """Read a specific resource by URI."""
        try:
            resp = await self._rpc("resources/read", {"uri": uri})
            contents = resp.get("contents", [])
            if contents:
                return contents[0].get("text", "")
            return None
        except MCPError as e:
            logger.warning("mcp: read_resource failed uri=%s: %s", uri, e)
            return None

    async def list_prompts(self) -> List[dict]:
        """List server-provided prompts."""
        try:
            resp = await self._rpc("prompts/list", {})
            return resp.get("prompts", [])
        except MCPError:
            return []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def server_info(self) -> Optional[MCPServerInfo]:
        return self._server_info

    @property
    def tool_names(self) -> List[str]:
        return [t.name for t in self._tools]

    # ------------------------------------------------------------------
    # Internal — JSON-RPC 2.0
    # ------------------------------------------------------------------

    async def _initialize(self) -> MCPServerInfo:
        """Perform MCP initialize handshake."""
        resp = await self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "clientInfo":      _CLIENT_INFO,
                "capabilities":    _CLIENT_CAPABILITIES,
            },
            timeout=_INIT_TIMEOUT,
        )
        info = MCPServerInfo(
            name             = resp.get("serverInfo", {}).get("name", self.name),
            version          = resp.get("serverInfo", {}).get("version", "?"),
            protocol_version = resp.get("protocolVersion", "?"),
            capabilities     = resp.get("capabilities", {}),
        )
        self._server_info = info

        # Send initialized notification
        await self._notify("notifications/initialized")
        return info

    async def _rpc(
        self,
        method:  str,
        params:  dict,
        timeout: float = _CALL_TIMEOUT,
    ) -> dict:
        """
        Send a JSON-RPC 2.0 request and return the result.
        Raises MCPError on JSON-RPC error responses.
        """
        self._req_counter += 1
        req_id  = self._req_counter
        message = {
            "jsonrpc": _JSONRPC,
            "id":      req_id,
            "method":  method,
            "params":  params,
        }

        # HTTP transport: use rpc() which does POST + parse
        if isinstance(self._transport, _HttpTransport):
            resp = await asyncio.wait_for(
                self._transport.rpc(message), timeout=timeout
            )
        else:
            await self._transport.send(message)
            while True:
                resp = await asyncio.wait_for(
                    self._transport.receive(), timeout=timeout
                )
                if resp.get("id") == req_id:
                    break   
        if "error" in resp:
            err = resp["error"]
            raise MCPError(
                err.get("message", "Unknown error"),
                err.get("code", -1),
            )

        return resp.get("result", {})

    async def _notify(self, method: str, params: dict = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        message = {
            "jsonrpc": _JSONRPC,
            "method":  method,
            "params":  params or {},
        }
        try:
            if isinstance(self._transport, _HttpTransport):
                await self._transport._client.post(
                    self._transport._url, json=message
                )
            else:
                await self._transport.send(message)
        except Exception as e:
            logger.debug("mcp: notification failed method=%s: %s", method, e)

    @staticmethod
    def _extract_content(resp: dict) -> Any:
        """
        Extract content from MCP tool result.
        MCP returns: {"content": [{"type": "text", "text": "..."}, ...]}
        """
        content_list = resp.get("content", [])
        if not content_list:
            return resp.get("result", "")

        if len(content_list) == 1 and content_list[0].get("type") == "text":
            return content_list[0].get("text", "")
        return content_list


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _name_from_url(url: str) -> str:
    """Extract a short server name from a URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.split(":")[0] or parsed.path.strip("/").split("/")[0]
    except Exception:
        return "mcp-server"