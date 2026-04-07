"""MCP (Model Context Protocol) client for connecting to external tool servers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from codaicli.types import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# MCP config location (follows Claude Desktop convention)
MCP_CONFIG_PATH = Path.home() / ".codaicli" / "mcp_servers.json"


def _mcp_available() -> bool:
    """Check if the mcp package is installed."""
    try:
        import mcp  # noqa: F401

        return True
    except ImportError:
        return False


class MCPManager:
    """Manages connections to MCP servers and routes tool calls to them.

    If the `mcp` package is not installed, all operations gracefully no-op.
    Install with: pip install codaicli[mcp]
    """

    def __init__(self):
        self._servers: dict[str, Any] = {}  # name -> session
        self._tool_map: dict[str, str] = {}  # prefixed_tool_name -> server_name
        self._transports: dict[str, Any] = {}  # name -> (read, write) or context

    async def connect(self, name: str, config: dict):
        """Connect to an MCP server.

        Args:
            name: Server name (used for tool prefixing).
            config: Server configuration dict with either:
                - {"command": "...", "args": [...], "env": {...}} for stdio
                - {"url": "..."} for HTTP/SSE
        """
        if not _mcp_available():
            logger.warning("MCP package not installed. Skipping server: %s", name)
            return

        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        if "command" in config:
            # stdio transport
            params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env"),
            )
            transport = stdio_client(params)
            read, write = await transport.__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()

            self._servers[name] = session
            self._transports[name] = transport

        elif "url" in config:
            # HTTP/SSE transport
            from mcp.client.sse import sse_client

            transport = sse_client(url=config["url"])
            read, write = await transport.__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()

            self._servers[name] = session
            self._transports[name] = transport

        else:
            logger.warning("Unknown MCP server config for '%s': %s", name, config)
            return

        # Discover tools from this server
        await self._discover_server_tools(name)

    async def _discover_server_tools(self, server_name: str):
        """Discover and register tools from a connected server."""
        session = self._servers.get(server_name)
        if not session:
            return

        tools_response = await session.list_tools()
        for tool in tools_response.tools:
            prefixed_name = f"{server_name}__{tool.name}"
            self._tool_map[prefixed_name] = server_name

    async def discover_tools(self) -> list[ToolDefinition]:
        """Return ToolDefinitions for all tools from all connected servers."""
        definitions = []

        for server_name, session in self._servers.items():
            tools_response = await session.list_tools()
            for tool in tools_response.tools:
                prefixed_name = f"{server_name}__{tool.name}"
                definitions.append(
                    ToolDefinition(
                        name=prefixed_name,
                        description=f"[{server_name}] {tool.description or ''}",
                        parameters=tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
                        is_destructive=True,  # MCP tools default to destructive (safe side)
                    )
                )

        return definitions

    def is_mcp_tool(self, name: str) -> bool:
        """Check if a tool name belongs to an MCP server."""
        return name in self._tool_map

    def get_server_for_tool(self, tool_name: str) -> str | None:
        """Get the server name that owns a tool."""
        return self._tool_map.get(tool_name)

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        """Forward a tool call to the appropriate MCP server."""
        server_name = self._tool_map.get(tool_name)
        if not server_name:
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                content=f"Unknown MCP tool: {tool_name}",
                is_error=True,
            )

        session = self._servers.get(server_name)
        if not session:
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                content=f"MCP server '{server_name}' is not connected",
                is_error=True,
            )

        # Strip the server prefix to get the original tool name
        original_name = tool_name.removeprefix(f"{server_name}__")

        try:
            result = await session.call_tool(name=original_name, arguments=arguments)

            # Extract text content from MCP result
            content_parts = []
            is_error = getattr(result, "is_error", False) or False
            for block in result.content:
                if hasattr(block, "text"):
                    content_parts.append(block.text)
                else:
                    content_parts.append(str(block))

            return ToolResult(
                tool_call_id="",
                name=tool_name,
                content="\n".join(content_parts) if content_parts else "No output",
                is_error=is_error,
            )

        except Exception as e:
            return ToolResult(
                tool_call_id="",
                name=tool_name,
                content=f"MCP error: {e}",
                is_error=True,
            )

    async def disconnect_all(self):
        """Clean shutdown of all MCP connections."""
        for name in list(self._servers.keys()):
            try:
                session = self._servers.pop(name)
                await session.__aexit__(None, None, None)
                transport = self._transports.pop(name, None)
                if transport:
                    await transport.__aexit__(None, None, None)
            except Exception:
                pass  # Best-effort cleanup

        self._tool_map.clear()

    @staticmethod
    def load_config() -> dict:
        """Load MCP server configuration from ~/.codaicli/mcp_servers.json."""
        if not MCP_CONFIG_PATH.exists():
            return {}
        try:
            return json.loads(MCP_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
