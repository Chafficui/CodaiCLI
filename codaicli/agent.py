"""Async agentic loop for CodaiCLI."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from codaicli.mcp_client import MCPManager
from codaicli.provider import Provider
from codaicli.tools import ToolRegistry
from codaicli.types import StreamEvent, ToolCall, ToolResult


class Agent:
    """Drives a multi-turn tool-use conversation loop.

    The agent sends messages to the LLM, handles tool calls (built-in and MCP),
    feeds results back, and repeats until the LLM produces a final text response.
    UI integration is via callbacks to keep the agent decoupled from Rich/terminal.
    """

    SYSTEM_PROMPT = (
        "You are CodaiCLI, an AI coding assistant running in a terminal. "
        "You help developers understand, modify, and manage their code projects. "
        "You have tools to read, write, edit, search, and list files in the project, "
        "as well as run shell commands. Always explore the codebase with tools before "
        "making changes. Explain your reasoning concisely."
    )

    def __init__(
        self,
        provider: Provider,
        tool_registry: ToolRegistry,
        mcp_manager: MCPManager | None = None,
        on_text: Callable[[str], None] | None = None,
        on_tool_call: Callable[[ToolCall], None] | None = None,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        confirm_tool: Callable[[ToolCall], Awaitable[bool]] | None = None,
        max_iterations: int = 25,
    ):
        self.provider = provider
        self.tools = tool_registry
        self.mcp = mcp_manager
        self.history: list[dict[str, Any]] = []
        self.max_iterations = max_iterations

        # Callbacks
        self._on_text = on_text or (lambda t: None)
        self._on_tool_call = on_tool_call or (lambda tc: None)
        self._on_tool_result = on_tool_result or (lambda tr: None)
        self._confirm_tool = confirm_tool

    async def run(self, user_message: str) -> str:
        """Process a user message through the agentic loop.

        Streams text to on_text() in real-time. Handles tool calls with
        confirmation for destructive tools. Returns the final text response.
        """
        self.history.append({"role": "user", "content": user_message})

        # Merge built-in + MCP tools
        all_tools = self.tools.get_litellm_tools()
        if self.mcp:
            mcp_defs = await self.mcp.discover_tools()
            for defn in mcp_defs:
                all_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": defn.name,
                            "description": defn.description,
                            "parameters": defn.parameters,
                        },
                    }
                )

        final_text = ""

        for _iteration in range(self.max_iterations):
            # Stream the response
            assistant_msg, tool_calls, text = await self._stream_response(all_tools)

            if text:
                final_text = text

            # Add assistant message to history
            self.history.append(assistant_msg)

            # No tool calls means we're done
            if not tool_calls:
                break

            # Execute tool calls and feed results back
            tool_result_messages = await self._handle_tool_calls(tool_calls)
            self.history.extend(tool_result_messages)

        else:
            final_text += "\n\n[Reached maximum iteration limit]"

        return final_text

    async def _stream_response(
        self, tools: list[dict]
    ) -> tuple[dict, list[ToolCall], str]:
        """Stream a provider response, forwarding text to callback.

        Returns (assistant_message_dict, tool_calls, accumulated_text).
        """
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage = None

        async for event in self.provider.stream(
            messages=self.history,
            tools=tools,
            system_prompt=self.SYSTEM_PROMPT,
        ):
            if event.type == "text" and event.text:
                text_parts.append(event.text)
                self._on_text(event.text)
            elif event.type == "tool_start" and event.text:
                # Tool name announced — could show a spinner
                pass
            elif event.type == "tool_done" and event.tool_call:
                tool_calls.append(event.tool_call)
            elif event.type == "usage" and event.usage:
                usage = event.usage

        text = "".join(text_parts)

        # Build assistant message dict
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ]

        return assistant_msg, tool_calls, text

    async def _handle_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[dict]:
        """Execute tool calls with confirmation and return result messages."""
        result_messages = []

        for tc in tool_calls:
            self._on_tool_call(tc)

            # Check if destructive and needs confirmation
            is_mcp = self.mcp and self.mcp.is_mcp_tool(tc.name)
            needs_confirm = (
                self.tools.is_destructive(tc.name) if not is_mcp else True
            )

            if needs_confirm and self._confirm_tool:
                approved = await self._confirm_tool(tc)
                if not approved:
                    result = ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content="User denied this action.",
                        is_error=True,
                    )
                    self._on_tool_result(result)
                    result_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": result.content,
                        }
                    )
                    continue

            # Execute the tool
            if is_mcp:
                result = await self.mcp.call_tool(tc.name, tc.arguments)
            else:
                result = await self.tools.execute(tc.name, tc.arguments)

            result.tool_call_id = tc.id
            self._on_tool_result(result)

            result_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": result.content,
                }
            )

        return result_messages

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()

    def estimate_tokens(self) -> int:
        """Rough estimate of tokens in current history (~4 chars per token)."""
        total_chars = 0
        for msg in self.history:
            total_chars += len(msg.get("content", "") or "")
            for tc in msg.get("tool_calls", []):
                total_chars += len(json.dumps(tc.get("function", {}).get("arguments", "")))
        return total_chars // 4

    def trim_history(self, max_tokens: int = 100_000):
        """Trim old messages if history exceeds token estimate.

        Removes oldest message groups, never orphaning tool results
        from their assistant messages.
        """
        while self.estimate_tokens() > max_tokens and len(self.history) > 2:
            # Remove from the front, but skip if it would orphan tool results
            removed = self.history.pop(0)
            # If we removed an assistant message with tool_calls,
            # also remove the following tool result messages
            if removed.get("role") == "assistant" and removed.get("tool_calls"):
                while self.history and self.history[0].get("role") == "tool":
                    self.history.pop(0)
            # If we hit a tool message at the front, remove it too
            elif removed.get("role") == "tool":
                pass  # Already removed
