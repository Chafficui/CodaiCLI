"""Thin wrapper around litellm for unified provider access."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import litellm

from codaicli.types import StreamEvent, ToolCall

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True
litellm.drop_params = True


class Provider:
    """Unified LLM provider using litellm.

    Supports 100+ providers via litellm model strings
    (e.g. "openai/gpt-4o", "anthropic/claude-sonnet-4-20250514", "ollama/llama3").
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _base_kwargs(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        """Build kwargs for litellm.acompletion."""
        # Prepend system message if provided
        all_messages = list(messages)
        if system_prompt:
            all_messages.insert(0, {"role": "system", "content": system_prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": all_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        return kwargs

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system_prompt: str = "",
    ) -> tuple[dict, list[ToolCall], str]:
        """Non-streaming completion.

        Returns (assistant_message_dict, tool_calls, stop_reason).
        """
        kwargs = self._base_kwargs(messages, tools, system_prompt)
        response = await litellm.acompletion(**kwargs)

        choice = response.choices[0]
        message = choice.message
        stop_reason = choice.finish_reason or "stop"

        # Parse tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        # Build assistant message dict (litellm/OpenAI format)
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": (
                            tc.function.arguments
                            if isinstance(tc.function.arguments, str)
                            else json.dumps(tc.function.arguments)
                        ),
                    },
                }
                for tc in message.tool_calls
            ]

        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                "output_tokens": getattr(response.usage, "completion_tokens", 0),
            }

        return assistant_msg, tool_calls, stop_reason, usage

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system_prompt: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion with tool support.

        Yields StreamEvents for text deltas and completed tool calls.
        Assembles incremental tool call argument chunks into complete ToolCalls.
        """
        kwargs = self._base_kwargs(messages, tools, system_prompt)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        response = await litellm.acompletion(**kwargs)

        # State for assembling tool calls
        pending_tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments_str}

        async for chunk in response:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                # Final chunk may contain usage
                if hasattr(chunk, "usage") and chunk.usage:
                    yield StreamEvent(
                        type="usage",
                        usage={
                            "input_tokens": getattr(
                                chunk.usage, "prompt_tokens", 0
                            ),
                            "output_tokens": getattr(
                                chunk.usage, "completion_tokens", 0
                            ),
                        },
                    )
                continue

            delta = choice.delta

            # Text content
            if delta and delta.content:
                yield StreamEvent(type="text", text=delta.content)

            # Tool call deltas (streamed incrementally)
            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index

                    if idx not in pending_tool_calls:
                        # New tool call starting
                        pending_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "name": (
                                tc_delta.function.name
                                if tc_delta.function
                                else ""
                            ),
                            "arguments_str": "",
                        }
                        if tc_delta.function and tc_delta.function.name:
                            yield StreamEvent(
                                type="tool_start",
                                text=tc_delta.function.name,
                            )

                    pending = pending_tool_calls[idx]

                    # Update ID if provided
                    if tc_delta.id:
                        pending["id"] = tc_delta.id
                    # Update name if provided
                    if tc_delta.function and tc_delta.function.name:
                        pending["name"] = tc_delta.function.name
                    # Accumulate arguments
                    if tc_delta.function and tc_delta.function.arguments:
                        pending["arguments_str"] += tc_delta.function.arguments

            # Check if stream ended (finish_reason present)
            if choice.finish_reason:
                # Emit all pending tool calls as complete
                for idx in sorted(pending_tool_calls.keys()):
                    pending = pending_tool_calls[idx]
                    try:
                        args = json.loads(pending["arguments_str"])
                    except (json.JSONDecodeError, ValueError):
                        args = {}

                    tool_call = ToolCall(
                        id=pending["id"],
                        name=pending["name"],
                        arguments=args,
                    )
                    yield StreamEvent(type="tool_done", tool_call=tool_call)

                pending_tool_calls.clear()

    def build_tool_result_messages(
        self, tool_calls: list[ToolCall], results: list[dict]
    ) -> list[dict]:
        """Build tool result messages in litellm/OpenAI format."""
        messages = []
        for tc, result in zip(tool_calls, results):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": result["content"],
                }
            )
        return messages
