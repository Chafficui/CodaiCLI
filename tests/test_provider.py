"""Tests for codaicli.provider."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codaicli.provider import Provider


class TestProviderInit:
    def test_defaults(self):
        p = Provider(model="openai/gpt-4o")
        assert p.model == "openai/gpt-4o"
        assert p.temperature == 0.2
        assert p.max_tokens == 4096
        assert p.api_key is None
        assert p.api_base is None

    def test_custom(self):
        p = Provider(
            model="ollama/llama3",
            api_key="test-key",
            api_base="http://localhost:11434",
            temperature=0.5,
            max_tokens=2048,
        )
        assert p.api_key == "test-key"
        assert p.api_base == "http://localhost:11434"


class TestBaseKwargs:
    def test_basic(self):
        p = Provider(model="openai/gpt-4o")
        kwargs = p._base_kwargs(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="You are helpful.",
        )
        assert kwargs["model"] == "openai/gpt-4o"
        assert kwargs["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert kwargs["messages"][1] == {"role": "user", "content": "hi"}
        assert "tools" not in kwargs

    def test_with_tools(self):
        p = Provider(model="test/model")
        tools = [{"type": "function", "function": {"name": "test"}}]
        kwargs = p._base_kwargs(
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"

    def test_with_api_key(self):
        p = Provider(model="test/model", api_key="sk-test")
        kwargs = p._base_kwargs(messages=[])
        assert kwargs["api_key"] == "sk-test"


class TestComplete:
    @pytest.mark.asyncio
    async def test_text_response(self):
        p = Provider(model="test/model")

        mock_message = MagicMock()
        mock_message.content = "Hello world"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with patch("codaicli.provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            msg, tool_calls, stop_reason, usage = await p.complete(
                messages=[{"role": "user", "content": "hi"}]
            )

        assert msg["content"] == "Hello world"
        assert tool_calls == []
        assert stop_reason == "stop"
        assert usage["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_tool_call_response(self):
        p = Provider(model="test/model")

        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "read_file"
        mock_tc.function.arguments = '{"path": "foo.py"}'

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        with patch("codaicli.provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            msg, tool_calls, stop_reason, usage = await p.complete(
                messages=[{"role": "user", "content": "read foo"}],
                tools=[{"type": "function", "function": {"name": "read_file"}}],
            )

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "read_file"
        assert tool_calls[0].arguments == {"path": "foo.py"}
        assert stop_reason == "tool_calls"


class TestBuildToolResultMessages:
    def test_basic(self):
        from codaicli.types import ToolCall

        p = Provider(model="test")
        tcs = [ToolCall(id="1", name="read_file", arguments={"path": "a.py"})]
        results = [{"content": "file content"}]
        msgs = p.build_tool_result_messages(tcs, results)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "1"
        assert msgs[0]["content"] == "file content"
