"""Tests for codaicli.agent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from codaicli.agent import Agent
from codaicli.types import StreamEvent, ToolCall, ToolResult


class MockProvider:
    """Mock provider that returns scripted responses."""

    def __init__(self, responses: list):
        """responses: list of (text, tool_calls) tuples."""
        self._responses = list(responses)
        self._call_count = 0
        self.model = "test/mock"

    async def stream(self, messages, tools, system_prompt=""):
        """Yield scripted StreamEvents."""
        if self._call_count >= len(self._responses):
            # Default: end turn with no text
            yield StreamEvent(type="text", text="Done.")
            return

        text, tool_calls = self._responses[self._call_count]
        self._call_count += 1

        if text:
            yield StreamEvent(type="text", text=text)

        for tc in (tool_calls or []):
            yield StreamEvent(type="tool_start", text=tc.name)
            yield StreamEvent(type="tool_done", tool_call=tc)


@pytest.fixture
def mock_tools():
    """Create a mock tool registry."""
    registry = MagicMock()
    registry.get_litellm_tools.return_value = [
        {"type": "function", "function": {"name": "read_file", "parameters": {}}}
    ]
    registry.is_destructive.return_value = False

    async def mock_execute(name, arguments):
        return ToolResult(
            tool_call_id="",
            name=name,
            content=f"Mock result for {name}",
        )

    registry.execute = AsyncMock(side_effect=mock_execute)
    return registry


class TestAgentTextOnly:
    @pytest.mark.asyncio
    async def test_simple_text(self, mock_tools):
        provider = MockProvider([("Hello world", [])])
        agent = Agent(provider=provider, tool_registry=mock_tools)
        result = await agent.run("hi")

        assert result == "Hello world"
        assert len(agent.history) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_text_callback(self, mock_tools):
        collected = []
        provider = MockProvider([("Hello", [])])
        agent = Agent(
            provider=provider,
            tool_registry=mock_tools,
            on_text=lambda t: collected.append(t),
        )
        await agent.run("hi")
        assert "Hello" in collected


class TestAgentToolCalls:
    @pytest.mark.asyncio
    async def test_single_tool_call(self, mock_tools):
        tc = ToolCall(id="1", name="read_file", arguments={"path": "main.py"})
        provider = MockProvider([
            ("Let me read that.", [tc]),  # First: tool call
            ("Here is the file content.", []),  # Second: final text
        ])

        agent = Agent(provider=provider, tool_registry=mock_tools)
        result = await agent.run("read main.py")

        assert "file content" in result
        mock_tools.execute.assert_called_once_with("read_file", {"path": "main.py"})
        # History: user, assistant+tool_call, tool_result, assistant
        assert len(agent.history) == 4

    @pytest.mark.asyncio
    async def test_multi_step_tool_calls(self, mock_tools):
        tc1 = ToolCall(id="1", name="read_file", arguments={"path": "a.py"})
        tc2 = ToolCall(id="2", name="read_file", arguments={"path": "b.py"})
        provider = MockProvider([
            ("Reading files.", [tc1]),
            ("Read another.", [tc2]),
            ("All done.", []),
        ])

        agent = Agent(provider=provider, tool_registry=mock_tools)
        result = await agent.run("read both files")

        assert "All done" in result
        assert mock_tools.execute.call_count == 2


class TestAgentToolDenial:
    @pytest.mark.asyncio
    async def test_denied_tool(self, mock_tools):
        mock_tools.is_destructive.return_value = True

        tc = ToolCall(id="1", name="write_file", arguments={"path": "x.py", "content": "x"})
        provider = MockProvider([
            ("I'll write the file.", [tc]),
            ("OK, I won't write it.", []),
        ])

        agent = Agent(
            provider=provider,
            tool_registry=mock_tools,
            confirm_tool=AsyncMock(return_value=False),
        )
        result = await agent.run("write x.py")

        mock_tools.execute.assert_not_called()
        # History should contain the denial
        tool_results = [m for m in agent.history if m.get("role") == "tool"]
        assert any("denied" in m.get("content", "").lower() for m in tool_results)


class TestAgentMaxIterations:
    @pytest.mark.asyncio
    async def test_iteration_limit(self, mock_tools):
        tc = ToolCall(id="1", name="read_file", arguments={"path": "a.py"})
        # Provider always returns tool calls
        provider = MockProvider(
            [("Reading.", [tc])] * 30
        )

        agent = Agent(provider=provider, tool_registry=mock_tools, max_iterations=3)
        result = await agent.run("infinite loop")

        assert "iteration limit" in result.lower()
        assert mock_tools.execute.call_count == 3


class TestAgentMCP:
    @pytest.mark.asyncio
    async def test_mcp_tool_routing(self, mock_tools):
        mock_mcp = MagicMock()
        mock_mcp.discover_tools = AsyncMock(return_value=[])
        mock_mcp.is_mcp_tool.return_value = True

        async def mock_call(name, args):
            return ToolResult(tool_call_id="", name=name, content="MCP result")

        mock_mcp.call_tool = AsyncMock(side_effect=mock_call)

        tc = ToolCall(id="1", name="github__list_repos", arguments={})
        provider = MockProvider([
            ("Checking repos.", [tc]),
            ("Found repos.", []),
        ])

        agent = Agent(
            provider=provider,
            tool_registry=mock_tools,
            mcp_manager=mock_mcp,
        )
        result = await agent.run("list repos")

        mock_mcp.call_tool.assert_called_once()
        mock_tools.execute.assert_not_called()


class TestHistoryManagement:
    @pytest.mark.asyncio
    async def test_clear_history(self, mock_tools):
        provider = MockProvider([("Hello", [])])
        agent = Agent(provider=provider, tool_registry=mock_tools)
        await agent.run("hi")
        assert len(agent.history) > 0
        agent.clear_history()
        assert len(agent.history) == 0

    def test_estimate_tokens(self, mock_tools):
        provider = MockProvider([])
        agent = Agent(provider=provider, tool_registry=mock_tools)
        agent.history = [
            {"role": "user", "content": "a" * 400},
            {"role": "assistant", "content": "b" * 400},
        ]
        estimate = agent.estimate_tokens()
        assert estimate == 200  # 800 chars / 4

    def test_trim_history(self, mock_tools):
        provider = MockProvider([])
        agent = Agent(provider=provider, tool_registry=mock_tools)
        # Fill history with lots of content
        for i in range(20):
            agent.history.append({"role": "user", "content": "x" * 10000})
            agent.history.append({"role": "assistant", "content": "y" * 10000})

        agent.trim_history(max_tokens=5000)
        assert agent.estimate_tokens() <= 5000
