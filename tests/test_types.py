"""Tests for codaicli.types."""

from codaicli.types import StreamEvent, ToolCall, ToolDefinition, ToolResult


class TestToolCall:
    def test_construction(self):
        tc = ToolCall(id="123", name="read_file", arguments={"path": "foo.py"})
        assert tc.id == "123"
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "foo.py"}


class TestToolResult:
    def test_defaults(self):
        tr = ToolResult(tool_call_id="1", name="read_file", content="hello")
        assert tr.is_error is False

    def test_error(self):
        tr = ToolResult(tool_call_id="1", name="read_file", content="fail", is_error=True)
        assert tr.is_error is True


class TestToolDefinition:
    def test_defaults(self):
        td = ToolDefinition(
            name="test", description="desc", parameters={"type": "object"}
        )
        assert td.is_destructive is False

    def test_destructive(self):
        td = ToolDefinition(
            name="test",
            description="desc",
            parameters={"type": "object"},
            is_destructive=True,
        )
        assert td.is_destructive is True


class TestStreamEvent:
    def test_text_event(self):
        e = StreamEvent(type="text", text="hello")
        assert e.text == "hello"
        assert e.tool_call is None

    def test_tool_done_event(self):
        tc = ToolCall(id="1", name="test", arguments={})
        e = StreamEvent(type="tool_done", tool_call=tc)
        assert e.tool_call is tc
        assert e.text is None
