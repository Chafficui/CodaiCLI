"""Canonical data types for CodaiCLI agent system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class ToolDefinition:
    """Schema for a tool the model can call."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    is_destructive: bool = False


@dataclass
class StreamEvent:
    """A single event from a streaming response."""

    type: str  # "text", "tool_start", "tool_done", "error", "usage"
    text: str | None = None
    tool_call: ToolCall | None = None
    usage: dict[str, int] | None = None
