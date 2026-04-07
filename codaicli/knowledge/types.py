"""Data types for the knowledge base system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeEntry:
    """A single knowledge base entry about the project."""

    id: str
    category: str  # "overview", "module", "architecture", "pattern"
    title: str
    content: str
    scope: str  # e.g. "codaicli/agent.py" or "project"
    source_files: list[str] = field(default_factory=list)
    source_hashes: dict[str, str] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: str = ""
    updated_at: str = ""
    is_stale: bool = False
    token_count: int = 0
