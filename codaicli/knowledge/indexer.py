"""Knowledge indexer — generates project documentation via LLM."""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import litellm

from codaicli.knowledge.prompts import (
    ARCHITECTURE_PROMPT,
    MODULE_PROMPT,
    OVERVIEW_PROMPT,
    PATTERN_PROMPT,
)
from codaicli.knowledge.store import KnowledgeStore
from codaicli.knowledge.types import KnowledgeEntry
from codaicli.provider import Provider


class KnowledgeIndexer:
    """Orchestrates LLM-based documentation generation for a project."""

    EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        provider: Provider,
        store: KnowledgeStore,
        project_path: str,
    ):
        self.provider = provider
        self.store = store
        self.project_path = Path(project_path).resolve()

    async def index_project(
        self,
        force: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ):
        """Generate knowledge for the entire project.

        Generates: overview, per-module summaries, architecture, patterns.
        Skips entries whose source files haven't changed unless force=True.
        """
        progress = on_progress or (lambda msg: None)
        py_files = self._find_python_files()

        if not py_files:
            progress("No Python files found.")
            return

        # 1. Overview
        progress("Generating project overview...")
        await self._index_overview(py_files, force)

        # 2. Module summaries
        for py_file in py_files:
            rel = str(py_file.relative_to(self.project_path))
            progress(f"Indexing module: {rel}")
            await self._index_module(py_file, force)

        # 3. Architecture (depends on module summaries)
        progress("Generating architecture overview...")
        await self._index_architecture(py_files, force)

        # 4. Patterns
        progress("Identifying code patterns...")
        await self._index_patterns(py_files, force)

        progress("Indexing complete.")

    async def refresh_stale(
        self,
        on_progress: Callable[[str], None] | None = None,
    ):
        """Re-generate only entries marked stale."""
        progress = on_progress or (lambda msg: None)
        stale = self.store.get_all(include_stale=True)
        stale = [e for e in stale if e.is_stale]

        if not stale:
            progress("No stale entries.")
            return

        py_files = self._find_python_files()

        for entry in stale:
            progress(f"Refreshing: {entry.title}")
            if entry.category == "overview":
                await self._index_overview(py_files, force=True)
            elif entry.category == "module":
                module_path = self.project_path / entry.scope
                if module_path.exists():
                    await self._index_module(module_path, force=True)
                else:
                    self.store.delete(entry.id)
            elif entry.category == "architecture":
                await self._index_architecture(py_files, force=True)
            elif entry.category == "pattern":
                await self._index_patterns(py_files, force=True)

        progress("Refresh complete.")

    async def _index_overview(self, py_files: list[Path], force: bool):
        """Generate or update the project overview entry."""
        scope = "project"
        existing = self.store.get_by_scope(scope)
        existing = [e for e in existing if e.category == "overview"]

        source_files = [str(f.relative_to(self.project_path)) for f in py_files]
        hashes = self._compute_file_hashes(source_files)

        if existing and not force:
            if not self._check_stale(existing[0], hashes):
                return

        file_list = self._build_file_list(py_files)
        prompt = OVERVIEW_PROMPT.format(file_list=file_list)
        content = await self._call_llm(prompt)

        entry_id = existing[0].id if existing else str(uuid.uuid4())
        created = existing[0].created_at if existing else self._now()

        entry = KnowledgeEntry(
            id=entry_id,
            category="overview",
            title="Project Overview",
            content=content,
            scope=scope,
            source_files=source_files,
            source_hashes=hashes,
            embedding=await self._embed(content),
            created_at=created,
            updated_at=self._now(),
            is_stale=False,
            token_count=len(content) // 4,
        )
        self.store.upsert(entry)

    async def _index_module(self, module_path: Path, force: bool):
        """Generate or update a module summary entry."""
        rel = str(module_path.relative_to(self.project_path))
        existing = self.store.get_by_scope(rel)
        existing = [e for e in existing if e.category == "module"]

        hashes = self._compute_file_hashes([rel])

        if existing and not force:
            if not self._check_stale(existing[0], hashes):
                return

        source_code = module_path.read_text(encoding="utf-8", errors="replace")
        # Truncate very large files
        if len(source_code) > 15_000:
            source_code = source_code[:15_000] + "\n... [truncated]"

        prompt = MODULE_PROMPT.format(module_path=rel, source_code=source_code)
        content = await self._call_llm(prompt)

        entry_id = existing[0].id if existing else str(uuid.uuid4())
        created = existing[0].created_at if existing else self._now()

        entry = KnowledgeEntry(
            id=entry_id,
            category="module",
            title=f"Module: {rel}",
            content=content,
            scope=rel,
            source_files=[rel],
            source_hashes=hashes,
            embedding=await self._embed(content),
            created_at=created,
            updated_at=self._now(),
            is_stale=False,
            token_count=len(content) // 4,
        )
        self.store.upsert(entry)

    async def _index_architecture(self, py_files: list[Path], force: bool):
        """Generate or update the architecture overview."""
        scope = "project"
        existing = self.store.get_by_scope(scope)
        existing = [e for e in existing if e.category == "architecture"]

        source_files = [str(f.relative_to(self.project_path)) for f in py_files]
        hashes = self._compute_file_hashes(source_files)

        if existing and not force:
            if not self._check_stale(existing[0], hashes):
                return

        # Gather module summaries
        all_entries = self.store.get_all()
        module_entries = [e for e in all_entries if e.category == "module"]
        summaries = "\n\n".join(
            f"### {e.scope}\n{e.content}" for e in module_entries
        )

        if not summaries:
            return

        prompt = ARCHITECTURE_PROMPT.format(module_summaries=summaries)
        content = await self._call_llm(prompt)

        entry_id = existing[0].id if existing else str(uuid.uuid4())
        created = existing[0].created_at if existing else self._now()

        entry = KnowledgeEntry(
            id=entry_id,
            category="architecture",
            title="Architecture Overview",
            content=content,
            scope=scope,
            source_files=source_files,
            source_hashes=hashes,
            embedding=await self._embed(content),
            created_at=created,
            updated_at=self._now(),
            is_stale=False,
            token_count=len(content) // 4,
        )
        self.store.upsert(entry)

    async def _index_patterns(self, py_files: list[Path], force: bool):
        """Generate or update the patterns entry."""
        scope = "project"
        existing = self.store.get_by_scope(scope)
        existing = [e for e in existing if e.category == "pattern"]

        source_files = [str(f.relative_to(self.project_path)) for f in py_files]
        hashes = self._compute_file_hashes(source_files)

        if existing and not force:
            if not self._check_stale(existing[0], hashes):
                return

        # Build snippets from first ~200 lines of each file
        snippets = []
        for f in py_files:
            rel = str(f.relative_to(self.project_path))
            try:
                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()[:200]
                snippets.append(f"### {rel}\n```python\n" + "\n".join(lines) + "\n```")
            except Exception:
                continue

        prompt = PATTERN_PROMPT.format(source_snippets="\n\n".join(snippets))
        content = await self._call_llm(prompt)

        entry_id = existing[0].id if existing else str(uuid.uuid4())
        created = existing[0].created_at if existing else self._now()

        entry = KnowledgeEntry(
            id=entry_id,
            category="pattern",
            title="Code Patterns & Conventions",
            content=content,
            scope=scope,
            source_files=source_files,
            source_hashes=hashes,
            embedding=await self._embed(content),
            created_at=created,
            updated_at=self._now(),
            is_stale=False,
            token_count=len(content) // 4,
        )
        self.store.upsert(entry)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a simple prompt and return text."""
        messages = [{"role": "user", "content": prompt}]
        msg, _, _, _ = await self.provider.complete(messages)
        return msg.get("content", "")

    async def _embed(self, text: str) -> list[float] | None:
        """Generate embedding for text. Returns None on failure."""
        try:
            response = await litellm.aembedding(
                model=self.EMBEDDING_MODEL,
                input=[text],
            )
            return response.data[0]["embedding"]
        except Exception:
            return None

    def _find_python_files(self) -> list[Path]:
        """Find all .py files in the project, excluding hidden/venv dirs."""
        exclude_dirs = {
            ".git", ".venv", "venv", "__pycache__", "node_modules",
            ".codaicli", ".tox", ".mypy_cache", ".pytest_cache", "dist",
            "build", "*.egg-info",
        }
        results = []
        for root, dirs, files in os.walk(self.project_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]
            for name in sorted(files):
                if name.endswith(".py"):
                    results.append(Path(root) / name)
        return results

    def _build_file_list(self, py_files: list[Path]) -> str:
        """Build a compact file list with docstrings for the overview prompt."""
        lines = []
        for f in py_files:
            rel = str(f.relative_to(self.project_path))
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                # Extract first docstring or comment
                first_line = ""
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        first_line = stripped.strip("\"'").strip()
                        break
                    if stripped.startswith("#") and not stripped.startswith("#!"):
                        first_line = stripped.lstrip("# ").strip()
                        break
                lines.append(f"- {rel}: {first_line}" if first_line else f"- {rel}")
            except Exception:
                lines.append(f"- {rel}")
        return "\n".join(lines)

    def _compute_file_hashes(self, files: list[str]) -> dict[str, str]:
        """Compute SHA-256 hashes for the given relative file paths."""
        hashes = {}
        for rel in files:
            full = self.project_path / rel
            if full.exists():
                hashes[rel] = hashlib.sha256(
                    full.read_bytes()
                ).hexdigest()
        return hashes

    def _check_stale(self, entry: KnowledgeEntry, current_hashes: dict[str, str]) -> bool:
        """Return True if the entry is stale (hashes differ)."""
        return entry.source_hashes != current_hashes

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
