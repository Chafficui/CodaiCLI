"""Tests for codaicli.knowledge.indexer."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codaicli.knowledge.indexer import KnowledgeIndexer
from codaicli.knowledge.store import KnowledgeStore
from codaicli.knowledge.types import KnowledgeEntry


@pytest.fixture
def project(tmp_path):
    """Create a minimal project with Python files."""
    (tmp_path / "main.py").write_text('"""Main entry point."""\ndef main():\n    pass\n')
    (tmp_path / "utils.py").write_text('"""Utility functions."""\ndef helper():\n    return 1\n')
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "app.py").write_text('"""App module."""\nclass App:\n    pass\n')
    return tmp_path


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / ".codaicli" / "knowledge.db")
    return KnowledgeStore(db_path)


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.model = "test/mock"

    async def mock_complete(messages, tools=None, system_prompt=""):
        return (
            {"role": "assistant", "content": "Generated documentation."},
            [],
            "stop",
            None,
        )

    provider.complete = AsyncMock(side_effect=mock_complete)
    return provider


class TestKnowledgeIndexer:
    @pytest.mark.asyncio
    async def test_index_project(self, project, store, mock_provider):
        indexer = KnowledgeIndexer(mock_provider, store, str(project))

        progress_msgs = []
        with patch.object(indexer, "_embed", return_value=None):
            await indexer.index_project(on_progress=progress_msgs.append)

        # Should have generated entries
        entries = store.get_all(include_stale=True)
        assert len(entries) > 0

        # Should have overview, modules, architecture, patterns
        categories = {e.category for e in entries}
        assert "overview" in categories
        assert "module" in categories
        assert "architecture" in categories
        assert "pattern" in categories

        # Progress messages should have been reported
        assert any("overview" in m.lower() for m in progress_msgs)
        assert any("complete" in m.lower() for m in progress_msgs)

    @pytest.mark.asyncio
    async def test_index_skips_unchanged(self, project, store, mock_provider):
        indexer = KnowledgeIndexer(mock_provider, store, str(project))

        with patch.object(indexer, "_embed", return_value=None):
            await indexer.index_project()
            first_count = mock_provider.complete.call_count

            # Second run should skip (files unchanged)
            mock_provider.complete.reset_mock()
            await indexer.index_project()
            assert mock_provider.complete.call_count == 0

    @pytest.mark.asyncio
    async def test_index_force_reindexes(self, project, store, mock_provider):
        indexer = KnowledgeIndexer(mock_provider, store, str(project))

        with patch.object(indexer, "_embed", return_value=None):
            await indexer.index_project()
            mock_provider.complete.reset_mock()

            await indexer.index_project(force=True)
            assert mock_provider.complete.call_count > 0

    @pytest.mark.asyncio
    async def test_refresh_stale(self, project, store, mock_provider):
        indexer = KnowledgeIndexer(mock_provider, store, str(project))

        with patch.object(indexer, "_embed", return_value=None):
            await indexer.index_project()

            # Mark one entry stale
            entries = store.get_all()
            module_entry = next(e for e in entries if e.category == "module")
            store.mark_stale(module_entry.id)

            mock_provider.complete.reset_mock()
            await indexer.refresh_stale()

            # Should have re-generated the stale entry
            assert mock_provider.complete.call_count >= 1

    @pytest.mark.asyncio
    async def test_refresh_no_stale(self, project, store, mock_provider):
        indexer = KnowledgeIndexer(mock_provider, store, str(project))

        progress_msgs = []
        with patch.object(indexer, "_embed", return_value=None):
            await indexer.refresh_stale(on_progress=progress_msgs.append)

        assert any("no stale" in m.lower() for m in progress_msgs)

    @pytest.mark.asyncio
    async def test_empty_project(self, tmp_path, mock_provider):
        store = KnowledgeStore(str(tmp_path / ".codaicli" / "knowledge.db"))
        indexer = KnowledgeIndexer(mock_provider, store, str(tmp_path))

        progress_msgs = []
        with patch.object(indexer, "_embed", return_value=None):
            await indexer.index_project(on_progress=progress_msgs.append)

        assert any("no python" in m.lower() for m in progress_msgs)

    def test_find_python_files(self, project):
        store = MagicMock()
        provider = MagicMock()
        indexer = KnowledgeIndexer(provider, store, str(project))

        files = indexer._find_python_files()
        names = [f.name for f in files]
        assert "main.py" in names
        assert "utils.py" in names
        assert "app.py" in names

    def test_find_excludes_hidden_dirs(self, project):
        hidden = project / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("x = 1\n")

        venv = project / ".venv"
        venv.mkdir()
        (venv / "lib.py").write_text("x = 1\n")

        store = MagicMock()
        provider = MagicMock()
        indexer = KnowledgeIndexer(provider, store, str(project))

        files = indexer._find_python_files()
        names = [f.name for f in files]
        assert "secret.py" not in names
        assert "lib.py" not in names

    def test_compute_file_hashes(self, project):
        store = MagicMock()
        provider = MagicMock()
        indexer = KnowledgeIndexer(provider, store, str(project))

        hashes = indexer._compute_file_hashes(["main.py", "utils.py"])
        assert "main.py" in hashes
        assert "utils.py" in hashes
        assert len(hashes["main.py"]) == 64  # SHA-256 hex

    def test_check_stale(self, project):
        store = MagicMock()
        provider = MagicMock()
        indexer = KnowledgeIndexer(provider, store, str(project))

        entry = KnowledgeEntry(
            id="1", category="module", title="test", content="test",
            scope="main.py", source_files=["main.py"],
            source_hashes={"main.py": "old_hash"},
        )
        current = {"main.py": "new_hash"}
        assert indexer._check_stale(entry, current) is True

        same = {"main.py": "old_hash"}
        assert indexer._check_stale(entry, same) is False
