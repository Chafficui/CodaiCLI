"""Tests for codaicli.knowledge.store."""

import json
import os

import pytest

from codaicli.knowledge.store import KnowledgeStore
from codaicli.knowledge.types import KnowledgeEntry


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / ".codaicli" / "knowledge.db")
    return KnowledgeStore(db_path)


def _make_entry(**kwargs) -> KnowledgeEntry:
    defaults = dict(
        id="test-1",
        category="module",
        title="Test Module",
        content="This is a test module.",
        scope="test.py",
        source_files=["test.py"],
        source_hashes={"test.py": "abc123"},
        embedding=None,
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        is_stale=False,
        token_count=10,
    )
    defaults.update(kwargs)
    return KnowledgeEntry(**defaults)


class TestKnowledgeStore:
    def test_init_creates_db(self, tmp_path):
        db_path = str(tmp_path / "sub" / "knowledge.db")
        store = KnowledgeStore(db_path)
        assert os.path.exists(db_path)

    def test_upsert_and_get(self, store):
        entry = _make_entry()
        store.upsert(entry)

        result = store.get("test-1")
        assert result is not None
        assert result.id == "test-1"
        assert result.title == "Test Module"
        assert result.content == "This is a test module."
        assert result.source_files == ["test.py"]
        assert result.source_hashes == {"test.py": "abc123"}

    def test_upsert_update(self, store):
        entry = _make_entry()
        store.upsert(entry)

        updated = _make_entry(content="Updated content", updated_at="2025-06-01T00:00:00Z")
        store.upsert(updated)

        result = store.get("test-1")
        assert result.content == "Updated content"

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_get_by_scope(self, store):
        store.upsert(_make_entry(id="1", scope="a.py"))
        store.upsert(_make_entry(id="2", scope="b.py"))
        store.upsert(_make_entry(id="3", scope="a.py"))

        results = store.get_by_scope("a.py")
        assert len(results) == 2
        assert all(r.scope == "a.py" for r in results)

    def test_get_all_excludes_stale(self, store):
        store.upsert(_make_entry(id="1", is_stale=False))
        store.upsert(_make_entry(id="2", is_stale=True))

        results = store.get_all(include_stale=False)
        assert len(results) == 1
        assert results[0].id == "1"

    def test_get_all_includes_stale(self, store):
        store.upsert(_make_entry(id="1", is_stale=False))
        store.upsert(_make_entry(id="2", is_stale=True))

        results = store.get_all(include_stale=True)
        assert len(results) == 2

    def test_mark_stale(self, store):
        store.upsert(_make_entry(id="1"))
        store.mark_stale("1")

        result = store.get("1")
        assert result.is_stale is True

    def test_mark_stale_by_file(self, store):
        store.upsert(_make_entry(id="1", source_files=["a.py", "b.py"]))
        store.upsert(_make_entry(id="2", source_files=["c.py"]))
        store.upsert(_make_entry(id="3", source_files=["a.py"]))

        store.mark_stale_by_file("a.py")

        assert store.get("1").is_stale is True
        assert store.get("2").is_stale is False
        assert store.get("3").is_stale is True

    def test_delete(self, store):
        store.upsert(_make_entry(id="1"))
        store.delete("1")
        assert store.get("1") is None

    def test_get_all_embeddings(self, store):
        store.upsert(_make_entry(id="1", embedding=[0.1, 0.2, 0.3]))
        store.upsert(_make_entry(id="2", embedding=None))
        store.upsert(_make_entry(id="3", embedding=[0.4, 0.5, 0.6]))

        embeddings = store.get_all_embeddings()
        assert len(embeddings) == 2
        ids = [eid for eid, _ in embeddings]
        assert "1" in ids
        assert "3" in ids

    def test_get_all_embeddings_excludes_stale(self, store):
        store.upsert(_make_entry(id="1", embedding=[0.1, 0.2], is_stale=True))
        store.upsert(_make_entry(id="2", embedding=[0.3, 0.4], is_stale=False))

        embeddings = store.get_all_embeddings()
        assert len(embeddings) == 1
        assert embeddings[0][0] == "2"
