"""Tests for codaicli.knowledge.retriever."""

from unittest.mock import AsyncMock, patch

import pytest

from codaicli.knowledge.retriever import KnowledgeRetriever
from codaicli.knowledge.store import KnowledgeStore
from codaicli.knowledge.types import KnowledgeEntry


def _make_entry(**kwargs) -> KnowledgeEntry:
    defaults = dict(
        id="1",
        category="module",
        title="Test Module",
        content="This module handles authentication and login.",
        scope="auth.py",
        source_files=["auth.py"],
        source_hashes={},
        embedding=None,
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        is_stale=False,
        token_count=10,
    )
    defaults.update(kwargs)
    return KnowledgeEntry(**defaults)


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / ".codaicli" / "knowledge.db")
    return KnowledgeStore(db_path)


@pytest.fixture
def retriever(store):
    return KnowledgeRetriever(store)


class TestKeywordSearch:
    def test_basic_search(self, store, retriever):
        store.upsert(_make_entry(id="1", content="handles authentication"))
        store.upsert(_make_entry(id="2", content="database connection pool"))

        results = retriever._keyword_search("authentication", top_k=5)
        assert len(results) == 1
        assert results[0].id == "1"

    def test_multi_word_search(self, store, retriever):
        store.upsert(_make_entry(id="1", content="handles authentication and login"))
        store.upsert(_make_entry(id="2", content="database authentication layer"))

        results = retriever._keyword_search("authentication login", top_k=5)
        # Both match "authentication", but first also matches "login"
        assert len(results) == 2
        assert results[0].id == "1"  # Higher score

    def test_no_matches(self, store, retriever):
        store.upsert(_make_entry(id="1", content="handles authentication"))
        results = retriever._keyword_search("zzz_nonexistent_zzz")
        assert len(results) == 0

    def test_top_k(self, store, retriever):
        for i in range(10):
            store.upsert(_make_entry(id=str(i), content=f"module {i} handles auth"))

        results = retriever._keyword_search("auth", top_k=3)
        assert len(results) == 3

    def test_excludes_stale(self, store, retriever):
        store.upsert(_make_entry(id="1", content="auth module", is_stale=True))
        store.upsert(_make_entry(id="2", content="auth helper", is_stale=False))

        results = retriever._keyword_search("auth")
        assert len(results) == 1
        assert results[0].id == "2"


class TestGetContext:
    def test_basic_context(self, store, retriever):
        store.upsert(_make_entry(
            id="1", title="Auth Module", content="Handles auth.",
            token_count=5,
        ))
        store.upsert(_make_entry(
            id="2", title="DB Module", content="Database layer.",
            token_count=5,
        ))

        ctx = retriever.get_context("auth", max_tokens=2000)
        assert "Auth Module" in ctx
        assert "Handles auth." in ctx

    def test_respects_token_budget(self, store, retriever):
        store.upsert(_make_entry(
            id="1", title="Small Auth", content="Auth handling.",
            token_count=5,
        ))
        store.upsert(_make_entry(
            id="2", title="Big Auth", content="A" * 4000,
            token_count=1000,
        ))

        ctx = retriever.get_context("auth", max_tokens=50)
        # Should include small entry but not big one
        assert "Small Auth" in ctx
        assert "A" * 100 not in ctx

    def test_empty_store(self, store, retriever):
        ctx = retriever.get_context("anything")
        assert ctx == ""


class TestSemanticSearch:
    @pytest.mark.asyncio
    async def test_with_embeddings(self, store, retriever):
        store.upsert(_make_entry(
            id="1", content="auth module", embedding=[1.0, 0.0, 0.0],
        ))
        store.upsert(_make_entry(
            id="2", content="database module", embedding=[0.0, 1.0, 0.0],
        ))

        with patch.object(retriever, "_embed", return_value=[0.9, 0.1, 0.0]):
            results = await retriever.search("authentication")

        assert len(results) >= 1
        assert results[0].id == "1"  # Closer to [1,0,0]

    @pytest.mark.asyncio
    async def test_fallback_to_keyword(self, store, retriever):
        store.upsert(_make_entry(id="1", content="auth module", embedding=None))

        with patch.object(retriever, "_embed", return_value=None):
            results = await retriever.search("auth")

        assert len(results) == 1
        assert results[0].id == "1"


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert KnowledgeRetriever._cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert KnowledgeRetriever._cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert KnowledgeRetriever._cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert KnowledgeRetriever._cosine_similarity([], []) == 0.0

    def test_zero_vector(self):
        assert KnowledgeRetriever._cosine_similarity([0, 0], [1, 0]) == 0.0
