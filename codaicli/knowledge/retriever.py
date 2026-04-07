"""Knowledge retriever — semantic search over the knowledge base."""

from __future__ import annotations

import math

import litellm

from codaicli.knowledge.store import KnowledgeStore
from codaicli.knowledge.types import KnowledgeEntry


class KnowledgeRetriever:
    """Searches the knowledge base using semantic similarity or keyword fallback."""

    EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(self, store: KnowledgeStore):
        self.store = store

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[KnowledgeEntry]:
        """Search for relevant knowledge entries.

        Uses embedding-based cosine similarity if embeddings are available,
        otherwise falls back to keyword matching.
        """
        embeddings = self.store.get_all_embeddings()

        if embeddings:
            # Try semantic search
            query_embedding = await self._embed(query)
            if query_embedding:
                scored = []
                for entry_id, emb in embeddings:
                    sim = self._cosine_similarity(query_embedding, emb)
                    scored.append((entry_id, sim))

                scored.sort(key=lambda x: x[1], reverse=True)
                top_ids = [eid for eid, _ in scored[:top_k]]

                results = []
                for eid in top_ids:
                    entry = self.store.get(eid)
                    if entry:
                        results.append(entry)
                return results

        # Fallback: keyword matching
        return self._keyword_search(query, top_k)

    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Return formatted knowledge for system prompt injection.

        Uses keyword search (sync) to avoid needing async in the hot path.
        Picks top entries, truncates to fit token budget.
        """
        entries = self._keyword_search(query, top_k=3)
        if not entries:
            return ""

        parts = []
        total_tokens = 0
        for entry in entries:
            if total_tokens + entry.token_count > max_tokens:
                break
            parts.append(f"### {entry.title}\n{entry.content}")
            total_tokens += entry.token_count

        return "\n\n".join(parts)

    def _keyword_search(self, query: str, top_k: int = 5) -> list[KnowledgeEntry]:
        """Simple keyword-based search as fallback."""
        query_lower = query.lower()
        words = query_lower.split()

        entries = self.store.get_all(include_stale=False)
        scored = []
        for entry in entries:
            text = f"{entry.title} {entry.content} {entry.scope}".lower()
            score = sum(1 for w in words if w in text)
            if score > 0:
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:top_k]]

    async def _embed(self, text: str) -> list[float] | None:
        """Generate embedding for the query text."""
        try:
            response = await litellm.aembedding(
                model=self.EMBEDDING_MODEL,
                input=[text],
            )
            return response.data[0]["embedding"]
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Pure Python cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)
