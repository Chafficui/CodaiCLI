"""Knowledge base system for CodaiCLI — self-documenting project context."""

from codaicli.knowledge.types import KnowledgeEntry
from codaicli.knowledge.store import KnowledgeStore
from codaicli.knowledge.indexer import KnowledgeIndexer
from codaicli.knowledge.retriever import KnowledgeRetriever

__all__ = ["KnowledgeEntry", "KnowledgeStore", "KnowledgeIndexer", "KnowledgeRetriever"]
