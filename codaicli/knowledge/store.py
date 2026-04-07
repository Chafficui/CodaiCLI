"""SQLite-backed storage for knowledge base entries."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from codaicli.knowledge.types import KnowledgeEntry


class KnowledgeStore:
    """Persistent storage for knowledge entries using SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create the entries table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    source_files TEXT NOT NULL DEFAULT '[]',
                    source_hashes TEXT NOT NULL DEFAULT '{}',
                    embedding TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_stale INTEGER NOT NULL DEFAULT 0,
                    token_count INTEGER NOT NULL DEFAULT 0
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _row_to_entry(self, row: tuple) -> KnowledgeEntry:
        """Convert a database row to a KnowledgeEntry."""
        return KnowledgeEntry(
            id=row[0],
            category=row[1],
            title=row[2],
            content=row[3],
            scope=row[4],
            source_files=json.loads(row[5]),
            source_hashes=json.loads(row[6]),
            embedding=json.loads(row[7]) if row[7] else None,
            created_at=row[8],
            updated_at=row[9],
            is_stale=bool(row[10]),
            token_count=row[11],
        )

    def upsert(self, entry: KnowledgeEntry):
        """Insert or update a knowledge entry."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO entries
                    (id, category, title, content, scope, source_files,
                     source_hashes, embedding, created_at, updated_at,
                     is_stale, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    category=excluded.category,
                    title=excluded.title,
                    content=excluded.content,
                    scope=excluded.scope,
                    source_files=excluded.source_files,
                    source_hashes=excluded.source_hashes,
                    embedding=excluded.embedding,
                    updated_at=excluded.updated_at,
                    is_stale=excluded.is_stale,
                    token_count=excluded.token_count
                """,
                (
                    entry.id,
                    entry.category,
                    entry.title,
                    entry.content,
                    entry.scope,
                    json.dumps(entry.source_files),
                    json.dumps(entry.source_hashes),
                    json.dumps(entry.embedding) if entry.embedding else None,
                    entry.created_at,
                    entry.updated_at,
                    int(entry.is_stale),
                    entry.token_count,
                ),
            )

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        """Get a single entry by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM entries WHERE id = ?", (entry_id,)
            ).fetchone()
            return self._row_to_entry(row) if row else None

    def get_by_scope(self, scope: str) -> list[KnowledgeEntry]:
        """Get all entries matching a scope."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM entries WHERE scope = ?", (scope,)
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]

    def get_all(self, include_stale: bool = False) -> list[KnowledgeEntry]:
        """Get all entries, optionally including stale ones."""
        with self._connect() as conn:
            if include_stale:
                rows = conn.execute("SELECT * FROM entries").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM entries WHERE is_stale = 0"
                ).fetchall()
            return [self._row_to_entry(r) for r in rows]

    def mark_stale(self, entry_id: str):
        """Mark a single entry as stale."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE entries SET is_stale = 1 WHERE id = ?", (entry_id,)
            )

    def mark_stale_by_file(self, filepath: str):
        """Mark all entries that depend on the given file as stale."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, source_files FROM entries WHERE is_stale = 0"
            ).fetchall()
            for row_id, source_files_json in rows:
                source_files = json.loads(source_files_json)
                if filepath in source_files:
                    conn.execute(
                        "UPDATE entries SET is_stale = 1 WHERE id = ?",
                        (row_id,),
                    )

    def delete(self, entry_id: str):
        """Delete an entry."""
        with self._connect() as conn:
            conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))

    def get_all_embeddings(self) -> list[tuple[str, list[float]]]:
        """Get (id, embedding) pairs for all non-stale entries with embeddings."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM entries WHERE embedding IS NOT NULL AND is_stale = 0"
            ).fetchall()
            return [
                (row[0], json.loads(row[1]))
                for row in rows
            ]
