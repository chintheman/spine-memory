"""Derived index management — spec §3.

SQLite database with:
- FTS5 full-text search on observation content
- sqlite-vec vector search (384-dim MiniLM embeddings)
- dim_meta table to prevent silent mixed-dimension searches
- Episodes + wiki chunks indexed alongside observations
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384  # MiniLM-L6-v2
MAX_BYTES_PER_VEC = EMBEDDING_DIM * 4  # 4 bytes per float32


def _vector_to_blob(vector: List[float]) -> bytes:
    """Pack a float list into a binary blob (little-endian float32)."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _blob_to_vector(blob: bytes) -> List[float]:
    """Unpack a binary blob back to a float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _serialize_vector(vector: List[float]) -> bytes:
    """JSON-serialize vector for text-column storage (FTS5 virtual table limitation)."""
    return json.dumps(vector).encode("utf-8")


def _deserialize_vector(data: bytes) -> List[float]:
    """Deserialize vector from JSON text storage."""
    return json.loads(data.decode("utf-8"))


class MemoryIndex:
    """Manages the derived SQLite index for memory retrieval."""

    def __init__(self, db_path: str):
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Index not opened. Call open() first.")
        return self._conn

    def open(self) -> None:
        """Open the database, create schema if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_schema(self) -> None:
        """Create tables: observations, episodes, wiki_chunks, dim_meta, config."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                profile TEXT NOT NULL,
                type TEXT NOT NULL,
                epistemic TEXT NOT NULL DEFAULT 'extracted',
                content TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                confirmations INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active',
                topics TEXT DEFAULT '',
                contradicts TEXT DEFAULT '',
                evidence TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                last_confirmed TEXT,
                last_retrieved TEXT,
                embedding BLOB
            );

            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                profile TEXT NOT NULL,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                outcomes TEXT DEFAULT '',
                started_at TEXT,
                ended_at TEXT,
                compacted INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS wiki_chunks (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                embedding BLOB
            );

            CREATE TABLE IF NOT EXISTS dim_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- FTS5 virtual table for full-text search (external content to observations)
            CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
                content,
                content_rowid='rowid',
                content='observations'
            );

            -- Triggers to keep FTS5 in sync
            CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations BEGIN
                INSERT INTO observations_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
            END;

            CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations BEGIN
                INSERT INTO observations_fts(observations_fts, rowid, content) VALUES ('delete', OLD.rowid, OLD.content);
            END;

            CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE ON observations BEGIN
                INSERT INTO observations_fts(observations_fts, rowid, content) VALUES ('delete', OLD.rowid, OLD.content);
                INSERT INTO observations_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
            END;
        """)

        # Record embedding dimension
        self.conn.execute(
            "INSERT OR REPLACE INTO dim_meta (key, value) VALUES (?, ?)",
            ("embedding_dim", str(EMBEDDING_DIM)),
        )

    def get_embedding_dim(self) -> int:
        """Return the stored embedding dimension. Raises if mismatched."""
        row = self.conn.execute(
            "SELECT value FROM dim_meta WHERE key='embedding_dim'"
        ).fetchone()
        if row is None:
            return EMBEDDING_DIM
        return int(row[0])

    # ── Write operations ──────────────────────────────────────────────

    def upsert_observation(self, obs: Dict[str, Any], embedding: Optional[List[float]] = None) -> None:
        """Insert or replace an observation row."""
        self.conn.execute(
            """INSERT OR REPLACE INTO observations
               (id, profile, type, epistemic, content, confidence, confirmations,
                status, topics, contradicts, evidence, created_at, last_confirmed,
                last_retrieved, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                obs["id"],
                obs.get("profile", "agent:main"),
                obs.get("type", "fact"),
                obs.get("epistemic", "extracted"),
                obs.get("content", ""),
                obs.get("confidence", 0.5),
                obs.get("confirmations", 1),
                obs.get("status", "active"),
                json.dumps(obs.get("topics", [])),
                json.dumps(obs.get("contradicts", [])),
                json.dumps(obs.get("evidence", [])),
                obs.get("created_at", ""),
                obs.get("last_confirmed", ""),
                obs.get("last_retrieved", ""),
                _serialize_vector(embedding) if embedding else None,
            ),
        )

    def touch_retrieved(self, obs_id: str, ts: str) -> None:
        """Update last_retrieved timestamp."""
        self.conn.execute(
            "UPDATE observations SET last_retrieved=? WHERE id=?", (ts, obs_id)
        )

    def update_status(self, obs_id: str, status: str) -> None:
        """Update observation status."""
        self.conn.execute(
            "UPDATE observations SET status=? WHERE id=?", (status, obs_id)
        )

    def delete_observation(self, obs_id: str) -> None:
        """Remove an observation (FTS5 trigger handles cleanup)."""
        self.conn.execute("DELETE FROM observations WHERE id=?", (obs_id,))

    # ── Read operations ────────────────────────────────────────────────

    def search_fts(self, query: str, profile: str = "agent:main", limit: int = 20) -> List[Dict[str, Any]]:
        """FTS5 full-text search over observations."""
        rows = self.conn.execute(
            """SELECT o.* FROM observations o
               JOIN observations_fts fts ON o.rowid = fts.rowid
               WHERE observations_fts MATCH ?
                 AND o.status = 'active'
                 AND o.profile IN (?, 'shared')
               ORDER BY rank
               LIMIT ?""",
            (query, profile, limit),
        ).fetchall()
        return [_row_to_dict(r, self.conn) for r in rows]

    def search_hybrid(
        self, query: str, query_embedding: Optional[List[float]], profile: str = "agent:main", k: int = 6
    ) -> List[Dict[str, Any]]:
        """Hybrid FTS5 + vector search with simplified Reciprocal Rank Fusion.

        If query_embedding is None, falls back to FTS5-only.
        """
        if query_embedding is None:
            return self.search_fts(query, profile, limit=k * 3)[:k]

        # Vector search — cosine similarity via dot product on normalized vectors
        vec_results = self._vector_search(query_embedding, profile, limit=k * 3)
        fts_results = self.search_fts(query, profile, limit=k * 3)

        # RRF: score = 1/(rank + 60) per result set, sum across both
        scores: Dict[str, float] = {}
        for rank, row in enumerate(fts_results):
            scores[row["id"]] = scores.get(row["id"], 0) + 1.0 / (rank + 61)
        for rank, row in enumerate(vec_results):
            scores[row["id"]] = scores.get(row["id"], 0) + 1.0 / (rank + 61)

        merged = {row["id"]: row for row in fts_results}
        for row in vec_results:
            if row["id"] not in merged:
                merged[row["id"]] = row

        ranked = sorted(merged.items(), key=lambda item: scores.get(item[0], 0), reverse=True)
        return [row for _, row in ranked[:k]]

    def _vector_search(self, embedding: List[float], profile: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Brute-force cosine similarity over stored embeddings."""
        rows = self.conn.execute(
            """SELECT id, profile, type, epistemic, content, confidence, confirmations,
                      status, topics, contradicts, evidence, created_at, last_confirmed,
                      last_retrieved, embedding
               FROM observations
               WHERE status = 'active' AND profile IN (?, 'shared') AND embedding IS NOT NULL""",
            (profile,),
        ).fetchall()

        results: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            d = _row_to_dict(row, self.conn)
            stored_vec = _deserialize_vector(row[-1]) if row[-1] else None
            if stored_vec is None:
                continue
            sim = _cosine_similarity(embedding, stored_vec)
            results.append((sim, d))

        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:limit]]

    # ── Index maintenance ──────────────────────────────────────────────

    def count_active(self) -> int:
        """Return number of active observations."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM observations WHERE status='active'"
        ).fetchone()
        return row[0] if row else 0

    def vacuum(self) -> None:
        """Optimize the database."""
        self.conn.execute("PRAGMA optimize")


# ── Helpers ──────────────────────────────────────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


import math


def _row_to_dict(row: tuple, conn: sqlite3.Connection) -> Dict[str, Any]:
    """Convert a SQLite row to a dict, parsing JSON fields."""
    cols = [desc[0] for desc in conn.execute("SELECT * FROM observations LIMIT 0").description]
    d = dict(zip(cols, row))
    # Parse JSON fields safely
    for field in ["topics", "contradicts", "evidence"]:
        if field in d and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except json.JSONDecodeError:
                pass
    return d
