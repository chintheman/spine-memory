"""Derived index management — spec §3.

SQLite database with:
- FTS5 full-text search on observation content
- brute-force vector search over packed float32 blobs (width set by the embedder)
- dim_meta table to prevent silent mixed-dimension searches
- Episodes + wiki chunks indexed alongside observations
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover — falls back to the pure-Python path
    np = None

logger = logging.getLogger(__name__)

# Vector width. Must match whatever the embedder actually produces: the packed
# float32 blob format is fixed-width, and _stack() reshapes on this number, so a
# mismatch does not error -- it silently reinterprets the bytes and returns
# nonsense similarity scores. Read it from the embedder rather than hardcoding,
# so swapping models cannot leave a stale 384 behind.
_DEFAULT_EMBEDDING_DIM = 384
_embedding_dim: Optional[int] = None
_dim_probe_failed = False


def embedding_dim() -> int:
    """Vector width, resolved lazily and cached.

    Was a module-level call, which meant importing spine.index loaded a
    SentenceTransformer: 8.1s and ~420MB on EVERY entry point, including
    keyword-only work like forget() or a read-only heartbeat check, and a
    ~420MB download triggered by an `import` statement on a cold machine.

    Resolution order is deliberate. The store's own dim_meta wins, because the
    width that matters is the width of the bytes already on disk. Only if the
    store is silent do we ask the embedder.
    """
    global _embedding_dim, _dim_probe_failed
    if _embedding_dim is not None:
        return _embedding_dim
    # Store first, as documented: the width that matters is the width of the
    # bytes already on disk. Read-only URI so a missing store is NOT created --
    # a plain connect() materialised a zero-byte memory.db. Closed in finally:
    # the previous version skipped close() whenever the SELECT raised, and this
    # is called once per vector, so it leaked a descriptor per row.
    #
    # Through connect_db (review round 2, 2026-08-30): this probe used a bare
    # connect with timeout=1.0, so under a long writer it gave up, cached the
    # miss, and let the embedder answer -- a wrong width would then be stamped
    # by the first _create_schema to see an empty dim_meta. The store is WAL,
    # so a reader never actually waits on a writer; the 30s is for the
    # rollback-journal edge only.
    if not _dim_probe_failed:
        con = None
        try:
            from .config import load_spine_config
            db = os.path.expanduser(load_spine_config().db)
            if os.path.exists(db):
                con = connect_db(f"file:{db}?mode=ro", uri=True)
                row = con.execute(
                    "SELECT value FROM dim_meta WHERE key='embedding_dim'").fetchone()
                if row:
                    _embedding_dim = int(row[0])
                    return _embedding_dim
        except Exception:  # noqa: BLE001 — no store, no table, or locked
            pass
        finally:
            if con is not None:
                con.close()
        # Remember the miss. Without this the probe re-ran per vector.
        _dim_probe_failed = True
    try:
        from .embedder import get_embedding_dim
        d = get_embedding_dim()
        if d:
            _embedding_dim = int(d)
            return _embedding_dim
    except Exception:  # noqa: BLE001 — embedder unavailable right now
        pass
    # Last resort: ask the DATA. A hardcoded fallback is how dim_meta ended up
    # claiming 384 over a store of 3,127 768-dim vectors — the probe failed once
    # (locked DB or embedder not yet loaded) and the constant got stamped in.
    # The stored vectors cannot be wrong about their own width.
    try:
        from .config import load_spine_config
        db = os.path.expanduser(load_spine_config().db)
        if os.path.exists(db):
            con = connect_db(f"file:{db}?mode=ro", uri=True)
            try:
                row = con.execute(
                    "SELECT LENGTH(embedding) FROM observations "
                    "WHERE embedding IS NOT NULL LIMIT 1").fetchone()
            finally:
                con.close()
            if row and row[0]:
                return int(row[0]) // 4
    except Exception:  # noqa: BLE001
        pass
    # Do NOT cache the fallback: a transient model-load failure must not freeze
    # the width for the life of the process.
    return _DEFAULT_EMBEDDING_DIM


def max_bytes_per_vec() -> int:
    return embedding_dim() * 4

# Recency scoring constants. NOTE (found 2026-07-30): despite the name and
# SpineConfig having matching `recency_half_life_hours`/`recency_weight`
# fields, nothing in this file's search path actually reads SpineConfig —
# `_compute_recency_factor` is always called with these hardcoded defaults,
# never with config-sourced values. Not fixed as part of this pass (real
# wiring means threading a SpineConfig instance through MemoryIndex/search()
# call sites, a bigger change than the config.py parsing gap this comment
# used to imply was the only issue) — flagged for a follow-up if these knobs
# need to be genuinely configurable.
_DEFAULT_RECENCY_HALF_LIFE_HOURS = 168  # 7 days
_DEFAULT_RECENCY_WEIGHT = 0.15  # contribution of recency to final score


def _compute_recency_factor(row: Dict[str, Any], half_life_hours: float = _DEFAULT_RECENCY_HALF_LIFE_HOURS,
                            weight: float = _DEFAULT_RECENCY_WEIGHT) -> float:
    """Compute recency multiplier for an observation row.

    Returns 1.0 for wiki chunks (no recency). For observations, computes
    exponential decay based on last_confirmed (fallback: created_at).
    Formula: (1 - weight) + weight * exp(-hours_idle / half_life)
    """
    if row.get("source") == "wiki":
        return 1.0
    ref_date = row.get("last_confirmed") or row.get("created_at", "")
    if not ref_date:
        return 1.0
    try:
        ref_dt = datetime.fromisoformat(ref_date)
        hours_idle = (datetime.now(timezone.utc) - ref_dt).total_seconds() / 3600
        if hours_idle <= 0:
            return 1.0
        decay = math.exp(-hours_idle / half_life_hours)
        return (1.0 - weight) + weight * decay
    except (ValueError, TypeError):
        return 1.0


def _fts5_safe_query(text: str) -> str:
    """Build a safe FTS5 MATCH expression from free text.

    FTS5 treats apostrophes, hyphens, parentheses, colons etc. as query
    syntax, not literal characters — "What is the user's name?" crashes
    MATCH with a syntax error otherwise. Quoting each token forces FTS5 to
    treat it as a literal string match, immune to being misparsed as an
    operator, at the cost of losing FTS5's own phrase/prefix operators in
    user-supplied queries (an acceptable tradeoff — those operators were
    never intentionally exposed to callers anyway).

    Stopwords are dropped (Aug 14 2026): the OR of every token including
    "what", "and", "am", "I" barely filters and lets bm25 rank carry all
    discrimination, which buried gold chunks at rank 60-470 on bench
    queries. Dropping them keeps the MATCH set tighter and the ranks
    meaningful. Token count floor: a query reduced to zero tokens returns
    empty (caller treats as no-match), and a query with 1-2 tokens still
    works as a plain OR of those.
    """
    STOPWORDS = {
        "the", "and", "are", "for", "you", "your", "that", "this", "with",
        "what", "how", "why", "does", "did", "was", "were", "have", "has",
        "had", "not", "but", "can", "from", "they", "them", "will", "would",
        "could", "should", "there", "their", "about", "into", "over", "after",
        "before", "between", "out", "off", "all", "any", "who", "whom",
        "whose", "which", "when", "where", "am", "is", "be", "been", "being",
        "to", "of", "in", "on", "at", "by", "it", "its", "i", "a", "an", "or",
    }
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    filtered = [t for t in tokens if t.lower() not in STOPWORDS]
    # Degenerate-query guard: if stopword filtering empties the query
    # (e.g. "how are you"), fall back to the unfiltered tokens rather than
    # returning no match at all — recall("how are you") should still work.
    tokens = filtered if filtered else tokens
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def _entity_match_boost(query_words: set, row: Dict[str, Any], boost: float = 1.25) -> float:
    """Multiplicative score boost when the query mentions one of this observation's
    tagged entities (people, tools, projects — populated by the observer's extraction
    pass, stored in the `topics` field). Neutral (1.0) if no entities or no match."""
    topics = row.get("topics")
    if not topics or not isinstance(topics, list):
        return 1.0
    for t in topics:
        if isinstance(t, str) and t.strip().lower() in query_words:
            return boost
    return 1.0

# NOTE: use max_bytes_per_vec() / embedding_dim(); the module-level constants
# were removed so a stale width cannot be frozen at import time.

# Statuses that recall is allowed to return.
#
# 'promoted' used to be excluded here, on the theory that a promoted
# observation already lives in MEMORY.md and therefore doesn't need to be
# searchable. That was backwards: promotion is applied to the *highest*
# confidence observations, so excluding them removed the best material from
# search — 90 of 158 rows at the time this was fixed. 'demoted' is the state
# an observation lands in when it gets pushed back out of MEMORY.md for space;
# it must stay searchable too, or demotion becomes deletion.
# Recall window. Measured 2026-08-20: at k=6 the correct memory for a
# multi-part question was regularly retrieved and ranked, then discarded just
# below the cutoff -- 7/12 multi-hop cases passed. k=20 passes 9/12 at the same
# latency (17.6ms vs 17.9ms), because the candidate pool was already being
# fetched and scored; only the final slice was throwing the results away.
#
# Raised 20 -> 25 on 2026-09-14 for the same reason as the candidate-pool bump
# (60 -> 200 on 2026-08-25): the store outgrew the tuning. At 2,330 observations
# the golden observation for eval case mh-memory-sync-latency (obs
# 094BMZDETBKM5RGB8H2WD0YY6K, profile agent:claude-code) sat at rank 21 -- one
# place outside the window -- while the content itself was intact and carried
# both needles. At k=25 all 29 eval cases pass; at k=20 one does not.
# Measured cost of the extra rows across all 29 eval queries: 23,168 -> 28,423
# chars of payload, +5,255 chars (~1,314 tokens, +22.7%) per recall CALL -- and
# recall is agent-invoked rather than per-turn. k=25 also matches what
# handle_reflect already used, so the read paths now share one depth.
# This buys margin, not a cure: 11 of that query's top 20 are near-duplicate
# wiki chunks, so the window will saturate again as the store grows. The durable
# fix is a wiki-share cap or chunk dedupe in search_hybrid, which costs no
# tokens and would let this return to 20.
DEFAULT_K = 25

SEARCHABLE_STATUSES = ("active", "promoted", "demoted")
_STATUS_PLACEHOLDERS = ",".join("?" * len(SEARCHABLE_STATUSES))

# How long a connection waits on a busy lock before raising "database is
# locked". sqlite3.connect() defaults to 5s, which is shorter than a nightly
# consolidation pass or a wiki reindex holding the WAL write lock. The live
# memory.db is shared by every cron consumer plus active sessions, and on
# 2026-08-30 16:01 the 4-hourly Claude Code sync crashed at its very first
# write (the dim_meta stamp in _create_schema) because another writer held
# the lock for >5s. 30s covers every writer this store has; a genuine hang
# still surfaces as an error rather than blocking forever.
DB_BUSY_TIMEOUT_S = 30.0


def connect_db(db_path: str, **kwargs: Any) -> sqlite3.Connection:
    """sqlite3.connect() with the spine-wide busy timeout applied.

    Every spine script that opens memory.db directly (heartbeat, hotcore sync,
    benchmark snapshot, propose_memories) should go through this so the
    timeout is set in one place. Extra kwargs (uri=True, ...) pass through.
    """
    kwargs.setdefault("timeout", DB_BUSY_TIMEOUT_S)
    return sqlite3.connect(db_path, **kwargs)


def _vector_to_blob(vector: List[float]) -> bytes:
    """Pack a float list into a binary blob (little-endian float32)."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _blob_to_vector(blob: bytes) -> List[float]:
    """Unpack a binary blob back to a float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _json_field(value: Any) -> str:
    """Serialize a list-ish column without double-encoding.

    upsert_observation used to call json.dumps() unconditionally. When the
    caller passed a value that was ALREADY a JSON string — which happens on
    any round-trip through the index — you got '"[\\"#11\\"]"'. That
    deserializes to a str, not a list, and _entity_match_boost requires a
    list: it silently returned the neutral 1.0 for every affected row, so
    entity matching was dead on exactly the observations that had been
    re-indexed. Present in live data when this was found.
    """
    if value is None:
        return "[]"
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return "[]"
        if s[0] in "[{":
            try:
                json.loads(s)
                return s  # already valid JSON — store verbatim
            except json.JSONDecodeError:
                pass
        return json.dumps([value])  # bare string -> single-element list
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value))
    return json.dumps(value)


def _normalize(vector: List[float]) -> List[float]:
    """Scale to unit length so cosine similarity is a plain dot product."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return list(vector)
    return [x / norm for x in vector]


def _serialize_vector(vector: List[float]) -> bytes:
    """Pack a vector for storage: pre-normalized little-endian float32.

    Was JSON text (~8.4KB per 384-dim vector, and json.loads dominated query
    time at ~65% of runtime). Packed float32 is 1,536 bytes and decodes with
    a zero-copy numpy view. Vectors are normalized on the way in so the search
    path never has to compute magnitudes.
    """
    return _vector_to_blob(_normalize(vector))


def _deserialize_vector(data: Any) -> Optional[List[float]]:
    """Decode a stored vector, accepting both the packed and legacy formats.

    Discriminate on LENGTH, not on a leading '[' byte. A packed vector is
    always exactly embedding_dim()*4 bytes; the legacy JSON encoding of a
    384-dim vector runs ~8.4KB and can never hit that size. Sniffing the first
    byte instead looks right but is wrong: 0x5B ('[') is a perfectly ordinary
    low mantissa byte, so roughly 1 packed vector in 256 starts with it. That
    misread cost a full silent fallback to the scalar path here once already.
    """
    if not data:
        return None
    if isinstance(data, str):
        data = data.encode("utf-8")
    if len(data) == max_bytes_per_vec():
        return _blob_to_vector(data)
    if data[:1] == b"[":
        try:
            return json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    if len(data) % 4 != 0:
        return None
    return _blob_to_vector(data)


class MemoryIndex:
    """Manages the derived SQLite index for memory retrieval."""

    def __init__(self, db_path: str):
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._canonical_root: Optional[str] = None
        # (signature, ids, blobs, stacked_matrix) — see _wiki_vectors
        self._wiki_cache: Optional[Tuple[Any, List[str], List[Any], Any]] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Index not opened. Call open() first.")
        return self._conn

    def open(self) -> None:
        """Open the database, create schema if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = connect_db(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        # INSERT OR REPLACE resolves a conflict by deleting the old row, but that
        # delete does NOT fire AFTER DELETE triggers unless recursive_triggers is
        # ON (SQLite default is OFF). With it off, upsert_observation() and the
        # wiki reindex leaked a stale FTS document on every re-write: the index
        # held 2,082 docs against 521 real rows (75% phantoms) and every IDF was
        # computed against a majority-ghost corpus. Set it per-connection in
        # open() so every writer (incl. rebuild-index.py, which also goes through
        # MemoryIndex.open()) is covered.
        self._conn.execute("PRAGMA recursive_triggers = ON")
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
            -- tokenize='porter unicode61' added 2026-08-23: unicode61 alone has no
            -- stemming, so "report" never matched "reporting" and a whole class of
            -- verb-form queries scored zero on the FTS channel (spine eval regression
            -- mh-verification-discipline). Existing DBs are migrated by
            -- _migrate_fts_tokenizer(); CREATE IF NOT EXISTS is a silent no-op.
            CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
                content,
                content_rowid='rowid',
                content='observations',
                tokenize='porter unicode61'
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

            -- FTS5 virtual table for wiki content search
            CREATE VIRTUAL TABLE IF NOT EXISTS wiki_chunks_fts USING fts5(
                content,
                content_rowid='rowid',
                content='wiki_chunks',
                tokenize='porter unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS wiki_chunks_ai AFTER INSERT ON wiki_chunks BEGIN
                INSERT INTO wiki_chunks_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
            END;

            CREATE TRIGGER IF NOT EXISTS wiki_chunks_ad AFTER DELETE ON wiki_chunks BEGIN
                INSERT INTO wiki_chunks_fts(wiki_chunks_fts, rowid, content) VALUES ('delete', OLD.rowid, OLD.content);
            END;

            CREATE TRIGGER IF NOT EXISTS wiki_chunks_au AFTER UPDATE ON wiki_chunks BEGIN
                INSERT INTO wiki_chunks_fts(wiki_chunks_fts, rowid, content) VALUES ('delete', OLD.rowid, OLD.content);
                INSERT INTO wiki_chunks_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
            END;
        """)

        # Record the embedding dimension ONLY if the store has none yet.
        # This was INSERT OR REPLACE on every open(), which made the guard
        # inert: whichever process opened the DB last stamped its own width
        # over the stored one, so a model swap without a re-embed could never
        # be detected -- and check_embedder compared the live embedder against
        # a value the live embedder had just written.
        #
        # Read before write: INSERT OR IGNORE still takes the WAL write lock
        # even when the row exists and nothing changes, so every open() --
        # including read-only consumers like recall -- contended with whichever
        # writer was active. Only reach for the lock when the row is missing.
        if self.conn.execute(
            "SELECT 1 FROM dim_meta WHERE key='embedding_dim'"
        ).fetchone() is None:
            self.conn.execute(
                "INSERT OR IGNORE INTO dim_meta (key, value) VALUES (?, ?)",
                ("embedding_dim", str(embedding_dim())),
            )
            # Python's sqlite3 opens an implicit transaction on that INSERT
            # and leaves it open; until the caller commits or closes, this
            # connection holds the WAL write lock against every sibling.
            self.conn.commit()

        self._migrate_fts_tokenizer()

    def _migrate_fts_tokenizer(self) -> None:
        """Recreate FTS5 tables whose stored DDL predates porter stemming.

        CREATE VIRTUAL TABLE IF NOT EXISTS is a silent no-op against an existing
        table, so changing the tokenizer in the DDL above does nothing to a live
        DB. Detect the drift and rebuild. Also repairs orphaned index entries
        left by INSERT OR REPLACE before recursive_triggers was turned on.
        """
        for tbl, src in (("observations_fts", "observations"),
                         ("wiki_chunks_fts", "wiki_chunks")):
            row = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (tbl,)
            ).fetchone()
            if not row or "porter" in row[0]:
                continue
            self.conn.executescript(f"""
                BEGIN;
                DROP TABLE {tbl};
                CREATE VIRTUAL TABLE {tbl} USING fts5(
                    content, content_rowid='rowid', content='{src}',
                    tokenize='porter unicode61');
                INSERT INTO {tbl}({tbl}) VALUES('rebuild');
                COMMIT;
            """)

    def get_embedding_dim(self) -> int:
        """The width the STORE was built at, which is what the bytes on disk are."""
        row = self.conn.execute(
            "SELECT value FROM dim_meta WHERE key='embedding_dim'"
        ).fetchone()
        if row is None:
            return embedding_dim()
        return int(row[0])

    def set_embedding_dim(self, dim: int) -> None:
        """Explicitly restamp the store's width. Only a full re-embed may call this."""
        self.conn.execute(
            "INSERT OR REPLACE INTO dim_meta (key, value) VALUES (?, ?)",
            ("embedding_dim", str(dim)),
        )

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
                _json_field(obs.get("topics", [])),
                _json_field(obs.get("contradicts", [])),
                _json_field(obs.get("evidence", [])),
                obs.get("created_at", ""),
                obs.get("last_confirmed", ""),
                obs.get("last_retrieved", ""),
                _serialize_vector(embedding) if embedding else None,
            ),
        )

    def upsert_episode(self, episode: Dict[str, Any]) -> None:
        """Insert or replace a session-episode row."""
        self.conn.execute(
            """INSERT OR REPLACE INTO episodes
               (id, profile, session_id, summary, outcomes, started_at, ended_at, compacted)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode["id"],
                episode.get("profile", "agent:main"),
                episode.get("session_id", "unknown"),
                episode.get("summary", ""),
                json.dumps(episode.get("outcomes", [])),
                episode.get("started_at", ""),
                episode.get("ended_at", ""),
                1 if episode.get("compacted") else 0,
            ),
        )

    def touch_retrieved(self, obs_id: str, ts: str) -> None:
        """Update last_retrieved timestamp."""
        self.conn.execute(
            "UPDATE observations SET last_retrieved=? WHERE id=?", (ts, obs_id)
        )

    def update_status(self, obs_id: str, status: str) -> None:
        """Update observation status in the DB AND the canonical JSONL.

        Status used to live only in the DB. Because rebuild-index.py rebuilds
        the table from the JSONL, that made a rebuild silently destructive: it
        reverted every demotion, flipped those rows back to 'active', and the
        next promote pass re-inflated the MEMORY.md hot core. 147 rows had
        drifted by the time it was found.

        The canonical store is append-only, so the write-back is a patch line,
        which load_observations() already merges in order.
        """
        self.conn.execute(
            "UPDATE observations SET status=? WHERE id=?", (status, obs_id)
        )
        self._write_back_status(obs_id, status)

    def _write_back_status(self, obs_id: str, status: str) -> None:
        row = self.conn.execute(
            "SELECT profile FROM observations WHERE id=?", (obs_id,)).fetchone()
        if not row:
            return
        path = self._canonical_jsonl(row[0])
        if not path or not os.path.exists(path):
            # No canonical file means nothing to keep in step. Never create one
            # here: a status patch is not a place to invent a store.
            return
        try:
            from .jsonl_writer import JSONLWriter
            JSONLWriter(path).append({
                "id": obs_id,
                "patch": {"status": status},
                "ts": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:  # noqa: BLE001
            # The DB write already landed. Losing the write-back is a drift bug,
            # not a data-loss bug, and the heartbeat's `divergence` check counts
            # exactly this. Log it rather than failing the consolidation run.
            logger.warning("status write-back failed for %s: %s", obs_id, e)

    def _canonical_jsonl(self, profile: str) -> Optional[str]:
        if self._canonical_root is None:
            try:
                from .config import load_spine_config
                self._canonical_root = os.path.expanduser(
                    load_spine_config().canonical_root)
            except Exception:  # noqa: BLE001
                self._canonical_root = ""
        if not self._canonical_root:
            return None
        return os.path.join(self._canonical_root, "observations", f"{profile}.jsonl")

    def delete_observation(self, obs_id: str) -> None:
        """Remove an observation (FTS5 trigger handles cleanup)."""
        self.conn.execute("DELETE FROM observations WHERE id=?", (obs_id,))

    # ── Read operations ────────────────────────────────────────────────

    @staticmethod
    def _profile_scope(profile: str) -> List[str]:
        """Profiles a search spans. '*' means every profile in the store.

        'shared' is the magic label wiki chunks are surfaced under, so it is
        always included: it is the corpus both agents draw on.
        """
        if profile in ("*", "all"):
            return ["*"]
        return [profile, "shared"]

    def _profile_sql(self, profile: str, column: str = "profile") -> Tuple[str, List[str]]:
        """SQL fragment + params restricting a query to the profile scope."""
        scope = self._profile_scope(profile)
        if scope == ["*"]:
            return "", []
        return f" AND {column} IN ({','.join('?' * len(scope))})", scope

    def search_fts(self, query: str, profile: str = "agent:main", limit: int = 20) -> List[Dict[str, Any]]:
        """FTS5 full-text search over observations AND wiki chunks."""
        fts_query = _fts5_safe_query(query)
        if not fts_query:
            return []

        profile_sql, profile_params = self._profile_sql(profile, "o.profile")

        # Observations
        obs_rows = self.conn.execute(
            f"""SELECT o.* FROM observations o
               JOIN observations_fts fts ON o.rowid = fts.rowid
               WHERE observations_fts MATCH ?
                 AND o.status IN ({_STATUS_PLACEHOLDERS})
                 {profile_sql}
               ORDER BY rank
               LIMIT ?""",
            (fts_query, *SEARCHABLE_STATUSES, *profile_params, limit),
        ).fetchall()

        # Wiki chunks
        wiki_rows = self.conn.execute(
            """SELECT wc.id, 'shared' AS profile, 'fact' AS type, 'extracted' AS epistemic,
                      (wc.title || ': ' || wc.content) AS content,
                      1.0 AS confidence, 1 AS confirmations, 'active' AS status, '' AS topics,
                      '' AS contradicts, '' AS evidence, '' AS created_at, '' AS last_confirmed,
                      '' AS last_retrieved,
                      'wiki' AS source, wc.path, wc.chunk_index
               FROM wiki_chunks wc
               JOIN wiki_chunks_fts fts ON wc.rowid = fts.rowid
               WHERE wiki_chunks_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (fts_query, limit),
        ).fetchall()

        results = [_row_to_dict(r, self.conn) for r in obs_rows]
        wiki_dicts = []
        for r in wiki_rows:
            d = {
                "id": r[0], "profile": r[1], "type": r[2], "epistemic": r[3],
                "content": r[4], "confidence": r[5], "confirmations": r[6],
                "status": r[7], "topics": r[8], "contradicts": r[9],
                "evidence": r[10], "created_at": r[11], "last_confirmed": r[12],
                "last_retrieved": r[13], "source": r[14], "path": r[15],
            }
            wiki_dicts.append(d)

        # Fair-share interleave (fix Aug 14 2026): previously observations were
        # appended first and results[:limit] truncated, so wiki rows were
        # silently discarded whenever observations saturated the limit
        # (verified: wiki FTS survivors = 0 on 20/20 bench queries). Both
        # sources now get comparable RRF rank footing and neither can starve.
        merged: List[Dict[str, Any]] = []
        oi, wi = 0, 0
        while len(merged) < limit and (oi < len(results) or wi < len(wiki_dicts)):
            if oi < len(results):
                merged.append(results[oi])
                oi += 1
            if len(merged) < limit and wi < len(wiki_dicts):
                merged.append(wiki_dicts[wi])
                wi += 1

        return merged

    def search_hybrid(
        self, query: str, query_embedding: Optional[List[float]], profile: str = "agent:main", k: int = DEFAULT_K
    ) -> List[Dict[str, Any]]:
        """Hybrid FTS5 + vector search with Reciprocal Rank Fusion and recency weighting.

        If query_embedding is None, falls back to FTS5-only.
        Recency: recent observations get a weighted boost via exponential decay
        (half-life from config, default 7 days/168h, 15% weight).
        """
        # Candidate pool (Aug 14 2026): k*3 was starving deep golds — FTS gold
        # chunks measured at rank 13-474 on bench queries were never fetched at
        # all, and the wiki-vector path already uncapped for the same reason
        # (see _vector_search comment). Grid-tested Aug 14: pool 40/60/100/300
        # all score recall 0.641 (vs 0.558 at k*3); 60 chosen as headroom
        # without paying for 300-candidate RRF every call.
        # Raised to 200 on 2026-08-25: at pool 60 the golden observation for
        # eval case mh-verification-discipline sat just outside the candidate
        # set (rank 9 at pool 90, absent at 60), so the eval regression gate
        # flipped after the store grew past the Aug-14 tuning size. Swept
        # Swept floors 60/100/150/200/300/450 against the 29-case eval set:
        # only >= 200 passes all (60 and 100/150 each starve one case).
        # Bench A/B on the current store: 200 costs 0.686->0.681 recall
        # (-0.7%) vs the 60 floor while closing the eval gap.
        pool = max(k * 3, 200)
        if query_embedding is None:
            return self.search_fts(query, profile, limit=pool)[:k]

        # Vector search — cosine similarity via dot product on normalized vectors
        vec_results = self._vector_search(query_embedding, profile, limit=pool)
        fts_results = self.search_fts(query, profile, limit=pool)

        # RRF: score = 1/(rank + 60) per result set, sum across both
        scores: Dict[str, float] = {}
        for rank, row in enumerate(fts_results):
            scores[row["id"]] = scores.get(row["id"], 0) + 1.0 / (rank + 61)
        for rank, row in enumerate(vec_results):
            scores[row["id"]] = scores.get(row["id"], 0) + 1.0 / (rank + 61)

        # Apply recency weighting (bonus for recent observations)
        merged = {row["id"]: row for row in fts_results}
        for row in vec_results:
            if row["id"] not in merged:
                merged[row["id"]] = row

        query_words = {w.lower() for w in query.split() if len(w) >= 3}
        recency_factor = _compute_recency_factor
        for obs_id, row in merged.items():
            if row.get("source") == "wiki":
                continue  # Wiki chunks get neutral recency (1.0)
            scores[obs_id] = scores.get(obs_id, 0) * recency_factor(row)
            scores[obs_id] = scores.get(obs_id, 0) * _entity_match_boost(query_words, row)

        ranked = sorted(merged.items(), key=lambda item: scores.get(item[0], 0), reverse=True)
        return [row for _, row in ranked[:k]]

    def _wiki_vectors(self) -> Tuple[List[str], List[Any]]:
        """Wiki chunk ids + vectors, with the stacked matrix cached per connection.

        The wiki table only changes on a reindex, but the vectors were being
        re-read and re-stacked on every query — a ~4MB copy each time. The
        cache is invalidated by a cheap (count, max rowid) signature, so a
        reindex is picked up without any explicit cache-busting call.
        """
        sig = self.conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM wiki_chunks WHERE embedding IS NOT NULL"
        ).fetchone()
        if self._wiki_cache and self._wiki_cache[0] == sig:
            return self._wiki_cache[1], self._wiki_cache[2]

        rows = self.conn.execute(
            "SELECT id, embedding FROM wiki_chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        ids = [r[0] for r in rows]
        blobs = [r[1] for r in rows]
        self._wiki_cache = (sig, ids, blobs, _stack(blobs))
        return ids, blobs

    def _vector_search(self, embedding: List[float], profile: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Brute-force cosine similarity over observations AND wiki chunks."""
        results: List[Tuple[float, Dict[str, Any]]] = []
        profile_sql, profile_params = self._profile_sql(profile)

        # Observations
        obs_rows = self.conn.execute(
            f"""SELECT id, profile, type, epistemic, content, confidence, confirmations,
                      status, topics, contradicts, evidence, created_at, last_confirmed,
                      last_retrieved, embedding
               FROM observations
               WHERE status IN ({_STATUS_PLACEHOLDERS})
                 {profile_sql}
                 AND embedding IS NOT NULL""",
            (*SEARCHABLE_STATUSES, *profile_params),
        ).fetchall()

        if obs_rows:
            sims = _similarities(embedding, [r[-1] for r in obs_rows])
            for row, sim in zip(obs_rows, sims):
                if sim <= -1.0:
                    continue  # undecodable vector
                results.append((sim, _row_to_dict(row, self.conn)))

        # Wiki chunks with embeddings.
        #
        # This query used to end `LIMIT ?` bound to limit*2 — with the default
        # k=6 that read 6*3*2 = 36 of 2,643 chunks, and not the best 36: there
        # was no ORDER BY, so SQLite returned the first 36 by rowid, i.e. an
        # arbitrary slice of insertion order with no relevance criterion at
        # all. 98.6% of the wiki was unreachable by semantic search. Keyword
        # search always covered the full set, which is why this stayed hidden.
        # Removing the cap is only affordable because scoring is now a single
        # numpy matmul over packed float32 rather than a per-row Python loop.
        #
        # Scoring is deliberately split into two passes: rank on (id, vector)
        # alone, then fetch title/content for the survivors only. Selecting the
        # text alongside the vectors pulled several MB of chunk bodies out of
        # SQLite on every single query to build rows that were then discarded.
        wiki_ids, wiki_blobs = self._wiki_vectors()
        if wiki_ids:
            cached_matrix = self._wiki_cache[3] if self._wiki_cache else None
            sims = _similarities(embedding, wiki_blobs, matrix=cached_matrix)
            top = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:limit]
            top = [i for i in top if sims[i] > -1.0]
            if top:
                ids = [wiki_ids[i] for i in top]
                placeholders = ",".join("?" * len(ids))
                fetched = {
                    r[0]: r for r in self.conn.execute(
                        f"""SELECT id, path, title, content
                            FROM wiki_chunks WHERE id IN ({placeholders})""", ids
                    ).fetchall()
                }
                for i in top:
                    row = fetched.get(wiki_ids[i])
                    if row is None:
                        continue
                    results.append((sims[i], {
                        "id": row[0], "profile": "shared", "type": "fact",
                        "epistemic": "extracted",
                        "content": f"{row[2]}: {row[3]}",
                        "confidence": 1.0, "confirmations": 1, "status": "active",
                        "topics": "", "contradicts": "", "evidence": "",
                        "created_at": "", "last_confirmed": "", "last_retrieved": "",
                        "source": "wiki", "path": row[1],
                    }))

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


def _stack(blobs: List[Any]) -> Optional["np.ndarray"]:
    """Stack packed float32 blobs into an (N, dim) matrix.

    Returns None if numpy is missing or any row isn't in the packed format —
    which happens mid-migration, when legacy JSON vectors are still present.
    """
    if np is None or not blobs:
        return None
    # Length alone identifies the packed format — see _deserialize_vector for
    # why a leading-byte check is actively wrong here.
    for b in blobs:
        if not isinstance(b, (bytes, memoryview)) or len(b) != max_bytes_per_vec():
            return None
    return np.frombuffer(b"".join(bytes(b) for b in blobs),
                         dtype="<f4").reshape(len(blobs), embedding_dim())


def _similarities(query: List[float], blobs: List[Any],
                  matrix: Optional["np.ndarray"] = None) -> List[float]:
    """Cosine similarity of `query` against every stored vector.

    Uses one numpy matmul over a stacked (N, dim) matrix — replacing a per-row
    Python loop that was the dominant cost once the wiki scan stopped being
    artificially capped at 36 rows. Pass `matrix` to reuse a cached stack and
    skip rebuilding it per query. Falls back to the scalar path when numpy is
    unavailable or the rows are ragged.
    """
    if matrix is None and not blobs:
        return []

    m = matrix if matrix is not None else _stack(blobs)
    if m is not None:
        q = np.asarray(query, dtype="<f4")
        qnorm = float(np.linalg.norm(q))
        if qnorm:
            q = q / qnorm
        # Stored vectors are pre-normalized by _serialize_vector, so a dot
        # product IS the cosine. Re-normalizing here would cost a second pass
        # over the matrix for no gain.
        return m.dot(q).tolist()

    out: List[float] = []
    for blob in blobs:
        vec = _deserialize_vector(blob)
        out.append(_cosine_similarity(query, vec) if vec else -1.0)
    return out


_OBS_COLUMNS: Optional[List[str]] = None


def _row_to_dict(row: tuple, conn: sqlite3.Connection) -> Dict[str, Any]:
    """Convert a SQLite row to a dict, parsing JSON fields."""
    global _OBS_COLUMNS
    if _OBS_COLUMNS is None:
        # Cached: this used to run a fresh SELECT per row purely to read
        # column names, which meant one extra query per search result.
        _OBS_COLUMNS = [d[0] for d in
                        conn.execute("SELECT * FROM observations LIMIT 0").description]
    d = dict(zip(_OBS_COLUMNS, row))
    # Parse JSON fields safely
    for field in ["topics", "contradicts", "evidence"]:
        if field in d and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except json.JSONDecodeError:
                pass
    # Raw embedding bytes are internal to vector search — never JSON-serializable, never needed by callers
    d.pop("embedding", None)
    return d
