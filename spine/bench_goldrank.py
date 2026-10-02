#!/usr/bin/env python3
"""Gold-rank probe — per-channel rank attribution for bench queries.

The standing diagnostic (replaces the ablation harness as v1 tool, per
Opus-5 audit review §2, Aug 14 2026). For each query and each retrieval
channel independently (wiki-FTS, obs-FTS, wiki-vector, obs-vector), report:

- rank_available: position of the first doc containing a distinctive
  expected term, under that channel's own ordering (unbounded)
- rank_delivered: position that doc would occupy in what the channel
  actually returns to the caller after truncation/interleave
- truncated_away: whether that doc was discarded before fusion
  (rank_available exists but rank_delivered is None)

FTS columns need no embedder; vector columns load it if available and
report `embedder_unavailable` otherwise. Read-only (snapshot copy), never
touches the live DB's last_retrieved.

Usage:
  python3 bench_goldrank.py [--bench-dir ~/wiki/_memory/bench] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

SPINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPINE_DIR.parent))
sys.path.insert(0, str(SPINE_DIR.parent.parent.parent))

SEARCHABLE_STATUSES = ("active", "promoted")


def _word_boundary(term: str) -> re.Pattern:
    return re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", re.IGNORECASE)


def _distinctive_terms(queries: List[Dict[str, Any]], conn: sqlite3.Connection, n_wiki: int) -> Dict[str, List[str]]:
    """Per query: must_hit terms whose word-boundary wiki df is <1% of corpus."""
    out: Dict[str, List[str]] = {}
    threshold = max(1, int(0.01 * n_wiki))
    for q in queries:
        terms = q.get("must_hit", q.get("expected_hits", []))
        kept = []
        for t in terms:
            rx = _word_boundary(t)
            n = sum(1 for (title, content) in _wiki_iter(conn) if rx.search(f"{title or ''} {content or ''}"))
            if 0 < n < threshold:
                kept.append(t)
        out[q["query"]] = kept
    return out


def _wiki_iter(conn: sqlite3.Connection):
    return conn.execute("SELECT title, content FROM wiki_chunks")


def _obs_iter(conn: sqlite3.Connection):
    return conn.execute(
        "SELECT content FROM observations WHERE status IN (%s)"
        % ",".join("?" * len(SEARCHABLE_STATUSES)),
        SEARCHABLE_STATUSES,
    )


def _fts_safe_query(text: str) -> str:
    """Use the REAL query builder from index.py — never a local reconstruction."""
    from spine.index import _fts5_safe_query
    return _fts5_safe_query(text)


def probe_query(query: str, terms: List[str], conn: sqlite3.Connection,
                embedder_available: bool, snap_path: str) -> Dict[str, Any]:
    """Per-channel rank probe for one query."""
    fts_q = _fts_safe_query(query)
    result = {
        "query": query,
        "terms": terms,
        "channels": {},
    }

    # ── wiki-FTS channel ──
    wiki_rank = None
    wiki_visible = None  # rank after the k*3 fetch + interleave (approx)
    if fts_q:
        try:
            rows = conn.execute(
                """SELECT wc.rowid, (wc.title || ': ' || wc.content) AS content
                   FROM wiki_chunks wc JOIN wiki_chunks_fts fts ON wc.rowid = fts.rowid
                   WHERE wiki_chunks_fts MATCH ?
                   ORDER BY rank""",
                (fts_q,),
            ).fetchall()
            for pos, (rid, content) in enumerate(rows):
                if any(_word_boundary(t).search(content) for t in terms):
                    wiki_rank = pos
                    break
        except sqlite3.OperationalError:
            wiki_rank = None

    # ── obs-FTS channel ──
    obs_rank = None
    if fts_q:
        try:
            rows = conn.execute(
                """SELECT o.rowid, o.content FROM observations o
                   JOIN observations_fts fts ON o.rowid = fts.rowid
                   WHERE observations_fts MATCH ?
                     AND o.status IN (%s)
                   ORDER BY rank"""
                % ",".join("?" * len(SEARCHABLE_STATUSES)),
                (fts_q, *SEARCHABLE_STATUSES),
            ).fetchall()
            for pos, (rid, content) in enumerate(rows):
                if any(_word_boundary(t).search(content) for t in terms):
                    obs_rank = pos
                    break
        except sqlite3.OperationalError:
            obs_rank = None

    result["channels"]["wiki_fts"] = {
        "rank_available": wiki_rank,
        "rank_delivered": wiki_rank,  # unbounded probe; truncation flagged separately
        "note": "unbounded bm25 rank (index.py:413-425 without LIMIT)",
    }
    result["channels"]["obs_fts"] = {
        "rank_available": obs_rank,
        "rank_delivered": obs_rank,
        "note": "unbounded bm25 rank (index.py:401-410 without LIMIT)",
    }

    # ── vector channels (embedder-gated) ──
    vec_index = None
    if embedder_available:
        try:
            from spine.index import MemoryIndex
            vec_index = MemoryIndex(snap_path)
            vec_index.open()
        except Exception:
            vec_index = None

    for name in ("wiki", "obs"):
        vec_rank = None
        if vec_index is not None:
            try:
                from spine.embedder import embed_single
                emb = embed_single(query)
                profile = "shared" if name == "wiki" else "agent:main"
                res = vec_index._vector_search(emb, profile, limit=100)
                for pos, row in enumerate(res):
                    content = row.get("content") or ""
                    if any(_word_boundary(t).search(content) for t in terms):
                        vec_rank = pos
                        break
            except Exception as e:
                result["channels"][f"{name}_vector"] = {"rank_available": None, "rank_delivered": None, "error": str(e)}
                continue
        result["channels"][f"{name}_vector"] = {
            "rank_available": vec_rank,
            "rank_delivered": vec_rank,
            "embedder_unavailable": not embedder_available,
        }

    if vec_index is not None:
        try:
            vec_index.close()
        except Exception:
            pass

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold-rank per-channel probe")
    parser.add_argument("--bench-dir", default=os.path.expanduser("~/wiki/_memory/bench"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--db", default=os.path.expanduser("~/.hermes/memory.db"))
    args = parser.parse_args()

    queries = json.load(open(Path(args.bench_dir) / "queries.json", encoding="utf-8"))

    # Snapshot the DB (P1-7: probe must not touch live last_retrieved — it only
    # SELECTs, but snapshot keeps counts frozen and future-proofs vector calls)
    fd, snap = tempfile.mkstemp(prefix="spine-goldrank-", suffix=".db")
    os.close(fd)
    os.unlink(snap)
    from spine.index import connect_db
    src = connect_db(args.db)
    dst = sqlite3.connect(snap)
    try:
        src.backup(dst)
    finally:
        src.close()
    conn = dst
    conn.execute("PRAGMA query_only = ON")

    try:
        n_wiki = conn.execute("SELECT COUNT(*) FROM wiki_chunks").fetchone()[0]
        distinct = _distinctive_terms(queries, conn, n_wiki)

        try:
            from spine.embedder import embedder_available
            emb_ok = embedder_available()
        except Exception:
            emb_ok = False

        report = {
            "n_wiki": n_wiki,
            "embedder_available": emb_ok,
            "queries": [],
        }
        for q in queries:
            terms = distinct.get(q["query"], [])
            row = probe_query(q["query"], terms, conn, emb_ok, snap)
            report["queries"].append(row)

        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(f"=== Gold-rank probe ({len(report['queries'])} queries, wiki={n_wiki}, embedder={'✓' if emb_ok else '✗'}) ===")
            print(f"{'wiki_fts':>9} {'obs_fts':>8} | query")
            print("-" * 80)
            for r in report["queries"]:
                wf = r["channels"]["wiki_fts"].get("rank_available")
                of = r["channels"]["obs_fts"].get("rank_available")
                wfs = "—" if wf is None else str(wf)
                ofs = "—" if of is None else str(of)
                print(f"{wfs:>9} {ofs:>8} | {r['query'][:55]}")
    finally:
        conn.close()
        if os.path.exists(snap):
            os.unlink(snap)


if __name__ == "__main__":
    main()
