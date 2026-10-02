#!/usr/bin/env python3
"""Provenance audit for benchmark queries — P0-2 gate.

For every expected_hit term in queries.json, record which tier(s) contain it
(observations vs wiki chunks), attach a derived `tiers` field per query, and
report the corpus coverage ceiling. Read-only: opens memory.db mode=ro,
never touches last_retrieved or any write path.

Usage:
  python3 bench_provenance.py [--db ~/.hermes/memory.db] [--queries ~/wiki/_memory/bench/queries.json] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

DB_DEFAULT = os.path.expanduser("~/.hermes/memory.db")
QUERIES_DEFAULT = os.path.expanduser("~/wiki/_memory/bench/queries.json")

# Mirrors SEARCHABLE_STATUSES in index.py
SEARCHABLE_STATUSES = ("active", "promoted")


def main() -> None:
    parser = argparse.ArgumentParser(description="Provenance audit for bench queries")
    parser.add_argument("--db", default=DB_DEFAULT)
    parser.add_argument("--queries", default=QUERIES_DEFAULT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    with open(args.queries, "r", encoding="utf-8") as f:
        queries = json.load(f)

    db_path = os.path.abspath(os.path.expanduser(args.db))
    # macOS/WAL quirk: `mode=ro` URI can fail at first query. Open normally
    # and enforce read-only at the engine level — this script only SELECTs.
    from spine.index import connect_db
    conn = connect_db(db_path)
    conn.execute("PRAGMA query_only = ON")

    # Load full text of both tiers (content only — small enough)
    obs_rows = conn.execute(
        "SELECT content FROM observations WHERE status IN (%s)"
        % ",".join("?" * len(SEARCHABLE_STATUSES)),
        SEARCHABLE_STATUSES,
    ).fetchall()
    obs_text = "\n".join(r[0] or "" for r in obs_rows).lower()

    wiki_rows = conn.execute(
        "SELECT title, content FROM wiki_chunks"
    ).fetchall()
    wiki_text = "\n".join(f"{r[0] or ''} {r[1] or ''}" for r in wiki_rows).lower()

    conn.close()

    total_terms = 0
    covered_terms = 0
    per_query = []

    for q in queries:
        # Schema: must_hit/may_hit (new) or expected_hits (legacy)
        if "must_hit" in q:
            terms = list(q.get("must_hit", [])) + list(q.get("may_hit", []))
        else:
            terms = q.get("expected_hits", [])
        term_rows = []
        for term in terms:
            tl = term.lower()
            in_obs = tl in obs_text
            in_wiki = tl in wiki_text
            if in_obs and in_wiki:
                tiers = "both"
            elif in_obs:
                tiers = "obs"
            elif in_wiki:
                tiers = "wiki"
            else:
                tiers = "neither"
            total_terms += 1
            if tiers != "neither":
                covered_terms += 1
            term_rows.append({"term": term, "tiers": tiers})

        # Derived tier = where the EXCLUSIVE content lives (what a diagonal
        # row can actually discriminate). Fixed Aug 14 2026: prior branches
        # mislabelled obs-exclusive queries as "both" and never emitted "obs".
        has_obs_excl = any(t["tiers"] == "obs" for t in term_rows)
        has_wiki_excl = any(t["tiers"] == "wiki" for t in term_rows)
        has_both = any(t["tiers"] == "both" for t in term_rows)
        has_neither = any(t["tiers"] == "neither" for t in term_rows)
        if has_obs_excl and has_wiki_excl:
            derived = "mixed"
        elif has_obs_excl:
            derived = "obs"
        elif has_wiki_excl:
            derived = "wiki"
        elif has_both:
            derived = "both"
        else:
            derived = "neither"

        q_out = {
            "query": q["query"],
            "source": q.get("source", ""),
            "k": q.get("k", 6),
            "n_terms": len(terms),
            "n_obs": sum(1 for t in term_rows if t["tiers"] in ("obs", "both")),
            "n_wiki": sum(1 for t in term_rows if t["tiers"] in ("wiki", "both")),
            "n_neither": sum(1 for t in term_rows if t["tiers"] == "neither"),
            "n_obs_exclusive": sum(1 for t in term_rows if t["tiers"] == "obs"),
            "n_wiki_exclusive": sum(1 for t in term_rows if t["tiers"] == "wiki"),
            "derived_tiers": derived,
            "terms": term_rows,
        }
        per_query.append(q_out)

    summary = {
        "n_queries": len(queries),
        "total_expected_terms": total_terms,
        "covered_terms": covered_terms,
        "coverage_ceiling": round(covered_terms / total_terms, 3) if total_terms else None,
        "queries_wiki_ge_obs": sum(1 for q in per_query if q["n_wiki"] >= q["n_obs"]),
        "queries_zero_obs": sum(1 for q in per_query if q["n_obs"] == 0),
        "derived_tier_counts": {
            t: sum(1 for q in per_query if q["derived_tiers"] == t)
            for t in ("obs", "wiki", "both", "mixed", "neither")
        },
        "queries": per_query,
    }

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    # Human-readable table
    print(f"=== Provenance audit: {args.queries} ===")
    print(f"Corpus: {len(obs_rows)} observations, {len(wiki_rows)} wiki chunks")
    print(f"Coverage ceiling: {summary['coverage_ceiling']} ({summary['covered_terms']}/{summary['total_expected_terms']} terms)")
    print(f"Queries wiki>=obs: {summary['queries_wiki_ge_obs']}/{summary['n_queries']}  |  Queries 0-obs: {summary['queries_zero_obs']}/{summary['n_queries']}")
    print(f"Derived tiers: {summary['derived_tier_counts']}")
    print()
    print(f"{'source':<12} {'#obs':<5} {'#wiki':<6} {'#nei':<5} {'derived':<8} query")
    print("-" * 100)
    for q in per_query:
        print(f"{q['source']:<12} {q['n_obs']:<5} {q['n_wiki']:<6} {q['n_neither']:<5} {q['derived_tiers']:<8} {q['query'][:60]}")
    print()
    print("=== Terms with NO coverage (neither tier) ===")
    for q in per_query:
        missing = [t["term"] for t in q["terms"] if t["tiers"] == "neither"]
        if missing:
            print(f"  [{q['source']}] {q['query'][:50]}: {missing}")


if __name__ == "__main__":
    main()
