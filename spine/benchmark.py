#!/usr/bin/env python3
"""Benchmark harness for spine recall — spec §6 (rewritten Aug 14 2026).

Runs a set of fixed queries with expected-hit annotations through recall(),
measures precision@k and recall, and records scores for trend tracking.

Rewrite changes (all from Opus-5 audit review, Aug 14 2026):
- **DB snapshot (P1-7):** runs against a `sqlite3 .backup` copy, so
  `touch_retrieved` (tools.py:311-315) can never pollute the live decay loop.
  Also freezes obs_count/wiki_chunk_count across configs.
- **must_hit/may_hit schema (P2-11):** `must_hit` terms are scored with
  word-boundary matching (no substring false positives); `may_hit` terms are
  reported but never required. Falls back to legacy `expected_hits` if the
  new fields are absent (compat with eval_set.json style files).
- **Per-result provenance (P1-10):** every result records {id, source, path,
  score, rank}; every query records a per-term hit/miss map.
- **Control set:** `controls.json` (separate file, P1-6) with trivial queries
  that ANY config must pass; a structural gate (db exists, obs>0, wiki>0,
  embedder state) runs on every invocation, and the score gate runs only on
  the full config (no-flag run).

Queries live in ~/wiki/_memory/bench/queries.json:
[
  {
    "query": "what are the user's communication preferences?",
    "must_hit": ["concise", "plan-first"],
    "may_hit": ["Telegram"],
    "k": 6,
    "source": "memory"
  }
]

Output: benchmark run report to bench/run-<date>.json and a summary line.
Regression >5% from baseline blocks a change.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add the spine package to path
SPINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPINE_DIR.parent))  # plugins/memory/ — makes 'spine' importable as package
sys.path.insert(0, str(SPINE_DIR.parent.parent.parent))  # hermes-agent root

CONTROLS_FILE = "controls.json"


def load_queries(bench_dir: str, filename: str = "queries.json") -> List[Dict[str, Any]]:
    """Load benchmark queries from the bench directory."""
    queries_path = Path(bench_dir) / filename
    if not queries_path.exists():
        print(f"No {filename} at {queries_path}")
        return []
    with open(queries_path, "r", encoding="utf-8") as f:
        return json.load(f)


def snapshot_db(live_db: str) -> str:
    """Copy the live DB to a temp snapshot via sqlite3 .backup (P1-7).

    Returns the snapshot path. The caller must delete it afterwards.
    """
    fd, snap = tempfile.mkstemp(prefix="spine-bench-", suffix=".db")
    os.close(fd)
    os.unlink(snap)  # backup() requires the target to not exist
    from spine.index import connect_db
    src = connect_db(live_db)
    dst = sqlite3.connect(snap)
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()
    return snap


def _word_boundary(term: str) -> re.Pattern:
    """Word-boundary regex for a term (P2-11: kills substring false positives)."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", re.IGNORECASE)


def score_query(query: str, must_hit: List[str], may_hit: List[str],
                retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score one query. Word-boundary match for must_hit; substring for may_hit.

    Returns per-term hit/miss map + per-result provenance + hit counts.
    """
    hit_map = {}
    must_hits = 0
    for term in must_hit:
        rx = _word_boundary(term)
        matched = any(rx.search((o.get("content") or "")) for o in retrieved)
        hit_map[term] = matched
        if matched:
            must_hits += 1

    may_map = {}
    may_hits = 0
    for term in may_hit:
        tl = term.lower()
        matched = any(tl in (o.get("content") or "").lower() for o in retrieved)
        may_map[term] = matched
        if matched:
            may_hits += 1

    n_must = len(must_hit)
    recall = must_hits / n_must if n_must else 1.0
    precision = must_hits / len(retrieved) if retrieved else 0.0
    return {
        "must_hit_map": hit_map,
        "may_hit_map": may_map,
        "must_hits": must_hits,
        "may_hits": may_hits,
        "recall": round(recall, 3),
        "precision_at_k": round(precision, 3),
    }


def run_benchmark(queries: List[Dict[str, Any]], config: Any) -> Dict[str, Any]:
    """Run all benchmark queries through recall() and compute scores.

    Returns the full report including per-result provenance and per-term maps.
    """
    from spine.tools import handle_recall
    from spine.embedder import embedder_available

    results = []
    total_precision = 0.0
    total_recall = 0.0
    queries_run = 0

    for q in queries:
        query = q["query"]
        k = q.get("k", 6)

        # Schema: must_hit/may_hit new, expected_hits legacy
        if "must_hit" in q:
            must_hit = q.get("must_hit", [])
            may_hit = q.get("may_hit", [])
        else:
            must_hit = q.get("expected_hits", [])
            may_hit = []

        try:
            raw = handle_recall({"query": query, "k": k}, config)
            resp = json.loads(raw)
            obs = resp.get("results", [])
        except Exception as e:
            obs = []
            resp = {"error": str(e)}

        # Per-result provenance (P1-10)
        provenance = []
        for rank, o in enumerate(obs):
            provenance.append({
                "id": o.get("id", ""),
                "source": o.get("source", "obs"),
                "path": o.get("path", ""),
                "rank": rank,
            })

        sc = score_query(query, must_hit, may_hit, obs)
        total_precision += sc["precision_at_k"]
        total_recall += sc["recall"]
        queries_run += 1

        results.append({
            "query": query,
            "source": q.get("source", ""),
            "k": k,
            "retrieved_count": len(obs),
            "provenance": provenance,
            "must_hit_map": sc["must_hit_map"],
            "may_hit_map": sc["may_hit_map"],
            "must_hits": sc["must_hits"],
            "may_hits": sc["may_hits"],
            "precision_at_k": sc["precision_at_k"],
            "recall": sc["recall"],
        })

    avg_precision = total_precision / queries_run if queries_run else 0.0
    avg_recall = total_recall / queries_run if queries_run else 0.0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "embedder_available": embedder_available(),
        "queries_run": queries_run,
        "scores": {
            "precision_at_k": round(avg_precision, 3),
            "recall": round(avg_recall, 3),
            "f1": round(2 * avg_precision * avg_recall / (avg_precision + avg_recall), 3) if (avg_precision + avg_recall) > 0 else 0.0,
        },
        "results": results,
    }


def structural_gate(db_path: str, bench_dir: str) -> List[str]:
    """P1-5 structural gate: catch misconfiguration before any scoring."""
    problems = []
    live_db = os.path.expanduser(db_path)
    if not os.path.exists(live_db):
        problems.append(f"DB missing: {live_db}")
        return problems

    from spine.index import connect_db
    conn = connect_db(f"file:{live_db}?mode=ro", uri=True)
    try:
        obs = conn.execute(
            "SELECT COUNT(*) FROM observations WHERE status IN ('active','promoted')"
        ).fetchone()[0]
        wiki = conn.execute("SELECT COUNT(*) FROM wiki_chunks").fetchone()[0]
        # Was key='dim', which does not exist -- the row is 'embedding_dim', so
        # this check has always been dead. And it asserted 384, which is wrong
        # since the 2026-08-21 model swap. Compare against the actual vectors
        # instead of a hardcoded number.
        dim = conn.execute(
            "SELECT value FROM dim_meta WHERE key='embedding_dim'").fetchall()
        dim_val = int(dim[0][0]) if dim else None
        widths = {len(b) // 4 for (b,) in conn.execute(
            "SELECT embedding FROM observations WHERE embedding IS NOT NULL LIMIT 200")
            if isinstance(b, (bytes, memoryview))}
        if obs == 0:
            problems.append("obs_count=0 — index empty")
        if wiki == 0:
            problems.append("wiki_chunk_count=0 — wiki not indexed")
        if widths and len(widths) > 1:
            problems.append(f"mixed vector widths {sorted(widths)} — store is corrupt")
        elif widths and dim_val is not None and dim_val not in widths:
            problems.append(f"dim_meta={dim_val} but vectors are {widths.pop()}-dim")
    except sqlite3.OperationalError as e:
        problems.append(f"DB read failed: {e}")
    finally:
        conn.close()

    if not os.path.exists(Path(bench_dir) / "queries.json"):
        problems.append("queries.json missing")
    return problems


def run_controls(config: Any, bench_dir: str) -> Dict[str, Any]:
    """Score-gate on the control set (full config only, P1-5b)."""
    controls = load_queries(bench_dir, CONTROLS_FILE)
    if not controls:
        return {"controls_loaded": False, "pass": True, "note": "controls.json absent — gate skipped"}
    rep = run_benchmark(controls, config)
    # Pass rule: >=50% recall on must_hit terms across all controls
    passed = rep["scores"]["recall"] >= 0.5
    return {
        "controls_loaded": True,
        "pass": passed,
        "recall": rep["scores"]["recall"],
        "n_controls": rep["queries_run"],
    }


def save_report(report: Dict[str, Any], bench_dir: str) -> str:
    """Save benchmark run to bench/run-<date>.json.

    <date> is the LOCAL calendar date. It was UTC, so the weekly cron firing
    at 04:36 SGT on the 30th wrote run-2026-08-29-*.json -- the file never
    matched the cron log or the day the operator looked at. The report's own
    `timestamp` field stays UTC ISO for machine comparison.
    """
    date_str = datetime.now().astimezone().strftime("%Y-%m-%d")
    run_dir = Path(bench_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(run_dir.glob(f"run-{date_str}*.json"))
    run_num = len(existing) + 1
    path = run_dir / f"run-{date_str}-{run_num:02d}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return str(path)


def check_regression(report: Dict[str, Any], bench_dir: str) -> Dict[str, Any]:
    """Compare against baseline. Flag regression >5%."""
    baseline_path = Path(bench_dir) / "baseline.json"
    if not baseline_path.exists():
        return {"baseline_exists": False, "regression": False}

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    current = report["scores"]
    baseline_scores = baseline.get("scores", {})

    regressions = {}
    for metric in ["precision_at_k", "recall", "f1"]:
        current_val = current.get(metric, 0)
        baseline_val = baseline_scores.get(metric, 0)
        if baseline_val > 0:
            pct_change = (current_val - baseline_val) / baseline_val * 100
            regressions[metric] = {
                "current": current_val,
                "baseline": baseline_val,
                "pct_change": round(pct_change, 1),
                "regression": pct_change < -5.0,
            }

    has_regression = any(r.get("regression") for r in regressions.values())

    return {
        "baseline_exists": True,
        "baseline_date": baseline.get("timestamp", "unknown"),
        "metrics": regressions,
        "regression": has_regression,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spine retrieval benchmark")
    parser.add_argument("--bench-dir", default=os.path.expanduser("~/wiki/_memory/bench"),
                        help="Benchmark directory (default: ~/wiki/_memory/bench)")
    parser.add_argument("--set-baseline", action="store_true",
                        help="Save current run as baseline")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON to stdout")
    parser.add_argument("--no-snapshot", action="store_true",
                        help="Run against the live DB (default: temp snapshot, P1-7)")
    args = parser.parse_args()

    # Load config
    try:
        from spine.config import load_spine_config
        config = load_spine_config()
    except Exception:
        config = None

    # Structural gate (P1-5a)
    db_for_gate = getattr(config, "db", "~/.hermes/memory.db") if config else "~/.hermes/memory.db"
    problems = structural_gate(db_for_gate, args.bench_dir)
    for p in problems:
        print(f"❌ GATE: {p}")
    if problems:
        print("Structural gate failed — refusing to run.")
        sys.exit(2)

    # DB snapshot (P1-7) — default on
    snapshot_path = None
    if not args.no_snapshot and config is not None:
        live = os.path.expanduser(getattr(config, "db", "~/.hermes/memory.db"))
        snapshot_path = snapshot_db(live)
        config.db = snapshot_path
        print(f"DB snapshot: {snapshot_path} (live DB untouched)")

    try:
        queries = load_queries(args.bench_dir)
        if not queries:
            print("No queries to run. Create ~/wiki/_memory/bench/queries.json")
            sys.exit(1)

        print(f"Running benchmark: {len(queries)} queries...")
        report = run_benchmark(queries, config)

        # Control score gate (full config only — P1-5b)
        controls = run_controls(config, args.bench_dir)
        report["controls"] = controls
        if controls.get("controls_loaded") and not controls.get("pass"):
            print(f"❌ CONTROL GATE FAILED: recall {controls.get('recall')} < 0.5 — results invalid.")
            sys.exit(2)

        # Save report
        path = save_report(report, args.bench_dir)
        print(f"Report saved: {path}")

        # Check regression
        regression = check_regression(report, args.bench_dir)

        # Print summary
        scores = report["scores"]
        embedder = "✓" if report["embedder_available"] else "✗"
        print(f"\nScores (embedder={embedder}, snapshot={'on' if snapshot_path else 'off'}):")
        print(f"  precision@k: {scores['precision_at_k']:.3f}")
        print(f"  recall:      {scores['recall']:.3f}")
        print(f"  f1:          {scores['f1']:.3f}")

        # Per-source breakdown (derived from provenance)
        src_stats: Dict[str, List[float]] = {}
        for r in report["results"]:
            src = r.get("source", "?")
            src_stats.setdefault(src, []).append(r["recall"])
        if src_stats:
            print("\n  recall by source tag:")
            for src, recalls in sorted(src_stats.items()):
                avg = sum(recalls) / len(recalls)
                print(f"    {src:<12} {avg:.3f} (n={len(recalls)})")

        if regression["baseline_exists"]:
            print(f"\nRegression vs baseline ({regression['baseline_date']}):")
            for metric, info in regression["metrics"].items():
                flag = "⚠️ REGRESSION" if info["regression"] else "✓ OK"
                print(f"  {metric}: {info['current']:.3f} vs {info['baseline']:.3f} ({info['pct_change']:+.1f}%) — {flag}")
            if regression["regression"]:
                print("\n⚠️ REGRESSION >5% DETECTED — block this change.")
                # P1-2 (2026-08-23): this used to print the warning and exit 0,
                # so the weekly cron showed "ok" while recall sat -5.6% below
                # baseline for weeks. The heartbeat only speaks on failures; a
                # bench that exits 0 is a bench that never spoke. Non-zero exit
                # makes the cron run FAIL and the Cron Health Watchdog re-flag it.
                sys.exit(3)

        if args.set_baseline:
            baseline_path = Path(args.bench_dir) / "baseline.json"
            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\nBaseline set: {baseline_path}")

        if args.json:
            print("\n--- JSON ---")
            print(json.dumps(report, indent=2))
    finally:
        if snapshot_path and os.path.exists(snapshot_path):
            os.unlink(snapshot_path)
            print(f"Snapshot cleaned: {snapshot_path}")


if __name__ == "__main__":
    main()
