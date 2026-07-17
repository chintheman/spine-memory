#!/usr/bin/env python3
"""Benchmark harness for spine recall — spec §6.

Runs a set of fixed queries with expected-hit annotations through recall(),
measures precision@k and recall, and records scores for trend tracking.

Queries live in ~/wiki/_memory/bench/queries.json:
[
  {
    "query": "what are the user's communication preferences?",
    "expected_hits": ["prefers concise replies", "Telegram-first"],
    "k": 6
  },
  ...
]

Output: benchmark run report to bench/run-<date>.json and a summary line.
Regression >5% from baseline blocks a change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add the spine package to path
SPINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPINE_DIR.parent.parent.parent))


def load_queries(bench_dir: str) -> List[Dict[str, Any]]:
    """Load benchmark queries from the bench directory."""
    queries_path = Path(bench_dir) / "queries.json"
    if not queries_path.exists():
        print(f"No queries file at {queries_path}")
        return []
    with open(queries_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark(queries: List[Dict[str, Any]], config: Any) -> Dict[str, Any]:
    """Run all benchmark queries through recall() and compute scores.

    Returns:
      {
        "timestamp": "...",
        "queries": [...results...],
        "scores": {"precision_at_k": 0.85, "recall": 0.72, ...},
        "embedder_available": true
      }
    """
    from spine.tools import handle_recall
    from spine.embedder import embedder_available

    results = []
    total_precision = 0.0
    total_recall = 0.0
    queries_run = 0

    for q in queries:
        query = q["query"]
        expected = q.get("expected_hits", [])
        k = q.get("k", 6)

        try:
            raw = handle_recall({"query": query, "k": k}, config)
            resp = json.loads(raw)
            obs = resp.get("results", [])
        except Exception as e:
            obs = []
            resp = {"error": str(e)}

        # Compute precision@k and recall
        retrieved_content = [o.get("content", "") for o in obs]
        hits = 0
        for expected_term in expected:
            for content in retrieved_content:
                if expected_term.lower() in content.lower():
                    hits += 1
                    break

        precision = hits / len(obs) if obs else 0.0
        recall = hits / len(expected) if expected else 1.0

        total_precision += precision
        total_recall += recall
        queries_run += 1

        results.append({
            "query": query,
            "expected_hits": expected,
            "retrieved_count": len(obs),
            "retrieved_content": retrieved_content[:k],
            "hits": hits,
            "precision_at_k": round(precision, 3),
            "recall": round(recall, 3),
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


def save_report(report: Dict[str, Any], bench_dir: str) -> str:
    """Save benchmark run to bench/run-<date>.json."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_dir = Path(bench_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Increment run counter
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
    args = parser.parse_args()

    # Load queries
    queries = load_queries(args.bench_dir)
    if not queries:
        print("No queries to run. Create ~/wiki/_memory/bench/queries.json")
        sys.exit(1)

    # Load config
    try:
        from spine.config import load_spine_config
        config = load_spine_config()
    except Exception:
        config = None

    print(f"Running benchmark: {len(queries)} queries...")
    report = run_benchmark(queries, config)

    # Save report
    path = save_report(report, args.bench_dir)
    print(f"Report saved: {path}")

    # Check regression
    regression = check_regression(report, args.bench_dir)

    # Print summary
    scores = report["scores"]
    embedder = "✓" if report["embedder_available"] else "✗"
    print(f"\nScores (embedder={embedder}):")
    print(f"  precision@{queries[0].get('k',6)}: {scores['precision_at_k']:.3f}")
    print(f"  recall:          {scores['recall']:.3f}")
    print(f"  f1:              {scores['f1']:.3f}")

    if regression["baseline_exists"]:
        print(f"\nRegression vs baseline ({regression['baseline_date']}):")
        for metric, info in regression["metrics"].items():
            flag = "⚠️ REGRESSION" if info["regression"] else "✓ OK"
            print(f"  {metric}: {info['current']:.3f} vs {info['baseline']:.3f} ({info['pct_change']:+.1f}%) — {flag}")
        if regression["regression"]:
            print("\n⚠️ REGRESSION >5% DETECTED — block this change.")

    if args.set_baseline:
        baseline_path = Path(args.bench_dir) / "baseline.json"
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nBaseline set: {baseline_path}")

    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
