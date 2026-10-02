#!/usr/bin/env python3
"""Phase 0 retrieval gate — run the eval set against spine and report pass/fail.

Usage:
    python3 eval_run.py                     # run, print table
    python3 eval_run.py --save before.json  # also write raw results
    python3 eval_run.py --compare before.json   # diff against a saved run

A case passes when any of its `expect_any` phrases appears (case-insensitively)
in the content of the top-k hybrid recall results. This is deliberately a
coarse gate: it measures "did the right memory surface at all", not ranking
quality. Nothing in spine's retrieval path ships without a before/after run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # .../plugins/memory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # hermes-agent root (for agent.* imports)

from spine.index import MemoryIndex  # noqa: E402
from spine import embedder  # noqa: E402

DB = os.path.expanduser("~/.hermes/memory.db")
EVAL = os.path.join(HERE, "eval_set.json")


def judge(case: Dict[str, Any], hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score one case against one result list.

    Module-level and importable ON PURPOSE: the first regression tests
    reimplemented this inline, so reverting the greedy sort left them green.
    """
    docs = [(h.get("content") or "").lower() for h in hits]
    blob = "\n".join(docs)
    hop = case.get("hop", "single")

    if hop in ("multi", "conj"):
        needles = case["expect_all"]
        matched = [n for n in needles if n.lower() in blob]
        # Count documents that each contribute a needle the OTHERS do not.
        per_doc = [{n for n in needles if n.lower() in d} for d in docs]
        covered, distinct = set(), 0
        # DESCENDING. Greedy set-cover takes the largest set first; ascending
        # counts {a} then {a,b} as two contributors, so one document holding
        # every needle plus one unrelated mention still passed min_sources=2.
        for owned in sorted(per_doc, key=len, reverse=True):
            fresh = owned - covered
            if fresh:
                covered |= fresh
                distinct += 1
        need = case.get("min_sources", 2) if hop == "multi" else 0
        passed = len(matched) == len(needles) and distinct >= need
        n_sources = distinct
    else:
        matched = [p for p in case["expect_any"] if p.lower() in blob]
        passed = bool(matched)
        n_sources = sum(1 for d in docs
                        if any(p.lower() in d for p in case["expect_any"]))
        need = 0
    return {"hop": hop, "passed": passed, "matched": matched,
            "n_sources": n_sources, "need_sources": need if hop == "multi" else 0}


def run(profile: str = "*") -> Dict[str, Any]:
    spec = json.load(open(EVAL, encoding="utf-8"))
    k = spec.get("k", 6)
    cases = spec["cases"]

    idx = MemoryIndex(DB)
    idx.open()

    have_embedder = embedder.embedder_available()
    results: List[Dict[str, Any]] = []

    for case in cases:
        q = case["q"]
        t0 = time.perf_counter()
        qvec = embedder.embed_single(q) if have_embedder else None
        t_embed = time.perf_counter() - t0

        t0 = time.perf_counter()
        hits = idx.search_hybrid(q, qvec or None, profile=profile, k=k)
        t_search = time.perf_counter() - t0

        verdict = judge(case, hits)

        results.append({
            "id": case["id"],
            "q": q,
            "hop": verdict["hop"],
            "passed": verdict["passed"],
            "matched": verdict["matched"],
            "n_sources": verdict["n_sources"],
            "need_sources": verdict["need_sources"],
            "search_ms": round(t_search * 1000, 1),
            "embed_ms": round(t_embed * 1000, 1),
            "n_hits": len(hits),
            "n_wiki": sum(1 for h in hits if h.get("source") == "wiki"),
            "top": [
                {"src": h.get("source") or "obs",
                 "content": (h.get("content") or "")[:110]}
                for h in hits
            ],
        })

    idx.close()
    passed = sum(1 for r in results if r["passed"])
    by_hop = {}
    for hop in ("single", "conj", "multi"):
        sub = [r for r in results if r["hop"] == hop]
        by_hop[hop] = {"passed": sum(1 for r in sub if r["passed"]), "total": len(sub)}
    if not any(v["total"] for v in by_hop.values()):
        raise RuntimeError("eval set produced no scored cases")
    return {
        "profile": profile,
        "embedder": have_embedder,
        "k": k,
        "passed": passed,
        "by_hop": by_hop,
        "total": len(results),
        "median_search_ms": sorted(r["search_ms"] for r in results)[len(results) // 2],
        "cases": results,
    }


def validate(verbose: bool = True) -> List[str]:
    """A multi-hop case is only meaningful if NO single document answers it.

    11 of 12 cases were mislabelled multi-hop while a single memory held every
    needle, so the section scored retrieval that never had to connect anything.
    """
    import sqlite3 as _sq
    spec = json.load(open(EVAL, encoding="utf-8"))
    con = _sq.connect(DB)
    con.execute("PRAGMA query_only = ON")
    docs = [r[0].lower() for r in con.execute("SELECT content FROM observations")]
    docs += [(r[0] + ": " + r[1]).lower()
             for r in con.execute("SELECT title, content FROM wiki_chunks")]
    con.close()
    bad = []
    for c in spec["cases"]:
        if c.get("hop") != "multi":
            continue
        ns = [n.lower() for n in c["expect_all"]]
        if any(all(n in d for n in ns) for d in docs):
            bad.append(c["id"])
    if verbose:
        for cid in bad:
            print(f"MISLABELLED: {cid} — a single document contains every needle; "
                  f"this cannot test multi-hop. Reclassify as conj or change the needles.")
        print(f"{len(bad)} invalid multi-hop case(s)")
    return bad


def show(rep: Dict[str, Any]) -> None:
    print(f"\nprofile={rep['profile']}  embedder={rep['embedder']}  k={rep['k']}")
    for hop in ("single", "conj", "multi"):
        sub = [r for r in rep["cases"] if r["hop"] == hop]
        if not sub:
            continue
        b = rep["by_hop"][hop]
        print(f"\n{hop.upper()}-HOP  {b['passed']}/{b['total']}")
        print(f"{'':2} {'case':26} {'result':6} {'ms':>7} {'src':>4}  matched")
        print("-" * 82)
        for r in sub:
            mark = "PASS" if r["passed"] else "FAIL"
            got = ", ".join(r["matched"])[:30] or "-"
            src = (f"{r['n_sources']}/{r['need_sources']}" if hop == "multi"
                   else str(r["n_sources"]))
            print(f"{'':2} {r['id']:26} {mark:6} {r['search_ms']:>7} {src:>4}  {got}")
        print("-" * 82)
    b = rep["by_hop"]
    parts = "   ".join(f"{h} {b[h]['passed']}/{b[h]['total']}"
                       for h in ("single", "conj", "multi") if b.get(h, {}).get("total"))
    print(f"\n   TOTAL {rep['passed']}/{rep['total']}   {parts}   "
          f"median search {rep['median_search_ms']} ms")
    wiki = sum(r["n_wiki"] for r in rep["cases"])
    print(f"   wiki chunks appearing in results: {wiki}\n")


def compare(new: Dict[str, Any], old_path: str) -> None:
    old = json.load(open(old_path, encoding="utf-8"))
    om = {c["id"]: c for c in old["cases"]}
    print(f"\n{'case':24} {'before':>8} {'after':>8}   {'ms before':>10} {'ms after':>9}")
    print("-" * 68)
    regressions = []
    for c in new["cases"]:
        o = om.get(c["id"])
        b = ("PASS" if o["passed"] else "FAIL") if o else "n/a"
        a = "PASS" if c["passed"] else "FAIL"
        flag = ""
        if o and o["passed"] and not c["passed"]:
            flag = "  <-- REGRESSION"
            regressions.append(c["id"])
        elif o and not o["passed"] and c["passed"]:
            flag = "  <-- fixed"
        ob = f"{o['search_ms']}" if o else "-"
        print(f"{c['id']:24} {b:>8} {a:>8}   {ob:>10} {c['search_ms']:>9}{flag}")
    print("-" * 68)
    print(f"{old['passed']}/{old['total']}  ->  {new['passed']}/{new['total']}")
    if regressions:
        print(f"REGRESSIONS: {', '.join(regressions)}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # "*" to match check_eval, which always runs run("*"). With agent:main here,
    # the documented `--save` refresh wrote an agent:main baseline (14/29) that a
    # "*" run (29/29) can never fall below, permanently disabling the gate.
    ap.add_argument("--profile", default="*")
    ap.add_argument("--validate", action="store_true",
                    help="check every multi-hop case really needs >1 document, then exit")
    ap.add_argument("--save")
    ap.add_argument("--compare")
    a = ap.parse_args()

    if a.validate:
        sys.exit(1 if validate() else 0)
    rep = run(a.profile)
    show(rep)
    if a.save:
        json.dump(rep, open(a.save, "w", encoding="utf-8"), indent=2)
        print(f"saved -> {a.save}")
    if a.compare:
        compare(rep, a.compare)
