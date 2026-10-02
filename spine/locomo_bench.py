#!/usr/bin/env python3
"""LoCoMo-style benchmark for spine memory — adapted to real Hermes data.

Unlike benchmark.py (which checks precision/recall against a fixed
expected-hits list for regression tracking), this mirrors the actual
LoCoMo/LongMemEval methodology used in published memory-system papers
(e.g. the Mem0 evaluation): questions are split into single-hop,
multi-hop, temporal, open-domain, and adversarial categories, and each
answer is scored two ways:

  - F1: token-level precision/recall overlap against a reference answer
        (the standard QA metric LoCoMo reports).
  - LLM-judge: a binary correct/incorrect verdict from an LLM given the
        question, reference answer, and system's answer — this is the
        *primary* metric in the original methodology (papers report the
        LLM-judge score as the headline number, not F1).

Difference from the original LoCoMo: questions are hand-curated from this
installation's own real observations (not LoCoMo's synthetic dialogues),
and the judge model is whatever spine's loop_model is configured to
(DeepSeek), not GPT-4o-mini. Scores are therefore comparable in *shape*
and *method* to published numbers, not a literal apples-to-apples
leaderboard entry — that caveat matters and should stay attached to any
score this produces.

Queries live in ~/wiki/_memory/bench/locomo_style_queries.json.
Output: a report to bench/locomo-run-<date>.json plus a category table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SPINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPINE_DIR.parent))
sys.path.insert(0, str(SPINE_DIR.parent.parent.parent))


def load_queries(bench_dir: str) -> List[Dict[str, Any]]:
    path = Path(bench_dir) / "locomo_style_queries.json"
    if not path.exists():
        print(f"No queries file at {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def f1_score(prediction: str, reference: str) -> float:
    """Standard token-overlap F1 (the same metric LoCoMo/SQuAD-style QA reports)."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: Dict[str, int] = defaultdict(int)
    for t in pred_tokens:
        pred_counts[t] += 1
    ref_counts: Dict[str, int] = defaultdict(int)
    for t in ref_tokens:
        ref_counts[t] += 1
    overlap = sum(min(pred_counts[t], ref_counts[t]) for t in pred_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def llm_judge(question: str, reference: str, prediction: str, model: str, expect_empty: bool = False) -> Dict[str, Any]:
    """Binary correctness verdict from an LLM, mirroring the LoCoMo/Mem0 judging protocol:
    judge sees the question, reference answer, and system's answer, and decides if the
    system's answer is semantically consistent with the reference. For adversarial/
    expect_empty questions, correctness means the system did NOT fabricate an answer."""
    from spine.llm_client import call_llm

    if expect_empty:
        judge_prompt = (
            f"Question: {question}\n"
            f"Correct behavior: the system has no information and should say so, NOT invent an answer.\n"
            f"System's actual answer: {prediction}\n\n"
            "Did the system correctly avoid fabricating information (i.e. it indicated it has "
            "no relevant data, or genuinely returned nothing usable), rather than making something up? "
            'Reply with strict JSON only: {"correct": true|false, "reason": "<one short sentence>"}'
        )
    else:
        judge_prompt = (
            f"Question: {question}\n"
            f"Reference (gold) answer: {reference}\n"
            f"System's actual answer: {prediction}\n\n"
            "Is the system's answer semantically consistent with the reference answer? "
            "It doesn't need to match wording, just the substantive facts. "
            'Reply with strict JSON only: {"correct": true|false, "reason": "<one short sentence>"}'
        )

    # Reasoning models (deepseek-v4-pro) spend tokens on hidden reasoning_content before
    # the visible JSON verdict — 150 was tight enough to sometimes truncate it. One retry
    # on a failed/empty call absorbs transient timeouts under back-to-back API load.
    import time as _time
    response = None
    for attempt in range(3):
        response = call_llm(
            [{"role": "user", "content": judge_prompt}],
            model=model,
            max_tokens=500,
            temperature=0.0,
        )
        if response:
            break
        _time.sleep(2)
    if not response:
        return {"correct": False, "reason": "judge call failed after retries"}

    text = response.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.startswith("```")]
        text = "\n".join(lines)

    # Pull the JSON object out even if there's stray text around it
    match = re.search(r"\{[^{}]*\"correct\"[^{}]*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        verdict = json.loads(candidate)
        return {"correct": bool(verdict.get("correct", False)), "reason": verdict.get("reason", "")}
    except json.JSONDecodeError:
        # Last resort: a truncated response still usually starts with the bool value
        if '"correct": true' in text or '"correct":true' in text:
            return {"correct": True, "reason": "(recovered from truncated judge response)"}
        if '"correct": false' in text or '"correct":false' in text:
            return {"correct": False, "reason": "(recovered from truncated judge response)"}
        return {"correct": False, "reason": f"unparseable judge response: {text[:150]}"}


def _extract_answer(tool: str, raw_response: str) -> str:
    """Pull a comparable answer string out of each tool's differently-shaped JSON."""
    try:
        resp = json.loads(raw_response)
    except json.JSONDecodeError:
        return ""

    if tool == "reflect":
        return resp.get("answer", "")

    # recall / recall_at both return {"results": [...]}
    results = resp.get("results", [])
    if not results:
        return ""
    return " | ".join(r.get("content", "") for r in results[:6])


def run_query(q: Dict[str, Any], config: Any) -> Dict[str, Any]:
    from spine.tools import handle_recall, handle_reflect, handle_recall_at

    tool = q["tool"]
    try:
        if tool == "recall":
            raw = handle_recall({"query": q["question"], "k": 6}, config)
        elif tool == "reflect":
            raw = handle_reflect({"question": q["question"]}, config)
        elif tool == "recall_at":
            raw = handle_recall_at({"query": q["question"], "as_of": q["as_of"], "k": 6}, config)
        else:
            raw = "{}"
    except Exception as e:
        raw = json.dumps({"error": str(e)})

    return {"raw": raw, "answer": _extract_answer(tool, raw)}


def run_benchmark(queries: List[Dict[str, Any]], config: Any, judge_model: str) -> Dict[str, Any]:
    per_query = []
    for q in queries:
        result = run_query(q, config)
        answer = result["answer"]
        expect_empty = q.get("expect_empty", False)

        # For raw recall()/recall_at() (no LLM synthesis), "nothing found" really is an
        # empty string, so the blunt empty-check is meaningful. reflect() always produces
        # a full sentence even when correctly declining ("I don't have that information"),
        # so for reflect()-sourced answers F1 is scored normally against the reference text
        # and the LLM judge (which is told this is an expect-no-fabrication case) carries
        # the real correctness signal instead.
        if expect_empty and q["tool"] in ("recall", "recall_at"):
            f1 = 1.0 if not answer.strip() else 0.0
        else:
            f1 = f1_score(answer, q["reference_answer"])
        judge = llm_judge(q["question"], q["reference_answer"], answer, judge_model, expect_empty=expect_empty)

        per_query.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "system_answer": answer[:300],
            "f1": f1,
            "llm_judge_correct": judge["correct"],
            "llm_judge_reason": judge["reason"],
        })

    by_category: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"f1": [], "judge": []})
    for r in per_query:
        by_category[r["category"]]["f1"].append(r["f1"])
        by_category[r["category"]]["judge"].append(1.0 if r["llm_judge_correct"] else 0.0)

    category_scores = {}
    for cat, vals in by_category.items():
        category_scores[cat] = {
            "n": len(vals["f1"]),
            "f1": round(sum(vals["f1"]) / len(vals["f1"]), 3),
            "llm_judge": round(sum(vals["judge"]) / len(vals["judge"]), 3),
        }

    all_f1 = [r["f1"] for r in per_query]
    all_judge = [1.0 if r["llm_judge_correct"] else 0.0 for r in per_query]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "methodology": "LoCoMo-style (single-hop/multi-hop/temporal/open-domain/adversarial), "
                        "judged by F1 token-overlap + LLM-as-judge binary verdict — adapted from "
                        "published LoCoMo/Mem0 evaluation methodology, run against this installation's "
                        "real observations rather than LoCoMo's synthetic dialogues. Judge model differs "
                        "from the original paper's GPT-4o-mini — scores are comparable in method, not a "
                        "literal leaderboard entry.",
        "judge_model": judge_model,
        "queries_run": len(per_query),
        "overall": {
            "f1": round(sum(all_f1) / len(all_f1), 3) if all_f1 else 0.0,
            "llm_judge": round(sum(all_judge) / len(all_judge), 3) if all_judge else 0.0,
        },
        "by_category": category_scores,
        "results": per_query,
    }


def save_report(report: Dict[str, Any], bench_dir: str) -> str:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_dir = Path(bench_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(run_dir.glob(f"locomo-run-{date_str}*.json"))
    run_num = len(existing) + 1
    path = run_dir / f"locomo-run-{date_str}-{run_num:02d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoCoMo-style spine memory benchmark")
    parser.add_argument("--bench-dir", default=os.path.expanduser("~/wiki/_memory/bench"))
    parser.add_argument("--judge-model", default="deepseek-v4-pro")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    queries = load_queries(args.bench_dir)
    if not queries:
        print("No queries to run.")
        sys.exit(1)

    try:
        from spine.config import load_spine_config
        config = load_spine_config()
    except Exception:
        config = None

    print(f"Running LoCoMo-style benchmark: {len(queries)} queries across "
          f"{len(set(q['category'] for q in queries))} categories...")
    report = run_benchmark(queries, config, args.judge_model)

    path = save_report(report, args.bench_dir)
    print(f"Report saved: {path}\n")

    print(f"{'Category':<15} {'N':>3} {'F1':>7} {'LLM-judge':>10}")
    print("-" * 38)
    for cat, scores in sorted(report["by_category"].items()):
        print(f"{cat:<15} {scores['n']:>3} {scores['f1']:>7.3f} {scores['llm_judge']:>10.3f}")
    print("-" * 38)
    print(f"{'OVERALL':<15} {report['queries_run']:>3} {report['overall']['f1']:>7.3f} {report['overall']['llm_judge']:>10.3f}")

    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
