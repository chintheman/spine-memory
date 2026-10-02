"""Point-in-time reconstruction over the canonical observation log.

Answers "what did the system believe as of date X" by replaying JSONL
patches only up to a cutoff timestamp — the canonical log already carries
full history (base record + ordered patches), this just stops applying
patches partway through instead of applying all of them.

No new dependency, no schema change — reuses the same JSONL structure
`rebuild-index.py` already reads, just filtered by time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .loops import _word_in


def load_observations_as_of(obs_dir: str, as_of: str, profile_filter: str = "") -> List[Dict[str, Any]]:
    """Reconstruct observation state as of a given ISO-8601 timestamp.

    An observation is included only if it existed by `as_of` (created_at <= as_of)
    and wasn't already deleted by then. Only patches with ts <= as_of are applied,
    so a later correction/promotion/deletion has no effect on the reconstructed view.
    """
    obs_path = Path(obs_dir)
    records: List[Dict[str, Any]] = []
    patches: Dict[str, List[Dict[str, Any]]] = {}

    if not obs_path.exists():
        return []

    for jsonl_file in sorted(obs_path.glob("*.jsonl")):
        profile = jsonl_file.stem
        # "*" is the all-profiles scope used by recall/reflect/forget. Compared
        # literally against a filename stem it matched nothing, so recall_at
        # silently returned [] instead of searching everything.
        if profile_filter in ("*", "all"):
            pass
        elif profile_filter and profile != profile_filter:
            continue
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "patch" in rec:
                    if rec.get("ts", "") <= as_of:
                        patches.setdefault(rec["id"], []).append(rec)
                else:
                    records.append(rec)

    result: List[Dict[str, Any]] = []
    for rec in records:
        if rec.get("created_at", "") > as_of:
            continue  # didn't exist yet as of this timestamp
        merged = dict(rec)
        for p in patches.get(rec["id"], []):
            merged.update(p["patch"])
        if merged.get("status") == "deleted":
            continue  # was already deleted by this timestamp
        result.append(merged)

    return result


# Common words that would otherwise let almost any observation "match" almost any
# query — without filtering these, a query with zero real matches still returns
# noise (e.g. matching just "the"/"user") instead of correctly returning nothing.
_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "did", "does", "know", "user",
    "users", "this", "that", "with", "have", "has", "had", "what", "who",
    "when", "where", "how", "system", "about", "you", "your",
}


def _score(query_words: List[str], content: str) -> int:
    content_lower = content.lower()
    return sum(1 for w in query_words if _word_in(w, content_lower))


def recall_as_of(obs_dir: str, query: str, as_of: str, profile_filter: str = "", k: int = 6) -> List[Dict[str, Any]]:
    """Simple keyword-relevance search over the as-of reconstruction.

    No embeddings here deliberately — historical vectors for superseded content
    were never stored, only current state is. Plain keyword overlap is honest
    about what this can and can't do, rather than faking semantic search.

    Requires at least 2 distinct meaningful (non-stopword) query words to match —
    without this floor, a query with genuinely nothing relevant as-of that date
    still returns weakly-overlapping noise instead of correctly returning empty.
    """
    observations = load_observations_as_of(obs_dir, as_of, profile_filter)
    query_words = [w.lower() for w in query.split() if len(w) >= 3 and w.lower() not in _STOPWORDS]
    if not query_words:
        return []
    min_matches = min(2, len(query_words))
    scored = [(_score(query_words, obs.get("content", "")), obs) for obs in observations]
    scored = [(s, obs) for s, obs in scored if s >= min_matches]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [obs for _, obs in scored[:k]]
