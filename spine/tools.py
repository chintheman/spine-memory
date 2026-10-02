"""Spine tool schemas and handlers — spec §4.

Five tools exposed to the agent:
- remember()  — write gate with two-signal secrets detector + dedupe
- recall()    — hybrid FTS5+vec retrieval with auto-degradation
- reflect()   — LLM synthesis over recalled observations
- forget()    — discovery → per-entry confirmation → ID deletion
- explain()   — provenance chain for a single observation (JSONL time-travel)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import SpineConfig
from .jsonl_writer import JSONLWriter
from .secrets_detector import detect_secrets
from .embedder import embedder_available, embed_single, embed
from .index import MemoryIndex, _fts5_safe_query, DEFAULT_K

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Tool schemas (OpenAI function-calling format)
# ═══════════════════════════════════════════════════════════════════════

REMEMBER_SCHEMA = {
    "name": "remember",
    "description": "Save a durable observation to spine memory. Events and task logs are rejected — use session_search() for those.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The observation content (present-tense, atomic claim, ≤500 chars)"},
            "type": {"type": "string", "enum": ["fact", "preference", "pattern", "correction", "identity"], "description": "Observation type"},
            "profile": {"type": "string", "description": "Profile name (default: agent:main)"},
            "topics": {"type": "array", "items": {"type": "string"}, "description": "Optional provenance topics (e.g. ['#11'])"},
            "epistemic": {"type": "string", "enum": ["extracted", "inferred"], "description": "How was this observed?"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Confidence 0-1"},
        },
        "required": ["content", "type"],
    },
}

RECALL_SCHEMA = {
    "name": "recall",
    "description": "Recall observations from spine memory using hybrid (keyword + semantic) search. Falls back to keyword-only if embedder unavailable.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "profile": {"type": "string", "description": "Profile filter (default: all profiles)"},
            "k": {"type": "integer", "minimum": 1, "maximum": 25, "description": "Number of results (default: 6)"},
        },
        "required": ["query"],
    },
}

RECALL_AT_SCHEMA = {
    "name": "recall_at",
    "description": "Time-travel recall — what did spine memory believe as of a past date, ignoring anything learned, corrected, or deleted after that point. Uses plain keyword matching over the historical log (not semantic search — vectors are only stored for current state), so phrase it with the actual words likely used at the time.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (keyword-based)"},
            "as_of": {"type": "string", "description": "ISO-8601 timestamp or date (e.g. '2026-07-18' or '2026-07-18T00:00:00+00:00') — reconstruct state as of this point in time"},
            "profile": {"type": "string", "description": "Profile filter (default: all profiles)"},
            "k": {"type": "integer", "minimum": 1, "maximum": 25, "description": "Number of results (default: 6)"},
        },
        "required": ["query", "as_of"],
    },
}

REFLECT_SCHEMA = {
    "name": "reflect",
    "description": "Deep analysis over recalled spine observations. Synthesizes patterns and answers questions about what the system knows.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "What to analyze or answer about the stored observations"},
            "profile": {"type": "string", "description": "Profile filter (default: all profiles)"},
        },
        "required": ["question"],
    },
}

FORGET_SCHEMA = {
    "name": "forget",
    "description": "Find and delete observations from spine memory. Discovery phase shows full-text matches with per-entry confirmation. Deletion is by exact ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "term": {"type": "string", "description": "Search term to find observations for deletion"},
            "action": {"type": "string", "enum": ["discover", "delete"], "description": "discover matches, or delete by exact id"},
            "obs_id": {"type": "string", "description": "Observation ID to delete (required when action=delete)"},
        },
        "required": [],
    },
}

EXPLAIN_SCHEMA = {
    "name": "explain",
    "description": "Show the full provenance chain for an observation — when and how it was created, confirmed, promoted, superseded, or contradicted. Traces the JSONL history for a single observation ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "obs_id": {"type": "string", "description": "Observation ID to explain (full 26-char ID or short prefix)"},
            "profile": {"type": "string", "description": "Profile filter (default: all profiles)"},
        },
        "required": ["obs_id"],
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _as_list(value: Any) -> List[Any]:
    """Coerce a scalar / None / list into a list, for list-typed record fields."""
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v not in (None, "")]
    if isinstance(value, (tuple, set)):
        return [v for v in value if v not in (None, "")]
    return [value] if value != "" else []


def _now_iso() -> str:
    """ISO 8601 timestamp in SGT."""
    return datetime.now(timezone.utc).isoformat()


def _generate_id() -> str:
    """Generate a ULID-like ID (26-char timestamp + random)."""
    import random
    import string

    ts = int(time.time() * 1000)
    ts_part = _encode_base32(ts, 10)  # 10 chars for timestamp
    rand_part = "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
    return ts_part + rand_part


def _encode_base32(n: int, length: int) -> str:
    """Encode integer to Crockford base32 string of given length."""
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    result = []
    for _ in range(length):
        result.append(alphabet[n % 32])
        n //= 32
    return "".join(reversed(result))


def _get_writer(config: SpineConfig, profile: str = "agent:main") -> JSONLWriter:
    """Get JSONL writer for a profile's observations file.

    A wildcard is a READ scope. Writing one would create a file literally named
    "*.jsonl" and the record would be invisible to every rebuild.
    """
    if profile in ("*", "all"):
        # Raise. Silently substituting a default log meant handle_remember, whose
        # schema takes a free-form profile, could write {"profile": "*"} into
        # agent:main.jsonl: a row consolidation never touches, recall by concrete
        # profile never sees, forget cannot reach, and whose status patches go to
        # a nonexistent observations/*.jsonl. A loud failure is the correct
        # outcome; read paths that legitimately span profiles must not ask for a
        # writer at all.
        raise ValueError("refusing to write to a wildcard profile; resolve it first")
    obs_dir = os.path.join(config.canonical_root, "observations")
    return JSONLWriter(os.path.join(obs_dir, f"{profile}.jsonl"))


def _get_index(config: SpineConfig) -> MemoryIndex:
    """Open and return the index, creating if needed."""
    idx = MemoryIndex(config.db)
    idx.open()
    return idx


# ═══════════════════════════════════════════════════════════════════════
# Tool handlers
# ═══════════════════════════════════════════════════════════════════════

def handle_remember(args: Dict[str, Any], config: SpineConfig) -> str:
    """remember() — write gate with secrets check + dedupe (§4.1)."""
    content = args["content"]
    obs_type = args.get("type", "fact")
# Caller-supplied and free-form in the schema. A wildcard is a read scope; a
    # write must name one profile.
    profile = args.get("profile", "agent:main")
    if profile in ("*", "all"):
        return json.dumps({"error": "profile must name a single profile, not a wildcard"})

    # Reject events/logs
    if obs_type in ("event", "log"):
        return json.dumps({
            "error": "Events and task logs are not suitable for spine memory. Use session_search() to find them in transcripts.",
            "rejected": True,
        })

    # Two-signal secrets detection
    secret_result = detect_secrets(content)
    if secret_result.get("blocked"):
        return json.dumps({
            "error": f"Content blocked by secrets detector: {secret_result.get('detail', 'unknown pattern')}",
            "rejected": True,
            "blocked_by": "secrets_detector",
        })
    if secret_result.get("held_for_review"):
        return json.dumps({
            "held": True,
            "tokens": secret_result.get("tokens", []),
            "message": "Content held for review — contains high-entropy tokens. Use show/discard to proceed.",
        })

    # Build observation record
    obs_id = _generate_id()
    now = _now_iso()
    confidence = args.get("confidence", 0.5)

    record = {
        "id": obs_id,
        "profile": profile,
        "type": obs_type,
        "epistemic": args.get("epistemic", "extracted"),
        "content": content,
        "confidence": confidence,
        "confirmations": 1,
        # Was hardcoded to []. The observer LLM returns a verbatim `quote`
        # supporting every observation and it was being dropped on the floor,
        # leaving the evidence column permanently empty — so nothing could
        # answer "why do we believe this?". handle_explain already reads it.
        "evidence": _as_list(args.get("evidence")),
        "status": "active",
        "supersedes": None,
        "contradicts": [],
        "topics": args.get("topics", []),
        "created_at": now,
        "last_confirmed": now,
        "last_retrieved": None,
    }

    # Dedupe check (cosine > 0.92 on existing actives)
    writer = _get_writer(config, profile)
    existing = writer.read_all()
    for existing_rec in existing:
        if existing_rec.get("id") == obs_id or "patch" in existing_rec:
            continue
        if (
            existing_rec.get("profile") == profile
            and existing_rec.get("type") == obs_type
            and existing_rec.get("content", "").strip().lower() == content.strip().lower()
        ):
            # Near-duplicate — increment confirmations via patch
            patch = {
                "id": existing_rec["id"],
                "patch": {
                    "confirmations": existing_rec.get("confirmations", 1) + 1,
                    "last_confirmed": now,
                },
                "ts": now,
            }
            writer.append(patch)

            # Update index
            try:
                idx = _get_index(config)
                idx.conn.execute(
                    "UPDATE observations SET confirmations=?, last_confirmed=? WHERE id=?",
                    (existing_rec.get("confirmations", 1) + 1, now, existing_rec["id"]),
                )
                idx.conn.commit()
                idx.close()
            except Exception as e:
                print(f"⚠️  spine: index update failed for {existing_rec['id']}: {e}", file=sys.stderr)

            return json.dumps({"success": True, "deduplicated": True, "id": existing_rec["id"], "confirmations_now": existing_rec.get("confirmations", 1) + 1})

    # Write new observation
    writer.append(record)

    # Index it
    embedding = None
    if embedder_available():
        try:
            embedding = embed_single(content)
        except Exception:
            pass

    try:
        idx = _get_index(config)
        idx.upsert_observation(record, embedding)
        idx.conn.commit()
        idx.close()
    except Exception as e:
        # Index failure is non-fatal for write (JSONL already has it) — log and continue
        print(f"⚠️  spine: index upsert failed for {obs_id}: {e}", file=sys.stderr)

    return json.dumps({"success": True, "id": obs_id, "indexed": embedding is not None})


def handle_recall(args: Dict[str, Any], config: SpineConfig) -> str:
    """recall() — hybrid retrieval with auto-degradation (§4.2)."""
    query = args["query"]
    # Read paths span EVERY profile by default: the whole point of importing
    # agent:claude-code was that one recall should reach both stores. Writes
    # and consolidation stay pinned to a single profile.
    profile = args.get("profile", "*")
    # Depth follows DEFAULT_K (raised 20 -> 25 on 2026-09-14, see index.py) so the
    # live path and the eval cannot drift apart again; the cap keeps a single
    # explicit request bounded.
    k = min(args.get("k", DEFAULT_K), 25)

    idx = _get_index(config)

    query_embedding = None
    embedder_ok = False
    if embedder_available():
        try:
            query_embedding = embed_single(query)
            embedder_ok = True
        except Exception:
            pass

    results = idx.search_hybrid(query, query_embedding, profile=profile, k=k)

    # Touch last_retrieved
    now = _now_iso()
    for r in results:
        idx.touch_retrieved(r["id"], now)
    idx.conn.commit()

    idx.close()

    note = ""
    if not embedder_ok:
        note = "Semantic search unavailable — keyword results only."

    return json.dumps({"results": results, "count": len(results), "note": note})


def handle_recall_at(args: Dict[str, Any], config: SpineConfig) -> str:
    """recall_at() — point-in-time recall over the canonical log (no schema/index involved)."""
    from .temporal import recall_as_of
    import os as _os

    query = args["query"]
    as_of = args["as_of"]
    # Same all-profiles default as recall/reflect/forget. Left at agent:main it
    # silently missed agent:claude-code, half the store.
    profile = args.get("profile", "*")
    k = min(args.get("k", 20), 25)

    obs_dir = _os.path.join(_os.path.expanduser(config.canonical_root), "observations")
    results = recall_as_of(obs_dir, query, as_of, profile_filter=profile, k=k)

    return json.dumps({
        "results": results,
        "count": len(results),
        "as_of": as_of,
        "note": "Keyword match over historical log — not semantic search. Reflects state as it existed at as_of, ignoring later corrections/deletions.",
    })


def handle_reflect(args: Dict[str, Any], config: SpineConfig) -> str:
    """reflect() — recall k=25 then LLM-synthesize an answer (§4.3).

    Recalls top-25 observations, then calls the LLM to produce a coherent
    synthesis citing specific observation IDs.
    """
    query = args["question"]
    # Read paths span EVERY profile by default: the whole point of importing
    # agent:claude-code was that one recall should reach both stores. Writes
    # and consolidation stay pinned to a single profile.
    profile = args.get("profile", "*")

    idx = _get_index(config)

    query_embedding = None
    if embedder_available():
        try:
            query_embedding = embed_single(query)
        except Exception:
            pass

    results = idx.search_hybrid(query, query_embedding, profile=profile, k=25)
    idx.close()

    if not results:
        return json.dumps({"answer": "No relevant observations found.", "observations": 0})

    # Format observations for the LLM
    obs_text = "\n".join(
        f"[{r['id'][:8]}] ({r['type']}, {r['epistemic']}, conf={r['confidence']:.2f}) {r['content']}"
        for r in results
    )

    # Call LLM for synthesis
    from .llm_client import call_llm

    model = getattr(config, "loop_model", "deepseek-v4-pro")
    messages = [
        {"role": "system", "content": "You are a memory analyst. Answer the user's question using ONLY the cited observations. Reference observation IDs in your answer (e.g., '[01JABC12]'). If observations conflict, note it. If there isn't enough data, say so."},
        {"role": "user", "content": f"Question: {query}\n\nObservations:\n{obs_text}\n\nSynthesize an answer:"},
    ]

    # deepseek-v4-pro is a reasoning model — it spends tokens on hidden reasoning_content
    # before producing visible content. With many observations to synthesize, a low budget
    # can be entirely consumed by reasoning, leaving an empty answer. 500 was too tight.
    answer = call_llm(messages, model=model, max_tokens=2000, temperature=0.3)

    if answer is None:
        # Fallback: return observations without synthesis
        citations = [f"[{r['id'][:8]}] ({r['type']}) {r['content']}" for r in results]
        return json.dumps({
            "question": query,
            "answer": "(LLM unavailable — raw observations below)",
            "observations": citations,
            "count": len(citations),
        })

    return json.dumps({
        "question": query,
        "answer": answer,
        "observations_sourced": len(results),
    })


def handle_forget(args: Dict[str, Any], config: SpineConfig) -> str:
    """forget() — discovery → per-entry confirmation → ID deletion (§4.4)."""
    term = args.get("term", "")
    action = args.get("action", "discover")
    # Recall and reflect default to every profile, so a memory can be found
    # in agent:claude-code and then reported "not found" by a forget that
    # only looked at agent:main. Same scope for find and delete.
    profile = args.get("profile", "*")

    idx = _get_index(config)
    # The writer is built AFTER the row's real profile is known: `profile` may
    # be the "*" read scope, and a wildcard has no JSONL to write to.
    writer = None

    if action == "discover":
        if not term:
            idx.close()
            return json.dumps({"error": "term required for action=discover"})
        # FTS5 fuzzy search
        results = idx.search_fts(term, profile=profile, limit=20)
        idx.close()

        entries = []
        for r in results:
            entries.append({
                "id": r["id"],
                "content": r["content"],
                "type": r["type"],
                "confidence": r["confidence"],
                "status": r["status"],
                "action_button": f"forget: {r['id'][:8]}... — confirm delete with action=delete",
            })

        return json.dumps({"matches": len(entries), "entries": entries, "instruction": "Review matches. To delete, call forget with action=delete and the exact obs_id."})

    elif action == "delete":
        obs_id = args.get("obs_id", "")
        if not obs_id:
            idx.close()
            return json.dumps({"error": "obs_id required for action=delete"})

        # Verify the observation exists
        # "*" means every profile. The literal comparison below matched a
        # profile actually named "*", of which there are none, so widening the
        # default silently broke delete for every row in the store.
        if profile in ("*", "all"):
            row = idx.conn.execute(
                "SELECT id, content, profile FROM observations WHERE id=?",
                (obs_id,),
            ).fetchone()
        else:
            row = idx.conn.execute(
                "SELECT id, content, profile FROM observations"
                " WHERE id=? AND profile IN (?, 'shared')",
                (obs_id, profile),
            ).fetchone()

        if row is None:
            idx.close()
            return json.dumps({"error": f"Observation {obs_id} not found"})

        content = row[1]
        # Resolve the concrete profile before writing: _get_writer("*") would
        # build JSONLWriter(observations/*.jsonl) and the tombstone would not
        # survive a rebuild.
        profile = row[2] or "agent:main"
        writer = _get_writer(config, profile)

        # Tombstone the SOURCE OF TRUTH first. The reverse order deleted the row
        # from the derived index and, if the append then failed, left it 'active'
        # in the canonical JSONL -- so the next rebuild-index.py resurrected a
        # memory the user had explicitly asked to forget. Demote already got this
        # ordering right; this path had it inverted.
        now = _now_iso()
        tombstone = {
            "id": obs_id,
            "patch": {"status": "deleted", "deleted_at": now},
            "ts": now,
        }
        writer.append(tombstone)

        # Only now drop it from the derived index.
        idx.delete_observation(obs_id)
        idx.conn.commit()
        idx.close()

        # Git commit the removal (spec §4.4)
        import subprocess as _subprocess
        import os as _os
        # Same filename the writer actually uses. This had `profile.replace(':','_')`,
        # producing observations/agent_main.jsonl, which has never existed -- so
        # the exists() check was always False and no forget was ever committed.
        jsonl_path = _os.path.expanduser(
            _os.path.join(config.canonical_root, "observations", f"{profile}.jsonl")
        )
        if _os.path.exists(jsonl_path):
            try:
                _subprocess.run(
                    ["git", "-C", str(_os.path.dirname(_os.path.dirname(jsonl_path))),
                     "add", jsonl_path],
                    capture_output=True, timeout=10,
                )
                # Pathspec is load-bearing. Without it this commits EVERYTHING
                # already staged in ~/wiki -- consolidation reports, archive
                # backups, index files -- under a "forget(...)" message. Round 3
                # made this path live for the first time, so it had never fired.
                _subprocess.run(
                    ["git", "-C", str(_os.path.dirname(_os.path.dirname(jsonl_path))),
                     "commit", "-m", f"forget({obs_id[:8]}): removed observation",
                     "--", jsonl_path],
                    capture_output=True, timeout=10,
                )
            except Exception:
                pass  # Non-fatal — deletion succeeded even if git commit fails

        # Scrub session DB for mentions
        scrub_count = _scrub_session_db(obs_id, content, config)

        return json.dumps({
            "success": True,
            "deleted_id": obs_id,
            "session_db_matches": scrub_count,
            "note": "Observation removed from current state. If previously committed to git, it persists in git history — BFG Repo-Cleaner is the nuclear option (see memory-architecture skill).",
        })
    idx.close()
    return json.dumps({"error": f"Unknown action: {action}"})


def handle_explain(args: Dict[str, Any], config: SpineConfig) -> str:
    """explain() — trace the full JSONL provenance chain for an observation.

    Walks the JSONL history to reconstruct creation, confirmations,
    promotion, contradictions, and patches for a single observation ID.
    """
    obs_id = args["obs_id"]
    profile = args.get("profile", "agent:main")

    writer = _get_writer(config, profile)
    all_records = writer.read_all()

    # --- Phase 1: find all records touching this ID ---
    timeline: List[Dict[str, Any]] = []
    found_original = False

    for rec in all_records:
        rid = rec.get("id", "")
        patch_target = rec.get("patch", {})

        # Direct match (the original record or a tombstone/patch targeting this ID)
        if rid == obs_id:
            if "patch" in rec:
                # This is a patch on the observation itself
                timeline.append({
                    "event": "patch",
                    "ts": rec.get("ts", ""),
                    "changes": rec.get("patch", {}),
                })
            else:
                # This is the original creation record
                timeline.append({
                    "event": "created",
                    "ts": rec.get("created_at", rec.get("ts", "")),
                    "type": rec.get("type"),
                    "epistemic": rec.get("epistemic"),
                    "content": rec.get("content"),
                    "confidence": rec.get("confidence"),
                    "topics": rec.get("topics"),
                    "profile": rec.get("profile"),
                })
                found_original = True

        # This record supersedes or contradicts the target
        if rec.get("supersedes") == obs_id:
            timeline.append({
                "event": "superseded_by",
                "ts": rec.get("created_at", rec.get("ts", "")),
                "replacement_id": rid,
                "replacement_content": rec.get("content", "")[:120],
            })
        if obs_id in (rec.get("contradicts") or []):
            timeline.append({
                "event": "contradicted_by",
                "ts": rec.get("created_at", rec.get("ts", "")),
                "contradictor_id": rid,
                "contradictor_content": rec.get("content", "")[:120],
            })

    if not found_original and not timeline:
        # Try prefix match
        for rec in all_records:
            if rec.get("id", "").startswith(obs_id) and "patch" not in rec:
                obs_id = rec["id"]
                return handle_explain({"obs_id": obs_id, "profile": profile}, config)

        return json.dumps({
            "error": f"No observation found for id or prefix '{obs_id}'",
            "hint": "Use recall() to find observation IDs, then explain(id) to trace their history.",
        })

    # --- Phase 2: build summary ---
    timeline.sort(key=lambda e: e.get("ts", ""))

    # Reconstruct current state by replaying patches
    current = {"id": obs_id}
    for event in timeline:
        if event["event"] == "created":
            current.update({
                "content": event.get("content", ""),
                "type": event.get("type"),
                "epistemic": event.get("epistemic"),
                "confidence": event.get("confidence"),
                "topics": event.get("topics"),
                "created_at": event.get("ts"),
            })
        elif event["event"] == "patch":
            current.update(event.get("changes", {}))

    # Count events
    events_summary = {}
    for e in timeline:
        events_summary[e["event"]] = events_summary.get(e["event"], 0) + 1

    return json.dumps({
        "obs_id": obs_id,
        "profile": profile,
        "current_state": {
            "content": current.get("content", ""),
            "type": current.get("type"),
            "epistemic": current.get("epistemic"),
            "confidence": current.get("confidence"),
            "status": current.get("status", "active"),
            "confirmations": current.get("confirmations", 1),
            "topics": current.get("topics", []),
            "created_at": current.get("created_at"),
            "last_confirmed": current.get("last_confirmed"),
        },
        "timeline": timeline,
        "summary": {
            "total_events": len(timeline),
            "event_counts": events_summary,
            "lifecycle": _describe_lifecycle(timeline),
        },
    })


def _describe_lifecycle(timeline: List[Dict[str, Any]]) -> str:
    """Produce a one-line lifecycle description from the timeline."""
    events = [e["event"] for e in timeline]
    if not events:
        return "No events recorded."

    parts = []
    if "created" in events:
        parts.append("created")
    confirm_count = sum(1 for e in timeline if e["event"] == "patch" and "confirmations" in str(e.get("changes", {})))
    if confirm_count:
        parts.append(f"confirmed {confirm_count}x")
    if "superseded_by" in events:
        parts.append("superseded")
    if "contradicted_by" in events:
        parts.append("contradicted")

    if not parts:
        parts.append("unknown lifecycle")
    return " → ".join(parts)


def _scrub_session_db(obs_id: str, content: str, config: SpineConfig) -> int:
    """Search session DB for mentions of deleted observation content.

    Non-destructive — logs matches to bench/scrub-<date>.json for review.
    Returns number of matching transcript rows found.
    """
    import os as _os
    from pathlib import Path as _Path
    import sqlite3 as _sqlite3

    state_db = _Path(_os.path.expanduser("~/.hermes/state.db"))
    if not state_db.exists():
        return 0

    conn = None
    try:
        conn = _sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
        conn.row_factory = _sqlite3.Row

        # Search FTS5 for the content or its key terms — reuses index.py's
        # shared safe-quoting helper instead of a second copy of the same
        # apostrophe/punctuation-safe tokenize-and-quote logic. Capped to the
        # first 8 words, same as before.
        import re as _re
        words = _re.findall(r"[A-Za-z0-9]+", content)[:8]
        terms = _fts5_safe_query(" ".join(words))
        if not terms:
            return 0
        rows = conn.execute(
            """SELECT m.id, m.session_id, m.role, m.content, m.timestamp
               FROM messages m
               JOIN messages_fts fts ON m.rowid = fts.rowid
               WHERE messages_fts MATCH ?
               LIMIT 50""",
            (terms,),
        ).fetchall()

        matches = [dict(r) for r in rows]

        # Save scrub report
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        scrub_dir = _Path(config.canonical_root) / "bench"
        scrub_dir.mkdir(parents=True, exist_ok=True)
        scrub_path = scrub_dir / f"scrub-{date_str}.json"

        existing = []
        if scrub_path.exists():
            with open(scrub_path, "r") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    pass

        existing.append({
            "obs_id": obs_id,
            "content": content,
            "scrubbed_at": datetime.now(timezone.utc).isoformat(),
            "session_db_matches": len(matches),
            "matches": matches,
        })

        with open(scrub_path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        return len(matches)
    except Exception as e:
        logger.warning("Session DB scrub failed: %s", e)
        return -1
    finally:
        if conn is not None:
            conn.close()
