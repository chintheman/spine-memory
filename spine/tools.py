"""Spine tool schemas and handlers — spec §4.

Four tools exposed to the agent:
- remember()  — write gate with two-signal secrets detector + dedupe
- recall()    — hybrid FTS5+vec retrieval with auto-degradation
- reflect()   — LLM synthesis over recalled observations
- forget()    — discovery → per-entry confirmation → ID deletion
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import SpineConfig
from .jsonl_writer import JSONLWriter
from .secrets_detector import detect_secrets
from .embedder import embedder_available, embed_single, embed
from .index import MemoryIndex

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
            "profile": {"type": "string", "description": "Profile filter (default: agent:main)"},
            "k": {"type": "integer", "minimum": 1, "maximum": 25, "description": "Number of results (default: 6)"},
        },
        "required": ["query"],
    },
}

REFLECT_SCHEMA = {
    "name": "reflect",
    "description": "Deep analysis over recalled spine observations. Synthesizes patterns and answers questions about what the system knows.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "What to analyze or answer about the stored observations"},
            "profile": {"type": "string", "description": "Profile filter (default: agent:main)"},
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
        "required": ["term"],
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

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
    """Get JSONL writer for a profile's observations file."""
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
    profile = args.get("profile", "agent:main")

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
        "evidence": [],
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
                idx.close()
            except Exception:
                pass

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
        idx.close()
    except Exception as e:
        # Index failure is non-fatal for write — log and continue
        pass

    return json.dumps({"success": True, "id": obs_id, "indexed": embedding is not None})


def handle_recall(args: Dict[str, Any], config: SpineConfig) -> str:
    """recall() — hybrid retrieval with auto-degradation (§4.2)."""
    query = args["query"]
    profile = args.get("profile", "agent:main")
    k = min(args.get("k", 6), 25)

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

    idx.close()

    note = ""
    if not embedder_ok:
        note = "Semantic search unavailable — keyword results only."

    return json.dumps({"results": results, "count": len(results), "note": note})


def handle_reflect(args: Dict[str, Any], config: SpineConfig) -> str:
    """reflect() — recall k=25 then LLM-synthesize an answer (§4.3).

    Recalls top-25 observations, then calls the LLM to produce a coherent
    synthesis citing specific observation IDs.
    """
    query = args["question"]
    profile = args.get("profile", "agent:main")

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

    answer = call_llm(messages, model=model, max_tokens=500, temperature=0.3)

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
    term = args["term"]
    action = args.get("action", "discover")
    profile = args.get("profile", "agent:main")

    idx = _get_index(config)
    writer = _get_writer(config, profile)

    if action == "discover":
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
        row = idx.conn.execute(
            "SELECT id, content FROM observations WHERE id=? AND profile IN (?, 'shared')",
            (obs_id, profile),
        ).fetchone()

        if row is None:
            idx.close()
            return json.dumps({"error": f"Observation {obs_id} not found"})

        content = row[1]

        # Delete from index
        idx.delete_observation(obs_id)
        idx.close()

        # Remove from JSONL (mark as deleted — append a tombstone)
        now = _now_iso()
        tombstone = {
            "id": obs_id,
            "patch": {"status": "deleted", "deleted_at": now},
            "ts": now,
        }
        writer.append(tombstone)

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


def _scrub_session_db(obs_id: str, content: str, config: SpineConfig) -> int:
    """Search session DB for mentions of deleted observation content.

    Non-destructive — logs matches to bench/scrub-<date>.json for review.
    Returns number of matching transcript rows found.
    """
    import os as _os
    from pathlib import Path as _Path
    import sqlite3 as _sqlite3

    state_db = _Path(_os.path.expanduser("~/.hermes/sessions/state.db"))
    if not state_db.exists():
        return 0

    try:
        conn = _sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
        conn.row_factory = _sqlite3.Row

        # Search FTS5 for the content or its key terms
        terms = " OR ".join(content.split()[:8])  # first 8 words as search terms
        rows = conn.execute(
            """SELECT m.id, m.session_id, m.role, m.content, m.timestamp
               FROM messages m
               JOIN messages_fts fts ON m.rowid = fts.rowid
               WHERE messages_fts MATCH ?
               LIMIT 50""",
            (terms,),
        ).fetchall()

        matches = [dict(r) for r in rows]
        conn.close()

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
