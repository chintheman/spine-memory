"""Spine loops — spec §5.

Three loops:
1. Observer — on_session_end → extract durable observations → remember()
2. Consolidation — nightly cron → merge, contradict, promote, demote, decay
3. Activation manifest — on_session_start → manifest + recall top-6
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import SpineConfig

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════
# Loop 1: Observer (on_session_end)
# ═══════════════════════════════════════════════════════════════════════

def run_observer(messages: List[Dict[str, Any]], config: SpineConfig) -> None:
    """Extract durable observations from a completed session.

    Called by on_session_end hook. Calls LLM to extract observations,
    then routes each through the remember() gate — no privileged write path.
    Writes one episode summary per session.
    """
    if not messages:
        return

    from .jsonl_writer import JSONLWriter
    from .tools import _generate_id, handle_remember
    from .llm_client import extract_observations
    import os

    # Extract session_id from messages if available
    session_id = "unknown"
    for msg in messages:
        if isinstance(msg, dict) and msg.get("session_id"):
            session_id = msg["session_id"]
            break

    # Count user + assistant turns
    user_turns = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "user")
    asst_turns = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")

    # Build compact transcript for LLM (last N turns to stay within context)
    transcript_lines = []
    for m in messages[-40:]:  # last 40 messages
        if isinstance(m, dict):
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                # Truncate long messages
                content = content[:500]
                transcript_lines.append(f"[{role}] {content}")
    transcript = "\n".join(transcript_lines)

    # LLM extraction
    observations = []
    try:
        model = getattr(config, "loop_model", "deepseek-v4-pro")
        observations = extract_observations(transcript, model=model)
        logger.info("Observer: extracted %d observations from session %s", len(observations), session_id)
    except Exception as e:
        logger.warning("Observer LLM extraction failed: %s", e)

    # Route each observation through remember() gate
    stored = 0
    for obs in observations:
        try:
            result = handle_remember({
                "content": obs.get("content", ""),
                "type": obs.get("type", "fact"),
                "epistemic": obs.get("epistemic", "extracted"),
                "confidence": obs.get("confidence", 0.5),
            }, config)
            resp = json.loads(result)
            if resp.get("success") or resp.get("deduplicated"):
                stored += 1
        except Exception as e:
            logger.warning("Observer remember() failed for obs: %s", e)

    # Write episode record
    now = _now_iso()
    episode_id = _generate_id()
    episode = {
        "id": episode_id,
        "profile": "agent:main",
        "session_id": session_id,
        "summary": f"Session with {user_turns} user turns, {asst_turns} assistant turns. Observer extracted {len(observations)} observations, stored {stored}.",
        "outcomes": [obs.get("content", "")[:100] for obs in observations[:5]],
        "started_at": now,
        "ended_at": now,
        "compacted": False,
    }

    ep_dir = os.path.join(config.canonical_root, "episodes")
    writer = JSONLWriter(os.path.join(ep_dir, "agent_main.jsonl"))
    writer.append(episode)

    logger.info("Observer: wrote episode %s (session=%s, %d obs, %d stored, %d turns)",
                episode_id, session_id, len(observations), stored, user_turns + asst_turns)


# ═══════════════════════════════════════════════════════════════════════
# Loop 2: Consolidation (nightly cron)
# ═══════════════════════════════════════════════════════════════════════

def run_consolidation(config: SpineConfig) -> Dict[str, Any]:
    """Run all five consolidation passes and return a report.

    Called by nightly cron. Passes:
    1. Decay — age observations, archive those below threshold
    2. Merge — cluster near-duplicates by exact content match
    3. Contradict — detect semantic conflicts, keep both
    4. Promote — auto-promote extracted+high-confidence or type=correction
    5. Demote — push stale hot-core entries back to spine
    """
    from .index import MemoryIndex
    from .jsonl_writer import JSONLWriter
    import os

    report = {
        "timestamp": _now_iso(),
        "passes": {},
        "notes": [],
    }

    idx = MemoryIndex(config.db)
    idx.open()
    now = _now_iso()

    # ── Pass 1: Decay ───────────────────────────────────────────────
    decayed = 0
    archived = 0
    rows = idx.conn.execute(
        "SELECT id, confidence, last_confirmed, last_retrieved FROM observations WHERE status='active'"
    ).fetchall()
    for row in rows:
        obs_id, conf, last_confirmed, last_retrieved = row
        if conf is None:
            continue
        # Decay: confidence −0.05 per 30 idle days
        # Simple check: if below archive threshold, archive it
        if conf < config.archive_threshold:
            idx.update_status(obs_id, "archived")
            archived += 1
            decayed += 1
    report["passes"]["decay"] = f"archived {archived} observations below threshold {config.archive_threshold}"

    # ── Pass 2: Merge near-duplicates ────────────────────────────────
    active_rows = idx.conn.execute(
        "SELECT id, profile, type, content, confirmations FROM observations WHERE status='active' ORDER BY profile, type"
    ).fetchall()

    merged = 0
    seen: Dict[Any, str] = {}  # key=(profile,type,content_lower) -> canonical_id
    for row in active_rows:
        obs_id, profile, obs_type, content, confirmations = row
        key = (profile, obs_type, content.lower().strip())
        if key in seen:
            canonical_id = seen[key]
            # Mark this one as superseded
            idx.update_status(obs_id, "superseded")
            # Increment confirmations on canonical
            idx.conn.execute(
                "UPDATE observations SET confirmations = confirmations + ? WHERE id=?",
                (confirmations, canonical_id),
            )
            merged += 1
        else:
            seen[key] = obs_id
    report["passes"]["merge"] = f"merged {merged} near-duplicates into {len(seen)} canonical entries"

    # ── Pass 3: Promote (soft gate — D10 Chin: revert to pure-soft) ────
    promoted = 0
    candidates = idx.conn.execute(
        """SELECT id, content, type, epistemic, confidence
           FROM observations WHERE status='active'
           AND confidence >= ?
           ORDER BY confidence DESC""",
        (config.promote_auto_min_confidence,),
    ).fetchall()

    for row in candidates:
        obs_id, content, obs_type, epistemic, confidence = row
        # Soft gate: all qualifying observations auto-promote with revert button
        # Correction always fast-paths; extracted+inferred both write immediately
        _promote_to_hotcore(obs_id, content, obs_type, config)
        idx.update_status(obs_id, "promoted")
        promoted += 1

    report["passes"]["promote"] = f"promoted {promoted} (soft gate — all with revert)"

    # ── Pass 4: Demote stale hot-core ────────────────────────────────
    demoted = 0
    # Check if MEMORY.md is over budget
    mem_path = os.path.expanduser("~/.hermes/memories/MEMORY.md")
    mem_size = 0
    if os.path.exists(mem_path):
        mem_size = os.path.getsize(mem_path)
    if mem_size > 20000:
        # Find promoted entries with no recent evidence → demote
        stale = idx.conn.execute(
            """SELECT id FROM observations WHERE status='promoted'
               ORDER BY last_confirmed ASC LIMIT 5"""
        ).fetchall()
        for (obs_id,) in stale:
            idx.update_status(obs_id, "active")  # demote back to active
            demoted += 1
    report["passes"]["demote"] = f"demoted {demoted} stale entries (MEMORY.md: {mem_size:,} bytes)"

    report["active_count"] = idx.count_active()
    idx.close()

    return report


def _promote_to_hotcore(obs_id: str, content: str, obs_type: str, config: SpineConfig) -> None:
    """Write an observation to MEMORY.md hot core with appropriate tag.

    Tags per spec: [R] rule, [C] correction, [F] fact, [W] workflow, [ID] identity.
    """
    import os as _os

    tag_map = {
        "correction": "[C]",
        "preference": "[W]",
        "pattern": "[W]",
        "fact": "[F]",
        "identity": "[ID]",
    }
    tag = tag_map.get(obs_type, "[F]")

    mem_path = _os.path.expanduser("~/.hermes/memories/MEMORY.md")
    entry = f"\n§\n{tag} {content}\n"

    try:
        with open(mem_path, "a", encoding="utf-8") as f:
            f.write(entry)
        logger.info("Promoted %s to MEMORY.md: %s", obs_id, content[:80])
    except Exception as e:
        logger.error("Failed to promote %s to MEMORY.md: %s", obs_id, e)


# ═══════════════════════════════════════════════════════════════════════
# Loop 3: Activation Manifest (on_session_start)
# ═══════════════════════════════════════════════════════════════════════

def build_activation_manifest(config: SpineConfig) -> Dict[str, Any]:
    """Build the session-start activation manifest (spec §5.3).

    Returns a dict with:
    - active_profile
    - relevant_skills
    - cross_topic_reminder
    - rule_checklist
    - recalled_top_6 (if index has observations)
    """
    manifest: Dict[str, Any] = {
        "active_profile": "agent:main",
        "cross_topic_reminder": "You have cross-topic context access across this Telegram group.",
        "rule_checklist": [
            "[R] Verify before asserting — tool calls before claims",
            "[R] Sweep after every fix — find all instances of the same bug",
            "[R] Never offer options when path is clear",
            "[R] Private stays private — never expose secrets",
        ],
        "recalled": [],
    }

    # Attempt recall of top-6 relevant observations
    try:
        from .index import MemoryIndex
        from .embedder import embedder_available, embed_single

        idx = MemoryIndex(config.db)
        idx.open()
        count = idx.count_active()

        if count > 0:
            # Recall observations relevant to the default profile
            query = "user preferences communication style workflow habits"
            query_embedding = None
            if embedder_available():
                try:
                    query_embedding = embed_single(query)
                except Exception:
                    pass

            results = idx.search_hybrid(query, query_embedding, k=6)
            manifest["recalled"] = [
                f"[{r['id'][:8]}] ({r['type']}, {r['epistemic']}) {r['content']}"
                for r in results
            ]
            manifest["note"] = f"{count} observations available, {len(results)} recalled"

        idx.close()
    except Exception as e:
        logger.warning("Activation manifest recall failed: %s", e)

    return manifest
