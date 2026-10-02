"""Spine loops — spec §5.

Three loops:
1. Observer — on_session_end → extract durable observations → remember()
2. Consolidation — nightly cron → merge, contradict, promote, demote, decay
3. Activation manifest — on_session_start → manifest + recall top-6
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .config import SpineConfig
from .embedder import embedder_available, embed_single
from .index import MemoryIndex, _cosine_similarity, _deserialize_vector

# The hot-core size budget, defined ONCE and imported by heartbeat.py.
# It used to be hard-coded here three times and again in heartbeat.py, so
# raising it on 2026-09-05 left the nightly warning still quoting 20,000
# while the heartbeat had already moved on. See heartbeat.HOTCORE_WARN_BYTES
# for the rationale behind the number.
#
# 2026-09-11: raised 26,000 -> 30,000 (user decision: "raise the cap and
# prune"). 26,000 was set against a 21,000-25,000 settling band; the file had
# drifted to 27,855 and the beat went red on Sep 8, 9, 10 and 11 with no
# automated remedy left — the demote pass reported "no promoted observations
# left to demote" and the LLM pass could find only -83 bytes against a
# 1,855-byte overrun, with zero near-duplicate blocks in the file.
#
# NOTE for whoever raises this again: the budget is also the demote TRIGGER
# (mem_size > HOTCORE_BUDGET_BYTES). Raising it does not reduce growth, it
# only stops the demote pass from firing until the next line is crossed, so
# pair every raise with a real prune. The durable fix is the tracked
# hotcore-retrieval-split (stop injecting situational corrections every call,
# retrieve them instead).
HOTCORE_BUDGET_BYTES = 30000

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
            quote = (obs.get("quote") or "").strip()
            result = handle_remember({
                "content": obs.get("content", ""),
                "type": obs.get("type", "fact"),
                "epistemic": obs.get("epistemic", "extracted"),
                "confidence": obs.get("confidence", 0.5),
                "topics": obs.get("entities", []),
                # The extraction prompt asks for a verbatim supporting snippet
                # and always returned one; it was never read. This is the
                # provenance trail behind every stored observation.
                "evidence": [quote] if quote else [],
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

    try:
        idx = MemoryIndex(config.db)
        idx.open()
        idx.upsert_episode(episode)
        idx.conn.commit()
        idx.close()
    except Exception as e:
        logger.warning("Observer: episode DB upsert failed for %s: %s", episode_id, e)

    logger.info("Observer: wrote episode %s (session=%s, %d obs, %d stored, %d turns)",
                episode_id, session_id, len(observations), stored, user_turns + asst_turns)


# ═══════════════════════════════════════════════════════════════════════
# Loop 2: Consolidation (nightly cron)
# ═══════════════════════════════════════════════════════════════════════

def run_consolidation(config: SpineConfig, profile: str = "") -> Dict[str, Any]:
    """Run all five consolidation passes and return a report.

    Called by nightly cron. Passes:
    1. Decay — age observations, archive those below threshold
    2. Merge — cluster near-duplicates by exact content match
    3. Contradict — detect semantic conflicts, keep both
    4. Promote — auto-promote extracted+high-confidence or type=correction
    5. Demote — push stale hot-core entries back to spine
    """
    # Every pass below is scoped to ONE profile. Before this, decay, merge,
    # promote, demote and contradiction-detection all ran globally, so importing
    # a second profile (agent:claude-code) would have decayed its curated
    # entries, auto-promoted them into Hermes's own MEMORY.md hot core, and
    # contradiction-scanned them against agent:main.
    profile = profile or CONSOLIDATION_PROFILE

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
    decayed, archived = _decay_pass(idx, config, profile)
    report["passes"]["decay"] = f"archived {archived} observations below threshold {config.archive_threshold}"

    # ── Pass 2: Merge near-duplicates ────────────────────────────────
    active_rows = idx.conn.execute(
        "SELECT id, profile, type, content, confirmations FROM observations"
        " WHERE status='active' AND profile = ? ORDER BY type",
        (profile,),
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

    # ── Pass 2.5: Semantic merge (cosine >0.88 clustering — spec §5.2.1) ────
    if embedder_available():
        try:
            semantic_merged = _semantic_merge(idx, config, profile)
            merged += semantic_merged
            report["passes"]["semantic_merge"] = f"semantically merged {semantic_merged} near-duplicates (cosine >0.88)"
        except Exception as e:
            report["passes"]["semantic_merge"] = f"semantic merge skipped: {e}"
    else:
        report["passes"]["semantic_merge"] = "skipped — embedder unavailable"

    # ── Pass 2.6: Contradict detection — flag, never resolve (spec §5.2.2) ────
    contradictions = _detect_contradictions(idx, profile)
    report["passes"]["contradict"] = (
        f"detected {len(contradictions)} potential contradictions"
        if contradictions else "no contradictions detected"
    )
    if contradictions:
        report["contradictions"] = contradictions

    # ── Pass 3: Promote (pure soft gate — D10: all ≥0.9 auto-promote + notify + revert) ────
    promoted = 0
    notified: List[str] = []

    # Pure soft: all observations ≥ 0.9 auto-promote (extracted + inferred + correction)
    auto_candidates = idx.conn.execute(
        """SELECT id, content, type, epistemic, confidence
           FROM observations WHERE status='active'
           AND profile = ?
           AND confidence >= ?
           ORDER BY confidence DESC""",
        (profile, config.promote_auto_min_confidence),
    ).fetchall()

    for row in auto_candidates:
        obs_id, content, obs_type, epistemic, confidence = row
        # Only skip on a genuine failure. Flipping status after a failed append
        # stranded the row forever (promote reads only 'active', demote cannot
        # excise text never written) and announced promotions that never
        # happened. A duplicate still marks promoted: the text IS in the hot
        # core, so the row is represented there and must become demotable.
        outcome = _promote_to_hotcore(obs_id, content, obs_type, config)
        if outcome == "failed":
            continue
        if outcome == "duplicate":
            # Another row already owns that block, and _excise_block removes
            # exactly one, so marking this 'promoted' stranded it forever:
            # never demotable (no block to match) and never re-promotable
            # (promote reads only 'active'). 'demoted' is the honest state --
            # represented in the hot core by another row, still searchable,
            # not retried every night.
            idx.update_status(obs_id, "demoted")
            continue
        idx.update_status(obs_id, "promoted")
        promoted += 1
        notified.append(f"[{obs_id[:8]}] ({obs_type}, {epistemic}, conf={confidence:.2f}) {content[:100]}")

    # Correction fast-path: promote even below 0.9 bar
    correction_candidates = idx.conn.execute(
        """SELECT id, content, type, epistemic, confidence
           FROM observations WHERE status='active'
           AND profile = ?
           AND type = 'correction'
           AND confidence < ?
           AND id NOT IN (SELECT id FROM observations WHERE status='promoted')""",
        (profile, config.promote_auto_min_confidence),
    ).fetchall()

    for row in correction_candidates:
        obs_id, content, obs_type, epistemic, confidence = row
        outcome = _promote_to_hotcore(obs_id, content, obs_type, config)
        if outcome == "failed":
            continue
        if outcome == "duplicate":
            # Another row already owns that block, and _excise_block removes
            # exactly one, so marking this 'promoted' stranded it forever:
            # never demotable (no block to match) and never re-promotable
            # (promote reads only 'active'). 'demoted' is the honest state --
            # represented in the hot core by another row, still searchable,
            # not retried every night.
            idx.update_status(obs_id, "demoted")
            continue
        idx.update_status(obs_id, "promoted")
        promoted += 1
        notified.append(f"[{obs_id[:8]}] (correction fast-path, conf={confidence:.2f}) {content[:100]}")

    report["passes"]["promote"] = (
        f"promoted {promoted} (pure soft gate — all ≥0.9, notify + revert)"
    )
    report["promoted_items"] = notified

    # Notification: build text for Telegram delivery
    if promoted > 0:
        report["promotion_notice"] = (
            f"🧠 **{promoted} observation(s) promoted to MEMORY.md**\n\n"
            + "\n".join(f"• {n}" for n in notified)
            + f"\n\n_To revert: use forget() with the content shown above. "
            f"Reverted observations get status=rejected and never re-promote._"
        )

    # ── Pass 4: Demote stale hot-core ────────────────────────────────
    demoted = 0
    mem_path = _hotcore_path()
    mem_size = os.path.getsize(mem_path) if os.path.exists(mem_path) else 0
    start_size = mem_size

    rewrite_failed = False
    if mem_size > HOTCORE_BUDGET_BYTES:
        # Oldest-confirmed promoted rows first. Two things were wrong here
        # before: the text was never removed from MEMORY.md (so the file only
        # ever grew), and rows were flipped back to 'active', which made them
        # eligible for promotion again on the very next run.
        stale = idx.conn.execute(
            """SELECT id, content, type FROM observations WHERE status='promoted'
               AND profile = ?
               ORDER BY last_confirmed ASC""",
            (profile,),
        ).fetchall()

        # Whole read-modify-write under the lock. Deciding what to excise from a
        # snapshot and then replacing the file is only safe if no other writer
        # can append in between; memory() can and does.
        with _hotcore_lock():
            blocks = _read_hotcore_blocks(mem_path)
            unmatched = 0
            demote_ids: List[str] = []

            for obs_id, content, obs_type in stale:
                if mem_size <= HOTCORE_TARGET_BYTES:
                    break
                remaining = _excise_block(blocks, content)
                if remaining is None:
                    # MEMORY.md has a second writer: the memory() tool and
                    # hotcore_consolidate.py's LLM-assisted pass rewrite and
                    # merge blocks wholesale, so a promoted row's text may no
                    # longer appear verbatim. Leave the row promoted rather than demoting one
                    # whose text could not be removed -- doing that empties the
                    # DB's view of the hot core while the file stays fat.
                    unmatched += 1
                    continue
                blocks = remaining
                # Bytes, not characters: every threshold here comes from
                # os.path.getsize. The live file is 17,309 chars / 17,515 bytes,
                # so a char count under-reads by ~1.2% and stops the loop early.
                mem_size = len(("\n§\n".join(blocks) + "\n").encode("utf-8"))
                demote_ids.append(obs_id)
                demoted += 1

            if demoted:
                # Durable state BEFORE the irreversible file write. The reverse
                # order left a crash window where MEMORY.md had lost the blocks
                # but the DB still called them promoted -- drift no check sees.
                # 'demoted' stays searchable (it is in SEARCHABLE_STATUSES) but
                # is never re-promoted: the promote pass only reads 'active'.
                # Write the file FIRST. If it raises, the rows are still
                # 'promoted' and next night retries. The reverse order left them
                # committed as 'demoted' with their text still in the file, and
                # since demote reads only 'promoted' those blocks became
                # permanently un-excisable -- the exact stuck-over-budget state
                # this pass exists to prevent.
                try:
                    _write_hotcore_blocks(mem_path, blocks)
                except Exception as e:  # noqa: BLE001
                    logger.error("hot-core rewrite failed, leaving %d rows promoted: %s",
                                 len(demote_ids), e)
                    report.setdefault("warnings", []).append(
                        f"MEMORY.md rewrite failed ({e}); {len(demote_ids)} rows left promoted")
                    demoted = 0
                    demote_ids = []
                    rewrite_failed = True
                    # Re-read the truth. mem_size still held the value the
                    # excise loop simulated down to, so the report claimed a
                    # shrink that never happened AND the figure fell under the
                    # budget, suppressing the over-budget alarm on the exact
                    # path this branch introduces.
                    mem_size = os.path.getsize(mem_path)
                else:
                    for oid in demote_ids:
                        idx.update_status(oid, "demoted")
                    idx.conn.commit()
                    mem_size = os.path.getsize(mem_path)


        # Gate on the state, not on why. The previous version only warned when a
        # promoted row could not be matched, which is the RARE case; the common
        # one is simply running out of promoted rows to demote, where
        # `unmatched` is 0. Verified live: 2026-08-18 finished at 32,481 bytes,
        # 62% over budget, and reported "demoted 2 stale entries" with no
        # warning at all. A human had to rescue the file.
        if mem_size > HOTCORE_BUDGET_BYTES:
            # Say so loudly. A pass that runs nightly, removes nothing, and
            # reports success is exactly how the embedder outage went unseen
            # for eight days.
            if rewrite_failed:
                why = "the hot-core rewrite failed, so nothing was removed"
            elif unmatched:
                why = (f"{unmatched} promoted observation(s) could not be matched "
                       f"to a block")
            else:
                why = "there are no promoted observations left to demote"
            warning = (
                f"MEMORY.md is {mem_size:,} bytes, over the {HOTCORE_BUDGET_BYTES:,} budget: {why}. "
                f"The demote pass cannot shrink the file further — run the "
                f"LLM-assisted consolidation pass: "
                f"`python3 consolidate.py --hotcore` (spine/hotcore_consolidate.py)."
            )
            report.setdefault("warnings", []).append(warning)
            logger.warning(warning)

    report["passes"]["demote"] = (
        f"demoted {demoted} stale entries "
        f"(MEMORY.md: {start_size:,} → {mem_size:,} bytes)"
    )

    report["active_count"] = idx.count_active()
    idx.conn.commit()
    idx.close()

    return report


CONSOLIDATION_PROFILE = "agent:main"

HOTCORE_PATH = "~/.hermes/memories/MEMORY.md"

# Demote until the file is back under this. Kept below the 20,000 trigger so a
# single pass actually clears the threshold instead of hovering on it and
# re-firing every night.
HOTCORE_TARGET_BYTES = 18000

_TAG_MAP = {
    "correction": "[C]",
    "preference": "[W]",
    "pattern": "[W]",
    "fact": "[F]",
    "identity": "[ID]",
}


try:
    import fcntl
except ImportError:  # pragma: no cover — Unix only; degrade to no locking
    fcntl = None


@contextmanager
def _hotcore_lock():
    """Exclusive lock on MEMORY.md, using the SAME sidecar the other writer uses.

    Hermes's memory() tool (tools/memory_tool.py `_file_lock`) holds an flock on
    `MEMORY.md.lock` across its own read-modify-write. The demote pass rewrites
    the whole file, so without taking that same lock a memory() append landing
    mid-pass is silently discarded by our os.replace -- and by definition a
    just-written hot-core entry exists nowhere else yet.
    """
    path = _hotcore_path() + ".lock"
    if fcntl is None:
        yield
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        fd.close()


def _hotcore_path() -> str:
    return os.path.expanduser(HOTCORE_PATH)


def _read_hotcore_blocks(path: str) -> List[str]:
    """Split MEMORY.md into its §-delimited blocks."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [b.strip() for b in f.read().split("\n§\n") if b.strip()]


def _write_hotcore_blocks(path: str, blocks: List[str]) -> None:
    """Rewrite MEMORY.md from blocks, atomically via a temp file + replace."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n§\n".join(blocks) + "\n")
    os.replace(tmp, path)


def _hotcore_entry(content: str, obs_type: str) -> str:
    return f"{_TAG_MAP.get(obs_type, '[F]')} {content}".strip()


_TAG_PREFIX = re.compile(r"^\s*\[(C|R|W|F|ID)\]\s*")


def _block_body(block: str) -> str:
    """A block's text with its [X] tag removed, for tag-insensitive matching.

    Legacy blocks carry an [R] tag that no observation type maps to, so
    comparing the rendered '{tag} {content}' string misses them entirely.
    """
    return _TAG_PREFIX.sub("", block).strip()


def _excise_block(blocks: List[str], content: str) -> Optional[List[str]]:
    """Return blocks minus the one matching content, or None if absent.

    Matching is exact on the tag-stripped body. Deliberately NOT a substring
    match: the consolidation pass merges several facts into one block, and
    removing a merged block to retire one of its facts would silently delete
    the others.
    """
    target = content.strip()
    for i, b in enumerate(blocks):
        if _block_body(b) == target:
            # Exactly one. Removing every match meant a legacy "[R] foo" and a
            # promoted "[F] foo" both vanished when a single row was demoted.
            # (The promote dedupe is now tag-INSENSITIVE, so such a pair can no
            # longer be created here — but pre-existing files still contain them,
            # and one demoted row must only ever retire one block.)
            return blocks[:i] + blocks[i + 1:]
    return None


# Calibrated on live data 2026-09-03, NOT inherited from _semantic_merge's 0.88
# (tuned for merging whole observations; at 0.88 this caught nothing, because
# the observer's rewordings sit far below it).
#
# Cosine ALONE cannot separate the two populations, which overlap:
#   true duplicates   brain-updater family min 0.545, median 0.705
#                     "[SILENT]" family    0.606 and 0.634
#                     username family      min 0.584, median 0.770
#   curated DISTINCT  75 hot-core blocks, 2,775 pairs: median 0.163,
#                     p99 0.489, max 0.683
#
# 0.55 is safe only BECAUSE of the second signal. Measured across all 2,775
# curated pairs, with _disagrees_on_specifics applied:
#
#   threshold   flagged by cosine   surviving the guard (= false positives)
#     0.55             10                        0
#     0.60              5                        0
#     0.65              1                        0
#
# Every pair the cosine flags in that band differs on a number, path or
# identifier, so the guard rejects all of them. Cosine-only at 0.55 would have
# produced 10 wrong merges; with the guard it produces none, and it catches the
# 0.606 / 0.634 duplicates that 0.70 let straight through into the hot core on
# 2026-09-03.
HOTCORE_DUP_COSINE = 0.55

# Near-misses are logged, not acted on, so what the threshold misses stays
# measurable instead of assumed.
HOTCORE_DUP_NEARMISS = 0.45

# Embedding the whole hot core on every candidate is the obvious waste here.
# Keyed on the block set, so it survives across candidates and invalidates the
# moment a write changes the file.
_HOTCORE_EMB_CACHE: Dict[int, Any] = {}


# Numbers, times, versions, ALL-CAPS identifiers, paths, env vars, flags.
# Anything whose EXACT value is the point of the sentence.
_HARD_TOKEN = re.compile(
    r"\b\d[\d:._/-]*\b"           # 0.9, 09:00-12:00, 3.11.15, 2026-08-26
    r"|\b[A-Z][A-Z0-9_]{3,}\b"     # DEEPSEEK_API_KEY, TOTO_JACKPOT
    r"|~?/[\w./-]+"                # ~/.hermes/.env, api/draw.ts
    r"|--[\w-]+"                   # --push, --no-verify
)


def _hard_tokens(text: str) -> set:
    return set(_HARD_TOKEN.findall(text or ""))


def _disagrees_on_specifics(candidate: str, existing: str) -> bool:
    """Does `candidate` carry an exact value that `existing` does not?

    The second signal, and the reason cosine alone cannot be trusted here.
    Measured on the live embedder:

        "peak billing is 09:00-12:00 SGT"
        "peak billing is 14:00-18:00 SGT"   cosine 0.974

    Those are two halves of one real fact in the owner's hot core, and a cosine-only
    check merges them and silently deletes a billing window. Same shape for
    "entry 0" vs "entry 1" (0.888) and any pair of sibling facts that differ
    only by a version, a path, a topic id or a flag.

    So a high cosine is necessary but not sufficient: if the candidate mentions
    a number, identifier, path or flag the existing block never mentions, it is
    a SIBLING fact, not a rewording. Deterministic, per the house rule that
    exactness gets enforced in code rather than inferred by a model.
    """
    return bool(_hard_tokens(candidate) - _hard_tokens(existing))


def _hotcore_semantic_duplicate(body: str, block_bodies: List[str]) -> Optional[str]:
    """Is `body` a REWORDING of something already in the hot core?

    Exact body match is the caller's job and is free. This catches what exact
    match structurally cannot, which is the failure that actually happened: the
    observer keeps re-forming the same fact in new words, every copy clears the
    0.9 confidence gate, and every copy promotes. Measured on the live store
    before this existed: 53 rows restating the brain-updater workflow, 31
    restating the macOS username, 9 restating the timezone. One promote pass
    pushed 10 rows into the hot core and 7 were rewordings of content already
    there, displacing older blocks that had to be demoted to make room.

    Returns the matching block body, or None.

    Returns None when the embedder is unavailable, which degrades to
    exact-match-only rather than blocking promotion. That is deliberate: a
    duplicate in the hot core is a cost, but a promote pass that silently stops
    promoting is an outage. The warning is logged so it is not silent.
    """
    if not body or not block_bodies:
        return None
    try:
        from . import embedder
        if not embedder.embedder_available():
            logger.warning(
                "hot-core dedupe degraded to exact match: embedder unavailable")
            return None

        key = hash(tuple(block_bodies))
        cached = _HOTCORE_EMB_CACHE.get(key)
        if cached is None:
            cached = embedder.embed(block_bodies)
            _HOTCORE_EMB_CACHE.clear()   # only ever hold the current file
            _HOTCORE_EMB_CACHE[key] = cached

        cand = embedder.embed_single(body)
        best, best_i = 0.0, -1
        for i, vec in enumerate(cached):
            num = sum(a * b for a, b in zip(cand, vec))
            da = sum(a * a for a in cand) ** 0.5
            db = sum(b * b for b in vec) ** 0.5
            if not da or not db:
                continue
            cos = num / (da * db)
            if cos > best:
                best, best_i = cos, i
        if best >= HOTCORE_DUP_COSINE:
            if _disagrees_on_specifics(body, block_bodies[best_i]):
                logger.info(
                    "hot-core: cos=%.3f but specifics differ, treating as a "
                    "sibling fact not a reword: %.60s", best, body)
                return None
            logger.info("hot-core near-duplicate (cos=%.3f): %.60s", best, body)
            return block_bodies[best_i]
        if best >= HOTCORE_DUP_NEARMISS:
            # Deliberately NOT treated as a duplicate. Logged so the
            # under-catching is measurable rather than assumed.
            logger.info("hot-core near-MISS (cos=%.3f, below %.2f): %.60s",
                        best, HOTCORE_DUP_COSINE, body)
    except Exception as e:  # noqa: BLE001
        # A broken dedupe must not take the promote pass down with it.
        logger.warning("hot-core dedupe check failed, falling back to exact: %s", e)
    return None


def _promote_to_hotcore(obs_id: str, content: str, obs_type: str,
                        config: SpineConfig) -> str:
    """Append an observation to the MEMORY.md hot core.

    Returns "written", "duplicate", or "failed" -- the caller must distinguish
    them. A plain bool conflated "already there" with "the write blew up", and
    the caller skipped update_status for both, so a row whose text was already a
    block stayed 'active' forever, was re-selected every night, and could never
    be demoted (demote reads only 'promoted').

    Writes through _write_hotcore_blocks rather than appending raw text. A raw
    append onto a file that already ends in a newline produces "…entry\n\n§\n",
    which fails memory_tool._detect_external_drift's round-trip check -- and a
    drifted file makes Hermes's own memory() refuse every mutation. The two
    writers have to agree on the format, and one serialiser is how you get that.
    """
    mem_path = _hotcore_path()
    entry = _hotcore_entry(content, obs_type)

    try:
        # Same lock as memory(): the dedupe check is a read-then-write, so
        # without it two writers can both observe "absent".
        with _hotcore_lock():
            blocks = _read_hotcore_blocks(mem_path)
            # Compare tag-stripped bodies. Comparing the full "[F] body" string
            # meant sync_hotcore's re-tagged copies ("[W] x" -> "[F] x") never
            # matched and appended a second block: a duplication engine inside
            # the file injected into every system prompt. Everything else in the
            # system (excise, coverage) already compares bodies.
            bodies = [_block_body(b) for b in blocks]
            if _block_body(entry) in set(bodies):
                logger.debug("Skipped promoting %s — already in MEMORY.md", obs_id)
                return "duplicate"
            # Exact match only catches a byte-identical restatement. The
            # observer rewords, so the same fact arrives again under new
            # wording and promotes again. See _hotcore_semantic_duplicate.
            near = _hotcore_semantic_duplicate(_block_body(entry), bodies)
            if near is not None:
                logger.debug("Skipped promoting %s — reword of an existing block",
                             obs_id)
                return "duplicate"
            _write_hotcore_blocks(mem_path, blocks + [entry])
        logger.info("Promoted %s to MEMORY.md: %s", obs_id, content[:80])
        return "written"
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to promote %s to MEMORY.md: %s", obs_id, e)
        return "failed"


# build_activation_manifest() removed 2026-09-05. It was orphaned — zero callers,
# zero tests, zero doc references — because its only consumer stored the result
# on spine's _manifest_text, which nothing ever read. Beyond being dead it was
# wrong three ways: a hard-coded rule_checklist of four rules matching none of
# the ten real [R] rules in the hot core; a cross_topic_reminder hard-coded to
# "this Telegram group"; and a docstring promising relevant_skills and
# recalled_top_6 while the body set `recalled` and `note`, so the caller's
# lookups always fell through to defaults. Per-turn rule delivery now goes
# through SpineProvider.prefetch(). See hotcore-retrieval-split.


def _semantic_merge(idx: MemoryIndex, config: SpineConfig, profile: str) -> int:
    """Semantic near-duplicate merge via cosine clustering + LLM.

    Clusters active observations by profile, merging paraphrases
    where cosine similarity >0.88 (spec §5.2.1). Uses LLM to produce
    a single canonical entry per cluster. Falls back to keep-both on
    LLM failure.

    Returns number of observations merged (superseded).
    """
    from .llm_client import call_llm

    # Fetch all active observations with embeddings
    rows = idx.conn.execute(
        """SELECT id, profile, type, content, confidence, confirmations,
                  epistemic, evidence, embedding
           FROM observations WHERE status='active' AND embedding IS NOT NULL
             AND profile = ?""",
        (profile,),
    ).fetchall()

    if len(rows) < 2:
        return 0

    # Group by profile
    by_profile: Dict[str, List[tuple]] = {}
    for row in rows:
        profile = row[1]
        by_profile.setdefault(profile, []).append(row)

    merged = 0
    for profile, profile_rows in by_profile.items():
        if len(profile_rows) < 2:
            continue

        # Build clusters via greedy clustering (cosine >0.88)
        clusters: List[List[int]] = []  # indices into profile_rows
        used: set = set()

        for i in range(len(profile_rows)):
            if i in used:
                continue
            emb_i = _deserialize_vector(profile_rows[i][8])
            if emb_i is None:
                continue
            cluster = [i]
            used.add(i)
            for j in range(i + 1, len(profile_rows)):
                if j in used:
                    continue
                emb_j = _deserialize_vector(profile_rows[j][8])
                if emb_j is None:
                    continue
                sim = _cosine_similarity(emb_i, emb_j)
                if sim > 0.88:
                    cluster.append(j)
                    used.add(j)
            if len(cluster) > 1:
                clusters.append(cluster)

        # Merge each cluster
        for cluster in clusters:
            members = [profile_rows[i] for i in cluster]
            canonical_idx = max(range(len(members)),
                                key=lambda i: members[i][4])  # highest confidence
            canonical = members[canonical_idx]

            # LLM merge for 3+ members; for pairs, just pick canonical
            if len(members) >= 3:
                obs_text = "\n".join(
                    f"[{m[0][:8]}] conf={m[4]:.2f} {m[3]}" for m in members
                )
                prompt = (
                    "Merge these near-duplicate observations into one concise, "
                    "present-tense atomic claim ≤500 chars. Preserve the most "
                    "precise version and combine confirmations.\n\n"
                    f"{obs_text}\n\nMerged claim:"
                )
                try:
                    merged_content = call_llm(
                        [{"role": "user", "content": prompt}],
                        model=getattr(config, "loop_model", "deepseek-v4-pro"),
                        max_tokens=200, temperature=0.2,
                    )
                    if merged_content and len(merged_content.strip()) > 10:
                        canonical = list(canonical)
                        canonical[3] = merged_content.strip()[:500]
                except Exception:
                    pass  # Keep original canonical content

            # Sum confirmations from all members
            total_confirmations = sum(m[5] for m in members)

            # Update canonical
            idx.conn.execute(
                "UPDATE observations SET confirmations=? WHERE id=?",
                (total_confirmations, canonical[0]),
            )

            # Supersede non-canonical members
            for i, m in enumerate(members):
                if i != canonical_idx:
                    idx.update_status(m[0], "superseded")
                    merged += 1

    return merged


_CONTRADICTION_CONFIDENCE_PENALTY = 0.15  # applied to the older/weaker side of a contradiction
_CONTRADICTION_CONFIDENCE_BOOST = 0.05     # applied to the more-recently-confirmed side


def _word_in(term: str, text: str) -> bool:
    """Whole-word match — avoids false positives like 'am' matching inside 'telegram'
    or 'pm' matching inside 'equipment'."""
    return re.search(r"\b" + re.escape(term) + r"\b", text) is not None


_CONTEXT_STOPWORDS = frozenset({
    "about", "after", "again", "also", "and", "any", "are", "because", "been",
    "before", "being", "both", "but", "can", "could", "did", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "into", "its", "itself", "just", "like", "more", "most", "much",
    "must", "not", "now", "only", "other", "our", "ours", "out", "over", "own",
    "same", "she", "should", "some", "such", "than", "that", "the", "their",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "very", "was", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "yourselves",
})


def _shared_context(a_topics: str, b_topics: str, a_lower: str, b_lower: str) -> bool:
    """True if two observations plausibly discuss the same subject.

    Context is shared when (1) they carry at least one common topic tag, or
    (2) their content shares at least one meaningful token (>=4 chars, not a
    stopword). Guards the opposition detector against flagging keyword
    opposites in unrelated observations (false positives).
    """
    try:
        ta = set(json.loads(a_topics)) if a_topics else set()
    except (json.JSONDecodeError, TypeError):
        ta = set()
    try:
        tb = set(json.loads(b_topics)) if b_topics else set()
    except (json.JSONDecodeError, TypeError):
        tb = set()
    if ta & tb:
        return True
    tok_a = {w for w in re.findall(r"[a-z]{4,}", a_lower) if w not in _CONTEXT_STOPWORDS}
    tok_b = {w for w in re.findall(r"[a-z]{4,}", b_lower) if w not in _CONTEXT_STOPWORDS}
    return bool(tok_a & tok_b)


_DECAY_LAST_RUN_KEY = "decay_last_run:{profile}"


def _decay_pass(idx: MemoryIndex, config: SpineConfig, profile: str,
                now: Optional[datetime] = None) -> Tuple[int, int]:
    """Age idle observations by config.decay_per_30d and archive the ones that
    fall below config.archive_threshold. Returns (decayed, archived).

    Decay is applied for the time elapsed SINCE THE LAST DECAY RUN, not since
    the observation was last confirmed. The original pass subtracted
    (days_idle / 30) * decay_per_30d from the already-decayed stored value on
    every nightly run, so an observation idle for d days lost d/30 * 0.05
    per NIGHT -- quadratic, not the documented -0.05 per 30 idle days. A 0.9
    observation reached the 0.3 archive floor in ~27 nights instead of 360
    days. Found 2026-08-30: the weekly retrieval benchmark regressed -11.8%
    recall because the golden observations for four queries had been
    archived by this runaway (53 agent:main rows archived Aug 19-29, every
    one of them >= 0.3 under the intended linear schedule).

    The last-run timestamp lives in dim_meta, keyed per profile, so a missed
    night is caught up on the next run and a manual re-run within the same
    day applies only the hours actually elapsed. On the first run after this
    fix there is no stamp; we record one and apply no decay, rather than
    guessing an elapsed window.
    """
    now = now or datetime.now(timezone.utc)
    key = _DECAY_LAST_RUN_KEY.format(profile=profile)
    row = idx.conn.execute(
        "SELECT value FROM dim_meta WHERE key=?", (key,)).fetchone()
    last_run: Optional[datetime] = None
    if row:
        try:
            last_run = datetime.fromisoformat(row[0])
        except (ValueError, TypeError):
            last_run = None

    decayed = 0
    archived = 0
    if last_run is not None:
        rows = idx.conn.execute(
            "SELECT id, confidence, last_confirmed, last_retrieved, type FROM observations"
            " WHERE status='active' AND profile = ?",
            (profile,),
        ).fetchall()
        for obs_id, conf, last_confirmed, last_retrieved, obs_type in rows:
            if conf is None:
                continue
            # Corrections carry the user's explicit words — the highest-value
            # memory class. Never let idle decay archive them; they leave active
            # status only via explicit demote/forget or supersede (2026-08-20,
            # compaction-prompt-tuning: preserve decisions over age).
            if obs_type == "correction":
                continue
            ref_date = last_confirmed or last_retrieved
            if not ref_date:
                continue
            try:
                ref_dt = datetime.fromisoformat(ref_date)
            except (ValueError, TypeError):
                continue
            if ref_dt.tzinfo is None:
                ref_dt = ref_dt.replace(tzinfo=timezone.utc)
            # Idle window that has not been charged yet: from the later of
            # (last confirmation, last decay run) up to now. A row confirmed
            # after the last run is charged only for the time since that
            # confirmation, so a fresh confirmation really does reset decay.
            window_start = max(ref_dt, last_run)
            idle_days = (now - window_start).total_seconds() / 86400.0
            if idle_days <= 0:
                continue
            conf = max(0.0, conf - (idle_days / 30.0) * config.decay_per_30d)
            idx.conn.execute(
                "UPDATE observations SET confidence=? WHERE id=?",
                (round(conf, 4), obs_id),
            )
            decayed += 1
            if conf < config.archive_threshold:
                idx.update_status(obs_id, "archived")
                archived += 1

    idx.conn.execute(
        "INSERT OR REPLACE INTO dim_meta (key, value) VALUES (?, ?)",
        (key, now.isoformat()),
    )
    idx.conn.commit()
    return decayed, archived


def _detect_contradictions(idx: MemoryIndex, profile: str) -> List[Dict[str, Any]]:
    """Detect potential contradictions between active observations (spec §5.2.2).

    Lightweight keyword-based detection — flags opposition pairs (e.g., "prefers"
    vs "hates", "morning" vs "evening") using whole-word matching. Never deletes
    or hides either side — instead applies a conservative, reversible confidence
    adjustment: the more recently-confirmed observation gets a small confidence
    boost, the other a larger penalty. Both sides remain visible via recall();
    a fully-decayed one only leaves MEMORY.md/active status through the normal
    decay pass, never immediately. This is a proportional nudge, not a Bayesian
    belief engine — full auto-resolution stays out of scope for a single-user system.

    Pairs are gated on shared context (topic tag or meaningful content-token
    overlap) before opposition keywords are compared, so unrelated observations
    (e.g. a cron-hygiene note containing "never" vs a research-dossier note
    containing "requires") are never flagged. Added 2026-08-04 after a false
    positive paired "cron prompts should not contain stale metadata" with
    "user evaluates tools via Instagram-sourced dossiers".

    Returns list of {a_id, a_content, b_id, b_content, reason, resolution}.
    """
    opposition_pairs = [
        (["prefer", "prefers", "like", "love"], ["hate", "dislike", "avoid", "never"]),
        (["morning", "early"], ["evening", "night", "late"]),
        (["always", "must", "require"], ["never", "optional", "flexible"]),
        (["concise", "short", "brief"], ["detailed", "long", "verbose", "comprehensive"]),
        (["yes", "approved", "accept"], ["no", "reject", "refuse", "deny"]),
    ]

    rows = idx.conn.execute(
        """SELECT id, type, content, confidence, last_confirmed, topics FROM observations
           WHERE status='active' AND profile = ? ORDER BY type""",
        (profile,),
    ).fetchall()

    if len(rows) < 2:
        return []

    contradictions: List[Dict[str, Any]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a_id, a_type, a_content, a_conf, a_last, a_topics = rows[i]
            b_id, b_type, b_content, b_conf, b_last, b_topics = rows[j]
            if a_type != b_type:
                continue
            a_lower = a_content.lower()
            b_lower = b_content.lower()
            # Context gate: skip pairs that don't plausibly discuss the same
            # subject — prevents keyword-opposition false positives.
            if not _shared_context(a_topics, b_topics, a_lower, b_lower):
                continue
            for pos_terms, neg_terms in opposition_pairs:
                a_pos = any(_word_in(t, a_lower) for t in pos_terms)
                a_neg = any(_word_in(t, a_lower) for t in neg_terms)
                b_pos = any(_word_in(t, b_lower) for t in pos_terms)
                b_neg = any(_word_in(t, b_lower) for t in neg_terms)
                if (a_pos and b_neg) or (a_neg and b_pos):
                    # Winner = more recently confirmed (tie-break: higher current confidence)
                    a_key = (a_last or "", a_conf or 0)
                    b_key = (b_last or "", b_conf or 0)
                    if a_key >= b_key:
                        winner_id, winner_conf = a_id, a_conf
                        loser_id, loser_conf = b_id, b_conf
                    else:
                        winner_id, winner_conf = b_id, b_conf
                        loser_id, loser_conf = a_id, a_conf

                    new_winner_conf = min(1.0, round((winner_conf or 0.5) + _CONTRADICTION_CONFIDENCE_BOOST, 4))
                    new_loser_conf = max(0.0, round((loser_conf or 0.5) - _CONTRADICTION_CONFIDENCE_PENALTY, 4))
                    idx.conn.execute("UPDATE observations SET confidence=? WHERE id=?", (new_winner_conf, winner_id))
                    idx.conn.execute("UPDATE observations SET confidence=? WHERE id=?", (new_loser_conf, loser_id))

                    contradictions.append({
                        "a_id": a_id[:8],
                        "a_content": a_content[:150],
                        "b_id": b_id[:8],
                        "b_content": b_content[:150],
                        "reason": f"Opposing keywords in {a_type} observations",
                        "resolution": (
                            f"{winner_id[:8]} confidence {winner_conf:.2f}->{new_winner_conf:.2f} (more recently confirmed), "
                            f"{loser_id[:8]} confidence {loser_conf:.2f}->{new_loser_conf:.2f} (unresolved — "
                            f"revert via forget() if wrong; neither side is deleted)"
                        ),
                    })
                    # Set contradicts[] on both — read-modify-write per row, JSON list.
                    # (Not string-concat against the schema's '' default: every real
                    # row is actually seeded "[]" by upsert_observation(), so a naive
                    # CASE WHEN contradicts='' never hits its clean-init branch, and a
                    # single shared-param UPDATE across both ids would wrongly write
                    # the same value to both rows instead of each other's id.)
                    for owner_id, other_id in ((a_id, b_id), (b_id, a_id)):
                        raw = idx.conn.execute(
                            "SELECT contradicts FROM observations WHERE id=?", (owner_id,)
                        ).fetchone()[0]
                        try:
                            current = json.loads(raw) if raw else []
                            if not isinstance(current, list):
                                current = []
                        except (json.JSONDecodeError, TypeError):
                            current = []
                        other_short = other_id[:8]
                        if other_short not in current:
                            current.append(other_short)
                        idx.conn.execute(
                            "UPDATE observations SET contradicts=? WHERE id=?",
                            (json.dumps(current), owner_id),
                        )
                    break
            if len(contradictions) >= 10:
                break
        if len(contradictions) >= 10:
            break

    return contradictions
