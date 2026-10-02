#!/usr/bin/env python3
"""LLM-assisted hot-core consolidation — the pass the demote warning names.

Why this exists
---------------
`loops.run_consolidation`'s demote pass is spine's only automatic way to shrink
MEMORY.md, and it works by excising blocks owned by observations with
status='promoted'. On 2026-09-01 that well ran dry: 0 rows promoted, 671
demoted, MEMORY.md still 21,618 bytes against a 20,000 budget. The pass did the
only honest thing available to it and warned

    "there are no promoted observations left to demote. Spine cannot shrink the
     file on its own — run the LLM-assisted consolidation pass."

which named a pass that had never been written. The heartbeat had FAILed on
`hotcore` and `consolidation` every night since. This module is that pass.

What it does
------------
Two phases, both LLM-proposed and code-verified:

  merge     Near-duplicate blocks (cosine > MERGE_SIMILARITY) collapse into one.
  compress  Remaining blocks are rewritten tersely, largest first, until the
            file is under target.

Measured on the live file, merge alone cannot do the job: the 66 blocks are
genuinely distinct (highest pairwise cosine 0.629, upper-bound saving ~810
bytes against a ~3,600-byte deficit). Compression is the load-bearing phase.
Merge stays because when duplicates DO appear, collapsing them is strictly
better than compressing both halves of the same fact.

Why the LLM output is never trusted
-----------------------------------
MEMORY.md is injected into every Hermes call and has a second writer; a block
lost here can exist nowhere else. So the model only ever *proposes*. Every
proposal must clear `hard_tokens()`: every path, filename, identifier, number,
date, ALL-CAPS term and hyphenated name in the original has to survive verbatim
in the replacement. Prose is compressible; facts are not. A proposal that drops
one is discarded and the original block is kept byte-for-byte. Same rule for
merges, against the union of the members' tokens.

The guard is deliberately over-strict — it fires on things like
"write-protected" that a rewrite could safely reword. That costs compression
ratio, never content, and this is a file where the two are not comparable.

Concurrency
-----------
The file is read under `loops._hotcore_lock`, the LLM calls then run with the
lock RELEASED (they take minutes; memory() must not block that long), and the
lock is re-taken to write. Each rewrite is applied only if its original block
is still present byte-identical in the re-read file. Anything Hermes wrote in
between is left alone, and a mid-flight edit to a block simply skips that
block's rewrite instead of clobbering it.

Usage:
    python3 consolidate.py --hotcore              # apply
    python3 consolidate.py --hotcore --dry-run    # report only, writes nothing
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage
from .config import SpineConfig
from .loops import (
    HOTCORE_TARGET_BYTES,
    _hotcore_lock,
    _read_hotcore_blocks,
    _write_hotcore_blocks,
)

logger = logging.getLogger(__name__)

# Blocks this similar are the same fact said twice. Set from the live file:
# the highest genuine pair sits at 0.629 and the next tier is unrelated topics
# in the 0.50s, so 0.60 is the gap, not a round number.
MERGE_SIMILARITY = 0.60

# How hard to squeeze a block. 0.70 asks for a 30% cut, which the guard then
# vetoes wherever the facts will not fit in that budget.
COMPRESS_RATIO = 0.55

# Never accept a "rewrite" shorter than this fraction of the original: passing
# the token guard with 40% of the bytes means the connective tissue that made
# the fact actionable is gone, even though every identifier survived.
MIN_KEEP_RATIO = 0.45

# Blocks per LLM call. Small enough that one bad batch costs little, large
# enough that 66 blocks is ~11 calls rather than 66.
BATCH = 6

# NOT config.loop_model. loop_model is deepseek-v4-pro, a reasoning model, and
# it cannot do this job: measured on a 6-block batch it spent its entire 8,000
# token completion budget on reasoning_tokens, returned empty content, and took
# 119s. deepseek-chat returned the same batch correctly in 4.4s. Compression is
# a transformation with a code-side verifier (`_accept`), not a reasoning
# problem, so the non-reasoning model is the right tool as well as the fast one.
HOTCORE_MODEL = "deepseek-chat"

# Per-request seconds. A batch is ~5s on deepseek-chat; 180 is headroom for a
# slow hop, not an expected wait. The default 60 in llm_client timed out on
# every batch of the first live run and every block "kept its original".
LLM_TIMEOUT = 180

# A merge has to actually save something. At the 0.60 clustering threshold the
# model will dutifully concatenate two related-but-distinct facts and hand back
# a block 4 characters shorter than the pair — losing the granularity that lets
# demote and coverage reason about one fact at a time, for nothing. Demand a
# real reduction or keep both blocks.
MIN_MERGE_SAVING_RATIO = 0.20

# Compression rounds. 4 is where the live file stops improving; the loop also
# breaks the moment a round accepts nothing, so this is a ceiling on a runaway,
# not the expected count.
MAX_ROUNDS = 4

_TAG_PREFIX = re.compile(r"^\s*\[(C|R|W|F|ID)\]\s*")

# Punctuation that is sentence furniture, not part of a token.
_STRIP = ".,;:!?()[]{}<>\"'`“”‘’—–…"


def split_tag(block: str) -> Tuple[str, str]:
    """('[W] ', 'body') — or ('', block) for the untagged blocks memory() wrote.

    The tag is held out of the LLM round-trip entirely and re-attached
    afterwards, so a rewrite cannot invent, drop or reclassify one.
    """
    m = _TAG_PREFIX.match(block)
    if not m:
        return "", block.strip()
    return f"[{m.group(1)}] ", block[m.end():].strip()


def hard_tokens(text: str) -> set:
    """Tokens a rewrite is forbidden to lose.

    Anything that carries a fact rather than phrasing: paths and filenames
    (`~/.hermes/config.yaml`), dotted or underscored identifiers
    (`coaching_engine.py`, `telegram:<chat_id>:<topic_id>`), every number and date
    fragment, ALL-CAPS terms (`PITFALL`, `CCA-F`), and multi-segment hyphenated
    names (`hermes-agent`, `all-mpnet-base-v2`).

    Over-inclusive on purpose — see the module docstring. A false positive
    blocks one compression; a false negative silently deletes a fact.
    """
    out = set()
    for raw in text.split():
        t = raw.strip(_STRIP)
        if len(t) < 2:
            continue
        if any(c in t for c in "_/.:@"):
            out.add(t)
        elif any(c.isdigit() for c in t):
            out.add(t)
        elif len(t) >= 3 and t.isupper():
            out.add(t)
        elif "-" in t and len(t) >= 6 and all(seg.isalnum() for seg in t.split("-") if seg):
            out.add(t)
    return out


def _missing_tokens(original: str, replacement: str) -> List[str]:
    """Hard tokens of `original` that do not appear verbatim in `replacement`.

    Substring, not token, comparison: a rewrite may legitimately re-punctuate
    around an identifier ("`~/.hermes/config.yaml`," -> "~/.hermes/config.yaml)")
    and that is not a lost fact.
    """
    return sorted(t for t in hard_tokens(original) if t not in replacement)


# ── LLM plumbing ────────────────────────────────────────────────────

_COMPRESS_SYSTEM = """You compress entries in an AI agent's long-term memory file.

Each entry is one durable fact, rule or correction about the user. The whole
file is injected into every model call, so it MUST get shorter. Hitting
max_chars is the job, not a suggestion — an entry returned at its original
length is a failed entry.

INVIOLABLE — a separate program checks this and throws away any entry that
breaks it:
- Every path, filename, command, identifier, number, date, proper noun and
  ALL-CAPS term appears in your output EXACTLY as in the input, character for
  character. Never reformat "20,000" to "20000", "~/.hermes" to "$HOME/.hermes",
  "Task-completion" to "task completion", or drop a word from an ALL-CAPS
  phrase like "OUTPUT FORMAT CORRECTION".
- Every distinct claim, rule, exception and reason survives. Only phrasing goes.

HOW to hit the budget — be aggressive, this is the only thing you may cut:
- Delete articles, copulas and hedges: "the", "a", "is", "are", "that",
  "which", "in order to", "make sure to", "it is important to".
- Delete narrative and meta-framing: "the user has noted that", "this means
  that", "as discussed", "for context", "going forward", story of how the rule
  came to be. Keep the rule.
- Replace phrases with symbols the entry already uses elsewhere: "then" -> "→",
  "instead of" -> "not", "for example" -> "e.g.", "results in" -> "→".
- Collapse enumerations: "X, Y, and Z are all broken" -> "X/Y/Z broken".
- Say each thing once. If the entry states a rule and then restates it as an
  example of itself, keep the rule.
- Telegraphic register throughout. Sentence fragments are correct here.
- Use ONLY words that already appear in the entry. Never coin a compound or a
  new hyphenation to save space: "cannot be driven through X" must not become
  "undrivable via X", "credential file" must not become "credential-file". A
  word that appears nowhere else in the user's records makes the entry
  unfindable, and that is checked and rejected.
- Preserve imperative force: "NEVER x" must not soften to "avoid x".
- Never add anything. No commentary, no headers, no markdown fences.

Return ONLY a JSON array: [{"i": <the entry's i>, "text": "<rewritten entry>"}]
One object per input entry, same i values."""

_MERGE_SYSTEM = """You merge duplicate entries in an AI agent's long-term memory file.

You are given entries that say the same thing. Produce ONE entry that carries
every claim from all of them.
- Keep EVERY path, filename, command, identifier, number, date, proper noun and
  ALL-CAPS term EXACTLY as written in the inputs.
- Lose nothing that is not literally repetition between the inputs.
- No commentary, no markdown fences.

Return ONLY the merged entry text."""


def _parse_json_array(response: str) -> Optional[list]:
    """Pull a JSON array out of an LLM response that may be fenced or chatty."""
    s = response.strip()
    if s.startswith("```"):
        s = "\n".join(l for l in s.split("\n") if not l.startswith("```"))
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _compress_batch(bodies: Dict[int, str], model: str) -> Dict[int, str]:
    """Ask for terse rewrites of {index: body}. Returns {index: proposal}.

    Returns whatever came back parseable; the caller does the verifying. An
    LLM failure yields {} and every block in the batch keeps its original text.
    """
    from .llm_client import call_llm

    payload = [
        {"i": i, "max_chars": max(60, int(len(b) * COMPRESS_RATIO)), "text": b}
        for i, b in bodies.items()
    ]
    budget = sum(len(b) for b in bodies.values())
    response = call_llm(
        [{"role": "system", "content": _COMPRESS_SYSTEM},
         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        model=model,
        max_tokens=min(8000, 1500 + budget),
        temperature=0.2,
        timeout=LLM_TIMEOUT,
    )
    if not response:
        return {}
    parsed = _parse_json_array(response)
    if parsed is None:
        logger.warning("compress batch returned unparseable JSON: %s", response[:200])
        return {}
    out: Dict[int, str] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        i, text = item.get("i"), item.get("text")
        if isinstance(i, int) and isinstance(text, str) and i in bodies:
            out[i] = text.strip()
    return out


def _merge_bodies(bodies: Sequence[str], model: str) -> Optional[str]:
    from .llm_client import call_llm

    joined = "\n\n---\n\n".join(bodies)
    response = call_llm(
        [{"role": "system", "content": _MERGE_SYSTEM},
         {"role": "user", "content": joined}],
        model=model,
        max_tokens=min(4000, 1000 + len(joined)),
        temperature=0.2,
        timeout=LLM_TIMEOUT,
    )
    if not response:
        return None
    text = response.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n") if not l.startswith("```")).strip()
    return text or None


# ── Verification ────────────────────────────────────────────────────

def _accept(original: str, proposal: str, conn=None) -> Tuple[bool, str]:
    """Is `proposal` a safe replacement for `original`? (ok, reason_if_not)."""
    if not proposal:
        return False, "empty"
    if "\n§\n" in proposal or proposal.strip() == "§":
        # A block containing the delimiter would split into two on the next
        # read and desynchronise every index built off block position.
        return False, "contains block delimiter"
    if len(proposal) >= len(original):
        return False, f"not shorter ({len(proposal)} >= {len(original)})"
    if len(proposal) < len(original) * MIN_KEEP_RATIO:
        return False, f"over-compressed ({len(proposal)}/{len(original)} chars)"
    missing = _missing_tokens(original, proposal)
    if missing:
        return False, f"drops {len(missing)} hard token(s): {', '.join(missing[:4])}"
    if conn is not None:
        # A rewrite must not COIN vocabulary either. Preserving every hard token
        # still let the first live run turn "cannot be driven through Hermes"
        # into "undrivable via Hermes" — no fact lost, but "undrivable" appears
        # in zero observations and zero wiki chunks, so the block stopped being
        # retrievable from spine and the heartbeat's hotcore_coverage check went
        # red on 5 blocks. coverage.py's own docstring is the rule here: check
        # is_retrievable before removing anything from the hot core, and a
        # compressing rewrite removes the words that made it findable.
        ok, absent = coverage.is_retrievable(coverage.strip_tag(proposal), conn)
        if not ok:
            return False, (f"coins {len(absent)} term(s) absent from spine: "
                           f"{', '.join(absent[:4])}")
    return True, ""


def _blocks_bytes(blocks: Sequence[str]) -> int:
    """Exactly what os.path.getsize will report for these blocks.

    Bytes, not chars: the file is 8% multi-byte (— § ’), and every threshold in
    spine comes from getsize. Counting chars here would stop the pass ~1,700
    bytes short of the budget it is trying to clear.
    """
    return len(("\n§\n".join(blocks) + "\n").encode("utf-8"))


# ── The pass ────────────────────────────────────────────────────────

def consolidate_hotcore(config: SpineConfig,
                        target_bytes: int = HOTCORE_TARGET_BYTES,
                        dry_run: bool = False,
                        model: str = "") -> Dict[str, Any]:  # noqa: D417
    """Shrink MEMORY.md under `target_bytes` by merging and compressing blocks.

    Returns a report dict. `applied` is False whenever nothing was written —
    because the file was already small enough, because --dry-run, or because
    every proposal failed verification.
    """
    import sqlite3

    from .loops import _hotcore_path

    model = model or HOTCORE_MODEL
    path = _hotcore_path()
    report: Dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "path": path,
        "model": model,
        "dry_run": dry_run,
        "applied": False,
        "merges": [],
        "rewrites": [],
        "rejected": [],
    }

    # Read-only: this pass never writes to the store, and a query_only handle
    # cannot become the thing that corrupts memory.db while the nightly
    # consolidation holds it.
    conn = sqlite3.connect(config.db)
    conn.execute("PRAGMA query_only = ON")

    with _hotcore_lock():
        original = _read_hotcore_blocks(path)
    start_bytes = _blocks_bytes(original) if original else 0
    report["start_bytes"] = start_bytes
    report["start_blocks"] = len(original)

    if not original:
        report["note"] = "MEMORY.md is empty or missing — nothing to consolidate"
        return report
    if start_bytes <= target_bytes:
        report["note"] = f"already {start_bytes:,} bytes, at or under the {target_bytes:,} target"
        report["end_bytes"] = start_bytes
        return report

    # Working copy keyed by the ORIGINAL block text, so the write-back step can
    # match against a file that may have moved under us.
    plan: Dict[str, str] = {}          # original block -> replacement block
    dropped: List[str] = []            # original blocks removed by a merge
    merge_keys: set = set()            # plan keys whose value came from a merge

    # The plan is keyed by block text, so two byte-identical blocks would both
    # receive the same replacement and the file would keep a duplicate pair
    # forever. memory() dedupes on the tag-stripped body, but a legacy "[R] x"
    # alongside "[F] x" is two distinct strings. Leave any such block alone.
    seen: Dict[str, int] = {}
    for b in original:
        seen[b] = seen.get(b, 0) + 1
    ambiguous = {b for b, n in seen.items() if n > 1}
    if ambiguous:
        report["ambiguous_duplicates"] = len(ambiguous)

    # ── Phase 1: merge near-duplicates ───────────────────────────────
    merged_pairs = _cluster(original, config)
    for cluster in merged_pairs:
        members = [original[i] for i in cluster]
        if any(m in ambiguous for m in members):
            continue
        tag, _ = split_tag(members[0])
        proposal = _merge_bodies([split_tag(m)[1] for m in members], model)
        if proposal is None:
            report["rejected"].append({"kind": "merge", "why": "LLM returned nothing",
                                       "blocks": [m[:60] for m in members]})
            continue
        merged_block = (tag + proposal).strip()
        union = "\n".join(members)
        missing = _missing_tokens(union, merged_block)
        floor = int(len(union) * (1 - MIN_MERGE_SAVING_RATIO))
        retrievable, coined = coverage.is_retrievable(
            coverage.strip_tag(merged_block), conn)
        if not retrievable:
            report["rejected"].append({
                "kind": "merge",
                "why": f"coins term(s) absent from spine: {', '.join(coined[:4])}",
                "blocks": [m[:60] for m in members]})
            continue
        if missing or len(merged_block) > floor:
            report["rejected"].append({
                "kind": "merge",
                "why": (f"drops {len(missing)} hard token(s): {', '.join(missing[:4])}"
                        if missing else
                        f"saves only {len(union) - len(merged_block)} of "
                        f"{len(union)} chars — these are distinct facts, not duplicates"),
                "blocks": [m[:60] for m in members]})
            continue
        plan[members[0]] = merged_block
        merge_keys.add(members[0])
        dropped.extend(members[1:])
        report["merges"].append({"members": [m[:60] for m in members],
                                 "saved": len(union) - len(merged_block)})

    # ── Phase 2: compress, largest block first ───────────────────────
    # Largest-first is what makes this terminate quickly: the top 20 blocks hold
    # a third of the file, so the deficit closes before the small blocks (where
    # a 30% cut is 40 bytes and the guard rejects most of them) are ever touched.
    order = sorted(
        (b for b in original if b not in dropped and b not in plan
         and b not in ambiguous),
        key=lambda b: len(b.encode("utf-8")), reverse=True)

    def _size() -> int:
        return _blocks_bytes([plan.get(b, b) for b in original if b not in dropped])

    # Rounds, not one pass. One pass over the live file cut 4%: the model is
    # conservative on a first look at an already-terse entry, and gets bolder
    # once it sees its own output as the input. Each round re-verifies against
    # the ORIGINAL block, so N rounds cannot compound past MIN_KEEP_RATIO or
    # quietly shed a hard token that survived round 1.
    rounds = 0
    while _size() > target_bytes and rounds < MAX_ROUNDS:
        rounds += 1
        accepted_this_round = 0
        pending = sorted(order, key=lambda b: len(plan.get(b, b).encode("utf-8")),
                         reverse=True)
        while pending and _size() > target_bytes:
            batch = pending[:BATCH]
            pending = pending[BATCH:]
            bodies = {i: split_tag(plan.get(b, b))[1] for i, b in enumerate(batch)}
            proposals = _compress_batch(bodies, model)
            for i, block in enumerate(batch):
                proposal = proposals.get(i)
                current_text = plan.get(block, block)
                if proposal is None:
                    if rounds == 1:
                        report["rejected"].append({"kind": "compress",
                                                   "why": "no proposal returned",
                                                   "block": block[:60]})
                    continue
                tag, _ = split_tag(block)
                candidate = (tag + proposal).strip()
                # Shorter than where we are now; still faithful to where we
                # started. Both, or the round is a no-op for this block.
                ok, why = _accept(block, candidate, conn)
                if ok and len(candidate) >= len(current_text):
                    ok, why = False, f"no further gain ({len(candidate)} >= {len(current_text)})"
                if not ok:
                    if rounds == 1:
                        report["rejected"].append({"kind": "compress", "why": why,
                                                   "block": block[:60]})
                    continue
                plan[block] = candidate
                accepted_this_round += 1
        report.setdefault("rounds", []).append(
            {"round": rounds, "accepted": accepted_this_round, "bytes": _size()})
        if accepted_this_round == 0:
            # The model has nothing left to give. Another round is another 11
            # API calls for the same answer.
            break

    # Report merges and compressions separately. `plan` holds both, keyed the
    # same way, so the merge survivors have to be excluded by identity here or
    # every merge is also counted as a rewrite.
    for block, text in plan.items():
        if block in dropped or block in merge_keys or text == block:
            continue
        report["rewrites"].append({"block": block[:60],
                                   "before": len(block), "after": len(text)})

    projected = [plan.get(b, b) for b in original if b not in dropped]
    report["projected_bytes"] = _blocks_bytes(projected)
    report["projected_blocks"] = len(projected)
    report["reached_target"] = report["projected_bytes"] <= target_bytes

    if not plan:
        report["note"] = "no proposal passed verification — MEMORY.md left untouched"
        report["end_bytes"] = start_bytes
        return report
    if dry_run:
        report["note"] = "dry run — nothing written"
        report["end_bytes"] = start_bytes
        return report

    # ── Write-back: re-read under the lock, apply only what still matches ──
    with _hotcore_lock():
        live = _read_hotcore_blocks(path)
        live_set = set(live)
        # A concurrent memory() write cannot be reconciled with a rewrite of the
        # same block, so the rewrite loses. Skipping one compression is a bad
        # day; overwriting a just-written memory that exists nowhere else is
        # the failure this whole module is built around.
        skipped = [b for b in plan if b not in live_set] + [b for b in dropped if b not in live_set]
        new_blocks: List[str] = []
        for b in live:
            if b in dropped:
                continue
            new_blocks.append(plan.get(b, b))

        # Last line of defence, on the bytes about to be written: every hard
        # token in the file we read at the start must still be in the file we
        # are about to write. Catches any bookkeeping slip above, not just a
        # bad LLM proposal.
        before_tokens = hard_tokens("\n".join(original))
        after_text = "\n".join(new_blocks)
        lost = sorted(t for t in before_tokens if t not in after_text)
        if lost:
            report["note"] = (f"ABORTED before writing: {len(lost)} hard token(s) would be "
                              f"lost ({', '.join(lost[:6])})")
            report["end_bytes"] = _blocks_bytes(live)
            report["aborted"] = True
            return report

        backup = f"{path}.bak.hotcore-consolidate.{time.strftime('%Y%m%dT%H%M%S')}"
        shutil.copy2(path, backup)
        _write_hotcore_blocks(path, new_blocks)

    report["backup"] = backup
    report["skipped_concurrent"] = [b[:60] for b in skipped]
    report["applied"] = True
    report["end_bytes"] = os.path.getsize(path)
    report["end_blocks"] = len(new_blocks)
    return report


def _cluster(blocks: Sequence[str], config: SpineConfig) -> List[List[int]]:
    """Indices of near-duplicate block groups, or [] if the embedder is down.

    No embedder means no similarity, and guessing at duplicates by keyword
    overlap is exactly the mistake coverage.py's docstring documents. Skip the
    merge phase and let compression carry the pass.
    """
    from . import embedder

    if not embedder.embedder_available():
        logger.warning("embedder unavailable — skipping the merge phase")
        return []
    from .index import _cosine_similarity

    vecs = embedder.embed(list(blocks))
    if len(vecs) != len(blocks):
        return []
    used, clusters = set(), []
    for i in range(len(blocks)):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in range(i + 1, len(blocks)):
            if j in used:
                continue
            if _cosine_similarity(vecs[i], vecs[j]) > MERGE_SIMILARITY:
                cluster.append(j)
                used.add(j)
        if len(cluster) > 1:
            clusters.append(cluster)
    return clusters


def format_report(report: Dict[str, Any]) -> str:
    """Human-readable summary — this pass is run by hand, at 2am, by someone
    who wants to know what happened to their memory file in four lines."""
    lines = [f"🧠 Hot-core consolidation ({report.get('model')})"]
    start, end = report.get("start_bytes", 0), report.get("end_bytes", 0)
    lines.append(f"MEMORY.md: {start:,} → {end:,} bytes, "
                 f"{report.get('start_blocks', 0)} → "
                 f"{report.get('end_blocks', report.get('start_blocks', 0))} blocks")
    if report.get("merges"):
        lines.append(f"Merged {len(report['merges'])} duplicate group(s)")
    if report.get("rewrites"):
        saved = sum(r["before"] - r["after"] for r in report["rewrites"])
        lines.append(f"Compressed {len(report['rewrites'])} block(s), −{saved:,} chars")
    if report.get("rejected"):
        lines.append(f"Rejected {len(report['rejected'])} proposal(s) — originals kept")
    if report.get("skipped_concurrent"):
        lines.append(f"Skipped {len(report['skipped_concurrent'])} block(s) "
                     f"changed by another writer mid-pass")
    if report.get("backup"):
        lines.append(f"Backup: {report['backup']}")
    if report.get("note"):
        lines.append(f"Note: {report['note']}")
    if not report.get("reached_target", True):
        lines.append("⚠️  Still over target — re-run, or trim by hand with the "
                     "coverage.uncovered_hotcore check first.")
    return "\n".join(lines)
