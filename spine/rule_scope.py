"""Decide which hot-core blocks a turn actually needs.

MEMORY.md is loaded whole into every call. Measured 2026-09-05, it sits in a
21,000-25,000 byte band and grows ~1,800/day net. Most of that is paid for when
its trigger is nowhere in sight: the screenshot rule is loaded during a crypto
conversation, the Garmin rule while debugging a cron.

A block declares when it is relevant with a `@when:` marker. Four kinds:

    (no marker)                      universal — always loaded. THE DEFAULT.
    @when:subject=ledger,prediction  fires when the query mentions a term
    @when:tool=garmin_*              fires when a matching tool is called
    @when:image                      fires when the payload carries an image

Two rules that keep this safe:

* **Untagged means universal.** A block nobody has classified stays hot. The
  cost of a stale byte is tokens; the cost of a missing behavioural rule is the
  agent misbehaving with nobody able to say why.
* **[R] blocks are never deferred**, whatever they are tagged with. They are the
  constitution, all ten of them, 3,019 bytes. Enforced here rather than trusted
  to the tagger.

Selection is deterministic on purpose. Tool and image triggers are facts — the
tool is being called or it is not, the payload has an image or it does not. No
similarity threshold, nothing to silently miss. Subject triggers carry their own
match terms rather than consulting a central taxonomy, so a block is readable on
its own and there is no hidden table to drift.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

WHEN_RE = re.compile(r"@when:([a-z]+)(?:=([^\s\]]+))?", re.IGNORECASE)

UNIVERSAL = "universal"
SUBJECT = "subject"
TOOL = "tool"
IMAGE = "image"

VALID_KINDS = {UNIVERSAL, SUBJECT, TOOL, IMAGE}

# Trigger kinds that something can actually DELIVER once they are filtered out
# of the frozen snapshot.
#
# prefetch()/queue_prefetch() receive only the user's query — no tool name, no
# attachment info — so today only SUBJECT can be delivered. Deferring a TOOL or
# IMAGE block would remove it from the snapshot with nothing able to bring it
# back: the rule would be silently deleted, which is the one outcome this whole
# design exists to prevent. Caught on 2026-09-05 by an end-to-end check, after
# the code had already been written to defer all three.
#
# TOOL landed 2026-09-05: the hotcore-rules plugin registers a
# transform_tool_result hook that appends a tool's rules to that tool's result.
# IMAGE still needs pre_llm_call, and is additionally blocked by
# compose_user_api_content returning None for list-shaped content — on an image
# turn the memory block is dropped entirely, which is exactly the turn the
# screenshot rule needs. Move a kind in here only once its hook lands.
DELIVERABLE_KINDS = {SUBJECT, TOOL}


@dataclass(frozen=True)
class Trigger:
    kind: str
    values: Tuple[str, ...] = ()

    def is_universal(self) -> bool:
        return self.kind == UNIVERSAL


@dataclass
class TurnContext:
    """What is observably true about the turn being assembled."""

    query: str = ""
    tool_name: str = ""
    has_image: bool = False

    def normalised_query(self) -> str:
        return (self.query or "").lower()


def parse_trigger(block: str) -> Trigger:
    """Read a block's `@when:` marker. No marker, or an unknown kind, is universal.

    An unknown kind deliberately falls back to universal rather than raising:
    a typo'd tag must not silently delete a rule from every prompt. The
    heartbeat reachability check is what surfaces the typo.
    """
    m = WHEN_RE.search(block or "")
    if not m:
        return Trigger(UNIVERSAL)
    kind = (m.group(1) or "").lower()
    if kind not in VALID_KINDS:
        return Trigger(UNIVERSAL)
    raw = m.group(2) or ""
    values = tuple(v.strip().lower() for v in raw.split(",") if v.strip())
    if kind in (SUBJECT, TOOL) and not values:
        # "@when:subject" with nothing to match on can never fire. Universal.
        return Trigger(UNIVERSAL)
    return Trigger(kind, values)


def fires(trigger: Trigger, ctx: TurnContext) -> bool:
    """Is this trigger satisfied by what the turn actually contains?"""
    if trigger.is_universal():
        return True
    if trigger.kind == IMAGE:
        return bool(ctx.has_image)
    if trigger.kind == TOOL:
        name = (ctx.tool_name or "").lower()
        if not name:
            return False
        return any(fnmatch.fnmatch(name, pat) for pat in trigger.values)
    if trigger.kind == SUBJECT:
        q = ctx.normalised_query()
        if not q:
            return False
        # Word-ish boundary so "led" does not match inside "ledger".
        return any(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", q)
                   for term in trigger.values)
    return True


def is_rule_block(block: str) -> bool:
    """[R] blocks are the constitution and are never deferred."""
    return (block or "").lstrip().startswith("[R]")


def split_blocks(text: str) -> List[str]:
    """Hot-core blocks are separated by a lone § line."""
    return [b.strip() for b in (text or "").split("\n§\n") if b.strip()]


def select(blocks: Sequence[str], ctx: TurnContext) -> Tuple[List[str], List[str]]:
    """Partition blocks into (hot, deferred) for this turn.

    hot      — must be in the prompt now
    deferred — not relevant to this turn
    """
    hot: List[str] = []
    deferred: List[str] = []
    for b in blocks:
        if is_rule_block(b):
            hot.append(b)
            continue
        trig = parse_trigger(b)
        (hot if fires(trig, ctx) else deferred).append(b)
    return hot, deferred


def reachable_kinds() -> set:
    """Trigger kinds `fires()` can actually satisfy — used by the heartbeat.

    A block tagged with a kind absent from this set would never load. That is
    the failure mode worth alarming on, because it is invisible from the outside.
    """
    return set(VALID_KINDS)


def summarise(blocks: Sequence[str], ctx: TurnContext) -> Dict[str, Any]:
    """Shadow-mode record: what WOULD have been deferred this turn."""
    hot, deferred = select(blocks, ctx)
    return {
        "query_len": len(ctx.query or ""),
        "tool": ctx.tool_name or None,
        "has_image": ctx.has_image,
        "hot_blocks": len(hot),
        "deferred_blocks": len(deferred),
        "hot_bytes": sum(len(b) for b in hot),
        "deferred_bytes": sum(len(b) for b in deferred),
        "deferred_triggers": sorted({
            f"{parse_trigger(b).kind}:{','.join(parse_trigger(b).values)}"
            for b in deferred
        }),
    }


def deliverable(blocks: Sequence[str], ctx: TurnContext) -> List[str]:
    """Blocks this turn needs that the frozen snapshot does not already carry.

    The snapshot is built with an empty TurnContext, so it holds exactly the
    universal and [R] blocks. Anything that fires for a real turn but is not in
    that set has to be delivered by prefetch(), or a tagged rule would simply
    vanish once scope filtering is on.

    Returned in the file's own order so repeated turns produce a stable string.
    """
    always_set = set(snapshot_keep(blocks))
    now, _ = select(blocks, ctx)
    return [b for b in now if b not in always_set]


def snapshot_keep(blocks: Sequence[str]) -> List[str]:
    """Entries the frozen system-prompt snapshot must carry.

    Universal entries and [R] rules always. PLUS any block whose trigger kind
    has no delivery path yet — filtering one of those out would delete a rule.
    """
    keep: List[str] = []
    for b in blocks:
        if is_rule_block(b):
            keep.append(b)
            continue
        trig = parse_trigger(b)
        if trig.is_universal() or trig.kind not in DELIVERABLE_KINDS:
            keep.append(b)
    return keep
