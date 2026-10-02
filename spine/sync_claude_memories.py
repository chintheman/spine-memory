#!/usr/bin/env python3
"""Mirror Claude Code's curated memory files into spine as `agent:claude-code`.

The markdown files under Claude Code's memory directory are the SOURCE OF TRUTH.
This profile is a derived, searchable copy, so every run regenerates it from
scratch rather than trying to patch it. That makes the sync idempotent and makes
deletions propagate, which an append-only import cannot do.

Object ids are derived from the filename, so a file's row keeps its identity
across runs and an edit updates in place instead of creating a duplicate.

Run from cron:  python3 spine/sync_claude_memories.py
Preview only:   python3 spine/sync_claude_memories.py --dry-run
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sqlite3
import sys

PROFILE = "agent:claude-code"
# Stamped on every derived record; classify_gone() uses it to tell rows this
# sync owns from rows another writer put here.
EVIDENCE_PREFIX = "claude-code memory file: "
# Claude Code keys its per-project memory dir off the flattened absolute path
# of the project's cwd (e.g. /Users/you/... -> -Users-you-...). There's no
# real default for that key, so set SPINE_CLAUDE_PROJECT_KEY to your own.
_CLAUDE_PROJECT_KEY = os.environ.get("SPINE_CLAUDE_PROJECT_KEY", "-Users-youruser")
MEM_DIR = os.path.expanduser(f"~/.claude/projects/{_CLAUDE_PROJECT_KEY}/memory")
# MEMORY.md is the human-facing index, not a memory. Its content is just
# one-line pointers to the other files, so importing it would add 55 duplicate
# summaries that outrank the real entries.
SKIP = {"MEMORY.md"}

TYPE_MAP = {"user": "identity", "feedback": "preference",
            "project": "fact", "reference": "fact"}
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def stable_id(filename: str) -> str:
    """A deterministic 26-char id for a file, so re-syncs update rather than duplicate."""
    h = int.from_bytes(hashlib.sha256(filename.encode()).digest()[:16], "big")
    out = []
    for _ in range(26):
        out.append(CROCKFORD[h & 31])
        h >>= 5
    return "".join(reversed(out))


def parse(path: str):
    raw = open(path, encoding="utf-8").read()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not m:
        return None
    fm, body = m.group(1), m.group(2).strip()
    if not body:
        return None

    def field(key: str) -> str:
        r = re.search(rf"^{key}:\s*(.+)$", fm, re.M)
        return r.group(1).strip().strip('"').strip("'") if r else ""

    mtype = ""
    tm = re.search(r"^metadata:\s*\n((?:\s+.*\n?)*)", fm, re.M)
    if tm:
        t = re.search(r"^\s+type:\s*(.+)$", tm.group(1), re.M)
        mtype = t.group(1).strip() if t else ""

    desc = field("description")
    # Lead with the description: it is the file's own one-line summary and is
    # what a recall query most often matches against.
    content = (desc + "\n\n" + body).strip() if desc else body
    return {
        "name": field("name") or os.path.basename(path)[:-3],
        "content": content,
        "type": TYPE_MAP.get(mtype, "fact"),
        # [[wikilinks]] are this file's outbound edges; carried as topics so the
        # Phase 3 graph work has real edge data waiting for it.
        "links": sorted(set(re.findall(r"\[\[([^\]]+)\]\]", body))),
    }


def expected_row_count(prior, recs, skipped):
    """Rows this profile should hold after a sync.

    THE single source of this number. check_sync maintained its own copy and
    rounds 4 and 5 broke it in opposite directions: first counting only
    parseable files, then counting every unparseable file even when it had
    never produced a row. Same duplicated-invariant disease classify_gone cured
    inside this module.
    """
    gone = classify_gone(prior, recs, skipped)
    retained = {o for o in gone["skipped"] | gone["foreign"] if o in prior}
    return len(gone["built"] | retained)


def classify_gone(prior, recs, skipped):
    """Partition the ids in the previous log that are not in this run's records.

    ONE place decides what "gone" means. `prior`, `removed`, `skipped_ids` and
    the lock-window carry-through were four overlapping answers to that question,
    maintained independently, and each review round broke one by fixing another:
    round 3 resurrected deletions, round 4 then erased unparseable files.

      built    — a record was rebuilt this run; nothing to decide
      skipped  — the .md exists but failed to parse. NOT a deletion. Keep the
                 row: a frontmatter typo must not silently destroy the memory.
      deleted  — the .md is genuinely gone. Drop the row.
      foreign  — never in `prior`, so it arrived after this run's first read;
                 the lock-window carry-through must preserve it.
    """
    built = {r["id"] for r in recs}
    skipped_ids = {stable_id(f) for f in skipped}
    # Only rows THIS sync owns may be deleted. A row is ours if it carries the
    # evidence marker build_records() stamps on every record derived from a .md.
    # Rows without it came from another writer (handle_remember/handle_forget can
    # both target this profile) and were previously carried through the lock
    # window, landing in `deleted` on the very next run 4h later -- preserved
    # briefly, then silently dropped from the append-only store.
    ours = {oid for oid, rec in prior.items() if _is_derived(rec)}
    return {
        "built": built,
        "skipped": skipped_ids - built,
        "foreign": set(prior) - ours,
        "deleted": ours - built - skipped_ids,
    }


def _is_derived(rec):
    """True if this record was generated from a Claude Code .md file."""
    return any(str(e).startswith(EVIDENCE_PREFIX) for e in (rec.get("evidence") or []))


def is_foreign(oid, prior, built, skipped_ids):
    """True if `oid` appeared during the lock window and must be carried through."""
    return oid not in prior and oid not in built and oid not in skipped_ids


def merge_prior(rec, prior):
    """Fold a previous record into a freshly built one; returns the change state.

    Extracted so it is testable. Two things must survive a rebuild: `created_at`
    (otherwise every sync resets the age of the whole corpus and hands the
    recency scorer a store that looks brand new) and `status` (every record is
    rebuilt as "active", so a demote or a forget tombstone was reverted within
    four hours).
    """
    if prior is None:
        return "added"
    rec["created_at"] = prior.get("created_at", rec["created_at"])
    rec["status"] = prior.get("status", rec["status"])
    if prior.get("content") != rec["content"]:
        return "changed"
    rec["last_confirmed"] = prior.get("last_confirmed", rec["last_confirmed"])
    return "unchanged"


def build_records(now_iso: str):
    if not os.path.isdir(MEM_DIR):
        sys.exit(f"memory dir not found: {MEM_DIR}")
    recs, skipped = [], []
    for fn in sorted(os.listdir(MEM_DIR)):
        if not fn.endswith(".md") or fn in SKIP:
            continue
        p = parse(os.path.join(MEM_DIR, fn))
        if not p:
            skipped.append(fn)
            continue
        recs.append({
            "confidence": 0.95,
            "confirmations": 1,
            "content": p["content"],
            "contradicts": [],
            "created_at": now_iso,
            "epistemic": "extracted",
            "evidence": [f"{EVIDENCE_PREFIX}{fn}"],
            "id": stable_id(fn),
            "last_confirmed": now_iso,
            "last_retrieved": None,
            "profile": PROFILE,
            "status": "active",
            "supersedes": None,
            "topics": sorted(set([p["name"]] + p["links"])),
            "type": p["type"],
        })
    return recs, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # spine/__init__.py imports agent.memory_provider, so the repo root has to
    # be on the path too or a manual run dies with ModuleNotFoundError: agent.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from spine.config import load_spine_config
    from spine.index import MemoryIndex
    from spine.embedder import embedder_available, embed_single

    config = load_spine_config()
    out = os.path.join(os.path.expanduser(config.canonical_root), "observations",
                       f"{PROFILE}.jsonl")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    recs, skipped = build_records(now)
    if not recs:
        sys.exit("refusing to sync: parsed 0 memory files (source dir empty or unreadable?)")

    # Preserve created_at from the existing copy so a re-sync doesn't reset the
    # age of every memory and hand the recency scorer a corpus that looks new.
    # Merge patch lines the way load_observations() does. Reading them as plain
    # records (last-write-wins) meant a patched row -- which has no created_at --
    # fell through to `now`, RESETTING the observation's age, which is exactly
    # what the carry-over below exists to prevent. It also forced a re-embed.
    prior = {}
    prior_patches = {}
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "patch" in d:
                prior_patches.setdefault(d["id"], []).append(d["patch"])
            else:
                prior[d["id"]] = d
    for oid, plist in prior_patches.items():
        if oid in prior:
            for patch in plist:
                prior[oid].update(patch)
    added = changed = unchanged = 0
    for r in recs:
        p = prior.get(r["id"])
        state = merge_prior(r, p)
        if state == "added":
            added += 1
        elif state == "changed":
            changed += 1
        else:
            unchanged += 1
    # A file that failed to parse is NOT a deletion. Treating it as one turned
    # "you broke the frontmatter" into "this memory is silently unretrievable".
    removed = sorted(classify_gone(prior, recs, skipped)["deleted"])

    print(f"profile   : {PROFILE}")
    print(f"files     : {len(recs)} parsed, {len(skipped)} skipped {skipped}")
    print(f"added     : {added}")
    print(f"changed   : {changed}")
    print(f"unchanged : {unchanged}")
    print(f"removed   : {len(removed)}")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        return

    if not embedder_available():
        # Writing without vectors would silently downgrade this profile to
        # keyword-only, which is exactly how the embedder outage hid for 8 days.
        sys.exit("refusing to sync: embedder unavailable, would write rows without vectors")

    # Under the same lock JSONLWriter uses for this file. A concurrent status
    # patch would otherwise land on the old inode and be discarded by the
    # replace below.
    from spine.jsonl_writer import JSONLWriter
    writer = JSONLWriter(out)
    tmp = out + ".tmp"
    with writer._lock():
        # Re-read under the lock and re-apply anything that landed between the
        # first read and now. Locking only the write left the original race open.
        if os.path.exists(out):
            late_base, late_patch = {}, {}
            for line in open(out, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "patch" in d:
                    late_patch.setdefault(d["id"], []).append(d["patch"])
                else:
                    late_base[d["id"]] = d
            by_id = {r["id"]: r for r in recs}
            for oid, plist in late_patch.items():
                if oid in by_id:
                    for patch in plist:
                        if "status" in patch:
                            by_id[oid]["status"] = patch["status"]
            # A whole record appended between the first (unlocked) read and this
            # lock would otherwise be erased by the os.replace below. The rebuild
            # is derived from the .md files, so anything else in this log is
            # foreign and must be carried through verbatim.
            gone = classify_gone(prior, recs, skipped)
            for oid, rec in late_base.items():
                # Carry through ONLY genuinely-new records. Records whose .md was
                # deleted must not come back (round 3 resurrected them); records
                # whose .md merely failed to parse are re-added below, not here.
                if is_foreign(oid, prior, gone["built"], gone["skipped"]) \
                        and rec.get("profile") == PROFILE:
                    recs.append(rec)
            # An unparseable .md keeps its previous record verbatim. Dropping it
            # erased the memory from the source of truth and hard-failed the
            # sync on every run until the file was repaired.
            for oid in gone["skipped"] | gone["foreign"]:
                if oid in prior and oid not in {r["id"] for r in recs}:
                    recs.append(prior[oid])
        with open(tmp, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        os.replace(tmp, out)

    idx = MemoryIndex(config.db)
    idx.open()
    try:
        for r in recs:
            if r["id"] in prior and prior[r["id"]].get("content") == r["content"]:
                # Content identical and already indexed: skip the re-embed.
                row = idx.conn.execute(
                    "SELECT 1 FROM observations WHERE id=? AND embedding IS NOT NULL",
                    (r["id"],)).fetchone()
                if row:
                    continue
            idx.upsert_observation(r, embed_single(r["content"]))
        # Deletions must propagate: the index is upsert-only, so a file removed
        # from Claude Code would otherwise answer queries forever.
        for obs_id in removed:
            idx.conn.execute("DELETE FROM observations WHERE id=? AND profile=?",
                             (obs_id, PROFILE))
        idx.conn.commit()
        total = idx.conn.execute(
            "SELECT COUNT(*) FROM observations WHERE profile=?", (PROFILE,)).fetchone()[0]
        novec = idx.conn.execute(
            "SELECT COUNT(*) FROM observations WHERE profile=? AND embedding IS NULL",
            (PROFILE,)).fetchone()[0]
    finally:
        idx.close()

    print(f"\nindexed   : {total} rows in {PROFILE}, {novec} missing vectors")
    if novec:
        sys.exit(f"POST-SYNC CHECK FAILED: {novec} row(s) have no vector")
    if total != len(recs):
        sys.exit(f"POST-SYNC CHECK FAILED: {total} rows indexed, expected {len(recs)}")
    print("sync OK")


if __name__ == "__main__":
    main()
