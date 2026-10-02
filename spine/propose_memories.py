#!/usr/bin/env python3
"""Propose Hermes memories for promotion into Claude Code's curated memory set.

Memory flows one way today: Claude Code's .md files sync INTO spine every 4h,
and nothing comes back. Hermes forms memories all day that Claude Code can never
see. This is the return leg.

    propose_memories.py --dry-run     # counts and a sample, writes nothing
    propose_memories.py               # write proposal files for review
    propose_memories.py --mark <id>   # record that a row was promoted

Hermes PROPOSES; it never writes into ~/.claude. Proposals land in
~/.hermes/proposals/claude-memory/ (inside the git-tracked ~/.hermes) and a
human-reviewed Claude Code session does the actual memory-file write. That
directory has no version control and no undo, so nothing automated may touch it.

## Why the filter looks the way it does

Two measured facts drove it, both counter-intuitive:

* `status='active' AND confidence >= 0.9` selects ZERO rows and always will.
  Hermes's own nightly promote pass drains everything >= 0.9 out of `active`,
  then the demote pass flips it to `demoted`. Max confidence among active rows
  is 0.838. `demoted` is not a quality signal -- it means "already passed
  through Hermes's hot core once", which is where the good material lives.
  A write path gated the obvious way would promote nothing, forever, silently.

* Confidence does not separate signal from noise. "HARD CORRECTION: never claim
  build progress without on-disk verification" and "User shared 4 Instagram
  reels on Jul 19" are both 0.95. `type` separates them: `fact` is 254 of 423
  rows and is where every transient event log lives.

So: filter on `type`, not confidence, and do not gate on `status`.
`confirmations` is also useless here -- 422 of 423 rows have exactly 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

SOURCE_PROFILE = "agent:main"
TARGET_PROFILE = "agent:claude-code"
PROPOSAL_DIR = os.path.expanduser("~/.hermes/proposals/claude-memory")

# Stamped into `evidence` when a row has been promoted, so it is never proposed
# twice. There is no dedicated column, and the 4h sync round-trip would
# otherwise duplicate: a promoted file returns as an agent:claude-code
# observation while agent:main still holds the original.
EXPORT_PREFIX = "exported to claude-code memory: "

# Types worth considering. `fact` is excluded deliberately -- see the module
# docstring. `identity` is included because a wrong identity fact makes a
# session behave wrongly, which is the bar.
CANDIDATE_TYPES = ("correction", "preference", "pattern", "identity")


def connect(db: str) -> sqlite3.Connection:
    from spine.index import connect_db
    con = connect_db(f"file:{db}?mode=ro", uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def already_exported(evidence_raw) -> bool:
    try:
        ev = json.loads(evidence_raw) if isinstance(evidence_raw, str) else (evidence_raw or [])
    except (json.JSONDecodeError, TypeError):
        return False
    return any(str(e).startswith(EXPORT_PREFIX) for e in ev or [])


def stage_a(con: sqlite3.Connection):
    """SQL pre-filter. No status gate, no confidence gate, no confirmations."""
    rows = con.execute(
        f"""SELECT id, type, content, confidence, status, created_at, evidence
            FROM observations
            WHERE profile = ?
              AND type IN ({','.join('?' * len(CANDIDATE_TYPES))})
              AND status NOT IN ('superseded', 'archived')
            ORDER BY
              CASE type WHEN 'correction' THEN 0 WHEN 'preference' THEN 1
                        WHEN 'pattern' THEN 2 ELSE 3 END,
              confidence DESC""",
        (SOURCE_PROFILE, *CANDIDATE_TYPES),
    ).fetchall()
    return [r for r in rows if not already_exported(r[6])]


def stage_b(con: sqlite3.Connection, rows):
    """Drop anything Claude Code's memory files already cover."""
    from spine.coverage import covered_by_profile
    fresh, dupes = [], []
    for r in rows:
        covered, _ = covered_by_profile(r[2], con, TARGET_PROFILE, n=4)
        (dupes if covered else fresh).append(r)
    return fresh, dupes


def mark_exported(db: str, obs_id: str, filename: str) -> None:
    """Append the export marker to BOTH stores, so a rebuild cannot undo it."""
    from spine.config import load_spine_config
    cfg = load_spine_config()
    stamp = f"{EXPORT_PREFIX}{filename}"

    from spine.index import connect_db
    con = connect_db(db)
    try:
        row = con.execute("SELECT evidence, profile FROM observations WHERE id=?",
                          (obs_id,)).fetchone()
        if not row:
            sys.exit(f"propose_memories: no such observation {obs_id}")
        try:
            ev = json.loads(row[0]) if row[0] else []
        except (json.JSONDecodeError, TypeError):
            ev = []
        if stamp not in ev:
            ev.append(stamp)
        con.execute("UPDATE observations SET evidence=? WHERE id=?",
                    (json.dumps(ev), obs_id))
        con.commit()
        profile = row[1]
    finally:
        con.close()

    # The DB is derived; the JSONL is the source of truth. Without this the next
    # rebuild-index.py run silently drops the marker and the row gets proposed
    # again -- the same class of bug as the status write-back.
    from spine.jsonl_writer import JSONLWriter
    path = os.path.join(os.path.expanduser(cfg.canonical_root), "observations",
                        f"{profile}.jsonl")
    if os.path.exists(path):
        JSONLWriter(path).append({
            "id": obs_id,
            "patch": {"evidence": ev},
            "ts": datetime.now(timezone.utc).isoformat(),
        })
    print(f"marked exported: {obs_id} -> {filename}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="counts and a sample; writes nothing")
    ap.add_argument("--mark", metavar="OBS_ID",
                    help="record that this observation was promoted")
    ap.add_argument("--filename", default="",
                    help="the memory filename it became (used with --mark)")
    ap.add_argument("--limit", type=int, default=0, help="cap the candidate list")
    a = ap.parse_args()

    from spine.config import load_spine_config
    db = os.path.expanduser(load_spine_config().db)

    if a.mark:
        if not a.filename:
            sys.exit("propose_memories: --mark requires --filename")
        mark_exported(db, a.mark, a.filename)
        return

    con = connect(db)
    try:
        rows = stage_a(con)
        fresh, dupes = stage_b(con, rows)
    finally:
        con.close()

    if a.limit:
        fresh = fresh[:a.limit]

    by_type = {}
    for r in fresh:
        by_type[r[1]] = by_type.get(r[1], 0) + 1

    print(f"stage A (type in {CANDIDATE_TYPES}, not exported) : {len(rows)}")
    print(f"stage B dropped as already covered by {TARGET_PROFILE}: {len(dupes)}")
    print(f"remaining for judgement                            : {len(fresh)}")
    print(f"  by type: {by_type}")
    print()
    print("Stage C is a judgement call and is NOT automated: for each candidate,")
    print("would a session not knowing this do something WRONG, and can a")
    print("description be written that makes it findable? Target 15-25 survivors.")

    if a.dry_run:
        print("\n--- sample (first 12) ---")
        for r in fresh[:12]:
            print(f"  [{r[1]:<10} {r[3]:.2f}] {r[0][:10]} {r[2][:110]}".replace("\n", " "))
        print("\nDRY RUN — nothing written.")
        return

    os.makedirs(PROPOSAL_DIR, exist_ok=True)
    out = os.path.join(PROPOSAL_DIR, "candidates.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump([{"id": r[0], "type": r[1], "content": r[2],
                    "confidence": r[3], "status": r[4], "created_at": r[5]}
                   for r in fresh], f, indent=2)
    print(f"\nwrote {len(fresh)} candidates to {out}")


if __name__ == "__main__":
    main()
