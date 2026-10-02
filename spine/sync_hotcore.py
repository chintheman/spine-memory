#!/usr/bin/env python3
"""Import MEMORY.md blocks that spine has never seen.

MEMORY.md has two writers. Spine's promote pass is one; Hermes's own memory()
tool is the other, and it writes straight into the hot core without ever
touching spine. So the always-loaded file steadily accumulates facts that exist
nowhere else -- 15 of them on 2026-08-20, which a "these are already in spine,
just delete them" cleanup would have destroyed outright.

This closes that leak: anything in the hot core that is not retrievable becomes
a spine observation, so the hot core can be trimmed safely at any time.

Imported rows are written as `demoted`: searchable, but never re-promoted back
into the file by the consolidation pass.

Silent when there is nothing to do.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

# Path setup MUST precede the spine import. The same 2026-08-30 change that
# broke heartbeat.py broke this file: a module-level `from spine.index import`
# was added while the sys.path inserts stayed inside main(), so running it as a
# script died with ModuleNotFoundError before main() was reached. This one is
# worse than the heartbeat, because it is the guard that imports hot-core
# blocks which exist nowhere else -- a MEMORY.md trim run while this is broken
# destroys them permanently.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# spine/__init__.py imports agent.memory_provider, so the repo root has to be on
# the path too or a manual run dies with ModuleNotFoundError: agent.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from spine.index import connect_db  # noqa: E402

PROFILE = "agent:main"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # sys.path is set at module level, above the spine import.
    from spine.config import load_spine_config
    from spine import coverage
    from spine.embedder import embedder_available
    from spine.tools import handle_remember

    config = load_spine_config()
    mem_path = os.path.expanduser("~/.hermes/memories/MEMORY.md")
    con = connect_db(os.path.expanduser(config.db))
    con.execute("PRAGMA query_only = ON")
    try:
        uncovered = coverage.uncovered_hotcore(mem_path, con)
        total = len(coverage.hotcore_blocks(mem_path))
    finally:
        con.close()

    if not uncovered:
        if args.dry_run:
            print(f"hot core: {total} blocks, all retrievable from spine")
        return

    print(f"hot core: {total} blocks, {len(uncovered)} not retrievable from spine")
    for block, missing in uncovered:
        print(f"  + {block[:88].replace(chr(10), ' ')}")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        return

    if not embedder_available():
        # Importing without vectors would put these rows in the store but leave
        # them findable by keyword only, which is the quiet half-failure this
        # whole exercise exists to prevent.
        sys.exit("refusing to import: embedder unavailable, rows would have no vectors")

    ids, withheld = [], []
    for block, _missing in uncovered:
        content = coverage.strip_tag(block)
        r = json.loads(handle_remember({
            "content": content,
            "type": "fact",
            "epistemic": "extracted",
            "confidence": 0.9,
            "topics": [],
            "evidence": ["MEMORY.md hot core, imported by sync_hotcore"],
        }, config))
        if r.get("success"):
            ids.append(r["id"])
        elif r.get("held") or r.get("rejected"):
            # The secrets detector holds on entropy and blocks 12-word prose.
            # The owner's memory notes are full of file paths, and a path like
            # /.hermes/skills/productivity/tracking-board/SKILL.md trips the
            # entropy hold. That is a fine tradeoff for a user-initiated write,
            # where the rejection is visible and recoverable; as a CRON gate it
            # meant one such block failed the run forever with nothing queued.
            # Report it and carry on.
            withheld.append((content[:70],
                             r.get("tokens") or r.get("blocked_by") or "hard block"))
        else:
            print(f"  ! failed: {content[:70]} -> {r}")

    # Demoted, not active: these have already had their turn in the hot core and
    # must not be promoted back into the file they came from.
    from spine.index import MemoryIndex
    idx = MemoryIndex(config.db)
    idx.open()
    try:
        for oid in ids:
            idx.update_status(oid, "demoted")   # writes DB + canonical JSONL
        idx.conn.commit()
    finally:
        idx.close()

    # Verify rather than assume: re-run the same coverage test.
    con = connect_db(os.path.expanduser(config.db))
    con.execute("PRAGMA query_only = ON")
    try:
        still = coverage.uncovered_hotcore(mem_path, con)
    finally:
        con.close()

    print(f"\nimported {len(ids)}, still uncovered: {len(still)}")
    for content, why in withheld:
        print(f"  ~ withheld by the secrets detector ({why}): {content}")
    # Withheld blocks can never become retrievable by this route, so excluding
    # them is what makes the assertion meaningful rather than permanently red.
    unresolved = len(still) - len(withheld)
    if unresolved > 0:
        sys.exit(f"POST-IMPORT CHECK FAILED: {unresolved} block(s) remain unretrievable")
    if withheld:
        print(f"{len(withheld)} block(s) withheld — review, then trim or reword them")
        return
    print("hot core fully retrievable")


if __name__ == "__main__":
    main()
