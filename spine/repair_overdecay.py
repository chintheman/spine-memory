#!/usr/bin/env python3
"""One-off repair for the quadratic-decay runaway (found 2026-08-30).

loops.py's decay pass subtracted (days_idle / 30) * decay_per_30d from the
ALREADY-decayed stored confidence on every nightly run, so idle observations
lost confidence quadratically instead of the documented -0.05 per 30 idle
days. 53 agent:main observations were archived between 2026-08-19 and
2026-08-29 that would all still be >= 0.3 under the intended schedule, and
98 of 178 active+archived rows sat a mean 0.32 below their linear value.

This script recomputes every active/archived observation of a profile from
its base confidence in the canonical JSONL (append-only source of truth,
never touched by decay) as:

    linear = max(0, base - idle_days / 30 * decay_per_30d)

where idle_days runs from last_confirmed (fallback last_retrieved, then
created_at) to now. Rows that were archived by the runaway and come out
>= archive_threshold are re-activated through MemoryIndex.update_status(),
which writes the status patch back to the JSONL so a rebuild-index cannot
resurrect the wrong state.

Not reconstructed: the contradiction pass's +/-0.05/-0.15 nudges. Those are
small, only 31 pairs were ever flagged, and the reports only carry 8-char id
prefixes -- not worth replaying. Run with --dry-run first.

Usage:
    venv/bin/python repair_overdecay.py [--profile agent:main] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SPINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPINE_DIR.parent))
sys.path.insert(0, str(SPINE_DIR.parent.parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="agent:main")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from spine.config import load_spine_config
    from spine.index import MemoryIndex

    cfg = load_spine_config()
    jsonl = Path(os.path.expanduser(cfg.canonical_root)) / "observations" / f"{args.profile}.jsonl"
    base = {}
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "patch" not in d and "confidence" in d:
                base[d["id"]] = float(d["confidence"])

    idx = MemoryIndex(cfg.db)
    idx.open()
    now = datetime.now(timezone.utc)
    rows = idx.conn.execute(
        "SELECT id, status, confidence, last_confirmed, last_retrieved, created_at, type"
        " FROM observations WHERE profile=? AND status IN ('active','archived')",
        (args.profile,),
    ).fetchall()

    updated = restored = skipped = 0
    try:
        for obs_id, status, conf, lc, lr, ca, typ in rows:
            if obs_id not in base:
                skipped += 1
                continue
            ref = lc or lr or ca
            try:
                idle_days = max(0.0, (now - datetime.fromisoformat(ref)).total_seconds() / 86400.0)
            except (ValueError, TypeError):
                skipped += 1
                continue
            if typ == "correction":
                linear = base[obs_id]  # corrections never decay (loops.py)
            else:
                linear = max(0.0, base[obs_id] - idle_days / 30.0 * cfg.decay_per_30d)
            linear = round(linear, 4)
            if abs(linear - (conf or 0.0)) > 1e-6:
                updated += 1
                if not args.dry_run:
                    idx.conn.execute("UPDATE observations SET confidence=? WHERE id=?", (linear, obs_id))
            if status == "archived" and linear >= cfg.archive_threshold:
                restored += 1
                print(f"  restore {obs_id[:8]} {typ:<10} base={base[obs_id]:.2f} idle={idle_days:5.1f}d "
                      f"db={conf:.3f} -> {linear:.3f}")
                if not args.dry_run:
                    idx.update_status(obs_id, "active")
        if not args.dry_run:
            # Stamp the decay clock so tonight's run charges only from now.
            idx.conn.execute(
                "INSERT OR REPLACE INTO dim_meta (key, value) VALUES (?, ?)",
                (f"decay_last_run:{args.profile}", now.isoformat()),
            )
            idx.conn.commit()
    finally:
        idx.close()

    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print(f"{mode}: profile={args.profile} rows={len(rows)} confidence_updated={updated} "
          f"restored_to_active={restored} skipped_no_base={skipped}")


if __name__ == "__main__":
    main()
