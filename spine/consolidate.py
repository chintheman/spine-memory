#!/usr/bin/env python3
"""Nightly consolidation runner — spec §5.2.

Called by cron at 4:00 AM SGT. Runs all five passes, writes report
to wiki _system/ and delivers summary to #11.

Usage:
    python3 consolidate.py
    python3 consolidate.py --dry-run
    python3 consolidate.py --hotcore              # LLM-assisted MEMORY.md shrink
    python3 consolidate.py --hotcore --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

SPINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPINE_DIR.parent.parent.parent))
# plugins/memory must be on the path too, or `python3 consolidate.py`
# (this file's own documented usage) dies on `import spine`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spine consolidation cycle")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't modify")
    parser.add_argument(
        "--hotcore", action="store_true",
        help="Run the LLM-assisted hot-core pass instead of the nightly cycle: "
             "merge and compress MEMORY.md blocks until the file is under budget. "
             "This is what the demote pass means when it warns that spine cannot "
             "shrink the file on its own.")
    args = parser.parse_args()

    from spine.config import load_spine_config
    from spine.loops import run_consolidation

    config = load_spine_config()

    if args.hotcore:
        from spine.hotcore_consolidate import consolidate_hotcore, format_report as hotcore_report
        report = consolidate_hotcore(config, dry_run=args.dry_run)
        print(hotcore_report(report))
        # Exit non-zero when the file is still over budget, so a cron or a shell
        # loop can tell "shrank it" from "tried and could not". The heartbeat
        # already treats over-budget as a failure; this must not disagree.
        raise SystemExit(0 if report.get("reached_target", False) else 1)

    if args.dry_run:
        print("Dry run — would consolidate")
        idx_path = config.db
        print(f"  DB: {idx_path}")
        print(f"  Promote threshold: {config.promote_auto_min_confidence}")
        print(f"  Archive threshold: {config.archive_threshold}")
        return

    print(f"Consolidation starting at {datetime.now(timezone.utc).isoformat()}...")
    report = run_consolidation(config)
    print(format_report(report))

    # Save report to wiki
    report_dir = Path(config.canonical_root).parent / "_system"
    report_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = report_dir / f"consolidation-{date_str}.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved: {report_path}")
    print(f"Active observations: {report.get('active_count', 0)}")
    print("Done.")


def format_report(report: Dict[str, Any]) -> str:
    """Human-readable summary for Telegram delivery (cron no_agent stdout).

    Replaces the raw JSON dict dump — the nightly delivery goes verbatim to
    topic #11, and a wall of JSON is noise. Report JSON is still saved to wiki.
    """
    passes = report.get("passes", {})
    ts = report.get("ts") or datetime.now(timezone.utc).isoformat()
    when = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M %Z")

    lines = [
        f"🧠 Spine Consolidation — {when}",
        "Memory hygiene pass complete:",
    ]
    for key, label in [
        ("decay", "Decay"),
        ("merge", "Merge"),
        ("semantic_merge", "Semantic merge"),
        ("contradict", "Contradictions"),
        ("promote", "Promoted"),
        ("demote", "Demoted"),
    ]:
        if key in passes:
            lines.append(f"• {label}: {passes[key]}")

    contradictions = report.get("contradictions") or []
    if contradictions:
        lines.append("")
        lines.append("⚠️ Contradictions flagged:")
        for c in contradictions[:5]:
            lines.append(f"• [{c.get('a_id')}] vs [{c.get('b_id')}] — {c.get('reason', 'opposition')}")
            resolution = c.get("resolution", "")
            if resolution:
                lines.append(f"  {resolution}")
        if len(contradictions) > 5:
            lines.append(f"  …and {len(contradictions) - 5} more")

    promoted = report.get("promoted_items") or []
    if promoted:
        lines.append("")
        lines.append(f"Promoted to MEMORY.md ({len(promoted)}):")
        for item in promoted[:5]:
            lines.append(f"• {item}")
        if len(promoted) > 5:
            lines.append(f"  …and {len(promoted) - 5} more (revert via forget() if wrong)")

    # Warnings were written to the JSON report but never rendered, so the demote
    # pass's "say so loudly" message never reached the nightly Telegram delivery
    # that is the only thing anyone actually reads.
    for w in report.get("warnings") or []:
        lines.append("")
        lines.append(f"⚠️  {w}")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
