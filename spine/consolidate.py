#!/usr/bin/env python3
"""Nightly consolidation runner — spec §5.2.

Called by cron at 4:00 AM SGT. Runs all five passes, writes report
to wiki _system/ and delivers summary to #11.

Usage:
    python3 consolidate.py
    python3 consolidate.py --dry-run
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spine consolidation cycle")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't modify")
    args = parser.parse_args()

    from spine.config import load_spine_config
    from spine.loops import run_consolidation

    config = load_spine_config()

    if args.dry_run:
        print("Dry run — would consolidate")
        idx_path = config.db
        print(f"  DB: {idx_path}")
        print(f"  Promote threshold: {config.promote_auto_min_confidence}")
        print(f"  Archive threshold: {config.archive_threshold}")
        return

    print(f"Consolidation starting at {datetime.now(timezone.utc).isoformat()}...")
    report = run_consolidation(config)
    print(json.dumps(report, indent=2))

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


if __name__ == "__main__":
    main()
