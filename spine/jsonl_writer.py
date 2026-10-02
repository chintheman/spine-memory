"""Append-only JSONL writer with file locking — spec §2.

Each observation or episode is one JSON line. Mutations are patch lines
referencing the original by id. The file lock ensures only one writer
modifies the file at a time (across threads/processes).

Patch-line format (appended, never modifies existing lines):
  {"id":"01J...","patch":{"confidence":0.92},"ts":"..."}
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JSONLWriter:
    """Append-only writer with file locking for a single JSONL log."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _lock(self):
        """Acquire an exclusive lock on the JSONL file."""
        lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def append(self, record: Dict[str, Any]) -> None:
        """Append a single JSON object as one line. Thread/process-safe."""
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock():
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all lines, parse as JSON. Returns list of dicts."""
        if not self._path.exists():
            return []
        lines = []
        with open(self._path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if raw:
                    try:
                        lines.append(json.loads(raw))
                    except json.JSONDecodeError as e:
                        logger.warning("Skipping malformed JSONL line in %s: %s", self._path, e)
        return lines

    def read_entry(self, obs_id: str) -> Optional[Dict[str, Any]]:
        """Find an entry by id (base record), applying any patches."""
        records = self.read_all()
        base = None
        patches: List[Dict[str, Any]] = []

        for r in records:
            if r.get("id") == obs_id:
                if "patch" in r:
                    patches.append(r)
                else:
                    base = r

        if base is None:
            return None

        # Apply patches in order
        result = dict(base)
        for p in patches:
            patch_data = p["patch"]
            for key, val in patch_data.items():
                result[key] = val
            result["last_patched"] = p.get("ts", "")
            result["patch_count"] = result.get("patch_count", 0) + 1

        return result

    def count(self) -> int:
        """Return number of base records (not patches)."""
        records = self.read_all()
        return sum(1 for r in records if "patch" not in r)

    def backup_and_compact(self) -> str:
        """Rewrite the log removing superseded/archived entries. Returns backup path.

        Not currently called anywhere (checked 2026-07-30) — available as a
        manual/occasional maintenance utility, not wired into the nightly
        consolidation loop. Left that way deliberately: auto-wiring a
        file-rewrite into an automated cron is a real behavior change (this
        class is otherwise strictly append-only, by design, per the module
        docstring), not something to add silently as part of a cleanup pass.
        """
        import shutil

        ts = int(time.time())
        bak = str(self._path) + f".bak.{ts}"
        with self._lock():
            if self._path.exists():
                shutil.copy2(self._path, bak)
            # Rewrite active entries only
            records = self.read_all()
            active = [r for r in records if r.get("status") not in ("archived", "superseded")]
            with open(self._path, "w", encoding="utf-8") as f:
                for r in active:
                    f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        return bak
