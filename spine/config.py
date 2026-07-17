"""Spine configuration reader — reads from config.yaml memory.spine block.

Fallback chain: config.yaml → defaults → sensible hard-coded values.
All paths resolve relative to HERMES_HOME.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class SpineConfig:
    """Resolved spine configuration, defaults applied."""

    canonical_root: str = "~/wiki/_memory"
    db: str = "~/.hermes/memory.db"
    embedder: str = "all-MiniLM-L6-v2"
    loop_model: str = "deepseek-v4-pro"
    consolidation_cron: str = "0 4 * * *"
    report_target: str = "telegram:-1004389869012:11"
    decay_per_30d: float = 0.05
    archive_threshold: float = 0.3
    promote_auto_epistemic: str = "extracted"
    promote_auto_min_confidence: float = 0.9
    episode_compact_days: int = 90
    hermes_home: str = "~/.hermes"


def _resolve_path(raw: str, hermes_home: str) -> str:
    """Resolve ~ and $HOME to absolute paths."""
    home = hermes_home if hermes_home else os.environ.get("HERMES_HOME", os.path.expanduser("~"))
    raw = raw.replace("~/.hermes", home) if raw.startswith("~/") else raw
    return os.path.expanduser(raw)


def load_spine_config(hermes_home: str = "") -> SpineConfig:
    """Read spine config from Hermes config.yaml, fill with defaults.

    Attempts a clean import from hermes_cli; falls back to file reading.
    """
    cfg = SpineConfig(hermes_home=hermes_home if hermes_home else os.environ.get("HERMES_HOME", "~/.hermes"))

    # Attempt to read from config.yaml
    try:
        from hermes_cli.config import load_config

        raw = load_config()
        spine = raw.get("memory", {}).get("spine", {})
    except Exception:
        spine = {}

    if spine:
        cfg.canonical_root = spine.get("canonical_root", cfg.canonical_root)
        cfg.db = spine.get("db", cfg.db)
        cfg.embedder = spine.get("embedder", cfg.embedder)
        cfg.loop_model = spine.get("loop_model", cfg.loop_model)
        cfg.consolidation_cron = spine.get("consolidation_cron", cfg.consolidation_cron)
        cfg.report_target = spine.get("report_target", cfg.report_target)
        cfg.decay_per_30d = float(spine.get("decay_per_30d", cfg.decay_per_30d))
        cfg.archive_threshold = float(spine.get("archive_threshold", cfg.archive_threshold))
        cfg.promote_auto_min_confidence = float(spine.get("promote_auto", {}).get("min_confidence", cfg.promote_auto_min_confidence))
        cfg.promote_auto_epistemic = spine.get("promote_auto", {}).get("epistemic", cfg.promote_auto_epistemic)
        cfg.episode_compact_days = int(spine.get("episode_compact_days", cfg.episode_compact_days))

    # Resolve paths
    cfg.canonical_root = _resolve_path(cfg.canonical_root, cfg.hermes_home)
    cfg.db = _resolve_path(cfg.db, cfg.hermes_home)

    # Ensure directories exist
    Path(cfg.canonical_root).mkdir(parents=True, exist_ok=True)
    for sub in ["observations", "episodes", "bench", "archive"]:
        Path(cfg.canonical_root, sub).mkdir(parents=True, exist_ok=True)

    return cfg
