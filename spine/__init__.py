"""Spine — Hermes Memory Provider v2 (plugins/memory/spine).

Implements the MemoryProvider ABC with canonical JSONL logs, a derived
SQLite FTS5+vec index, and four agent tools (remember, recall, reflect, forget).
Three loops (observer, consolidation, activation manifest) run via plugin hooks.

Spec: memory-system-v2.1.0-spec.md
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

__version__ = "2.1.0"


class SpineProvider(MemoryProvider):
    """Hermes Memory v2 — "The Sleeping Brain" provider."""

    # ------------------------------------------------------------------
    # MemoryProvider ABC
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "spine"

    def is_available(self) -> bool:
        """Spine is local-only — always available if config is present."""
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        """Read config, open index, warm embedder if needed."""
        from .config import load_spine_config

        self._session_id = session_id
        self._config = load_spine_config(kwargs.get("hermes_home", ""))
        logger.info("Spine initialized — session=%s", session_id)

    def system_prompt_block(self) -> str:
        """Spine contributes no static system prompt text."""
        return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return the four spine tool schemas."""
        from .tools import REMEMBER_SCHEMA, RECALL_SCHEMA, REFLECT_SCHEMA, FORGET_SCHEMA

        return [REMEMBER_SCHEMA, RECALL_SCHEMA, REFLECT_SCHEMA, FORGET_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch tool call to the appropriate handler."""
        from .tools import handle_remember, handle_recall, handle_reflect, handle_forget

        handlers = {
            "remember": handle_remember,
            "recall": handle_recall,
            "reflect": handle_reflect,
            "forget": handle_forget,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return '{"error": "Unknown tool: ' + tool_name + '"}'
        return handler(args, config=self._config)

    def shutdown(self) -> None:
        """Unload embedder, close index."""
        from .embedder import unload_embedder

        unload_embedder()
        logger.info("Spine shutdown complete.")

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    # TODO Phase 2.3: on_session_start is a general Hermes plugin hook,
    # NOT a MemoryProvider ABC hook. Spine must also register as a general
    # plugin to receive it. Until then, activation manifest is deferred.
    # def on_session_start(self, session_id: str, **kwargs) -> None:
    #     from .loops import build_activation_manifest
    #     self._last_manifest = build_activation_manifest(self._config)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Observer pass — extract durable observations (spec §5.1)."""
        from .loops import run_observer

        run_observer(messages, config=self._config)

    def on_session_reset(self, **kwargs) -> None:
        """Clear tracking state on /reset."""
        self._last_manifest = None
