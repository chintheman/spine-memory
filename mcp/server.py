#!/usr/bin/env python3
"""Spine Memory MCP server — exposes Hermes's spine memory to other agents.

Spine (plugins/memory/spine in the hermes-agent checkout) is Hermes's memory
system: hybrid FTS5 + vector recall over ~/.hermes/memory.db, with local MiniLM
embeddings. Its six operations were previously reachable only as in-process
Python calls, so Claude Code and Zo could not read or write the same memory
Hermes uses. This server closes that gap.

Design note — why everything is lazy:
`hermes mcp serve` is unusable because it boots Hermes's entire agent runtime
(connecting every downstream MCP server) BEFORE answering the client's
`initialize` request, blowing past the 30s connect timeout. This server must
not repeat that. Nothing heavy is imported at module load: spine, torch and
sentence-transformers are pulled in on the first tool call, not at startup, so
`initialize` returns immediately. Do not move these imports to the top.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

# mcp 2.0 (pinned by hermes-agent's pyproject for CVE-2026-48710) removed
# `mcp.server.fastmcp`. `mcp.server.MCPServer` is the same surface: the
# `@server.tool()` decorator and `run()` are unchanged. This server borrows
# hermes-agent's venv -- it has to, it imports the spine plugin from that
# checkout -- so a dependency bump there lands here silently. It did on
# 2026-08-28 and went unnoticed until 09-09.
from mcp.server import MCPServer

HERMES_AGENT = os.path.expanduser("~/.hermes/hermes-agent")
PLUGINS_MEMORY = os.path.join(HERMES_AGENT, "plugins", "memory")

mcp = MCPServer("spine-memory")

_spine: Optional[Dict[str, Any]] = None


def _load():
    """Import spine and resolve config. Cached; first call pays the model load."""
    global _spine
    if _spine is not None:
        return _spine

    for p in (HERMES_AGENT, PLUGINS_MEMORY):
        if p not in sys.path:
            sys.path.insert(0, p)

    from spine.config import load_spine_config
    from spine.tools import (
        handle_explain,
        handle_forget,
        handle_recall,
        handle_recall_at,
        handle_reflect,
        handle_remember,
    )

    _spine = {
        "config": load_spine_config(),
        "recall": handle_recall,
        "recall_at": handle_recall_at,
        "remember": handle_remember,
        "reflect": handle_reflect,
        "forget": handle_forget,
        "explain": handle_explain,
    }
    return _spine


def _call(op: str, args: Dict[str, Any]) -> str:
    """Dispatch to a spine handler, returning its JSON string verbatim.

    Spine handlers return JSON strings already; errors are surfaced as JSON so
    the caller gets a structured failure rather than an MCP transport error.
    """
    try:
        s = _load()
        return s[op](args, s["config"])
    except Exception as exc:  # noqa: BLE001 - surface everything to the caller
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "op": op})


@mcp.tool()
def memory_recall(query: str, k: int = 6, profile: str = "agent:main") -> str:
    """Search Hermes's long-term memory by meaning, not just keywords.

    Hybrid FTS5 keyword + vector similarity search fused with Reciprocal Rank
    Fusion, then weighted for recency. This is the main read path — use it
    before asserting anything about the user's history, decisions or projects.

    Args:
        query: What you want to know, in natural language.
        k: Number of results (max 25).
        profile: Memory profile. "agent:main" is the real one; wiki content is
            stored under "shared" and is returned by the same search.
    """
    return _call("recall", {"query": query, "k": k, "profile": profile})


@mcp.tool()
def memory_remember(
    content: str,
    type: str = "fact",
    confidence: float = 0.5,
    epistemic: str = "extracted",
    topics: Optional[List[str]] = None,
    profile: str = "agent:main",
) -> str:
    """Write a durable observation into Hermes's memory.

    Passes through spine's write gate (secrets detection, dedupe) and is
    appended to the canonical JSONL log as well as the index, so it survives a
    rebuild. Prefer one clear fact per call.

    Args:
        content: The fact, in a self-contained sentence.
        type: "fact", "preference", "event", or "decision".
        confidence: 0.0-1.0. Use >=0.9 only for directly stated, verified facts.
        epistemic: "stated" (user said it) or "extracted" (inferred).
        topics: Optional tags for retrieval.
        profile: Memory profile to write to.
    """
    return _call(
        "remember",
        {
            "content": content,
            "type": type,
            "confidence": confidence,
            "epistemic": epistemic,
            "topics": topics or [],
            "profile": profile,
        },
    )


@mcp.tool()
def memory_recall_at(query: str, as_of: str, k: int = 6, profile: str = "agent:main") -> str:
    """Search memory as it stood at a past point in time.

    Answers "what did I believe/know then", excluding anything learned after
    the cutoff. Useful for auditing why a past decision was made.

    Args:
        query: What you want to know.
        as_of: ISO-8601 timestamp or date, e.g. "2026-07-15".
        k: Number of results (max 25).
        profile: Memory profile.
    """
    return _call("recall_at", {"query": query, "as_of": as_of, "k": k, "profile": profile})


@mcp.tool()
def memory_reflect(question: str, profile: str = "agent:main") -> str:
    """Synthesise an answer across many memories using an LLM.

    Slower and costlier than memory_recall — it makes an external LLM call
    (DeepSeek) to reason over retrieved observations. Use memory_recall for
    lookups; use this only when the answer requires combining several memories.

    Args:
        question: The question to reason about.
        profile: Memory profile.
    """
    return _call("reflect", {"question": question, "profile": profile})


@mcp.tool()
def memory_forget(
    term: str = "",
    action: str = "discover",
    obs_id: str = "",
    profile: str = "agent:main",
) -> str:
    """Find and optionally archive memories matching a term.

    Defaults to "discover", which only reports what matches and changes
    nothing. Always run discover first and confirm with the user before any
    destructive action — this edits their durable memory.

    Args:
        term: Text to search for.
        action: "discover" (safe, default) or an archive action.
        obs_id: Specific observation id, when acting on one record.
        profile: Memory profile.
    """
    return _call("forget", {"term": term, "action": action, "obs_id": obs_id, "profile": profile})


@mcp.tool()
def memory_explain(obs_id: str, profile: str = "agent:main") -> str:
    """Show the provenance of one observation — when and why it was recorded.

    Args:
        obs_id: The observation id, as returned by memory_recall.
        profile: Memory profile.
    """
    return _call("explain", {"obs_id": obs_id, "profile": profile})


if __name__ == "__main__":
    mcp.run()
