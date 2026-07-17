"""Lightweight LLM client for spine loops — spec §5.1, §5.2.

Calls DeepSeek API (OpenAI-compatible) for observer extraction and
consolidation passes. Reads DEEPSEEK_API_KEY from environment.
Uses the configured loop_model (default: deepseek-v4-pro).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-pro"


def _get_api_key() -> Optional[str]:
    """Get DeepSeek API key from environment."""
    return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_KEY")


def call_llm(
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.3,
) -> Optional[str]:
    """Call DeepSeek chat completions API. Returns response text or None on failure."""
    api_key = _get_api_key()
    if not api_key:
        logger.error("No DEEPSEEK_API_KEY in environment — cannot call LLM")
        return None

    try:
        import urllib.request

        payload = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return None


# ── Observer extraction prompt (spec §5.1) ──────────────────────────

OBSERVER_SYSTEM_PROMPT = """You are a memory observer. Extract durable facts about the user from this conversation transcript.

Return a JSON array. Each object:
{
  "type": "fact" | "preference" | "pattern" | "correction" | "identity",
  "content": "≤500 chars, present-tense, atomic claim about the user",
  "epistemic": "extracted" (user stated it) | "inferred" (you deduced it),
  "confidence": 0.0-1.0,
  "quote": "verbatim snippet supporting this observation"
}

Record ALL durable observations — 5–15 per session is normal. An empty array is valid for pure-chore sessions.
EXCLUDE: one-time events, task statuses, build logs, secrets, temporal facts.
PRIORITIZE: corrections (user pushed back on agent behavior) — highest priority.
"""


def extract_observations(transcript: str, model: str = DEFAULT_MODEL) -> List[Dict[str, Any]]:
    """Run observer LLM pass over a transcript. Returns parsed observations list."""
    messages = [
        {"role": "system", "content": OBSERVER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract durable observations from this conversation:\n\n{transcript}"},
    ]

    response = call_llm(messages, model=model, max_tokens=2000, temperature=0.3)
    if not response:
        return []

    # Parse JSON from response (may have markdown ```json fences)
    response = response.strip()
    if response.startswith("```"):
        # Strip markdown fences
        lines = response.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        response = "\n".join(lines)

    try:
        observations = json.loads(response)
        if isinstance(observations, list):
            return observations
        return []
    except json.JSONDecodeError:
        logger.warning("Observer LLM returned unparseable JSON: %s...", response[:200])
        return []
