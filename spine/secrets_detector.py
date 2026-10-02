"""Two-signal secrets detector — spec §4.1.

Signal 1: Pattern hard-reject — known key prefixes, private key blocks,
seed phrases, Ethereum keys, bearer tokens. These always block.

Signal 2: Entropy flag-for-review — any token >20 chars with >4.5 bits/char
Shannon entropy → held, not written, Telegram notice with show/discard.
Allowlist exempts known ID shapes (ULIDs, git SHAs, session ids).
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple

# ── Signal 1: Pattern hard-reject ──────────────────────────────────────

HARD_BLOCK_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # API key prefixes
    (re.compile(r'sk-[a-zA-Z0-9]{20,}'), "OpenAI API key"),
    # Anthropic keys (sk-ant-api03-...) never matched the OpenAI pattern above —
    # the hyphens 3 chars into "ant-api03-" break its 20+ consecutive-alnum
    # requirement, so a real Anthropic key only ever hit the weaker entropy
    # hold instead of a hard block.
    (re.compile(r'sk-ant-[a-zA-Z0-9_-]{20,}'), "Anthropic API key"),
    (re.compile(r'r8_[a-zA-Z0-9]{20,}'), "Replicate API key"),
    (re.compile(r'rf_[a-zA-Z0-9]{20,}'), "Together AI API key"),
    (re.compile(r'ghp_[a-zA-Z0-9]{20,}'), "GitHub personal access token"),
    (re.compile(r'xox[bprs]-[a-zA-Z0-9-]{20,}'), "Slack bot token"),
    (re.compile(r'AKIA[A-Z0-9]{16,}'), "AWS access key"),
    # Private key blocks
    (re.compile(r'-----BEGIN\s+.*PRIVATE KEY-----'), "Private key block"),
    # Seed phrases (12 or 24 words, with optional leading/trailing context).
    # Case-insensitive — a capitalized paste ("Abandon Ability Able...") is a
    # real seed phrase too and must not silently fall through the hard block.
    # Known accepted tradeoff (confirmed 2026-07-30, user decision: keep as
    # is): this widens the false-positive surface to ordinary capitalized
    # 12+-word prose (e.g. "The Quick Brown Fox Jumps Over The Lazy Dog
    # While Running Fast Today" now hard-blocks) — no BIP-39 wordlist check
    # here to filter that out. Deliberate: a wrongly-blocked normal write is
    # recoverable/visible; a missed real seed phrase is not.
    (re.compile(r'\b([a-z]{3,12}\s){11}[a-z]{3,12}\b', re.IGNORECASE), "12-word seed phrase"),
    (re.compile(r'\b([a-z]{3,12}\s){23}[a-z]{3,12}\b', re.IGNORECASE), "24-word seed phrase"),
    # Ethereum private key (0x + 64 hex)
    (re.compile(r'\b0x[a-fA-F0-9]{64}\b'), "Ethereum private key"),
    # Bearer tokens
    (re.compile(r'bearer\s+[a-zA-Z0-9._~+/-]{20,}', re.IGNORECASE), "Bearer token"),
]

# ── Signal 2: Entropy detection ────────────────────────────────────────

# Allowlisted patterns (exempt from entropy check)
ALLOWLIST_PATTERNS: List[re.Pattern] = [
    re.compile(r'^[0-9A-HJKMNP-TV-Z]{26}$'),            # ULID
    re.compile(r'^[a-fA-F0-9]{7,40}$'),                  # git SHA (7-40 hex)
    re.compile(r'^s_[a-z0-9_]+$'),                       # session id
    re.compile(r'^0x[a-fA-F0-9]{8,16}$'),                # short Ethereum address (not a key)
    re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),  # UUID
]


def shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy in bits per character.

    Returns 0.0 for strings shorter than 2 characters.
    """
    if len(text) < 2:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _is_allowlisted(token: str) -> bool:
    """Check if token matches any allowlist pattern (exempt from entropy check)."""
    for pattern in ALLOWLIST_PATTERNS:
        if pattern.fullmatch(token):
            return True
    return False


def detect_secrets(content: str) -> dict:
    """Scan content for secrets. Returns result dict.

    Returns:
      {"blocked": False, "reason": ""}  — clean
      {"blocked": True, "reason": "hard", "detail": "..."}  — hard reject
      {"blocked": False, "held_for_review": True, "tokens": ["..."]}  — entropy hold
    """
    # Signal 1: Hard pattern rejection
    for pattern, label in HARD_BLOCK_PATTERNS:
        if pattern.search(content):
            return {"blocked": True, "reason": "hard", "detail": f"Matched: {label}"}

    # Signal 2: Entropy scan
    # Split into word-like tokens (alphanumeric + common symbol chars, >20 chars)
    tokens = re.findall(r'[A-Za-z0-9_/+=.-]{21,}', content)
    high_entropy_tokens = []
    for token in tokens:
        if _is_allowlisted(token):
            continue
        if shannon_entropy(token) > 4.5:
            high_entropy_tokens.append(token)

    if high_entropy_tokens:
        return {
            "blocked": False,
            "held_for_review": True,
            "tokens": high_entropy_tokens,
        }

    return {"blocked": False, "reason": ""}
