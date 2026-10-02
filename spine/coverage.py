"""Is a piece of text actually retrievable from spine?

Exists because the obvious way to answer that question is wrong. A token-overlap
score between the text and the corpus reported "0 orphans" for a MEMORY.md block
whose distinctive term (`OnVUE`) appeared in exactly zero observations and zero
wiki chunks. Common words -- "the", "before", "system", "test" -- pushed the
overlap above threshold on their own.

The fix is to ignore common words entirely and probe only the rarest ones. If a
text's three least-common terms are all absent from the corpus, the text is not
in there, whatever a similarity score says.

Use `is_retrievable()` before deleting anything from the MEMORY.md hot core.
Never hand-roll a fuzzy version.
"""
from __future__ import annotations

import re
import sqlite3
from typing import List, Optional, Tuple

# Candidate distinctive terms. Trailing punctuation is stripped afterwards:
# without that, "manifest." and "inversion:" score as ultra-rare purely because
# of the period, and every block containing one gets flagged as unretrievable.
_TOKEN = re.compile(r"[A-Za-z0-9_./:\-]{6,}")
_TRIM = ".,;:!?)('\"`/-"

# Tag prefixes used by MEMORY.md blocks; not part of the content.
_TAG = re.compile(r"^\s*\[(C|R|W|F|ID)\]\s*")


def _corpus_count(conn: sqlite3.Connection, token: str,
                  profile: Optional[str] = None) -> int:
    """How many corpus rows contain `token`.

    `profile` scopes the observation half to one profile and drops the wiki
    entirely — that is the form needed to ask "is this already covered by
    agent:claude-code specifically", which the unscoped version cannot answer
    because it counts every profile plus 2,643 wiki chunks.
    """
    if profile:
        return conn.execute(
            "SELECT COUNT(*) FROM observations WHERE profile = ? AND content LIKE ?",
            (profile, "%" + token + "%")).fetchone()[0]
    n = conn.execute("SELECT COUNT(*) FROM observations WHERE content LIKE ?",
                     ("%" + token + "%",)).fetchone()[0]
    n += conn.execute("SELECT COUNT(*) FROM wiki_chunks WHERE content LIKE ?",
                      ("%" + token + "%",)).fetchone()[0]
    return n


def covered_by_profile(text: str, conn: sqlite3.Connection, profile: str,
                       n: int = 4) -> Tuple[bool, List[str]]:
    """Is `text` already represented in `profile`?

    Ranks tokens by rarity across the WHOLE corpus (so "the rarest terms in this
    text" stays meaningful), then asks whether those terms appear in the target
    profile. Returns (covered, tokens_found_there).

    Deliberately strict: all n rarest tokens must be present. Measured on the
    live store, that bar puts overlap between agent:main and agent:claude-code
    at 11%, with corrections 88% novel.
    """
    probes = distinctive_tokens(text, conn, n)
    if not probes:
        return False, []
    found = [t for t in probes if _corpus_count(conn, t, profile=profile) > 0]
    return len(found) == len(probes), found


def distinctive_tokens(text: str, conn: sqlite3.Connection, n: int = 3) -> List[str]:
    """The n rarest tokens in `text`, by how often they occur in the corpus."""
    tokens = {t.strip(_TRIM) for t in _TOKEN.findall(text)}
    tokens = {t for t in tokens if len(t) >= 6}
    if not tokens:
        return []
    ranked = sorted(((_corpus_count(conn, t), t) for t in tokens), key=lambda x: (x[0], x[1]))
    return [t for _, t in ranked[:n]]


def is_retrievable(text: str, conn: sqlite3.Connection,
                   exclude_self: Optional[str] = None) -> Tuple[bool, List[str]]:
    """True if every one of the text's rarest tokens exists somewhere in the corpus.

    Returns (retrievable, missing_tokens). A text with no usable tokens is
    treated as retrievable: there is nothing distinctive to lose, and failing
    open here beats blocking on trivia.

    `exclude_self` is an observation id to ignore, for checking whether a text
    would still be findable if that row were removed.
    """
    probes = distinctive_tokens(text, conn)
    if not probes:
        return True, []
    missing = []
    for t in probes:
        if exclude_self:
            n = conn.execute(
                "SELECT COUNT(*) FROM observations WHERE content LIKE ? AND id != ?",
                ("%" + t + "%", exclude_self)).fetchone()[0]
            n += conn.execute("SELECT COUNT(*) FROM wiki_chunks WHERE content LIKE ?",
                              ("%" + t + "%",)).fetchone()[0]
        else:
            n = _corpus_count(conn, t)
        if n == 0:
            missing.append(t)
    return (not missing), missing


def strip_tag(block: str) -> str:
    return _TAG.sub("", block).strip()


def hotcore_blocks(path: str) -> List[str]:
    import os
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [b.strip() for b in f.read().split("\n§\n") if b.strip()]


def uncovered_hotcore(path: str, conn: sqlite3.Connection) -> List[Tuple[str, List[str]]]:
    """MEMORY.md blocks that spine has never heard of.

    These arise because MEMORY.md has a second writer: Hermes's memory() tool
    writes straight into the hot core and never passes through spine. 15 blocks
    were in this state on 2026-08-20, and deleting them would have destroyed
    them outright.
    """
    out = []
    for block in hotcore_blocks(path):
        ok, missing = is_retrievable(strip_tag(block), conn)
        if not ok:
            out.append((block, missing))
    return out
