#!/usr/bin/env python3
"""rebuild-index.py — Rebuild the derived SQLite index from canonical JSONL logs.

Idempotent: running twice produces an identical database (same row counts,
same content checksums). Reads all `<profile>.jsonl` files from
~/wiki/_memory/observations/, applies patches in order, embeds content
with MiniLM, and populates the memory.db derived index.

Usage:
    python3 rebuild-index.py                    # full rebuild
    python3 rebuild-index.py --profile agent:main  # single profile only
    python3 rebuild-index.py --dry-run              # validate, don't write
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the spine package to path
SPINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SPINE_DIR.parent))  # plugins/memory/ — makes 'spine' importable as package
sys.path.insert(0, str(SPINE_DIR.parent.parent.parent))  # hermes-agent root


def load_observations(obs_dir: str, profile_filter: str = "") -> List[Dict[str, Any]]:
    """Load all observations from JSONL files, applying patches in order.

    Returns a list of fully-patched observation dicts, with patches merged.
    """
    obs_path = Path(obs_dir)
    records: List[Dict[str, Any]] = []
    patches: Dict[str, List[Dict[str, Any]]] = {}

    for jsonl_file in sorted(obs_path.glob("*.jsonl")):
        profile = jsonl_file.stem
        if profile_filter and profile != profile_filter:
            continue

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping malformed line in {jsonl_file}: {e}", file=sys.stderr)
                    continue

                if "patch" in rec:
                    pid = rec["id"]
                    if pid not in patches:
                        patches[pid] = []
                    patches[pid].append(rec)
                else:
                    records.append(rec)

    # Apply patches
    patched: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        oid = rec["id"]
        merged = dict(rec)
        if oid in patches:
            for p in patches[oid]:
                for key, val in p["patch"].items():
                    merged[key] = val
            merged["patch_count"] = len(patches[oid])
        # Skip deleted/tombstoned
        if merged.get("status") == "deleted":
            continue
        patched[oid] = merged

    return list(patched.values())


def load_episodes(episodes_dir: str, profile_filter: str = "") -> List[Dict[str, Any]]:
    """Load all episode records from JSONL files. Episodes are append-only, no patches."""
    ep_path = Path(episodes_dir)
    records: List[Dict[str, Any]] = []
    if not ep_path.exists():
        return records

    for jsonl_file in sorted(ep_path.glob("*.jsonl")):
        profile = jsonl_file.stem
        if profile_filter and profile != profile_filter:
            continue
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping malformed line in {jsonl_file}: {e}", file=sys.stderr)
    return records


def embed_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Embed a list of texts in batches. Returns empty lists on failure.

    Reuses spine.embedder's cached model loader instead of an independent
    second SentenceTransformer load — avoids duplicating model-loading logic
    (and a second copy of the numpy-import-safety issue embedder.py itself
    had to fix).
    """
    try:
        from spine.embedder import _load_model
        import spine.embedder as _embedder_mod
        model = _load_model()
        if model is None:
            print("⚠️  Embedder unavailable — index built without vectors", file=sys.stderr)
            return []
        all_vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        np = _embedder_mod.np
        if np is not None and isinstance(all_vectors, np.ndarray):
            return all_vectors.tolist()
        return []
    except Exception as e:
        print(f"⚠️  Embedder unavailable — index built without vectors: {e}", file=sys.stderr)
        return []


def build_index(observations: List[Dict[str, Any]], db_path: str, dry_run: bool = False,
                 episodes: Optional[List[Dict[str, Any]]] = None) -> Dict[str, int]:
    """Rebuild the derived index from observations and episodes.

    Returns stats: {observations: N, episodes: N, wiki_chunks: N}
    """
    from spine.index import MemoryIndex

    episodes = episodes or []
    stats: Dict[str, int] = {"observations": 0, "episodes": 0, "wiki_chunks": 0}

    if dry_run:
        print(f"Dry run: {len(observations)} observations, {len(episodes)} episodes would be indexed to {db_path}")
        stats["observations"] = len(observations)
        stats["episodes"] = len(episodes)
        return stats

    # Embed all content first
    texts = [obs["content"] for obs in observations if obs.get("content")]
    vectors = embed_batch(texts) if texts else []

    idx = MemoryIndex(db_path)
    idx.open()

    # Check dimension consistency
    # A full rebuild re-embeds everything, so it is the ONE place allowed to
    # restamp the store width. dim_meta is INSERT OR IGNORE elsewhere, so
    # without this a legitimate model swap leaves the store permanently
    # mislabelled and check_embedder FAILs every night with no way back.
    stored_dim = idx.get_embedding_dim()
    # From the EMBEDDER, not index.embedding_dim(): that resolves store-first, so
    # comparing it against the store compared the store to itself and the branch
    # below was unreachable -- set_embedding_dim() had no caller at all.
    from spine.embedder import get_embedding_dim as _live_dim
    try:
        live_dim = _live_dim()
    except Exception:  # noqa: BLE001 — embedder down; leave the stamp alone
        live_dim = stored_dim
    if live_dim and live_dim != stored_dim:
        print(f"  embedding width {stored_dim} -> {live_dim}; restamping dim_meta")
        idx.set_embedding_dim(live_dim)

    # Upsert observations
    for i, obs in enumerate(observations):
        embedding = vectors[i] if i < len(vectors) else None
        idx.upsert_observation(obs, embedding)
        stats["observations"] += 1

    # Upsert episodes
    for ep in episodes:
        idx.upsert_episode(ep)
        stats["episodes"] += 1

    idx.conn.commit()
    idx.vacuum()
    idx.close()

    return stats


def chunk_wiki(wiki_root: str, db_path: str, dry_run: bool = False) -> int:
    """Chunk wiki markdown files and index them.

    Each h2 section becomes a chunk. Skips context-digests, learnings, .git.
    Returns number of chunks indexed.
    """
    import re
    from spine.index import MemoryIndex, _serialize_vector

    wiki_path = Path(wiki_root)
    if not wiki_path.exists():
        print(f"Wiki root not found: {wiki_root}")
        return 0

    md_files = []
    for f in wiki_path.rglob("*.md"):
        rel = str(f.relative_to(wiki_path))
        # Skip noise
        if any(skip in rel for skip in ["context-digests", "learnings", ".git", "__pycache__", "node_modules", "/dist/", "/.astro/"]):
            continue
        md_files.append(f)

    print(f"Chunking {len(md_files)} wiki files from {wiki_root}...")
    chunks = []

    for fpath in md_files:
        try:
            content = fpath.read_text(encoding="utf-8")
        except Exception:
            continue
        rel = str(fpath.relative_to(wiki_path))
        # Extract title from first h1
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else fpath.stem

        # Split by h2 sections
        sections = re.split(r'(?=^##\s)', content, flags=re.MULTILINE)
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            chunk_id = f"wiki:{rel}:{i}"
            chunks.append({
                "id": chunk_id,
                "path": rel,
                "title": title,
                "content": section[:3000],  # cap chunk size
                "chunk_index": i,
            })
    print(f"  Created {len(chunks)} chunks")

    if dry_run:
        print("Dry run — would index chunks")
        return len(chunks)

    # Embed chunks
    texts = [c["content"] for c in chunks]
    vectors = embed_batch(texts, batch_size=64) if texts else []

    idx = MemoryIndex(db_path)
    idx.open()

    # Upsert each chunk
    for i, chunk in enumerate(chunks):
        embedding = vectors[i] if i < len(vectors) else None
        idx.conn.execute(
            """INSERT OR REPLACE INTO wiki_chunks (id, path, title, content, chunk_index, embedding)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (chunk["id"], chunk["path"], chunk["title"], chunk["content"],
             chunk["chunk_index"], _serialize_vector(embedding) if embedding else None),
        )
    idx.conn.commit()
    idx.close()
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild spine derived index from canonical logs")
    parser.add_argument("--canonical-root", default=os.path.expanduser("~/wiki/_memory"),
                        help="Root directory for canonical logs (default: ~/wiki/_memory)")
    parser.add_argument("--wiki-root", default=os.path.expanduser("~/wiki"),
                        help="Wiki root for chunking (default: ~/wiki)")
    parser.add_argument("--db", default=os.path.expanduser("~/.hermes/memory.db"),
                        help="SQLite database path")
    parser.add_argument("--profile", default="",
                        help="Rebuild for a specific profile only")
    parser.add_argument("--skip-wiki", action="store_true",
                        help="Skip wiki chunking")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate logs only, don't write the database")
    args = parser.parse_args()

    obs_dir = os.path.join(args.canonical_root, "observations")

    # Wiki chunking (Phase 1.5)
    if not args.skip_wiki and os.path.isdir(args.wiki_root):
        chunk_wiki(args.wiki_root, args.db, dry_run=args.dry_run)

    # Observations
    if not os.path.exists(obs_dir):
        print(f"No observations at {obs_dir} — wiki-only index rebuilt. Done.")
        sys.exit(0)

    print(f"Loading observations from {obs_dir}...")
    observations = load_observations(obs_dir, args.profile)
    print(f"  Found {len(observations)} observations")

    ep_dir = os.path.join(args.canonical_root, "episodes")
    episodes = load_episodes(ep_dir, args.profile)
    print(f"  Found {len(episodes)} episodes")

    if not observations and not episodes:
        print("Nothing to index. Done.")
        sys.exit(0)

    print(f"Building index at {args.db}...")
    stats = build_index(observations, args.db, dry_run=args.dry_run, episodes=episodes)
    print(f"  Indexed {stats['observations']} observations, {stats['episodes']} episodes")
    print("Done.")


if __name__ == "__main__":
    main()
