# Spine — Local-First Memory System

**"The Sleeping Brain" — govern writes, derive reads, batch intelligence.**

Spine is a portable, self-owned memory module for AI agents. It captures observations from conversations, consolidates them nightly, and surfaces relevant context at session start — all locally, with zero per-turn API cost.

## Architecture

```
canonical log (JSONL, append-only)  ← write gate (remember)
         ↓
derived index (SQLite FTS5 + vectors)  ← rebuild-index.py
         ↓
recall (hybrid keyword + semantic)
         ↓
synthesis (reflect — LLM answers from observations)
```

## Four Tools

| Tool | What it does |
|---|---|
| `remember(content, type)` | Write gate — two-signal secrets detector + dedupe |
| `recall(query, k=6)` | Hybrid FTS5+vector search, auto-degradation |
| `reflect(question)` | LLM synthesis over recalled observations |
| `forget(term)` | Discovery → per-entry confirmation → ID deletion |

## Three Loops

| Loop | Trigger | What |
|---|---|---|
| **Observer** | `on_session_end` | LLM extracts durable observations → remember() gate |
| **Consolidation** | Nightly cron | Merge → Promote → Demote → Decay → Compact |
| **Activation** | `on_session_start` | Manifest + recall top-6 → system prompt |

## Getting Started

### 1. Copy the spine module
```bash
cp -r spine/ your-agent/plugins/memory/spine/
```

### 2. Install dependencies
```bash
pip install sentence-transformers  # MiniLM embedder (384-dim)
```

### 3. Configure
```yaml
memory:
  provider: spine
  spine:
    canonical_root: ~/data/memory
    db: ~/data/memory.db
    embedder: all-MiniLM-L6-v2
    loop_model: deepseek-v4-pro
    consolidation_cron: "0 4 * * *"
    promote_auto:
      min_confidence: 0.9
    archive_threshold: 0.3
```

### 4. Rebuild index from canonical logs
```bash
python3 spine/rebuild-index.py
```

### 5. Run benchmark
```bash
python3 spine/benchmark.py --set-baseline
```

## Design Principles

1. **Canonical store is plain text** — JSONL, git-tracked, human-readable.
2. **Index is derived and disposable** — rebuild from logs at any time.
3. **Intelligence in batch** — observe at session end, consolidate nightly.
4. **Epistemic honesty** — every observation carries provenance and confidence.
5. **Forgetting is a feature** — decay, archival, hard delete.
6. **Secrets never stored** — two-signal detector at write time.

## Cost

~$0.20–0.50/month on deepseek-v4-pro at ~1 session/day. No per-turn API calls. Zero third-party dependencies for core operation in Phase 1–2.

## Files

| File | Purpose |
|---|---|
| `spine/__init__.py` | MemoryProvider ABC implementation |
| `spine/config.py` | Config reader with defaults |
| `spine/jsonl_writer.py` | Append-only JSONL with file lock |
| `spine/secrets_detector.py` | Two-signal secrets gate |
| `spine/embedder.py` | MiniLM on-demand load/unload |
| `spine/index.py` | SQLite FTS5 + vector index |
| `spine/tools.py` | remember/recall/reflect/forget handlers |
| `spine/loops.py` | Observer + consolidation + manifest |
| `spine/llm_client.py` | DeepSeek API client |
| `spine/rebuild-index.py` | Idempotent JSONL→DB rebuild |
| `spine/benchmark.py` | Retrieval benchmark harness |
| `spine/consolidate.py` | Nightly cron runner |
| `spec/memory-system-v2.1.0-spec.md` | Full architecture spec |
| `bench/queries.json` | Example benchmark queries |
| `README.md` | This file |
