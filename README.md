# Spine: Local-First Memory System (v2.2.0)

**"The Sleeping Brain", govern writes, derive reads, batch intelligence.**

Spine is a portable, self-owned memory module for AI agents. It captures observations from conversations, consolidates them nightly, and surfaces relevant context at session start, all locally, with zero per-turn API cost.

## Architecture

```
canonical log (JSONL, append-only)  ← write gate (remember)
         ↓
derived index (SQLite FTS5 + vectors)  ← rebuild-index.py
         ↓
recall (hybrid keyword + brute-force cosine)
         ↓
synthesis (reflect, LLM answers from observations)
```

Recall is hybrid FTS5 keyword search plus a brute-force cosine similarity scan over
packed float32 vector blobs stored directly in SQLite, a plain dot product against
every pre-normalized stored vector, combined with FTS5 via Reciprocal Rank Fusion and
a recency weight. There is no vector-DB extension (no sqlite-vec, no FAISS, no ANN
index): `index.py`'s `_vector_search` / `_cosine_similarity` walk the stored rows
directly. That's a deliberate simplicity tradeoff, see `index.py` for the scaling
notes, not an oversight.

## Four Tools

| Tool | What it does |
|---|---|
| `remember(content, type)` | Write gate, two-signal secrets detector + dedupe |
| `recall(query, k=6)` | Hybrid FTS5 + brute-force cosine search, auto-degradation |
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
    report_target: "telegram:<chat_id>:<topic_id>"   # or set SPINE_REPORT_TARGET
    promote_auto:
      min_confidence: 0.9
    archive_threshold: 0.3
```

`report_target` has no real default in this repo, it ships as the literal
placeholder `telegram:<chat_id>:<topic_id>`. Set it in `config.yaml` or via the
`SPINE_REPORT_TARGET` env var before running consolidation or the heartbeat.

### 4. Rebuild index from canonical logs
```bash
python3 spine/rebuild-index.py
```

### 5. Run benchmark
```bash
python3 spine/benchmark.py --set-baseline
python3 spine/eval_run.py            # retrieval gate against spine/eval_set.json
```

## Design Principles

1. **Canonical store is plain text**, JSONL, git-tracked, human-readable.
2. **Index is derived and disposable**, rebuild from logs at any time.
3. **Intelligence in batch**, observe at session end, consolidate nightly.
4. **Epistemic honesty**, every observation carries provenance and confidence.
5. **Forgetting is a feature**, decay, archival, hard delete.
6. **Secrets never stored**, two-signal detector at write time.

## Cost

~$0.20-0.50/month on deepseek-v4-pro at ~1 session/day. No per-turn API calls. Zero third-party dependencies for core operation in Phase 1-2.

## Files

Core:

| File | Purpose |
|---|---|
| `spine/__init__.py` | MemoryProvider ABC implementation |
| `spine/config.py` | Config reader with defaults |
| `spine/jsonl_writer.py` | Append-only JSONL with file lock |
| `spine/secrets_detector.py` | Two-signal secrets gate |
| `spine/embedder.py` | MiniLM on-demand load/unload; tolerates both the old and new sentence-transformers dimension-lookup method name |
| `spine/index.py` | SQLite FTS5 + brute-force cosine vector index |
| `spine/tools.py` | remember/recall/reflect/forget handlers; commits the DB write before closing the connection so a mid-close failure can't silently drop an entry |
| `spine/loops.py` | Observer + consolidation + manifest |
| `spine/llm_client.py` | DeepSeek API client |
| `spine/rebuild-index.py` | Idempotent JSONL→DB rebuild; same commit-before-close ordering as `tools.py` |
| `spine/consolidate.py` | Nightly cron runner (spec §5.2) |

Operational / hardening (added since v2.1.0):

| File | Purpose |
|---|---|
| `spine/heartbeat.py` | Daily monitor for failure modes that otherwise degrade silently (embedder rename, stale wiki index, cron drift, etc.), see its module docstring for the incident each check maps to |
| `spine/hotcore_consolidate.py` | LLM-assisted compression pass for MEMORY.md (the hot core), preserving hard tokens (paths, identifiers, numbers) exactly |
| `spine/sync_hotcore.py` | Imports MEMORY.md blocks written by other tools directly into spine, so they're not invisible to recall |
| `spine/sync_claude_memories.py` | Mirrors Claude Code's curated memory markdown files into spine as a derived, searchable `agent:claude-code` profile. **Depends on Claude Code's on-disk memory layout**, its per-project directory name (`SPINE_CLAUDE_PROJECT_KEY` env var; Claude Code derives this from the project's absolute path) is Claude-Code-specific and not part of spine itself |
| `spine/propose_memories.py` | The return leg of `sync_claude_memories.py`, proposes spine observations for promotion into Claude Code's curated memory set |
| `spine/rule_scope.py` | Tags hot-core blocks with `@when:` scope so only relevant rules load per turn, instead of the whole file on every call |
| `spine/temporal.py` | Point-in-time reconstruction, replays canonical-log patches up to a cutoff to answer "what did the system believe as of date X" |
| `spine/repair_overdecay.py` | One-off repair script for a quadratic-decay bug (documented in its own docstring); kept as a worked example of a targeted data repair, not something you run blind |
| `spine/coverage.py` | Checks whether a given piece of text is actually retrievable from spine, token-overlap scoring alone under-reports orphaned content |
| `spine/eval_run.py` + `spine/eval_set.json` | Phase-0 retrieval gate: runs a fixed query set against spine and reports pass/fail per case (single-hop, conjunctive, multi-hop). `eval_set.json`'s cases are illustrative/synthetic, not this installation's real query log |
| `spine/bench_goldrank.py` | Per-channel rank attribution (wiki-FTS / obs-FTS / wiki-vector / obs-vector) for benchmark queries |
| `spine/bench_provenance.py` | Read-only audit of which corpus tier(s) contain each benchmark query's expected-hit terms |
| `spine/locomo_bench.py` | LoCoMo/LongMemEval-style benchmark: single/multi-hop/temporal/open-domain/adversarial categories, scored by F1 and LLM-judge, not just fixed-hit precision/recall |
| `spine/benchmark.py` | Fixed expected-hits precision/recall harness for regression tracking |

Tests, spec, MCP:

| File | Purpose |
|---|---|
| `tests/test_spine_regressions.py` | Regression suite covering dedupe, hot-core promotion/rewrite, decay, and more |
| `tests/test_rule_scope.py` | Tests for `rule_scope.py`'s `@when:` tagging |
| `spec/memory-system-v2.1.0-spec.md` | Full architecture spec |
| `bench/queries.json` | Example benchmark queries |
| `mcp/server.py` | MCP server exposing spine's remember/recall/reflect/forget over MCP so other agents (Claude Code, etc.) can share the same memory store. **Imports the spine plugin from a Hermes Agent checkout** (`~/.hermes/hermes-agent/plugins/memory`) and runs inside that checkout's Python environment, it is not a standalone server outside a Hermes Agent install |
| `README.md` | This file |

### Dependency note: code that imports Hermes internals

`sync_claude_memories.py`, `propose_memories.py`, `heartbeat.py`, and `mcp/server.py`
either import from a Hermes Agent checkout, read Claude Code's on-disk memory layout,
or both. They're included because they're a real, working part of this system and
useful as reference, but they are not portable as-is to a non-Hermes install without
adapting those integration points. Everything else under `spine/` is
Hermes-independent.

Run tests with `python3 -m pytest tests -q` from the repo root. 4 of 67 tests in
`tests/test_rule_scope.py` require `tools/memory_tool.py` and `agent/turn_context.py`
from a full Hermes Agent checkout, neither exists in this repo, so those 4 fail here
by design (2 also fail in the source checkout itself, from an unrelated signature
drift between `rule_scope.py`'s tests and the current `MemoryStore.__init__`). The
other 63 are self-contained and pass standalone.

## CHANGELOG

### v2.2.0 (2026-09-28)
- **Fixed:** data-loss bug where `tools.py` and `rebuild-index.py` could `close()`
  the DB connection before `commit()`, silently dropping the last write.
  Commit now always precedes close.
- **Fixed:** `embedder.py` now tries both `get_embedding_dimension` and
  `get_sentence_embedding_dimension`, tolerating the sentence-transformers rename
  that previously broke semantic recall silently (fell back to keyword-only for
  8 days before detection, the origin story for `heartbeat.py`, below).
- **Added:** `heartbeat.py`, daily monitor for silent-failure modes.
- **Added:** hot-core sync/consolidate pair (`sync_hotcore.py`,
  `hotcore_consolidate.py`) and `sync_claude_memories.py` /
  `propose_memories.py` for two-way sync with Claude Code's memory files.
- **Added:** `rule_scope.py` (scoped hot-core loading), `temporal.py`
  (point-in-time reconstruction), `repair_overdecay.py` (worked repair example),
  `propose_memories.py`, `coverage.py`.
- **Added:** `eval_run.py` + `eval_set.json` (retrieval gate), `bench_goldrank.py`,
  `bench_provenance.py`, `locomo_bench.py`.
- **Added:** `tests/` (regression suite + rule-scope tests).
- **Added:** `mcp/server.py`, MCP server exposing spine to other agents.
- **Docs:** README recall description corrected to brute-force cosine over stored
  vectors (there was never a vector-DB extension in the retrieval path); full
  module table; CHANGELOG added.
- **Privacy:** removed a real Telegram chat/topic id from `config.py` and the spec
  (now a placeholder + `SPINE_REPORT_TARGET` env var), and genericized absolute
  paths, the owner's name, and other installation-specific identifiers throughout.
  The real id remains visible in this repo's git history prior to this commit;
  history was not rewritten.

### v2.1.0 (2026-07-17)
- Initial public release.
