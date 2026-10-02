# Hermes Memory System v2.1.0, "The Sleeping Brain"

> **Status:** Build-ready. Supersedes spec v2.0.0 entirely.
> **Incorporates:** SCOPE.md amendments A1-A11 · Hermes review (REVIEW.md) S1-S7, S-A-S-E, Q1-Q7.
> **Build phases & milestones:** SCOPE.md · **Environment & findings:** CONTEXT.md · **Decisions:** PROJECT.md §4
> **Host:** M1 MacBook Air 8GB · **Loop model:** deepseek-v4-pro · **Cost ceiling:** ≤ $2/month

---

## 0. Thesis

v1 had every memory organ and none of the loops. v2 adds three cheap batch loops, observe, consolidate, recall, over local, git-owned storage. No per-turn LLM or API
cost. Doctrine (from the Gbrain evaluation): **governed writes, derived reads, epistemic
honesty.** The whole system is a portable module: log format + rebuild script + four
tools + three loops.

## 1. Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│ HOT CORE, injected every turn                                     │
│   MEMORY.md    rules/corrections/workflow · cap 20,000 chars       │
│                fed by nightly promotion · [R][C][F][W] tags kept    │
│   IDENTITY.md  slim identity · cap 5,000 chars · replaces USER.md  │
├───────────────────────────────────────────────────────────────────┤
│ CANONICAL STORE (git SSOT), ~/wiki/_memory/                       │
│   observations/<profile>.jsonl   append-only, file-locked writer   │
│   episodes/<profile>.jsonl       append-only                       │
│   bench/                          retrieval benchmark + history     │
│   archive/                        archived entries, retired configs │
│   Pre-commit secrets hook on the repo (S-E).                        │
├───────────────────────────────────────────────────────────────────┤
│ DERIVED INDEX (disposable), ~/.hermes/memory.db                   │
│   SQLite: FTS5 + sqlite-vec (384-dim MiniLM now; re-embed on swap) │
│   Rebuilt any time: rebuild-index.py (idempotent, JSONL → db)      │
│   Wiki chunks indexed here too (source='wiki'), replaces FAISS    │
├───────────────────────────────────────────────────────────────────┤
│ SESSION DB, unchanged (raw transcripts, FTS5, 30-day prune)       │
│   forget() gains scrub access to it                                 │
└───────────────────────────────────────────────────────────────────┘
Decommissioned: Honcho (plugin + 30 peers), standalone FAISS,
USER.md as every-turn injection.
```

## 2. Canonical Log Format (JSONL, one object per line)

### observations/<profile>.jsonl
```json
{"id":"01J...ULID","profile":"agent:main","topics":["#11"],
 "type":"preference","epistemic":"extracted",
 "content":"Prefers concise Telegram replies; long docs as files",
 "confidence":0.85,"confirmations":2,
 "evidence":[{"session_id":"s_abc","ts":"2026-07-14T09:12:00+08:00","quote":"..."}],
 "status":"active","supersedes":null,"contradicts":[],
 "created_at":"...","last_confirmed":"...","last_retrieved":null}
```
- `type`: fact | preference | pattern | correction | identity
- `epistemic`: **extracted** (user stated it) | **inferred** (observer deduced it)
- `status`: active | promoted | archived | superseded | **rejected** (reverted promotion, never re-promotes)
- `topics`: optional provenance; recall does **not** filter by topic (cross-topic awareness is a goal)
- Mutations (confidence, status, confirmations) are appended as patch lines
  `{"id":..., "patch":{...},"ts":...}`, the log stays append-only; the index applies
  patches in order. Compaction (§5.2 pass 5) may rewrite a log file wholesale in a
  single git commit.

### episodes/<profile>.jsonl
```json
{"id":"01J...","profile":"agent:main","session_id":"s_abc",
 "summary":"≤1500 chars","outcomes":["decided X","shipped Y"],
 "started_at":"...","ended_at":"...","compacted":false}
```

## 3. Derived Index Schema (memory.db)

Per spec v2.0.0 §2 with these deltas: add `epistemic`, `topics`, `contradicts` columns;
add `status='rejected'`; embedding dim = 384 (MiniLM) with dim recorded in a meta table
so an embedder swap forces a rebuild rather than silently mixing spaces. `media` table
remains reserved (Phase 3). FTS5 mirrors + vec0 tables for observations, episodes, wiki.

## 4. Tools

### 4.1 remember(content, type, profile?, topics?)
Write gate. Routing by declared type (corrections/rules → correction observations;
identity → identity observations; fact/preference/pattern → standard; events/logs →
**rejected** with pointer to session DB). Dedupe on write: cosine > 0.92 within
profile+type → increment confirmations instead of insert.

**Secrets detector (two signals):**
1. *Pattern hard-reject:* known key prefixes (sk-, r8_, rf_, ghp_, xox, AKIA…),
   `-----BEGIN`, bearer tokens, 12/24-word seed phrases, 0x-64-hex.
2. *Entropy flag-for-review (S1):* any token >20 chars with >4.5 bits/char Shannon
   entropy → held, not written, Telegram notice with show/discard buttons.
   Allowlist exempts known ID shapes (ULIDs, git SHAs, session ids) to cut false
   positives.

### 4.2 recall(query, profile?, k=6)
1. If `embedder_available`: embed query (MiniLM, loaded on demand, unloaded after), else **automatic** FTS5-only mode (S6): no config toggle; one-line note in output
   ("semantic search unavailable, keyword results only") + one Telegram alert per day
   max to #11.
2. Hybrid: RRF over vec + FTS5, `profile IN (:p,'shared')`, `status='active'`, top-20 → top-k.
3. Touch `last_retrieved` (feeds decay).
Reranker: none in Phase 1 (local-first); revisit at Voyage upgrade.

### 4.3 reflect(question, profile?)
recall(k=25, all sources) → one deepseek-v4-pro synthesis with observation-id citations.
On demand only.

### 4.4 forget(term | id)
- **Discovery:** fuzzy/FTS match → present **every** matched entry in full text with
  per-entry "🗑 Delete" / "❌ Keep" inline buttons (callback `fg:<8-char-hash>`, 64-byte
  cap respected via gateway-side mapping, same pattern as `_approval_state`).
- **Deletion:** by exact observation id only (S4). Removes from JSONL (git commit records
  removal), rebuilds affected index rows, scrubs matching session-DB rows, confirms scope.
- **Git-history warning (S2/S-A):** if the entry was ever committed, output includes:
  "removed from current state; persists in git history, BFG Repo-Cleaner is the nuclear
  option (documented in memory-architecture SKILL.md)."

### 4.5 session_search(), unchanged
recall() finds what the system *knows*; session_search finds what was *said*.

## 5. The Loops

### 5.1 Loop 1, Observer
**Trigger:** `on_session_end` plugin hook (Q1, confirmed; no cron fallback needed).
Receives `session_id`, `messages`, metadata. One deepseek-v4-pro call per session
(~1 session/day per Q3, cost negligible).

Prompt contract:
```
Extract durable observations from this transcript. JSON array.
Each: {type, content ≤500 chars present-tense atomic claim, epistemic:
"extracted" if the user stated it / "inferred" if you deduced it,
confidence 0-1, quote}.
Record all durable observations, 5-15 per session is normal (empty is
also valid). Exclude only: one-time events, task status, logs, secrets.
Corrections (user pushed back on agent behavior) are highest priority.
```
Writes flow through the remember() gate (dedupe + secrets, the observer gets no bypass;
single governed write path, Gbrain doctrine). Also writes one episode summary.

### 5.2 Loop 2, Consolidation (cron `0 4 * * *` **Asia/Singapore** (S7) · deliver `telegram:<chat_id>:<topic_id>` (Q7) · model deepseek-v4-pro)
1. **Merge**, cluster actives (cosine > 0.88) per profile → canonical entry, sum
   confirmations, members `superseded`.
2. **Contradict (A8)**, semantic conflicts between actives → set `contradicts`, keep
   both, surface in report. Never auto-resolve.
3. **Promote, hybrid gate (S3/S-B refinement of D2):**
   - `epistemic=extracted` AND `confidence ≥ 0.9` (or type=correction) → **auto-write**
     to MEMORY.md/IDENTITY.md + Telegram notice with "↩️ Revert" button
     (`rv:<8-char-hash>`). Revert → demote + `status=rejected`.
   - `epistemic=inferred` (any confidence) → **approval queue**: Telegram card with
     "✅ Promote" / "❌ Keep in spine" buttons. Never auto-enters the hot core.
4. **Demote**, hot-core entries lacking recent evidence and over budget → spine, with
   provenance.
5. **Decay & compact**, confidence −0.05 per 30 idle days; <0.3 → archived (reversible).
   Episodes >90 days → monthly digest, originals `compacted`.
Report: one-liner to #11 + full entry in wiki `_system/`; feeds the upgraded health report.

### 5.3 Loop 3, Session-start recall & activation manifest
**Mechanism:** `on_session_start` hook in the spine provider (wiki-aware-start Step 7
simplifies to invoking it). Injects once, turn 1:
1. **Activation manifest** (targets the four observed failure modes, CONTEXT §3.1):
   active profile · available skills relevant to the opening context · **explicit
   reminder: "you have cross-topic context access across this Telegram group"** ·
   [R]-tagged standing auto-execute rules restated as a checklist.
2. **Relevant memory:** recall(profile mission + latest context-digest headline, k=6).

## 6. Embedder & Benchmark

- **Local backend: all-MiniLM-L6-v2** (Q5, voyage-4-nano does not exist as a pip
  package; MiniLM is installed, ~500MB load / ~1.2GB batch peak, loaded on-demand only).
  Honest expectation: this matches the autopsy's flagged quality level, so Phase 1
  retrieval ≈ today's wiki index. The benchmark exists precisely to measure the upgrade.
- **Benchmark (A7):** ~20 fixed queries with expected-hit annotations in
  `~/wiki/_memory/bench/`, drawn from real past questions (mine the session DB). Run at:
  baseline, any embedder/retrieval change, weekly via health report. Regression >5%
  blocks a change. **Rebuild-parity bar (S5): scores within ±1% and same top-3 per query, not bit-identical** (float tie-breaking).
- **Voyage upgrade path (Phase 3):** after 2-4 wk observation window and the owner's go, voyage-4-lite via API key, `rebuild-index.py` re-embed, benchmark; commit only on ≥5%
  improvement, else revert. Embedder is behind an `embed(texts)->vectors` abstraction;
  swap = config + rebuild.

## 7. Integration Surface (from Q2/Q6/Q7 + review §4)

- **Provider:** implement the `MemoryProvider` ABC (`on_turn_start`, `on_session_end`,
  `on_session_switch`, `store`) as plugin `~/.hermes/hermes-agent/plugins/memory/spine/`;
  set `memory_provider: spine`. Not a tool-shadowing workaround.
- **Hooks:** `on_session_start` (manifest+recall) · `on_session_end` (observer) ·
  `on_session_reset` (context tracking reset). `pre_llm_call` reserved, do NOT use for
  per-turn injection (anti-Honcho principle).
- **Telegram:** InlineKeyboardMarkup confirmed; `callback_data` ≤64 bytes → compact
  hashes + gateway-side state map. All memory notifications/reports → **#11 System
  Admin** (`telegram:<chat_id>:<topic_id>`), consistent with existing memory crons.
- **Cron hygiene:** existing `Memory Space Monitor` (daily 07:20) becomes redundant once
  consolidation manages capacity by construction, review/retire in Phase 2.6. Weekly
  `Memory Health Report` is upgraded, not duplicated.

### config.yaml (target state)
```yaml
memory:
  memory_enabled: true
  memory_provider: spine
  memory_char_limit: 20000          # MEMORY.md (was 90000)
  identity_file: IDENTITY.md
  identity_char_limit: 5000         # replaces user_char_limit/USER.md
  memory_notifications: 'on'
  spine:
    canonical_root: ~/wiki/_memory
    db: ~/.hermes/memory.db
    embedder: all-MiniLM-L6-v2      # swap to voyage-4-lite at Phase 3.3
    loop_model: deepseek-v4-pro
    consolidation_cron: "0 4 * * *"  # Asia/Singapore
    report_target: "telegram:<chat_id>:<topic_id>"
    decay_per_30d: 0.05
    archive_threshold: 0.3
    promote_auto: {epistemic: extracted, min_confidence: 0.9}
    promote_queue: {epistemic: inferred}
    episode_compact_days: 90
```

## 8. Safety & Privacy Summary

| Concern | Handling |
|---|---|
| Secrets entering memory | Two-signal write gate (patterns hard-reject; entropy holds for review) + pre-commit hook on ~/wiki (Phase 0) |
| Secrets already in git history | forget() warns; BFG documented as nuclear option; known limitation |
| Hot-core poisoning | Hybrid promotion gate, inferred never auto-enters; extracted auto-writes are one-tap revertible and marked `rejected` on revert |
| Over-deletion | forget() deletes by exact id only, per-entry confirmation with full text shown |
| Index corruption/drift | db fully derived; rebuild drill is milestone M1; parity = ±1% + same top-3 |
| Embedder failure | Automatic FTS5-only degradation, flagged in output + daily-capped alert |
| Data ownership | Everything local + git; zero API dependency for core operation in Phase 1-2 |

## 9. Cost (steady state, post-Q3 volumes)

~1 session/day → observer ~$0.05-0.15/mo · consolidation ~$0.10-0.30/mo ·
reflect ad hoc ~$0.05 · embeddings/storage $0 → **≈ $0.20-0.50/month** on pro.
Ceiling ≤$2/mo holds with 4-10× volume growth. Tracked in the health report.

## 10. Pointers
Build order, milestones M1-M3, owner-only actions → **SCOPE.md**.
Environment, interview findings, Gbrain doctrine, path map, Q1-Q7 detail → **CONTEXT.md**.
Mission, principles, decision log D1-D14, success criteria → **PROJECT.md**.
