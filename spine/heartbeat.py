#!/usr/bin/env python3
"""Spine heartbeat — catches the failures that degrade silently.

Every check below is an incident that already happened, not a hypothetical:

  embedder    Semantic recall was dead for 8 days after a sentence-transformers
              method rename. It failed gracefully, fell back to keyword-only,
              and said so in its own output. Nobody read that output.
  vectors     A rebuild run under the wrong interpreter wiped every observation
              vector, printed a warning, and exited 0.
  hotcore     MEMORY.md grew unchecked to 86KB against a 90KB cap. Over the cap
              the memory tool rejects writes, so Hermes keeps its old memories
              and quietly stops forming new ones.
  sync        The Claude Code profile is a derived copy. If the sync stops
              running, recall keeps answering confidently from a stale mirror.
  divergence  Observation `status` lives only in the DB and is never written
              back to the canonical JSONL, so a rebuild silently reverts
              demotions and re-inflates the hot core.
  consolidate The nightly consolidation ran for weeks doing nothing at all.
  proposals   Hermes proposes Claude Code memories but never writes them. A
              queue nobody reviews is the same failure as a hot core nobody
              trims: it just sits there until someone happens to look.
  eval        A regression nobody notices is the whole point of this file.

DESIGN RULE, from the owner's own alerting principle: never fire on missing data.
A check that cannot get its evidence returns SKIP, not FAIL. Silent when blind.
And the script is silent when everything is healthy, because a heartbeat that
reports every day is one you stop reading.

Exit 0 = healthy or skipped. Exit 1 = something needs attention.
"""
from __future__ import annotations

import glob
import json
import re
import os
import sqlite3
import sys
import time

# The sys.path setup has to happen BEFORE the spine import, not inside main().
# A module-level `from spine.index import ...` was added on 2026-08-30 while the
# path inserts stayed in main(), so running this file as a script (which is how
# the daily cron runs it) died with ModuleNotFoundError before main() was ever
# reached. The heartbeat was down from 2026-08-30 to 09-03 and, because it is
# the thing that reports silent failures, nothing reported its own.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# spine/__init__.py imports agent.memory_provider, so the repo root has to be on
# the path too or this dies with ModuleNotFoundError: agent.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from spine.index import connect_db  # noqa: E402

OK, FAIL, SKIP = "OK", "FAIL", "SKIP"

# Must match propose_memories.EXPORT_PREFIX. Compared against the raw evidence
# JSON so this check needs no import from that module.
EXPORT_MARK = "exported to claude-code memory: "

MEM_MD = os.path.expanduser("~/.hermes/memories/MEMORY.md")
# Claude Code keys its per-project memory dir off the flattened absolute path
# of the project's cwd (e.g. /Users/you/... -> -Users-you-...). There's no
# real default for that key, so set SPINE_CLAUDE_PROJECT_KEY to your own.
_CLAUDE_PROJECT_KEY = os.environ.get("SPINE_CLAUDE_PROJECT_KEY", "-Users-youruser")
CLAUDE_MEM_DIR = os.path.expanduser(f"~/.claude/projects/{_CLAUDE_PROJECT_KEY}/memory")
CC_PROFILE = "agent:claude-code"
PROPOSAL_DIR = os.path.expanduser("~/.hermes/proposals/claude-memory")

# 2026-09-05: raised 20,000 -> 26,000 after measuring 44 nightly consolidation
# reports. The file adds ~4,200 bytes/day and the nightly demote removes ~2,450,
# so it nets about +1,800/day and settles in a 21,000-25,000 band. Against a
# 20,000 line that meant the check was red nearly every day and the only thing
# that ever brought it down was a human doing it by hand — the file hit 39,672
# on 2026-08-25 and was manually cut by 15,212 the next day.
#
# The content is not the problem. [R] rules plus [C] corrections alone are
# ~10,900 bytes, over half the old budget, and both are permanent and grow-only.
# Squeezing below that band is what cost an [R] rule on 2026-09-05.
#
# 20,000 was never a hard limit either: nothing in the codebase rejects a write
# at any size. It is a warn threshold, so it should fire when something is
# genuinely wrong rather than every single day. 26,000 sits above the natural
# band and leaves the weekly compression pass to handle real spikes.
#
# The durable fix is to stop loading situational corrections on every call and
# retrieve them instead. Tracked separately as hotcore-retrieval-split.
from spine.loops import HOTCORE_BUDGET_BYTES as HOTCORE_WARN_BYTES  # noqa: E402
SYNC_STALE_HOURS = 12             # cron runs every 4h; 3 misses is a real fault
CONSOLIDATE_STALE_HOURS = 48      # cron runs daily
EVAL_REGRESSION_TOLERANCE = 0     # any drop below baseline is a regression

HOTCORE_PATH = os.path.expanduser("~/.hermes/memories/MEMORY.md")
RULE_WATERMARK = os.path.expanduser("~/.hermes/state/hotcore-rule-watermark.json")

WIKI_INDEX_DIR = os.path.expanduser("~/.hermes/wiki_index")
WIKI_INDEX_STALE_HOURS = 48       # LaunchAgent runs daily; 2 misses is a real fault


def check_embedder(cfg):
    from spine import embedder
    if not embedder.embedder_available():
        return FAIL, "embedder unavailable — recall has silently dropped to keyword-only"
    dim = embedder.get_embedding_dim()
    con = connect_db(cfg.db)
    con.execute("PRAGMA query_only = ON")
    row = con.execute("SELECT value FROM dim_meta WHERE key='embedding_dim'").fetchone()
    con.close()
    if not row:
        return SKIP, "no embedding_dim recorded, nothing to compare against"
    if int(row[0]) != dim:
        return FAIL, f"embedder returns dim {dim} but the index was built at {row[0]}"
    return OK, f"loads, dim {dim}"


def check_vector_width(cfg):
    """Every stored vector must be the width the live embedder produces.

    Added after a real incident on 2026-08-20: an embedding-model swap was
    started, the re-embed was killed part way, and the code was left expecting
    768-dim vectors against a 384-dim store. Search did not error. It returned
    three confident, entirely wrong results. `check_vectors` passed throughout,
    because every row still had *a* vector -- just the wrong shape.

    A width mismatch is unrecoverable by retry: it needs a completed re-embed or
    a revert, so it is reported loudly rather than left to look healthy.
    """
    from spine import embedder
    if not embedder.embedder_available():
        return SKIP, "embedder unavailable, cannot compare widths"
    want = embedder.get_embedding_dim()
    if not want:
        return SKIP, "embedder reports no dimension"
    con = connect_db(cfg.db)
    con.execute("PRAGMA query_only = ON")
    try:
        widths = set()
        for table in ("observations", "wiki_chunks"):
            for (blob,) in con.execute(
                    f"SELECT embedding FROM {table} WHERE embedding IS NOT NULL"):
                if isinstance(blob, (bytes, memoryview)):
                    widths.add(len(blob) // 4)
    finally:
        con.close()
    if not widths:
        return SKIP, "no vectors stored"
    if widths != {want}:
        return FAIL, (f"embedder produces {want}-dim vectors but the store holds "
                      f"{sorted(widths)} — search will return confident nonsense; "
                      f"finish the re-embed or revert the model")
    return OK, f"all vectors {want}-dim, matching the live embedder"


def check_vectors(cfg):
    con = connect_db(cfg.db)
    con.execute("PRAGMA query_only = ON")
    total = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    novec = con.execute(
        "SELECT COUNT(*) FROM observations WHERE embedding IS NULL").fetchone()[0]
    wiki_novec = con.execute(
        "SELECT COUNT(*) FROM wiki_chunks WHERE embedding IS NULL").fetchone()[0]
    con.close()
    if not total:
        return SKIP, "no observations indexed"
    if novec or wiki_novec:
        return FAIL, (f"{novec}/{total} observations and {wiki_novec} wiki chunks have no "
                      f"vector — a rebuild probably ran without the embedder")
    return OK, f"{total} observations, all vectorised"


def check_hotcore(cfg):
    if not os.path.exists(MEM_MD):
        return SKIP, "MEMORY.md not found"
    size = os.path.getsize(MEM_MD)
    if size > HOTCORE_WARN_BYTES:
        return FAIL, (f"MEMORY.md is {size:,} bytes, over the {HOTCORE_WARN_BYTES:,} budget "
                      f"— roughly {size // 4:,} tokens on every Hermes call")
    return OK, f"{size:,} bytes ({size * 100 // HOTCORE_WARN_BYTES}% of budget)"


def check_sync(cfg):
    if not os.path.isdir(CLAUDE_MEM_DIR):
        return SKIP, "Claude Code memory dir not reachable"
    # Ask the SYNC what it expects rather than re-deriving it. Two private
    # copies of "how many rows should exist" is why rounds 4 and 5 broke this
    # check in opposite directions: first it counted only parseable files, then
    # it counted every unparseable file even when the file had never produced a
    # row at all (a born-broken .md has no prior record to retain), so a single
    # frontmatter typo pinned the check red while the sync itself ran green.
    # Explicit package import. A bare `import sync_claude_memories` resolves
    # to the cron shim of the same name on sys.path[0], which re-execs the
    # plugin source and clobbers sys.argv.
    from spine import sync_claude_memories as sync
    jsonl = os.path.join(os.path.expanduser(cfg.canonical_root), "observations",
                         f"{CC_PROFILE}.jsonl")
    if not os.path.exists(jsonl):
        return FAIL, f"{CC_PROFILE} has never been synced"

    prior = {}
    for line in open(jsonl, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "patch" not in d:
            prior[d["id"]] = d

    sync.MEM_DIR = CLAUDE_MEM_DIR
    recs, unparseable = sync.build_records("1970-01-01T00:00:00+00:00")
    expected = sync.expected_row_count(prior, recs, unparseable)
    n_files = len(recs)

    age_h = (time.time() - os.path.getmtime(jsonl)) / 3600
    con = connect_db(cfg.db)
    con.execute("PRAGMA query_only = ON")
    try:
        n_rows = con.execute("SELECT COUNT(*) FROM observations WHERE profile=?",
                             (CC_PROFILE,)).fetchone()[0]
    finally:
        con.close()

    if n_rows != expected:
        return FAIL, (f"expected {expected} rows for {CC_PROFILE} but found {n_rows} — "
                      f"the sync is not keeping up")
    if age_h > SYNC_STALE_HOURS:
        return FAIL, f"last sync was {age_h:.0f}h ago, expected within {SYNC_STALE_HOURS}h"

    note = (f", {len(unparseable)} unparseable ({', '.join(unparseable[:2])})"
            if unparseable else "")
    return OK, f"{n_rows} rows mirrored, last sync {age_h:.1f}h ago{note}"


def check_divergence(cfg):
    """DB status vs the canonical JSONL, with patch lines merged.

    Two bugs before: it compared each line's top-level `status`, last-write-wins,
    but `_write_back_status` appends {"id":..., "patch":{"status":...}} lines
    with NO top-level status -- so any row it had ever touched read as None and
    was treated as agreeing. Measured 2026-08-21: 190 of 421 ids, 45% of the
    store, were invisible to this check. And it returned OK unconditionally,
    while index.py cited it as the safety net for a swallowed write-back.
    """
    obs_dir = os.path.join(os.path.expanduser(cfg.canonical_root), "observations")
    if not os.path.isdir(obs_dir):
        return SKIP, "canonical store not reachable"

    on_disk = {}
    patches = {}
    for path in glob.glob(os.path.join(obs_dir, "*.jsonl")):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "patch" in d:
                patches.setdefault(d["id"], []).append(d["patch"])
            else:
                on_disk[d["id"]] = d.get("status")
    # Same merge order load_observations() uses, so this check agrees with what
    # a rebuild would actually produce.
    for oid, plist in patches.items():
        if oid not in on_disk:
            continue
        for patch in plist:
            if "status" in patch:
                on_disk[oid] = patch["status"]

    if not on_disk:
        return SKIP, "canonical store is empty"

    con = connect_db(cfg.db)
    con.execute("PRAGMA query_only = ON")
    try:
        diff = [i for i, s in con.execute("SELECT id, status FROM observations")
                if i in on_disk and on_disk[i] != s]
    finally:
        con.close()

    if diff:
        return FAIL, (f"{len(diff)} row(s) disagree between the DB and the canonical "
                      f"store — a rebuild would revert them; status write-back has "
                      f"failed somewhere (e.g. {', '.join(diff[:3])})")
    return OK, f"DB and canonical store agree on all {len(on_disk)} rows"


def check_consolidation(cfg):
    reports = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.expanduser(cfg.canonical_root)),
        "_system", "consolidation-*.json")))
    if not reports:
        return SKIP, "no consolidation reports found"
    age_h = (time.time() - os.path.getmtime(reports[-1])) / 3600
    if age_h > CONSOLIDATE_STALE_HOURS:
        return FAIL, f"last consolidation was {age_h / 24:.1f} days ago"
    try:
        rep = json.load(open(reports[-1], encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return SKIP, "latest consolidation report unreadable"
    warns = rep.get("warnings") or []
    if warns:
        return FAIL, "consolidation reported: " + "; ".join(warns)
    return OK, f"ran {age_h:.0f}h ago, no warnings"


def check_eval(cfg):
    # NOT __file__. The cron wrapper execs this file's source, so __file__ is
    # the wrapper's path (~/.hermes/scripts/spine) where no baseline exists --
    # the check returned "no eval baseline saved" and the regression gate has
    # never actually run under cron. Resolve from the imported module instead.
    import spine.heartbeat as _self
    here = os.path.dirname(os.path.abspath(_self.__file__))
    baselines = sorted(glob.glob(os.path.join(here, "eval_baseline_*.json")))
    if not baselines:
        return SKIP, "no eval baseline saved"
    try:
        base = json.load(open(baselines[-1], encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return SKIP, f"baseline unreadable: {type(e).__name__}"
    try:
        sys.path.insert(0, here)
        import eval_run
        # Compare like with like: a baseline saved under a different
        # profile is not a valid floor.
        # Validate the SET before trusting the score. A multi-hop case answerable
        # from one document scores retrieval that never had to connect anything;
        # 11 of 12 were silently in that state once. --validate was otherwise a
        # flag a human had to remember to type.
        mislabelled = eval_run.validate(verbose=False)
        now = eval_run.run(base.get("profile", "*"))
    except Exception as e:  # noqa: BLE001
        # SKIP is for ABSENT evidence. A harness that raises is a broken tool,
        # and this is the regression gate -- the one check that must not be able
        # to go quietly dark.
        return FAIL, f"eval harness failed to run: {type(e).__name__}: {e}"
    out = []
    # All three sections, or a regression in the one left out goes unnoticed --
    # which is the entire failure class this file exists for.
    for hop in ("single", "conj", "multi"):
        b = base.get("by_hop", {}).get(hop, {}).get("passed")
        n = now.get("by_hop", {}).get(hop, {}).get("passed")
        tot = now.get("by_hop", {}).get(hop, {}).get("total")
        if b is None:
            continue
        out.append(f"{hop} {n}/{tot} (was {b})")
        if n < b - EVAL_REGRESSION_TOLERANCE:
            return FAIL, f"{hop}-hop recall regressed: {n} vs baseline {b}"
    if mislabelled:
        # Lead with the REASON. This returned ", ".join(out) -- the scores only --
        # so a red check reported perfect numbers with no case id and no cause.
        return FAIL, (f"eval set has {len(mislabelled)} mislabelled multi-hop "
                      f"case(s) ({', '.join(mislabelled)}): a single document "
                      f"answers them, so they test nothing. Scores: "
                      + ", ".join(out[1:] if out and out[0].startswith(str(len(mislabelled))) else out))
    return OK, ", ".join(out) if out else "baseline has no sections to compare"


def check_hotcore_coverage(cfg):
    """Every hot-core block must exist in spine, or trimming it destroys it.

    MEMORY.md has a second writer: Hermes's memory() tool writes straight into
    the file and never touches spine. 27 blocks were in that state when this
    check was written. A cleanup that assumed "these are in spine already"
    would have deleted all of them.
    """
    from spine import coverage
    if not os.path.exists(MEM_MD):
        return SKIP, "MEMORY.md not found"
    con = connect_db(cfg.db)
    con.execute("PRAGMA query_only = ON")
    try:
        total = len(coverage.hotcore_blocks(MEM_MD))
        uncovered = coverage.uncovered_hotcore(MEM_MD, con)
    finally:
        con.close()
    if not total:
        return SKIP, "hot core is empty"
    if uncovered:
        # Separate what sync_hotcore CAN fix from what it cannot. The secrets
        # detector holds blocks containing high-entropy tokens (file paths trip
        # it), and sync_hotcore now treats that as a terminal state -- so telling
        # the operator to "run sync_hotcore.py" for those is advice that exits 0
        # and changes nothing. A permanently red guard is a guard people stop
        # reading, and this one is the last thing between a MEMORY.md trim and
        # destroying blocks that exist nowhere else.
        from spine.coverage import strip_tag
        from spine.secrets_detector import detect_secrets
        importable, withheld = [], []
        for block, _missing in uncovered:
            verdict = detect_secrets(strip_tag(block))
            # detect_secrets' own contract is {"blocked": bool} /
            # {"held_for_review": bool} -- NOT handle_remember's outward
            # {"held", "rejected"} keys. Guessing the wrong shape here would
            # have made this branch dead code.
            (withheld if (verdict.get("blocked") or verdict.get("held_for_review"))
             else importable).append(block)
        if importable:
            return FAIL, (f"{len(importable)}/{total} hot-core blocks are not retrievable "
                          f"from spine — run sync_hotcore.py before trimming MEMORY.md")
        return FAIL, (f"{len(withheld)}/{total} hot-core blocks cannot be imported: the "
                      f"secrets detector withholds them. sync_hotcore cannot fix these — "
                      f"reword or trim them by hand, and do NOT trim MEMORY.md until then")
    return OK, f"all {total} hot-core blocks retrievable from spine"


def check_proposals(cfg):
    """Two ways the Hermes -> Claude Code write path fails silently.

    Pending: a proposal is written and then nobody reviews it. Nothing breaks,
    so nothing complains, and the queue just sits there.

    Unmarked: a proposal is promoted into ~/.claude but the source spine row is
    never stamped exported. Nothing looks wrong, and the row is proposed again
    on every future run. The 4h sync then returns the promoted file as an
    agent:claude-code observation while agent:main still holds the original, so
    the duplicate compounds rather than sitting still.
    """
    if not os.path.isdir(PROPOSAL_DIR):
        return SKIP, "no proposal queue"

    pending = [f for f in glob.glob(os.path.join(PROPOSAL_DIR, "*.md"))]
    problems = []
    if pending:
        oldest = min(os.path.getmtime(f) for f in pending)
        age_d = (time.time() - oldest) / 86400
        problems.append(f"{len(pending)} proposal(s) awaiting review, oldest {age_d:.0f}d old")

    if not os.path.isdir(CLAUDE_MEM_DIR):
        return (FAIL, "; ".join(problems)) if problems else (SKIP, "Claude Code memory dir not reachable")

    promoted = {}
    for path in glob.glob(os.path.join(CLAUDE_MEM_DIR, "*.md")):
        if os.path.basename(path) == "MEMORY.md":
            continue
        try:
            head = open(path, encoding="utf-8").read(2000)
        except OSError:
            continue
        m = re.search(r"^\s*source_obs_id:\s*(\S+)\s*$", head, re.M)
        if m:
            promoted[m.group(1)] = os.path.basename(path)

    if promoted:
        con = connect_db(f"file:{cfg.db}?immutable=1", uri=True)
        try:
            qs = ",".join("?" * len(promoted))
            rows = dict(con.execute(
                f"SELECT id, evidence FROM observations WHERE id IN ({qs})",
                list(promoted)).fetchall())
        finally:
            con.close()
        unmarked = []
        for oid, fname in sorted(promoted.items()):
            ev = rows.get(oid)
            if ev is None:
                continue  # row is gone from this profile; not this check's job
            if EXPORT_MARK not in (ev or ""):
                unmarked.append(fname)
        if unmarked:
            problems.append(
                f"{len(unmarked)} promoted memor(y/ies) whose spine row is not marked "
                f"exported and will be proposed again: {', '.join(unmarked[:3])}"
                + (" ..." if len(unmarked) > 3 else ""))

    if problems:
        return FAIL, "; ".join(problems)
    return OK, f"no proposals pending, {len(promoted)} promoted memories marked exported"


def check_fts_index(cfg):
    """Watch the FTS index for phantom-document drift and tokenizer regressions.

    2026-08-23: the index held 2,082 docs against 521 real observations (75%
    phantoms) and 4,952 against 2,643 wiki chunks. INSERT OR REPLACE did not
    fire the FTS delete trigger (recursive_triggers was off), so every re-write
    leaked a stale document. count(*) on an external-content FTS table reads
    the CONTENT table (looks right), integrity-check validates structure only,
    and search_fts JOINs orphans away from results — so it was invisible to
    every existing check while bm25 computed every IDF against a
    majority-ghost corpus. The docsize shadow table is the one count that
    reflects the index itself.
    """
    con = connect_db(cfg.db)
    con.execute("PRAGMA query_only = ON")
    try:
        problems = []
        for tbl, src in (("observations_fts", "observations"),
                         ("wiki_chunks_fts", "wiki_chunks")):
            indexed = con.execute(
                f"SELECT count(*) FROM {tbl}_docsize").fetchone()[0]
            real = con.execute(
                f"SELECT count(*) FROM {src}").fetchone()[0]
            if indexed != real:
                problems.append(
                    f"{tbl} indexes {indexed} docs but {src} has {real} "
                    f"rows ({indexed - real} phantoms)")
            row = con.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (tbl,)).fetchone()
            if not row or "porter" not in (row[0] or ""):
                problems.append(
                    f"{tbl} missing porter tokenizer — stemming broken")
    finally:
        con.close()
    if problems:
        return FAIL, "; ".join(problems)
    return OK, "obs/wiki FTS indexes match row counts, porter tokenizer present"


def check_wiki_index(cfg):
    """Watch the wiki semantic index for the staleness that already happened.

    2026-09-04: index.faiss was last built 2026-06-12 and nothing had rebuilt
    it in the 84 days since. It was never automated at all — no cron, no
    LaunchAgent, one manual run in June. So Hermes `/wiki_search` and
    session-kickoff §0g kept answering from a June snapshot of a vault that had
    grown to 1,207 indexable notes. Same shape as the `embedder` check above:
    it never errored, it just answered from the past.

    Also asserts the metadata/vector alignment invariant. wiki-search.py maps a
    FAISS row position straight into metadata[idx], with no dedup and no bounds
    reconciliation, so if those two files ever drift out of step every result
    silently points at the wrong note.
    """
    index_file = os.path.join(WIKI_INDEX_DIR, "index.faiss")
    meta_file = os.path.join(WIKI_INDEX_DIR, "metadata.jsonl")
    vecs_file = os.path.join(WIKI_INDEX_DIR, "vectors.npy")

    if not os.path.exists(index_file):
        return SKIP, "no wiki index on disk — embedder has never run"

    problems = []
    age_h = (time.time() - os.path.getmtime(index_file)) / 3600
    if age_h > WIKI_INDEX_STALE_HOURS:
        problems.append(
            f"index.faiss is {age_h / 24:.1f}d old (limit "
            f"{WIKI_INDEX_STALE_HOURS}h) — the wiki-embedder cron "
            f"is not running")

    rows = None
    if os.path.exists(meta_file):
        with open(meta_file) as fh:
            rows = sum(1 for line in fh if line.strip())

    if os.path.exists(vecs_file) and rows is not None:
        import numpy as np
        # mmap: read the header and shape, never pull 11MB of floats into a
        # heartbeat that is supposed to be cheap enough to run daily.
        n_vecs = np.load(vecs_file, mmap_mode="r").shape[0]
        if rows != n_vecs:
            problems.append(
                f"metadata.jsonl has {rows} rows but vectors.npy has "
                f"{n_vecs} — search results would point at the wrong notes")

    if problems:
        return FAIL, "; ".join(problems)
    return OK, (f"wiki index {age_h:.0f}h old"
                + (f", {rows} chunks aligned" if rows is not None else ""))


def check_hotcore_rules(cfg):
    """[R] rules are permanent. Assert the count never drops.

    2026-09-05: the consolidation cron was told, in capitals, never to touch a
    block tagged [R]. It compressed one anyway — the two-Macs rule — folding it
    into an untagged block and dropping the fact that the M1 and M4 are in
    different physical locations. Nothing was permanently lost, because the
    coverage step had already pushed every block into spine, but the hot core
    lost a rule and its protection tag in one pass and nothing noticed.

    A prompt instruction has a non-zero failure rate; a count comparison does
    not. This is the deterministic half of that rule.

    The watermark is the high-water mark of [R] blocks ever seen. Deleting a
    rule on purpose is legitimate, so the failure message says how to accept the
    new count rather than treating every drop as corruption.
    """
    if not os.path.exists(HOTCORE_PATH):
        return SKIP, "no MEMORY.md on disk"

    with open(HOTCORE_PATH, encoding="utf-8", errors="ignore") as fh:
        current = sum(1 for line in fh if line.startswith("[R]"))

    mark = None
    if os.path.exists(RULE_WATERMARK):
        try:
            mark = json.load(open(RULE_WATERMARK, encoding="utf-8")).get("rules")
        except (json.JSONDecodeError, OSError):
            mark = None

    if mark is None or current > mark:
        os.makedirs(os.path.dirname(RULE_WATERMARK), exist_ok=True)
        with open(RULE_WATERMARK, "w", encoding="utf-8") as fh:
            json.dump({"rules": current, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh)
        return OK, f"{current} [R] rules (watermark set)"

    if current < mark:
        return FAIL, (f"MEMORY.md has {current} [R] rules but {mark} were seen "
                      f"before — a permanent rule was removed or lost its tag. "
                      f"Restore it, or if the deletion was deliberate update "
                      f"{RULE_WATERMARK}")

    return OK, f"{current} [R] rules, none lost"


def check_hotcore_triggers(cfg):
    """Every @when: marker in the hot core must be one fires() can satisfy.

    A block tagged with a kind rule_scope does not understand falls back to
    universal, so it still reaches the model — safe, but silently not doing what
    whoever wrote the tag intended. "@when:tool" with no glob, or
    "@when:planetry=mars", would sit there looking scoped and never scope
    anything. That is invisible from the outside, which is exactly the class of
    fault this file exists for.

    Added 2026-09-05 with hotcore-retrieval-split phase 2.
    """
    if not os.path.exists(HOTCORE_PATH):
        return SKIP, "no MEMORY.md on disk"
    try:
        from spine.rule_scope import (
            WHEN_RE, SUBJECT, TOOL, VALID_KINDS, split_blocks, parse_trigger,
        )
    except Exception as e:  # noqa: BLE001
        return SKIP, f"rule_scope unavailable ({type(e).__name__})"

    with open(HOTCORE_PATH, encoding="utf-8", errors="ignore") as fh:
        blocks = split_blocks(fh.read())

    problems = []
    tagged = 0
    for b in blocks:
        m = WHEN_RE.search(b)
        if not m:
            continue
        tagged += 1
        kind = (m.group(1) or "").lower()
        head = b.lstrip()[:60].replace("\n", " ")
        if kind not in VALID_KINDS:
            problems.append(f"unknown trigger '{kind}' — treated as universal: {head}")
            continue
        if kind in (SUBJECT, TOOL) and not parse_trigger(b).values:
            problems.append(f"'{kind}' with no match terms — can never scope: {head}")

    if problems:
        return FAIL, "; ".join(problems[:3]) + (
            f" (+{len(problems) - 3} more)" if len(problems) > 3 else "")
    if tagged == 0:
        return OK, f"no @when: markers yet, all {len(blocks)} blocks universal"
    return OK, f"{tagged} tagged block(s), all triggers reachable"


MCP_CONFIG = os.path.expanduser("~/.claude.json")
MCP_SERVER_NAME = "spine-memory"
MCP_INIT_TIMEOUT = 20             # Claude Code allows 30; fail before it does
MCP_EXPECTED_TOOLS = 6            # memory_recall/remember/recall_at/reflect/forget/explain


def check_mcp_server(cfg):
    """spine-memory must complete an MCP handshake and still expose its tools.

    2026-08-28 to 09-09: this server was dead for twelve days and nobody knew.
    hermes-agent's venv moved to mcp 2.0.0 (pinned for CVE-2026-48710), which
    removed `mcp.server.fastmcp`; spine-mcp borrows that venv because it imports
    this plugin from this checkout, so the bump killed it at import. Thirteen
    Claude Code sessions ran with no spine memory, and the only signal was a
    greyed-out server name. Every other MCP server survived because each has its
    own pinned venv. spine-mcp cannot -- it needs this one -- so a watcher is the
    fix instead of isolation.

    Only the handshake and the tool list are checked. Calling a tool would load
    torch and MiniLM, which `embedder` already covers, and would make this check
    depend on model state rather than on the server being alive.
    """
    import json as _json
    import subprocess
    import threading

    try:
        with open(MCP_CONFIG) as fh:
            spec = (_json.load(fh).get("mcpServers") or {}).get(MCP_SERVER_NAME)
    except (OSError, ValueError) as exc:
        # Blind is not broken: Claude Code's config is not ours to depend on.
        return SKIP, f"cannot read {MCP_CONFIG} ({type(exc).__name__}) — nothing to check"

    if not spec or not spec.get("command"):
        return SKIP, f"no local `{MCP_SERVER_NAME}` server configured for Claude Code"
    if not os.path.exists(spec["command"]):
        return FAIL, f"interpreter missing: {spec['command']}"

    argv = [spec["command"]] + list(spec.get("args") or [])
    env = dict(os.environ)
    env.update(spec.get("env") or {})

    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=env,
        )
    except OSError as exc:
        return FAIL, f"cannot start the server: {exc}"

    def send(payload):
        try:
            proc.stdin.write(_json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass  # it died; stderr below says why

    def read_line(timeout):
        box = {}
        t = threading.Thread(target=lambda: box.update(line=proc.stdout.readline()), daemon=True)
        t.start()
        t.join(timeout)
        return box.get("line")

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2026-07-28", "capabilities": {},
                         "clientInfo": {"name": "spine-heartbeat", "version": "1"}}})
        init = read_line(MCP_INIT_TIMEOUT)
        if not init:
            err = (proc.stderr.read() or "").strip().splitlines()
            last = err[-1] if err else f"no initialize response within {MCP_INIT_TIMEOUT}s"
            return FAIL, f"server does not answer — Claude Code has no spine memory ({last})"

        try:
            payload = _json.loads(init)
        except ValueError:
            return FAIL, f"non-JSON on stdout, which corrupts the protocol: {init.strip()[:80]}"
        if "result" not in payload:
            return FAIL, f"initialize rejected: {_json.dumps(payload.get('error'))[:80]}"

        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        listed = read_line(MCP_INIT_TIMEOUT)
        if not listed:
            return FAIL, "handshake ok but tools/list never answered"
        try:
            tools = _json.loads(listed)["result"]["tools"]
        except (ValueError, KeyError, TypeError):
            return FAIL, f"tools/list returned nothing usable: {listed.strip()[:80]}"

        if len(tools) < MCP_EXPECTED_TOOLS:
            names = ", ".join(sorted(t.get("name", "?") for t in tools)) or "none"
            return FAIL, (f"only {len(tools)} of {MCP_EXPECTED_TOOLS} tools exposed ({names}) — "
                          f"a decorator or an import is broken")
        return OK, f"handshake ok, {len(tools)} tools exposed"
    finally:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001 - teardown must never mask the verdict
            pass


CHECKS = [
    ("embedder", check_embedder),
    ("mcp_server", check_mcp_server),
    ("vectors", check_vectors),
    ("vector_width", check_vector_width),
    ("fts_index", check_fts_index),
    ("wiki_index", check_wiki_index),
    ("hotcore", check_hotcore),
    ("hotcore_rules", check_hotcore_rules),
    ("hotcore_triggers", check_hotcore_triggers),
    ("hotcore_coverage", check_hotcore_coverage),
    ("sync", check_sync),
    ("divergence", check_divergence),
    ("consolidation", check_consolidation),
    ("proposals", check_proposals),
    ("eval", check_eval),
]


def main() -> None:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # spine/__init__.py imports agent.memory_provider, so the repo root has to
    # be on the path too or a manual run dies with ModuleNotFoundError: agent.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from spine.config import load_spine_config
    cfg = load_spine_config()

    results = []
    for name, fn in CHECKS:
        try:
            status, msg = fn(cfg)
        except Exception as e:  # noqa: BLE001
            # A check that crashes is itself a fault worth reporting. Never let
            # one broken check hide the other six.
            status, msg = FAIL, f"check crashed: {type(e).__name__}: {e}"
        results.append((name, status, msg))

    failures = [r for r in results if r[1] == FAIL]

    if failures or verbose:
        print("🫀 **Spine heartbeat**\n")
        for name, status, msg in results:
            icon = {"OK": "✅", "FAIL": "🔴", "SKIP": "⚪"}[status]
            print(f"{icon} `{name}` — {msg}")
        if failures:
            print(f"\n**{len(failures)} check(s) need attention.**")
    # Silent on a clean run: a daily all-clear is a message you stop reading.

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
