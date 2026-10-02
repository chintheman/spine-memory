"""Regression tests for defects found by two independent reviews of spine.

Every test here corresponds to a bug that actually shipped. The plugin had no
tests at all, so each fix rested on a claim in a commit message. These pin the
behaviour instead.

Run: python3 -m pytest tests -q   (from the repo root)
"""
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time

# Repo layout here is `<root>/spine/` + `<root>/tests/` (this file), one level
# shallower than the original plugins/memory/spine/tests/ checkout path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                          # for `from spine import loops`
# spine/ itself: eval_run.py and sync_claude_memories.py are scripts, not part
# of the package, so they import as top-level modules.
sys.path.insert(0, os.path.join(_ROOT, "spine"))

from spine import loops  # noqa: E402


# ── hot-core file format ──────────────────────────────────────────────

def _round_trips(raw: str) -> bool:
    """The exact check memory_tool._detect_external_drift performs."""
    parts = [e.strip() for e in raw.split("\n§\n") if e.strip()]
    return raw.strip() == "\n§\n".join(parts)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _never_touch_the_real_hotcore():
    """Point HOTCORE_PATH somewhere disposable for EVERY test, and restore it.

    loops.HOTCORE_PATH defaults to the user's real ~/.hermes/memories/MEMORY.md,
    which is injected into every agent call. Tests mutated that module global
    with no teardown, so any future test reaching _promote_to_hotcore or
    run_consolidation before setting it would write to production -- and this
    suite now runs inside the shared 35k-test session.
    """
    original = loops.HOTCORE_PATH
    guard = tempfile.mktemp(suffix=".md")
    open(guard, "w", encoding="utf-8").write("[F] sandbox\n")
    loops.HOTCORE_PATH = guard
    try:
        yield
    finally:
        loops.HOTCORE_PATH = original
        if os.path.exists(guard):
            os.remove(guard)


def _hotcore(tmp_path_str, body):
    open(tmp_path_str, "w", encoding="utf-8").write(body)
    loops.HOTCORE_PATH = tmp_path_str


def test_promote_keeps_the_file_round_trippable():
    """A raw append produced '\\n\\n§\\n', which makes memory() refuse ALL writes."""
    f = tempfile.mktemp(suffix=".md")
    _hotcore(f, "[F] alpha\n§\n[F] beta\n")
    assert _round_trips(open(f).read())
    # Distinct facts on purpose: "entry 0/1/2" are rewords of each other
    # (cosine 0.888) and the promote dedupe now refuses them, which is correct
    # and is not what this test is about.
    for i, text in enumerate(["gamma is a colour space",
                              "delta airlines flies to Tokyo",
                              "epsilon is a Greek letter"]):
        assert loops._promote_to_hotcore(f"id{i}", text, "fact", None) == "written"
        assert _round_trips(open(f).read()), f"drifted after promote {i}"
    os.remove(f)


def test_promote_is_tristate():
    """duplicate must be distinguishable from failed, or rows strand as 'active'."""
    f = tempfile.mktemp(suffix=".md")
    _hotcore(f, "[F] alpha\n")
    assert loops._promote_to_hotcore("a", "brand new", "fact", None) == "written"
    assert loops._promote_to_hotcore("a", "brand new", "fact", None) == "duplicate"
    loops.HOTCORE_PATH = "/nonexistent-dir-xyz/MEMORY.md"
    assert loops._promote_to_hotcore("a", "x", "fact", None) == "failed"
    os.remove(f)


def test_promote_dedupe_is_tag_insensitive():
    """A re-tagged copy of an existing block must not append a second one.

    sync_hotcore re-remembers hot-core blocks as type "fact", so "[W] x" comes
    back as "[F] x". A full-string comparison never matched and the file grew a
    duplicate every run -- a growth engine inside the prompt injected on every
    call.
    """
    f = tempfile.mktemp(suffix=".md")
    _hotcore(f, "[W] the user prefers short replies\n§\n[R] never use em dashes\n")
    before = len(loops._read_hotcore_blocks(f))
    # same bodies, different type -> different tag
    assert loops._promote_to_hotcore("a", "the user prefers short replies", "fact", None) == "duplicate"
    assert loops._promote_to_hotcore("b", "never use em dashes", "fact", None) == "duplicate"
    assert len(loops._read_hotcore_blocks(f)) == before, "dedupe appended a re-tagged copy"
    os.remove(f)


def test_excise_removes_exactly_one_block():
    blocks = ["[R] shared text", "[F] shared text", "[W] other"]
    out = loops._excise_block(blocks, "shared text")
    assert len(out) == 2, "removed more than one block"
    assert loops._excise_block(["[F] a"], "absent") is None


def test_hotcore_lock_serialises_a_concurrent_writer():
    """The demote rewrite must not discard a memory() append landing mid-pass."""
    f = tempfile.mktemp(suffix=".md")
    _hotcore(f, "[F] alpha\n§\n[F] beta\n")
    order = []

    def rewriter():
        with loops._hotcore_lock():
            order.append("rewrite:start")
            time.sleep(0.6)
            blocks = loops._read_hotcore_blocks(f)
            loops._write_hotcore_blocks(f, [b for b in blocks if "beta" not in b])
            order.append("rewrite:done")

    def appender():
        time.sleep(0.2)
        with loops._hotcore_lock():
            order.append("append:acquired")
            blocks = loops._read_hotcore_blocks(f)
            loops._write_hotcore_blocks(f, blocks + ["[W] concurrent"])

    t1, t2 = threading.Thread(target=rewriter), threading.Thread(target=appender)
    t1.start(); t2.start(); t1.join(); t2.join()
    body = open(f).read()
    assert order.index("rewrite:done") < order.index("append:acquired")
    assert "concurrent" in body, "concurrent write was destroyed"
    assert "beta" not in body, "excision did not happen"
    assert _round_trips(body)
    os.remove(f)


# ── eval scorer ───────────────────────────────────────────────────────

def _judge(case, docs):
    """Score through eval_run's REAL judge(), not a reimplementation."""
    import eval_run
    return eval_run.judge(case, [{"content": d} for d in docs])


def test_min_sources_rejects_one_doc_holding_every_needle():
    case = {"id": "t", "q": "q", "hop": "multi",
            "expect_all": ["alpha", "beta"], "min_sources": 2}
    r = _judge(case, ["alpha and beta together", "alpha only"])
    assert r["n_sources"] == 1, f"expected 1 contributor, got {r['n_sources']}"
    assert not r["passed"]


def test_min_sources_accepts_two_genuine_contributors():
    case = {"id": "t", "q": "q", "hop": "multi",
            "expect_all": ["alpha", "beta"], "min_sources": 2}
    r = _judge(case, ["alpha only", "beta only"])
    assert r["n_sources"] == 2 and r["passed"]


def test_conj_ignores_source_count_but_still_needs_every_needle():
    case = {"id": "t", "q": "q", "hop": "conj", "expect_all": ["alpha", "beta"]}
    assert _judge(case, ["alpha and beta"])["passed"]
    assert not _judge(case, ["alpha only"])["passed"]


# ── canonical store ───────────────────────────────────────────────────

def test_sync_carries_status_and_age_forward():
    """Records are rebuilt as 'active'; a demote must not be reverted."""
    import sync_claude_memories as sync
    fresh = {"content": "same", "status": "active",
             "created_at": "2026-08-01", "last_confirmed": "2026-08-01"}
    prior = {"content": "same", "status": "demoted",
             "created_at": "2026-01-01", "last_confirmed": "2026-02-02"}
    assert sync.merge_prior(dict(fresh), None) == "added"
    r = dict(fresh)
    assert sync.merge_prior(r, prior) == "unchanged"
    assert r["status"] == "demoted", "status reverted to active"
    assert r["created_at"] == "2026-01-01", "age was reset"
    r2 = dict(fresh)
    assert sync.merge_prior(r2, dict(prior, content="different")) == "changed"
    assert r2["status"] == "demoted" and r2["created_at"] == "2026-01-01"


def test_wildcard_profile_write_is_refused():
    """A wildcard is a READ scope; writing one must fail loudly, not silently.

    Briefly it returned the agent:main log instead, which let handle_remember
    (free-form profile in its schema) write {"profile": "*"} into agent:main's
    file: a row consolidation never touches, recall by concrete profile never
    sees, and forget cannot reach.
    """
    from spine.tools import _get_writer, handle_remember
    from spine.config import SpineConfig
    cfg = SpineConfig(canonical_root=tempfile.mkdtemp())
    try:
        _get_writer(cfg, "*")
    except ValueError:
        pass
    else:
        raise AssertionError("_get_writer accepted a wildcard profile")
    r = json.loads(handle_remember({"content": "x", "profile": "*"}, cfg))
    assert "error" in r, "handle_remember accepted a wildcard profile"

def test_fixture_restores_the_real_hotcore_path():
    """Assert on the RESTORED value, not the guard.

    The previous version read HOTCORE_PATH inside the test body, where the
    fixture has already swapped it, so deleting the fixture's `finally` left all
    12 tests green -- the one test guarding the fixture could not see it break.
    """
    import contextlib
    original = loops.HOTCORE_PATH
    gen = _never_touch_the_real_hotcore.__wrapped__()
    next(gen)
    assert loops.HOTCORE_PATH != original, "fixture did not redirect"
    with contextlib.suppress(StopIteration):
        next(gen)
    assert loops.HOTCORE_PATH == original, "fixture did not restore the real path"


def _sync_fixture(files):
    """A memory dir plus the previous JSONL, ready for classify_gone()."""
    import sync_claude_memories as sync
    d = tempfile.mkdtemp()
    md = os.path.join(d, "mem")
    os.makedirs(md)
    for name, ok in files.items():
        body = (f"---\nname: {name}\ndescription: d\nmetadata:\n  type: project\n"
                f"---\n\nbody of {name}\n") if ok else "BROKEN NO FRONTMATTER\n"
        open(os.path.join(md, f"{name}.md"), "w").write(body)
    sync.MEM_DIR = md
    return sync


def test_deleted_file_is_dropped_but_unparseable_file_is_kept():
    """Four notions of "gone" lived in this module and each round broke one.

    Round 3 resurrected deleted memories into the source of truth. Round 4 fixed
    that and started ERASING memories whose frontmatter merely broke. Both are
    data loss in the append-only store.
    """
    sync = _sync_fixture({"alpha": True, "bravo": True, "charlie": True})
    recs, skipped = sync.build_records("2026-01-01T00:00:00+00:00")
    prior = {r["id"]: r for r in recs}
    assert len(prior) == 3 and not skipped

    # bravo deleted, charlie's frontmatter broken
    os.remove(os.path.join(sync.MEM_DIR, "bravo.md"))
    open(os.path.join(sync.MEM_DIR, "charlie.md"), "w").write("BROKEN\n")
    recs2, skipped2 = sync.build_records("2026-01-02T00:00:00+00:00")
    assert skipped2 == ["charlie.md"]

    gone = sync.classify_gone(prior, recs2, skipped2)
    bravo = sync.stable_id("bravo.md")
    charlie = sync.stable_id("charlie.md")
    assert bravo in gone["deleted"], "a deleted file must be removed"
    assert charlie in gone["skipped"], "an unparseable file must NOT be a deletion"
    assert charlie not in gone["deleted"]


def test_lock_window_carries_new_records_but_not_deleted_ones():
    sync = _sync_fixture({"alpha": True, "bravo": True})
    recs, skipped = sync.build_records("2026-01-01T00:00:00+00:00")
    prior = {r["id"]: r for r in recs}
    os.remove(os.path.join(sync.MEM_DIR, "bravo.md"))
    recs2, skipped2 = sync.build_records("2026-01-02T00:00:00+00:00")
    gone = sync.classify_gone(prior, recs2, skipped2)
    built, skips = gone["built"], gone["skipped"]

    assert sync.is_foreign("brand_new_id", prior, built, skips), \
        "a record appended during the lock window must be carried through"
    assert not sync.is_foreign(sync.stable_id("bravo.md"), prior, built, skips), \
        "a deleted memory was resurrected"
    assert not sync.is_foreign(recs2[0]["id"], prior, built, skips)


def test_born_broken_file_does_not_inflate_the_expected_row_count():
    """A .md that has NEVER parsed produces no row, so it must not be expected.

    check_sync kept its own copy of this number and rounds 4 and 5 broke it in
    opposite directions: counting only parseable files, then counting every
    unparseable file even when it had never produced a row -- pinning the check
    red while the sync ran green.
    """
    sync = _sync_fixture({"alpha": True, "bravo": True})
    recs, skipped = sync.build_records("2026-01-01T00:00:00+00:00")
    prior = {r["id"]: r for r in recs}
    assert sync.expected_row_count(prior, recs, skipped) == 2

    # a brand-new broken file: no prior record, so no row is ever created
    open(os.path.join(sync.MEM_DIR, "delta.md"), "w").write("BROKEN\n")
    recs2, skipped2 = sync.build_records("2026-01-02T00:00:00+00:00")
    assert skipped2 == ["delta.md"]
    assert sync.expected_row_count(prior, recs2, skipped2) == 2, \
        "a file that never parsed must not be counted"

    # a file that HAD parsed and is now broken keeps its row, so it IS counted
    open(os.path.join(sync.MEM_DIR, "bravo.md"), "w").write("BROKEN\n")
    recs3, skipped3 = sync.build_records("2026-01-03T00:00:00+00:00")
    assert sync.expected_row_count(prior, recs3, skipped3) == 2


def test_foreign_records_are_never_deleted():
    """Rows another writer put in this profile are not this sync's to remove.

    They were carried through the lock window, then landed in `deleted` on the
    very next run 4h later and were dropped from the append-only store.
    """
    sync = _sync_fixture({"alpha": True})
    recs, skipped = sync.build_records("2026-01-01T00:00:00+00:00")
    prior = {r["id"]: r for r in recs}
    prior["FOREIGN"] = {"id": "FOREIGN", "profile": sync.PROFILE,
                        "evidence": ["handle_remember"]}
    gone = sync.classify_gone(prior, recs, skipped)
    assert "FOREIGN" in gone["foreign"]
    assert "FOREIGN" not in gone["deleted"], "a foreign record was scheduled for deletion"
    # and a derived record whose file is gone still IS deleted
    recs2, skipped2 = sync.build_records("2026-01-02T00:00:00+00:00")
    os.remove(os.path.join(sync.MEM_DIR, "alpha.md"))
    recs3, skipped3 = sync.build_records("2026-01-03T00:00:00+00:00")
    assert sync.classify_gone(prior, recs3, skipped3)["deleted"] == {recs[0]["id"]}


# ── the Hermes -> Claude Code write path ──────────────────────────────

def _proposals_env(tmp, proposals, memories, evidence_by_id):
    """Point heartbeat's proposals check at throwaway dirs and a throwaway DB."""
    import heartbeat

    pdir = os.path.join(tmp, "proposals")
    mdir = os.path.join(tmp, "memories")
    os.makedirs(pdir)
    os.makedirs(mdir)
    for name in proposals:
        with open(os.path.join(pdir, name), "w", encoding="utf-8") as fh:
            fh.write("---\nname: x\n---\n")
    for name, obs_id in memories:
        with open(os.path.join(mdir, name), "w", encoding="utf-8") as fh:
            fh.write(f"---\nname: x\nmetadata:\n  source_obs_id: {obs_id}\n---\nbody\n")

    db = os.path.join(tmp, "t.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE observations (id TEXT PRIMARY KEY, evidence TEXT)")
    for obs_id, ev in evidence_by_id.items():
        con.execute("INSERT INTO observations VALUES (?, ?)", (obs_id, ev))
    con.commit()
    con.close()

    heartbeat.PROPOSAL_DIR = pdir
    heartbeat.CLAUDE_MEM_DIR = mdir
    return heartbeat, type("Cfg", (), {"db": db})()


def test_proposals_check_is_silent_when_the_queue_is_clear():
    """The healthy state. If this ever fails the check is pure noise."""
    with tempfile.TemporaryDirectory() as tmp:
        hb, cfg = _proposals_env(
            tmp, [], [("a.md", "OBS1")],
            {"OBS1": json.dumps(["exported to claude-code memory: a.md"])})
        status, msg = hb.check_proposals(cfg)
        assert status == hb.OK, msg


def test_proposals_check_fires_on_a_pending_queue():
    """NEGATIVE CONTROL for the pending half. A check that cannot fail is not
    a check -- the whole reason this exists is that an unreviewed queue looks
    exactly like a healthy one from the outside."""
    with tempfile.TemporaryDirectory() as tmp:
        hb, cfg = _proposals_env(tmp, ["p1.md", "p2.md"], [], {})
        status, msg = hb.check_proposals(cfg)
        assert status == hb.FAIL, msg
        assert "2 proposal" in msg


def test_proposals_check_fires_on_a_promoted_but_unmarked_row():
    """NEGATIVE CONTROL for the duplicate half.

    A promoted memory whose spine row was never stamped exported gets proposed
    again forever, and the 4h sync returns the promoted file as a second
    observation while the original still stands. Nothing errors, so nothing
    surfaces it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        hb, cfg = _proposals_env(
            tmp, [], [("promoted.md", "OBS2")],
            {"OBS2": json.dumps(["hot core import"])})   # no export stamp
        status, msg = hb.check_proposals(cfg)
        assert status == hb.FAIL, msg
        assert "promoted.md" in msg


def test_proposals_check_ignores_a_memory_whose_source_row_is_gone():
    """A row deleted or moved to another profile is not this check's problem.
    Failing on it would make the check permanently red for reasons the operator
    cannot act on, which is how guards stop being read."""
    with tempfile.TemporaryDirectory() as tmp:
        hb, cfg = _proposals_env(tmp, [], [("orphan.md", "MISSING")], {})
        status, msg = hb.check_proposals(cfg)
        assert status == hb.OK, msg


def test_proposals_check_skips_when_there_is_no_queue():
    """Missing evidence is a SKIP, never a FAIL. Silent when blind."""
    with tempfile.TemporaryDirectory() as tmp:
        hb, cfg = _proposals_env(tmp, [], [], {})
        hb.PROPOSAL_DIR = os.path.join(tmp, "does-not-exist")
        status, _ = hb.check_proposals(cfg)
        assert status == hb.SKIP


# ── 2026-08-30: quadratic decay + database-is-locked ─────────────────

def _fresh_index(tmp_path):
    from spine.index import MemoryIndex
    idx = MemoryIndex(str(tmp_path / "m.db"))
    idx.open()
    idx._canonical_root = ""  # no JSONL write-back in tests
    return idx


def _seed(idx, obs_id, conf, last_confirmed, status="active", typ="fact"):
    idx.conn.execute(
        "INSERT INTO observations (id, profile, type, content, confidence, status, "
        "created_at, last_confirmed) VALUES (?, 'agent:main', ?, ?, ?, ?, ?, ?)",
        (obs_id, typ, f"content {obs_id}", conf, status, last_confirmed, last_confirmed))
    idx.conn.commit()


def test_decay_is_linear_in_elapsed_time_not_in_run_count(tmp_path):
    """The old pass charged days_idle/30*rate on EVERY run: a 0.9 row idle 40
    days lost ~0.067 per night and hit the 0.3 archive floor in weeks. The
    charge must depend only on wall-clock elapsed since the last run."""
    from datetime import datetime, timedelta, timezone
    from spine.config import SpineConfig
    idx = _fresh_index(tmp_path)
    cfg = SpineConfig()  # decay_per_30d=0.05, archive_threshold=0.3
    t0 = datetime(2026, 8, 1, tzinfo=timezone.utc)
    _seed(idx, "old", 0.9, (t0 - timedelta(days=40)).isoformat())

    # First run: no stamp yet -> records the clock, charges nothing.
    assert loops._decay_pass(idx, cfg, "agent:main", now=t0) == (0, 0)
    assert idx.conn.execute("SELECT confidence FROM observations WHERE id='old'").fetchone()[0] == 0.9

    # Ten "nightly" runs across 10 days: total charge is 10/30*0.05, not
    # sum(days_idle/30*0.05) over 10 nights (which would be > 0.75).
    for n in range(1, 11):
        loops._decay_pass(idx, cfg, "agent:main", now=t0 + timedelta(days=n))
    conf, status = idx.conn.execute(
        "SELECT confidence, status FROM observations WHERE id='old'").fetchone()
    assert abs(conf - (0.9 - 10 / 30 * 0.05)) < 1e-3, conf
    assert status == "active"

    # Re-running with no time elapsed is a no-op.
    loops._decay_pass(idx, cfg, "agent:main", now=t0 + timedelta(days=10))
    assert idx.conn.execute("SELECT confidence FROM observations WHERE id='old'").fetchone()[0] == conf


def test_decay_catches_up_a_missed_night_and_respects_fresh_confirmation(tmp_path):
    from datetime import datetime, timedelta, timezone
    from spine.config import SpineConfig
    idx = _fresh_index(tmp_path)
    cfg = SpineConfig()
    t0 = datetime(2026, 8, 1, tzinfo=timezone.utc)
    _seed(idx, "idle", 0.5, (t0 - timedelta(days=5)).isoformat())
    loops._decay_pass(idx, cfg, "agent:main", now=t0)
    # Confirmed 2 days into a 6-day gap: charged only for the 4 days since.
    _seed(idx, "fresh", 0.5, (t0 + timedelta(days=2)).isoformat())
    loops._decay_pass(idx, cfg, "agent:main", now=t0 + timedelta(days=6))
    idle = idx.conn.execute("SELECT confidence FROM observations WHERE id='idle'").fetchone()[0]
    fresh = idx.conn.execute("SELECT confidence FROM observations WHERE id='fresh'").fetchone()[0]
    assert abs(idle - (0.5 - 6 / 30 * 0.05)) < 1e-3, idle
    assert abs(fresh - (0.5 - 4 / 30 * 0.05)) < 1e-3, fresh


def test_decay_still_archives_below_threshold_and_spares_corrections(tmp_path):
    from datetime import datetime, timedelta, timezone
    from spine.config import SpineConfig
    idx = _fresh_index(tmp_path)
    cfg = SpineConfig()
    t0 = datetime(2026, 8, 1, tzinfo=timezone.utc)
    _seed(idx, "weak", 0.31, (t0 - timedelta(days=1)).isoformat())
    _seed(idx, "corr", 0.31, (t0 - timedelta(days=1)).isoformat(), typ="correction")
    loops._decay_pass(idx, cfg, "agent:main", now=t0)
    decayed, archived = loops._decay_pass(idx, cfg, "agent:main", now=t0 + timedelta(days=30))
    assert (decayed, archived) == (1, 1)
    rows = dict(idx.conn.execute("SELECT id, status FROM observations").fetchall())
    assert rows == {"weak": "archived", "corr": "active"}


def test_open_sets_a_real_busy_timeout(tmp_path):
    """sqlite3.connect() defaults to 5s; the 4-hourly Claude Code sync died
    with 'database is locked' at its first write on 2026-08-30 because a
    sibling writer held the WAL lock longer than that."""
    from spine.index import DB_BUSY_TIMEOUT_S
    idx = _fresh_index(tmp_path)
    assert idx.conn.execute("PRAGMA busy_timeout").fetchone()[0] == int(DB_BUSY_TIMEOUT_S * 1000)
    assert idx.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_open_on_an_existing_store_does_not_wait_for_the_write_lock(tmp_path):
    """The dim_meta stamp was INSERT OR IGNORE on every open(), which takes
    the write lock even when the row already exists -- so a read-only recall
    queued behind whichever writer was active. Read first, insert only when
    missing."""
    import sqlite3 as _sq
    from spine.index import MemoryIndex
    db = str(tmp_path / "m.db")
    boot = MemoryIndex(db)
    boot.open()  # creates schema + stamp
    boot.close()
    holder = _sq.connect(db, timeout=1)
    holder.execute("BEGIN IMMEDIATE")  # take and hold the write lock
    try:
        t = time.monotonic()
        idx = MemoryIndex(db)
        idx.open()
        elapsed = time.monotonic() - t
        idx.close()
    finally:
        holder.rollback()
        holder.close()
    assert elapsed < 1.0, f"open() blocked on the write lock for {elapsed:.1f}s"


def test_open_waits_out_a_short_writer_collision_instead_of_failing(tmp_path):
    """A sibling holding the write lock for longer than the old 5s default no
    longer crashes the opener; it waits and proceeds."""
    import sqlite3 as _sq
    from spine.index import MemoryIndex
    db = str(tmp_path / "m.db")
    boot = MemoryIndex(db)
    boot.open()
    boot.close()
    locked = threading.Event()
    released = threading.Event()

    def _hold_write_lock():
        # sqlite3 connections are thread-bound: the holder lives here entirely.
        holder = _sq.connect(db, timeout=1)
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("INSERT INTO dim_meta (key, value) VALUES ('probe', 'x')")
        locked.set()
        time.sleep(1.5)
        holder.commit()
        holder.close()
        released.set()
    threading.Thread(target=_hold_write_lock, daemon=True).start()
    assert locked.wait(5)

    idx = MemoryIndex(db)
    idx.open()
    t = time.monotonic()
    # A real write from the opener: must block until the holder commits.
    idx.conn.execute("INSERT INTO dim_meta (key, value) VALUES ('probe2', 'y')")
    idx.conn.commit()
    assert time.monotonic() - t >= 1.0
    assert released.is_set()
    idx.close()


def test_dim_probe_waits_out_a_writer_instead_of_caching_a_miss(tmp_path, monkeypatch):
    """embedding_dim()'s store probe used a bare connect with timeout=1.0.
    Under a writer holding the DB longer than that it swallowed the BUSY,
    cached the miss, and let the embedder answer -- so the first schema
    create to see an empty dim_meta could stamp the model's width over the
    store's. Review round 2 (2026-08-30) routed it through connect_db."""
    import sqlite3 as _sq
    from spine import index as _index
    from spine.config import SpineConfig
    db = str(tmp_path / "m.db")
    # Raw rollback-journal store (no WAL): BEGIN EXCLUSIVE makes a mode=ro
    # reader see SQLITE_BUSY, which is the only way to exercise the timeout.
    boot = _sq.connect(db)
    boot.execute("CREATE TABLE dim_meta (key TEXT PRIMARY KEY, value TEXT)")
    # A width no embedder produces, so an accidental embedder answer cannot pass.
    boot.execute("INSERT INTO dim_meta VALUES ('embedding_dim', '1234')")
    boot.commit()
    boot.close()

    cfg = SpineConfig()
    cfg.db = db
    monkeypatch.setattr("spine.config.load_spine_config", lambda *a, **k: cfg)
    monkeypatch.setattr(_index, "_embedding_dim", None)
    monkeypatch.setattr(_index, "_dim_probe_failed", False)

    locked = threading.Event()

    def _hold_exclusive():
        holder = _sq.connect(db, timeout=1)
        holder.execute("BEGIN EXCLUSIVE")
        locked.set()
        time.sleep(1.5)
        holder.rollback()
        holder.close()
    threading.Thread(target=_hold_exclusive, daemon=True).start()
    assert locked.wait(5)

    t = time.monotonic()
    width = _index.embedding_dim()
    elapsed = time.monotonic() - t
    assert width == 1234, width
    assert elapsed >= 1.0, f"probe gave up after {elapsed:.2f}s instead of waiting"
    assert _index._dim_probe_failed is False


# ── hot-core promote dedupe ───────────────────────────────────────────

class _StubEmbedder:
    """Deterministic stand-in so these tests never load a 768-dim model.

    Vectors are hand-placed so the cosines are exact and readable:
    identical text -> 1.0, the reword -> ~0.98, the distinct fact -> 0.0.
    """
    def __init__(self, available=True):
        self._available = available
        self.vectors = {}

    def embedder_available(self):
        return self._available

    def embed(self, texts):
        return [self.vectors.get(t, [0.0, 0.0, 1.0]) for t in texts]

    def embed_single(self, text):
        return self.vectors.get(text, [0.0, 0.0, 1.0])


def _with_stub(monkeypatch, stub):
    import spine.embedder as real
    for name in ("embedder_available", "embed", "embed_single"):
        monkeypatch.setattr(real, name, getattr(stub, name))


def test_promote_dedupe_catches_a_reword(monkeypatch):
    """NEGATIVE CONTROL for the whole point of the change.

    Exact-body match structurally cannot catch this, and it is the failure that
    actually happened: 53 rows restating the brain-updater workflow, 31 the
    macOS username, 9 the timezone, each worded differently, each clearing the
    0.9 gate, each promoting into the always-loaded file.
    """
    stub = _StubEmbedder()
    existing = "User's macOS account username is exampleuser."
    reword = "The user's system username on macOS is exampleuser."
    stub.vectors[existing] = [1.0, 0.0, 0.0]
    stub.vectors[reword] = [0.98, 0.2, 0.0]      # cos ~= 0.980
    _with_stub(monkeypatch, stub)

    assert loops._hotcore_semantic_duplicate(reword, [existing]) == existing


def test_promote_dedupe_does_not_flag_a_distinct_fact(monkeypatch):
    """The expensive error. A false positive keeps a genuinely new fact OUT of
    the hot core and nothing downstream would ever report that it happened."""
    stub = _StubEmbedder()
    existing = "User's macOS account username is exampleuser."
    novel = "Quarterly VAT filing in Singapore is due on the last day of the month."
    stub.vectors[existing] = [1.0, 0.0, 0.0]
    stub.vectors[novel] = [0.0, 1.0, 0.0]        # cos = 0.0
    _with_stub(monkeypatch, stub)

    assert loops._hotcore_semantic_duplicate(novel, [existing]) is None


def test_promote_dedupe_respects_the_threshold_boundary(monkeypatch):
    """Just below HOTCORE_DUP_COSINE must pass through, just above must not.
    Pins the calibrated number so a future edit to it fails loudly."""
    stub = _StubEmbedder()
    existing = "block text"
    stub.vectors[existing] = [1.0, 0.0, 0.0]

    import math
    # No digits in the candidate text: _disagrees_on_specifics treats a number
    # the existing block lacks as proof of a sibling fact, which would mask the
    # cosine behaviour this test is pinning.
    for label, delta, expect_dup in (("above", +0.02, True), ("below", -0.05, False)):
        target = loops.HOTCORE_DUP_COSINE + delta
        cand = f"candidate {label}"
        # a vector whose cosine with [1,0,0] is exactly `target`
        stub.vectors[cand] = [target, math.sqrt(max(0.0, 1 - target ** 2)), 0.0]
        _with_stub(monkeypatch, stub)
        got = loops._hotcore_semantic_duplicate(cand, [existing]) is not None
        assert got is expect_dup, f"cos={target:.3f} expected dup={expect_dup}"


def test_promote_dedupe_degrades_to_exact_match_without_an_embedder(monkeypatch):
    """Silent-when-blind, applied to a write path.

    A duplicate block is a cost. A promote pass that stops promoting because
    the embedder died is an outage, and spine has shipped exactly that failure
    before (the embedder was silently dead for 8 days). Returning None keeps
    exact-match dedupe working and lets promotion continue.
    """
    stub = _StubEmbedder(available=False)
    _with_stub(monkeypatch, stub)
    assert loops._hotcore_semantic_duplicate("anything", ["a block"]) is None


def test_promote_dedupe_survives_a_broken_embedder(monkeypatch):
    """A crash inside the dedupe must not take the promote pass down."""
    import spine.embedder as real
    monkeypatch.setattr(real, "embedder_available", lambda: True)
    def _boom(*a, **k):
        raise RuntimeError("embedder exploded")
    monkeypatch.setattr(real, "embed", _boom)
    monkeypatch.setattr(real, "embed_single", _boom)
    assert loops._hotcore_semantic_duplicate("x", ["y"]) is None


def test_promote_dedupe_handles_an_empty_hot_core(monkeypatch):
    """First promotion into an empty file must not be called a duplicate."""
    stub = _StubEmbedder()
    _with_stub(monkeypatch, stub)
    assert loops._hotcore_semantic_duplicate("first ever block", []) is None


def test_promote_dedupe_keeps_sibling_facts_that_differ_by_a_specific():
    """NEGATIVE CONTROL for the second signal, and the reason it exists.

    Measured on the live embedder, "peak billing is 09:00-12:00 SGT" and
    "peak billing is 14:00-18:00 SGT" have cosine 0.974. Both are real halves
    of one block in the owner's hot core. A cosine-only check merges them and
    silently deletes a billing window, so a high cosine is necessary but not
    sufficient: differing exact values mean sibling facts, not a rewording.

    No embedder needed. This pins the deterministic half on its own.
    """
    assert loops._disagrees_on_specifics(
        "peak billing is 14:00-18:00 SGT", "peak billing is 09:00-12:00 SGT")
    assert loops._disagrees_on_specifics("entry 1", "entry 0")
    assert loops._disagrees_on_specifics(
        "Hermes venv Python is 3.9.6.", "Hermes venv Python is 3.11.15.")
    assert loops._disagrees_on_specifics(
        "set DEEPSEEK_API_KEY", "set ANTHROPIC_API_KEY")
    assert loops._disagrees_on_specifics(
        "config lives at ~/.hermes/config.yaml", "config lives at ~/.claude/settings.json")


def test_promote_dedupe_allows_a_reword_carrying_the_same_specifics():
    """The other side of the same guard: identical exact values must NOT block
    a genuine rewording, or the second signal would disable the first."""
    assert not loops._disagrees_on_specifics(
        "The user's system username on macOS is exampleuser.",
        "User's macOS account username is exampleuser.")
    assert not loops._disagrees_on_specifics("plain prose", "other plain prose")


def test_promote_to_hotcore_actually_calls_the_dedupe(monkeypatch):
    """INTEGRATION, and the one a mutation test caught missing.

    Every other dedupe test here exercises _hotcore_semantic_duplicate in
    isolation. Stubbing the call out of _promote_to_hotcore entirely still left
    all of them green, which means they proved the helper works and proved
    NOTHING about whether the promote path uses it. That is the tautology shape
    that has bitten this suite before.

    This asserts the wiring: a reword offered to _promote_to_hotcore comes back
    "duplicate" and is NOT appended to the file.
    """
    f = tempfile.mktemp(suffix=".md")
    existing = "User's macOS account username is exampleuser."
    _hotcore(f, f"[F] {existing}\n")

    stub = _StubEmbedder()
    reword = "The user's system username on macOS is exampleuser."
    stub.vectors[existing] = [1.0, 0.0, 0.0]
    stub.vectors[reword] = [0.98, 0.2, 0.0]
    _with_stub(monkeypatch, stub)

    before = open(f, encoding="utf-8").read()
    assert loops._promote_to_hotcore("obs1", reword, "fact", None) == "duplicate"
    assert open(f, encoding="utf-8").read() == before, "reword was appended anyway"

    # and a genuinely new fact still gets through the same path
    novel = "Quarterly VAT filing in Singapore is due on the last day of the month."
    stub.vectors[novel] = [0.0, 1.0, 0.0]
    assert loops._promote_to_hotcore("obs2", novel, "fact", None) == "written"
    assert novel in open(f, encoding="utf-8").read()
    os.remove(f)
