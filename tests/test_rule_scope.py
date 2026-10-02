"""Tests for hot-core rule scoping (hotcore-retrieval-split, phase 2).

The failure that matters here is silent: a behavioural rule that stops reaching
the model makes the agent misbehave with nobody able to say why. So the tests
lean on the safe-default and never-defer guarantees, not just the happy path.
"""
import os
import sys

# Repo layout here is `<root>/spine/` + `<root>/tests/` (this file), one level
# shallower than the original plugins/memory/spine/tests/ checkout path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                          # for `from spine.rule_scope import ...`
sys.path.insert(0, os.path.join(_ROOT, "spine"))   # flat imports of spine/ scripts

from spine.rule_scope import (  # noqa: E402
    IMAGE, SUBJECT, TOOL, UNIVERSAL, TurnContext, fires, parse_trigger,
    reachable_kinds, select, split_blocks, summarise,
)


# ── parse_trigger: unclassified and malformed must both be universal ──

def test_no_marker_is_universal():
    assert parse_trigger("[C] Execute, do not hand homework.").kind == UNIVERSAL


def test_unknown_kind_falls_back_to_universal():
    # A typo must not silently delete a rule from every prompt.
    assert parse_trigger("[W] thing @when:planetary=mars").kind == UNIVERSAL


def test_valueless_subject_is_universal():
    # "@when:subject" with nothing to match could never fire.
    assert parse_trigger("[W] thing @when:subject").kind == UNIVERSAL


def test_parses_kinds_and_values():
    t = parse_trigger("[W] ledger note @when:subject=ledger,prediction")
    assert t.kind == SUBJECT and t.values == ("ledger", "prediction")
    assert parse_trigger("[W] garmin @when:tool=garmin_*").kind == TOOL
    assert parse_trigger("[C] screenshot rule @when:image").kind == IMAGE


# ── fires: triggers are facts, not guesses ──

def test_image_trigger_matches_payload_only():
    t = parse_trigger("[C] screenshot @when:image")
    assert fires(t, TurnContext(query="anything", has_image=True))
    assert not fires(t, TurnContext(query="anything", has_image=False))


def test_tool_trigger_matches_glob_and_nothing_else():
    t = parse_trigger("[W] garmin @when:tool=garmin_*")
    assert fires(t, TurnContext(tool_name="garmin_get_activities"))
    assert not fires(t, TurnContext(tool_name="web_search"))
    assert not fires(t, TurnContext())  # no tool this turn


def test_subject_trigger_respects_word_boundaries():
    t = parse_trigger("[W] ledger @when:subject=ledger")
    assert fires(t, TurnContext(query="check the ledger please"))
    assert fires(t, TurnContext(query="LEDGER grading"))
    # must not match inside another word
    assert not fires(t, TurnContext(query="the ledgerdemain of it"))
    assert not fires(t, TurnContext(query="unrelated question"))


def test_universal_always_fires_even_on_an_empty_turn():
    assert fires(parse_trigger("[C] always"), TurnContext())


# ── select: the two safety guarantees ──

def test_untagged_block_is_never_deferred():
    hot, deferred = select(["[C] no marker here"], TurnContext(query="x"))
    assert hot and not deferred


def test_rule_block_is_never_deferred_even_if_tagged():
    # [R] is the constitution. A mis-tagged rule must still load.
    blocks = ["[R] durable observations go to remember() @when:image"]
    hot, deferred = select(blocks, TurnContext(has_image=False))
    assert hot == blocks and deferred == []


def test_irrelevant_block_defers_and_relevant_one_does_not():
    blocks = [
        "[W] garmin restart @when:tool=garmin_*",
        "[C] screenshot rule @when:image",
        "[C] universal correction",
    ]
    hot, deferred = select(blocks, TurnContext(query="hi", tool_name="garmin_get"))
    assert blocks[0] in hot        # its tool is being called
    assert blocks[2] in hot        # universal
    assert blocks[1] in deferred   # no image this turn


def test_a_trigger_that_always_fires_is_not_a_trigger():
    # Negative control for the whole design: on a turn with no query, no tool
    # and no image, only universal/[R] content should survive.
    blocks = [
        "[W] a @when:subject=ledger",
        "[W] b @when:tool=garmin_*",
        "[C] c @when:image",
        "[R] d",
        "[C] e",
    ]
    hot, deferred = select(blocks, TurnContext())
    assert sorted(hot) == sorted(["[R] d", "[C] e"])
    assert len(deferred) == 3


# ── plumbing ──

def test_split_blocks_handles_the_section_separator():
    assert split_blocks("[C] one\n§\n[W] two\n§\n") == ["[C] one", "[W] two"]


def test_summarise_reports_byte_savings():
    blocks = ["[C] universal", "[W] x @when:image"]
    s = summarise(blocks, TurnContext(query="q", has_image=False))
    assert s["hot_blocks"] == 1 and s["deferred_blocks"] == 1
    assert s["deferred_bytes"] > 0
    assert s["deferred_triggers"] == ["image:"]


def test_reachable_kinds_covers_every_kind_fires_understands():
    for kind in reachable_kinds():
        assert kind in {UNIVERSAL, SUBJECT, TOOL, IMAGE}


# ── deliverable(): the snapshot + delivery must lose nothing ──

from spine.rule_scope import deliverable  # noqa: E402

SAMPLE = [
    "[R] constitution rule",
    "[C] universal correction",
    "[W] garmin note @when:tool=garmin_*",
    "[C] screenshot rule @when:image",
    "[W] ledger note @when:subject=ledger",
]


def _snapshot():
    """What the frozen system-prompt snapshot holds — must mirror the store."""
    from spine.rule_scope import snapshot_keep
    return snapshot_keep(SAMPLE)


def test_snapshot_holds_universal_rules_and_undeliverable_kinds():
    # Not just universal + [R]: an image block stays too, because nothing can
    # deliver it back yet. Tool blocks left on 2026-09-05 when the
    # hotcore-rules transform_tool_result hook landed.
    assert _snapshot() == [
        "[R] constitution rule",
        "[C] universal correction",
        "[C] screenshot rule @when:image",
    ]


def test_delivery_never_duplicates_the_snapshot():
    extra = deliverable(SAMPLE, TurnContext(query="ledger check"))
    assert not set(extra) & set(_snapshot())


def test_nothing_is_lost_for_any_turn():
    """snapshot + delivered must cover every block that fires. The whole design
    rests on this: a rule that fires but reaches neither is silently gone."""
    contexts = [
        TurnContext(),
        TurnContext(query="ledger check"),
        TurnContext(query="anything", has_image=True),
        TurnContext(tool_name="garmin_get_activities"),
        TurnContext(query="ledger", tool_name="garmin_x", has_image=True),
    ]
    for ctx in contexts:
        should_fire, _ = select(SAMPLE, ctx)
        delivered = set(_snapshot()) | set(deliverable(SAMPLE, ctx))
        missing = set(should_fire) - delivered
        assert not missing, f"lost {missing} for {ctx}"


def test_delivery_is_empty_when_nothing_extra_fires():
    assert deliverable(SAMPLE, TurnContext(query="totally unrelated")) == []


def test_rule_blocks_are_never_delivered_twice():
    # [R] is always in the snapshot, so it must never appear in the delta.
    extra = deliverable(SAMPLE, TurnContext(query="ledger", has_image=True))
    assert not any(b.startswith("[R]") for b in extra)


# ── the dangerous one: filtering the snapshot must never reach disk ──

def test_scope_filter_never_shrinks_what_gets_persisted(tmp_path, monkeypatch):
    """If a write path ever read the filtered snapshot, enabling scoping would
    delete every tagged rule from MEMORY.md. save_to_disk must always persist
    live entries, which the filter does not touch."""
    import tools.memory_tool as mt

    mem_dir = tmp_path / "memories"
    mem_dir.mkdir()
    blocks = [
        "[R] constitution rule",
        "[C] universal correction",
        "[W] ledger note @when:subject=ledger",
        "[C] screenshot rule @when:image",
    ]
    (mem_dir / "MEMORY.md").write_text("\n§\n".join(blocks), encoding="utf-8")
    (mem_dir / "USER.md").write_text("", encoding="utf-8")
    monkeypatch.setattr(mt, "get_memory_dir", lambda: mem_dir)

    store = mt.MemoryStore(memory_char_limit=90000, user_char_limit=1375,
                           memory_enabled=True, user_profile_enabled=True,
                           scope_filter=True)
    store.load_from_disk()

    # The snapshot IS filtered ...
    snap = store._system_prompt_snapshot["memory"]
    assert "ledger note" not in snap        # subject: deferred
    assert "screenshot rule" in snap        # image: no delivery path, stays

    # ... but live state is whole, and so is anything persisted from it.
    assert len(store.memory_entries) == len(blocks)
    store.save_to_disk("memory")
    on_disk = (mem_dir / "MEMORY.md").read_text(encoding="utf-8")
    for b in blocks:
        assert b in on_disk, f"scope filter destroyed {b!r} on disk"


def test_scope_filter_off_is_byte_identical_to_unfiltered(tmp_path, monkeypatch):
    """The flag must be a true no-op when off — the revert path depends on it."""
    import tools.memory_tool as mt

    mem_dir = tmp_path / "memories"
    mem_dir.mkdir()
    (mem_dir / "MEMORY.md").write_text(
        "[C] plain\n§\n[W] tagged @when:subject=ledger", encoding="utf-8")
    (mem_dir / "USER.md").write_text("", encoding="utf-8")
    monkeypatch.setattr(mt, "get_memory_dir", lambda: mem_dir)

    def build(flag):
        s = mt.MemoryStore(memory_char_limit=90000, user_char_limit=1375,
                           memory_enabled=True, user_profile_enabled=True,
                           scope_filter=flag)
        s.load_from_disk()
        return s._system_prompt_snapshot["memory"]

    assert "tagged" in build(False)
    assert "tagged" not in build(True)


# ── a block may only leave the snapshot if something can deliver it back ──

from spine.rule_scope import DELIVERABLE_KINDS, snapshot_keep  # noqa: E402


def test_undeliverable_kinds_stay_in_the_snapshot():
    """A block may only leave the snapshot if something can deliver it back.

    Image blocks still have no path: pre_llm_call is unwired, and
    compose_user_api_content drops the memory block on list-shaped content
    anyway. Filtering one out would delete the rule outright.
    """
    keep = snapshot_keep(SAMPLE)
    assert "[C] screenshot rule @when:image" in keep
    # Both of these have a delivery path now.
    assert "[W] ledger note @when:subject=ledger" not in keep   # prefetch()
    assert "[W] garmin note @when:tool=garmin_*" not in keep    # hotcore-rules hook


def test_only_deliverable_kinds_are_ever_deferred():
    from spine.rule_scope import parse_trigger
    deferred = [b for b in SAMPLE if b not in snapshot_keep(SAMPLE)]
    for b in deferred:
        assert parse_trigger(b).kind in DELIVERABLE_KINDS


def test_nothing_lost_including_undeliverable_kinds():
    for ctx in [TurnContext(),
                TurnContext(query="ledger check"),
                TurnContext(query="x", has_image=True),
                TurnContext(tool_name="garmin_get_activities"),
                TurnContext(query="ledger", tool_name="garmin_x", has_image=True)]:
        should_fire, _ = select(SAMPLE, ctx)
        reachable = set(snapshot_keep(SAMPLE)) | set(deliverable(SAMPLE, ctx))
        assert not (set(should_fire) - reachable), f"lost a rule for {ctx}"


# ── prefetch ordering: the two bugs that only showed up live ──

def test_prefetch_uses_the_current_query_not_a_cache(tmp_path, monkeypatch):
    """Two ordering bugs, both found on 2026-09-05 against the live gateway.

    queue_prefetch() runs AFTER a turn to prepare the next, so the first turn of
    a session has nothing cached — and one-shot runs (cron, hermes-run) are
    always a first turn, which silently lost every deferred rule. Caching also
    means turn 2 delivers the selection for turn 1's query.
    """
    import spine as spine_mod

    hot = tmp_path / "MEMORY.md"
    hot.write_text(
        "[C] universal\n§\n"
        "[W] ledger note @when:subject=ledger\n§\n"
        "[W] hip note @when:subject=hip",
        encoding="utf-8")
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(hot) if p.endswith("MEMORY.md") else p)
    monkeypatch.setattr(spine_mod, "_scope_filter_enabled", lambda: True)

    p = spine_mod.SpineProvider()
    p.initialize("t")

    # First turn, cold: must still deliver.
    first = p.prefetch("check the ledger")
    assert "ledger note" in first, "first turn of a session delivered nothing"

    # Next turn on a different subject must NOT replay the previous selection.
    second = p.prefetch("my hip hurts")
    assert "hip note" in second
    assert "ledger note" not in second, "delivered the previous turn's rules"

    # A turn matching nothing delivers nothing.
    assert p.prefetch("unrelated chatter") == ""


def test_prefetch_is_silent_when_the_flag_is_off(monkeypatch):
    import spine as spine_mod
    monkeypatch.setattr(spine_mod, "_scope_filter_enabled", lambda: False)
    p = spine_mod.SpineProvider()
    p.initialize("t")
    assert p.prefetch("check the ledger") == ""


# ── the gateway path: a non-string user message must still key recall ──

def test_query_text_for_recall_handles_every_message_shape():
    """A live gateway turn got no recall because original_user_message is a list
    of content parts there, and both memory hooks guarded on isinstance(str)
    and fell back to "". Silently: prefetch_all returns "" for an empty query
    and the caller swallows exceptions."""
    from agent.turn_context import _memory_query_text as q

    assert q("check the ledger") == "check the ledger"
    assert q([{"type": "text", "text": "check the ledger"},
              {"type": "image_url", "image_url": {}}]) == "check the ledger"
    assert q(["a", "b"]) == "a\nb"
    # nothing to key a search on
    assert q(None) == ""
    assert q([{"type": "image_url", "image_url": {}}]) == ""


def test_multimodal_turn_still_selects_rules():
    """End of the chain: a list-shaped message must reach rule selection."""
    from agent.turn_context import _memory_query_text as q
    content = [{"type": "text", "text": "the ledger scorecard looks wrong"},
               {"type": "image_url", "image_url": {}}]
    hot, deferred = select(SAMPLE, TurnContext(query=q(content)))
    assert "[W] ledger note @when:subject=ledger" in hot


# ── the flag itself: read per turn, never frozen for the process ──

def test_flag_turned_on_mid_process_takes_effect(tmp_path, monkeypatch):
    """The defect that made three live tests read as "prefetch is broken".

    _scope_filter_enabled() used to memoise into a process-global set on the
    first prefetch of the process's life. Half 1 (MemoryStore) re-reads the
    flag for every agent the gateway builds, so turning the flag on against a
    RUNNING gateway filtered the snapshot while prefetch() kept returning ""
    from the stale False. The deferred rules were then in no prompt at all.
    """
    import hermes_cli.config as cfg_mod
    import spine as spine_mod

    hot = tmp_path / "MEMORY.md"
    hot.write_text(
        "[C] universal\n§\n[W] ledger note @when:subject=ledger",
        encoding="utf-8")
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(hot) if p.endswith("MEMORY.md") else p)

    flag = {"on": False}
    monkeypatch.setattr(cfg_mod, "load_config_readonly",
                        lambda: {"memory": {"scope_filter": flag["on"]}})

    p = spine_mod.SpineProvider()
    p.initialize("t")
    assert p.prefetch("check the ledger") == "", "delivered with the flag off"

    flag["on"] = True  # operator flips it on a gateway that is already up
    assert "ledger note" in p.prefetch("check the ledger"), (
        "prefetch froze the flag for the life of the process")


def test_flag_defaults_off_when_config_is_unreadable(monkeypatch):
    """Fails closed: no flag, no filtering, full hot core."""
    import hermes_cli.config as cfg_mod
    import spine as spine_mod

    def _boom():
        raise OSError("config unreadable")

    monkeypatch.setattr(cfg_mod, "load_config_readonly", _boom)
    assert spine_mod._scope_filter_enabled() is False
    monkeypatch.setattr(cfg_mod, "load_config_readonly", lambda: {})
    assert spine_mod._scope_filter_enabled() is False
