"""Tests for the synthetic miscue injector (T1.3a).

Written test-first.  Everything here is pure + seeded + CI-safe: no audio, no
TTS, no network.  The load-bearing property is the TEXT-LEVEL ROUND-TRIP:
feeding the *miscued text* (what a reader would have said) back into the
detector recovers the gold labels we planted — the audio-free upper bound the
benchmark is built against.

EMPIRICAL TEXT-LEVEL DETECTION FACTS (jiwer 4.0.0, probed 2026-06-10)
--------------------------------------------------------------------
The injector and these tests assume the following about ``miscue.detect`` run
on PLAIN TEXT (timing rule inactive).  Probed directly against the detector:

  * substitution / omission / insertion  -> recovered exactly (class + index,
    and said/target words for sub/ins).  This is the named round-trip property.
  * self_correction  -> ALSO recovered from text.  A wrong attempt that is an
    orthographic near-miss (SequenceMatcher ratio >= 0.5) of the next target
    word, read immediately before the correct word, aligns as
    ``insert(wrong) + equal(correct)`` (variant a) and is classified
    self_correction at the target index, said_word = the wrong attempt.
  * hesitation, render="filler"  -> recovered from text.  The filler token
    ("um"/"uh") is stripped pre-alignment and re-emitted as a hesitation bound
    to the next surviving target index, said_word = the filler.
  * hesitation, render="silence"  -> INVISIBLE at text level.  The silence is a
    TTS-only ``[[slnc ...]]`` marker that never appears in the spoken/miscued
    text, so text-level detection yields nothing for it (it is an audio-timing
    signal only, recoverable by a later timed-mode ticket).
"""
from __future__ import annotations

import random

import pytest

from readcoach.inject import (
    InjectedItem,
    MiscueSpec,
    Passage,
    inject,
    load_passages,
    plan_benchmark,
    validate_coverage,
)
from readcoach.miscue import Miscue, detect

PASSAGE_DIR = "data/passages"
SEED = 42

# Text-recoverable classes: detect() on the miscued *text* must recover these.
# (silence-hesitations are deliberately excluded — audio-only.)
_TEXT_RECOVERABLE = {"substitution", "omission", "insertion", "self_correction"}


# ===========================================================================
# Passage loading
# ===========================================================================

def test_load_passages_count_and_schema():
    passages = load_passages(PASSAGE_DIR)
    assert len(passages) == 8
    ids = [p.id for p in passages]
    assert ids == sorted(ids), "passages must be returned in stable id order"
    assert ids == [f"p0{i}" for i in range(1, 9)]
    for p in passages:
        assert isinstance(p, Passage)
        assert p.id and p.phonics_focus and p.text
        assert p.band in {1, 2, 3, 4}
        assert p.words == tuple(p.text.split())


def test_load_passages_bands_two_each():
    passages = load_passages(PASSAGE_DIR)
    counts: dict[int, int] = {}
    for p in passages:
        counts[p.band] = counts.get(p.band, 0) + 1
    assert counts == {1: 2, 2: 2, 3: 2, 4: 2}


def test_load_passages_word_counts_in_range():
    for p in load_passages(PASSAGE_DIR):
        assert 25 <= len(p.words) <= 45, f"{p.id} has {len(p.words)} words"


def test_passage_is_frozen():
    p = load_passages(PASSAGE_DIR)[0]
    with pytest.raises(Exception):
        p.band = 99  # type: ignore[misc]


# ===========================================================================
# Per-class injection — gold is intended AND detector constraints hold
# ===========================================================================

def _passage() -> Passage:
    # p01 is a CVC passage with simple, distinct words — a clean injection target.
    return load_passages(PASSAGE_DIR)[0]


def test_inject_substitution():
    p = _passage()
    rng = random.Random(1)
    item = inject(p, [MiscueSpec(type="substitution", index=1)], rng)
    assert len(item.gold) == 1
    g = item.gold[0]
    assert g.type == "substitution" and g.index == 1
    assert g.target_word == p.words[1].strip(".,").casefold()
    # The substitution must actually differ from the target (detector requirement).
    assert g.said_word != g.target_word
    # Round-trips through the detector at text level.
    assert detect(item.miscued_text, item.target_text) == item.gold


def test_inject_omission():
    p = _passage()
    item = inject(p, [MiscueSpec(type="omission", index=2)], random.Random(1))
    assert [(m.type, m.index) for m in item.gold] == [("omission", 2)]
    # The omitted word is gone from the miscued text (one fewer word).
    assert len(item.miscued_text.split()) == len(p.words) - 1
    assert detect(item.miscued_text, item.target_text) == item.gold


def test_inject_insertion_constraints():
    p = _passage()
    item = inject(p, [MiscueSpec(type="insertion", index=2)], random.Random(1))
    assert len(item.gold) == 1 and item.gold[0].type == "insertion"
    said = item.gold[0].said_word
    assert said is not None
    # Constraint (a): inserted token != preceding target word.
    prev_target = p.words[item.gold[0].index - 1].strip(".,").casefold()
    assert said != prev_target
    # Constraint (b): inserted token not >= 0.5 similar to FOLLOWING target word.
    import difflib
    following = p.words[item.gold[0].index].strip(".,").casefold()
    assert difflib.SequenceMatcher(None, said, following).ratio() < 0.5
    # Constraint (c): inserted token is not a detector filler.
    from readcoach.miscue import _FILLERS
    assert said not in _FILLERS
    assert detect(item.miscued_text, item.target_text) == item.gold


def test_inject_self_correction_similarity_gate():
    import difflib
    p = _passage()
    item = inject(p, [MiscueSpec(type="self_correction", index=1)], random.Random(1))
    assert len(item.gold) == 1 and item.gold[0].type == "self_correction"
    said = item.gold[0].said_word
    target = item.gold[0].target_word
    assert said is not None and target is not None
    # The published self_correction gate: wrong attempt is >=0.5 similar to target,
    # != target, and != the preceding target word.
    assert difflib.SequenceMatcher(None, said, target).ratio() >= 0.5
    assert said != target
    if item.gold[0].index > 0:
        prev = p.words[item.gold[0].index - 1].strip(".,").casefold()
        assert said != prev
    assert detect(item.miscued_text, item.target_text) == item.gold


def test_inject_hesitation_filler_recoverable_from_text():
    p = _passage()
    item = inject(
        p, [MiscueSpec(type="hesitation", index=3, render="filler")], random.Random(1)
    )
    assert len(item.gold) == 1 and item.gold[0].type == "hesitation"
    # filler render: the filler appears in BOTH miscued_text and tts_text.
    assert item.miscued_text == item.tts_text
    # Recoverable from text.
    assert detect(item.miscued_text, item.target_text) == item.gold


def test_inject_hesitation_silence_invisible_at_text():
    p = _passage()
    item = inject(
        p, [MiscueSpec(type="hesitation", index=3, render="silence")], random.Random(1)
    )
    assert len(item.gold) == 1 and item.gold[0].type == "hesitation"
    # The silence marker is in tts_text only; the spoken/miscued text is unchanged.
    assert "[[slnc" in item.tts_text
    assert "[[slnc" not in item.miscued_text
    assert item.miscued_text == item.target_text
    # Text-level detection sees NOTHING (audio-timing signal only).
    assert detect(item.miscued_text, item.target_text) == []


def test_clean_item_has_no_miscues():
    p = _passage()
    item = inject(p, [], random.Random(1))
    assert item.gold == []
    assert item.miscued_text == item.target_text == item.tts_text
    assert detect(item.miscued_text, item.target_text) == []


def test_injected_item_gold_render_length_mismatch_raises():
    """Constructing an InjectedItem with mismatched gold/gold_render lengths
    must raise ValueError at construction time."""
    from readcoach.miscue import Miscue

    gold = [Miscue("substitution", target_word="cat", said_word="bat", index=0)]
    # gold_render is empty while gold has 1 item — should raise.
    with pytest.raises(ValueError, match="gold_render length.*must equal.*gold length"):
        InjectedItem(
            utt_id="test-mismatch",
            passage_id="p01",
            target_text="cat sat mat",
            miscued_text="bat sat mat",
            tts_text="bat sat mat",
            gold=gold,
            gold_render=[],  # Mismatch: gold has 1, gold_render has 0.
        )


# ===========================================================================
# THE ROUND-TRIP PROPERTY (the plan's named test)
# ===========================================================================

def _text_recoverable_gold(item: InjectedItem) -> list[Miscue]:
    """Gold miscues that text-level detection is expected to recover.

    Drops silence-render hesitations (audio-only).  Filler hesitations and
    self_corrections ARE kept (both empirically recover from text).
    """
    out: list[Miscue] = []
    for g, render in zip(item.gold, item.gold_render, strict=True):
        if g.type == "hesitation" and render == "silence":
            continue
        out.append(g)
    return sorted(out, key=lambda m: (m.index, m.type))


def test_round_trip_every_planned_item():
    items = plan_benchmark(load_passages(PASSAGE_DIR), seed=SEED)
    assert items, "plan must be non-empty"
    for item in items:
        predicted = detect(item.miscued_text, item.target_text)
        expected = _text_recoverable_gold(item)
        pred_sorted = sorted(predicted, key=lambda m: (m.index, m.type))
        # Class + index match, said/target words match for the text-recoverable set.
        assert pred_sorted == expected, (
            f"round-trip mismatch for {item.utt_id}\n"
            f"  predicted={pred_sorted}\n  expected ={expected}\n"
            f"  miscued  ={item.miscued_text!r}\n  target   ={item.target_text!r}"
        )


def test_silence_hesitations_absent_from_text_detection():
    """Explicit: a planned item whose ONLY miscue is a silence hesitation must
    yield nothing at the text level (and the round-trip filters it out)."""
    p = _passage()
    item = inject(
        p, [MiscueSpec(type="hesitation", index=5, render="silence")], random.Random(7)
    )
    assert detect(item.miscued_text, item.target_text) == []
    assert _text_recoverable_gold(item) == []


# ===========================================================================
# Coverage matrix — enforced in code, no bypass
# ===========================================================================

_ALL_CLASSES = (
    "substitution",
    "omission",
    "insertion",
    "self_correction",
    "hesitation",
)


def test_full_plan_passes_coverage():
    items = plan_benchmark(load_passages(PASSAGE_DIR), seed=SEED)
    matrix = validate_coverage(items)
    # Every (class, passage) cell is populated.
    for pid in [f"p0{i}" for i in range(1, 9)]:
        for cls in _ALL_CLASSES:
            assert matrix[pid][cls] >= 1, f"{pid} missing {cls}"


def test_coverage_raises_when_class_missing_for_a_passage():
    items = plan_benchmark(load_passages(PASSAGE_DIR), seed=SEED)
    # Remove every item that contributes a 'self_correction' to passage p03.
    target_pid = "p03"
    pruned = [
        it
        for it in items
        if not (
            it.passage_id == target_pid
            and any(g.type == "self_correction" for g in it.gold)
        )
    ]
    with pytest.raises(Exception):
        validate_coverage(pruned)


def test_plan_benchmark_targets_item_count_and_clean_per_passage():
    items = plan_benchmark(load_passages(PASSAGE_DIR), seed=SEED)
    # ~88 items: 8 passages * (1 clean + ~10 variants).
    assert 80 <= len(items) <= 96, f"got {len(items)} items"
    # Exactly one clean (no-miscue) item per passage.
    for pid in [f"p0{i}" for i in range(1, 9)]:
        cleans = [it for it in items if it.passage_id == pid and not it.gold]
        assert len(cleans) == 1, f"{pid} has {len(cleans)} clean items"


# ===========================================================================
# Determinism + purity
# ===========================================================================

def test_plan_benchmark_deterministic():
    passages = load_passages(PASSAGE_DIR)
    a = plan_benchmark(passages, seed=SEED)
    b = plan_benchmark(passages, seed=SEED)
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.utt_id == y.utt_id
        assert x.miscued_text == y.miscued_text
        assert x.tts_text == y.tts_text
        assert x.gold == y.gold


def test_plan_benchmark_different_seed_differs():
    passages = load_passages(PASSAGE_DIR)
    a = plan_benchmark(passages, seed=1)
    b = plan_benchmark(passages, seed=2)
    # Same structure (utt_ids are structural), but the injected word choices /
    # variant composition should differ somewhere.
    assert [x.miscued_text for x in a] != [x.miscued_text for x in b]


def test_injection_is_pure_no_global_random():
    """Two interleaved generations with independent rngs must not perturb each
    other — i.e. inject() uses only its passed-in rng, never the global module."""
    p = _passage()
    spec = [MiscueSpec(type="substitution", index=2)]

    # Baseline: a fresh rng(123) used alone.
    solo = inject(p, spec, random.Random(123))

    # Now interleave: seed the GLOBAL random heavily between constructing the rng
    # and calling inject.  If inject touched the global, output would shift.
    random.seed(999)
    rng = random.Random(123)
    random.random()  # perturb global state
    random.random()
    interleaved = inject(p, spec, rng)

    assert solo.miscued_text == interleaved.miscued_text
    assert solo.gold == interleaved.gold
