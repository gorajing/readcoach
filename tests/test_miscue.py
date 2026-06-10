"""Tests for the miscue detector (T1.2).

Written test-first.  Every Word list here is hand-built with synthetic timings —
no model, no audio, no network.  These tests are CI-safe.

EMPIRICAL JIWER FINDING (jiwer 4.0.0, probed 2026-06-10)
-------------------------------------------------------
jiwer's word-level Levenshtein backtrace, for a "wrong word then correct word"
self-correction, ALWAYS emits the ``insert(wrong) + equal(correct)`` shape
(variant *a*).  An exhaustive small-alphabet search (ref/hyp lengths 1..4) found
ZERO cases of the ``substitute(correct->wrong) + insert(correct)`` shape
(variant *b*) — both shapes cost one edit op, and jiwer's tie-break prefers
insert-then-equal.

The detector must nonetheless classify variant *b* as a self-correction, because
other alignment backends (and older jiwer) DO emit it.  We therefore test
variant *a* through real jiwer output (a string pair) and variant *b* by feeding
the classifier a hand-built ``AlignmentChunk`` stream (``_classify_alignment``),
which is the load-bearing internal seam.
"""
from __future__ import annotations

import math

import pytest

from readcoach.asr import AsrResult, Word
from readcoach.miscue import (
    GAP_THRESHOLD_S,
    Miscue,
    _classify_alignment,
    detect,
    score,
)


# ---------------------------------------------------------------------------
# Helpers — build a timed Word list from (text, start, end) triples.
# Whitespace/punctuation/casing deliberately mimic real Whisper tokens
# (leading space, attached punctuation) so the normalization path is exercised.
# ---------------------------------------------------------------------------

def words(*triples: tuple[str, float, float]) -> list[Word]:
    return [Word(text=t, start=s, end=e, confidence=0.9) for (t, s, e) in triples]


def evenly_timed(tokens: list[str], dur: float = 0.3, gap: float = 0.05) -> list[Word]:
    """Build a Word list with uniform durations and small inter-word gaps."""
    out: list[Word] = []
    t = 0.0
    for tok in tokens:
        out.append(Word(text=tok, start=t, end=t + dur, confidence=0.9))
        t += dur + gap
    return out


def types_at(miscues: list[Miscue]) -> list[tuple[str, int]]:
    return sorted((m.type, m.index) for m in miscues)


# ===========================================================================
# CLEAN READ  — the false-positive guard
# ===========================================================================

def test_clean_read_text_only_no_miscues():
    assert detect("the cat sat on the mat", "the cat sat on the mat") == []


def test_clean_read_timed_no_miscues():
    w = evenly_timed(["the", "cat", "sat", "on", "the", "mat"])
    assert detect(w, "the cat sat on the mat") == []


def test_clean_read_normalization_punctuation_and_casing():
    """Leading spaces, attached punctuation, and casing must NOT create miscues."""
    w = words(
        (" The", 0.0, 0.3),
        (" cat", 0.35, 0.6),
        (" sat", 0.65, 0.9),
        (" on", 0.95, 1.2),
        (" the", 1.25, 1.5),
        (" mat.", 1.55, 1.8),
    )
    assert detect(w, "The cat sat on the mat.") == []


def test_clean_read_asrresult_wrapper():
    """AsrResult input behaves like its .words list."""
    w = evenly_timed(["the", "cat", "sat"])
    res = AsrResult(text="the cat sat", words=w)
    assert detect(res, "the cat sat") == []


# ===========================================================================
# SINGLE-CLASS detection — substitution / omission / insertion
# ===========================================================================

def test_single_substitution():
    # target index 1 ("cat") read as "bat"
    m = detect("the bat sat", "the cat sat")
    assert len(m) == 1
    sub = m[0]
    assert sub.type == "substitution"
    assert sub.target_word == "cat"
    assert sub.said_word == "bat"
    assert sub.index == 1


def test_single_omission():
    # "cat" (target index 1) skipped
    m = detect("the sat on the mat", "the cat sat on the mat")
    assert len(m) == 1
    omit = m[0]
    assert omit.type == "omission"
    assert omit.target_word == "cat"
    assert omit.said_word is None
    assert omit.index == 1


def test_single_insertion_non_repeat():
    # extra non-repeat word "big" inserted before "cat"
    m = detect("the big cat sat", "the cat sat")
    assert len(m) == 1
    ins = m[0]
    assert ins.type == "insertion"
    assert ins.target_word is None
    assert ins.said_word == "big"
    # index = target position the insertion sits before (here target idx 1, "cat")
    assert ins.index == 1


# ===========================================================================
# SELF-CORRECTION — both jiwer alignment variants
# ===========================================================================

def test_self_correction_variant_a_insert_then_equal_real_jiwer():
    """Variant (a): jiwer emits insert('bat') + equal('cat').

    This is the shape jiwer 4.0 ACTUALLY produces for 'the bat cat sat' over
    target 'the cat sat' (verified empirically).  The wrong attempt 'bat'
    immediately precedes the correct token 'cat' for target index 1.
    """
    m = detect("the bat cat sat", "the cat sat")
    assert len(m) == 1
    sc = m[0]
    assert sc.type == "self_correction"
    assert sc.target_word == "cat"   # the corrected (target) word
    assert sc.said_word == "bat"     # the wrong attempt
    assert sc.index == 1


def test_self_correction_variant_b_substitute_then_insert_handbuilt_chunks():
    """Variant (b): substitute('cat'->'bat') + insert('cat').

    jiwer 4.0 never emits this shape for a natural correction (see module
    docstring), so we drive the classifier directly with a hand-built chunk
    stream representing it.  target = ['the','cat','sat'],
    hyp = ['the','bat','cat','sat']:
        equal      ref[0:1] hyp[0:1]   the
        substitute ref[1:2] hyp[1:2]   cat -> bat   (wrong attempt at slot)
        insert     ref[2:2] hyp[2:3]   cat          (the correction)
        equal      ref[2:3] hyp[3:4]   sat
    Must classify as ONE self_correction at index 1 (cat corrected, bat said).
    """
    from jiwer.process import AlignmentChunk

    target = ["the", "cat", "sat"]
    hyp = ["the", "bat", "cat", "sat"]
    chunks = [
        AlignmentChunk(type="equal", ref_start_idx=0, ref_end_idx=1, hyp_start_idx=0, hyp_end_idx=1),
        AlignmentChunk(type="substitute", ref_start_idx=1, ref_end_idx=2, hyp_start_idx=1, hyp_end_idx=2),
        AlignmentChunk(type="insert", ref_start_idx=2, ref_end_idx=2, hyp_start_idx=2, hyp_end_idx=3),
        AlignmentChunk(type="equal", ref_start_idx=2, ref_end_idx=3, hyp_start_idx=3, hyp_end_idx=4),
    ]
    m = _classify_alignment(chunks, target, hyp)
    assert len(m) == 1
    sc = m[0]
    assert sc.type == "self_correction"
    assert sc.target_word == "cat"
    assert sc.said_word == "bat"
    assert sc.index == 1


def test_self_correction_variant_b_with_trailing_insertion():
    """Variant (b) where the correcting insert chunk also carries an EXTRA inserted
    word ('dog') after the correction: substitute('cat'->'bat') + insert('cat','dog').
    The correction 'cat' is consumed by the self_correction; 'dog' must surface as a
    plain insertion.  Exercises the insert-tail path."""
    from jiwer.process import AlignmentChunk

    target = ["the", "cat", "sat"]
    hyp = ["the", "bat", "cat", "dog", "sat"]
    chunks = [
        AlignmentChunk(type="equal", ref_start_idx=0, ref_end_idx=1, hyp_start_idx=0, hyp_end_idx=1),
        AlignmentChunk(type="substitute", ref_start_idx=1, ref_end_idx=2, hyp_start_idx=1, hyp_end_idx=2),
        AlignmentChunk(type="insert", ref_start_idx=2, ref_end_idx=2, hyp_start_idx=2, hyp_end_idx=4),
        AlignmentChunk(type="equal", ref_start_idx=2, ref_end_idx=3, hyp_start_idx=4, hyp_end_idx=5),
    ]
    m = _classify_alignment(chunks, target, hyp)
    assert types_at(m) == [("insertion", 2), ("self_correction", 1)]
    sc = next(x for x in m if x.type == "self_correction")
    assert sc.target_word == "cat" and sc.said_word == "bat"
    ins = next(x for x in m if x.type == "insertion")
    assert ins.said_word == "dog" and ins.index == 2


# ===========================================================================
# REPEAT vs SELF-CORRECTION disambiguation
# ===========================================================================

def test_repeat_the_the_is_hesitation_not_correction():
    """'the the cat' over target 'the cat' → ONE hesitation, never self_correction
    and never plain insertion.  The repeated 'the' is a disfluency."""
    m = detect("the the cat", "the cat")
    assert len(m) == 1
    h = m[0]
    assert h.type == "hesitation"
    assert h.type != "self_correction"
    # repeated token sits at the target index of the word being repeated (0 = "the")
    assert h.index == 0
    assert h.said_word == "the"


def test_repeat_midsentence_is_hesitation():
    m = detect("the cat cat sat", "the cat sat")
    assert len(m) == 1
    assert m[0].type == "hesitation"
    assert m[0].index == 1
    assert m[0].said_word == "cat"


# ===========================================================================
# FILLER lexicon — stripped pre-alignment, surfaces as hesitation not insertion
# ===========================================================================

def test_filler_um_midsentence_is_hesitation_not_insertion():
    w = words(
        (" the", 0.0, 0.3),
        (" um", 0.35, 0.6),
        (" cat", 0.65, 0.9),
        (" sat", 0.95, 1.2),
    )
    m = detect(w, "the cat sat")
    assert len(m) == 1
    h = m[0]
    assert h.type == "hesitation"
    assert h.type != "insertion"
    # hesitation aligned before the target word that follows the filler ("cat", idx 1)
    assert h.index == 1


def test_filler_does_not_break_alignment():
    """A filler removed pre-alignment must leave the rest a clean read."""
    w = words(
        (" um", 0.0, 0.3),
        (" the", 0.35, 0.6),
        (" cat", 0.65, 0.9),
    )
    m = detect(w, "the cat")
    # exactly one hesitation, no spurious substitution/insertion
    assert [x.type for x in m] == ["hesitation"]


# ===========================================================================
# TIMING hesitation — timed mode only
# ===========================================================================

def test_timing_gap_creates_hesitation_no_textual_deviation():
    """A long inter-word gap with a perfect textual read → hesitation at the
    index of the word that follows the gap."""
    w = words(
        (" the", 0.0, 0.3),
        (" cat", 0.4, 0.7),
        (" sat", 0.7 + GAP_THRESHOLD_S + 0.5, 0.7 + GAP_THRESHOLD_S + 0.8),  # big gap before "sat"
    )
    m = detect(w, "the cat sat")
    assert len(m) == 1
    h = m[0]
    assert h.type == "hesitation"
    assert h.index == 2  # before "sat"
    assert h.target_word == "sat"


def test_timing_gap_just_under_threshold_no_hesitation():
    w = words(
        (" the", 0.0, 0.3),
        (" cat", 0.4, 0.7),
        (" sat", 0.7 + GAP_THRESHOLD_S - 0.01, 0.7 + GAP_THRESHOLD_S + 0.29),  # gap just under
    )
    assert detect(w, "the cat sat") == []


def test_text_only_mode_no_timing_hesitation():
    """Same logical content as the timing-gap case but text-only: the timing rule
    is INACTIVE, so no hesitation is produced.  Documents the modal difference."""
    assert detect("the cat sat", "the cat sat") == []


# ===========================================================================
# MULTI-MISCUE passage — combine >=3 classes
# ===========================================================================

def test_multi_miscue_passage_all_found():
    """target: the quick brown fox jumps
    hyp    : the slow brown jumps  (sub quick->slow, omit fox)  + filler 'um' + gap
    Build timed words so we also get a timing hesitation; expect sub + omission
    + hesitation all present."""
    w = words(
        (" the", 0.0, 0.3),
        (" slow", 0.35, 0.6),           # substitution for "quick" (idx 1)
        (" brown", 0.65, 0.9),
        # "fox" (idx 3) omitted
        (" um", 0.95, 1.2),             # filler -> hesitation before "jumps"
        (" jumps", 1.25, 1.5),
    )
    m = detect(w, "the quick brown fox jumps")
    found = {x.type for x in m}
    assert "substitution" in found
    assert "omission" in found
    assert "hesitation" in found
    # substitution at idx1, omission at idx3
    subs = [x for x in m if x.type == "substitution"]
    assert subs[0].index == 1 and subs[0].said_word == "slow" and subs[0].target_word == "quick"
    omits = [x for x in m if x.type == "omission"]
    assert omits[0].index == 3 and omits[0].target_word == "fox"


# ===========================================================================
# CONFIDENCE field default
# ===========================================================================

def test_rule_based_confidence_defaults_to_one():
    m = detect("the bat sat", "the cat sat")
    assert m[0].confidence == 1.0


# ===========================================================================
# score() — per-class P/R/F1 (+/-1 tolerance) + FP-per-100-correct-words
# ===========================================================================

def test_score_handcomputed_precision_recall():
    """2 predicted subs, 1 gold sub within +/-1 → P=0.5, R=1.0, F1=2/3."""
    gold = [Miscue("substitution", "cat", "bat", 1)]
    pred = [
        Miscue("substitution", "cat", "bat", 1),   # matches gold (exact)
        Miscue("substitution", "dog", "log", 5),   # false positive
    ]
    out = score(pred, gold, n_target_words=6)
    s = out["substitution"]
    assert s["n_gold"] == 1
    assert s["n_pred"] == 2
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(1.0)
    assert s["f1"] == pytest.approx(2 / 3)


def test_score_plus_minus_one_tolerance_match():
    """index off by exactly 1 → match."""
    gold = [Miscue("omission", "cat", None, 3)]
    pred = [Miscue("omission", "cat", None, 4)]   # off by 1 -> match
    out = score(pred, gold, n_target_words=10)
    o = out["omission"]
    assert o["recall"] == pytest.approx(1.0)
    assert o["precision"] == pytest.approx(1.0)


def test_score_off_by_two_is_no_match():
    """index off by 2 → NO match (FN for gold, FP for pred)."""
    gold = [Miscue("omission", "cat", None, 3)]
    pred = [Miscue("omission", "cat", None, 5)]   # off by 2 -> no match
    out = score(pred, gold, n_target_words=10)
    o = out["omission"]
    assert o["recall"] == pytest.approx(0.0)
    assert o["precision"] == pytest.approx(0.0)


def test_score_class_must_match_for_a_match():
    """A predicted hesitation at the same index as a gold omission does NOT match."""
    gold = [Miscue("omission", "cat", None, 3)]
    pred = [Miscue("hesitation", None, "cat", 3)]
    out = score(pred, gold, n_target_words=10)
    assert out["omission"]["recall"] == pytest.approx(0.0)
    assert out["hesitation"]["precision"] == pytest.approx(0.0)


def test_score_absent_class_reports_none_not_one():
    """A class absent from BOTH gold and predicted reports None, never a
    fabricated 1.0."""
    gold = [Miscue("substitution", "cat", "bat", 1)]
    pred = [Miscue("substitution", "cat", "bat", 1)]
    out = score(pred, gold, n_target_words=5)
    assert out["insertion"]["precision"] is None
    assert out["insertion"]["recall"] is None
    assert out["insertion"]["f1"] is None
    assert out["insertion"]["n_gold"] == 0
    assert out["insertion"]["n_pred"] == 0
    # present class is real
    assert out["substitution"]["precision"] == pytest.approx(1.0)


def test_score_fp_per_100_handcomputed():
    """n_target_words=10, gold has 2 miscues → 8 correctly-read words.
    pred has 1 true positive + 3 false positives → fp_per_100 = 3 / 8 * 100 = 37.5."""
    gold = [
        Miscue("substitution", "a", "b", 1),
        Miscue("omission", "c", None, 4),
    ]
    pred = [
        Miscue("substitution", "a", "b", 1),   # TP
        Miscue("insertion", None, "x", 6),     # FP
        Miscue("insertion", None, "y", 7),     # FP
        Miscue("hesitation", None, "z", 8),    # FP
    ]
    out = score(pred, gold, n_target_words=10)
    assert out["fp_per_100_correct_words"] == pytest.approx(37.5)


# ===========================================================================
# GOLD-vs-GOLD property test — score(gold, gold, n) is perfect
# ===========================================================================

@pytest.mark.parametrize(
    "gold",
    [
        [],
        [Miscue("substitution", "cat", "bat", 1)],
        [Miscue("omission", "cat", None, 1), Miscue("insertion", None, "the", 3)],
        [
            Miscue("substitution", "quick", "slow", 1),
            Miscue("omission", "fox", None, 3),
            Miscue("hesitation", None, "um", 4),
            Miscue("self_correction", "cat", "bat", 1),
            Miscue("insertion", None, "big", 2),
        ],
        # two miscues of the SAME class at adjacent indices (matching must stay 1:1)
        [
            Miscue("substitution", "a", "x", 1),
            Miscue("substitution", "b", "y", 2),
        ],
    ],
)
def test_gold_vs_gold_is_perfect(gold):
    n = 10
    out = score(gold, gold, n_target_words=n)
    present = {m.type for m in gold}
    for cls in ["substitution", "omission", "insertion", "self_correction", "hesitation"]:
        entry = out[cls]
        if cls in present:
            assert entry["precision"] == pytest.approx(1.0), cls
            assert entry["recall"] == pytest.approx(1.0), cls
            assert entry["f1"] == pytest.approx(1.0), cls
        else:
            assert entry["precision"] is None, cls
            assert entry["recall"] is None, cls
            assert entry["f1"] is None, cls
    assert out["fp_per_100_correct_words"] == pytest.approx(0.0)


def test_gold_vs_gold_adjacent_same_class_no_double_match():
    """Two gold subs at indices 1 and 2 scored against themselves must yield
    P=R=1.0 — the +/-1 tolerance must NOT let one pred match two golds (or vice
    versa).  This is the greedy one-to-one matching guard."""
    gold = [
        Miscue("substitution", "a", "x", 1),
        Miscue("substitution", "b", "y", 2),
    ]
    out = score(gold, gold, n_target_words=10)
    assert out["substitution"]["precision"] == pytest.approx(1.0)
    assert out["substitution"]["recall"] == pytest.approx(1.0)
    assert math.isclose(out["substitution"]["f1"], 1.0)
