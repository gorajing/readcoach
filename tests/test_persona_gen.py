"""T5.3 — persona generator: calibrated, grounded-by-construction (red first).

These tests pin the persona profile loader (fail-loud on a missing rate field),
generation determinism, the band-ceiling grounding (no session uses a passage
above the persona's ceiling), the corpus-level rate-calibration assertion (and
that it FIRES when a persona spec is perturbed), and that every generated event
carries one of the 5 miscue classes with a real said_word drawn from the
passage's own band via the inject.py machinery.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from readcoach.inject import load_passages
from readcoach.miscue import _ALL_CLASSES
from readcoach.persona_gen import (
    CalibrationError,
    Persona,
    SessionEvent,
    SessionItem,
    assert_corpus_calibration,
    generate_all,
    generate_session,
    load_personas,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PERSONAS_DIR = _PROJECT_ROOT / "data" / "personas"
_PASSAGES_DIR = _PROJECT_ROOT / "data" / "passages"

# The 5 detector miscue classes plus the two non-miscue event kinds the stream
# can carry. "clean" = a correctly-read word; "page_end" = a page boundary.
_EVENT_KINDS = set(_ALL_CLASSES) | {"clean", "page_end"}


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------

def test_load_personas_returns_three_sorted_by_id():
    personas = load_personas(_PERSONAS_DIR)
    ids = [p.id for p in personas]
    assert ids == sorted(ids)
    assert set(ids) == {"emergent", "ell_profile", "dyslexic_profile"}


def test_persona_has_all_five_rate_classes_and_citations():
    personas = {p.id: p for p in load_personas(_PERSONAS_DIR)}
    for p in personas.values():
        for cls in _ALL_CLASSES:
            assert cls in p.rates, f"{p.id} missing rate for {cls}"
            assert p.rates[cls] >= 0.0
        assert p.citations, f"{p.id} has no citations"
        assert p.calibration_notes, f"{p.id} has no calibration_notes"
        assert 1 <= p.band_ceiling <= 4


def test_load_personas_fails_loud_on_missing_rate_field(tmp_path):
    # A persona YAML missing one of the 5 rate classes must raise (fail loud),
    # not silently default the rate to 0.
    bad = tmp_path / "broken.yaml"
    bad.write_text(
        "id: broken\n"
        "label: Broken\n"
        "description: missing the insertion rate\n"
        "band_ceiling: 2\n"
        "rates:\n"
        "  substitution: 5.0\n"
        "  omission: 3.0\n"
        # insertion intentionally omitted
        "  self_correction: 1.5\n"
        "  hesitation: 8.0\n"
        "struggle_escalation_p: 0.4\n"
        "citations: [x]\n"
        "calibration_notes: [y]\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="insertion"):
        load_personas(tmp_path)


def test_load_personas_fails_loud_on_band_ceiling_out_of_range(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text(
        "id: broken\n"
        "label: Broken\n"
        "description: band ceiling too high\n"
        "band_ceiling: 7\n"
        "rates:\n"
        "  substitution: 5.0\n"
        "  omission: 3.0\n"
        "  insertion: 1.0\n"
        "  self_correction: 1.5\n"
        "  hesitation: 8.0\n"
        "struggle_escalation_p: 0.4\n"
        "citations: [x]\n"
        "calibration_notes: [y]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="band_ceiling"):
        load_personas(tmp_path)


# ---------------------------------------------------------------------------
# Generation: shape + grounding
# ---------------------------------------------------------------------------

def _emergent() -> Persona:
    return {p.id: p for p in load_personas(_PERSONAS_DIR)}["emergent"]


def test_generate_session_shape_and_event_kinds():
    persona = _emergent()
    passage = load_passages(_PASSAGES_DIR)[0]
    item = generate_session(persona, passage, random.Random(7), session_index=0)

    assert isinstance(item, SessionItem)
    assert item.persona_id == persona.id
    assert item.passage_id == passage.id
    assert item.events, "session must have events"
    for ev in item.events:
        assert isinstance(ev, SessionEvent)
        assert ev.kind in _EVENT_KINDS
        # Every miscue event carries a class in the 5 detector classes (grounding).
        if ev.kind in _ALL_CLASSES:
            assert ev.kind in set(_ALL_CLASSES)


def test_generated_said_words_are_real_words_from_the_passage_band():
    """Substitution/self_correction said_words are inject.py orthographic
    neighbours of an ACTUAL target word in the passage — never a placeholder."""
    persona = _emergent()
    passage = load_passages(_PASSAGES_DIR)[0]
    item = generate_session(persona, passage, random.Random(11), session_index=0)
    saw_real_said = False
    for ev in item.events:
        if ev.kind in ("substitution", "self_correction"):
            assert ev.said_word, "substitution/self_correction must carry a said_word"
            assert ev.said_word != "?", "said_word must be a real word, not a placeholder"
            # target_word is a real word from the passage at that index.
            assert ev.target_word == passage.words[ev.word_index].strip(".,!?;:\"'").casefold()
            saw_real_said = True
    assert saw_real_said, "expected at least one substitution/self_correction in this session"


def test_generate_session_is_deterministic_for_fixed_seed():
    persona = _emergent()
    passage = load_passages(_PASSAGES_DIR)[0]
    a = generate_session(persona, passage, random.Random(99), session_index=0)
    b = generate_session(persona, passage, random.Random(99), session_index=0)
    assert a == b


def test_generate_session_differs_with_different_seed():
    persona = _emergent()
    passage = load_passages(_PASSAGES_DIR)[0]
    a = generate_session(persona, passage, random.Random(1), session_index=0)
    b = generate_session(persona, passage, random.Random(2), session_index=0)
    assert a != b


# ---------------------------------------------------------------------------
# generate_all: corpus size, determinism, band-ceiling grounding
# ---------------------------------------------------------------------------

def test_generate_all_is_deterministic_and_sized_90_to_120():
    personas = load_personas(_PERSONAS_DIR)
    passages = load_passages(_PASSAGES_DIR)
    items_a = generate_all(personas, passages, seed=2026)
    items_b = generate_all(personas, passages, seed=2026)
    assert items_a == items_b
    assert 90 <= len(items_a) <= 120, f"corpus size {len(items_a)} outside 90-120"
    # unique ids
    assert len({it.id for it in items_a}) == len(items_a)


def test_generate_all_respects_band_ceiling_grounding():
    """No session may use a passage whose band exceeds the persona's ceiling.

    This IS the 'never vocabulary above the passage/level' grounding, asserted by
    construction: a band-2-ceiling persona never reads a band-3/4 passage.
    """
    personas = {p.id: p for p in load_personas(_PERSONAS_DIR)}
    passages = {p.id: p for p in load_passages(_PASSAGES_DIR)}
    items = generate_all(list(personas.values()), list(passages.values()), seed=2026)
    for it in items:
        ceiling = personas[it.persona_id].band_ceiling
        band = passages[it.passage_id].band
        assert band <= ceiling, (
            f"session {it.id}: passage band {band} > persona ceiling {ceiling}"
        )


def test_generate_all_covers_every_persona():
    personas = load_personas(_PERSONAS_DIR)
    passages = load_passages(_PASSAGES_DIR)
    items = generate_all(personas, passages, seed=2026)
    seen = {it.persona_id for it in items}
    assert seen == {p.id for p in personas}


# ---------------------------------------------------------------------------
# Calibration assertion: green for the real specs, FIRES on perturbation
# ---------------------------------------------------------------------------

def test_corpus_calibration_passes_for_real_personas():
    personas = load_personas(_PERSONAS_DIR)
    passages = load_passages(_PASSAGES_DIR)
    items = generate_all(personas, passages, seed=2026)
    # Must not raise: observed per-100 rates are within tolerance of each
    # persona's spec across the corpus.
    assert_corpus_calibration(personas, passages, items, tolerance=0.5)


def test_corpus_calibration_fires_when_persona_spec_perturbed():
    """Construct a violating generator config: claim a tiny substitution rate but
    generate with the real (high) one -> the observed rate is far above spec, so
    the assertion must FIRE."""
    personas = load_personas(_PERSONAS_DIR)
    passages = load_passages(_PASSAGES_DIR)
    items = generate_all(personas, passages, seed=2026)

    # Perturb the emergent persona's substitution rate to a value the generated
    # corpus cannot match (claim 0.1/100 when items were made at 5.5/100).
    perturbed = []
    for p in personas:
        if p.id == "emergent":
            bad_rates = dict(p.rates)
            bad_rates["substitution"] = 0.1
            perturbed.append(
                Persona(
                    id=p.id,
                    label=p.label,
                    description=p.description,
                    band_ceiling=p.band_ceiling,
                    rates=bad_rates,
                    struggle_escalation_p=p.struggle_escalation_p,
                    citations=p.citations,
                    calibration_notes=p.calibration_notes,
                )
            )
        else:
            perturbed.append(p)

    with pytest.raises(CalibrationError, match="substitution"):
        assert_corpus_calibration(perturbed, passages, items, tolerance=0.5)
