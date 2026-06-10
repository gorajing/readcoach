"""T4.3 — SessionTrace serde (red first).

The trace is what the policy-compiler checks consume and what the tutor harness
emits.  Frozen dataclasses, JSON round-trip, fail-loud on malformed input.
"""
from __future__ import annotations

import json

import pytest

from readcoach.trace import (
    SessionTrace,
    TurnRecord,
    trace_from_dict,
    trace_from_json,
    trace_to_dict,
    trace_to_json,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _turn(**kw) -> TurnRecord:
    base = dict(
        turn_index=0,
        at_page_end=False,
        miscue_type=None,
        action_move="WAIT",
        hint_level=None,
        served_reason=None,
        utterance=None,
        is_ai_reminder=False,
        skill_id=None,
    )
    base.update(kw)
    return TurnRecord(**base)


def _trace(turns=None, **kw) -> SessionTrace:
    base = dict(
        child_id="kid-1",
        policy_version="1.0.0",
        completed_skills_at_start=(),
        turns=tuple(turns or ()),
    )
    base.update(kw)
    return SessionTrace(**base)


# ---------------------------------------------------------------------------
# Frozen contract
# ---------------------------------------------------------------------------

def test_turn_record_is_frozen():
    t = _turn()
    with pytest.raises(Exception):
        t.turn_index = 9  # type: ignore[misc]


def test_session_trace_is_frozen():
    s = _trace()
    with pytest.raises(Exception):
        s.child_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_dict_round_trip_preserves_all_fields():
    s = _trace(
        turns=[
            _turn(turn_index=0, action_move="WAIT"),
            _turn(
                turn_index=1,
                at_page_end=True,
                miscue_type="substitution",
                action_move="MODEL_THE_WORD",
                hint_level="phonetic",
                served_reason="new",
                utterance="The word is cat.",
                is_ai_reminder=True,
                skill_id="cvc_short_a",
            ),
        ],
        completed_skills_at_start=("digraph_ch",),
    )
    d = trace_to_dict(s)
    back = trace_from_dict(d)
    assert back == s


def test_json_round_trip_preserves_all_fields():
    s = _trace(
        turns=[_turn(turn_index=0), _turn(turn_index=1, is_ai_reminder=True)],
        completed_skills_at_start=("silent_e",),
    )
    text = trace_to_json(s)
    # It must be real JSON.
    json.loads(text)
    back = trace_from_json(text)
    assert back == s


def test_turns_are_a_tuple_after_deserialization():
    """Frozen means the turns container must be hashable/immutable (a tuple)."""
    s = trace_from_dict(trace_to_dict(_trace(turns=[_turn()])))
    assert isinstance(s.turns, tuple)
    assert isinstance(s.completed_skills_at_start, tuple)


# ---------------------------------------------------------------------------
# Fail-loud
# ---------------------------------------------------------------------------

def test_from_dict_missing_required_key_raises():
    d = trace_to_dict(_trace(turns=[_turn()]))
    del d["turns"][0]["action_move"]
    with pytest.raises((KeyError, ValueError)):
        trace_from_dict(d)


def test_from_dict_unknown_turn_key_raises():
    d = trace_to_dict(_trace(turns=[_turn()]))
    d["turns"][0]["bogus_field"] = 1
    with pytest.raises((TypeError, ValueError, KeyError)):
        trace_from_dict(d)


def test_from_json_malformed_raises():
    with pytest.raises(json.JSONDecodeError):
        trace_from_json("{not json")


def test_negative_turn_index_raises():
    with pytest.raises(ValueError):
        _turn(turn_index=-1)
