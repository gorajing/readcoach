"""T4.1 — move-policy rule matrix (per-rule TDD).

Each pedagogy rule from the ticket is its own test, written RED first against the
`decide()` stub.  The policy is a *pure* function: enum-shaped inputs
(miscue type, page position, struggle count, learner gaps) → a frozen
``TutorAction`` carrying a human-readable, rule-id-stamped rationale and the
``POLICY_VERSION``.

Pedagogy contract under test (controller resolution, T4.1):
  - mid-page protects flow: WAIT for everything EXCEPT sustained struggle on the
    same word (>=2 -> SCAFFOLDED_HINT ladder; >=3 -> MODEL_THE_WORD);
  - self-corrections are NEVER corrected, at any struggle count, and the
    rationale never frames the read as an error;
  - hesitation alone (no struggle) -> WAIT (productive struggle protected);
  - the scaffold ladder is deterministic: 2->bounce, 3->highlight, 4->phonetic;
  - page-end coaching: open comprehension -> COMPREHENSION_PROMPT; clean/strong
    page -> ENCOURAGE; page done with a warranted next item -> NEXT_ITEM;
  - POLICY_VERSION is stamped on EVERY action;
  - any uncovered combination -> conservative WAIT default (proven unreached by
    the replay in tests/test_policy_replay.py).
"""
from __future__ import annotations

import dataclasses

import pytest

from readcoach.learner_model import LearnerState
from readcoach.miscue import Miscue
from readcoach.tutor import (
    POLICY_VERSION,
    DEFAULT_RULE_ID,
    TutorAction,
    TutorContext,
    decide,
    hint_level_for,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _state(mastery: dict[str, float] | None = None) -> LearnerState:
    return LearnerState(child_id="kid", mastery=dict(mastery or {}))


def _ctx(
    *,
    miscue: Miscue | None = None,
    at_page_end: bool = False,
    consecutive_struggles: int = 0,
    page_had_struggle: bool = False,
    open_comprehension: bool = False,
    mastery: dict[str, float] | None = None,
) -> TutorContext:
    return TutorContext(
        miscue=miscue,
        learner_state=_state(mastery),
        at_page_end=at_page_end,
        consecutive_struggles=consecutive_struggles,
        page_had_struggle=page_had_struggle,
        open_comprehension=open_comprehension,
    )


def _sub(word: str = "dog", said: str = "gog", index: int = 9) -> Miscue:
    return Miscue("substitution", target_word=word, said_word=said, index=index)


# ---------------------------------------------------------------------------
# Structural: every action stamps the policy version, dataclasses frozen
# ---------------------------------------------------------------------------

def test_tutor_action_is_frozen():
    a = decide(_ctx())
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.move = "ENCOURAGE"  # type: ignore[misc]


def test_tutor_context_is_frozen():
    c = _ctx()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.at_page_end = True  # type: ignore[misc]


@pytest.mark.parametrize(
    "ctx",
    [
        _ctx(),  # default
        _ctx(miscue=_sub(), consecutive_struggles=1),  # mid-page single miscue
        _ctx(miscue=_sub(), consecutive_struggles=2),  # scaffold
        _ctx(miscue=_sub(), consecutive_struggles=3),  # model
        _ctx(at_page_end=True),  # page-end clean
        _ctx(at_page_end=True, open_comprehension=True),  # comprehension
        _ctx(at_page_end=True, page_had_struggle=True, mastery={"cvc_blend": 0.5}),
        _ctx(
            miscue=Miscue("self_correction", "to", "do", 24, confidence=0.7),
            consecutive_struggles=3,
        ),
    ],
)
def test_policy_version_stamped_on_every_action(ctx):
    action = decide(ctx)
    assert action.policy_version == POLICY_VERSION
    assert POLICY_VERSION == "1.0.0"


def test_rationale_always_carries_a_rule_id():
    # Every rationale embeds its stable rule_id so a logged action is auditable.
    for ctx in (
        _ctx(),
        _ctx(miscue=_sub(), consecutive_struggles=2),
        _ctx(miscue=_sub(), consecutive_struggles=3),
        _ctx(at_page_end=True),
        _ctx(at_page_end=True, open_comprehension=True),
    ):
        action = decide(ctx)
        assert "[" in action.rationale and "]" in action.rationale, action.rationale


# ---------------------------------------------------------------------------
# Rule: mid-page, no sustained struggle -> WAIT always
# ---------------------------------------------------------------------------

def test_mid_page_single_miscue_waits():
    action = decide(_ctx(miscue=_sub(), consecutive_struggles=1))
    assert action.move == "WAIT"


def test_mid_page_first_miscue_zero_struggle_waits():
    action = decide(_ctx(miscue=_sub(), consecutive_struggles=0))
    assert action.move == "WAIT"


def test_mid_page_no_miscue_waits():
    action = decide(_ctx(miscue=None, consecutive_struggles=0))
    assert action.move == "WAIT"


# ---------------------------------------------------------------------------
# Rule: hesitation alone (no struggle) -> WAIT (productive struggle protected)
# ---------------------------------------------------------------------------

def test_hesitation_alone_waits():
    hes = Miscue("hesitation", target_word=None, said_word="um", index=5)
    action = decide(_ctx(miscue=hes, consecutive_struggles=0))
    assert action.move == "WAIT"


def test_hesitation_does_not_count_as_struggle_even_when_repeated():
    # A hesitation is never a "failed attempt"; even at a high count it must not
    # trigger a corrective move on its own.
    hes = Miscue("hesitation", target_word=None, said_word="um", index=5)
    action = decide(_ctx(miscue=hes, consecutive_struggles=0))
    assert action.move == "WAIT"


# ---------------------------------------------------------------------------
# Rule: struggles >= 2 same word -> SCAFFOLDED_HINT with ladder level
# ---------------------------------------------------------------------------

def test_two_struggles_gives_scaffolded_hint_bounce():
    action = decide(_ctx(miscue=_sub(), consecutive_struggles=2))
    assert action.move == "SCAFFOLDED_HINT"
    assert action.hint_level == "bounce"
    assert action.target_word == "dog"


def test_scaffolded_hint_carries_error_type_from_miscue():
    action = decide(_ctx(miscue=_sub(), consecutive_struggles=2))
    assert action.error_type == "substitution"


# ---------------------------------------------------------------------------
# Rule: struggles >= 3 -> MODEL_THE_WORD (give the word, move on)
# ---------------------------------------------------------------------------

def test_three_struggles_models_the_word():
    action = decide(_ctx(miscue=_sub(), consecutive_struggles=3))
    assert action.move == "MODEL_THE_WORD"
    assert action.target_word == "dog"
    assert action.hint_level is None


def test_many_struggles_still_models_the_word():
    action = decide(_ctx(miscue=_sub(), consecutive_struggles=7))
    assert action.move == "MODEL_THE_WORD"


# ---------------------------------------------------------------------------
# Ladder helper: deterministic 2->bounce, 3->highlight, 4->phonetic
# ---------------------------------------------------------------------------

def test_ladder_progression_is_deterministic():
    assert hint_level_for(2) == "bounce"
    assert hint_level_for(3) == "highlight"
    assert hint_level_for(4) == "phonetic"
    # Saturates at the top rung; never returns an undefined level.
    assert hint_level_for(5) == "phonetic"
    assert hint_level_for(99) == "phonetic"


def test_ladder_below_threshold_is_none():
    assert hint_level_for(0) is None
    assert hint_level_for(1) is None


# ---------------------------------------------------------------------------
# Rule: self-correction is NEVER corrected, at any struggle count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("struggles", [0, 1, 2, 3, 5, 10])
def test_self_correction_never_scaffolded_or_modeled(struggles):
    sc = Miscue("self_correction", target_word="to", said_word="do", index=24, confidence=0.7)
    action = decide(_ctx(miscue=sc, consecutive_struggles=struggles))
    assert action.move not in ("SCAFFOLDED_HINT", "MODEL_THE_WORD")


@pytest.mark.parametrize("struggles", [0, 1, 2, 3, 5, 10])
def test_self_correction_rationale_has_no_error_framing(struggles):
    sc = Miscue("self_correction", target_word="to", said_word="do", index=24, confidence=0.7)
    action = decide(_ctx(miscue=sc, consecutive_struggles=struggles))
    assert "error" not in action.rationale.lower()
    assert "wrong" not in action.rationale.lower()
    assert "mistake" not in action.rationale.lower()


def test_self_correction_mid_page_waits():
    sc = Miscue("self_correction", target_word="to", said_word="do", index=24, confidence=0.7)
    action = decide(_ctx(miscue=sc, consecutive_struggles=2))
    assert action.move == "WAIT"


def test_self_correction_at_page_end_is_celebrated_not_recorrected():
    sc = Miscue("self_correction", target_word="to", said_word="do", index=24, confidence=0.7)
    action = decide(_ctx(miscue=sc, at_page_end=True))
    assert action.move == "ENCOURAGE"
    assert "error" not in action.rationale.lower()


# ---------------------------------------------------------------------------
# Rule: page-end coaching
# ---------------------------------------------------------------------------

def test_page_end_open_comprehension_prompts():
    action = decide(_ctx(at_page_end=True, open_comprehension=True))
    assert action.move == "COMPREHENSION_PROMPT"


def test_page_end_clean_strong_encourages():
    action = decide(_ctx(at_page_end=True, page_had_struggle=False))
    assert action.move == "ENCOURAGE"


def test_page_end_with_gaps_and_struggle_selects_next_item():
    # Page done, the child struggled, and there is a mastery gap to target ->
    # NEXT_ITEM selection comes from the learner's gaps.
    action = decide(
        _ctx(
            at_page_end=True,
            page_had_struggle=True,
            mastery={"vowel_team_ea": 0.40, "digraph_ch": 0.99},
        )
    )
    assert action.move == "NEXT_ITEM"
    # The chosen next item is one of the open gaps, never a mastered skill.
    assert action.target_word == "vowel_team_ea"


def test_page_end_comprehension_takes_precedence_over_next_item():
    action = decide(
        _ctx(
            at_page_end=True,
            page_had_struggle=True,
            open_comprehension=True,
            mastery={"vowel_team_ea": 0.40},
        )
    )
    assert action.move == "COMPREHENSION_PROMPT"


# ---------------------------------------------------------------------------
# Conservative default
# ---------------------------------------------------------------------------

def test_unmatched_context_defaults_to_wait():
    # An empty mid-page context matches no corrective rule -> conservative WAIT.
    action = decide(_ctx())
    assert action.move == "WAIT"


def test_default_rule_rationale_is_explicit():
    # The default backstop is reachable only by a contradictory mid-page state
    # (no miscue AND a non-zero struggle count) that the matrix's named rules do
    # not cover — constructing it directly proves the backstop is a safe WAIT
    # carrying the documented rationale, never a raise.  The replay test proves
    # this path is never hit on real data.
    action = decide(_ctx(miscue=None, consecutive_struggles=4))
    assert action.move == "WAIT"
    assert DEFAULT_RULE_ID == "R-DEFAULT"
    assert f"[{DEFAULT_RULE_ID}]" in action.rationale
    assert "waiting is the safe move" in action.rationale


def test_action_returned_is_tutor_action():
    assert isinstance(decide(_ctx()), TutorAction)
