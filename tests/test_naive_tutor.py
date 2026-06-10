"""T5.0 — NaiveTutor baseline tests (red first, then green).

Covers:
  * naive turn construction (action_move=None, utterance=response text)
  * generalized mid-page check: None-action + non-empty utterance mid-page → finding
  * empty utterance with None action mid-page → no finding
  * policy-harness traces still 0 violations (clean-trace regression)
  * stub determinism (same event → same canned response)
  * audit integration: stub run → ≥1 violation per profile, specific rules asserted
  * missing key raises RuntimeError loud
"""
from __future__ import annotations

from pathlib import Path

import pytest

from readcoach.naive_tutor import NaiveTutor, StubTransport
from readcoach.policy_compiler import audit, compile_rules, load_policies
from readcoach.trace import SessionTrace, TurnRecord

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_POLICIES_DIR = _PROJECT_ROOT / "policies"


# ---------------------------------------------------------------------------
# Helpers
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


def _trace(turns, *, completed=()) -> SessionTrace:
    reindexed = tuple(
        TurnRecord(
            turn_index=i,
            at_page_end=t.at_page_end,
            miscue_type=t.miscue_type,
            action_move=t.action_move,
            hint_level=t.hint_level,
            served_reason=t.served_reason,
            utterance=t.utterance,
            is_ai_reminder=t.is_ai_reminder,
            skill_id=t.skill_id,
        )
        for i, t in enumerate(turns)
    )
    return SessionTrace(
        child_id="kid",
        policy_version="naive-stub",
        completed_skills_at_start=tuple(completed),
        turns=reindexed,
    )


def _checks():
    return compile_rules(load_policies(_POLICIES_DIR))


# ---------------------------------------------------------------------------
# Generalized never_coaches_mid_page check (None-action utterance)
# ---------------------------------------------------------------------------

def test_mid_page_none_action_with_utterance_is_violation():
    """action_move=None + non-empty utterance on a mid-page turn → finding."""
    checks = {c.rule_id: c for c in _checks()}
    c = checks["never_coaches_mid_page"]
    trace = _trace([_turn(action_move=None, utterance="The word is cat.", at_page_end=False)])
    findings = c(trace)
    assert len(findings) == 1
    assert findings[0].rule_id == "never_coaches_mid_page"
    assert findings[0].severity == "error"


def test_mid_page_none_action_empty_utterance_is_clean():
    """action_move=None + empty/None utterance on a mid-page turn → no finding."""
    checks = {c.rule_id: c for c in _checks()}
    c = checks["never_coaches_mid_page"]
    for utt in (None, "", "   "):
        trace = _trace([_turn(action_move=None, utterance=utt, at_page_end=False)])
        assert c(trace) == [], f"expected clean for utterance={utt!r}"


def test_mid_page_none_action_at_page_end_utterance_is_clean():
    """action_move=None + utterance on a PAGE-END turn → no finding (coaching allowed there)."""
    checks = {c.rule_id: c for c in _checks()}
    c = checks["never_coaches_mid_page"]
    trace = _trace([_turn(action_move=None, utterance="Great work!", at_page_end=True)])
    assert c(trace) == []


def test_policy_harness_wait_no_utterance_is_clean():
    """WAIT move with no utterance (policy harness) → no mid-page finding."""
    checks = {c.rule_id: c for c in _checks()}
    c = checks["never_coaches_mid_page"]
    trace = _trace([_turn(action_move="WAIT", utterance=None, at_page_end=False)])
    assert c(trace) == []


# ---------------------------------------------------------------------------
# Policy-harness traces stay at 0 violations under the generalized check
# ---------------------------------------------------------------------------

def test_policy_harness_mid_page_moves_with_utterances_not_flagged():
    """Policy-harness SCAFFOLDED_HINT / MODEL_THE_WORD mid-page still clean.

    These moves have action_move != None, so the None-action branch never fires.
    The named-coaching-move branch only fires on ENCOURAGE/COMPREHENSION_PROMPT/
    NEXT_ITEM, so mid-page hint/model turns remain compliant.
    """
    checks = {c.rule_id: c for c in _checks()}
    c = checks["never_coaches_mid_page"]
    trace = _trace([
        _turn(action_move="SCAFFOLDED_HINT", utterance="Look at the first sound.", at_page_end=False),
        _turn(action_move="MODEL_THE_WORD", utterance="That word is 'chip'.", at_page_end=False),
    ])
    assert c(trace) == []


# ---------------------------------------------------------------------------
# NaiveTutor turn construction
# ---------------------------------------------------------------------------

def test_naive_tutor_turn_has_none_action_move():
    """NaiveTutor always sets action_move=None (no policy taxonomy)."""
    stub = StubTransport()
    tutor = NaiveTutor(client_factory=lambda: stub)
    record = tutor.react(
        turn_index=0,
        at_page_end=False,
        miscue_type="substitution",
        target_word="chip",
    )
    assert record.action_move is None


def test_naive_tutor_turn_has_utterance():
    """NaiveTutor sets utterance to the model's response text."""
    stub = StubTransport()
    tutor = NaiveTutor(client_factory=lambda: stub)
    record = tutor.react(
        turn_index=0,
        at_page_end=False,
        miscue_type="substitution",
        target_word="chip",
    )
    assert record.utterance is not None
    assert record.utterance.strip() != ""


def test_naive_tutor_turn_fields_match_inputs():
    """TurnRecord fields come from the inputs passed to react()."""
    stub = StubTransport()
    tutor = NaiveTutor(client_factory=lambda: stub)
    record = tutor.react(
        turn_index=5,
        at_page_end=True,
        miscue_type="self_correction",
        target_word="wagon",
    )
    assert record.turn_index == 5
    assert record.at_page_end is True
    assert record.miscue_type == "self_correction"


def test_naive_tutor_none_miscue_still_produces_record():
    """react() with no miscue (clean word read) still produces a TurnRecord."""
    stub = StubTransport()
    tutor = NaiveTutor(client_factory=lambda: stub)
    record = tutor.react(
        turn_index=2,
        at_page_end=False,
        miscue_type=None,
        target_word="the",
    )
    assert record.action_move is None
    assert record.utterance is not None


# ---------------------------------------------------------------------------
# Stub determinism
# ---------------------------------------------------------------------------

def test_stub_transport_is_deterministic():
    """Same event type yields the same canned response every time."""
    stub = StubTransport()
    tutor = NaiveTutor(client_factory=lambda: stub)
    r1 = tutor.react(turn_index=0, at_page_end=False, miscue_type="substitution", target_word="cat")
    r2 = tutor.react(turn_index=1, at_page_end=False, miscue_type="substitution", target_word="cat")
    assert r1.utterance == r2.utterance


def test_stub_transport_supplies_word_on_miscue():
    """The stub gives away the word (typical unconstrained helpful-assistant behavior)."""
    stub = StubTransport()
    tutor = NaiveTutor(client_factory=lambda: stub)
    record = tutor.react(
        turn_index=0, at_page_end=False, miscue_type="substitution", target_word="chip"
    )
    # The stub is documented to immediately supply the correct word on any miscue.
    assert "chip" in record.utterance


def test_stub_transport_corrects_self_correction():
    """The stub corrects even self-corrections (a documented violation pattern)."""
    stub = StubTransport()
    tutor = NaiveTutor(client_factory=lambda: stub)
    record = tutor.react(
        turn_index=0, at_page_end=False, miscue_type="self_correction", target_word="wagon"
    )
    # The stub gives the word on self-corrections too — auditable violation.
    assert "wagon" in record.utterance


def test_stub_transport_praises_effusively():
    """The stub adds effusive praise on clean reads (auditable never_says_wrong pattern)."""
    stub = StubTransport()
    tutor = NaiveTutor(client_factory=lambda: stub)
    record = tutor.react(
        turn_index=0, at_page_end=True, miscue_type=None, target_word="the"
    )
    # The stub is documented to praise effusively.
    assert record.utterance is not None


# ---------------------------------------------------------------------------
# Audit integration: stub run → violations per profile, specific rules
# ---------------------------------------------------------------------------

def _build_naive_stub_traces():
    """Run the stub transport through all 3 scripted profiles; return traces."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "naive_replay", _PROJECT_ROOT / "scripts" / "naive_replay.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_stub()


def test_stub_run_produces_three_traces():
    traces = _build_naive_stub_traces()
    assert len(traces) == 3
    names = {t.child_id for t in traces}
    assert names == {"struggling-decoder", "fluent-but-hesitant", "self-corrector"}


def test_stub_run_every_profile_has_violations():
    """Every profile should have ≥1 violation when the naive stub runs."""
    checks = _checks()
    traces = _build_naive_stub_traces()
    for trace in traces:
        report = audit(trace, checks)
        assert report.violations >= 1, (
            f"profile {trace.child_id!r} expected violations; got 0"
        )


def test_stub_run_never_coaches_mid_page_fires():
    """The stub utters mid-page → never_coaches_mid_page violations."""
    checks = _checks()
    traces = _build_naive_stub_traces()
    for trace in traces:
        report = audit(trace, checks)
        rule_ids = {f.rule_id for f in report.findings}
        assert "never_coaches_mid_page" in rule_ids, (
            f"profile {trace.child_id!r}: expected never_coaches_mid_page to fire"
        )


def test_stub_run_never_says_wrong_fires():
    """The stub uses corrective phrasing ('wrong word, the word is X') → never_says_wrong."""
    checks = _checks()
    traces = _build_naive_stub_traces()
    # At least one profile should trip never_says_wrong.
    all_rule_ids = set()
    for trace in traces:
        report = audit(trace, checks)
        all_rule_ids.update(f.rule_id for f in report.findings)
    assert "never_says_wrong" in all_rule_ids


def test_stub_run_periodic_ai_reminder_fires():
    """The stub never sets is_ai_reminder → periodic_ai_reminder fires."""
    checks = _checks()
    traces = _build_naive_stub_traces()
    for trace in traces:
        report = audit(trace, checks)
        rule_ids = {f.rule_id for f in report.findings}
        assert "periodic_ai_reminder" in rule_ids, (
            f"profile {trace.child_id!r}: expected periodic_ai_reminder to fire"
        )


def test_stub_run_never_corrects_self_correction_fires():
    """The stub corrects self-corrections → never_corrects_self_correction fires."""
    checks = _checks()
    traces = _build_naive_stub_traces()
    all_rule_ids = set()
    for trace in traces:
        report = audit(trace, checks)
        all_rule_ids.update(f.rule_id for f in report.findings)
    assert "never_corrects_self_correction" in all_rule_ids


# ---------------------------------------------------------------------------
# Missing API key raises loud
# ---------------------------------------------------------------------------

def test_missing_api_key_raises_loud(monkeypatch):
    """NaiveTutor with the default transport raises RuntimeError when key absent."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    tutor = NaiveTutor()  # default transport
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        tutor.react(
            turn_index=0, at_page_end=False, miscue_type="substitution", target_word="cat"
        )
