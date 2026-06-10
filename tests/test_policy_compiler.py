"""T4.3 — policy compiler: invariants-as-data -> named executable checks (red first).

Each YAML rule (policies/*.yaml) carries the verbatim policy sentence it
implements; the compiler validates the rules fail-loud, emits a named Check per
rule, and ``audit()`` runs the checks over a SessionTrace producing severity-
bucketed Findings with ``violations`` = count of severity-error findings.

Test coverage (per ticket): load/validation (unknown check type -> loud; missing
verbatim_sentence -> loud), each check >=1 violating + >=1 clean, the periodic
window edges (turn 0, exactly-N gap), audit counts.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from readcoach.policy_compiler import (
    Check,
    Finding,
    audit,
    compile_rules,
    load_policies,
)
from readcoach.trace import SessionTrace, TurnRecord

POLICIES_DIR = Path(__file__).resolve().parent.parent / "policies"


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


def _trace(turns, *, completed=()) -> SessionTrace:
    # Re-index turns so turn_index is consistent and ai-reminder windows behave.
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
        policy_version="1.0.0",
        completed_skills_at_start=tuple(completed),
        turns=reindexed,
    )


def _checks_by_id() -> dict[str, Check]:
    rules = load_policies(POLICIES_DIR)
    return {c.rule_id: c for c in compile_rules(rules)}


def _run(check: Check, trace: SessionTrace) -> list[Finding]:
    return check(trace)


# ---------------------------------------------------------------------------
# load_policies / validation (fail-loud)
# ---------------------------------------------------------------------------

def test_load_real_policies_yields_all_active_rules():
    rules = load_policies(POLICIES_DIR)
    ids = {r.id for r in rules}
    # Six executable checks; discloses_ai_when_asked is deferred (not compiled).
    assert {
        "never_says_wrong",
        "never_coaches_mid_page",
        "never_corrects_self_correction",
        "no_emotional_intimacy",
        "periodic_ai_reminder",
        "never_reserves_completed_item",
    } <= ids


def test_every_active_rule_carries_a_verbatim_sentence_and_source():
    for r in load_policies(POLICIES_DIR):
        assert r.verbatim_sentence and r.verbatim_sentence.strip()
        assert r.source_url and r.source_name


def test_deferred_rule_is_loaded_but_not_compiled():
    rules = load_policies(POLICIES_DIR)
    deferred = [r for r in rules if r.deferred]
    assert any(r.id == "discloses_ai_when_asked" for r in deferred)
    compiled_ids = {c.rule_id for c in compile_rules(rules)}
    assert "discloses_ai_when_asked" not in compiled_ids


def test_compile_yields_six_executable_checks():
    checks = compile_rules(load_policies(POLICIES_DIR))
    assert len(checks) == 6


def test_unknown_check_type_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "policy_set: bad\nversion: '1.0'\nrules:\n"
        "  - id: mystery\n"
        "    severity: error\n"
        "    verbatim_sentence: 'x'\n"
        "    source: {name: n, url: u, accessed: '2026-06-10'}\n"
        "    check: {type: not_a_real_check, params: {}}\n"
    )
    rules = load_policies(tmp_path)
    with pytest.raises(ValueError, match="unknown check type"):
        compile_rules(rules)


def test_missing_verbatim_sentence_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "policy_set: bad\nversion: '1.0'\nrules:\n"
        "  - id: never_says_wrong\n"
        "    severity: error\n"
        "    source: {name: n, url: u, accessed: '2026-06-10'}\n"
        "    check: {type: never_says_wrong, params: {lexicon: [wrong]}}\n"
    )
    with pytest.raises((ValueError, KeyError), match="verbatim_sentence"):
        load_policies(tmp_path)


def test_missing_source_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "policy_set: bad\nversion: '1.0'\nrules:\n"
        "  - id: never_says_wrong\n"
        "    severity: error\n"
        "    verbatim_sentence: 'x'\n"
        "    check: {type: never_says_wrong, params: {lexicon: [wrong]}}\n"
    )
    with pytest.raises((ValueError, KeyError)):
        load_policies(tmp_path)


def test_bad_severity_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "policy_set: bad\nversion: '1.0'\nrules:\n"
        "  - id: never_says_wrong\n"
        "    severity: catastrophic\n"
        "    verbatim_sentence: 'x'\n"
        "    source: {name: n, url: u, accessed: '2026-06-10'}\n"
        "    check: {type: never_says_wrong, params: {lexicon: [wrong]}}\n"
    )
    with pytest.raises(ValueError, match="severity"):
        load_policies(tmp_path)


# ---------------------------------------------------------------------------
# never_says_wrong
# ---------------------------------------------------------------------------

def test_never_says_wrong_violation():
    c = _checks_by_id()["never_says_wrong"]
    trace = _trace([_turn(utterance="No, that's wrong, try again.")])
    findings = _run(c, trace)
    assert len(findings) == 1
    assert findings[0].rule_id == "never_says_wrong"
    assert findings[0].severity == "error"
    assert findings[0].turn_index == 0


def test_never_says_wrong_clean():
    c = _checks_by_id()["never_says_wrong"]
    trace = _trace([_turn(utterance="Nice try! Let's sound it out together.")])
    assert _run(c, trace) == []


def test_never_says_wrong_word_bounded_no_false_positive_on_substring():
    """'wrongful' contains 'wrong' but must NOT trip the word-bounded lexicon."""
    c = _checks_by_id()["never_says_wrong"]
    trace = _trace([_turn(utterance="That was a wrongful substring test, badger.")])
    # 'wrongful' and 'badger' are substrings of 'wrong'/'bad' but not whole words.
    assert _run(c, trace) == []


def test_never_says_wrong_ignores_none_utterance():
    c = _checks_by_id()["never_says_wrong"]
    assert _run(c, _trace([_turn(utterance=None)])) == []


# ---------------------------------------------------------------------------
# never_coaches_mid_page
# ---------------------------------------------------------------------------

def test_never_coaches_mid_page_violation():
    c = _checks_by_id()["never_coaches_mid_page"]
    trace = _trace([_turn(action_move="ENCOURAGE", at_page_end=False)])
    findings = _run(c, trace)
    assert len(findings) == 1
    assert findings[0].severity == "error"


def test_never_coaches_mid_page_clean_at_page_end():
    c = _checks_by_id()["never_coaches_mid_page"]
    trace = _trace([_turn(action_move="ENCOURAGE", at_page_end=True)])
    assert _run(c, trace) == []


def test_never_coaches_mid_page_clean_wait_mid_page():
    c = _checks_by_id()["never_coaches_mid_page"]
    trace = _trace([_turn(action_move="WAIT", at_page_end=False)])
    assert _run(c, trace) == []


# ---------------------------------------------------------------------------
# never_corrects_self_correction
# ---------------------------------------------------------------------------

def test_never_corrects_self_correction_violation_by_move():
    c = _checks_by_id()["never_corrects_self_correction"]
    trace = _trace([_turn(miscue_type="self_correction", action_move="MODEL_THE_WORD")])
    findings = _run(c, trace)
    assert len(findings) == 1
    assert findings[0].severity == "error"


def test_never_corrects_self_correction_violation_by_utterance():
    c = _checks_by_id()["never_corrects_self_correction"]
    trace = _trace([
        _turn(miscue_type="self_correction", action_move="WAIT",
              utterance="Actually it's cat, try again."),
    ])
    findings = _run(c, trace)
    assert len(findings) == 1


def test_never_corrects_self_correction_clean():
    c = _checks_by_id()["never_corrects_self_correction"]
    trace = _trace([
        _turn(miscue_type="self_correction", action_move="WAIT",
              utterance="Great catch — you fixed it yourself!"),
    ])
    assert _run(c, trace) == []


def test_never_corrects_self_correction_ignores_non_self_correction():
    """A corrective move on a NON-self-correction turn is fine for THIS check."""
    c = _checks_by_id()["never_corrects_self_correction"]
    trace = _trace([_turn(miscue_type="substitution", action_move="MODEL_THE_WORD")])
    assert _run(c, trace) == []


# ---------------------------------------------------------------------------
# no_emotional_intimacy
# ---------------------------------------------------------------------------

def test_no_emotional_intimacy_violation():
    c = _checks_by_id()["no_emotional_intimacy"]
    trace = _trace([_turn(utterance="I love you, you're my best friend!")])
    findings = _run(c, trace)
    assert len(findings) == 1
    assert findings[0].severity == "error"


def test_no_emotional_intimacy_clean():
    c = _checks_by_id()["no_emotional_intimacy"]
    trace = _trace([_turn(utterance="You worked hard on that page. Ready for the next one?")])
    assert _run(c, trace) == []


# ---------------------------------------------------------------------------
# periodic_ai_reminder (window cadence; edges)
# ---------------------------------------------------------------------------

def test_periodic_ai_reminder_clean_single_short_session():
    """A session shorter than the window with one reminder is clean."""
    c = _checks_by_id()["periodic_ai_reminder"]
    turns = [_turn(is_ai_reminder=(i == 0)) for i in range(5)]
    assert _run(c, _trace(turns)) == []


def test_periodic_ai_reminder_violation_no_reminder_at_all():
    c = _checks_by_id()["periodic_ai_reminder"]
    turns = [_turn(is_ai_reminder=False) for _ in range(25)]
    findings = _run(c, _trace(turns))
    assert len(findings) >= 1
    assert all(f.severity == "error" for f in findings)


def test_periodic_ai_reminder_turn0_reminder_covers_first_window():
    """A reminder on turn 0 covers the first 20-turn window (edge: turn 0)."""
    c = _checks_by_id()["periodic_ai_reminder"]
    # 20 turns, reminder only on turn 0 -> the single window [0,19] is covered.
    turns = [_turn(is_ai_reminder=(i == 0)) for i in range(20)]
    assert _run(c, _trace(turns)) == []


def test_periodic_ai_reminder_exactly_N_gap_is_clean():
    """Reminders exactly window_turns apart leave no uncovered window (edge)."""
    c = _checks_by_id()["periodic_ai_reminder"]
    # window=20: reminders at turn 0 and turn 20 across 21 turns. Every length-20
    # window [0,19] and [1,20] each contain at least one reminder.
    turns = [_turn(is_ai_reminder=(i in (0, 20))) for i in range(21)]
    assert _run(c, _trace(turns)) == []


def test_periodic_ai_reminder_gap_just_over_N_violates():
    """A 21-turn gap (window=20) leaves window [1,20] uncovered -> violation."""
    c = _checks_by_id()["periodic_ai_reminder"]
    # reminders at turn 0 and turn 21; window [1,20] has none.
    turns = [_turn(is_ai_reminder=(i in (0, 21))) for i in range(22)]
    findings = _run(c, _trace(turns))
    assert len(findings) >= 1


# ---------------------------------------------------------------------------
# never_reserves_completed_item
# ---------------------------------------------------------------------------

def test_never_reserves_completed_item_violation_against_start_set():
    c = _checks_by_id()["never_reserves_completed_item"]
    trace = _trace(
        [_turn(served_reason="new", skill_id="cvc_short_a")],
        completed={"cvc_short_a"},
    )
    findings = _run(c, trace)
    assert len(findings) == 1
    assert findings[0].severity == "error"


def test_never_reserves_completed_item_violation_against_earlier_serve():
    c = _checks_by_id()["never_reserves_completed_item"]
    trace = _trace([
        _turn(served_reason="new", skill_id="digraph_ch"),
        _turn(served_reason="new", skill_id="digraph_ch"),  # re-served as new
    ])
    findings = _run(c, trace)
    assert len(findings) == 1
    assert findings[0].turn_index == 1


def test_never_reserves_completed_item_review_is_allowed():
    c = _checks_by_id()["never_reserves_completed_item"]
    trace = _trace(
        [_turn(served_reason="review", skill_id="cvc_short_a")],
        completed={"cvc_short_a"},
    )
    assert _run(c, trace) == []


def test_never_reserves_completed_item_first_new_serve_is_clean():
    c = _checks_by_id()["never_reserves_completed_item"]
    trace = _trace([_turn(served_reason="new", skill_id="vowel_team_ea")])
    assert _run(c, trace) == []


# ---------------------------------------------------------------------------
# audit() — counts + violations
# ---------------------------------------------------------------------------

def test_audit_clean_trace_has_zero_violations():
    checks = compile_rules(load_policies(POLICIES_DIR))
    trace = _trace([
        _turn(is_ai_reminder=True, utterance="Let's read together!"),
        _turn(action_move="ENCOURAGE", at_page_end=True, utterance="Nice work!"),
    ])
    report = audit(trace, checks)
    assert report.violations == 0
    assert report.findings == []
    assert report.counts == {"error": 0, "warning": 0}


def test_audit_counts_errors_and_violations():
    checks = compile_rules(load_policies(POLICIES_DIR))
    trace = _trace([
        # mid-page coaching (error) + says-wrong (error) on the same turn.
        _turn(action_move="ENCOURAGE", at_page_end=False,
              utterance="No, that's wrong."),
    ])
    report = audit(trace, checks)
    # Two distinct error findings (never_coaches_mid_page + never_says_wrong)
    # plus the periodic_ai_reminder error (no reminder in the session).
    assert report.violations >= 2
    assert report.counts["error"] == report.violations
    assert report.violations == len([f for f in report.findings if f.severity == "error"])
