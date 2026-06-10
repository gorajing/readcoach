"""T4.3 — invariants metric wired into the gate (red first).

`invariants_metrics(trace)` computes the REAL `invariants.violations` value the
gate already gates (min/0).  This test proves a violating trace, run through the
ACTUAL gate rule table (imported from scripts/gate.py), drives compare() to
exit 1 — i.e. the gate carries the real computed value, not a hard-coded 0.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from evals.harness import EvalReport, compare
from readcoach.invariants import invariants_metrics
from readcoach.trace import SessionTrace, TurnRecord

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_gate_rules():
    """Import the REAL GATE_RULES table from scripts/gate.py."""
    spec = importlib.util.spec_from_file_location(
        "gate_mod", _PROJECT_ROOT / "scripts" / "gate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.GATE_RULES


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


def _trace(turns) -> SessionTrace:
    return SessionTrace(
        child_id="kid",
        policy_version="1.0.0",
        completed_skills_at_start=(),
        turns=tuple(turns),
    )


# A baseline (prev) report with zero violations + the miscue metrics the gate
# table references, so the ONLY thing under test here is invariants.violations.
def _full_metrics(violations: int) -> dict:
    f1_block = {"f1": 1.0}
    return {
        "miscue": {
            "substitution": f1_block,
            "omission": f1_block,
            "insertion": f1_block,
            "self_correction": f1_block,
            "hesitation": f1_block,
            "fp_per_100_correct_words": 0.0,
        },
        "invariants": {"violations": violations},
        "latency": {"decision_ms_p95": None},
    }


def test_invariants_metrics_clean_trace_is_zero():
    trace = _trace([
        _turn(is_ai_reminder=True, utterance="Let's read together!"),
        _turn(at_page_end=True, action_move="ENCOURAGE", utterance="Nice work!"),
    ])
    m = invariants_metrics(trace)
    assert m == {"violations": 0}


def test_invariants_metrics_counts_real_violations():
    trace = _trace([
        _turn(action_move="ENCOURAGE", at_page_end=False, utterance="No, that's wrong."),
    ])
    m = invariants_metrics(trace)
    # mid-page coaching + says-wrong + no periodic reminder -> >= 2 errors.
    assert m["violations"] >= 2


def test_violating_trace_makes_real_gate_exit_1():
    """End-to-end: a violating trace -> invariants.violations > 0 -> gate exit 1."""
    gate_rules = _load_gate_rules()
    violating = _trace([
        _turn(action_move="ENCOURAGE", at_page_end=False, utterance="No, that's wrong."),
    ])
    viol = invariants_metrics(violating)["violations"]
    assert viol > 0

    prev = EvalReport(version="prev", metrics=_full_metrics(0), metadata={})
    new = EvalReport(version="new", metrics=_full_metrics(viol), metadata={})

    # Drop the report-only latency rule (None values -> skipped by gate.py main;
    # here we just need the gating rules).
    active = [r for r in gate_rules if not r.report_only]
    result = compare(prev, new, active)
    assert result.exit_code == 1
    assert any("invariants.violations" in b for b in result.breaches)


def test_clean_trace_passes_real_gate():
    gate_rules = _load_gate_rules()
    clean = _trace([
        _turn(is_ai_reminder=True, utterance="Let's read!"),
        _turn(at_page_end=True, action_move="ENCOURAGE", utterance="Great job!"),
    ])
    viol = invariants_metrics(clean)["violations"]
    assert viol == 0

    prev = EvalReport(version="prev", metrics=_full_metrics(0), metadata={})
    new = EvalReport(version="new", metrics=_full_metrics(viol), metadata={})
    active = [r for r in gate_rules if not r.report_only]
    result = compare(prev, new, active)
    assert result.exit_code == 0
