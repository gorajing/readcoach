"""T5.4 — pre-registered A/B + flywheel + sensitivity (TDD).

Pins the deterministic dev-split slice:

  * VERSION CONTRACTS:
      - v1 is STATE-BLIND — the learner store is never consulted (mastery is the
        empty dict every turn; proven via a store-spy AND by construction since v1
        takes no store).
      - v2 is MASTERY-CONDITIONED — it produces planner-driven targeted NEXT_ITEM
        serves that v1 (state-blind) cannot.
      - v3 is the ALWAYS-INTERVENE villain — it NEVER WAITs on a mid-page miscue
        (every mid-page miscue is met with MODEL_THE_WORD).
  * WAIT-RATE BAND RULES behave on synthetic reports (floor 0.35, ceiling 0.50).
  * v3's report -> compare() -> exit 1 with the invariants breach NAMED (receipt #2).
  * HOLDOUT-PATH REFUSAL (the freeze stays auditable).
  * promote_failure IDEMPOTENCY (re-promoting the same batch leaves the golden
    unchanged).
  * AGGREGATION MATH (wait_rate denominator = decision turns; violation sums).

The runner is imported in-process (it is a script, not a package) so the tests
share the exact replay machinery ``scripts/run_ab.py`` uses.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from evals.harness import EvalReport, compare  # noqa: E402
from readcoach.learner_store import InMemoryLearnerStore  # noqa: E402
from readcoach.planner import load_curriculum  # noqa: E402
from readcoach.trace import SessionTrace  # noqa: E402
from readcoach.tutor_versions import is_decision_turn, run_session  # noqa: E402

_CURRICULUM_PATH = _PROJECT_ROOT / "data" / "curriculum" / "scope_sequence.yaml"
_DEV_FILE = _PROJECT_ROOT / "evals" / "golden" / "persona_sessions_dev.jsonl"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_ab", _PROJECT_ROOT / "scripts" / "run_ab.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_ab"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


@pytest.fixture(scope="module")
def curriculum():
    return load_curriculum(_CURRICULUM_PATH)


@pytest.fixture(scope="module")
def dev_sessions():
    return [
        json.loads(line)
        for line in _DEV_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# A store spy: every get_state call is logged, and mastery is forced non-empty.
# If v1 EVER consulted it, the spy would see a get_state call AND the policy
# would have non-empty gaps -> a NEXT_ITEM with a skill_id.  v1 produces neither.
# ---------------------------------------------------------------------------


class _SpyStore(InMemoryLearnerStore):
    """An InMemoryLearnerStore that records get_state calls (state-blindness spy)."""

    def __init__(self) -> None:
        super().__init__()
        self.get_state_calls = 0

    def get_state(self, child_id: str):  # noqa: ANN001
        self.get_state_calls += 1
        return super().get_state(child_id)


def _one_session_with_struggle(dev_sessions: list[dict]) -> dict:
    """Pick a dev session that has a page with struggle (drives NEXT_ITEM in v2)."""
    for item in dev_sessions:
        if any(ev["page_had_struggle"] for ev in item["events"]):
            return item
    raise AssertionError("no dev session with a struggling page found")


# ---------------------------------------------------------------------------
# VERSION CONTRACT — v1 is state-blind (mastery never consulted)
# ---------------------------------------------------------------------------


def test_v1_is_state_blind_never_emits_next_item(dev_sessions):
    """v1 must NEVER emit the NEXT_ITEM move on ANY dev session.

    NEXT_ITEM is the page-end move the policy emits ONLY when the learner state has
    a gap (``R-PE-NEXT-ITEM`` requires ``page_had_struggle`` AND a non-empty
    ``gaps()``).  A state-blind v1 feeds an EMPTY learner state, so no gap can ever
    exist -> the move cannot appear.  Asserting on the MOVE (not skill_id, which
    the memory-free path never stamps) is what makes this catch a leak of mastery
    into v1's decision context.
    """
    for item in dev_sessions:
        trace = run_session(item, "v1")
        assert all(t.action_move != "NEXT_ITEM" for t in trace.turns), (
            f"v1 emitted NEXT_ITEM on session {item['id']} — that move is only "
            f"reachable from a NON-empty learner state, so v1 is not state-blind"
        )


def test_v1_ignores_an_injected_store_by_construction(dev_sessions):
    """Passing a spy store to v1 leaves it UNTOUCHED — v1 builds its own blind state.

    v1's signature takes no store; even when one is handed to run_session, the v1
    path never reads it.  The spy proves get_state was not called via the store.
    """
    item = _one_session_with_struggle(dev_sessions)
    spy = _SpyStore()
    # Seed the spy with strong, non-empty mastery so that IF v1 consulted it, the
    # behaviour would visibly change (gaps present -> NEXT_ITEM).
    from datetime import datetime, timezone

    spy.record_observation(
        child_id=item["persona_id"], skill="cvc_short_a", correct=False,
        confidence=0.9, session_id="seed", ts=datetime(2026, 6, 10, tzinfo=timezone.utc),
        miscue_class="substitution",
    )
    calls_before = spy.get_state_calls
    trace = run_session(item, "v1", store=spy)
    # The spy was never consulted by the v1 path.
    assert spy.get_state_calls == calls_before
    # And v1 still emitted no NEXT_ITEM move (blind by construction — the seeded
    # mastery in the spy never reached decide()).
    assert all(t.action_move != "NEXT_ITEM" for t in trace.turns)


# ---------------------------------------------------------------------------
# VERSION CONTRACT — v2 is mastery-conditioned (consults the store, targets gaps)
# ---------------------------------------------------------------------------


def test_v2_consults_the_store_and_targets_next_items(dev_sessions, curriculum):
    """v2 must consult the store (spy sees get_state) and emit targeted NEXT_ITEMs."""
    item = _one_session_with_struggle(dev_sessions)
    spy = _SpyStore()
    trace = run_session(item, "v2", curriculum=curriculum, store=spy)
    assert spy.get_state_calls > 0, "v2 must read live learner state from the store"
    targeted = [
        t for t in trace.turns if t.action_move == "NEXT_ITEM" and t.skill_id is not None
    ]
    assert targeted, "v2 should serve at least one planner-targeted NEXT_ITEM"


def test_v2_requires_a_curriculum(dev_sessions):
    """v2 without a curriculum fails loud (it needs the band→skill anchor)."""
    item = dev_sessions[0]
    with pytest.raises(ValueError, match="curriculum"):
        run_session(item, "v2")


# ---------------------------------------------------------------------------
# VERSION CONTRACT — v3 villain never WAITs on a mid-page miscue
# ---------------------------------------------------------------------------


def test_v3_never_waits_on_a_mid_page_miscue(dev_sessions):
    """v3 (always-intervene) must model EVERY mid-page miscue — never WAIT."""
    waited_on_miscue = 0
    modeled_on_miscue = 0
    for item in dev_sessions:
        trace = run_session(item, "v3")
        for t in trace.turns:
            if t.miscue_type is not None and not t.at_page_end:
                assert t.action_move != "WAIT", (
                    f"v3 WAITed on a mid-page {t.miscue_type} — the villain must "
                    f"always intervene"
                )
                if t.action_move == "MODEL_THE_WORD":
                    modeled_on_miscue += 1
                waited_on_miscue += 0
    assert modeled_on_miscue > 0, "v3 should have modeled at least one mid-page miscue"


def test_v3_models_self_corrections_the_real_violation(dev_sessions):
    """v3 modeling a self_correction is the never_corrects_self_correction breach."""
    found = False
    for item in dev_sessions:
        trace = run_session(item, "v3")
        for t in trace.turns:
            if (
                t.miscue_type == "self_correction"
                and not t.at_page_end
                and t.action_move == "MODEL_THE_WORD"
            ):
                found = True
    assert found, (
        "v3 must hit at least one self_correction with MODEL_THE_WORD — that is the "
        "honest action-level invariant violation the gate catches"
    )


# ---------------------------------------------------------------------------
# WAIT-RATE BAND RULES behave on synthetic reports
# ---------------------------------------------------------------------------


def _metrics(wait_rate: float, violations: int = 0, serve: int = 0) -> dict:
    return {
        "invariants": {"violations": violations},
        "wait_rate": wait_rate,
        "wait_rate_ceiling": wait_rate,
        "serve_violations": serve,
    }


def test_wait_rate_band_passes_inside_band(runner):
    prev = EvalReport("prev", _metrics(0.42), {})
    new = EvalReport("new", _metrics(0.42), {})
    result = compare(prev, new, runner.TUTOR_GATE_RULES)
    assert result.passed and result.exit_code == 0


def test_wait_rate_band_blocks_below_floor(runner):
    prev = EvalReport("prev", _metrics(0.42), {})
    new = EvalReport("new", _metrics(0.10), {})  # below 0.35 floor
    result = compare(prev, new, runner.TUTOR_GATE_RULES)
    assert not result.passed
    assert any("wait_rate" in b for b in result.breaches)


def test_wait_rate_band_blocks_above_ceiling(runner):
    prev = EvalReport("prev", _metrics(0.42), {})
    new = EvalReport("new", _metrics(0.80), {})  # above 0.50 ceiling
    result = compare(prev, new, runner.TUTOR_GATE_RULES)
    assert not result.passed
    assert any("wait_rate_ceiling" in b for b in result.breaches)


# ---------------------------------------------------------------------------
# v3 report -> compare -> exit 1 with the invariants breach named (receipt #2)
# ---------------------------------------------------------------------------


def test_v3_report_is_blocked_with_invariants_breach_named(runner):
    """Receipt #2: a v3-shaped report (violations > 0) is BLOCKED, breach named."""
    v2 = EvalReport("tutor-v2", _metrics(0.42, violations=0), {})
    v3 = EvalReport("tutor-v3", _metrics(0.0, violations=30), {})
    result = compare(v2, v3, runner.TUTOR_GATE_RULES)
    assert result.exit_code == 1, "v3 must regress -> exit 1"
    assert not result.passed
    assert any("invariants.violations" in b for b in result.breaches), (
        "the gate must NAME the invariants.violations breach (receipt #2)"
    )


def test_clean_v2_passes_the_gate_vs_v1(runner):
    """The primary A/B claim: a clean v2 (0 violations, in-band wait) PASSES vs v1."""
    v1 = EvalReport("tutor-v1", _metrics(0.40, violations=0), {})
    v2 = EvalReport("tutor-v2", _metrics(0.40, violations=0), {})
    result = compare(v1, v2, runner.TUTOR_GATE_RULES)
    assert result.passed and result.exit_code == 0


# ---------------------------------------------------------------------------
# Holdout-path refusal (the freeze stays auditable)
# ---------------------------------------------------------------------------


def test_runner_refuses_a_holdout_path(runner):
    holdout = _PROJECT_ROOT / "evals" / "golden" / "persona_sessions_holdout.jsonl"
    with pytest.raises(ValueError, match="holdout"):
        runner._assert_not_holdout(holdout)


def test_runner_accepts_the_dev_path(runner):
    # Must NOT raise.
    runner._assert_not_holdout(_DEV_FILE)


def test_load_dev_sessions_refuses_holdout(runner):
    holdout = _PROJECT_ROOT / "evals" / "golden" / "persona_sessions_holdout.jsonl"
    with pytest.raises(ValueError, match="holdout"):
        runner._load_dev_sessions(holdout)


# ---------------------------------------------------------------------------
# promote_failure idempotency
# ---------------------------------------------------------------------------


def test_promote_batch_is_idempotent(runner, tmp_path):
    golden = tmp_path / "tutor_failures.jsonl"
    batch = [
        {"trace_id": "v3__a__turn1", "rule_violated": "never_corrects_self_correction"},
        {"trace_id": "v3__b__turn2", "rule_violated": "never_corrects_self_correction"},
    ]
    after_first = runner._promote_batch(batch, golden)
    assert after_first == 2
    content_after_first = golden.read_text(encoding="utf-8")
    # Re-promote the SAME batch -> unchanged.
    after_second = runner._promote_batch(batch, golden)
    assert after_second == 2
    assert golden.read_text(encoding="utf-8") == content_after_first


def test_v3_violation_trace_ids_are_unique(runner, dev_sessions):
    """Promotable v3 traces carry session-unique ids (no collision across 49)."""
    v3_traces = [run_session(item, "v3") for item in dev_sessions]
    dicts = runner._v3_violation_traces(dev_sessions, v3_traces)
    ids = [d["trace_id"] for d in dicts]
    assert len(ids) == len(set(ids)), "trace_ids must be unique (session id in the id)"
    assert dicts, "v3 must produce at least one promotable violation"


# ---------------------------------------------------------------------------
# Aggregation math
# ---------------------------------------------------------------------------


def test_is_decision_turn_excludes_clean_mid_page():
    """Only miscue turns and page-end turns are decision turns (band denominator)."""
    from readcoach.trace import TurnRecord

    clean_mid = TurnRecord(
        turn_index=0, at_page_end=False, miscue_type=None, action_move="WAIT",
        hint_level=None, served_reason=None, utterance=None, is_ai_reminder=False,
    )
    miscue_mid = TurnRecord(
        turn_index=1, at_page_end=False, miscue_type="substitution",
        action_move="WAIT", hint_level=None, served_reason=None, utterance=None,
        is_ai_reminder=False,
    )
    page_end = TurnRecord(
        turn_index=2, at_page_end=True, miscue_type=None, action_move="ENCOURAGE",
        hint_level=None, served_reason=None, utterance=None, is_ai_reminder=False,
    )
    assert not is_decision_turn(clean_mid)
    assert is_decision_turn(miscue_mid)
    assert is_decision_turn(page_end)


def test_aggregate_wait_rate_uses_decision_denominator(runner, dev_sessions):
    """wait_rate = (# WAIT among decision turns) / (# decision turns), pooled."""
    traces = [run_session(item, "v1") for item in dev_sessions]
    metrics = runner._aggregate(traces)

    # Recompute independently and assert equality.
    n_dec = 0
    n_wait = 0
    for tr in traces:
        for t in tr.turns:
            if is_decision_turn(t):
                n_dec += 1
                if t.action_move == "WAIT":
                    n_wait += 1
    assert metrics["n_decision_turns"] == n_dec
    assert metrics["wait_rate"] == pytest.approx(n_wait / n_dec)
    # In-band on the frozen dev split.
    assert 0.35 <= metrics["wait_rate"] <= 0.50


def test_aggregate_v1_v2_clean_v3_dirty(runner, dev_sessions, curriculum):
    """v1/v2 carry zero invariant violations; v3 carries > 0 (the villain)."""
    v1 = runner._aggregate([run_session(it, "v1") for it in dev_sessions])
    v3 = runner._aggregate([run_session(it, "v3") for it in dev_sessions])
    v2_traces = []
    stores: dict[str, InMemoryLearnerStore] = {}
    for it in dev_sessions:
        store = stores.setdefault(it["persona_id"], InMemoryLearnerStore())
        v2_traces.append(run_session(it, "v2", curriculum=curriculum, store=store))
    v2 = runner._aggregate(v2_traces)

    assert v1["invariants"]["violations"] == 0
    assert v2["invariants"]["violations"] == 0
    assert v3["invariants"]["violations"] > 0
    # v2's mastery-conditioning value-add: targeted next items v1 cannot produce.
    assert v2["targeted_next_items"] > 0
    assert v1["targeted_next_items"] == 0


def test_replay_is_deterministic(dev_sessions, curriculum):
    """Replaying the same session twice yields identical traces (no RNG)."""
    item = dev_sessions[0]
    a = run_session(item, "v1")
    b = run_session(item, "v1")
    assert _trace_moves(a) == _trace_moves(b)
    s1 = InMemoryLearnerStore()
    s2 = InMemoryLearnerStore()
    c = run_session(item, "v2", curriculum=curriculum, store=s1)
    d = run_session(item, "v2", curriculum=curriculum, store=s2)
    assert _trace_moves(c) == _trace_moves(d)


def _trace_moves(trace: SessionTrace) -> list:
    return [(t.turn_index, t.action_move, t.served_reason, t.skill_id) for t in trace.turns]
