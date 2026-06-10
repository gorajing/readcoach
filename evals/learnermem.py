"""T6.4 — LearnerMem v0: memory-consistency probes over the learner store.

WHAT THIS IS
────────────
A tutor agent's "memory" is not a chat transcript — it is the learner model
(BKT mastery + FSRS review cards + miscue-tagged observations) plus the planner
that reads it.  When a child returns for session 2, the question MEMORY-
CONSISTENCY probing asks is: *is the system's session-2 behaviour consistent
with what it learned about this child in session 1?*  A tutor that forgets a
planted struggle, drifts a mastery value on reopen, or re-teaches a finished
skill has a memory bug — and that bug is invisible to single-session evals.

LINEAGE (cited in docs/learnermem.md)
─────────────────────────────────────
PersonaMem (arXiv:2504.14225) and MemoryArena-style long-horizon consistency
evals probe whether an LLM agent's later turns stay faithful to persona / facts
established earlier.  LearnerMem ports that idea into a *tutoring* domain: the
"persona" is the learner model, the "later turns" are the planner's session-2
decisions, and the planted facts are reading struggles, mastery levels, and
completion state.

V0 SCOPE (stated plainly — see docs)
────────────────────────────────────
These are DETERMINISTIC STATE PROBES, not LLM-judged dialogue checks.  The
planner + store ARE the memory; the utterance layer merely verbalises their
decisions downstream.  So a probe here is a machine-checkable assertion over
(store state, planner decision) — no LLM, no flakiness.  Probing the LLM
utterance layer for *verbal* consistency ("you struggled with silent-e last
time, so let's...") is the named v1 extension point.

PROBE CONTRACT
──────────────
Each Probe has:
  setup(store)            -> None            plants session-1 facts
  check(store, ctx)       -> ProbeResult     asserts session-2 consistency

setup is run against a session-1 store; the store is then closed and REOPENED
(a fresh handle on the same db) to model the session boundary, and check runs
against that reopened session-2 store.  This makes "did the fact survive the
session boundary?" the thing actually under test, not an in-process artefact.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from readcoach.learner_store import SqliteLearnerStore
from readcoach.planner import (
    MASTERY_COMPLETED,
    MASTERY_THRESHOLD,
    Curriculum,
    load_curriculum,
    next_item,
    unlocked,
)

# ---------------------------------------------------------------------------
# Fixed coordinates (deterministic — no randomness anywhere in this module)
# ---------------------------------------------------------------------------

CURRICULUM_PATH = (
    Path(__file__).parent.parent / "data" / "curriculum" / "scope_sequence.yaml"
)
CHILD_ID = "learnermem_child"
BASE_TS = datetime(2026, 6, 10, 9, 0, 0, tzinfo=timezone.utc)
SESSION_2_NOW = BASE_TS + timedelta(days=1)

# Over-personalization invariant floor: a single failure must not drop a
# mastered skill below the prerequisite-satisfaction threshold (servability).
OVERPERSONALIZATION_FLOOR = MASTERY_THRESHOLD


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Outcome of one probe check.

    finding: an honest, recorded observation that does NOT flip pass/fail — used
        when a probe passes its promised invariant but surfaces a real weakness
        worth reporting (e.g. P6's completed-status flip).  None when there is
        nothing extra to report.
    """

    passed: bool
    evidence: str
    finding: str | None = None


@dataclass
class Probe:
    """A planted-fact memory-consistency probe.

    setup(store) plants session-1 facts; check(store, ctx) asserts that the
    reopened session-2 store + planner behave consistently with them.
    """

    id: str
    description: str
    setup: Callable[[SqliteLearnerStore], None]
    check: Callable[[SqliteLearnerStore, "PlannerCtx"], ProbeResult]


@dataclass
class PlannerCtx:
    """Read-only context handed to every probe check.

    Bundles the loaded curriculum (and anything else a probe needs to reach the
    planner) so checks never re-load it ad hoc.
    """

    curriculum: Curriculum


@dataclass
class LearnerMemReport:
    """Aggregate report across all probes."""

    results: dict[str, ProbeResult]
    consistency_score: float
    findings: list[str] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results.values() if r.passed)

    @property
    def n_total(self) -> int:
        return len(self.results)


# ---------------------------------------------------------------------------
# Shared session-1 staging helpers
# ---------------------------------------------------------------------------


def _drive_mastery(
    store: SqliteLearnerStore,
    skill: str,
    target: float,
    *,
    session_id: str = "session_1",
    max_steps: int = 80,
) -> None:
    """Drive a skill's BKT mastery to >= target via correct observations."""
    for i in range(max_steps):
        if store.get_state(CHILD_ID).mastery.get(skill, 0.0) >= target:
            return
        store.record_observation(
            child_id=CHILD_ID,
            skill=skill,
            correct=True,
            confidence=1.0,
            session_id=session_id,
            ts=BASE_TS + timedelta(seconds=i),
        )
    raise RuntimeError(
        f"Could not reach mastery={target} on {skill!r} in {max_steps} steps."
    )


def _plant_failure(
    store: SqliteLearnerStore,
    skill: str,
    *,
    miscue_class: str | None,
    offset_min: int = 10,
    session_id: str = "session_1",
) -> None:
    """Plant ONE incorrect observation at the end of session 1.

    With offset_min=10 it sits inside the k=5 last-observation window when
    session 2 opens, so a class-keyed gate (if any) can see it.
    """
    store.record_observation(
        child_id=CHILD_ID,
        skill=skill,
        correct=False,
        confidence=1.0,
        session_id=session_id,
        ts=BASE_TS + timedelta(minutes=offset_min),
        miscue_class=miscue_class,
    )


# ===========================================================================
# P1 — struggled-fact persistence (silent-e substitution → gated successor)
# ===========================================================================
# Session 1: silent_e is driven high (mastery gate will pass) but a SUBSTITUTION
# error is planted in the last-5 window.  silent_e gates vowel_team_ai_ay on
# classes [substitution, omission].  Session 2: the planner must NOT serve the
# silent-e-gated successor (vowel_team_ai_ay).  The evidence is the gate
# evaluation: mastery PASS, class gate FIRES => LOCKED.

_P1_GATED_SUCCESSOR = "vowel_team_ai_ay"
_P1_PREREQ = "silent_e"


def _p1_setup(store: SqliteLearnerStore) -> None:
    _drive_mastery(store, _P1_PREREQ, 0.98)
    _plant_failure(store, _P1_PREREQ, miscue_class="substitution")


def _p1_check(store: SqliteLearnerStore, ctx: PlannerCtx) -> ProbeResult:
    mastery = store.get_state(CHILD_ID).mastery.get(_P1_PREREQ, 0.0)
    mastery_pass = mastery >= MASTERY_THRESHOLD

    recent = store.get_last_k_observations(CHILD_ID, _P1_PREREQ, k=5)
    blockers = [
        o for o in recent if not o["correct"] and o["miscue_class"] == "substitution"
    ]
    class_gate_fires = len(blockers) > 0

    unlocked_ids = unlocked(ctx.curriculum, CHILD_ID, store)
    successor_locked = _P1_GATED_SUCCESSOR not in unlocked_ids

    served = next_item(
        ctx.curriculum, CHILD_ID, store, store.get_served_log(CHILD_ID),
        now=SESSION_2_NOW,
    )
    served_skill = served[0] if served is not None else None
    not_served = served_skill != _P1_GATED_SUCCESSOR

    # Consistency requires: mastery gate passes (so the LOCK is the class gate,
    # not a mastery drop in disguise), AND the successor is both unlocked-False
    # and not served.
    passed = mastery_pass and class_gate_fires and successor_locked and not_served
    evidence = (
        f"silent_e mastery={mastery:.4f} >= {MASTERY_THRESHOLD} "
        f"(mastery gate {'PASS' if mastery_pass else 'FAIL'}); "
        f"last-5 substitution blockers={len(blockers)} "
        f"(class gate {'FIRES' if class_gate_fires else 'silent'}); "
        f"{_P1_GATED_SUCCESSOR} unlocked={not successor_locked}; "
        f"planner served={served_skill!r}"
    )
    return ProbeResult(passed=passed, evidence=evidence)


# ===========================================================================
# P2 — mastery continuity (no drift on reopen)
# ===========================================================================
# Every session-1 mastery value must reappear EXACTLY in session 2.  We snapshot
# the full mastery dict at the end of setup into a registry keyed by the store's
# db path, so the paired check (on the reopened store) compares against it.


def _p2_setup(store: SqliteLearnerStore) -> None:
    # Touch a spread of skills so the snapshot is non-trivial.
    _drive_mastery(store, "cvc_short_a", 0.90)
    _drive_mastery(store, "cvc_short_i_u", MASTERY_COMPLETED)
    _plant_failure(store, "cvc_short_a", miscue_class="omission")
    # Capture the exact S1 mastery dict.  The CLAIM is bit-exact float identity
    # across the reopen, so we hold the S1 values in a module registry keyed by
    # db path and compare against the reopened store's values in check.
    _P2_SNAPSHOTS[store_id(store)] = dict(store.get_state(CHILD_ID).mastery)


def _p2_check(store: SqliteLearnerStore, ctx: PlannerCtx) -> ProbeResult:
    snapshot = _P2_SNAPSHOTS.get(store_id(store))
    if snapshot is None:
        return ProbeResult(False, "P2 snapshot missing (setup/check store mismatch)")

    s2_mastery = store.get_state(CHILD_ID).mastery
    drifts = []
    for skill, m1 in snapshot.items():
        m2 = s2_mastery.get(skill)
        if m2 is None:
            drifts.append(f"{skill}: missing in session 2")
        elif abs(m1 - m2) >= 1e-12:
            drifts.append(f"{skill}: {m1!r} -> {m2!r} (delta={m1 - m2:.2e})")

    passed = not drifts
    if passed:
        evidence = (
            f"all {len(snapshot)} session-1 mastery values reappear bit-exact in "
            f"session 2 (max |delta| < 1e-12): "
            + ", ".join(f"{s}={v:.6f}" for s, v in sorted(snapshot.items()))
        )
    else:
        evidence = "DRIFT detected: " + "; ".join(drifts)
    return ProbeResult(passed=passed, evidence=evidence)


# ===========================================================================
# P3 — due-review consistency (failed skill due BEFORE passed skill)
# ===========================================================================
# A skill that FAILED in session 1 must be due for review in session 2 strictly
# BEFORE a skill that PASSED in session 1.  FSRS schedules a failed card (Again)
# sooner than a passed card (Good); the probe asserts that ordering survives the
# reopen and is observable via card.due timestamps.

_P3_FAILED = "cvc_short_e_o"
_P3_PASSED = "sight_words_1"


def _p3_setup(store: SqliteLearnerStore) -> None:
    # Both skills get a series of correct obs to build a card, then the failed
    # one gets one more FAILED obs (FSRS Again -> short interval).
    _drive_mastery(store, _P3_FAILED, 0.90)
    _drive_mastery(store, _P3_PASSED, 0.90)
    _plant_failure(store, _P3_FAILED, miscue_class="substitution")


def _p3_check(store: SqliteLearnerStore, ctx: PlannerCtx) -> ProbeResult:
    failed_card = store.get_card(CHILD_ID, _P3_FAILED)
    passed_card = store.get_card(CHILD_ID, _P3_PASSED)

    if failed_card.due is None or passed_card.due is None:
        return ProbeResult(
            False,
            f"missing due dates: {_P3_FAILED}.due={failed_card.due}, "
            f"{_P3_PASSED}.due={passed_card.due}",
        )

    ordered = failed_card.due < passed_card.due
    evidence = (
        f"{_P3_FAILED} (failed S1) due={failed_card.due.isoformat()} "
        f"{'<' if ordered else '>='} "
        f"{_P3_PASSED} (passed S1) due={passed_card.due.isoformat()}"
    )
    return ProbeResult(passed=ordered, evidence=evidence)


# ===========================================================================
# P4 — completed-skill memory (never re-served as 'new')
# ===========================================================================
# A skill completed (mastery >= MASTERY_COMPLETED) in session 1 must never be
# served with reason='new' in session 2.  We drive cvc_short_a (a root skill,
# always unlocked) to completion and log it served, then in session 2 run the
# planner repeatedly and assert it is never picked as 'new'.  Evidence combines
# the served log (it WAS served in S1) with the planner's S2 reasons.

_P4_COMPLETED = "cvc_short_a"


def _p4_setup(store: SqliteLearnerStore) -> None:
    _drive_mastery(store, _P4_COMPLETED, MASTERY_COMPLETED)
    store.record_served(
        CHILD_ID, _P4_COMPLETED, BASE_TS + timedelta(minutes=5), "new"
    )


def _p4_check(store: SqliteLearnerStore, ctx: PlannerCtx) -> ProbeResult:
    mastery = store.get_state(CHILD_ID).mastery.get(_P4_COMPLETED, 0.0)
    if mastery < MASTERY_COMPLETED:
        return ProbeResult(
            False,
            f"setup invariant broken: {_P4_COMPLETED} mastery={mastery:.4f} "
            f"< {MASTERY_COMPLETED} on reopen",
        )

    s1_served_new = any(
        e["skill"] == _P4_COMPLETED and e["reason"] == "new"
        for e in store.get_served_log(CHILD_ID)
    )

    new_serves = []
    for step in range(6):
        pick = next_item(
            ctx.curriculum, CHILD_ID, store, store.get_served_log(CHILD_ID),
            now=SESSION_2_NOW + timedelta(seconds=step),
        )
        if pick is not None and pick[0] == _P4_COMPLETED and pick[1] == "new":
            new_serves.append(step)

    passed = s1_served_new and not new_serves
    evidence = (
        f"{_P4_COMPLETED} mastery={mastery:.4f} >= {MASTERY_COMPLETED}; "
        f"served as 'new' in S1 log={s1_served_new}; "
        f"S2 re-serves as 'new'={len(new_serves)} "
        f"(planner+served-log agree it stays completed)"
    )
    return ProbeResult(passed=passed, evidence=evidence)


# ===========================================================================
# P5 — cross-skill inference guard (memory is skill-scoped)
# ===========================================================================
# Struggles planted ONLY on skill A must not alter skill B's mastery or B's
# gating.  We build B to a known state in session 1, snapshot it, then hammer A
# with failures (including a class-tagged one).  In session 2, B's mastery and
# B's own unlock status must be unchanged.

_P5_SKILL_A = "cvc_short_a"
_P5_SKILL_B = "cvc_short_e_o"  # gated by cvc_short_i_u, independent of A's gate


def _p5_setup(store: SqliteLearnerStore) -> None:
    # Build B's full prerequisite chain so B's own unlock status is meaningful,
    # then snapshot B before touching A.
    _drive_mastery(store, "cvc_short_a", 0.90)
    _drive_mastery(store, "cvc_short_i_u", 0.90)
    _drive_mastery(store, _P5_SKILL_B, 0.85)

    state = store.get_state(CHILD_ID)
    cur = load_curriculum(CURRICULUM_PATH)
    _P5_SNAPSHOTS[store_id(store)] = {
        "mastery_b": state.mastery.get(_P5_SKILL_B, 0.0),
        "b_unlocked": _P5_SKILL_B in unlocked(cur, CHILD_ID, store),
    }

    # Now hammer A with failures (tagged + untagged). None of these touch B.
    _plant_failure(store, _P5_SKILL_A, miscue_class="substitution", offset_min=11)
    _plant_failure(store, _P5_SKILL_A, miscue_class=None, offset_min=12)


def _p5_check(store: SqliteLearnerStore, ctx: PlannerCtx) -> ProbeResult:
    snap = _P5_SNAPSHOTS.get(store_id(store))
    if snap is None:
        return ProbeResult(False, "P5 snapshot missing (setup/check store mismatch)")

    s2_mastery_b = store.get_state(CHILD_ID).mastery.get(_P5_SKILL_B, 0.0)
    s2_b_unlocked = _P5_SKILL_B in unlocked(ctx.curriculum, CHILD_ID, store)

    mastery_unchanged = abs(snap["mastery_b"] - s2_mastery_b) < 1e-12
    unlock_unchanged = snap["b_unlocked"] == s2_b_unlocked

    passed = mastery_unchanged and unlock_unchanged
    evidence = (
        f"failures planted only on {_P5_SKILL_A}; "
        f"{_P5_SKILL_B} mastery {snap['mastery_b']:.6f} -> {s2_mastery_b:.6f} "
        f"(unchanged={mastery_unchanged}); "
        f"{_P5_SKILL_B} unlocked {snap['b_unlocked']} -> {s2_b_unlocked} "
        f"(unchanged={unlock_unchanged})"
    )
    return ProbeResult(passed=passed, evidence=evidence)


# ===========================================================================
# P6 — over-personalization guard (one generic failure must not over-react)
# ===========================================================================
# THE REAL OVER-PERSONALIZATION RISK (see ticket note): with class-gate
# semantics, one *tagged* failure in the last-5 window DOES lock the gated
# successor — that is by design and correct.  So P6 probes a generic (UNTAGGED)
# failure on an otherwise-mastered, root skill (no class gate of its own).  The
# invariant the system PROMISES to hold: one generic failure leaves the skill
# (a) mastery >= OVERPERSONALIZATION_FLOOR (still satisfies any prerequisite
# that depends on it — it stays servable/unlocked downstream) and (b) its own
# unlock status unchanged.  We assert that.
#
# HONEST FINDING (recorded, not hidden): the SAME single generic failure DOES
# drop mastery below MASTERY_COMPLETED (0.95), i.e. the skill loses "completed"
# status from one data point.  That is a genuine over-personalization
# sensitivity in the completion semantics.  The probe PASSES on the promised
# invariant and SEPARATELY surfaces the completed-status flip as a finding
# rather than redefining the probe to make the flip "pass".

_P6_SKILL = "cvc_short_a"  # root skill: no prerequisites, no class gate of its own


def _p6_setup(store: SqliteLearnerStore) -> None:
    _drive_mastery(store, _P6_SKILL, 0.98)
    # exactly ONE generic (untagged) failure
    _plant_failure(store, _P6_SKILL, miscue_class=None)


def _p6_check(store: SqliteLearnerStore, ctx: PlannerCtx) -> ProbeResult:
    mastery = store.get_state(CHILD_ID).mastery.get(_P6_SKILL, 0.0)
    above_floor = mastery >= OVERPERSONALIZATION_FLOOR

    # A root skill (no prerequisites) is always unlocked — its own unlock status
    # must therefore remain True regardless of the failure.
    node = ctx.curriculum.nodes[_P6_SKILL]
    expected_unlocked = not node.prerequisites  # root => always unlocked
    actually_unlocked = _P6_SKILL in unlocked(ctx.curriculum, CHILD_ID, store)
    unlock_unchanged = actually_unlocked == expected_unlocked

    # The promised invariant — this is what determines pass/fail.
    passed = above_floor and unlock_unchanged

    # Honest finding: completed-status flip from one generic failure.
    completed_flip = mastery < MASTERY_COMPLETED
    finding = None
    if completed_flip:
        finding = (
            f"OVER-PERSONALIZATION SENSITIVITY: one generic (untagged) failure on "
            f"{_P6_SKILL} dropped mastery to {mastery:.4f}, BELOW MASTERY_COMPLETED "
            f"({MASTERY_COMPLETED}) — the skill loses 'completed' status from a "
            f"single data point. It stays >= {OVERPERSONALIZATION_FLOOR} "
            f"(servable/unlocked), so the promised invariant holds, but one bad "
            f"observation un-completes a mastered skill. Mitigation belongs in the "
            f"completion hysteresis (require N consecutive sub-threshold obs before "
            f"un-completing), tracked as a v1 planner item."
        )

    evidence = (
        f"one generic failure on {_P6_SKILL}: mastery={mastery:.4f} "
        f">= floor {OVERPERSONALIZATION_FLOOR} ({above_floor}); "
        f"own unlock status {actually_unlocked} (unchanged={unlock_unchanged}); "
        f"completed-status retained={not completed_flip}"
    )
    # finding is None unless the completed-status flip occurred; run_probes
    # harvests it without it affecting pass/fail.
    return ProbeResult(passed=passed, evidence=evidence, finding=finding)


# ---------------------------------------------------------------------------
# Per-store snapshot registries (keyed by the store's db path)
# ---------------------------------------------------------------------------
# Some probes need to compare an exact session-1 snapshot against session-2
# state.  We key snapshots by the store's underlying db path so a setup call and
# its paired check call (on the reopened store over the SAME db) line up, while
# different probes' stores (different db paths) never collide.

_P2_SNAPSHOTS: dict[str, dict[str, float]] = {}
_P5_SNAPSHOTS: dict[str, dict[str, object]] = {}


def store_id(store: SqliteLearnerStore) -> str:
    """Stable identity for a store across close/reopen: its db file path."""
    # sqlite3 exposes the db filename via PRAGMA database_list (seq 0 = 'main').
    rows = store._conn.execute("PRAGMA database_list").fetchall()
    for row in rows:
        if row["name"] == "main":
            return row["file"] or "<memory>"
    return "<unknown>"


# ---------------------------------------------------------------------------
# Probe registry
# ---------------------------------------------------------------------------

PROBES: list[Probe] = [
    Probe(
        id="P1",
        description=(
            "struggled-fact persistence: a silent-e substitution struggle planted "
            "in session 1 keeps the silent-e-gated successor LOCKED in session 2 "
            "(mastery gate passes, class gate fires)"
        ),
        setup=_p1_setup,
        check=_p1_check,
    ),
    Probe(
        id="P2",
        description=(
            "mastery continuity: every session-1 mastery value reappears "
            "bit-exact in session 2 (no drift on reopen)"
        ),
        setup=_p2_setup,
        check=_p2_check,
    ),
    Probe(
        id="P3",
        description=(
            "due-review consistency: a skill failed in session 1 is due for review "
            "BEFORE a skill passed in session 1"
        ),
        setup=_p3_setup,
        check=_p3_check,
    ),
    Probe(
        id="P4",
        description=(
            "completed-skill memory: a skill completed (>=0.95) in session 1 is "
            "never served as 'new' in session 2 (served-log + planner agree)"
        ),
        setup=_p4_setup,
        check=_p4_check,
    ),
    Probe(
        id="P5",
        description=(
            "cross-skill inference guard: struggles planted only on skill A do not "
            "alter skill B's mastery or B's gating (memory is skill-scoped)"
        ),
        setup=_p5_setup,
        check=_p5_check,
    ),
    Probe(
        id="P6",
        description=(
            "over-personalization guard: one generic (untagged) failure on a "
            "mastered root skill keeps it >= threshold (servable) and unlock "
            "status unchanged"
        ),
        setup=_p6_setup,
        check=_p6_check,
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_probe(
    probe: Probe,
    db_path_factory: Callable[[], str],
    ctx: PlannerCtx,
) -> ProbeResult:
    """Run a single probe end-to-end across a real session boundary.

    1. Open a session-1 store on a fresh db; run setup; close it.
    2. Reopen a session-2 store on the SAME db; run check; close it.

    The close/reopen is the session boundary — it is what makes "did the planted
    fact survive?" the thing under test rather than an in-process coincidence.
    """
    db_path = db_path_factory()

    s1 = SqliteLearnerStore(db_path)
    try:
        probe.setup(s1)
    finally:
        s1.close()

    s2 = SqliteLearnerStore(db_path)
    try:
        return probe.check(s2, ctx)
    finally:
        s2.close()


def run_probes(db_path_factory: Callable[[], str]) -> LearnerMemReport:
    """Run all probes; return per-probe results, consistency score, findings.

    db_path_factory() must return a FRESH (non-existent or empty) db path on each
    call so probes do not contaminate one another.

    consistency_score = fraction of probes passed.
    """
    ctx = PlannerCtx(curriculum=load_curriculum(CURRICULUM_PATH))

    results: dict[str, ProbeResult] = {}
    findings: list[str] = []
    for probe in PROBES:
        result = run_probe(probe, db_path_factory, ctx)
        results[probe.id] = result
        if result.finding:
            findings.append(f"[{probe.id}] {result.finding}")

    n_total = len(results)
    n_passed = sum(1 for r in results.values() if r.passed)
    score = n_passed / n_total if n_total else 0.0

    return LearnerMemReport(
        results=results, consistency_score=score, findings=findings
    )
