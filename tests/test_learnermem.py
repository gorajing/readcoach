"""T6.4 — LearnerMem v0 probe tests.

Each probe is exercised TWICE:
  * a PASS case: setup plants the fact, the reopened store + planner behave
    consistently, the probe returns passed=True.
  * a SABOTAGE case: between session 1 and session 2 we corrupt the store (or
    plant a contradictory fact) so the probe MUST return passed=False.  A probe
    that cannot fail is decoration — the sabotage case proves it has teeth.

Plus: report shape and score arithmetic.

All probes are deterministic state checks (no LLM, no network, no randomness),
so every assertion below is exact.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from readcoach.learner_store import SqliteLearnerStore

import evals.learnermem as lm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_factory(tmp_path):
    """Return a factory producing fresh, monotonically-named db paths."""
    counter = {"n": 0}

    def factory() -> str:
        counter["n"] += 1
        return str(tmp_path / f"probe_{counter['n']}.db")

    return factory


def _setup_then_reopen(probe, db_path):
    """Run a probe's setup on a session-1 store, close, return reopened store."""
    s1 = SqliteLearnerStore(db_path)
    probe.setup(s1)
    s1.close()
    return SqliteLearnerStore(db_path)


def _ctx():
    return lm.PlannerCtx(curriculum=lm.load_curriculum(lm.CURRICULUM_PATH))


def _probe(probe_id: str):
    return next(p for p in lm.PROBES if p.id == probe_id)


# ---------------------------------------------------------------------------
# P1 — struggled-fact persistence
# ---------------------------------------------------------------------------


class TestP1StruggledFactPersistence:
    def test_pass(self, tmp_path):
        db = str(tmp_path / "p1.db")
        store = _setup_then_reopen(_probe("P1"), db)
        result = _probe("P1").check(store, _ctx())
        store.close()
        assert result.passed, result.evidence
        # The lock must be the CLASS gate, not a mastery drop in disguise.
        assert "mastery gate PASS" in result.evidence
        assert "class gate FIRES" in result.evidence

    def test_sabotage_forget_the_struggle(self, tmp_path):
        # Sabotage: between sessions, DELETE the planted substitution failure so
        # the system "forgets" the struggle.  The class gate then no longer
        # fires, the successor unlocks, and the probe must FAIL.
        db = str(tmp_path / "p1_sab.db")
        s1 = SqliteLearnerStore(db)
        _probe("P1").setup(s1)
        s1.close()

        # Corrupt: remove the incorrect silent_e observations.
        conn = sqlite3.connect(db)
        conn.execute(
            "DELETE FROM observations WHERE skill='silent_e' AND correct=0"
        )
        conn.commit()
        conn.close()

        store = SqliteLearnerStore(db)
        result = _probe("P1").check(store, _ctx())
        store.close()
        assert not result.passed, (
            "P1 must FAIL when the planted struggle is forgotten: " + result.evidence
        )


# ---------------------------------------------------------------------------
# P2 — mastery continuity
# ---------------------------------------------------------------------------


class TestP2MasteryContinuity:
    def test_pass(self, tmp_path):
        db = str(tmp_path / "p2.db")
        store = _setup_then_reopen(_probe("P2"), db)
        result = _probe("P2").check(store, _ctx())
        store.close()
        assert result.passed, result.evidence
        assert "bit-exact" in result.evidence

    def test_sabotage_drift_a_mastery_value(self, tmp_path):
        # Sabotage: nudge one mastery value on reopen.  Continuity is violated,
        # the probe must FAIL and name the drifted skill.
        db = str(tmp_path / "p2_sab.db")
        s1 = SqliteLearnerStore(db)
        _probe("P2").setup(s1)
        s1.close()

        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE mastery SET p_mastery = p_mastery + 0.05 WHERE skill='cvc_short_a'"
        )
        conn.commit()
        conn.close()

        store = SqliteLearnerStore(db)
        result = _probe("P2").check(store, _ctx())
        store.close()
        assert not result.passed, "P2 must FAIL on mastery drift: " + result.evidence
        assert "DRIFT" in result.evidence
        assert "cvc_short_a" in result.evidence


# ---------------------------------------------------------------------------
# P3 — due-review consistency
# ---------------------------------------------------------------------------


class TestP3DueReviewConsistency:
    def test_pass(self, tmp_path):
        db = str(tmp_path / "p3.db")
        store = _setup_then_reopen(_probe("P3"), db)
        result = _probe("P3").check(store, _ctx())
        store.close()
        assert result.passed, result.evidence
        assert "<" in result.evidence  # failed.due < passed.due

    def test_sabotage_swap_due_dates(self, tmp_path):
        # Sabotage: overwrite the FAILED skill's card so its due date is far in
        # the future (later than the passed skill).  The ordering inverts and the
        # probe must FAIL.
        db = str(tmp_path / "p3_sab.db")
        s1 = SqliteLearnerStore(db)
        _probe("P3").setup(s1)
        s1.close()

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT card_json FROM reviews WHERE skill=?", (lm._P3_FAILED,)
        ).fetchone()
        card = json.loads(row["card_json"])
        # Push the failed card's due far into the future.
        far = datetime(2099, 1, 1, tzinfo=timezone.utc)
        card["due"] = far.isoformat().replace("+00:00", "Z")
        conn.execute(
            "UPDATE reviews SET card_json=? WHERE skill=?",
            (json.dumps(card), lm._P3_FAILED),
        )
        conn.commit()
        conn.close()

        store = SqliteLearnerStore(db)
        result = _probe("P3").check(store, _ctx())
        store.close()
        assert not result.passed, (
            "P3 must FAIL when due ordering is inverted: " + result.evidence
        )


# ---------------------------------------------------------------------------
# P4 — completed-skill memory
# ---------------------------------------------------------------------------


class TestP4CompletedSkillMemory:
    def test_pass(self, tmp_path):
        db = str(tmp_path / "p4.db")
        store = _setup_then_reopen(_probe("P4"), db)
        result = _probe("P4").check(store, _ctx())
        store.close()
        assert result.passed, result.evidence
        assert "S2 re-serves as 'new'=0" in result.evidence

    def test_sabotage_uncomplete_the_skill(self, tmp_path):
        # Sabotage: drop the completed skill's mastery below MASTERY_COMPLETED on
        # reopen.  The planner will then re-serve it as 'new', and the probe's
        # setup-invariant guard must trip -> FAIL.
        db = str(tmp_path / "p4_sab.db")
        s1 = SqliteLearnerStore(db)
        _probe("P4").setup(s1)
        s1.close()

        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE mastery SET p_mastery=0.5 WHERE skill=?", (lm._P4_COMPLETED,)
        )
        conn.commit()
        conn.close()

        store = SqliteLearnerStore(db)
        result = _probe("P4").check(store, _ctx())
        store.close()
        assert not result.passed, (
            "P4 must FAIL when the completed skill is un-completed: "
            + result.evidence
        )


# ---------------------------------------------------------------------------
# P5 — cross-skill inference guard
# ---------------------------------------------------------------------------


class TestP5CrossSkillInferenceGuard:
    def test_pass(self, tmp_path):
        db = str(tmp_path / "p5.db")
        store = _setup_then_reopen(_probe("P5"), db)
        result = _probe("P5").check(store, _ctx())
        store.close()
        assert result.passed, result.evidence
        assert "unchanged=True" in result.evidence

    def test_sabotage_leak_a_struggle_into_b(self, tmp_path):
        # Sabotage: between sessions, mutate skill B's mastery (simulating a
        # cross-skill leak from A's struggles).  B is no longer unchanged, so the
        # probe must FAIL.
        db = str(tmp_path / "p5_sab.db")
        s1 = SqliteLearnerStore(db)
        _probe("P5").setup(s1)
        s1.close()

        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE mastery SET p_mastery=p_mastery-0.2 WHERE skill=?",
            (lm._P5_SKILL_B,),
        )
        conn.commit()
        conn.close()

        store = SqliteLearnerStore(db)
        result = _probe("P5").check(store, _ctx())
        store.close()
        assert not result.passed, (
            "P5 must FAIL when a struggle leaks into skill B: " + result.evidence
        )


# ---------------------------------------------------------------------------
# P6 — over-personalization guard (+ honest finding)
# ---------------------------------------------------------------------------


class TestP6OverPersonalizationGuard:
    def test_pass_on_promised_invariant(self, tmp_path):
        db = str(tmp_path / "p6.db")
        store = _setup_then_reopen(_probe("P6"), db)
        result = _probe("P6").check(store, _ctx())
        store.close()
        # The promised invariant holds: one generic failure keeps the skill
        # servable (>= floor) and its unlock status unchanged.
        assert result.passed, result.evidence

    def test_records_completed_status_flip_finding(self, tmp_path):
        # The honest finding: the SAME single generic failure drops the skill
        # below MASTERY_COMPLETED.  The probe still passes (promised invariant)
        # but MUST surface the flip as a finding rather than hide it.
        db = str(tmp_path / "p6_find.db")
        store = _setup_then_reopen(_probe("P6"), db)
        result = _probe("P6").check(store, _ctx())
        store.close()
        assert result.finding is not None, (
            "P6 must record the completed-status flip finding"
        )
        assert "OVER-PERSONALIZATION" in result.finding
        assert "MASTERY_COMPLETED" in result.finding
        # The skill DID lose completed status (that's the whole finding).
        assert "completed-status retained=False" in result.evidence

    def test_sabotage_drop_below_floor_fails(self, tmp_path):
        # Sabotage: drive mastery below the servability floor on reopen.  Now the
        # promised invariant is violated and the probe must FAIL — proving the
        # pass is load-bearing, not vacuous.
        db = str(tmp_path / "p6_sab.db")
        s1 = SqliteLearnerStore(db)
        _probe("P6").setup(s1)
        s1.close()

        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE mastery SET p_mastery=0.5 WHERE skill=?", (lm._P6_SKILL,)
        )
        conn.commit()
        conn.close()

        store = SqliteLearnerStore(db)
        result = _probe("P6").check(store, _ctx())
        store.close()
        assert not result.passed, (
            "P6 must FAIL when mastery drops below the servability floor: "
            + result.evidence
        )


# ---------------------------------------------------------------------------
# Report shape and score arithmetic
# ---------------------------------------------------------------------------


class TestReportShapeAndScore:
    def test_all_six_probes_run_and_pass(self, tmp_path):
        report = lm.run_probes(_db_factory(tmp_path))
        assert report.n_total == 6
        assert set(report.results) == {"P1", "P2", "P3", "P4", "P5", "P6"}
        # All six promised invariants hold on a clean store.
        assert report.n_passed == 6, {
            k: v.evidence for k, v in report.results.items() if not v.passed
        }
        assert report.consistency_score == 1.0

    def test_score_is_fraction_passed(self, tmp_path):
        report = lm.run_probes(_db_factory(tmp_path))
        assert report.consistency_score == report.n_passed / report.n_total

    def test_p6_finding_surfaced_in_report(self, tmp_path):
        report = lm.run_probes(_db_factory(tmp_path))
        # The over-personalization finding must propagate to the report.
        assert any("OVER-PERSONALIZATION" in f for f in report.findings), report.findings
        assert any(f.startswith("[P6]") for f in report.findings)

    def test_score_arithmetic_with_a_forced_failure(self, tmp_path):
        # Build a report by hand with one failing probe to pin the arithmetic.
        results = {
            "P1": lm.ProbeResult(True, "ok"),
            "P2": lm.ProbeResult(False, "drift"),
            "P3": lm.ProbeResult(True, "ok"),
            "P4": lm.ProbeResult(True, "ok"),
        }
        report = lm.LearnerMemReport(
            results=results, consistency_score=3 / 4, findings=[]
        )
        assert report.n_passed == 3
        assert report.n_total == 4
        assert report.consistency_score == 0.75


# ---------------------------------------------------------------------------
# Probe contract sanity: setup actually plants something
# ---------------------------------------------------------------------------


class TestProbeContract:
    def test_every_probe_has_id_description_setup_check(self):
        for p in lm.PROBES:
            assert p.id
            assert p.description
            assert callable(p.setup)
            assert callable(p.check)

    def test_setup_writes_to_the_store(self, tmp_path):
        # A probe whose setup is a no-op would make its check vacuous.  Verify
        # each setup leaves at least one observation OR mastery row behind.
        for p in lm.PROBES:
            db = str(tmp_path / f"contract_{p.id}.db")
            s1 = SqliteLearnerStore(db)
            p.setup(s1)
            state = s1.get_state(lm.CHILD_ID)
            s1.close()
            assert state.mastery, f"{p.id} setup planted no mastery"
