"""T4.2 — Two-session continuity test via tmp_path SQLite.

Mirrors the flow of scripts/two_session_demo.py but uses pytest fixtures
so it runs in-process and cleans up automatically.

Observable claims tested:
  (a) Mastery carried over from session 1 survives store close/reopen.
  (b) FSRS due-reviews are present in session 2 (at far-future cutoff).
  (c) CLASS gate bites independently of the mastery gate: session 2 mastery
      on cvc_short_a is >= 0.80 (mastery gate PASSES) AND cvc_short_i_u is
      LOCKED (class gate fires because last-5 window contains a substitution).
  (d) A completed skill (cvc_short_i_u, mastery >= MASTERY_COMPLETED from
      session 1) is never served with reason='new' in session 2.
  (e) served_log persists across close/reopen.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from readcoach.learner_store import SqliteLearnerStore
from readcoach.planner import (
    MASTERY_COMPLETED,
    MASTERY_THRESHOLD,
    load_curriculum,
    next_item,
    unlocked,
)

CURRICULUM_PATH = (
    Path(__file__).parent.parent / "data" / "curriculum" / "scope_sequence.yaml"
)

_BASE_TS = datetime(2026, 6, 10, 9, 0, 0, tzinfo=timezone.utc)
_CHILD = "demo_child"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs(store, skill, correct, *, miscue_class=None, n=1, offset_s=0,
         session_id="s1"):
    for i in range(n):
        store.record_observation(
            child_id=_CHILD,
            skill=skill,
            correct=correct,
            confidence=1.0,
            session_id=session_id,
            ts=_BASE_TS + timedelta(seconds=offset_s + i),
            miscue_class=miscue_class,
        )


def _mastery_to(store, skill, target, session_id="s1"):
    for i in range(60):
        state = store.get_state(_CHILD)
        if state.mastery.get(skill, 0.0) >= target:
            return
        store.record_observation(
            child_id=_CHILD,
            skill=skill,
            correct=True,
            confidence=1.0,
            session_id=session_id,
            ts=_BASE_TS + timedelta(seconds=i),
        )
    raise RuntimeError(f"Could not reach mastery={target} on {skill!r}")


# ---------------------------------------------------------------------------
# Shared session-1 setup (mirrors scripts/two_session_demo.py staging)
# ---------------------------------------------------------------------------


def _build_session1(db_path: str) -> dict:
    """Run session 1 and return a mastery snapshot for session-2 assertions.

    Staging mirrors two_session_demo.py:
      1. Drive cvc_short_a to >= 0.98 (one failure from there → ~0.90,
         still >= MASTERY_THRESHOLD=0.80 so mastery gate passes in S2).
      2. Complete cvc_short_i_u (mastery >= MASTERY_COMPLETED=0.95) while
         cvc_short_a is still clean — makes claim (d) non-vacuous.
      3. Inject ONE substitution-tagged failure on cvc_short_a at the END of
         the session so it sits in the k=5 window when session 2 opens.
    """
    s1 = SqliteLearnerStore(db_path)

    # Step 1: build cvc_short_a mastery to >= 0.98
    _mastery_to(s1, "cvc_short_a", 0.98, session_id="s1")

    # Step 2: complete cvc_short_i_u while cvc_short_a is clean
    _mastery_to(s1, "cvc_short_i_u", MASTERY_COMPLETED, session_id="s1")
    s1.record_served(_CHILD, "cvc_short_i_u", _BASE_TS + timedelta(minutes=5), "new")

    # Step 3: ONE substitution failure on cvc_short_a (stays in k=5 window)
    s1.record_observation(
        child_id=_CHILD,
        skill="cvc_short_a",
        correct=False,
        confidence=1.0,
        session_id="s1",
        ts=_BASE_TS + timedelta(minutes=10),
        miscue_class="substitution",
    )

    snapshot = s1.get_state(_CHILD).mastery.copy()
    s1.close()
    return snapshot


# ---------------------------------------------------------------------------
# Claim (a) — mastery carries over
# ---------------------------------------------------------------------------


class TestMasteryCarryOver:
    def test_mastery_survives_close_reopen(self, tmp_path):
        db_path = str(tmp_path / "demo.db")
        snapshot = _build_session1(db_path)

        s2 = SqliteLearnerStore(db_path)
        state_s2 = s2.get_state(_CHILD)
        s2.close()

        for skill, m_s1 in snapshot.items():
            m_s2 = state_s2.mastery.get(skill, None)
            assert m_s2 is not None, f"Skill {skill!r} missing in session 2"
            assert abs(m_s1 - m_s2) < 1e-9, (
                f"Mastery mismatch for {skill!r}: S1={m_s1}, S2={m_s2}"
            )


# ---------------------------------------------------------------------------
# Claim (b) — due reviews present in session 2
# ---------------------------------------------------------------------------


class TestDueReviewsInSession2:
    def test_failed_obs_creates_due_review(self, tmp_path):
        db_path = str(tmp_path / "demo.db")
        _build_session1(db_path)

        s2 = SqliteLearnerStore(db_path)
        # Use far-future cutoff to see all due skills
        due = s2.due_reviews(_CHILD, datetime(2099, 1, 1, tzinfo=timezone.utc))
        s2.close()

        assert len(due) > 0, "Session 2 should have at least one due review"


# ---------------------------------------------------------------------------
# Claim (c) — CLASS gate blocks cvc_short_i_u INDEPENDENTLY of mastery gate
# ---------------------------------------------------------------------------


class TestClassGateBlocksSuccessorWithMasteryAboveThreshold:
    """The novel mechanism tested here: in session 2, cvc_short_a mastery is
    above MASTERY_THRESHOLD (mastery gate passes) AND cvc_short_i_u is still
    LOCKED because the last-5 window contains a substitution error (class gate
    fires independently).

    The guard assertion ensures the test fails loudly if BKT params change
    such that the mastery gate becomes the active blocker instead.
    """

    def test_mastery_above_threshold_AND_gated_skill_locked(self, tmp_path):
        db_path = str(tmp_path / "demo.db")
        snapshot = _build_session1(db_path)

        curriculum = load_curriculum(CURRICULUM_PATH)
        s2 = SqliteLearnerStore(db_path)

        # Guard assertion: mastery on cvc_short_a must be >= MASTERY_THRESHOLD.
        # If this fails, the test setup (staging in _build_session1) must be
        # revised — the mastery gate is the active blocker, not the class gate.
        m_a_s2 = snapshot.get("cvc_short_a", 0.0)
        assert m_a_s2 >= MASTERY_THRESHOLD, (
            f"GUARD FAILED: cvc_short_a mastery is {m_a_s2:.4f} < "
            f"{MASTERY_THRESHOLD} (MASTERY_THRESHOLD).  The mastery gate would "
            f"be the active blocker, not the class gate.  Update _build_session1 "
            f"to drive cvc_short_a to >= 0.98 before injecting the failure."
        )

        # Verify the substitution error is in the last-5 window
        recent = s2.get_last_k_observations(_CHILD, "cvc_short_a", k=5)
        wrong_sub = [
            o for o in recent
            if not o["correct"] and o["miscue_class"] == "substitution"
        ]
        assert wrong_sub, (
            "Test setup: no substitution error in last-5 window for cvc_short_a. "
            "Check _build_session1 staging."
        )

        # Assert cvc_short_i_u is LOCKED (class gate fires)
        unlocked_s2 = unlocked(curriculum, _CHILD, s2)
        assert "cvc_short_i_u" not in unlocked_s2, (
            f"cvc_short_i_u should be LOCKED by the class gate. "
            f"mastery[cvc_short_a]={m_a_s2:.4f} >= {MASTERY_THRESHOLD} "
            f"(mastery gate PASSES) but last-5 window has "
            f"{len(wrong_sub)} substitution error(s) — class gate must fire."
        )

        # next_item must not serve cvc_short_i_u
        served_log = s2.get_served_log(_CHILD)
        pick = next_item(
            curriculum, _CHILD, s2, served_log,
            now=_BASE_TS + timedelta(days=1)
        )
        if pick is not None:
            assert pick[0] != "cvc_short_i_u", (
                "Planner served gated successor cvc_short_i_u; "
                "class gate should block it even though mastery gate passes"
            )

        s2.close()


# ---------------------------------------------------------------------------
# Claim (d) — completed skill never re-served as 'new' across sessions
# ---------------------------------------------------------------------------


class TestCompletedSkillNeverReservedNew:
    def test_completed_in_s1_not_new_in_s2(self, tmp_path):
        db_path = str(tmp_path / "demo.db")
        snapshot = _build_session1(db_path)
        curriculum = load_curriculum(CURRICULUM_PATH)

        # cvc_short_i_u was completed in session 1
        m_iu = snapshot.get("cvc_short_i_u", 0.0)
        assert m_iu >= MASTERY_COMPLETED, (
            f"Test setup: cvc_short_i_u mastery={m_iu:.4f} < MASTERY_COMPLETED. "
            f"Check _build_session1."
        )

        # Session 2: reopen, pick items, never re-serve cvc_short_i_u as 'new'
        s2 = SqliteLearnerStore(db_path)
        for step in range(6):
            sl = s2.get_served_log(_CHILD)
            pick = next_item(
                curriculum, _CHILD, s2, sl,
                now=_BASE_TS + timedelta(seconds=step)
            )
            if pick is not None:
                skill, reason = pick
                if skill == "cvc_short_i_u":
                    assert reason == "review", (
                        f"Completed skill cvc_short_i_u returned as 'new' in step {step}"
                    )
        s2.close()


# ---------------------------------------------------------------------------
# Claim (e) — served_log persists across sessions
# ---------------------------------------------------------------------------


class TestServedLogPersists:
    def test_served_entries_survive_reopen(self, tmp_path):
        db_path = str(tmp_path / "demo.db")
        _build_session1(db_path)

        # _build_session1 records cvc_short_i_u as served in session 1
        s2 = SqliteLearnerStore(db_path)
        log = s2.get_served_log(_CHILD)
        s2.close()

        assert any(e["skill"] == "cvc_short_i_u" for e in log), (
            "served_log entry for cvc_short_i_u should persist across close/reopen"
        )

    def test_served_log_next_item_appends(self, tmp_path):
        db_path = str(tmp_path / "demo.db")
        curriculum = load_curriculum(CURRICULUM_PATH)

        s1 = SqliteLearnerStore(db_path)
        sl = s1.get_served_log(_CHILD)
        pick = next_item(curriculum, _CHILD, s1, sl, now=_BASE_TS)
        s1.close()

        assert pick is not None
        s2 = SqliteLearnerStore(db_path)
        log = s2.get_served_log(_CHILD)
        s2.close()

        assert len(log) >= 1
        assert log[-1]["skill"] == pick[0]
        assert log[-1]["reason"] == pick[1]
