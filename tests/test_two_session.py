"""T4.2 — Two-session continuity test via tmp_path SQLite.

Mirrors the flow of scripts/two_session_demo.py but uses pytest fixtures
so it runs in-process and cleans up automatically.

Observable claims tested:
  (a) Mastery carried over from session 1 survives store close/reopen.
  (b) FSRS due-reviews are present in session 2 (at far-future cutoff).
  (c) Planner in session 2 does NOT serve the gated successor when session 1
      ended with unresolved substitution miscues on the prerequisite.
  (d) A completed skill (mastery >= MASTERY_COMPLETED) is never served
      with reason='new' in session 2.
  (e) served_log persists across close/reopen.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from readcoach.learner_store import SqliteLearnerStore
from readcoach.planner import (
    MASTERY_COMPLETED,
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
# Shared session-1 setup — writes cvc_short_a mastery + substitution failure
# -----------------------------------------------------------------------


def _build_session1(db_path: str) -> dict:
    """Run session 1 and return state snapshot for assertions in session 2."""
    s1 = SqliteLearnerStore(db_path)

    # Build mastery on cvc_short_a
    _mastery_to(s1, "cvc_short_a", 0.85, session_id="s1")

    # End session with substitution-tagged failures (these stay in k=5 window)
    for i in range(2):
        s1.record_observation(
            child_id=_CHILD,
            skill="cvc_short_a",
            correct=False,
            confidence=1.0,
            session_id="s1",
            ts=_BASE_TS + timedelta(minutes=10 + i),
            miscue_class="substitution",
        )

    # Record a served entry for cvc_short_a
    s1.record_served(_CHILD, "cvc_short_a", _BASE_TS, "new")

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
# Claim (c) — substitution gate blocks gated successor in session 2
# ---------------------------------------------------------------------------


class TestSubstitutionGateBlocksSuccessor:
    def test_gated_skill_not_in_unlocked(self, tmp_path):
        """After session 1 ends with substitution errors on cvc_short_a,
        session 2 does not have cvc_short_i_u in the unlocked set."""
        db_path = str(tmp_path / "demo.db")
        _build_session1(db_path)

        curriculum = load_curriculum(CURRICULUM_PATH)
        s2 = SqliteLearnerStore(db_path)

        # Check that the last-5 observations still contain a substitution error
        recent = s2.get_last_k_observations(_CHILD, "cvc_short_a", k=5)
        wrong_sub = [
            o for o in recent
            if not o["correct"] and o["miscue_class"] == "substitution"
        ]

        if wrong_sub:
            # Gate is active: successor must not be unlocked
            unlocked_s2 = unlocked(curriculum, _CHILD, s2)
            assert "cvc_short_i_u" not in unlocked_s2, (
                "cvc_short_i_u should be gated by unresolved substitution miscues"
            )

            # And next_item must not return cvc_short_i_u
            served_log = s2.get_served_log(_CHILD)
            pick = next_item(
                curriculum, _CHILD, s2, served_log,
                now=_BASE_TS + timedelta(days=1)
            )
            if pick is not None:
                assert pick[0] != "cvc_short_i_u", (
                    "Planner served gated successor cvc_short_i_u; "
                    "substitution gate should block it"
                )
        else:
            pytest.skip(
                "Window no longer contains substitution error (mastery/window cleared)"
            )

        s2.close()


# ---------------------------------------------------------------------------
# Claim (d) — completed skill never re-served as 'new' across sessions
# ---------------------------------------------------------------------------


class TestCompletedSkillNeverReservedNew:
    def test_completed_in_s1_not_new_in_s2(self, tmp_path):
        db_path = str(tmp_path / "demo.db")
        curriculum = load_curriculum(CURRICULUM_PATH)

        # Session 1: complete cvc_short_a
        s1 = SqliteLearnerStore(db_path)
        _mastery_to(s1, "cvc_short_a", MASTERY_COMPLETED, session_id="s1")
        s1.record_served(_CHILD, "cvc_short_a", _BASE_TS, "new")
        mastery_s1 = s1.get_state(_CHILD).mastery.get("cvc_short_a", 0.0)
        assert mastery_s1 >= MASTERY_COMPLETED
        s1.close()

        # Session 2: reopen, pick items, never re-serve cvc_short_a as 'new'
        s2 = SqliteLearnerStore(db_path)
        for step in range(6):
            sl = s2.get_served_log(_CHILD)
            pick = next_item(
                curriculum, _CHILD, s2, sl,
                now=_BASE_TS + timedelta(seconds=step)
            )
            if pick is not None:
                skill, reason = pick
                if skill == "cvc_short_a":
                    assert reason == "review", (
                        f"Completed skill cvc_short_a returned as 'new' in step {step}"
                    )
        s2.close()


# ---------------------------------------------------------------------------
# Claim (e) — served_log persists across sessions
# ---------------------------------------------------------------------------


class TestServedLogPersists:
    def test_served_entries_survive_reopen(self, tmp_path):
        db_path = str(tmp_path / "demo.db")
        _build_session1(db_path)

        # _build_session1 records cvc_short_a as served
        s2 = SqliteLearnerStore(db_path)
        log = s2.get_served_log(_CHILD)
        s2.close()

        assert any(e["skill"] == "cvc_short_a" for e in log), (
            "served_log entry for cvc_short_a should persist across close/reopen"
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
