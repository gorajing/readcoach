"""T4.2 — Quest planner tests (TDD: write red first, then green).

Tests are grouped into:
  P1  Curriculum validation (cycle → loud; unknown prereq → loud)
  P2  Unlock semantics (mastery + miscue-class gates)
  P3  next_item: never-re-serve completed without review intent
  P4  next_item: review intent re-serve carries "review" reason
  P5  next_item: argmax mastery-gap among unlocked non-completed
  P6  Parity: InMemory and SQLite backends give identical planner decisions
  P7  Two-session continuity (from SQLite)
"""
from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

from readcoach.learner_store import InMemoryLearnerStore, SqliteLearnerStore
from readcoach.planner import (
    MASTERY_COMPLETED,
    MASTERY_THRESHOLD,
    CyclicCurriculumError,
    UnknownPrerequisiteError,
    load_curriculum,
    next_item,
    unlocked,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
_CURRICULUM_PATH = (
    Path(__file__).parent.parent / "data" / "curriculum" / "scope_sequence.yaml"
)


def _obs(store, child, skill, correct, *, miscue_class=None, n=1, session="s1"):
    """Record n observations on skill, optionally tagged with a miscue class."""
    from datetime import timedelta

    base = _NOW
    for i in range(n):
        store.record_observation(
            child_id=child,
            skill=skill,
            correct=correct,
            confidence=1.0,
            session_id=session,
            ts=base + timedelta(seconds=i),
            miscue_class=miscue_class,
        )


def _mastery_to(store, child, skill, target_mastery):
    """Drive mastery on skill to >= target via repeated correct observations."""
    from datetime import timedelta

    base = _NOW
    for i in range(50):
        state = store.get_state(child)
        if state.mastery.get(skill, 0.0) >= target_mastery:
            break
        store.record_observation(
            child_id=child,
            skill=skill,
            correct=True,
            confidence=1.0,
            session_id="setup",
            ts=base + timedelta(seconds=i),
        )
    else:
        raise RuntimeError(
            f"Could not reach mastery={target_mastery} on {skill!r} in 50 steps"
        )


# ---------------------------------------------------------------------------
# P1 — Curriculum validation
# ---------------------------------------------------------------------------


class TestCurriculumValidation:
    def test_load_real_curriculum(self):
        """The shipped scope_sequence.yaml loads without error."""
        c = load_curriculum(_CURRICULUM_PATH)
        assert len(c.nodes) == 15
        assert "cvc_short_a" in c.nodes

    def test_cycle_raises_loud(self, tmp_path):
        """A → B → A cycle raises CyclicCurriculumError."""
        yml = tmp_path / "cyc.yaml"
        yml.write_text(
            textwrap.dedent("""
            nodes:
              - id: skill_a
                band: 1
                label: "A"
                description: "node A"
                prerequisites:
                  - skill: skill_b
                    mastery_min: 0.80
                    classes: [substitution]
              - id: skill_b
                band: 1
                label: "B"
                description: "node B"
                prerequisites:
                  - skill: skill_a
                    mastery_min: 0.80
                    classes: [substitution]
            """)
        )
        with pytest.raises(CyclicCurriculumError):
            load_curriculum(yml)

    def test_unknown_prereq_raises_loud(self, tmp_path):
        """A prerequisite that references a non-existent node id raises."""
        yml = tmp_path / "unk.yaml"
        yml.write_text(
            textwrap.dedent("""
            nodes:
              - id: skill_a
                band: 1
                label: "A"
                description: "node A"
                prerequisites:
                  - skill: skill_nonexistent
                    mastery_min: 0.80
                    classes: [substitution]
            """)
        )
        with pytest.raises(UnknownPrerequisiteError, match="skill_nonexistent"):
            load_curriculum(yml)

    def test_self_loop_raises_loud(self, tmp_path):
        """A node that lists itself as a prerequisite is a cycle."""
        yml = tmp_path / "self.yaml"
        yml.write_text(
            textwrap.dedent("""
            nodes:
              - id: skill_a
                band: 1
                label: "A"
                description: "self loop"
                prerequisites:
                  - skill: skill_a
                    mastery_min: 0.80
                    classes: [substitution]
            """)
        )
        with pytest.raises(CyclicCurriculumError):
            load_curriculum(yml)


# ---------------------------------------------------------------------------
# P2 — Unlock semantics
# ---------------------------------------------------------------------------


class TestUnlockSemantics:
    def _simple_curriculum(self, tmp_path, *, edge_classes):
        """Two-node curriculum: skill_root (no prereqs) → skill_dep."""
        yml = tmp_path / "simple.yaml"
        yml.write_text(
            textwrap.dedent(f"""
            nodes:
              - id: skill_root
                band: 1
                label: "Root"
                description: "root node"
                prerequisites: []
              - id: skill_dep
                band: 2
                label: "Dep"
                description: "dependent node"
                prerequisites:
                  - skill: skill_root
                    mastery_min: 0.80
                    classes: {edge_classes}
            """)
        )
        return load_curriculum(yml)

    def test_no_prereqs_always_unlocked(self, tmp_path):
        c = self._simple_curriculum(tmp_path, edge_classes="[substitution]")
        store = InMemoryLearnerStore()
        result = unlocked(c, "child_u1", store)
        assert "skill_root" in result

    def test_dep_locked_by_low_mastery(self, tmp_path):
        """skill_dep is locked when mastery on skill_root is below threshold."""
        c = self._simple_curriculum(tmp_path, edge_classes="[substitution]")
        store = InMemoryLearnerStore()
        # Record a few wrong answers so mastery is low
        _obs(store, "child_u2", "skill_root", False, n=3)
        result = unlocked(c, "child_u2", store)
        assert "skill_dep" not in result

    def test_dep_unlocked_when_mastery_met_no_blocking_miscues(self, tmp_path):
        """When mastery >= 0.80 and no tagged errors, dep is unlocked."""
        c = self._simple_curriculum(tmp_path, edge_classes="[substitution]")
        store = InMemoryLearnerStore()
        _mastery_to(store, "child_u3", "skill_root", MASTERY_THRESHOLD)
        result = unlocked(c, "child_u3", store)
        assert "skill_dep" in result

    def test_dep_locked_when_mastery_met_but_recent_tagged_miscue(self, tmp_path):
        """Mastery >= 0.80 but last-5 obs contain an incorrect substitution → locked."""
        c = self._simple_curriculum(tmp_path, edge_classes="[substitution]")
        store = InMemoryLearnerStore()
        # Drive mastery high first
        _mastery_to(store, "child_u4", "skill_root", MASTERY_THRESHOLD)
        # Then add a recent substitution error
        _obs(store, "child_u4", "skill_root", False, miscue_class="substitution")
        result = unlocked(c, "child_u4", store)
        assert "skill_dep" not in result

    def test_class_mismatch_does_not_block(self, tmp_path):
        """An error tagged with a class NOT on the edge does NOT block the gate.

        We isolate the class-gate logic by adding correct observations AFTER
        the wrong one to restore mastery above threshold.  The window (k=5)
        still contains the wrong obs but since it's tagged 'hesitation' (not
        'substitution') the gate must not fire.
        """
        c = self._simple_curriculum(tmp_path, edge_classes="[substitution]")
        store = InMemoryLearnerStore()
        # Drive mastery above threshold
        _mastery_to(store, "child_u5", "skill_root", MASTERY_THRESHOLD)
        # Add a 'hesitation' error — not in the edge's classes=[substitution]
        _obs(store, "child_u5", "skill_root", False, miscue_class="hesitation")
        # Restore mastery above threshold with a correct obs (1 correct after 1 wrong)
        _mastery_to(store, "child_u5", "skill_root", MASTERY_THRESHOLD)
        # The wrong hesitation obs is still within the k=5 window
        recent = store.get_last_k_observations("child_u5", "skill_root", k=5)
        wrong_obs = [o for o in recent if not o["correct"]]
        assert any(o["miscue_class"] == "hesitation" for o in wrong_obs), (
            "Test setup: hesitation obs should still be in last-5"
        )
        result = unlocked(c, "child_u5", store)
        # Should be unlocked because hesitation is not in the gate's [substitution]
        assert "skill_dep" in result

    def test_generic_untagged_error_does_not_block(self, tmp_path):
        """An incorrect observation with no miscue_class tag is generic and never blocks."""
        c = self._simple_curriculum(tmp_path, edge_classes="[substitution]")
        store = InMemoryLearnerStore()
        # Drive mastery above threshold
        _mastery_to(store, "child_u6", "skill_root", MASTERY_THRESHOLD)
        # An error with no class tag
        _obs(store, "child_u6", "skill_root", False, miscue_class=None)
        # Restore mastery above threshold
        _mastery_to(store, "child_u6", "skill_root", MASTERY_THRESHOLD)
        # The untagged wrong obs is still within the k=5 window
        recent = store.get_last_k_observations("child_u6", "skill_root", k=5)
        wrong_obs = [o for o in recent if not o["correct"]]
        assert any(o["miscue_class"] is None for o in wrong_obs), (
            "Test setup: untagged error should still be in last-5"
        )
        result = unlocked(c, "child_u6", store)
        # Untagged error is generic — never triggers any class gate
        assert "skill_dep" in result

    def test_old_tagged_errors_beyond_k5_do_not_block(self, tmp_path):
        """Errors more than 5 observations back (the window) do not block."""
        c = self._simple_curriculum(tmp_path, edge_classes="[substitution]")
        store = InMemoryLearnerStore()
        _mastery_to(store, "child_u7", "skill_root", MASTERY_THRESHOLD)
        # Add an old substitution error, then 5 subsequent correct observations
        _obs(store, "child_u7", "skill_root", False, miscue_class="substitution")
        _obs(store, "child_u7", "skill_root", True, n=5)
        result = unlocked(c, "child_u7", store)
        # The substitution is now beyond the k=5 window
        assert "skill_dep" in result

    def test_multi_edge_both_prereqs_needed(self, tmp_path):
        """A node with 2 prereqs is unlocked only when BOTH are met."""
        yml = tmp_path / "multi.yaml"
        yml.write_text(
            textwrap.dedent("""
            nodes:
              - id: skill_a
                band: 1
                label: "A"
                description: "A"
                prerequisites: []
              - id: skill_b
                band: 1
                label: "B"
                description: "B"
                prerequisites: []
              - id: skill_c
                band: 2
                label: "C"
                description: "C"
                prerequisites:
                  - skill: skill_a
                    mastery_min: 0.80
                    classes: [substitution]
                  - skill: skill_b
                    mastery_min: 0.80
                    classes: [substitution]
            """)
        )
        c = load_curriculum(yml)
        store = InMemoryLearnerStore()
        _mastery_to(store, "child_u8", "skill_a", MASTERY_THRESHOLD)
        # Only skill_a is mastered; skill_b is not
        result = unlocked(c, "child_u8", store)
        assert "skill_c" not in result

        _mastery_to(store, "child_u8", "skill_b", MASTERY_THRESHOLD)
        result = unlocked(c, "child_u8", store)
        assert "skill_c" in result


# ---------------------------------------------------------------------------
# P3 — never-re-serve completed without review intent
# ---------------------------------------------------------------------------


class TestNeverReserveCompleted:
    """Completed = mastery >= MASTERY_COMPLETED (0.95).  next_item must not
    return these unless they appear on the FSRS due list (review intent)."""

    def _two_node_curriculum(self, tmp_path):
        yml = tmp_path / "two.yaml"
        yml.write_text(
            textwrap.dedent("""
            nodes:
              - id: root_skill
                band: 1
                label: "Root"
                description: "root"
                prerequisites: []
              - id: next_skill
                band: 2
                label: "Next"
                description: "next"
                prerequisites:
                  - skill: root_skill
                    mastery_min: 0.80
                    classes: [substitution]
            """)
        )
        return load_curriculum(yml)

    def test_completed_skill_not_returned_as_new(self, tmp_path):
        """Once root_skill is completed (mastery >= 0.95), next_item never
        returns it with reason='new'."""
        c = self._two_node_curriculum(tmp_path)
        store = InMemoryLearnerStore()
        # Complete root_skill
        _mastery_to(store, "child_p3a", "root_skill", MASTERY_COMPLETED)
        served_log: list = []

        result = next_item(c, "child_p3a", store, served_log, now=_NOW)
        assert result is not None
        skill_id, reason = result
        # The only non-completed candidate is next_skill (after root unlocks it)
        # root_skill is completed so should not be 'new'
        if skill_id == "root_skill":
            assert reason == "review", (
                "root_skill is completed; it may only be returned with reason='review'"
            )

    def test_completed_never_served_twice_new(self, tmp_path):
        """Completed skill in served_log with reason='new' is never served again."""
        c = self._two_node_curriculum(tmp_path)
        store = InMemoryLearnerStore()

        _mastery_to(store, "child_p3b", "root_skill", MASTERY_COMPLETED)

        # Simulate that root_skill was already served
        served_log = [
            {"skill": "root_skill", "ts": _NOW.isoformat(), "reason": "new"}
        ]
        store.record_served("child_p3b", "root_skill", _NOW, "new")

        result = next_item(c, "child_p3b", store, served_log, now=_NOW)
        if result is not None:
            skill_id, reason = result
            assert skill_id != "root_skill" or reason == "review"


# ---------------------------------------------------------------------------
# P4 — review re-serve carries "review" reason
# ---------------------------------------------------------------------------


class TestReviewReserve:
    def test_due_review_returns_review_reason(self, tmp_path):
        """When a skill is on the FSRS due list, next_item returns it with
        reason='review'."""
        yml = tmp_path / "rev.yaml"
        yml.write_text(
            textwrap.dedent("""
            nodes:
              - id: skill_rev
                band: 1
                label: "Rev"
                description: "review node"
                prerequisites: []
            """)
        )
        c = load_curriculum(yml)
        store = InMemoryLearnerStore()

        # Make skill_rev completed (mastery >= 0.95)
        _mastery_to(store, "child_p4", "skill_rev", MASTERY_COMPLETED)

        # Force it onto the due list by using a far-future `now`
        far_future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        served_log: list = []

        result = next_item(c, "child_p4", store, served_log, now=far_future)
        assert result is not None
        skill_id, reason = result
        assert skill_id == "skill_rev"
        assert reason == "review"


# ---------------------------------------------------------------------------
# P5 — argmax mastery-gap among unlocked non-completed
# ---------------------------------------------------------------------------


class TestArgmaxMasteryGap:
    def test_picks_skill_with_largest_gap(self, tmp_path):
        """next_item picks the unlocked non-completed skill with the lowest mastery
        (largest gap from MASTERY_COMPLETED threshold)."""
        yml = tmp_path / "gap.yaml"
        yml.write_text(
            textwrap.dedent("""
            nodes:
              - id: skill_high
                band: 1
                label: "High"
                description: "high mastery skill"
                prerequisites: []
              - id: skill_low
                band: 1
                label: "Low"
                description: "low mastery skill"
                prerequisites: []
            """)
        )
        c = load_curriculum(yml)
        store = InMemoryLearnerStore()

        # Give skill_high higher mastery than skill_low
        _mastery_to(store, "child_p5", "skill_high", 0.85)
        # skill_low stays at cold-start prior (0.3)
        served_log: list = []

        result = next_item(c, "child_p5", store, served_log, now=_NOW)
        assert result is not None
        skill_id, reason = result
        # skill_low has bigger gap, should be picked
        assert skill_id == "skill_low"
        assert reason == "new"


# ---------------------------------------------------------------------------
# P6 — Parity: InMemory vs SQLite
# ---------------------------------------------------------------------------


class TestParity:
    def test_identical_decisions_both_backends(self, tmp_path):
        """Both backends produce the same next_item decision given same sequence."""
        c = load_curriculum(_CURRICULUM_PATH)

        sql_store = SqliteLearnerStore(str(tmp_path / "parity.db"))
        mem_store = InMemoryLearnerStore()

        child = "parity_child"

        for store in (sql_store, mem_store):
            # Give cvc_short_a some mastery, one recent substitution failure
            _mastery_to(store, child, "cvc_short_a", 0.85)
            _obs(store, child, "cvc_short_a", False, miscue_class="substitution")

        sql_result = next_item(c, child, sql_store, [], now=_NOW)
        mem_result = next_item(c, child, mem_store, [], now=_NOW)

        assert sql_result == mem_result, (
            f"Parity mismatch: sqlite={sql_result!r}, inmem={mem_result!r}"
        )
        sql_store.close()


# ---------------------------------------------------------------------------
# P7 — Two-session continuity (SQLite)
# ---------------------------------------------------------------------------


class TestTwoSessionContinuity:
    """Mirror the two_session_demo.py flow but via pytest tmp_path."""

    def test_mastery_carries_over(self, tmp_path):
        db_path = str(tmp_path / "sess.db")

        # Session 1: build some mastery on cvc_short_a
        s1 = SqliteLearnerStore(db_path)
        _mastery_to(s1, "child_demo", "cvc_short_a", 0.75)
        mastery_s1 = s1.get_state("child_demo").mastery.get("cvc_short_a", 0.0)
        s1.close()

        # Session 2: reopen and verify mastery persisted
        s2 = SqliteLearnerStore(db_path)
        mastery_s2 = s2.get_state("child_demo").mastery.get("cvc_short_a", 0.0)
        s2.close()

        assert abs(mastery_s1 - mastery_s2) < 1e-9

    def test_substitution_gate_blocks_successor_in_session2(self, tmp_path):
        """After session 1 ends with unresolved substitution on cvc_short_a,
        session 2 planner does not unlock cvc_short_i_u."""
        db_path = str(tmp_path / "gate.db")
        c = load_curriculum(_CURRICULUM_PATH)

        # Session 1
        s1 = SqliteLearnerStore(db_path)
        _mastery_to(s1, "child_gate", "cvc_short_a", MASTERY_THRESHOLD)
        # Add an unresolved substitution error at the end of session 1
        _obs(s1, "child_gate", "cvc_short_a", False, miscue_class="substitution")
        s1.close()

        # Session 2: reopen
        s2 = SqliteLearnerStore(db_path)
        unlocked_s2 = unlocked(c, "child_gate", s2)
        s2.close()

        assert "cvc_short_i_u" not in unlocked_s2

    def test_completed_skill_never_reserved_new_across_sessions(self, tmp_path):
        """A skill completed in session 1 is not returned with reason='new' in
        session 2."""
        db_path = str(tmp_path / "norepeat.db")
        c = load_curriculum(_CURRICULUM_PATH)

        # Session 1: complete cvc_short_a, record it served
        s1 = SqliteLearnerStore(db_path)
        _mastery_to(s1, "child_nr", "cvc_short_a", MASTERY_COMPLETED)
        s1.record_served("child_nr", "cvc_short_a", _NOW, "new")
        s1.close()

        # Session 2: served_log reconstructed from store
        s2 = SqliteLearnerStore(db_path)
        served_log = s2.get_served_log("child_nr")
        result = next_item(c, "child_nr", s2, served_log, now=_NOW)
        s2.close()

        if result is not None:
            skill_id, reason = result
            assert not (skill_id == "cvc_short_a" and reason == "new")

    def test_served_log_persists_across_sessions(self, tmp_path):
        """record_served entries survive store close/reopen."""
        db_path = str(tmp_path / "servlog.db")
        s1 = SqliteLearnerStore(db_path)
        s1.record_served("child_sl", "cvc_short_a", _NOW, "new")
        s1.close()

        s2 = SqliteLearnerStore(db_path)
        log = s2.get_served_log("child_sl")
        s2.close()

        assert any(e["skill"] == "cvc_short_a" for e in log)
