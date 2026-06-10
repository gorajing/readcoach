"""T3.4 — LearnerState store: SQLite/in-memory parity tests.

Parametrized over both backends so every behavioral test runs on both unless
explicitly skipped (e.g. round-trip persistence only makes sense on SQLite).

Rating mapping (documented here, not buried in implementation):
    correct=True  → fsrs.Rating.Good
    correct=False → fsrs.Rating.Again

These tests are the behavioural spec; the implementation must make them green.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from readcoach.bkt import bkt_update
from readcoach.learner_store import (
    DEFAULT_BKT_PARAMS,
    InMemoryLearnerStore,
    SchemaVersionError,
    SqliteLearnerStore,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sqlite_store(tmp_path: Path) -> SqliteLearnerStore:
    return SqliteLearnerStore(str(tmp_path / "test.db"))


def _inmem_store() -> InMemoryLearnerStore:
    return InMemoryLearnerStore()


@pytest.fixture(params=["sqlite", "inmemory"])
def store(request, tmp_path):
    """Parametrized fixture yielding both backends."""
    if request.param == "sqlite":
        return _sqlite_store(tmp_path)
    return _inmem_store()


# ---------------------------------------------------------------------------
# T1 — mastery moves per bkt_update (correct observation)
# ---------------------------------------------------------------------------

class TestMasteryUpdate:
    def test_correct_observation_moves_mastery(self, store):
        """record_observation → mastery == direct bkt_update call."""
        child = "child_001"
        skill = "digraph_ch"
        # Store has prior = BktParams.L0
        prior = DEFAULT_BKT_PARAMS.L0

        store.record_observation(child, skill, correct=True, confidence=1.0,
                                 session_id="s1", ts=_now())

        state = store.get_state(child)
        expected = bkt_update(prior, True, 1.0,
                               DEFAULT_BKT_PARAMS.s,
                               DEFAULT_BKT_PARAMS.g,
                               DEFAULT_BKT_PARAMS.t)
        assert abs(state.mastery[skill] - expected) < 1e-9

    def test_incorrect_observation_moves_mastery_down(self, store):
        prior = DEFAULT_BKT_PARAMS.L0
        child = "child_002"
        skill = "vowel_team_ea"

        store.record_observation(child, skill, correct=False, confidence=1.0,
                                 session_id="s1", ts=_now())

        state = store.get_state(child)
        expected = bkt_update(prior, False, 1.0,
                               DEFAULT_BKT_PARAMS.s,
                               DEFAULT_BKT_PARAMS.g,
                               DEFAULT_BKT_PARAMS.t)
        assert abs(state.mastery[skill] - expected) < 1e-9


# ---------------------------------------------------------------------------
# T2 — soft evidence: conf=0.5 leaves mastery unchanged (only transit fires)
# ---------------------------------------------------------------------------

class TestSoftEvidence:
    def test_half_confidence_observation_only_applies_transit(self, store):
        """conf=0.5 carries zero information; only transit is applied."""
        child = "child_003"
        skill = "cvc_blend"
        prior = DEFAULT_BKT_PARAMS.L0
        t = DEFAULT_BKT_PARAMS.t

        # conf=0.5 → information-free → posterior == prior before transit
        # After transit: prior + (1 - prior) * t
        expected = bkt_update(prior, True, 0.5,
                               DEFAULT_BKT_PARAMS.s,
                               DEFAULT_BKT_PARAMS.g,
                               DEFAULT_BKT_PARAMS.t)
        direct = prior + (1 - prior) * t
        assert abs(expected - direct) < 1e-9, "bkt_update sanity: conf=0.5 should equal transit-only"

        store.record_observation(child, skill, correct=True, confidence=0.5,
                                 session_id="s1", ts=_now())

        state = store.get_state(child)
        assert abs(state.mastery[skill] - expected) < 1e-9


# ---------------------------------------------------------------------------
# T3 — FSRS due_reviews: failed obs makes skill due sooner than passed
# ---------------------------------------------------------------------------

class TestFsrsDueReviews:
    def test_failed_obs_due_sooner_than_passed(self, store):
        """A wrong answer → Due sooner (Again) vs right answer → Due later (Good)."""
        now = _now()
        store.record_observation("child_a", "skill_pass", correct=True,
                                 confidence=1.0, session_id="s1", ts=now)
        store.record_observation("child_b", "skill_fail", correct=False,
                                 confidence=1.0, session_id="s1", ts=now)

        # Retrieve due reviews with a far-future cutoff so both appear
        far_future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        due_a = store.due_reviews("child_a", far_future)
        due_b = store.due_reviews("child_b", far_future)

        # Both skills show up (they were observed once, so they have a card)
        assert "skill_pass" in due_a
        assert "skill_fail" in due_b

        # Get card state to compare due dates
        card_pass = store.get_card("child_a", "skill_pass")
        card_fail = store.get_card("child_b", "skill_fail")

        assert card_fail.due < card_pass.due, (
            f"Failed skill due={card_fail.due} should be sooner than "
            f"passed skill due={card_pass.due}"
        )

    def test_due_reviews_respects_cutoff(self, store):
        """due_reviews only returns skills whose card.due <= now."""
        now = _now()
        # Record a correct answer; card will be due in the future
        store.record_observation("child_c", "skill_future", correct=True,
                                 confidence=1.0, session_id="s1", ts=now)
        # Check right now — skill should not be due yet (Good pushed it out)
        due_now = store.due_reviews("child_c", now)
        assert "skill_future" not in due_now


# ---------------------------------------------------------------------------
# T4 — WCPM and hesitation trend
# ---------------------------------------------------------------------------

class TestEngagementMetrics:
    def test_wcpm_hand_computed(self, store):
        """120 words in 90 s → 80.0 WCPM."""
        store.record_session_metrics(
            child_id="child_wcpm",
            session_id="sess_wcpm",
            n_words_read=120,
            n_hesitations=6,
            duration_s=90.0,
            ts=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        trend = store.engagement_trend("child_wcpm")
        assert len(trend) == 1
        ts, wcpm, hrate = trend[0]
        assert abs(wcpm - 80.0) < 1e-6
        assert abs(hrate - 6 / 120) < 1e-9

    def test_hesitation_trend_ordering(self, store):
        """engagement_trend returns rows in ascending timestamp order."""
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(3):
            from datetime import timedelta
            store.record_session_metrics(
                child_id="child_trend",
                session_id=f"sess_{i}",
                n_words_read=100,
                n_hesitations=i * 2,
                duration_s=60.0,
                ts=base + timedelta(hours=i),
            )
        trend = store.engagement_trend("child_trend")
        assert len(trend) == 3
        timestamps = [row[0] for row in trend]
        assert timestamps == sorted(timestamps)

    def test_wcpm_two_sessions(self, store):
        """Multiple sessions produce correct per-session WCPM."""
        base = datetime(2026, 1, 2, tzinfo=timezone.utc)
        from datetime import timedelta
        store.record_session_metrics(
            child_id="child_multi",
            session_id="sess_a",
            n_words_read=60,
            n_hesitations=3,
            duration_s=60.0,
            ts=base,
        )
        store.record_session_metrics(
            child_id="child_multi",
            session_id="sess_b",
            n_words_read=90,
            n_hesitations=9,
            duration_s=60.0,
            ts=base + timedelta(hours=1),
        )
        trend = store.engagement_trend("child_multi")
        assert len(trend) == 2
        assert abs(trend[0][1] - 60.0) < 1e-6   # 60 words / 1 min
        assert abs(trend[1][1] - 90.0) < 1e-6   # 90 words / 1 min


# ---------------------------------------------------------------------------
# T5 — Parity: identical sequence → identical state on both backends
# ---------------------------------------------------------------------------

class TestParity:
    """Run a fixed sequence of ~10 mixed operations on both backends and
    verify LearnerState (mastery, due_reviews) + trend come out identical."""

    @staticmethod
    def _run_sequence(s):
        """Shared scripted sequence; s is any store instance."""
        base = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
        from datetime import timedelta

        child = "parity_child"

        obs = [
            ("digraph_ch",  True,  1.0),
            ("vowel_ea",    False, 1.0),
            ("cvc_blend",   True,  0.5),
            ("digraph_sh",  True,  0.8),
            ("sight_the",   False, 0.9),
            ("digraph_ch",  True,  1.0),  # second obs on same skill
            ("vowel_ea",    True,  1.0),
            ("cvc_blend",   False, 0.7),
            ("sight_the",   True,  1.0),
            ("digraph_sh",  False, 0.6),
        ]
        for i, (skill, correct, conf) in enumerate(obs):
            s.record_observation(child, skill, correct=correct, confidence=conf,
                                 session_id=f"sess_{i // 3}",
                                 ts=base + timedelta(minutes=i))

        # Three session-metric rows
        for i in range(3):
            s.record_session_metrics(
                child_id=child,
                session_id=f"sess_{i}",
                n_words_read=80 + i * 10,
                n_hesitations=i + 1,
                duration_s=60.0,
                ts=base + timedelta(hours=i),
            )
        return child

    def test_mastery_parity(self, tmp_path):
        sql = _sqlite_store(tmp_path)
        mem = _inmem_store()
        child_sql = self._run_sequence(sql)
        child_mem = self._run_sequence(mem)

        state_sql = sql.get_state(child_sql)
        state_mem = mem.get_state(child_mem)

        assert set(state_sql.mastery.keys()) == set(state_mem.mastery.keys())
        for skill in state_sql.mastery:
            assert abs(state_sql.mastery[skill] - state_mem.mastery[skill]) < 1e-9, \
                f"mastery mismatch on {skill!r}"

    def test_due_reviews_parity(self, tmp_path):
        sql = _sqlite_store(tmp_path)
        mem = _inmem_store()
        child_sql = self._run_sequence(sql)
        child_mem = self._run_sequence(mem)

        far_future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        due_sql = sorted(sql.due_reviews(child_sql, far_future))
        due_mem = sorted(mem.due_reviews(child_mem, far_future))
        assert due_sql == due_mem

    def test_trend_parity(self, tmp_path):
        sql = _sqlite_store(tmp_path)
        mem = _inmem_store()
        child_sql = self._run_sequence(sql)
        child_mem = self._run_sequence(mem)

        trend_sql = sql.engagement_trend(child_sql)
        trend_mem = mem.engagement_trend(child_mem)
        assert len(trend_sql) == len(trend_mem)
        for (ts_s, wcpm_s, hr_s), (ts_m, wcpm_m, hr_m) in zip(trend_sql, trend_mem):
            assert abs(wcpm_s - wcpm_m) < 1e-6
            assert abs(hr_s - hr_m) < 1e-9


# ---------------------------------------------------------------------------
# T6 — Two-session round-trip (SQLite only; in-memory is not persistent)
# ---------------------------------------------------------------------------

class TestTwoSessionRoundTrip:
    """Close the SQLite store, reopen it on the same path, verify state persists."""

    def test_mastery_survives_reopen(self, tmp_path):
        db_path = str(tmp_path / "roundtrip.db")
        # Session 1: write
        s1 = SqliteLearnerStore(db_path)
        s1.record_observation("child_rt", "skill_rt", correct=True, confidence=1.0,
                               session_id="sess_1", ts=_now())
        state1 = s1.get_state("child_rt")
        s1.close()

        # Session 2: reopen and read
        s2 = SqliteLearnerStore(db_path)
        state2 = s2.get_state("child_rt")
        s2.close()

        assert abs(state2.mastery["skill_rt"] - state1.mastery["skill_rt"]) < 1e-9

    def test_due_reviews_survives_reopen(self, tmp_path):
        db_path = str(tmp_path / "roundtrip2.db")
        s1 = SqliteLearnerStore(db_path)
        s1.record_observation("child_rr", "skill_rr", correct=False, confidence=1.0,
                               session_id="sess_1", ts=_now())
        s1.close()

        s2 = SqliteLearnerStore(db_path)
        due = s2.due_reviews("child_rr", datetime(2099, 1, 1, tzinfo=timezone.utc))
        assert "skill_rr" in due
        s2.close()

    def test_session_metrics_survive_reopen(self, tmp_path):
        db_path = str(tmp_path / "roundtrip3.db")
        ts = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
        s1 = SqliteLearnerStore(db_path)
        s1.record_session_metrics("child_sm", "sess_sm", 120, 6, 90.0, ts)
        s1.close()

        s2 = SqliteLearnerStore(db_path)
        trend = s2.engagement_trend("child_sm")
        assert len(trend) == 1
        _, wcpm, _ = trend[0]
        assert abs(wcpm - 80.0) < 1e-6
        s2.close()

    def test_second_session_continues_mastery(self, tmp_path):
        """Session 2 can observe the same skill and mastery continues updating."""
        db_path = str(tmp_path / "roundtrip4.db")
        s1 = SqliteLearnerStore(db_path)
        s1.record_observation("child_cont", "skill_cont", correct=True, confidence=1.0,
                               session_id="sess_1", ts=_now())
        mastery_after_s1 = s1.get_state("child_cont").mastery["skill_cont"]
        s1.close()

        s2 = SqliteLearnerStore(db_path)
        s2.record_observation("child_cont", "skill_cont", correct=True, confidence=1.0,
                               session_id="sess_2", ts=_now())
        mastery_after_s2 = s2.get_state("child_cont").mastery["skill_cont"]
        s2.close()

        # Second correct obs should push mastery higher
        assert mastery_after_s2 > mastery_after_s1


# ---------------------------------------------------------------------------
# T7 — Schema version mismatch → loud error
# ---------------------------------------------------------------------------

class TestSchemaVersionMismatch:
    def test_wrong_version_raises(self, tmp_path):
        db_path = str(tmp_path / "old_schema.db")
        # Craft a db with version=99
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);")
        conn.execute("INSERT INTO meta VALUES ('schema_version', '99');")
        conn.commit()
        conn.close()

        with pytest.raises(SchemaVersionError, match="99"):
            SqliteLearnerStore(db_path)


# ---------------------------------------------------------------------------
# T8 — Transaction rollback: partial failure leaves no rows
# ---------------------------------------------------------------------------

class TestTransactionRollback:
    def test_rollback_on_mid_write_failure(self, tmp_path):
        """A forced failure mid-operation leaves the db in clean pre-op state."""
        db_path = str(tmp_path / "rollback.db")
        store = SqliteLearnerStore(db_path)

        # Record a baseline obs to confirm the db is operable
        store.record_observation("child_rb", "skill_good", correct=True, confidence=1.0,
                                 session_id="sess_good", ts=_now())

        # Now trigger rollback by passing an invalid confidence (should raise)
        with pytest.raises(Exception):
            store.record_observation("child_rb", "skill_bad", correct=True,
                                     confidence=99.0,  # invalid — bkt_update will raise
                                     session_id="sess_bad", ts=_now())

        # skill_bad should NOT be in mastery (no partial row committed)
        state = store.get_state("child_rb")
        assert "skill_bad" not in state.mastery
        # skill_good should still be fine
        assert "skill_good" in state.mastery
