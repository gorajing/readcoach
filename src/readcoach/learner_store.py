"""T3.4 — LearnerState store: SQLite and in-memory backends.

Two backends behind one interface:
    SqliteLearnerStore(db_path)  — WAL discipline, session_scope transactions
    InMemoryLearnerStore()       — for tests and ephemeral use; not persistent

Schema v1 tables (SQLite):
    meta         (key TEXT PK, value TEXT)
    learners     (child_id TEXT PK)
    sessions     (session_id TEXT PK, child_id TEXT, started_at TEXT)
    observations (id INTEGER PK AUTOINCREMENT, child_id TEXT, skill TEXT,
                  correct INTEGER, confidence REAL, session_id TEXT, ts TEXT,
                  miscue_class TEXT NULL)
    mastery      (child_id TEXT, skill TEXT, p_mastery REAL,
                  PRIMARY KEY (child_id, skill))
    reviews      (child_id TEXT, skill TEXT, card_json TEXT,
                  PRIMARY KEY (child_id, skill))
    session_metrics (session_id TEXT PK, child_id TEXT, n_words_read INTEGER,
                     n_hesitations INTEGER, duration_s REAL, ts TEXT)
    served_log   (id INTEGER PK AUTOINCREMENT, child_id TEXT, skill TEXT,
                  ts TEXT, reason TEXT)

FSRS rating mapping (documented, not buried):
    correct=True  → fsrs.Rating.Good
    correct=False → fsrs.Rating.Again

BKT default parameters (constant; fitted-per-skill params arrive with later
curriculum work per plan note):
    L0 = 0.3   # moderate cold-start prior
    s  = 0.1   # slip: 10% chance of error when mastered
    g  = 0.3   # guess: 30% chance correct when unmastered
    t  = 0.1   # transit: 10% learning rate per opportunity

WAL discipline lifted from Music/music-analyzer jobs/db.py (Rule 10):
    PRAGMA journal_mode=WAL
    PRAGMA busy_timeout=5000
    PRAGMA foreign_keys=ON

session_scope pattern lifted from TheDose database.py:
    contextmanager → commit on exit, rollback+re-raise on any exception.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from fsrs import Card, Rating, Scheduler

from readcoach.bkt import BktParams, bkt_update
from readcoach.learner_model import LearnerState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

# Default BKT parameter bundle — one set for all skills.
# Rationale: fitted-per-skill params arrive with curriculum work (later ticket).
# Values chosen to be identifiable and conservative for a cold start.
DEFAULT_BKT_PARAMS = BktParams(s=0.1, g=0.3, t=0.1, L0=0.3)

_FSRS = Scheduler()

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SchemaVersionError(RuntimeError):
    """Raised when an existing db has a different schema_version than expected."""


# ---------------------------------------------------------------------------
# Store protocol (structural)
# ---------------------------------------------------------------------------


class LearnerStoreProtocol(Protocol):
    def record_observation(
        self,
        child_id: str,
        skill: str,
        correct: bool,
        confidence: float,
        session_id: str,
        ts: datetime,
        miscue_class: str | None = None,
    ) -> None: ...

    def record_session_metrics(
        self,
        child_id: str,
        session_id: str,
        n_words_read: int,
        n_hesitations: int,
        duration_s: float,
        ts: datetime,
    ) -> None: ...

    def get_state(self, child_id: str) -> LearnerState: ...

    def due_reviews(self, child_id: str, now: datetime) -> list[str]: ...

    def get_card(self, child_id: str, skill: str) -> Card: ...

    def engagement_trend(
        self, child_id: str
    ) -> list[tuple[datetime, float, float]]: ...

    def get_last_k_observations(
        self,
        child_id: str,
        skill: str,
        k: int = 5,
    ) -> list[dict]: ...

    def record_served(
        self,
        child_id: str,
        skill: str,
        ts: datetime,
        reason: str,
    ) -> None: ...

    def get_served_log(self, child_id: str) -> list[dict]: ...


# ---------------------------------------------------------------------------
# Shared logic (backend-agnostic)
# ---------------------------------------------------------------------------


def _apply_bkt(prior: float, correct: bool, confidence: float) -> float:
    """Apply one soft-evidence BKT step with DEFAULT_BKT_PARAMS."""
    p = DEFAULT_BKT_PARAMS
    return bkt_update(prior, correct, confidence, p.s, p.g, p.t)


def _fsrs_update(card: Card, correct: bool, ts: datetime) -> Card:
    """Apply one FSRS review.  Rating mapping: correct→Good, incorrect→Again."""
    rating = Rating.Good if correct else Rating.Again
    updated, _ = _FSRS.review_card(card, rating, review_datetime=ts)
    return updated


def _wcpm(n_words: int, duration_s: float) -> float:
    """Words correct per minute = words / (duration_s / 60)."""
    return n_words / (duration_s / 60.0)


def _hrate(n_hesitations: int, n_words: int) -> float:
    """Hesitation rate = hesitations / words."""
    if n_words == 0:
        return 0.0
    return n_hesitations / n_words


# ---------------------------------------------------------------------------
# SQLite helpers (WAL discipline)
# ---------------------------------------------------------------------------

_DDL = """\
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learners (
    child_id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    child_id   TEXT NOT NULL REFERENCES learners(child_id)
);

CREATE TABLE IF NOT EXISTS observations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    child_id     TEXT    NOT NULL REFERENCES learners(child_id),
    skill        TEXT    NOT NULL,
    correct      INTEGER NOT NULL,
    confidence   REAL    NOT NULL,
    session_id   TEXT    NOT NULL,
    ts           TEXT    NOT NULL,
    miscue_class TEXT    NULL
);

CREATE TABLE IF NOT EXISTS mastery (
    child_id  TEXT NOT NULL REFERENCES learners(child_id),
    skill     TEXT NOT NULL,
    p_mastery REAL NOT NULL,
    PRIMARY KEY (child_id, skill)
);

CREATE TABLE IF NOT EXISTS reviews (
    child_id  TEXT NOT NULL REFERENCES learners(child_id),
    skill     TEXT NOT NULL,
    card_json TEXT NOT NULL,
    PRIMARY KEY (child_id, skill)
);

CREATE TABLE IF NOT EXISTS session_metrics (
    session_id   TEXT    PRIMARY KEY,
    child_id     TEXT    NOT NULL REFERENCES learners(child_id),
    n_words_read INTEGER NOT NULL,
    n_hesitations INTEGER NOT NULL,
    duration_s   REAL    NOT NULL,
    ts           TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS served_log (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    child_id TEXT    NOT NULL REFERENCES learners(child_id),
    skill    TEXT    NOT NULL,
    ts       TEXT    NOT NULL,
    reason   TEXT    NOT NULL
);
"""


def _open_connection(db_path: str) -> sqlite3.Connection:
    """Open a raw sqlite3 connection with WAL mode and busy timeout."""
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _session_scope(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One transaction boundary: commit on success, rollback+re-raise on exception.

    Pattern lifted from TheDose/src/thedose/infrastructure/persistence/database.py.
    The caller must NOT call conn.commit() themselves — this context manager
    owns the transaction.
    """
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _init_sqlite(db_path: str) -> sqlite3.Connection:
    """Create or open a db, run DDL, check/set schema_version."""
    conn = _open_connection(db_path)

    # Run DDL statements individually (sqlite3 module doesn't support
    # executescript inside a transaction cleanly with WAL)
    for stmt in _DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()

    # Check or set schema version
    row = conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
    else:
        stored = int(row["value"])
        if stored != SCHEMA_VERSION:
            conn.close()
            raise SchemaVersionError(
                f"Database schema version mismatch: expected {SCHEMA_VERSION}, "
                f"found {stored}. Manual migration required."
            )

    return conn


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


class SqliteLearnerStore:
    """SQLite-backed learner store — WAL mode, session_scope transactions.

    Not thread-safe across processes; WAL handles concurrent readers.
    """

    def __init__(self, db_path: str) -> None:
        self._conn = _init_sqlite(db_path)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_learner(self, conn: sqlite3.Connection, child_id: str) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO learners (child_id) VALUES (?)", (child_id,)
        )

    def _ensure_session(
        self, conn: sqlite3.Connection, child_id: str, session_id: str
    ) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, child_id) VALUES (?, ?)",
            (session_id, child_id),
        )

    def _get_mastery(
        self, conn: sqlite3.Connection, child_id: str, skill: str
    ) -> float:
        row = conn.execute(
            "SELECT p_mastery FROM mastery WHERE child_id=? AND skill=?",
            (child_id, skill),
        ).fetchone()
        return float(row["p_mastery"]) if row else DEFAULT_BKT_PARAMS.L0

    def _set_mastery(
        self, conn: sqlite3.Connection, child_id: str, skill: str, p: float
    ) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO mastery (child_id, skill, p_mastery) VALUES (?, ?, ?)",
            (child_id, skill, p),
        )

    def _get_card(
        self, conn: sqlite3.Connection, child_id: str, skill: str
    ) -> Card:
        row = conn.execute(
            "SELECT card_json FROM reviews WHERE child_id=? AND skill=?",
            (child_id, skill),
        ).fetchone()
        if row is None:
            return Card()
        return Card.from_dict(json.loads(row["card_json"]))

    def _set_card(
        self, conn: sqlite3.Connection, child_id: str, skill: str, card: Card
    ) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO reviews (child_id, skill, card_json) VALUES (?, ?, ?)",
            (child_id, skill, json.dumps(card.to_dict())),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_observation(
        self,
        child_id: str,
        skill: str,
        correct: bool,
        confidence: float,
        session_id: str,
        ts: datetime,
        miscue_class: str | None = None,
    ) -> None:
        """Record one observation; update BKT mastery and FSRS card atomically.

        miscue_class: optional tag (e.g. "substitution", "omission", "hesitation")
            used by the planner's prerequisite-edge gate.  None = generic.
        """
        # Validate confidence early — bkt_update would raise, but we want the
        # rollback to fire cleanly if it's caught by session_scope.
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence={confidence!r} is out of range [0, 1].")

        with _session_scope(self._conn) as conn:
            self._ensure_learner(conn, child_id)
            self._ensure_session(conn, child_id, session_id)

            # Write raw observation (miscue_class may be NULL)
            conn.execute(
                """INSERT INTO observations
                   (child_id, skill, correct, confidence, session_id, ts,
                    miscue_class)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (child_id, skill, int(correct), confidence, session_id,
                 ts.isoformat(), miscue_class),
            )

            # BKT update
            prior = self._get_mastery(conn, child_id, skill)
            new_p = _apply_bkt(prior, correct, confidence)
            self._set_mastery(conn, child_id, skill, new_p)

            # FSRS update
            card = self._get_card(conn, child_id, skill)
            updated_card = _fsrs_update(card, correct, ts)
            self._set_card(conn, child_id, skill, updated_card)

    def record_session_metrics(
        self,
        child_id: str,
        session_id: str,
        n_words_read: int,
        n_hesitations: int,
        duration_s: float,
        ts: datetime,
    ) -> None:
        """Store per-session engagement metrics (WCPM, hesitation rate)."""
        with _session_scope(self._conn) as conn:
            self._ensure_learner(conn, child_id)
            self._ensure_session(conn, child_id, session_id)
            conn.execute(
                """INSERT OR REPLACE INTO session_metrics
                   (session_id, child_id, n_words_read, n_hesitations, duration_s, ts)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, child_id, n_words_read, n_hesitations, duration_s,
                 ts.isoformat()),
            )

    def get_state(self, child_id: str) -> LearnerState:
        """Return the current LearnerState for a child."""
        conn = self._conn
        rows = conn.execute(
            "SELECT skill, p_mastery FROM mastery WHERE child_id=?", (child_id,)
        ).fetchall()
        mastery = {row["skill"]: float(row["p_mastery"]) for row in rows}
        due = self.due_reviews(child_id, datetime.now(timezone.utc))
        return LearnerState(child_id=child_id, mastery=mastery, due_reviews=due)

    def due_reviews(self, child_id: str, now: datetime) -> list[str]:
        """Return skills whose FSRS card.due <= now."""
        conn = self._conn
        rows = conn.execute(
            "SELECT skill, card_json FROM reviews WHERE child_id=?", (child_id,)
        ).fetchall()
        result = []
        for row in rows:
            card = Card.from_dict(json.loads(row["card_json"]))
            if card.due is not None and card.due <= now:
                result.append(row["skill"])
        return result

    def get_card(self, child_id: str, skill: str) -> Card:
        """Return the FSRS card for a specific child/skill pair."""
        return self._get_card(self._conn, child_id, skill)

    def engagement_trend(
        self, child_id: str
    ) -> list[tuple[datetime, float, float]]:
        """Return per-session (ts, wcpm, hesitation_rate) ordered by ts asc."""
        conn = self._conn
        rows = conn.execute(
            """SELECT ts, n_words_read, n_hesitations, duration_s
               FROM session_metrics
               WHERE child_id=?
               ORDER BY ts ASC""",
            (child_id,),
        ).fetchall()
        result = []
        for row in rows:
            ts = datetime.fromisoformat(row["ts"])
            wcpm = _wcpm(row["n_words_read"], row["duration_s"])
            hr = _hrate(row["n_hesitations"], row["n_words_read"])
            result.append((ts, wcpm, hr))
        return result

    def get_last_k_observations(
        self,
        child_id: str,
        skill: str,
        k: int = 5,
    ) -> list[dict]:
        """Return the last k observations for a child/skill pair, newest first.

        Each dict has keys: correct (bool), miscue_class (str | None), ts (str).
        """
        rows = self._conn.execute(
            """SELECT correct, miscue_class, ts
               FROM observations
               WHERE child_id=? AND skill=?
               ORDER BY id DESC
               LIMIT ?""",
            (child_id, skill, k),
        ).fetchall()
        return [
            {
                "correct": bool(row["correct"]),
                "miscue_class": row["miscue_class"],
                "ts": row["ts"],
            }
            for row in rows
        ]

    def record_served(
        self,
        child_id: str,
        skill: str,
        ts: datetime,
        reason: str,
    ) -> None:
        """Persist a served-item record (skill, ts, reason) to the served_log table."""
        with _session_scope(self._conn) as conn:
            self._ensure_learner(conn, child_id)
            conn.execute(
                """INSERT INTO served_log (child_id, skill, ts, reason)
                   VALUES (?, ?, ?, ?)""",
                (child_id, skill, ts.isoformat(), reason),
            )

    def get_served_log(self, child_id: str) -> list[dict]:
        """Return all served_log entries for a child, ordered by ts asc."""
        rows = self._conn.execute(
            """SELECT skill, ts, reason
               FROM served_log
               WHERE child_id=?
               ORDER BY id ASC""",
            (child_id,),
        ).fetchall()
        return [
            {"skill": row["skill"], "ts": row["ts"], "reason": row["reason"]}
            for row in rows
        ]


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


@dataclass
class _SessionMetricRow:
    session_id: str
    child_id: str
    n_words_read: int
    n_hesitations: int
    duration_s: float
    ts: datetime


@dataclass
class _ObservationRow:
    child_id: str
    skill: str
    correct: bool
    miscue_class: str | None
    ts: str  # ISO string, matches SQLite storage


@dataclass
class _ServedRow:
    child_id: str
    skill: str
    ts: str  # ISO string
    reason: str


class InMemoryLearnerStore:
    """Ephemeral in-memory learner store.

    Functionally identical to SqliteLearnerStore but stores data in dicts.
    Does NOT survive process restart or store.close() — document this clearly
    so callers do not accidentally rely on persistence.
    """

    def __init__(self) -> None:
        # child_id -> skill -> float
        self._mastery: dict[str, dict[str, float]] = {}
        # child_id -> skill -> Card
        self._cards: dict[str, dict[str, Card]] = {}
        # list of _SessionMetricRow
        self._metrics: list[_SessionMetricRow] = []
        # list of _ObservationRow (append-only, ordered by insertion)
        self._observations: list[_ObservationRow] = []
        # list of _ServedRow
        self._served: list[_ServedRow] = []

    def _get_mastery(self, child_id: str, skill: str) -> float:
        return self._mastery.get(child_id, {}).get(skill, DEFAULT_BKT_PARAMS.L0)

    def _set_mastery(self, child_id: str, skill: str, p: float) -> None:
        self._mastery.setdefault(child_id, {})[skill] = p

    def _get_card(self, child_id: str, skill: str) -> Card:
        return self._cards.get(child_id, {}).get(skill, Card())

    def _set_card(self, child_id: str, skill: str, card: Card) -> None:
        self._cards.setdefault(child_id, {})[skill] = card

    def record_observation(
        self,
        child_id: str,
        skill: str,
        correct: bool,
        confidence: float,
        session_id: str,
        ts: datetime,
        miscue_class: str | None = None,
    ) -> None:
        """Record one observation — BKT + FSRS update, no partial state on error.

        miscue_class: optional tag used by the planner's prerequisite-edge gate.
        """
        # Validate first; if anything below raises the _mastery dict is not touched
        # because Python assignment is atomic at the dict-entry level.
        # We deliberately compute both updates before writing either, mimicking
        # the transactional behaviour of the SQLite backend.
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence={confidence!r} is out of range [0, 1].")

        prior = self._get_mastery(child_id, skill)
        new_p = _apply_bkt(prior, correct, confidence)  # may raise (validated above)

        card = self._get_card(child_id, skill)
        updated_card = _fsrs_update(card, correct, ts)

        # Atomically (in CPython GIL sense) write both
        self._set_mastery(child_id, skill, new_p)
        self._set_card(child_id, skill, updated_card)
        self._observations.append(
            _ObservationRow(
                child_id=child_id,
                skill=skill,
                correct=correct,
                miscue_class=miscue_class,
                ts=ts.isoformat(),
            )
        )

    def record_session_metrics(
        self,
        child_id: str,
        session_id: str,
        n_words_read: int,
        n_hesitations: int,
        duration_s: float,
        ts: datetime,
    ) -> None:
        """Store per-session engagement metrics."""
        self._metrics.append(
            _SessionMetricRow(
                session_id=session_id,
                child_id=child_id,
                n_words_read=n_words_read,
                n_hesitations=n_hesitations,
                duration_s=duration_s,
                ts=ts,
            )
        )

    def get_state(self, child_id: str) -> LearnerState:
        mastery = dict(self._mastery.get(child_id, {}))
        due = self.due_reviews(child_id, datetime.now(timezone.utc))
        return LearnerState(child_id=child_id, mastery=mastery, due_reviews=due)

    def due_reviews(self, child_id: str, now: datetime) -> list[str]:
        result = []
        for skill, card in self._cards.get(child_id, {}).items():
            if card.due is not None and card.due <= now:
                result.append(skill)
        return result

    def get_card(self, child_id: str, skill: str) -> Card:
        return self._get_card(child_id, skill)

    def engagement_trend(
        self, child_id: str
    ) -> list[tuple[datetime, float, float]]:
        rows = sorted(
            [m for m in self._metrics if m.child_id == child_id],
            key=lambda r: r.ts,
        )
        return [
            (r.ts, _wcpm(r.n_words_read, r.duration_s),
             _hrate(r.n_hesitations, r.n_words_read))
            for r in rows
        ]

    def get_last_k_observations(
        self,
        child_id: str,
        skill: str,
        k: int = 5,
    ) -> list[dict]:
        """Return the last k observations for a child/skill pair, newest first."""
        matching = [
            o for o in self._observations
            if o.child_id == child_id and o.skill == skill
        ]
        # Return newest first (last inserted = most recent)
        recent = matching[-k:] if len(matching) > k else matching
        recent = list(reversed(recent))
        return [
            {
                "correct": o.correct,
                "miscue_class": o.miscue_class,
                "ts": o.ts,
            }
            for o in recent
        ]

    def record_served(
        self,
        child_id: str,
        skill: str,
        ts: datetime,
        reason: str,
    ) -> None:
        """Persist a served-item record (skill, ts, reason)."""
        self._served.append(
            _ServedRow(child_id=child_id, skill=skill, ts=ts.isoformat(), reason=reason)
        )

    def get_served_log(self, child_id: str) -> list[dict]:
        """Return all served entries for a child, ordered by insertion."""
        return [
            {"skill": r.skill, "ts": r.ts, "reason": r.reason}
            for r in self._served
            if r.child_id == child_id
        ]

    def close(self) -> None:
        """No-op; provided for API symmetry with SqliteLearnerStore."""


# ---------------------------------------------------------------------------
# LearnerModel facade (keeps backward-compat with learner_model.py stub API)
# ---------------------------------------------------------------------------


class LearnerModel:
    """BKT mastery + FSRS review scheduling, persisted via a store backend.

    Default backend is SqliteLearnerStore.  Pass store= to inject a custom
    backend (e.g. InMemoryLearnerStore for tests).
    """

    def __init__(
        self,
        db_path: str | None = None,
        store: LearnerStoreProtocol | None = None,
    ) -> None:
        if store is not None:
            self._store = store
        elif db_path is not None:
            self._store = SqliteLearnerStore(db_path)
        else:
            raise ValueError("Provide either db_path or store.")

    def update(self, child_id: str, observations: list) -> None:
        """Fold observations (readcoach.learner_model.Observation) into store."""
        now = datetime.now(timezone.utc)
        for i, obs in enumerate(observations):
            self._store.record_observation(
                child_id=child_id,
                skill=obs.skill,
                correct=obs.correct,
                confidence=obs.confidence,
                session_id="default",
                ts=now,
            )

    def state(self, child_id: str) -> LearnerState:
        return self._store.get_state(child_id)
