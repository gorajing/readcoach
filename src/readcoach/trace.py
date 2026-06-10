"""SessionTrace — the record the policy compiler audits and the harness emits.

T4.3 — the utterance/invariant layer's shared data shape.
================================================================

A *trace* is the inspectable, replayable record of one tutoring session: an
ordered list of turns, each capturing the decision context (page position,
miscue class), the chosen move + scaffold rung, the SERVED reason for a quest
item, the LLM-verbalized utterance (if any), and whether the turn carried the
periodic AI-identity reminder.

The trace is **frozen** (``TurnRecord`` / ``SessionTrace`` are frozen
dataclasses, turns stored as a tuple) so a logged session cannot be mutated
out from under the checks that consume it.  JSON (de)serialization is provided
so a trace can be written to ``evals/results/*.jsonl`` and re-loaded by the
audit; deserialization is FAIL-LOUD — a missing required field, an unknown
field, or a negative turn index raises rather than silently defaulting.

This module deliberately holds NO policy: it is the substrate the
``policy_compiler`` checks run over.  The move/hint/served-reason strings mirror
``readcoach.tutor`` (``Move`` / ``HintLevel``) but are stored as plain strings
so a trace can be (de)serialized without importing the policy.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class TurnRecord:
    """One turn of a tutoring session.

    Fields
    ------
    turn_index            0-based position in the session (must be >= 0).
    at_page_end           was the child at a page boundary on this turn?
    miscue_type           the miscue class driving this turn, or None.
    action_move           the move the policy chose (``readcoach.tutor.Move``).
    hint_level            scaffold-ladder rung for SCAFFOLDED_HINT, else None.
    served_reason         why a quest item was served this turn
                          ("new" | "review" | ...), or None when no item served.
    utterance             the LLM-verbalized text said to the child, or None.
    is_ai_reminder        did this turn carry the periodic AI-identity reminder?
    skill_id              the phonics skill this turn served, or None.
    """

    turn_index: int
    at_page_end: bool
    miscue_type: str | None
    action_move: str | None
    hint_level: str | None
    served_reason: str | None
    utterance: str | None
    is_ai_reminder: bool
    skill_id: str | None = None

    def __post_init__(self) -> None:
        if self.turn_index < 0:
            raise ValueError(
                f"turn_index must be >= 0; got {self.turn_index}"
            )


@dataclass(frozen=True)
class SessionTrace:
    """A full tutoring session: ordered turns plus session-level context.

    ``completed_skills_at_start`` is the set of skills the child had already
    mastered when this session opened — the ``never_reserves_completed_item``
    check needs it to distinguish a legitimate first-serve from a re-serve of
    already-completed content.  Stored as a tuple (frozen/hashable).
    """

    child_id: str
    policy_version: str
    completed_skills_at_start: tuple[str, ...]
    turns: tuple[TurnRecord, ...]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

_TURN_FIELD_NAMES = frozenset(f.name for f in fields(TurnRecord))
_TRACE_FIELD_NAMES = frozenset(f.name for f in fields(SessionTrace))


def trace_to_dict(trace: SessionTrace) -> dict[str, Any]:
    """Plain-dict form of a trace (JSON-serializable; lists, not tuples)."""
    return {
        "child_id": trace.child_id,
        "policy_version": trace.policy_version,
        "completed_skills_at_start": list(trace.completed_skills_at_start),
        "turns": [asdict(t) for t in trace.turns],
    }


def _turn_from_dict(d: dict[str, Any]) -> TurnRecord:
    if not isinstance(d, dict):
        raise ValueError(f"turn must be an object; got {type(d).__name__}")
    keys = set(d)
    unknown = keys - _TURN_FIELD_NAMES
    if unknown:
        raise ValueError(f"unknown turn field(s): {sorted(unknown)}")
    # Required fields = every field without a default.  ``skill_id`` is the only
    # optional one; everything else must be present (missing -> KeyError).
    missing = (_TURN_FIELD_NAMES - {"skill_id"}) - keys
    if missing:
        raise KeyError(f"turn missing required field(s): {sorted(missing)}")
    return TurnRecord(
        turn_index=d["turn_index"],
        at_page_end=d["at_page_end"],
        miscue_type=d["miscue_type"],
        action_move=d["action_move"],
        hint_level=d["hint_level"],
        served_reason=d["served_reason"],
        utterance=d["utterance"],
        is_ai_reminder=d["is_ai_reminder"],
        skill_id=d.get("skill_id"),
    )


def trace_from_dict(d: dict[str, Any]) -> SessionTrace:
    """Reconstruct a SessionTrace from its dict form.  Fail-loud.

    Raises ``KeyError`` on a missing required field, ``ValueError`` on an
    unknown field / wrong shape, propagating from ``TurnRecord.__post_init__``
    for an invalid turn_index.
    """
    if not isinstance(d, dict):
        raise ValueError(f"trace must be an object; got {type(d).__name__}")
    unknown = set(d) - _TRACE_FIELD_NAMES
    if unknown:
        raise ValueError(f"unknown trace field(s): {sorted(unknown)}")
    missing = _TRACE_FIELD_NAMES - set(d)
    if missing:
        raise KeyError(f"trace missing required field(s): {sorted(missing)}")
    raw_turns = d["turns"]
    if not isinstance(raw_turns, list):
        raise ValueError(
            f"'turns' must be a list; got {type(raw_turns).__name__}"
        )
    raw_completed = d["completed_skills_at_start"]
    if not isinstance(raw_completed, list):
        raise ValueError(
            f"'completed_skills_at_start' must be a list; got "
            f"{type(raw_completed).__name__}"
        )
    return SessionTrace(
        child_id=d["child_id"],
        policy_version=d["policy_version"],
        completed_skills_at_start=tuple(raw_completed),
        turns=tuple(_turn_from_dict(t) for t in raw_turns),
    )


def trace_to_json(trace: SessionTrace, *, indent: int | None = None) -> str:
    """Serialize a trace to a JSON string."""
    return json.dumps(trace_to_dict(trace), indent=indent, sort_keys=True)


def trace_from_json(text: str) -> SessionTrace:
    """Parse a trace from JSON.  Malformed JSON raises ``JSONDecodeError``."""
    return trace_from_dict(json.loads(text))
