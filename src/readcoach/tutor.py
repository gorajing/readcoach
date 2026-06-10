"""Tutoring agent harness — an explicit decision policy, NOT a chat wrapper.

A raw LLM is RLHF'd to be helpful and give answers — the opposite of teaching. The
harness enforces the withholding/scaffolding policy the model won't hold on its own
(MathTutorBench arXiv:2502.18940; MetaCLASS arXiv:2602.02457; Tutor CoPilot).

Decision: given (miscue, learner_state, position_in_page) -> one discrete move.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .learner_model import LearnerState
from .miscue import Miscue

Move = Literal["WAIT", "ENCOURAGE", "SCAFFOLDED_HINT", "MODEL_THE_WORD", "NEXT_ITEM"]


@dataclass
class TutorContext:
    miscue: Miscue | None
    learner_state: LearnerState
    at_page_end: bool
    consecutive_struggles: int = 0


@dataclass
class TutorAction:
    move: Move
    target_word: str | None
    rationale: str       # inspectable: WHY this move (for the eval + the demo)
    error_type: str | None = None  # for SCAFFOLDED_HINT (sound-card / segment-blend / ...)


# Golden invariants (regression-tested in evals/): the harness must NEVER violate these.
GOLDEN_INVARIANTS = (
    "never_says_wrong",          # protect motivation
    "never_coaches_mid_page",    # coach at page-end, like a teacher
    "never_false_positive_correction",  # don't 'correct' a self-correction or ASR error
)


def decide(ctx: TutorContext) -> TutorAction:
    """Choose a single pedagogical move.

    Rules (Day 3):
      - coach at page-end, not mid-page;
      - protect productive struggle — intervene only on UNPRODUCTIVE struggle
        (repeated failure / no progress), not on every miscue (MetaCLASS: aim ~35-50% WAIT);
      - scaffold support by error type (sound-card -> segment-and-blend -> model the word);
      - pick NEXT_ITEM at the edge of the ZPD using learner_state.mastery.
    """
    raise NotImplementedError("Day 3: discrete move policy + intervention-timing gate")
