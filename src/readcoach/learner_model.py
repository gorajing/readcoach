"""Learner model — per-child, per-skill mastery from noisy, speech-derived observations.

- Mastery: BKT per phonics skill (advance at 0.95). Simple beats deep here
  (arXiv:2206.11460, 2302.06881), and population priors mitigate cold start.
- Observations carry detector confidence, folded in as virtual evidence (Pearl) —
  a noisy observation should move the posterior less than a certain one.
- Review scheduling: FSRS (which grapheme/sight-word to review when).
- State: SQLite by default (Redis behind a flag).
- Surfaced via an open-learner dashboard (mastery heatmap, WCPM growth, review queue).

Honesty: observations come from imperfect miscue detection, so the model and its eval
must acknowledge observation uncertainty (calibration, not just AUC).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Observation:
    skill: str       # e.g. "vowel_team_ea", "digraph_ch", "cvc_blend"
    correct: bool
    confidence: float = 1.0  # how sure the miscue detector is (propagates uncertainty)


@dataclass
class LearnerState:
    child_id: str
    mastery: dict[str, float] = field(default_factory=dict)  # skill -> P(mastered)
    due_reviews: list[str] = field(default_factory=list)

    def gaps(self, threshold: float = 0.95) -> list[str]:
        return [s for s, p in self.mastery.items() if p < threshold]


class LearnerModel:
    """pyBKT mastery + FSRS review, persisted in SQLite (Redis behind a flag)."""

    def __init__(self, db_path: str | None = None) -> None:
        raise NotImplementedError("Day 3: pyBKT + py-fsrs + SQLite-backed state")

    def update(self, child_id: str, observations: list[Observation]) -> None:
        """Fold new observations into per-skill mastery + the review schedule."""
        raise NotImplementedError

    def state(self, child_id: str) -> LearnerState:
        raise NotImplementedError
