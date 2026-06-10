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
    """BKT mastery + FSRS review, persisted in SQLite (Redis behind a flag).

    Delegates to readcoach.learner_store.LearnerModel; kept here so existing
    imports continue to work (``from readcoach.learner_model import LearnerModel``).
    """

    def __init__(self, db_path: str | None = None) -> None:
        # Lazy import to avoid circular dependency during module initialisation.
        from readcoach.learner_store import LearnerModel as _Impl  # noqa: PLC0415
        self._impl = _Impl(db_path=db_path)

    def update(self, child_id: str, observations: list[Observation]) -> None:
        """Fold new observations into per-skill mastery + the review schedule."""
        self._impl.update(child_id, observations)

    def state(self, child_id: str) -> LearnerState:
        return self._impl.state(child_id)
