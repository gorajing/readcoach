"""The eval flywheel — prove the tutor improves, gated, with a validated judge.

Pieces (docs/ARCHITECTURE.md):
  - golden set of (passage, child-read-attempt, tutor-response) cases
  - LLM judge: literacy axes (miscue-F1, WCPM, intervention appropriateness) + pedagogy
    axes (BEA: mistake-ID / location / guidance / actionability; + ICAP engagement)
  - judge VALIDATION: judge-vs-human agreement per dimension on ~25 hand-labeled turns
  - simulated students (emergent / ELL-accented / dyslexic) for OFFLINE A/B
  - child-safety eval + decodability scorer
  - golden-invariant regression tests + a CI gate (beat prior rubric AND stay under latency)
  - slices by accent / persona

Day 2 builds the skeleton (golden set + evaluate + compare + gate); Day 4 adds the
judge, judge-validation, simulated students, and the improvement proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalReport:
    version: str
    rubric: dict[str, float] = field(default_factory=dict)   # dimension -> score
    miscue_f1: float | None = None
    latency_ms_p50: float | None = None
    invariant_violations: list[str] = field(default_factory=list)
    slices: dict[str, dict] = field(default_factory=dict)     # e.g. by accent/persona


def evaluate(tutor_version: str, golden_set: list[dict]) -> EvalReport:
    """Score a tutor version on the golden set. Day 2: skeleton; Day 4: full judge."""
    raise NotImplementedError


def compare(prev: EvalReport, new: EvalReport, latency_budget_ms: float) -> dict:
    """The regression GATE: 'new' ships only if it beats 'prev' on the rubric AND stays
    under the latency budget AND violates no golden invariant. Returns {ship: bool, ...}.
    """
    raise NotImplementedError


def validate_judge(hand_labeled_turns: list[dict]) -> dict[str, float]:
    """Report judge-vs-human agreement PER DIMENSION. A judge you haven't measured is
    theater (BEA 2025: judge F1 0.82 vs human 0.91, dimension-dependent)."""
    raise NotImplementedError
