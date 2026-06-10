"""Miscue detection — the hard, measurable part (transcription != miscue detection).

Align the ASR hypothesis to the KNOWN target text and classify each deviation.
Even with target-text prompting, per-class miscue F1 sits well below transcription
accuracy (arXiv:2406.07060, 2505.23627, 2506.11079) — this is where the honest
numbers live.

Deliverable: `python -m readcoach.miscue --demo` prints per-class precision/recall/F1
across bias settings on the synthetic benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MiscueType = Literal["substitution", "omission", "insertion", "self_correction", "hesitation"]


@dataclass
class Miscue:
    type: MiscueType
    target_word: str | None
    said_word: str | None
    index: int  # position in the target text


def detect(hypothesis: str, target_text: str) -> list[Miscue]:
    """Align ``hypothesis`` to ``target_text`` (jiwer edit ops) and classify deviations.

    Heuristics: substitution/omission/insertion from the op stream; self_correction when a
    wrong token is immediately followed by the correct target token; hesitation from
    fillers / repeats (+ word timings if available). Keep false positives LOW — a
    false "correction" on a child who read correctly is a product-killing failure.
    """
    raise NotImplementedError("Day 1: jiwer alignment + deviation classification")


def score(predicted: list[Miscue], gold: list[Miscue]) -> dict[str, float]:
    """Return precision / recall / f1 for miscue detection against gold labels."""
    raise NotImplementedError("Day 1: miscue precision/recall/F1")
