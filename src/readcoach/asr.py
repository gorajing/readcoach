"""ASR layer — swappable on purpose. The interesting measurement lives above it.

Backends sit behind one interface so accuracy vs. latency can be compared:
  - faster-whisper (default)
  - whisper.cpp / Moonshine v2  (on-device stretch)

The key feature is the *bias-strength knob*: prompting/biasing the model with the
known passage text lowers WER (arXiv:2505.23627, 2506.11079) but can also make the
model hallucinate the expected text over real reading errors — that tradeoff is what
this project measures.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    text: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None


@dataclass
class AsrResult:
    text: str
    words: list[Word] = field(default_factory=list)
    rtf: float | None = None  # real-time factor (latency proxy)
    backend: str = "faster-whisper"


def transcribe(audio_path: str, target_text: str | None = None,
               backend: str = "faster-whisper") -> AsrResult:
    """Transcribe ``audio_path``. If ``target_text`` is given, apply the target-text
    prior (initial_prompt / biasing). Day 1: implement faster-whisper; report RTF.
    """
    raise NotImplementedError("Day 1: faster-whisper backend + target-text prior")
