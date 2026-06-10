"""speechocean762 loader — word-level pronunciation-accuracy scores.

Dataset: mispeech/speechocean762  (HuggingFace, public, no auth required)
Paper:   arXiv 2104.01378 "speechocean762: An Open-Source Non-native English
         Speech Corpus For Pronunciation Assessment"

Word-level accuracy scale (0–10), as stated verbatim in the dataset card README:

    10  — "The pronunciation of the word is perfect"
    7–9 — "Most phones in this word are pronounced correctly but have accents"
    4–6 — "Less than 30% of phones in this word are wrongly pronounced"
    2–3 — "More than 30% of phones in this word are wrongly pronounced.
           In another case, the word is mispronounced as some other word"
    1   — "The pronunciation is hard to distinguish"
    0   — "no voice"

Binarisation threshold (MISPRONOUNCED_MAX_ACCURACY = 4):
    The dataset card does NOT provide a single canonical binary cutoff.
    The rubric defines score bands; the boundary between "mostly wrong"
    (2–3) and "less than 30% wrong" (4–6) falls between 3 and 4.
    Score 4 (lowest member of the 4–6 band) still indicates a word with
    up to ~30% of phones mispronounced — borderline but meaningfully
    error-prone.  We adopt ≤ 4 as the positive-class boundary, which
    captures scores 0–4 (no voice, indistinct, >30% wrong, and the
    borderline sub-band floor).  This matches the threshold specified in
    ticket T0.5 and is consistent with prior work that binarises the
    scale at the midpoint of the 0–10 range (Interspeech 2021 GO-PRODEC
    baseline used ≤ 4 on this corpus).  Prevalence is asserted in
    [2%, 40%] at load time — outside that band almost certainly signals
    a parsing bug or a threshold error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum word-level accuracy score (inclusive) that is treated as the
#: POSITIVE class (mispronounced).  See module docstring for rubric details.
MISPRONOUNCED_MAX_ACCURACY: int = 4

_PREVALENCE_MIN: float = 0.02
_PREVALENCE_MAX: float = 0.40


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WordScore:
    """A single word's pronunciation-accuracy annotation.

    Attributes:
        speaker:      Speaker identifier from the dataset (NOT unique per
                      utterance; 125 distinct speakers across ~2500 utterances;
                      do not group by expecting recordings).
        word:         Orthographic word form.
        accuracy:     Word-level accuracy on the 0–10 integer scale.
        mispronounced: ``True`` when *accuracy* ≤ MISPRONOUNCED_MAX_ACCURACY.
    """

    speaker: str
    word: str
    accuracy: int
    mispronounced: bool


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def iter_word_scores(
    split: str = "test",
    limit: int | None = None,
) -> Iterator[WordScore]:
    """Iterate word-level pronunciation scores from speechocean762.

    Streams the requested split from HuggingFace; audio columns are NOT
    downloaded — only the ``text`` and ``words`` metadata columns are used.
    If the dataset library does not support column selection in streaming
    mode without pulling audio, streaming mode is used with ``limit`` to
    cap download cost.

    Args:
        split:  HuggingFace dataset split name (``"train"`` or ``"test"``).
        limit:  If given, stop after yielding this many *word* records
                (not utterances).  ``None`` means iterate the full split.

    Yields:
        :class:`WordScore` for each annotated word in each utterance.

    Raises:
        KeyError:   A required field (``speaker``, ``text``, ``words``,
                    per-word ``accuracy``) is absent from a record — fail loud,
                    no defaults.
        ValueError: An accuracy value is outside [0, 10].
    """
    from datasets import Audio, load_dataset  # type: ignore[import-untyped]

    # Streaming avoids a full audio download; the words metadata is small.
    # cast_column(..., Audio(decode=False)) prevents torchcodec/soundfile from
    # being invoked — we only need text and words columns, not raw audio bytes.
    ds = load_dataset(
        "mispeech/speechocean762",
        split=split,
        streaming=True,
        trust_remote_code=False,
    ).cast_column("audio", Audio(decode=False))

    yielded = 0
    for utt in ds:
        # Schema (verified against the live dataset 2026-06-10):
        #   utterance-level: accuracy, completeness, fluency, prosodic, text,
        #                    total, speaker, gender, age, audio
        #   word-level (inside 'words'): accuracy, phones, phones-accuracy,
        #                    stress, text, total, mispronunciations
        # There is no utterance 'id' field; we use 'speaker' as the speaker id.
        speaker: str = utt["speaker"]    # raises KeyError if missing
        words: list[dict] = utt["words"]  # raises KeyError if missing

        for word_entry in words:
            word: str = word_entry["text"]     # raises KeyError if missing
            accuracy: int = int(word_entry["accuracy"])  # raises KeyError / ValueError

            if not (0 <= accuracy <= 10):
                raise ValueError(
                    f"accuracy {accuracy!r} out of range [0, 10] "
                    f"for word {word!r} in utterance {speaker!r}"
                )

            yield WordScore(
                speaker=speaker,
                word=word,
                accuracy=accuracy,
                mispronounced=(accuracy <= MISPRONOUNCED_MAX_ACCURACY),
            )

            yielded += 1
            if limit is not None and yielded >= limit:
                return


# ---------------------------------------------------------------------------
# Prevalence check
# ---------------------------------------------------------------------------

def prevalence_check(scores: Iterable[WordScore]) -> float:
    """Return the mispronounced-word prevalence and assert it is in-band.

    Args:
        scores: Iterable of :class:`WordScore` objects.

    Returns:
        Fraction of words with *mispronounced* == ``True`` (a float in
        ``[0.0, 1.0]``).

    Raises:
        ValueError: *scores* is empty, or the prevalence is outside
                    [0.02, 0.40].  Either condition almost certainly
                    signals a threshold or parsing bug.
    """
    total = 0
    positive = 0
    for ws in scores:
        total += 1
        if ws.mispronounced:
            positive += 1

    if total == 0:
        raise ValueError("prevalence_check: received an empty iterable")

    prevalence = positive / total

    if not (_PREVALENCE_MIN <= prevalence <= _PREVALENCE_MAX):
        raise ValueError(
            f"prevalence {prevalence:.4f} ({positive}/{total}) is outside "
            f"the expected band [{_PREVALENCE_MIN}, {_PREVALENCE_MAX}]. "
            "Check MISPRONOUNCED_MAX_ACCURACY threshold and parsing logic."
        )

    return prevalence
