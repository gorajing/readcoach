"""Miscue detection — the hard, measurable part (transcription != miscue detection).

Align the ASR hypothesis to the KNOWN target text and classify each deviation.
Even with target-text prompting, per-class miscue F1 sits well below transcription
accuracy (arXiv:2406.07060, 2505.23627, 2506.11079) — this is where the honest
numbers live.

Deliverable: `python -m readcoach.miscue --demo` prints per-class precision/recall/F1
across bias settings on the synthetic benchmark.

===========================================================================
Design notes (T1.2)
===========================================================================
Two input modes, both required:

  * text-only  — ``detect(hyp_str, target)``.  No word timings, so the
    timing-based hesitation rule is INACTIVE.  This is the audio-free upper
    bound a later ticket round-trips against.
  * timed      — ``detect(list[Word] | AsrResult, target)``.  Full 5-class:
    the inter-word-gap hesitation rule is live.

Normalization: ``Word.text`` from Whisper carries a LEADING SPACE (" cat"),
punctuation is token-attached (" mat."), and casing/punctuation vary with the
bias setting.  Every token is normalized (strip whitespace + leading/trailing
punctuation, casefold) before alignment; Miscue fields report the normalized
form.

Filler lexicon (um/uh/er/hmm/mm) is removed from the hypothesis BEFORE
alignment so fillers never surface as insertions; each removed filler instead
emits a hesitation bound to the next surviving target index.

Self-correction has two jiwer alignment shapes:
  (a) insert(wrong) + equal(correct)            — what jiwer 4.0 emits
  (b) substitute(correct->wrong) + insert(correct)  — emitted by other backends
Both are classified as ``self_correction``.

Precedence (a token yields at most ONE miscue):
  self_correction  >  repeat-hesitation  >  substitution / insertion / omission
Timing/filler hesitations are added on top and never collide with an alignment
miscue at the same hypothesis position (the filler/gap word is not part of the
alignment op stream).

``confidence`` on a Miscue: rule-based detectors emit 1.0 (no principled
discount exists yet); the field is here for T3's learner model, which consumes
per-observation confidence.  Keep values honest — 1.0 unless a real discount
is justified.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Literal

import jiwer

MiscueType = Literal["substitution", "omission", "insertion", "self_correction", "hesitation"]

# Inter-word gap (seconds) above which a hesitation is flagged in timed mode.
# PLACEHOLDER until real-kid timing data exists — child readers pause far more
# than adults, so 1.0s is a deliberately conservative starting point.
GAP_THRESHOLD_S = 1.0

# Disfluency fillers, stripped from the hypothesis BEFORE alignment.  Normalized
# (casefold + punctuation-stripped) forms.  A filler lexicon alone has ~0 expected
# recall (Whisper usually normalizes fillers out of its transcript), but it is kept
# as a cheap second signal alongside the timing rule.
_FILLERS = frozenset({"um", "uh", "er", "hmm", "mm", "uhh", "umm", "erm"})

# Characters stripped from token edges during normalization (leading/trailing only).
_EDGE_PUNCT = ".,!?;:\"'`()[]{}…—–-"

# A "wrong attempt followed by the correct word" only counts as a SELF-CORRECTION
# when the wrong token is an orthographic near-miss of the target word (a partial
# decoding: "bat"->"cat"), not an unrelated inserted word ("big cat").  We have no
# phonetic model, so SequenceMatcher char-ratio is the proxy; 0.5 cleanly separates
# sound-out attempts (bat/cat=0.67, ran/run=0.67) from real insertions (big/cat=0.0,
# slow/quick=0.0).  Self-corrections detected via this heuristic carry a confidence
# discount (the structural rule alone is ambiguous without phonetics).
_SELF_CORRECTION_SIM_THRESHOLD = 0.5
_SELF_CORRECTION_CONFIDENCE = 0.7


def _orthographic_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

_ALL_CLASSES: tuple[MiscueType, ...] = (
    "substitution",
    "omission",
    "insertion",
    "self_correction",
    "hesitation",
)


@dataclass
class Miscue:
    type: MiscueType
    target_word: str | None
    said_word: str | None
    index: int  # position in the target text
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize(token: str) -> str:
    """Strip whitespace + leading/trailing punctuation, then casefold.

    Mirrors the empirical Whisper token shape (" Cat." -> "cat").  Internal
    punctuation (e.g. "don't") is preserved; only edge punctuation is stripped.
    """
    return token.strip().strip(_EDGE_PUNCT).casefold()


# ---------------------------------------------------------------------------
# Input handling — unify str / list[Word] / AsrResult into normalized tokens
# (plus the surviving Word objects in timed mode) with fillers removed.
# ---------------------------------------------------------------------------

@dataclass
class _Hyp:
    """Normalized hypothesis after filler removal.

    tokens         : normalized non-filler tokens, in order (what we align).
    timed          : aligned Word objects (same length as ``tokens``) or None
                     in text-only mode.
    filler_after   : map {surviving-token-index -> count of fillers that
                     appeared immediately BEFORE that token}.  Index ``len(tokens)``
                     keys trailing fillers.  Used to emit filler hesitations.
    """

    tokens: list[str]
    timed: list["object"] | None  # list[Word] | None  (Word imported lazily for typing)
    filler_before: dict[int, list[str]]


def _build_hyp(hypothesis) -> _Hyp:  # noqa: ANN001
    from readcoach.asr import AsrResult, Word

    raw_words: list[Word] | None
    if isinstance(hypothesis, str):
        raw_words = None
        raw_tokens = [_normalize(t) for t in hypothesis.split()]
    elif isinstance(hypothesis, AsrResult):
        raw_words = list(hypothesis.words)
        raw_tokens = [_normalize(w.text) for w in raw_words]
    elif isinstance(hypothesis, list):
        if hypothesis and not isinstance(hypothesis[0], Word):
            raise TypeError(
                f"detect() list input must be list[Word]; got element of type "
                f"{type(hypothesis[0]).__name__}"
            )
        raw_words = list(hypothesis)
        raw_tokens = [_normalize(w.text) for w in raw_words]
    else:
        raise TypeError(
            "detect() hypothesis must be str | AsrResult | list[Word]; "
            f"got {type(hypothesis).__name__}"
        )

    tokens: list[str] = []
    timed: list[Word] | None = [] if raw_words is not None else None
    filler_before: dict[int, list[str]] = {}

    for i, tok in enumerate(raw_tokens):
        if tok == "":
            # Pure-punctuation / empty token — drop it (not a spoken word).
            continue
        if tok in _FILLERS:
            filler_before.setdefault(len(tokens), []).append(tok)
            continue
        tokens.append(tok)
        if timed is not None:
            timed.append(raw_words[i])

    return _Hyp(tokens=tokens, timed=timed, filler_before=filler_before)


# ---------------------------------------------------------------------------
# Alignment-chunk classification (the load-bearing seam — independently tested)
# ---------------------------------------------------------------------------

def _classify_alignment(
    chunks,  # noqa: ANN001  list[jiwer AlignmentChunk]
    target_norm: list[str],
    hyp_norm: list[str],
) -> list[Miscue]:
    """Classify a jiwer alignment-chunk stream into Miscues.

    Pure function of (chunks, normalized target, normalized hypothesis).  Handles
    both self-correction variants and the repeat-vs-correction disambiguation.

    Precedence per token: self_correction > repeat-hesitation > sub/ins/omit.
    A hypothesis token consumed as the "wrong attempt" of a self-correction, or
    as a repeat, is not re-emitted as an insertion/substitution.
    """
    miscues: list[Miscue] = []
    n = len(chunks)
    i = 0
    while i < n:
        c = chunks[i]
        ctype = c.type

        if ctype == "equal":
            i += 1
            continue

        if ctype == "delete":
            # Omission: target words present, no hypothesis words.
            for r in range(c.ref_start_idx, c.ref_end_idx):
                miscues.append(
                    Miscue("omission", target_word=target_norm[r], said_word=None, index=r)
                )
            i += 1
            continue

        if ctype == "substitute":
            # Each ref/hyp pair in the span is one substitution UNLESS it forms
            # variant-(b) self-correction with a following insert of the correct
            # word, or the said word is a repeat.
            span = c.ref_end_idx - c.ref_start_idx
            for k in range(span):
                ref_idx = c.ref_start_idx + k
                hyp_idx = c.hyp_start_idx + k
                said = hyp_norm[hyp_idx]
                target = target_norm[ref_idx]

                # Variant (b): substitute(target->said) immediately followed by an
                # insert whose token == the correct target word -> self_correction.
                # Only triggers for the LAST pair of the span (the insert chunk
                # follows the whole substitute chunk).
                if k == span - 1 and i + 1 < n:
                    nxt = chunks[i + 1]
                    if (
                        nxt.type == "insert"
                        and nxt.hyp_end_idx - nxt.hyp_start_idx >= 1
                        and hyp_norm[nxt.hyp_start_idx] == target
                        and said != target
                        and _orthographic_similarity(said, target)
                        >= _SELF_CORRECTION_SIM_THRESHOLD
                    ):
                        miscues.append(
                            Miscue(
                                "self_correction",
                                target_word=target,
                                said_word=said,
                                index=ref_idx,
                                confidence=_SELF_CORRECTION_CONFIDENCE,
                            )
                        )
                        # Consume exactly the one correcting insert token.
                        _emit_insert_tail(miscues, nxt, target_norm, hyp_norm, skip_first=True)
                        i += 2
                        break

                # Repeat: said word equals the immediately preceding target word.
                if ref_idx > 0 and said == target_norm[ref_idx - 1]:
                    miscues.append(
                        Miscue("hesitation", target_word=None, said_word=said, index=ref_idx - 1)
                    )
                    continue

                miscues.append(
                    Miscue("substitution", target_word=target, said_word=said, index=ref_idx)
                )
            else:
                i += 1
                continue
            # (break path) already advanced i
            continue

        if ctype == "insert":
            # Variant (a) self-correction lives here: insert(wrong) followed by an
            # equal that reads the correct target word at the same target index.
            # Delegated to _handle_insert (also handles repeat-hesitation / plain
            # insertion).
            consumed = _handle_insert(miscues, chunks, i, target_norm, hyp_norm)
            i += consumed
            continue

        raise ValueError(f"Unknown jiwer alignment chunk type: {ctype!r}")

    return miscues


def _handle_insert(
    miscues: list[Miscue],
    chunks,  # noqa: ANN001
    i: int,
    target_norm: list[str],
    hyp_norm: list[str],
) -> int:
    """Classify an ``insert`` chunk.  Returns how many chunks were consumed.

    Precedence (per the controller): self_correction > repeat-hesitation > insertion.
    But a REPEAT (inserted token == preceding target word) is always a disfluency,
    never a correction attempt, so the repeat test gates the self_correction branch.

    For the last inserted token (the one adjacent to the following equal), decide:
      - repeat-hesitation: it equals the immediately preceding target word
        (``target[ref_at - 1]``) -> a repeat disfluency ("the the").
      - self_correction (variant a): it is an orthographic near-miss of the next
        target word (a partial decoding: "bat"->"cat"), AND that target word is
        read next (an ``equal`` chunk begins at ``ref_at``).
      - plain insertion otherwise ("big cat" — unrelated inserted word).
    Earlier inserted tokens in a multi-token chunk are classified the same way.
    """
    c = chunks[i]
    ref_at = c.ref_start_idx  # target position the insertion sits BEFORE
    nxt = chunks[i + 1] if i + 1 < len(chunks) else None

    inserted = list(range(c.hyp_start_idx, c.hyp_end_idx))

    last_hyp = inserted[-1]
    wrong = hyp_norm[last_hyp]
    is_repeat = ref_at > 0 and wrong == target_norm[ref_at - 1]

    # Self-correction variant (a): wrong attempt is an orthographic near-miss of the
    # correct target word read next.  Repeats never qualify (they are disfluencies).
    if (
        not is_repeat
        and nxt is not None
        and nxt.type == "equal"
        and nxt.ref_start_idx == ref_at
        and ref_at < len(target_norm)
    ):
        correct = target_norm[ref_at]
        if (
            wrong != correct
            and _orthographic_similarity(wrong, correct) >= _SELF_CORRECTION_SIM_THRESHOLD
        ):
            # Any earlier inserted tokens are additional disfluent attempts.
            _emit_insert_tail_tokens(miscues, inserted[:-1], ref_at, target_norm, hyp_norm)
            miscues.append(
                Miscue(
                    "self_correction",
                    target_word=correct,
                    said_word=wrong,
                    index=ref_at,
                    confidence=_SELF_CORRECTION_CONFIDENCE,
                )
            )
            return 1

    # No self-correction: classify each inserted token as repeat-hesitation or
    # plain insertion.
    _emit_insert_tail_tokens(miscues, inserted, ref_at, target_norm, hyp_norm)
    return 1


def _emit_insert_tail_tokens(
    miscues: list[Miscue],
    inserted_hyp_indices: list[int],
    ref_at: int,
    target_norm: list[str],
    hyp_norm: list[str],
) -> None:
    """Emit repeat-hesitation or plain insertion for each inserted hyp token.

    Repeat rule: an inserted token identical (normalized) to the IMMEDIATELY
    PRECEDING TARGET token (target[ref_at - 1]) is a REPEAT -> hesitation at that
    preceding target index.  We use the preceding TARGET token (not hypothesis)
    because the index space of every Miscue is the target text, and a repeat is a
    re-utterance of a word the reader just (correctly) read from the passage —
    anchoring it to the target keeps "the the" over "the cat" reporting the
    hesitation at target index 0.
    """
    for h in inserted_hyp_indices:
        said = hyp_norm[h]
        if ref_at > 0 and said == target_norm[ref_at - 1]:
            miscues.append(
                Miscue("hesitation", target_word=None, said_word=said, index=ref_at - 1)
            )
        else:
            miscues.append(
                Miscue("insertion", target_word=None, said_word=said, index=ref_at)
            )


def _emit_insert_tail(
    miscues: list[Miscue],
    insert_chunk,  # noqa: ANN001
    target_norm: list[str],
    hyp_norm: list[str],
    skip_first: bool,
) -> None:
    """Emit insertions for an insert chunk used as a variant-(b) correction tail.

    ``skip_first`` drops the leading inserted token (it was the correction itself,
    already consumed by the self_correction Miscue).  Remaining inserted tokens are
    classified as plain insertions/repeats before the next target position.
    """
    indices = list(range(insert_chunk.hyp_start_idx, insert_chunk.hyp_end_idx))
    if skip_first:
        indices = indices[1:]
    _emit_insert_tail_tokens(
        miscues, indices, insert_chunk.ref_start_idx, target_norm, hyp_norm
    )


# ---------------------------------------------------------------------------
# Timing / filler hesitations
# ---------------------------------------------------------------------------

def _timing_hesitations(
    chunks,  # noqa: ANN001
    timed,  # noqa: ANN001  list[Word]
    target_norm: list[str],
    gap_threshold_s: float,
) -> list[Miscue]:
    """Flag a hesitation before any hypothesis word preceded by a gap > threshold.

    The hesitation index is the TARGET index aligned to the gapped hypothesis word
    (via the equal/substitute chunk covering it).  Words with no aligned target
    position (pure insertions) are skipped — a gap before an inserted word is
    already covered by the insertion/repeat signal.

    **Limitation — untimed words are silently skipped.**  The gap check requires
    both ``prev.end`` and ``cur.start`` to be non-None (``Word`` fields may be
    None when ASR returns partial timings).  A long pause that spans a word with
    no timing information is therefore invisible: the hesitation will be missed and
    recall degrades silently on partial-timing ASR output.  This is an accepted
    limitation until full per-word timing is guaranteed.
    """
    # Map hyp index -> target index (only for equal & substitute alignments,
    # which place a hypothesis word at a definite target slot).
    hyp_to_ref: dict[int, int] = {}
    for c in chunks:
        if c.type in ("equal", "substitute"):
            span = min(c.ref_end_idx - c.ref_start_idx, c.hyp_end_idx - c.hyp_start_idx)
            for k in range(span):
                hyp_to_ref[c.hyp_start_idx + k] = c.ref_start_idx + k

    out: list[Miscue] = []
    for hi in range(1, len(timed)):
        prev, cur = timed[hi - 1], timed[hi]
        if prev.end is None or cur.start is None:
            continue
        gap = cur.start - prev.end
        if gap > gap_threshold_s and hi in hyp_to_ref:
            ref_idx = hyp_to_ref[hi]
            out.append(
                Miscue(
                    "hesitation",
                    target_word=target_norm[ref_idx],
                    said_word=None,
                    index=ref_idx,
                )
            )
    return out


def _filler_hesitations(
    hyp: _Hyp,
    chunks,  # noqa: ANN001
    target_norm: list[str],
) -> list[Miscue]:
    """Emit a hesitation for each removed filler, bound to the next surviving
    target index (the target position the following hypothesis word aligns to)."""
    if not hyp.filler_before:
        return []

    # surviving-hyp-index -> target index (equal/substitute alignments).
    hyp_to_ref: dict[int, int] = {}
    for c in chunks:
        if c.type in ("equal", "substitute"):
            span = min(c.ref_end_idx - c.ref_start_idx, c.hyp_end_idx - c.hyp_start_idx)
            for k in range(span):
                hyp_to_ref[c.hyp_start_idx + k] = c.ref_start_idx + k

    out: list[Miscue] = []
    for surviving_idx, fillers in sorted(hyp.filler_before.items()):
        # Determine the target index this filler precedes.
        if surviving_idx in hyp_to_ref:
            ref_idx = hyp_to_ref[surviving_idx]
        elif surviving_idx >= len(hyp.tokens):
            # Trailing filler -> after the last target word.
            ref_idx = len(target_norm)
        else:
            # Following word is itself an insertion (no target slot) — bind to the
            # nearest known following target slot, else end of target.
            ref_idx = min(
                (hyp_to_ref[j] for j in hyp_to_ref if j >= surviving_idx),
                default=len(target_norm),
            )
        for filler in fillers:
            out.append(
                Miscue("hesitation", target_word=None, said_word=filler, index=ref_idx)
            )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(
    hypothesis,  # noqa: ANN001  str | AsrResult | list[Word]
    target_text: str,
    *,
    gap_threshold_s: float = GAP_THRESHOLD_S,
) -> list[Miscue]:
    """Align ``hypothesis`` to ``target_text`` (jiwer edit ops) and classify deviations.

    ``hypothesis`` may be a plain string (text-only mode — timing rule inactive)
    or a ``list[Word]`` / ``AsrResult`` (timed mode — full 5-class).  See the
    module docstring for normalization, precedence, and the two self-correction
    alignment variants.

    Keep false positives LOW — a false "correction" on a child who read correctly
    is a product-killing failure.
    """
    target_norm = [_normalize(t) for t in target_text.split()]
    hyp = _build_hyp(hypothesis)

    # jiwer needs at least one token on each side; handle empties explicitly.
    if not target_norm:
        raise ValueError("target_text must contain at least one word")

    out = jiwer.process_words(
        reference=" ".join(target_norm),
        hypothesis=" ".join(hyp.tokens) if hyp.tokens else "",
    )
    chunks = out.alignments[0]

    miscues = _classify_alignment(chunks, target_norm, hyp.tokens)

    # Filler hesitations (fillers were stripped pre-alignment).
    miscues.extend(_filler_hesitations(hyp, chunks, target_norm))

    # Timing hesitations — timed mode only.
    if hyp.timed is not None:
        miscues.extend(
            _timing_hesitations(chunks, hyp.timed, target_norm, gap_threshold_s)
        )

    miscues.sort(key=lambda m: (m.index, _ALL_CLASSES.index(m.type)))
    return miscues


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _validate_score_inputs(
    predicted: list[Miscue],
    gold: list[Miscue],
    n_target_words: int,
) -> None:
    """Raise ValueError on any out-of-range index or non-positive n_target_words.

    Legal index range: 0 <= index <= n_target_words.  The upper bound is INCLUSIVE
    because detect() legitimately produces index == n_target_words for two trailing
    cases (empirically verified, 2026-06-10, jiwer 4.0.0):
      - Trailing insertion: "the cat sat dog" / "the cat sat" → insertion index 3
        with n_target_words=3.
      - Trailing filler: a filler word after the last target word → hesitation
        index 3 with n_target_words=3.
    Nothing detect() produces has index > n_target_words, so values above that
    bound are always a caller/data bug (e.g. Finding I-1: gold omission index 8
    with n_target_words=3 silently skewed fp_per_100_correct_words).
    """
    if n_target_words <= 0:
        raise ValueError(
            f"n_target_words must be a positive integer; got {n_target_words}"
        )
    for label, lst in (("gold", gold), ("predicted", predicted)):
        for m in lst:
            if m.index < 0 or m.index > n_target_words:
                raise ValueError(
                    f"Invalid {label} miscue: {m.type} at index {m.index} is "
                    f"outside the legal range [0, {n_target_words}] for "
                    f"n_target_words={n_target_words}"
                )


def score(
    predicted: list[Miscue],
    gold: list[Miscue],
    n_target_words: int,
) -> dict:
    """Per-class precision/recall/F1 (+/-1 index tolerance) + FP-per-100-correct-words.

    A predicted miscue matches a gold miscue iff same class AND
    ``|index_pred - index_gold| <= 1``.  Matching is greedy one-to-one: within a
    class, golds and preds are sorted by index and matched by smallest index
    distance first (ties broken by the lower pred index, then lower gold index),
    so neither side is double-counted.

    ``n_target_words`` = number of words in the target text.  Correctly-read words
    = target words with NO gold miscue at their index.
    ``fp_per_100_correct_words`` = false positives / correct_words * 100.

    A class absent from BOTH gold and predicted reports precision/recall/f1 =
    ``None`` (never a fabricated 1.0).  ``n_gold`` / ``n_pred`` are always real
    counts.

    Raises ``ValueError`` if any miscue index is outside ``[0, n_target_words]``
    or if ``n_target_words <= 0``.  The upper bound is inclusive because
    ``detect()`` legitimately emits index == n_target_words for trailing insertions
    and trailing filler hesitations (end-of-passage position).
    """
    _validate_score_inputs(predicted, gold, n_target_words)

    result: dict = {}
    total_fp = 0

    for cls in _ALL_CLASSES:
        g = sorted([m for m in gold if m.type == cls], key=lambda m: m.index)
        p = sorted([m for m in predicted if m.type == cls], key=lambda m: m.index)
        n_gold, n_pred = len(g), len(p)

        tp = _greedy_match_count(
            [m.index for m in p],
            [m.index for m in g],
            tolerance=1,
        )
        fp = n_pred - tp
        total_fp += fp

        if n_gold == 0 and n_pred == 0:
            precision = recall = f1 = None
        else:
            precision = tp / n_pred if n_pred else 0.0
            recall = tp / n_gold if n_gold else 0.0
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

        result[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_gold": n_gold,
            "n_pred": n_pred,
        }

    # Correctly-read words = target words with no gold miscue at their index.
    gold_indices = {m.index for m in gold}
    correct_words = sum(1 for idx in range(n_target_words) if idx not in gold_indices)
    if correct_words > 0:
        fp_per_100 = total_fp / correct_words * 100
    else:
        # No correctly-read words to normalize against — report the raw FP count
        # scaled to 100 only if there are FPs, else 0.0.  Fail-soft is wrong here;
        # 0 correct words is a degenerate passage, so we surface 0.0 when there
        # are also no FPs and float('inf') would mislead — use total_fp * 100 as a
        # defined fallback so the number is finite and monotonic in FPs.
        fp_per_100 = float(total_fp * 100)

    result["fp_per_100_correct_words"] = fp_per_100
    return result


def _greedy_match_count(
    pred_indices: list[int],
    gold_indices: list[int],
    tolerance: int,
) -> int:
    """Count one-to-one matches where |pred - gold| <= tolerance.

    Greedy by smallest distance: build all candidate (distance, pred_i, gold_j)
    pairs, sort, and consume each pred/gold at most once.  Ties broken by lower
    pred index then lower gold index (the sort key), giving a deterministic result.
    """
    candidates: list[tuple[int, int, int]] = []
    for pi, p in enumerate(pred_indices):
        for gi, g in enumerate(gold_indices):
            d = abs(p - g)
            if d <= tolerance:
                candidates.append((d, pi, gi))
    candidates.sort()

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matched = 0
    for _d, pi, gi in candidates:
        if pi in used_pred or gi in used_gold:
            continue
        used_pred.add(pi)
        used_gold.add(gi)
        matched += 1
    return matched
