"""Synthetic miscue injector (T1.3a) — pure, seeded, no I/O except YAML loading.

We hand-wrote the decodable passages (``data/passages/*.yaml``); this module
plants miscues into them with KNOWN gold labels, so the benchmark's ground truth
is correct by construction rather than by annotation.

The injector is built AGAINST the detector's published class boundaries
(``readcoach.miscue``).  Every generated item is constructed so that
``miscue.detect`` classifies it as intended; the relevant detector constraints
are re-asserted here at build time (fail loud — a violated constraint is a bug in
this module, not data to skip).  Specifically:

  * substitution     — said word != target (normalized).
  * insertion        — inserted token (a) != preceding target word, (b) has
                       SequenceMatcher ratio < 0.5 vs the FOLLOWING target word
                       (else the detector calls it self_correction), (c) is not in
                       the detector filler lexicon (else it becomes a hesitation).
  * self_correction  — wrong attempt with ratio >= 0.5 vs the target word, != the
                       target, and != the preceding target word, placed
                       immediately BEFORE the correct target word.
  * omission         — drop one target word.
  * hesitation       — two render modes, distinguished in gold metadata:
                         render="filler"  -> "um"/"uh" inserted into the spoken
                                             text (recoverable from text).
                         render="silence" -> a ``[[slnc 1200]]`` marker emitted
                                             ONLY into the TTS text (invisible at
                                             text level; an audio-timing signal).

Text spaces
-----------
Three parallel renderings come out of ``inject``:
  * ``target_text``  — the passage as written (the detector's reference).
  * ``miscued_text`` — what a reader would have SAID (round-trip + TTS source).
                       Silence hesitations leave this UNCHANGED.
  * ``tts_text``     — identical to ``miscued_text`` except silence hesitations
                       appear as ``[[slnc 1200]]`` markers for the TTS engine.

All gold indices are in TARGET word space.
"""
from __future__ import annotations

import difflib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from readcoach.miscue import (
    _EDGE_PUNCT,
    _FILLERS,
    _SELF_CORRECTION_SIM_THRESHOLD,
    Miscue,
    MiscueType,
)

RenderMode = Literal["filler", "silence"]

# Silence marker emitted into the TTS text only (macOS `say`/AVSpeech style).
SILENCE_MARKER = "[[slnc 1200]]"

# Filler tokens for render="filler" hesitations.  Both are in the detector's
# filler lexicon, so the detector strips them and emits a hesitation.
_HESITATION_FILLERS: tuple[str, ...] = ("um", "uh")

# Orthographic-neighbor pools for generating substitution / self-correction
# attempts.  We swap the initial consonant or an interior vowel of the target to
# produce a realistic misreading; candidates are then validated against the
# detector's class gates (and retried with the rng on failure).
_INITIAL_CONSONANTS = "bcdfghjklmnprstvw"
_VOWELS = "aeiou"

# Insertion pool: short function words / adjectives a reader might interject.
# None are in the detector filler lexicon.  Each candidate is still validated
# per-site against constraints (a)/(b)/(c) before use.
_INSERTION_POOL: tuple[str, ...] = (
    "and",
    "the",
    "a",
    "so",
    "very",
    "big",
    "little",
    "then",
    "just",
    "now",
    "here",
    "good",
)

# Bounded retry budget for word-choice search before we fail loud.
_MAX_ATTEMPTS = 200


# ---------------------------------------------------------------------------
# Passage loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Passage:
    id: str
    band: int
    phonics_focus: str
    text: str
    words: tuple[str, ...]


def load_passages(dir: str | Path) -> list[Passage]:
    """Load every ``*.yaml`` passage in ``dir``, sorted by id (stable order).

    Fail-loud: a missing field, an empty file, or a non-1..4 band raises.
    """
    directory = Path(dir)
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"No passage YAML files found in {directory!r}")

    passages: list[Passage] = []
    for f in files:
        raw = yaml.safe_load(f.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{f.name}: expected a YAML mapping, got {type(raw).__name__}")
        # Explicit key access — KeyError on any missing field (fail loud).
        pid = raw["id"]
        band = raw["band"]
        phonics_focus = raw["phonics_focus"]
        text = raw["text"].strip()
        if band not in (1, 2, 3, 4):
            raise ValueError(f"{f.name}: band must be one of 1..4, got {band!r}")
        if not text:
            raise ValueError(f"{f.name}: empty text")
        passages.append(
            Passage(
                id=pid,
                band=band,
                phonics_focus=phonics_focus.strip(),
                text=text,
                words=tuple(text.split()),
            )
        )
    passages.sort(key=lambda p: p.id)
    return passages


# ---------------------------------------------------------------------------
# Injection spec / result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MiscueSpec:
    """A planned injection at a target word index.

    ``said_word`` / ``render`` are normally chosen by the injector; they may be
    pinned by a caller (used for clean items: an empty spec list).
    """

    type: MiscueType
    index: int
    said_word: str | None = None
    render: RenderMode | None = None


@dataclass(frozen=True)
class InjectedItem:
    utt_id: str
    passage_id: str
    target_text: str
    miscued_text: str
    tts_text: str
    gold: list[Miscue]
    # Parallel to ``gold``: the render mode for each hesitation gold ("filler" /
    # "silence"), and None for every non-hesitation class.  Lets consumers know
    # which gold entries are audio-only without re-deriving it.
    gold_render: list[RenderMode | None] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalization helpers (mirror the detector exactly)
# ---------------------------------------------------------------------------

def _normalize(token: str) -> str:
    return token.strip().strip(_EDGE_PUNCT).casefold()


def _split_affixes(word: str) -> tuple[str, str, str]:
    """Split a raw target word into (leading_punct, core, trailing_punct).

    ``core`` is the normalized (casefolded, edge-stripped) spoken form — what the
    detector compares.  Leading/trailing punctuation is preserved so the miscued
    text can be reassembled to read naturally for TTS.
    """
    stripped = word.strip()
    lead_len = len(stripped) - len(stripped.lstrip(_EDGE_PUNCT))
    trail_len = len(stripped) - len(stripped.rstrip(_EDGE_PUNCT))
    lead = stripped[:lead_len]
    trail = stripped[len(stripped) - trail_len:] if trail_len else ""
    core = stripped[lead_len: len(stripped) - trail_len] if trail_len else stripped[lead_len:]
    return lead, core.casefold(), trail


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Word-choice generators (rng-driven, validated against detector gates)
# ---------------------------------------------------------------------------

def _orthographic_neighbors(word: str, rng: random.Random) -> list[str]:
    """Candidate misreadings of ``word``: initial-consonant swaps + vowel swaps.

    Returned in a deterministic, rng-shuffled order so the search is seeded.
    """
    cands: list[str] = []
    if word:
        # Initial-sound swaps (realistic: "cat" -> "bat", "sat" -> "mat").
        for c in _INITIAL_CONSONANTS:
            if c != word[0]:
                cands.append(c + word[1:])
        # Interior vowel confusions ("cat" -> "cot", "pig" -> "pug").
        for i, ch in enumerate(word):
            if ch in _VOWELS:
                for v in _VOWELS:
                    if v != ch:
                        cands.append(word[:i] + v + word[i + 1:])
    # Deterministic dedupe, then seeded shuffle.
    seen: set[str] = set()
    uniq = [c for c in cands if not (c in seen or seen.add(c))]
    rng.shuffle(uniq)
    return uniq


def _choose_substitution(
    target_core: str,
    rng: random.Random,
) -> str:
    """A misreading that the detector will classify as a substitution.

    Only requirement (detector side): said != target after normalization.  We
    still prefer an orthographic neighbor for realism, and additionally keep it
    BELOW the self-correction similarity gate is NOT required here (a substitution
    has no following correct word, so similarity is irrelevant to the class).
    """
    for cand in _orthographic_neighbors(target_core, rng):
        if cand and cand != target_core:
            return cand
    # Fallback (target too short / degenerate): mutate first char deterministically.
    for c in _INITIAL_CONSONANTS:
        cand = c + target_core[1:] if target_core else c
        if cand != target_core:
            return cand
    raise RuntimeError(f"could not generate a substitution for {target_core!r}")


def _choose_self_correction_attempt(
    target_core: str,
    prev_core: str | None,
    rng: random.Random,
) -> str:
    """A wrong attempt PASSING the self-correction gate vs ``target_core``.

    Gate (published): ratio >= 0.5 vs target, != target, != preceding target word.
    Orthographic neighbors of the target are exactly the right generator — a
    single-edit neighbor of a >=2-char word always clears 0.5.
    """
    for cand in _orthographic_neighbors(target_core, rng):
        if not cand or cand == target_core or cand == prev_core:
            continue
        if _similarity(cand, target_core) >= _SELF_CORRECTION_SIM_THRESHOLD:
            return cand
    raise RuntimeError(
        f"could not generate a self-correction attempt for {target_core!r} "
        f"(prev={prev_core!r}) clearing the >= {_SELF_CORRECTION_SIM_THRESHOLD} gate"
    )


def _choose_insertion(
    prev_core: str | None,
    next_core: str | None,
    rng: random.Random,
) -> str:
    """An inserted token satisfying detector insertion constraints (a)/(b)/(c).

    (a) != preceding target word, (b) ratio < 0.5 vs the FOLLOWING target word,
    (c) not in the filler lexicon.
    """
    pool = list(_INSERTION_POOL)
    rng.shuffle(pool)
    for cand in pool:
        if cand in _FILLERS:  # (c)
            continue
        if prev_core is not None and cand == prev_core:  # (a)
            continue
        if next_core is not None and _similarity(cand, next_core) >= 0.5:  # (b)
            continue
        return cand
    raise RuntimeError(
        f"no insertion candidate satisfied constraints (prev={prev_core!r}, "
        f"next={next_core!r}); enlarge _INSERTION_POOL"
    )


# ---------------------------------------------------------------------------
# Core injection
# ---------------------------------------------------------------------------

def _variant_tag(specs: list[MiscueSpec]) -> str:
    """Deterministic, stable tag for an item built from ``specs``.

    Clean (no specs) -> "clean".  Otherwise the specs' (type-abbrev, index) pairs
    in index order, e.g. "sub1" or "sub1_om4_hesF7".
    """
    if not specs:
        return "clean"
    abbr = {
        "substitution": "sub",
        "omission": "om",
        "insertion": "ins",
        "self_correction": "sc",
        "hesitation": "hes",
    }
    parts: list[str] = []
    for s in sorted(specs, key=lambda x: x.index):
        tag = abbr[s.type]
        if s.type == "hesitation":
            tag += "F" if s.render == "filler" else "S"
        parts.append(f"{tag}{s.index}")
    return "_".join(parts)


def inject(passage: Passage, specs: list[MiscueSpec], rng: random.Random) -> InjectedItem:
    """Apply ``specs`` to ``passage`` and return an :class:`InjectedItem`.

    Pure w.r.t. ``rng`` (no global random use).  ``specs`` must target distinct,
    in-range indices; multi-miscue specs are additionally required non-adjacent by
    :func:`plan_benchmark` (this function tolerates any distinct indices).
    """
    words = passage.words
    n = len(words)

    # Validate spec indices up front (fail loud on a planning bug).
    seen_idx: set[int] = set()
    for s in specs:
        if not (0 <= s.index < n):
            raise IndexError(
                f"{passage.id}: spec index {s.index} out of range [0, {n})"
            )
        if s.index in seen_idx:
            raise ValueError(f"{passage.id}: duplicate spec index {s.index}")
        seen_idx.add(s.index)

    spec_by_index = {s.index: s for s in specs}

    # Pre-split every target word into (lead, core, trail).
    affixes = [_split_affixes(w) for w in words]
    cores = [c for (_, c, _) in affixes]

    out_tokens: list[str] = []      # miscued (spoken) text tokens
    tts_tokens: list[str] = []      # tts text tokens (== miscued + silence markers)
    gold: list[Miscue] = []
    gold_render: list[RenderMode | None] = []

    for i in range(n):
        lead, core, trail = affixes[i]
        spec = spec_by_index.get(i)
        prev_core = cores[i - 1] if i > 0 else None
        # For insertion/self_correction, the "following" target word is the word
        # at this index (the inserted/attempt token sits immediately before it).
        next_core = cores[i]

        if spec is None:
            out_tokens.append(words[i])
            tts_tokens.append(words[i])
            continue

        if spec.type == "substitution":
            said = spec.said_word or _choose_substitution(core, rng)
            if said == core:
                raise AssertionError(
                    f"{passage.id}: substitution at {i} equals target {core!r}"
                )
            tok = lead + said + trail
            out_tokens.append(tok)
            tts_tokens.append(tok)
            gold.append(Miscue("substitution", target_word=core, said_word=said, index=i))
            gold_render.append(None)

        elif spec.type == "omission":
            # Drop the word entirely from spoken + tts text.
            gold.append(Miscue("omission", target_word=core, said_word=None, index=i))
            gold_render.append(None)

        elif spec.type == "insertion":
            said = spec.said_word or _choose_insertion(prev_core, next_core, rng)
            # Re-assert constraints (a)/(b)/(c) (fail loud on a generator bug).
            if said in _FILLERS:
                raise AssertionError(f"{passage.id}: insertion {said!r} is a filler")
            if prev_core is not None and said == prev_core:
                raise AssertionError(
                    f"{passage.id}: insertion {said!r} repeats preceding target"
                )
            if next_core is not None and _similarity(said, next_core) >= 0.5:
                raise AssertionError(
                    f"{passage.id}: insertion {said!r} too similar to following "
                    f"target {next_core!r} (would be self_correction)"
                )
            out_tokens.append(said)
            tts_tokens.append(said)
            gold.append(Miscue("insertion", target_word=None, said_word=said, index=i))
            gold_render.append(None)
            # Then the real word.
            out_tokens.append(words[i])
            tts_tokens.append(words[i])

        elif spec.type == "self_correction":
            said = spec.said_word or _choose_self_correction_attempt(core, prev_core, rng)
            # Re-assert the published gate (fail loud).
            if said == core:
                raise AssertionError(f"{passage.id}: self_correction attempt == target")
            if said == prev_core:
                raise AssertionError(
                    f"{passage.id}: self_correction attempt == preceding target"
                )
            if _similarity(said, core) < _SELF_CORRECTION_SIM_THRESHOLD:
                raise AssertionError(
                    f"{passage.id}: self_correction attempt {said!r} below similarity "
                    f"gate vs target {core!r}"
                )
            # Wrong attempt then the correct word.
            out_tokens.append(said)
            tts_tokens.append(said)
            out_tokens.append(words[i])
            tts_tokens.append(words[i])
            gold.append(
                Miscue(
                    "self_correction",
                    target_word=core,
                    said_word=said,
                    index=i,
                    confidence=0.7,
                )
            )
            gold_render.append(None)

        elif spec.type == "hesitation":
            render = spec.render
            if render not in ("filler", "silence"):
                raise ValueError(
                    f"{passage.id}: hesitation spec must set render='filler'|'silence', "
                    f"got {render!r}"
                )
            if render == "filler":
                filler = spec.said_word or _HESITATION_FILLERS[
                    rng.randrange(len(_HESITATION_FILLERS))
                ]
                if filler not in _FILLERS:
                    raise AssertionError(
                        f"{passage.id}: hesitation filler {filler!r} not in detector lexicon"
                    )
                out_tokens.append(filler)
                tts_tokens.append(filler)
                out_tokens.append(words[i])
                tts_tokens.append(words[i])
                gold.append(
                    Miscue("hesitation", target_word=None, said_word=filler, index=i)
                )
                gold_render.append("filler")
            else:  # silence
                # Spoken text unchanged; TTS gets a silence marker before the word.
                out_tokens.append(words[i])
                tts_tokens.append(SILENCE_MARKER)
                tts_tokens.append(words[i])
                gold.append(
                    Miscue(
                        "hesitation", target_word=cores[i], said_word=None, index=i
                    )
                )
                gold_render.append("silence")

        else:  # pragma: no cover — exhaustive over MiscueType
            raise ValueError(f"{passage.id}: unknown miscue type {spec.type!r}")

    # Sort gold and its parallel render list together (gold/gold_render are built
    # one-to-one in spec-encounter order; zip-sort keeps them aligned).
    paired = sorted(zip(gold, gold_render), key=lambda mr: (mr[0].index, mr[0].type))
    sorted_gold = [m for m, _ in paired]
    sorted_render = [r for _, r in paired]

    utt_id = f"{passage.id}-{_variant_tag(specs)}"
    return InjectedItem(
        utt_id=utt_id,
        passage_id=passage.id,
        target_text=passage.text,
        miscued_text=" ".join(out_tokens),
        tts_text=" ".join(tts_tokens),
        gold=sorted_gold,
        gold_render=sorted_render,
    )


# ---------------------------------------------------------------------------
# Benchmark planning
# ---------------------------------------------------------------------------

_ALL_CLASSES: tuple[MiscueType, ...] = (
    "substitution",
    "omission",
    "insertion",
    "self_correction",
    "hesitation",
)


# Classes whose generated attempt is an orthographic NEIGHBOUR of the target and
# therefore needs a target word long enough to clear the 0.5 similarity gate
# (a 1-char target like "a" has no single-edit neighbour at ratio >= 0.5).  Both
# self_correction and substitution draw from the neighbour generator, so both are
# steered onto length >= 2 targets.
_NEIGHBOUR_CLASSES: frozenset[MiscueType] = frozenset({"self_correction", "substitution"})
_MIN_NEIGHBOUR_LEN = 2


def _eligible_indices(passage: Passage, cls: MiscueType) -> list[int]:
    """Target indices an injection of ``cls`` may use, in ascending order.

    * index 0 is excluded for insertion / self_correction (they need a preceding
      word to differ from for clean gold).
    * neighbour classes are restricted to targets of length >= 2 (gate clearance).
    """
    n = len(passage.words)
    cores = [_split_affixes(w)[1] for w in passage.words]
    out: list[int] = []
    for i in range(n):
        if i == 0 and cls in ("insertion", "self_correction"):
            continue
        if cls in _NEIGHBOUR_CLASSES and len(cores[i]) < _MIN_NEIGHBOUR_LEN:
            continue
        out.append(i)
    if not out:  # pragma: no cover — passages are long enough that this never trips
        raise RuntimeError(f"{passage.id}: no eligible index for class {cls}")
    return out


def _pick_index(passage: Passage, cls: MiscueType, rng: random.Random) -> int:
    return rng.choice(_eligible_indices(passage, cls))


def _pick_multi_indices(
    passage: Passage, classes: list[MiscueType], rng: random.Random
) -> list[int]:
    """Assign each class a distinct, pairwise NON-ADJACENT, eligible target index.

    Non-adjacency keeps multi-miscue items unambiguous for the detector (two edits
    on neighbouring words can merge into one alignment chunk).  Each class's index
    is drawn from its own eligibility set (length / position constraints), so e.g.
    a self_correction never lands on a 1-char word.  Greedy with bounded retry.
    """
    for _ in range(_MAX_ATTEMPTS):
        used: set[int] = set()
        chosen: list[int] = []
        ok = True
        for cls in classes:
            candidates = [
                i
                for i in _eligible_indices(passage, cls)
                if i not in used and all(abs(i - u) >= 2 for u in used)
            ]
            if not candidates:
                ok = False
                break
            pick = rng.choice(candidates)
            used.add(pick)
            chosen.append(pick)
        if ok:
            return chosen
    raise RuntimeError(
        f"{passage.id}: could not assign non-adjacent eligible indices to {classes}"
    )


def plan_benchmark(passages: list[Passage], seed: int) -> list[InjectedItem]:
    """Plan 88 injected items (11 per passage), deterministic for a fixed seed.

    Per passage (11 items):
      * 1 clean item (no miscues).
      * 6 single-miscue items — one for each of the 5 classes, with hesitation
        rendered BOTH ways (filler + silence), so every class has a clean
        single-miscue exemplar.
      * 4 multi-miscue items — 2-3 distinct, non-adjacent indices mixing classes.
    8 passages x 11 = 88 items.

    Calls :func:`validate_coverage` before returning (no bypass).
    """
    rng = random.Random(seed)
    items: list[InjectedItem] = []

    single_class_plan: list[tuple[MiscueType, RenderMode | None]] = [
        ("substitution", None),
        ("omission", None),
        ("insertion", None),
        ("self_correction", None),
        ("hesitation", "filler"),
        ("hesitation", "silence"),
    ]

    # Multi-miscue compositions (2-3 classes each).  Chosen so that, summed with
    # the single-miscue items, every class appears for every passage even before
    # the coverage check.
    multi_plans: list[list[MiscueType]] = [
        ["substitution", "omission"],
        ["insertion", "self_correction"],
        ["substitution", "hesitation", "omission"],
        ["self_correction", "insertion", "hesitation"],
    ]

    for p in passages:
        # 1) Clean item.
        items.append(inject(p, [], rng))

        # 2) One single-miscue item per class (hesitation x2 render modes).
        for cls, render in single_class_plan:
            idx = _pick_index(p, cls, rng)
            items.append(inject(p, [MiscueSpec(type=cls, index=idx, render=render)], rng))

        # 3) Multi-miscue items: distinct, non-adjacent, per-class-eligible indices.
        for classes in multi_plans:
            indices = _pick_multi_indices(p, classes, rng)
            specs: list[MiscueSpec] = []
            for cls, idx in zip(classes, indices, strict=True):
                render: RenderMode | None = None
                if cls == "hesitation":
                    render = "filler" if rng.random() < 0.5 else "silence"
                specs.append(MiscueSpec(type=cls, index=idx, render=render))
            items.append(inject(p, specs, rng))

    validate_coverage(items)  # raises if any (class, passage) cell empty — no bypass
    return items


# ---------------------------------------------------------------------------
# Coverage enforcement (no bypass)
# ---------------------------------------------------------------------------

def validate_coverage(items: list[InjectedItem]) -> dict[str, dict[str, int]]:
    """Build the class x passage coverage matrix and ENFORCE full coverage.

    Returns ``{passage_id: {class: count}}``.  RAISES ``CoverageError`` if any of
    the 5 miscue classes is missing for any passage present in ``items``.  There
    is no flag to skip this check — incomplete coverage is always a failure.
    """
    if not items:
        raise CoverageError("no items to validate")

    passage_ids = sorted({it.passage_id for it in items})
    matrix: dict[str, dict[str, int]] = {
        pid: {cls: 0 for cls in _ALL_CLASSES} for pid in passage_ids
    }
    for it in items:
        for g in it.gold:
            matrix[it.passage_id][g.type] += 1

    missing: list[str] = []
    for pid in passage_ids:
        for cls in _ALL_CLASSES:
            if matrix[pid][cls] == 0:
                missing.append(f"{pid}:{cls}")
    if missing:
        raise CoverageError(
            "coverage matrix is incomplete — every class must appear for every "
            f"passage.  Missing cells: {', '.join(missing)}"
        )
    return matrix


class CoverageError(Exception):
    """Raised when the class x passage coverage matrix has an empty cell."""
