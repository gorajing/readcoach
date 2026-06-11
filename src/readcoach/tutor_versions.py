"""T5.4 — three tutor versions over one replay interface (deterministic A/B).

The pre-registered A/B compares a STATE-BLIND tutor (v1) against a
MASTERY-CONDITIONED tutor (v2) on the FROZEN dev split, plus a deliberately-worse
ALWAYS-INTERVENE villain (v3) the gate must catch.  All three share one replay
signature::

    run_session(session_item: dict, version: str) -> SessionTrace

``session_item`` is one parsed line of ``evals/golden/persona_sessions_dev.jsonl``
(see :mod:`readcoach.persona_gen` ``SessionItem`` / ``SessionEvent`` for the
shape: a per-word event stream with ``kind`` ∈ the 5 detector classes ∪
{clean, page_end}, ``at_page_end`` / ``consecutive_struggles`` /
``page_had_struggle`` driving the policy, and the page band).

NO LLM ANYWHERE
---------------
These are ACTION-LEVEL traces: every ``TurnRecord.utterance`` is ``None``.  The
policy-compiler's utterance lexicon checks (``never_says_wrong`` /
``no_emotional_intimacy``) therefore pass trivially — a ``None`` utterance is
skipped by those checks.  The violations that matter here come from ACTION-LEVEL
rules (``never_corrects_self_correction`` keys on ``action_move`` ∈
{SCAFFOLDED_HINT, MODEL_THE_WORD} applied to a self_correction turn;
``never_coaches_mid_page`` keys on ``action_move`` ∈ {ENCOURAGE,
COMPREHENSION_PROMPT, NEXT_ITEM} mid-page).  That is the honest seam: v3's
"model every miscue, even a self-correction" trips ``never_corrects_self_correction``
on real dev-split self_correction events, while v1/v2 honor self-corrections and
stay clean.

THE THREE VERSIONS
------------------
* **v1 — state-blind.**  ``decide()`` is fed a BLINDED ``LearnerState`` (empty
  mastery dict every turn — no memory across turns or sessions).  The planner is
  NOT consulted: a page-end NEXT_ITEM carries ``target=None`` (scripted next
  passage in the corpus order; we do not select a skill from learner state).
* **v2 — mastery-conditioned.**  The full pipeline: a fresh
  :class:`~readcoach.learner_store.InMemoryLearnerStore` per persona-child,
  observations recorded per MISCUE event (incorrect evidence on the
  band-representative skill, tagged with the detector ``miscue_class`` and a
  constant detector-style confidence of 0.9 — clean reads are NOT recorded; see
  ``_run_v2`` for why all-events saturates BKT and erases the gaps),
  ``decide()`` fed the LIVE ``LearnerState`` from the store, and a page-end
  NEXT_ITEM target selected by :func:`readcoach.planner.next_item`.
* **v3 — always-intervene (DELIBERATELY WORSE — DEMO-VILLAIN).**  Never WAITs:
  every mid-page miscue is met with an immediate MODEL_THE_WORD.  This is the bad
  tutor the gate must block; it is clearly labelled in code and exists only to
  prove the evaluation discriminates.

SKILL MAPPING (documented, honest)
----------------------------------
The persona corpus carries no per-word phonics-skill labels — only a passage
``band`` (1..4) and free-text ``phonics_focus``.  So v2 associates each session
with the curriculum skills AT THE PASSAGE'S BAND and records every word event
against a single BAND-REPRESENTATIVE skill (the first curriculum skill at that
band in topological order).  This is a deliberate, documented modelling choice:
a band→skill mapping is the principled substitute for a word→skill mapping the
data does not provide.  It keeps the v2 pipeline REAL (store + BKT + planner)
without fabricating labels.

DETERMINISM
-----------
Pure of any RNG.  Every output is a function of the (frozen) session item and the
version.  ``run_session`` is idempotent and order-independent.
"""
from __future__ import annotations

from datetime import datetime, timezone

from readcoach.learner_model import LearnerState
from readcoach.learner_store import InMemoryLearnerStore
from readcoach.miscue import Miscue, _ALL_CLASSES
from readcoach.planner import Curriculum, next_item
from readcoach.trace import SessionTrace, TurnRecord
from readcoach.tutor import POLICY_VERSION, TutorContext, decide

# The three version ids.  Public so the runner / tests reference them by name.
VERSIONS: tuple[str, ...] = ("v1", "v2", "v3")

# Constant detector-style confidence stamped on every v2 observation.  The
# persona corpus is synthetic and carries no real detector confidence, so we use
# one documented constant rather than fabricate a per-event score (mirrors the
# learner-store BKT soft-evidence contract: a confidence < 1.0 discounts a noisy
# observation).
V2_DETECTOR_CONFIDENCE: float = 0.9

# A fixed timestamp base — the replay is offline and time-independent; we only
# need monotonic ordering for the store's FSRS cards, not real wall-clock time.
_BASE_TS = datetime(2026, 6, 10, tzinfo=timezone.utc)

# The 5 detector miscue classes that count as a "miscue" event (everything else
# in an event stream is "clean" or "page_end").
_MISCUE_KINDS = frozenset(_ALL_CLASSES)


def is_decision_turn(turn) -> bool:  # noqa: ANN001 — TurnRecord
    """A pedagogically-meaningful decision turn: a miscue turn OR a page-end turn.

    The wait_rate band [0.35, 0.50] (MetaCLASS arXiv:2602.02457) is defined over
    turns where the policy actually faced a non-trivial choice — exactly the
    contexts ``scripts/policy_replay.py`` emits (one per miscue, one per page-end).
    Clean mid-page reads are trivially WAIT and not skill-diagnostic, so including
    them would inflate the denominator and make the band incomparable to the
    miscue-only replay the band was calibrated on.  This helper is the single
    source of truth for that denominator across the runner and its tests.
    """
    return turn.miscue_type is not None or turn.at_page_end


def _band_representative_skill(curriculum: Curriculum, band: int) -> str | None:
    """First curriculum skill at ``band`` in topological order, or None.

    The session-to-skill anchor for v2: deterministic and stable across runs
    because ``curriculum.topological_order`` is itself deterministic.
    """
    for skill_id in curriculum.topological_order:
        if curriculum.nodes[skill_id].band == band:
            return skill_id
    return None


def _miscue_from_event(event: dict) -> Miscue | None:
    """Build a :class:`Miscue` from a session event, or None for clean/page_end."""
    kind = event["kind"]
    if kind not in _MISCUE_KINDS:
        return None
    confidence = 0.7 if kind == "self_correction" else 1.0
    return Miscue(
        type=kind,
        target_word=event.get("target_word"),
        said_word=event.get("said_word"),
        index=event["word_index"],
        confidence=confidence,
    )


def _v3_decide(ctx: TutorContext) -> "object":
    """DELIBERATELY-WORSE villain policy (DEMO ONLY) — never WAITs mid-page.

    Every mid-page miscue is met with an immediate MODEL_THE_WORD, INCLUDING a
    self_correction — which is exactly the ``never_corrects_self_correction``
    violation the gate is built to catch.  Page-end turns defer to the real
    policy (the villain's defining flaw is mid-page over-intervention, not
    page-end behaviour), so v3's page-end NEXT_ITEM/ENCOURAGE stay well-formed and
    the ONLY new violations come from the mid-page modeling.

    Returns a real :class:`~readcoach.tutor.TutorAction`; typed loosely to avoid
    importing the action class name into this villain helper's signature.
    """
    from readcoach.tutor import TutorAction  # local import: villain is self-contained

    if not ctx.at_page_end and ctx.miscue is not None:
        return TutorAction(
            move="MODEL_THE_WORD",
            target_word=ctx.miscue.target_word,
            rationale=(
                "[V3-ALWAYS-INTERVENE] DEMO-VILLAIN: model every miscue immediately "
                "(never WAIT) — deliberately violates productive-struggle protection"
            ),
            error_type=ctx.miscue.type,
        )
    # Page-end (and the rare mid-page no-miscue turn) -> the real policy.
    return decide(ctx)


def run_session(
    session_item: dict,
    version: str,
    *,
    curriculum: Curriculum | None = None,
    store: InMemoryLearnerStore | None = None,
) -> SessionTrace:
    """Replay one persona session under ``version`` -> an action-level SessionTrace.

    Parameters
    ----------
    session_item:
        One parsed line of the frozen dev split (a ``SessionItem`` dict).
    version:
        ``"v1"`` (state-blind), ``"v2"`` (mastery-conditioned), or ``"v3"``
        (always-intervene villain).
    curriculum:
        Required for v2 (band→skill anchor + planner traversal); ignored for
        v1/v3.  Passed in so the runner loads it once.
    store:
        Optional v2 store to reuse across a persona's sessions (memory persists
        across sessions for the same child).  When None, a fresh per-call
        :class:`InMemoryLearnerStore` is created (no cross-session memory).

    Every ``TurnRecord.utterance`` is ``None`` (action-level trace; no LLM).
    """
    if version not in VERSIONS:
        raise ValueError(f"unknown tutor version {version!r}; expected one of {VERSIONS}")

    persona_id = session_item["persona_id"]
    band = int(session_item["band"])
    child_id = persona_id  # one child per persona

    if version == "v2":
        if curriculum is None:
            raise ValueError("v2 requires a curriculum (band→skill anchor + planner)")
        if store is None:
            store = InMemoryLearnerStore()
        return _run_v2(session_item, curriculum, store, child_id, band)
    if version == "v3":
        return _run_simple(session_item, child_id, decide_fn=_v3_decide, blind=True)
    # v1
    return _run_simple(session_item, child_id, decide_fn=decide, blind=True)


def _run_simple(
    session_item: dict,
    child_id: str,
    *,
    decide_fn,  # noqa: ANN001 — Callable[[TutorContext], TutorAction]
    blind: bool,
) -> SessionTrace:
    """Replay for the memory-free versions (v1 state-blind, v3 villain).

    ``blind`` -> a fresh EMPTY ``LearnerState`` every turn (no mastery, no memory).
    Page-end NEXT_ITEM is never produced here because the empty learner state has
    no gaps to target (the page-end rule requires ``page_had_struggle`` AND a
    non-empty gaps() list) — so v1's page-end mass lands on ENCOURAGE /
    COMPREHENSION_PROMPT, never a planner-selected NEXT_ITEM.  That is the
    intended state-blindness: no learner state -> no targeted next item.
    """
    records: list[TurnRecord] = []
    turn_index = 0
    for event in session_item["events"]:
        if event["kind"] == "page_end":
            ctx = TutorContext(
                miscue=None,
                learner_state=LearnerState(child_id=child_id, mastery={}),
                at_page_end=True,
                consecutive_struggles=0,
                page_had_struggle=bool(event["page_had_struggle"]),
                open_comprehension=False,
            )
            action = decide_fn(ctx)
            records.append(_record(turn_index, event, action, served_reason=None, skill_id=None))
            turn_index += 1
            continue

        miscue = _miscue_from_event(event)
        ctx = TutorContext(
            miscue=miscue,
            learner_state=LearnerState(child_id=child_id, mastery={}),
            at_page_end=bool(event["at_page_end"]),
            consecutive_struggles=int(event["consecutive_struggles"]),
            page_had_struggle=bool(event["page_had_struggle"]),
            open_comprehension=False,
        )
        action = decide_fn(ctx)
        records.append(_record(turn_index, event, action, served_reason=None, skill_id=None))
        turn_index += 1

    return SessionTrace(
        child_id=child_id,
        policy_version=POLICY_VERSION,
        completed_skills_at_start=(),
        turns=tuple(records),
    )


def _run_v2(
    session_item: dict,
    curriculum: Curriculum,
    store: InMemoryLearnerStore,
    child_id: str,
    band: int,
) -> SessionTrace:
    """Full mastery-conditioned pipeline for v2.

    OBSERVATION MODEL (documented, honest)
    --------------------------------------
    A v2 observation is recorded ONLY on a MISCUE event, against the
    band-representative skill, as INCORRECT evidence (with the detector
    ``miscue_class`` and the constant ``V2_DETECTOR_CONFIDENCE``).  Clean reads are
    NOT recorded: in this corpus clean reading is dominated by high-frequency words
    that are not targeted practice of the passage's focus skill, so feeding every
    clean word as "correct" evidence would saturate BKT to mastered for every skill
    and erase the very gaps the planner exists to target (empirically verified on
    the dev split: all-events -> mastery 1.0 everywhere -> no gaps -> v2 collapses
    onto v1).  Recording the diagnostic signal (the miscue) keeps the learner model
    honest: a skill the child miscued on stays a GAP, and the planner serves it.

    DIVERGENCE FROM v1
    ------------------
    With a populated, sub-mastery learner state, the page-end policy's
    ``R-PE-NEXT-ITEM`` rule fires (it needs ``page_had_struggle`` AND a non-empty
    ``LearnerState.gaps()``); for that NEXT_ITEM turn the PLANNER selects the real
    next item (skill + reason).  v1's empty state has no gaps, so the same page-end
    lands on ENCOURAGE — that ENCOURAGE → NEXT_ITEM swing is the behavioural
    signature of mastery-conditioning in the deterministic A/B.
    """
    skill = _band_representative_skill(curriculum, band)
    records: list[TurnRecord] = []
    turn_index = 0
    session_id = session_item["id"]
    # Skills already introduced as "new" in THIS trace.  Re-serving one as "new"
    # would trip never_reserves_completed_item (a completed/introduced item must
    # not be re-introduced as new).  A re-encounter of a still-unmastered skill is
    # continued practice -> recorded as "review", which is always compliant.
    served_new: set[str] = set()

    for event in session_item["events"]:
        kind = event["kind"]

        if kind == "page_end":
            live_state = store.get_state(child_id)
            ctx = TutorContext(
                miscue=None,
                learner_state=live_state,
                at_page_end=True,
                consecutive_struggles=0,
                page_had_struggle=bool(event["page_had_struggle"]),
                open_comprehension=False,
            )
            action = decide(ctx)
            served_reason: str | None = None
            served_skill: str | None = None
            if action.move == "NEXT_ITEM":
                # Consult the planner for the real next item (skill + reason).
                served_log = store.get_served_log(child_id)
                pick = next_item(
                    curriculum,
                    child_id,
                    store,
                    served_log,
                    now=_BASE_TS,
                )
                if pick is not None:
                    served_skill, planner_reason = pick
                    # Downgrade a repeat "new" serve of the same skill to "review":
                    # the skill was already introduced this session, so re-serving
                    # it is review intent, not a fresh introduction.
                    if planner_reason == "new" and served_skill in served_new:
                        served_reason = "review"
                    else:
                        served_reason = planner_reason
                        if planner_reason == "new":
                            served_new.add(served_skill)
            records.append(
                _record(
                    turn_index, event, action,
                    served_reason=served_reason, skill_id=served_skill,
                )
            )
            turn_index += 1
            continue

        # Miscue event -> record INCORRECT evidence against the band-rep skill.
        # Clean reads are NOT recorded (see the observation-model docstring above).
        if skill is not None and kind in _MISCUE_KINDS:
            store.record_observation(
                child_id=child_id,
                skill=skill,
                correct=False,
                confidence=V2_DETECTOR_CONFIDENCE,
                session_id=session_id,
                ts=_BASE_TS,
                miscue_class=kind,
            )

        miscue = _miscue_from_event(event)
        live_state = store.get_state(child_id)
        ctx = TutorContext(
            miscue=miscue,
            learner_state=live_state,
            at_page_end=bool(event["at_page_end"]),
            consecutive_struggles=int(event["consecutive_struggles"]),
            page_had_struggle=bool(event["page_had_struggle"]),
            open_comprehension=False,
        )
        action = decide(ctx)
        records.append(_record(turn_index, event, action, served_reason=None, skill_id=None))
        turn_index += 1

    return SessionTrace(
        child_id=child_id,
        policy_version=POLICY_VERSION,
        completed_skills_at_start=(),  # fresh child each persona; nothing pre-completed
        turns=tuple(records),
    )


def _record(
    turn_index: int,
    event: dict,
    action,  # noqa: ANN001 — TutorAction
    *,
    served_reason: str | None,
    skill_id: str | None,
) -> TurnRecord:
    """Build one action-level TurnRecord (utterance always None — no LLM)."""
    kind = event["kind"]
    miscue_type = kind if kind in _MISCUE_KINDS else None
    return TurnRecord(
        turn_index=turn_index,
        at_page_end=bool(event["at_page_end"]),
        miscue_type=miscue_type,
        action_move=action.move,
        hint_level=action.hint_level,
        served_reason=served_reason,
        utterance=None,  # action-level trace — no LLM verbalization
        is_ai_reminder=False,
        skill_id=skill_id,
    )
