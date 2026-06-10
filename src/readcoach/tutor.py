"""Tutoring agent harness — an explicit decision policy, NOT a chat wrapper.

A raw LLM is RLHF'd to be helpful and give answers — the opposite of teaching. The
harness enforces the withholding/scaffolding policy the model won't hold on its own
(MathTutorBench arXiv:2502.18940; MetaCLASS arXiv:2602.02457; Tutor CoPilot).

Decision: given (miscue, learner_state, position_in_page) -> one discrete move.

T4.1 — the MOVE POLICY (pure rules)
===================================
``decide()`` is the tutor's decision core: a *pure*, *versioned* function from
enum-shaped inputs to a frozen ``TutorAction``.  An LLM later only VERBALIZES the
move this policy chooses; it never decides.

The rule matrix is an explicit if-chain, one rule per block, ordered by priority.
Every rule has a stable ``rule_id`` (embedded in the rationale, e.g. ``[R-MID-MODEL]``)
so any logged action is auditable, and every action is stamped with
``POLICY_VERSION``.  An uncovered combination falls through to a conservative WAIT
default (``R-DEFAULT``) — never a raise mid-lesson — but the matrix is built to
cover the replay exhaustively, so the default is a backstop, not a load-bearing
path (proven unreached in ``tests/test_policy_replay.py``).

Pedagogy contract (controller resolution, T4.1) — true BY CONSTRUCTION:
  * Mid-page protects flow.  While NOT at page-end the policy returns WAIT for
    everything EXCEPT sustained struggle on the SAME word:
      - consecutive_struggles >= MODEL_THRESHOLD (3)  -> MODEL_THE_WORD
      - consecutive_struggles >= HINT_THRESHOLD  (2)  -> SCAFFOLDED_HINT (ladder)
    A teacher does not interrupt the flow for one miscue, but does rescue a stuck
    child.  ENCOURAGE / COMPREHENSION_PROMPT / NEXT_ITEM are page-end-only.
  * Self-corrections are NEVER corrected, at any struggle count.  Mid-page they
    WAIT; at page-end they are celebrated via ENCOURAGE.  The rationale never
    frames the read as an error (the action must read clean for the verbalizer).
  * Hesitation alone is not a "failed attempt": productive struggle is protected,
    so a hesitation miscue without a sustained struggle count WAITs.
  * The scaffold ladder is deterministic: 2->bounce, 3->highlight, 4->phonetic
    (``hint_level_for``).  The MODEL_THRESHOLD gate means the live policy hands a
    stuck child the word at 3 rather than climbing to the top rung; the higher
    rungs are the documented ladder ordering the verbalizer uses and are reachable
    if MODEL_THRESHOLD is raised (not dead — ``decide`` stamps the level via the
    same helper the tests pin).
  * Page-end coaching: an open comprehension opportunity -> COMPREHENSION_PROMPT;
    else a page the child struggled on with an open mastery gap -> NEXT_ITEM
    selected from ``LearnerState.gaps()``; else a clean/strong page -> ENCOURAGE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .learner_model import LearnerState
from .miscue import Miscue

Move = Literal[
    "WAIT",
    "ENCOURAGE",
    "SCAFFOLDED_HINT",
    "MODEL_THE_WORD",
    "COMPREHENSION_PROMPT",
    "NEXT_ITEM",
]

HintLevel = Literal["bounce", "highlight", "phonetic"]

# Bumped only via a tracked code change; stamped on EVERY action so a logged
# decision is attributable to the exact rule matrix that produced it.
POLICY_VERSION = "1.0.0"

# Struggle thresholds (consecutive failed attempts on the SAME word, mid-page).
HINT_THRESHOLD = 2   # >= here -> SCAFFOLDED_HINT ladder
MODEL_THRESHOLD = 3  # >= here -> give the word and move on

# Ladder ordering, indexed by (consecutive_struggles - HINT_THRESHOLD).  Single
# source of truth for the bounce -> highlight -> phonetic progression; saturates
# at the top rung.
_LADDER: tuple[HintLevel, ...] = ("bounce", "highlight", "phonetic")

# Stable rule id for the conservative backstop.  A replay/decide path that lands
# here means the matrix failed to name a context; the replay test asserts this is
# never hit.
DEFAULT_RULE_ID = "R-DEFAULT"


@dataclass(frozen=True)
class TutorContext:
    """Enum-shaped decision inputs.  Page-end state has compatible defaults so
    callers that only have a mid-page miscue need not construct page-end fields."""

    miscue: Miscue | None
    learner_state: LearnerState
    at_page_end: bool
    consecutive_struggles: int = 0
    # Page-end state — minimal fields the page-end rules need.
    page_had_struggle: bool = False     # did the child struggle anywhere on this page?
    open_comprehension: bool = False    # is there an unanswered comprehension opportunity?


@dataclass(frozen=True)
class TutorAction:
    move: Move
    target_word: str | None
    rationale: str               # inspectable: WHY this move (rule_id embedded)
    error_type: str | None = None        # miscue class driving a SCAFFOLDED_HINT
    hint_level: HintLevel | None = None  # scaffold ladder rung (SCAFFOLDED_HINT only)
    policy_version: str = POLICY_VERSION  # stamped on EVERY action


# Golden invariants (regression-tested in evals/): the harness must NEVER violate these.
GOLDEN_INVARIANTS = (
    "never_says_wrong",          # protect motivation
    "never_coaches_mid_page",    # coach at page-end, like a teacher
    "never_false_positive_correction",  # don't 'correct' a self-correction or ASR error
)


def hint_level_for(consecutive_struggles: int) -> HintLevel | None:
    """Deterministic scaffold-ladder rung for a struggle count.

    Below ``HINT_THRESHOLD`` there is no hint (None).  At/above it the rung climbs
    bounce -> highlight -> phonetic and saturates at the top:

        2 -> "bounce", 3 -> "highlight", 4 -> "phonetic", 5+ -> "phonetic".
    """
    if consecutive_struggles < HINT_THRESHOLD:
        return None
    idx = min(consecutive_struggles - HINT_THRESHOLD, len(_LADDER) - 1)
    return _LADDER[idx]


def _rationale(rule_id: str, text: str) -> str:
    """Stamp a stable rule id onto a human-readable reason."""
    return f"[{rule_id}] {text}"


def decide(ctx: TutorContext) -> TutorAction:
    """Choose a single pedagogical move from the versioned rule matrix.

    Pure function of ``ctx``.  Priority-ordered if-chain (one rule per block);
    the FIRST matching rule wins.  Every return is a frozen ``TutorAction`` with
    ``POLICY_VERSION`` stamped and the rule id embedded in the rationale.
    """
    m = ctx.miscue
    struggles = ctx.consecutive_struggles
    is_self_correction = m is not None and m.type == "self_correction"

    # -- Page-end coaching (ENCOURAGE / COMPREHENSION_PROMPT / NEXT_ITEM) --------
    if ctx.at_page_end:
        # R-PE-COMPREHENSION: an open comprehension opportunity takes precedence —
        # checking understanding is the highest-value page-end move.
        if ctx.open_comprehension:
            return TutorAction(
                move="COMPREHENSION_PROMPT",
                target_word=None,
                rationale=_rationale(
                    "R-PE-COMPREHENSION",
                    "page-end with an open comprehension opportunity; prompt for "
                    "understanding before moving on",
                ),
            )

        # R-PE-NEXT-ITEM: the page is done, the child struggled, and there is an
        # open mastery gap to target -> advance to the next item at the edge of the
        # ZPD.  The next item is selected from the learner's gaps (lowest mastery
        # first), never a mastered skill.
        if ctx.page_had_struggle:
            gaps = ctx.learner_state.gaps()
            if gaps:
                next_skill = min(gaps, key=lambda s: ctx.learner_state.mastery[s])
                return TutorAction(
                    move="NEXT_ITEM",
                    target_word=next_skill,
                    rationale=_rationale(
                        "R-PE-NEXT-ITEM",
                        f"page done after struggle; next item targets the weakest "
                        f"open gap ({next_skill})",
                    ),
                )

        # R-PE-ENCOURAGE: a clean/strong page (or no gap left to target) -> warm
        # acknowledgement.  Self-corrections land here too and are CELEBRATED, never
        # re-corrected; the rationale stays free of error framing.
        return TutorAction(
            move="ENCOURAGE",
            target_word=None,
            rationale=_rationale(
                "R-PE-ENCOURAGE",
                "page-end on a strong read; acknowledge the effort and keep momentum",
            ),
        )

    # -- Mid-page: protect flow ------------------------------------------------
    # R-MID-SELF-CORRECTION: a self-correction is the child fixing their own read.
    # Never corrective, at ANY struggle count; mid-page that means WAIT.  Placed
    # ABOVE the struggle gates so an inflated struggle count can never turn a
    # self-correction into a hint/model.  Rationale carries no error framing.
    if is_self_correction:
        return TutorAction(
            move="WAIT",
            target_word=None,
            rationale=_rationale(
                "R-MID-SELF-CORRECTION",
                "the reader caught and fixed it themselves; honor the self-correction "
                "and let them keep reading",
            ),
        )

    # R-MID-MODEL: sustained, unproductive struggle on the same word -> give the
    # word and move on (rescue the stuck child).
    if m is not None and struggles >= MODEL_THRESHOLD:
        return TutorAction(
            move="MODEL_THE_WORD",
            target_word=m.target_word,
            rationale=_rationale(
                "R-MID-MODEL",
                f"{struggles} unproductive attempts on '{m.target_word}'; model the "
                f"word so the reader can continue",
            ),
            error_type=m.type,
        )

    # R-MID-SCAFFOLD: struggle is mounting but still recoverable -> a graded hint
    # on the deterministic ladder (bounce -> highlight -> phonetic).
    if m is not None and struggles >= HINT_THRESHOLD:
        level = hint_level_for(struggles)
        return TutorAction(
            move="SCAFFOLDED_HINT",
            target_word=m.target_word,
            rationale=_rationale(
                "R-MID-SCAFFOLD",
                f"{struggles} attempts on '{m.target_word}'; offer a '{level}' scaffold "
                f"to keep the reader working productively",
            ),
            error_type=m.type,
            hint_level=level,
        )

    # R-MID-WAIT: a single miscue, a hesitation, or no miscue at all mid-page ->
    # WAIT.  Productive struggle is protected; we do not interrupt the flow.
    if m is not None or struggles == 0:
        return TutorAction(
            move="WAIT",
            target_word=None,
            rationale=_rationale(
                "R-MID-WAIT",
                "mid-page; protect the reading flow and let productive struggle run",
            ),
        )

    # R-DEFAULT: conservative backstop for any combination the matrix did not name.
    # WAIT is always the safe move; never raise mid-lesson.
    return TutorAction(
        move="WAIT",
        target_word=None,
        rationale=_rationale(
            DEFAULT_RULE_ID,
            "default: no rule matched; waiting is the safe move",
        ),
    )
