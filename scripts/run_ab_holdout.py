#!/usr/bin/env python3
"""T5.4 — the HOLDOUT A/B runner: implements docs/ab_holdout_plan.md VERBATIM.

This is the SANCTIONED held-out evaluation.  ``scripts/run_ab.py`` REFUSES the
holdout path on purpose (the dev A/B must never read the freeze); this script is
the one place the held-out half is read, and only under the pre-registered
protocol committed in ``docs/ab_holdout_plan.md`` BEFORE any result existed.

THE PRE-REGISTERED PLAN IS LAW
==============================
docs/ab_holdout_plan.md (committed 2026-06-11, frozen) specifies exactly:

  * v1 (state-blind) and v2 (mastery-conditioned) over ALL 49 holdout sessions.
  * Deterministic metrics (invariant violations, WAIT-rate band, targeted next
    items, serve violations) over every holdout session, same aggregation as the
    dev A/B (scripts/run_ab.py is imported and REUSED — one aggregation, not two).
  * From each version's replay, 60 decision turns sampled, stratified by persona
    (20/20/20), seed 7331.  THE SAMPLE IS A FUNCTION OF THE REPLAY ONLY — it never
    sees a judged score, so it cannot be tuned to a result.
  * Each sampled turn is verbalized live (claude-cli transport, prompt 1.0) and
    judged on ONLY the dimensions ``judge_validation.json`` marks gate_eligible.
  * Prediction-#5 adjudication mapping (CONFIRMED / PARTIAL / MISSED /
    UNADJUDICABLE) exactly as written in the plan.

HONEST PAIRING (the one place the plan leaves a real choice)
------------------------------------------------------------
The plan says "seeded paired bootstrap 95% CI over sampled turns".  But the two
versions are sampled from DIFFERENT action streams (v1 and v2 make different
moves, so their decision-turn sets differ).  TRUE turn-level pairing is therefore
impossible: there is no canonical 1:1 correspondence between a v1 sampled turn and
a v2 sampled turn.  We do NOT silently pretend otherwise.

We implement the CI as an INDEPENDENT seeded bootstrap of the difference of means
(resample the v1 sample and the v2 sample independently, each within its own
60-turn draw, take mean(v2) - mean(v1), repeat; one fixed seed).  The output
records ``pairing="independent"`` with an explicit honesty note explaining that
true pairing is impossible given different action streams, so the closest correct
construction is an independent-resample difference-of-means CI.  This is stated in
``ab_holdout.json`` as a finding, not buried.

CHECKPOINTING (interruptions WILL happen)
-----------------------------------------
The live phase is ~120 verbalizations + up to 120×(#eligible dims) judgments.
BOTH phases are checkpointed with the same loud-resume design as
scripts/judge_turns.py: every result is appended atomically to a workfile; on
resume the workfile is loaded and every line validated (malformed → abort, never
skip); the phase's final artifact is written all-or-nothing only when every unit
is present, then the workfile is removed.  A transport failure ABORTS (the plan:
"any sampled turn skipped … aborts rather than degrades").

GATING THE LIVE-ONLY SIDE EFFECT
--------------------------------
Appending the prediction-#5 verdict to docs/results_vs_predictions.md is a
side effect that must happen ONLY when the run actually completes live (real
verbalizations + real judgments, real judge_validation.json).  It is therefore
gated behind ``--live`` AND structured as a single function
(``append_prediction5_verdict``) the controller's live run calls — never during a
mocked test run.

Usage
-----
    # The controller's real run (reads judge_validation.json; live transports):
    uv run python scripts/run_ab_holdout.py --i-am-the-preregistered-run --live

    # Resume after an interruption (same flags; workfiles auto-loaded):
    uv run python scripts/run_ab_holdout.py --i-am-the-preregistered-run --live
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from evals.harness import EvalReport, compare, evaluate  # noqa: E402
from evals.judge import JudgeError, judge_turn  # noqa: E402
from readcoach.learner_store import InMemoryLearnerStore  # noqa: E402
from readcoach.llm_client import ClaudeCliTransport  # noqa: E402
from readcoach.planner import Curriculum, load_curriculum  # noqa: E402
from readcoach.learner_model import LearnerState  # noqa: E402
from readcoach.tutor import TutorAction, TutorContext, decide  # noqa: E402
from readcoach.tutor_versions import (  # noqa: E402
    V2_DETECTOR_CONFIDENCE,
    _BASE_TS,
    _band_representative_skill,
    _miscue_from_event,
    is_decision_turn,
)

# Import the dev A/B runner to REUSE its aggregation + report machinery verbatim.
# (It is a script, not a package; load it by path the same way its tests do.)
_run_ab_spec = importlib.util.spec_from_file_location(
    "run_ab", _PROJECT_ROOT / "scripts" / "run_ab.py"
)
run_ab = importlib.util.module_from_spec(_run_ab_spec)
sys.modules["run_ab"] = run_ab
_run_ab_spec.loader.exec_module(run_ab)  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
# Paths + frozen constants (the plan's seed/size are NOT runtime-configurable)
# ---------------------------------------------------------------------------

_GOLDEN_DIR = _PROJECT_ROOT / "evals" / "golden"
_HOLDOUT_FILE = _GOLDEN_DIR / "persona_sessions_holdout.jsonl"
_LOCK_FILE = _GOLDEN_DIR / "holdout.lock"
_RESULTS_DIR = _PROJECT_ROOT / "evals" / "results"
_VALIDATION_FILE = _RESULTS_DIR / "judge_validation.json"
_AB_HOLDOUT_JSON = _RESULTS_DIR / "ab_holdout.json"
_RESULTS_DOC = _PROJECT_ROOT / "docs" / "results_vs_predictions.md"
_CURRICULUM_PATH = _PROJECT_ROOT / "data" / "curriculum" / "scope_sequence.yaml"

# Default workdir (manifest + checkpoints).  Overridable in tests via run().
_DEFAULT_WORKDIR = _RESULTS_DIR / "ab_holdout_work"

# Pre-registered sampling parameters — FROZEN (changing them post-hoc invalidates
# the run; the plan: "any post-hoc change to the sampling seed/size … aborts").
SAMPLE_SEED = 7331
PER_PERSONA = 20  # 20/20/20 stratified -> 60 per version
PROMPT_VERSION = "1.0"

# Prediction #5 is adjudicated on these two judged dimensions (the plan).  icap is
# reported but never part of the verdict.
PRED5_DIMS: tuple[str, ...] = ("guidance", "actionability")

# Bootstrap config for the difference-of-means CI (seeded, deterministic).
_N_BOOT = 2000
_BOOT_SEED = 7331
_CI = 0.95


# ---------------------------------------------------------------------------
# Sampled decision turn (carries everything verbalization + judging need)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionTurn:
    """One sampled decision turn — a pure function of the replay (no judged data).

    ``turn_uid`` is stable and unique across a version's sample: it embeds the
    version, session id and turn index, so the verbalize/judge workfiles key on it
    deterministically and resume correctly.
    """

    turn_uid: str
    version: str
    persona_id: str
    session_id: str
    turn_index: int
    move: str
    miscue_type: str | None
    at_page_end: bool
    target_word: str | None
    hint_level: str | None
    skill_id: str | None
    served_reason: str | None

    def to_action(self) -> TutorAction:
        """Rebuild the TutorAction the policy produced (for verbalization).

        The verbalizer reads only move / target_word / hint_level / error_type;
        the rationale is informational.  Reconstructing from the recorded fields is
        exact for the verbalizer's inputs and avoids re-deriving the policy here.
        """
        return TutorAction(
            move=self.move,
            target_word=self.target_word,
            rationale=f"[holdout-replay {self.version}] reconstructed for verbalization",
            error_type=self.miscue_type,
            hint_level=self.hint_level,  # type: ignore[arg-type]
        )

    def ctx_summary(self) -> dict:
        """The context summary handed to the verbalizer (mirrors generate_turns)."""
        return {
            "at_page_end": self.at_page_end,
            "struggles": 0 if self.at_page_end else 0,
            "miscue": self.miscue_type or "none",
        }


# ---------------------------------------------------------------------------
# Abort helper (loud, non-zero)
# ---------------------------------------------------------------------------


class HoldoutAbort(RuntimeError):
    """Raised to abort the holdout run loudly (mapped to a non-zero exit in main)."""


def _abort(msg: str) -> None:
    raise HoldoutAbort(msg)


# ---------------------------------------------------------------------------
# Phase 0 — freeze verification (in-process hash check) + eligibility loading
# ---------------------------------------------------------------------------


def verify_holdout_lock(
    lock_file: Path = _LOCK_FILE, holdout_file: Path = _HOLDOUT_FILE
) -> dict:
    """Re-hash the holdout JSONL against holdout.lock; abort on any mismatch.

    Equivalent to ``scripts/freeze_split.py --verify`` for the holdout file, in
    process.  Returns the lock dict on success.  Aborts (HoldoutAbort) if the lock
    is missing, the file is missing, or the content hash does not match.
    """
    import hashlib

    if not lock_file.exists():
        _abort(f"holdout lock not found: {lock_file} — cannot verify the freeze")
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    meta = lock["files"].get(holdout_file.name)
    if meta is None:
        _abort(f"holdout.lock has no entry for {holdout_file.name}")
    if not holdout_file.exists():
        _abort(f"holdout corpus not found: {holdout_file}")
    actual = hashlib.sha256(
        holdout_file.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    if actual != meta["sha256"]:
        _abort(
            "HOLDOUT HASH MISMATCH — the frozen corpus does not match holdout.lock.\n"
            f"  expected: {meta['sha256']}\n"
            f"  actual  : {actual}\n"
            "The run is INVALID (the plan: any holdout-lock hash mismatch aborts)."
        )
    return lock


def load_eligible_dims(validation_file: Path | None = None) -> tuple[set[str], dict]:
    """Load gate-eligible judged dimensions from judge_validation.json.

    The plan: judge ONLY the dimensions the validation marks ``gate_eligible``.
    Returns (eligible_dim_names, raw_validation_dict).  Aborts if the file is
    absent ("validation has not run") — the run cannot proceed without it.

    ``validation_file`` defaults to the module-level ``_VALIDATION_FILE`` resolved
    at CALL time (so tests can monkeypatch the module attribute).
    """
    if validation_file is None:
        validation_file = _VALIDATION_FILE
    if not validation_file.exists():
        _abort(
            f"judge validation has not run — {validation_file} does not exist. "
            "The holdout judging gates on gate_eligible dims, which only "
            "judge_validation.json defines. Run scripts/validate_judge.py first."
        )
    data = json.loads(validation_file.read_text(encoding="utf-8"))
    eligible = {
        d["dimension"]
        for d in data.get("dimensions", [])
        if d.get("gate_eligible") is True
    }
    return eligible, data


# ---------------------------------------------------------------------------
# Phase 1 — deterministic replay + aggregation (REUSE run_ab) and sampling capture
# ---------------------------------------------------------------------------


def _load_holdout_sessions() -> list[dict]:
    """Load the 49 frozen holdout sessions (this is the SANCTIONED holdout read)."""
    return [
        json.loads(line)
        for line in _HOLDOUT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def replay_with_decisions(
    version: str,
    sessions: list[dict],
    curriculum: Curriculum,
) -> tuple[list, list[DecisionTurn]]:
    """Replay ``version`` over ``sessions``; return (traces, decision_turns).

    The trace list feeds run_ab._aggregate (identical deterministic metrics).  The
    decision-turn list is the sampling population (one DecisionTurn per miscue or
    page-end turn), capturing the TutorAction inputs so the sampled turns can be
    verbalized later.  This mirrors readcoach.tutor_versions.run_session exactly so
    the traces are byte-for-byte the same as the dev A/B's replay; capturing the
    action alongside is the only addition.

    NOTE: v1/v3 are state-blind; v2 reuses ONE store per persona (cross-session
    memory) — same as run_ab.replay_version.
    """
    if version not in ("v1", "v2"):
        raise ValueError(f"holdout A/B compares v1 and v2 only; got {version!r}")

    # Canonical traces from the shared replay (for aggregation parity).
    traces = run_ab.replay_version(version, sessions, curriculum=curriculum)

    # Capture decision-turn actions by re-deriving them the same way the replay
    # does.  We walk traces + sessions together (run_session preserves event order
    # and one TurnRecord per event), so trace.turns[i] corresponds to the i-th
    # non-trivial event the policy saw.  We re-decide to recover target_word/etc.
    decisions: list[DecisionTurn] = []
    v2_stores: dict[str, InMemoryLearnerStore] = {}
    for item, trace in zip(sessions, traces, strict=True):
        persona_id = item["persona_id"]
        session_id = item["id"]
        if version == "v2":
            store = v2_stores.setdefault(persona_id, InMemoryLearnerStore())
            actions = _v2_actions(item, curriculum, store)
        else:
            actions = _v1_actions(item)
        # actions and trace.turns are 1:1 (both one-per-event, same order).
        for turn, (action, event) in zip(trace.turns, actions, strict=True):
            if not is_decision_turn(turn):
                continue
            decisions.append(
                DecisionTurn(
                    turn_uid=f"{version}__{session_id}__t{turn.turn_index:03d}",
                    version=version,
                    persona_id=persona_id,
                    session_id=session_id,
                    turn_index=turn.turn_index,
                    move=action.move,
                    miscue_type=turn.miscue_type,
                    at_page_end=turn.at_page_end,
                    target_word=action.target_word,
                    hint_level=action.hint_level,
                    skill_id=turn.skill_id,
                    served_reason=turn.served_reason,
                )
            )
    return traces, decisions


def _v1_actions(item: dict) -> list[tuple[TutorAction, dict]]:
    """Re-derive v1 (state-blind) actions per event — mirrors _run_simple."""
    child_id = item["persona_id"]
    out: list[tuple[TutorAction, dict]] = []
    for event in item["events"]:
        if event["kind"] == "page_end":
            ctx = TutorContext(
                miscue=None,
                learner_state=LearnerState(child_id=child_id, mastery={}),
                at_page_end=True,
                consecutive_struggles=0,
                page_had_struggle=bool(event["page_had_struggle"]),
                open_comprehension=False,
            )
        else:
            ctx = TutorContext(
                miscue=_miscue_from_event(event),
                learner_state=LearnerState(child_id=child_id, mastery={}),
                at_page_end=bool(event["at_page_end"]),
                consecutive_struggles=int(event["consecutive_struggles"]),
                page_had_struggle=bool(event["page_had_struggle"]),
                open_comprehension=False,
            )
        out.append((decide(ctx), event))
    return out


def _v2_actions(
    item: dict, curriculum: Curriculum, store: InMemoryLearnerStore
) -> list[tuple[TutorAction, dict]]:
    """Re-derive v2 (mastery-conditioned) actions per event — mirrors _run_v2.

    Records observations into ``store`` exactly as _run_v2 does so the live state
    fed to decide() matches the canonical trace.  ``store`` is the per-persona
    store reused across that persona's sessions (cross-session memory), so this
    must be called in session order with the same store the canonical replay used.
    """
    from readcoach.miscue import _ALL_CLASSES

    child_id = item["persona_id"]
    band = int(item["band"])
    session_id = item["id"]
    skill = _band_representative_skill(curriculum, band)
    miscue_kinds = frozenset(_ALL_CLASSES)
    out: list[tuple[TutorAction, dict]] = []
    for event in item["events"]:
        kind = event["kind"]
        if kind == "page_end":
            ctx = TutorContext(
                miscue=None,
                learner_state=store.get_state(child_id),
                at_page_end=True,
                consecutive_struggles=0,
                page_had_struggle=bool(event["page_had_struggle"]),
                open_comprehension=False,
            )
            out.append((decide(ctx), event))
            continue
        if skill is not None and kind in miscue_kinds:
            store.record_observation(
                child_id=child_id,
                skill=skill,
                correct=False,
                confidence=V2_DETECTOR_CONFIDENCE,
                session_id=session_id,
                ts=_BASE_TS,
                miscue_class=kind,
            )
        ctx = TutorContext(
            miscue=_miscue_from_event(event),
            learner_state=store.get_state(child_id),
            at_page_end=bool(event["at_page_end"]),
            consecutive_struggles=int(event["consecutive_struggles"]),
            page_had_struggle=bool(event["page_had_struggle"]),
            open_comprehension=False,
        )
        out.append((decide(ctx), event))
    return out


# ---------------------------------------------------------------------------
# Phase 2 — stratified sampling (deterministic, replay-only)
# ---------------------------------------------------------------------------


def sample_turns(
    decisions: list[DecisionTurn],
    *,
    per_persona: int = PER_PERSONA,
    seed: int = SAMPLE_SEED,
) -> list[DecisionTurn]:
    """Stratified sample of ``per_persona`` decision turns per persona.

    Deterministic: the population is sorted by a stable key (turn_uid) and a seeded
    random.Random draws ``per_persona`` per persona.  Persona order is sorted, and
    each persona's RNG is reseeded from (seed, persona) so the per-stratum draw is
    independent and reproducible.  The sample depends ONLY on the replay (the
    DecisionTurn list), never on any judged score.

    Aborts if any persona stratum has fewer than ``per_persona`` decision turns
    (the plan's 20/20/20 cannot be honored — degrade is not allowed).
    """
    by_persona: dict[str, list[DecisionTurn]] = {}
    for d in decisions:
        by_persona.setdefault(d.persona_id, []).append(d)

    sampled: list[DecisionTurn] = []
    for persona in sorted(by_persona):
        pool = sorted(by_persona[persona], key=lambda d: d.turn_uid)
        if len(pool) < per_persona:
            _abort(
                f"persona {persona!r} has only {len(pool)} decision turns "
                f"(< {per_persona}); the pre-registered 20/20/20 stratified sample "
                "cannot be honored — aborting rather than degrading."
            )
        # Per-stratum seeded draw: reseed from (seed, persona) for independence.
        rng = random.Random(f"{seed}:{persona}")
        picks = rng.sample(pool, per_persona)
        picks.sort(key=lambda d: d.turn_uid)  # stable output order
        sampled.extend(picks)
    return sampled


# ---------------------------------------------------------------------------
# Workfile checkpointing (shared loud-resume design for verbalize + judge)
# ---------------------------------------------------------------------------


def _load_workfile(
    workfile: Path, required: set[str], key_fields: tuple[str, ...]
) -> dict[tuple, dict]:
    """Load a checkpoint workfile → {key_tuple: row}.  Aborts on any malformation.

    Mirrors scripts/judge_turns._load_workfile: malformed line, missing required
    field, or a duplicate key all ABORT (never skip).  ``key_fields`` is the tuple
    of fields forming the resume key.
    """
    done: dict[tuple, dict] = {}
    if not workfile.exists():
        return done
    for lineno, line in enumerate(
        workfile.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            _abort(
                f"MALFORMED workfile line {lineno} in {workfile}: {exc}. "
                "Fix or delete the workfile to proceed."
            )
        missing = required - set(obj)
        if missing:
            _abort(
                f"MALFORMED workfile line {lineno} in {workfile}: "
                f"missing {sorted(missing)!r}. Fix or delete the workfile."
            )
        key = tuple(str(obj[f]) for f in key_fields)
        if key in done:
            _abort(
                f"MALFORMED workfile: duplicate key {key} at line {lineno} in "
                f"{workfile}. Fix or delete the workfile."
            )
        done[key] = obj
    return done


def _append_workfile(workfile: Path, row: dict) -> None:
    """Atomically append one JSON row (line-append) to the checkpoint workfile."""
    workfile.parent.mkdir(parents=True, exist_ok=True)
    with workfile.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Phase 3 — verbalize sampled turns (checkpointed)
# ---------------------------------------------------------------------------

# A verbalizer is any object with ``.verbalize(action, ctx_summary, prompt_version,
# is_ai_reminder=...)`` — ClaudeCliTransport for the live run, a stub in tests.
VerbalizeFn = Callable[..., str]


def verbalize_phase(
    sample: list[DecisionTurn],
    verbalizer,  # noqa: ANN001 — .verbalize(...) contract
    workfile: Path,
) -> dict[str, str]:
    """Verbalize every sampled turn → {turn_uid: utterance}.  Checkpointed.

    Resume-safe: completed utterances are loaded from ``workfile`` and skipped; new
    ones are appended atomically.  A transport failure ABORTS (the plan forbids a
    skipped sampled turn).  All-or-nothing: returns the full mapping only after
    every sampled turn has an utterance.
    """
    required = {"turn_uid", "utterance"}
    done = _load_workfile(workfile, required, ("turn_uid",))
    print(
        f"  verbalize: resuming {len(done)}/{len(sample)} done",
        file=sys.stderr,
    )
    for d in sample:
        if (d.turn_uid,) in done:
            continue
        try:
            utterance = verbalizer.verbalize(
                d.to_action(), d.ctx_summary(), PROMPT_VERSION, is_ai_reminder=False
            )
        except Exception as exc:  # noqa: BLE001 — abort loud on ANY transport failure
            _abort(
                f"verbalization failed for turn_uid={d.turn_uid!r}: {exc}. "
                "Workfile preserved; rerun to resume. (The plan: a skipped sampled "
                "turn aborts rather than degrades.)"
            )
        row = {"turn_uid": d.turn_uid, "utterance": utterance}
        _append_workfile(workfile, row)
        done[(d.turn_uid,)] = row
    return {uid[0]: row["utterance"] for uid, row in done.items()}


# ---------------------------------------------------------------------------
# Phase 4 — judge each utterance × each eligible dim (checkpointed)
# ---------------------------------------------------------------------------


def judge_phase(
    sample: list[DecisionTurn],
    utterances: dict[str, str],
    judge_dims: list[str],
    transport,  # noqa: ANN001 — CodexCliTransport or a stub
    workfile: Path,
    *,
    judge_fn: Callable[..., object] = judge_turn,
) -> dict[tuple[str, str], dict]:
    """Judge every (sampled turn, eligible dim) → {(turn_uid, dim): verdict_row}.

    Checkpointed and resume-safe, same loud-resume design as verbalize_phase.  A
    JudgeError ABORTS (no default verdicts, no skipped pairs).  All-or-nothing.
    """
    required = {"turn_uid", "dimension", "score", "passing"}
    done = _load_workfile(workfile, required, ("turn_uid", "dimension"))
    total = len(sample) * len(judge_dims)
    print(f"  judge: resuming {len(done)}/{total} done", file=sys.stderr)

    by_uid = {d.turn_uid: d for d in sample}
    for d in sample:
        for dim in judge_dims:
            key = (d.turn_uid, dim)
            if key in done:
                continue
            turn = {
                "utterance": utterances[d.turn_uid],
                "move": d.move,
                "miscue": d.miscue_type or "",
            }
            try:
                verdict = judge_fn(turn, dim, transport=transport)
            except JudgeError as exc:
                _abort(
                    f"judge failed for turn_uid={d.turn_uid!r}, dim={dim!r}: {exc}. "
                    "Workfile preserved; rerun to resume."
                )
            row = {
                "turn_uid": d.turn_uid,
                "dimension": verdict.dimension,
                "score": verdict.score,
                "passing": verdict.passing,
                "persona_id": by_uid[d.turn_uid].persona_id,
                "version": d.version,
            }
            _append_workfile(workfile, row)
            done[key] = row
    return done


# ---------------------------------------------------------------------------
# Phase 5 — analysis (all-or-nothing) + adjudication
# ---------------------------------------------------------------------------


def _diff_of_means_ci(
    v1_scores: list[float], v2_scores: list[float], *, seed: int
) -> tuple[float, float]:
    """INDEPENDENT seeded bootstrap CI for mean(v2) - mean(v1).

    The two samples are drawn from DIFFERENT action streams (see the module
    docstring), so they are resampled INDEPENDENTLY within their own 60-turn draws.
    One numpy Generator seeded once drives both index draws per replicate, so the
    whole computation is deterministic for a fixed seed + fixed score vectors.

    Returns the (lo, hi) percentile-method 95% CI bounds of the difference.  If
    either sample is empty the difference is undefined and (nan, nan) is returned
    (the caller's mean_v1/mean_v2 are None in that case and v2_beats_v1 is False).
    """
    import numpy as np

    if not v1_scores or not v2_scores:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    a = np.asarray(v1_scores, dtype=float)
    b = np.asarray(v2_scores, dtype=float)
    n1, n2 = len(a), len(b)
    diffs = np.empty(_N_BOOT, dtype=float)
    for i in range(_N_BOOT):
        ia = rng.integers(0, n1, size=n1)
        ib = rng.integers(0, n2, size=n2)
        diffs[i] = b[ib].mean() - a[ia].mean()
    alpha = 1.0 - _CI
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return lo, hi


def _scores_for(
    verdicts: dict[tuple[str, str], dict], dim: str, version: str
) -> list[float]:
    """All judged scores for one dimension and one version."""
    return [
        float(row["score"])
        for (_, d), row in verdicts.items()
        if d == dim and row["version"] == version
    ]


def analyze(
    verdicts: dict[tuple[str, str], dict],
    judge_dims: list[str],
    eligible: set[str],
) -> dict:
    """Per-dimension means + diff-of-means CI; prediction-#5 adjudication.

    All-or-nothing: this is only called when every (turn, dim) verdict exists.
    """
    per_dim: dict[str, dict] = {}
    for dim in judge_dims:
        v1 = _scores_for(verdicts, dim, "v1")
        v2 = _scores_for(verdicts, dim, "v2")
        mean_v1 = sum(v1) / len(v1) if v1 else None
        mean_v2 = sum(v2) / len(v2) if v2 else None
        lo, hi = _diff_of_means_ci(v1, v2, seed=_BOOT_SEED)
        diff = (mean_v2 - mean_v1) if (mean_v1 is not None and mean_v2 is not None) else None
        v2_beats_v1 = bool(diff is not None and diff > 0 and lo > 0)
        per_dim[dim] = {
            "dimension": dim,
            "eligible": dim in eligible,
            "n_v1": len(v1),
            "n_v2": len(v2),
            "mean_v1": mean_v1,
            "mean_v2": mean_v2,
            "diff_v2_minus_v1": diff,
            "diff_ci_95": [lo, hi],
            "v2_beats_v1": v2_beats_v1,
            "note": "icap is reported, not adjudicated for prediction #5."
            if dim == "icap"
            else None,
        }

    verdict, finding = _adjudicate_pred5(per_dim, eligible)
    return {
        "per_dimension": per_dim,
        "pairing": "independent",
        "pairing_note": (
            "True turn-level pairing is IMPOSSIBLE: v1 and v2 are sampled from "
            "different action streams (the two policies make different moves), so "
            "there is no canonical 1:1 turn correspondence. The plan's 'paired "
            "bootstrap' is therefore implemented as an INDEPENDENT seeded bootstrap "
            "of the difference of means (each 60-turn sample resampled within "
            "itself; fixed seed). This is the closest correct construction; we do "
            "NOT silently pretend the samples are paired."
        ),
        "prediction_5": {
            "verdict": verdict,
            "adjudicated_dims": list(PRED5_DIMS),
            "finding": finding,
            "mapping": (
                "both eligible {guidance, actionability} positive-and-significant -> "
                "CONFIRMED; one of two -> PARTIAL; neither -> MISSED; any of the two "
                "ineligible -> that part UNADJUDICABLE (reported, not dropped)."
            ),
        },
    }


def _adjudicate_pred5(per_dim: dict[str, dict], eligible: set[str]) -> tuple[str, str]:
    """Map per-dimension results to the plan's CONFIRMED/PARTIAL/MISSED/UNADJUDICABLE.

    Plan rule (verbatim): for each eligible dim among {guidance, actionability},
    v2 "beats" v1 iff mean diff (v2-v1) > 0 AND its 95% CI excludes zero.

      * any of the two dims INELIGIBLE -> that part is UNADJUDICABLE (a finding).
      * both eligible + both beat -> CONFIRMED.
      * both eligible + exactly one beats -> PARTIAL.
      * both eligible + neither beats -> MISSED.
    """
    ineligible = [d for d in PRED5_DIMS if d not in eligible]
    if ineligible:
        return "UNADJUDICABLE", (
            f"dimension(s) {ineligible} failed judge validation (not gate_eligible), "
            "so prediction #5's claim on them cannot be adjudicated. Reported as a "
            "finding, not silently dropped."
        )
    beats = [d for d in PRED5_DIMS if per_dim[d]["v2_beats_v1"]]
    if len(beats) == len(PRED5_DIMS):
        return "CONFIRMED", (
            "v2 beat v1 (positive mean diff, CI excludes zero) on BOTH "
            f"{list(PRED5_DIMS)}."
        )
    if len(beats) == 1:
        return "PARTIAL", (
            f"v2 beat v1 on {beats} but not on "
            f"{[d for d in PRED5_DIMS if d not in beats]}."
        )
    return "MISSED", "v2 did not beat v1 on either guidance or actionability."


# ---------------------------------------------------------------------------
# Phase 1 reports (REUSE run_ab.evaluate + compare on TUTOR_GATE_RULES)
# ---------------------------------------------------------------------------


def _holdout_report(version: str, metrics: dict) -> EvalReport:
    """Write the immutable per-version holdout report (versions v1/v2-holdout)."""
    return evaluate(
        f"{version}-holdout",
        str(_HOLDOUT_FILE),
        metrics=metrics,
        results_dir=str(_RESULTS_DIR),
    )


# ---------------------------------------------------------------------------
# Live-only doc append (gated; the controller's live run triggers this)
# ---------------------------------------------------------------------------


def append_prediction5_verdict(analysis: dict, results_doc: Path = _RESULTS_DOC) -> None:
    """Append the prediction-#5 verdict block to docs/results_vs_predictions.md.

    ONLY called on a completed LIVE run (gated by --live in main).  Replaces the
    'Status: pending' stub under the Prediction 5 heading with the adjudicated
    verdict + per-dimension table + the honest-pairing note.  Idempotent-ish: if a
    holdout-verdict block already exists it is NOT duplicated (a second live run
    would no-op the append).
    """
    pred5 = analysis["prediction_5"]
    marker = "<!-- holdout-prediction-5-verdict -->"
    text = results_doc.read_text(encoding="utf-8")
    if marker in text:
        return  # already appended

    lines = [
        "",
        marker,
        "### Holdout adjudication (run via scripts/run_ab_holdout.py --live)",
        "",
        f"**Verdict: {pred5['verdict']}**",
        "",
        f"{pred5['finding']}",
        "",
        "| dimension | eligible | mean v1 | mean v2 | diff (v2-v1) | 95% CI | v2 beats v1 |",
        "|-----------|----------|---------|---------|--------------|--------|-------------|",
    ]
    for dim, d in analysis["per_dimension"].items():
        ci = d["diff_ci_95"]
        lines.append(
            f"| {dim} | {d['eligible']} | {_fmt(d['mean_v1'])} | {_fmt(d['mean_v2'])} | "
            f"{_fmt(d['diff_v2_minus_v1'])} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{d['v2_beats_v1']} |"
        )
    lines += [
        "",
        f"_Pairing: {analysis['pairing']}._ {analysis['pairing_note']}",
        "",
    ]
    results_doc.write_text(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _fmt(x: float | None) -> str:
    return "N/A" if x is None else f"{x:.3f}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    *,
    verbalizer,  # noqa: ANN001
    judge_transport,  # noqa: ANN001
    workdir: Path = _DEFAULT_WORKDIR,
    live: bool = False,
    judge_fn: Callable[..., object] = judge_turn,
    write_reports: bool = True,
) -> dict:
    """Execute all phases and return the assembled ab_holdout payload.

    ``verbalizer`` / ``judge_transport`` are injected so tests stub the transports.
    ``write_reports`` writes the immutable per-version reports (skipped in tests to
    avoid the evaluate() immutability collision against the real results dir).
    ``live`` gates the prediction-#5 doc append (only a real run appends).
    """
    workdir.mkdir(parents=True, exist_ok=True)

    # --- Phase 0: freeze + eligibility -----------------------------------
    verify_holdout_lock()
    eligible, validation = load_eligible_dims()
    # Judge eligible judged dims, plus icap if eligible (reported, not adjudicated).
    from evals.judge import JUDGED_DIMENSIONS

    judge_dims = [d for d in JUDGED_DIMENSIONS if d in eligible]
    if not judge_dims:
        _abort(
            "no judged dimension is gate_eligible in judge_validation.json — there "
            "is nothing to judge on the holdout. Reported as a finding (the judge "
            "failed validation on every dimension)."
        )

    curriculum = load_curriculum(_CURRICULUM_PATH)
    sessions = _load_holdout_sessions()
    if len(sessions) != 49:
        _abort(f"expected 49 holdout sessions, found {len(sessions)}")

    # --- Phase 1: deterministic replay + aggregation + sampling capture ---
    v1_traces, v1_decisions = replay_with_decisions("v1", sessions, curriculum)
    v2_traces, v2_decisions = replay_with_decisions("v2", sessions, curriculum)
    v1_metrics = run_ab._aggregate(v1_traces)
    v2_metrics = run_ab._aggregate(v2_traces)

    if write_reports:
        v1_report = _holdout_report("v1", v1_metrics)
        v2_report = _holdout_report("v2", v2_metrics)
    else:
        v1_report = EvalReport("v1-holdout", v1_metrics, {})
        v2_report = EvalReport("v2-holdout", v2_metrics, {})
    gate = compare(v1_report, v2_report, run_ab.TUTOR_GATE_RULES)

    # --- Phase 2: stratified sample (replay-only) -------------------------
    v1_sample = sample_turns(v1_decisions)
    v2_sample = sample_turns(v2_decisions)
    sample = v1_sample + v2_sample
    _write_sample_manifest(workdir / "sample_manifest.json", v1_sample, v2_sample)

    # --- Phase 3: verbalize (checkpointed) --------------------------------
    utterances = verbalize_phase(sample, verbalizer, workdir / ".verbalize_work.jsonl")

    # --- Phase 4: judge (checkpointed) ------------------------------------
    verdicts = judge_phase(
        sample, utterances, judge_dims, judge_transport,
        workdir / ".judge_work.jsonl", judge_fn=judge_fn,
    )

    # --- Phase 5: analysis (all-or-nothing) -------------------------------
    analysis = analyze(verdicts, judge_dims, eligible)

    payload = {
        "ticket": "T5.4",
        "what_this_is": (
            "HOLDOUT A/B (v1 state-blind vs v2 mastery-conditioned) implementing "
            "docs/ab_holdout_plan.md verbatim. Deterministic metrics over all 49 "
            "holdout sessions; 60 seeded stratified judged turns per version; "
            "prediction-#5 adjudication."
        ),
        "split": "holdout",
        "holdout_file": _HOLDOUT_FILE.name,
        "date": date.today().isoformat(),
        "live": live,
        "sampling": {
            "seed": SAMPLE_SEED,
            "per_persona": PER_PERSONA,
            "n_per_version": len(v1_sample),
            "stratified_by": "persona (20/20/20)",
            "depends_on": "replay only (no judged data)",
        },
        "judge_eligibility": {
            "gate_eligible_dims": sorted(eligible),
            "judged_dims": judge_dims,
            "validation_file": _VALIDATION_FILE.name,
            "gate_conditions": validation.get("gate_conditions"),
        },
        "deterministic": {
            "n_sessions": len(sessions),
            "metrics": {"v1": v1_metrics, "v2": v2_metrics},
            "tutor_gate_rules": [
                {"metric": r.metric, "direction": r.direction, "threshold": r.threshold}
                for r in run_ab.TUTOR_GATE_RULES
            ],
            "gate_v1_to_v2": {
                "passed": gate.passed,
                "exit_code": gate.exit_code,
                "breaches": gate.breaches,
            },
        },
        "judged": analysis,
    }
    return payload


def _write_sample_manifest(
    path: Path, v1_sample: list[DecisionTurn], v2_sample: list[DecisionTurn]
) -> None:
    """Write the audit manifest: every sampled turn's context + action (no scores).

    This is the replay-only sample, committed to the workdir BEFORE any judged data
    exists, so the sample's independence from results is auditable.
    """
    manifest = {
        "seed": SAMPLE_SEED,
        "per_persona": PER_PERSONA,
        "v1": [asdict(d) for d in v1_sample],
        "v2": [asdict(d) for d in v2_sample],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Holdout A/B runner — the pre-registered held-out evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--i-am-the-preregistered-run",
        action="store_true",
        help=(
            "Required acknowledgement that this is the SANCTIONED holdout read "
            "executed under docs/ab_holdout_plan.md. Without it the runner refuses "
            "to touch the frozen holdout."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Build the REAL claude-cli verbalizer + codex judge transport and append "
            "the prediction-#5 verdict to docs/results_vs_predictions.md on success. "
            "Omitting --live is an error here (there is no mocked CLI entrypoint; "
            "tests call run() directly with stubbed transports)."
        ),
    )
    args = parser.parse_args(argv)

    if not args.i_am_the_preregistered_run:
        print(
            "REFUSING to run the holdout A/B without "
            "--i-am-the-preregistered-run.\n"
            "This is the one sanctioned read of the frozen held-out split; it must "
            "be invoked deliberately, under docs/ab_holdout_plan.md.",
            file=sys.stderr,
        )
        return 2
    if not args.live:
        print(
            "REFUSING to run without --live. The CLI entrypoint performs the REAL "
            "verbalization + judging. Mocked runs go through run() in tests.",
            file=sys.stderr,
        )
        return 2

    # Build the real transports (fail loud if the binaries are absent at call time).
    from evals.judge import CodexCliTransport

    verbalizer = ClaudeCliTransport()
    judge_transport = CodexCliTransport()

    try:
        payload = run(
            verbalizer=verbalizer,
            judge_transport=judge_transport,
            live=True,
        )
    except HoldoutAbort as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _AB_HOLDOUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Live-only side effect: append the adjudicated verdict to the predictions doc.
    append_prediction5_verdict(payload["judged"])

    _print_summary(payload)
    print(f"\nWrote {_AB_HOLDOUT_JSON.relative_to(_PROJECT_ROOT)}")
    return 0


def _print_summary(payload: dict) -> None:
    print("=" * 72)
    print("HOLDOUT A/B — pre-registered (docs/ab_holdout_plan.md)")
    print("=" * 72)
    det = payload["deterministic"]
    print(f"holdout sessions: {det['n_sessions']}")
    g = det["gate_v1_to_v2"]
    print(f"deterministic gate v1->v2: exit {g['exit_code']}  "
          f"{'PASS' if g['passed'] else 'BLOCKED'}")
    for b in g["breaches"]:
        print(f"  breach: {b}")
    je = payload["judge_eligibility"]
    print(f"\njudged dims (gate_eligible): {je['judged_dims']}")
    pred5 = payload["judged"]["prediction_5"]
    print(f"prediction #5 verdict: {pred5['verdict']}")
    print(f"  {pred5['finding']}")
    for dim, d in payload["judged"]["per_dimension"].items():
        ci = d["diff_ci_95"]
        print(
            f"  {dim:<14} mean_v1={_fmt(d['mean_v1'])} mean_v2={_fmt(d['mean_v2'])} "
            f"diff={_fmt(d['diff_v2_minus_v1'])} CI=[{ci[0]:.3f},{ci[1]:.3f}] "
            f"beats={d['v2_beats_v1']} eligible={d['eligible']}"
        )
    print(f"\npairing: {payload['judged']['pairing']} (independent — see note)")


if __name__ == "__main__":
    sys.exit(main())
