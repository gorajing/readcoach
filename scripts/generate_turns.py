"""Generate verbalized tutoring turns for the labeling set (T4.3).

THREE hand-scripted reader profiles (struggling-decoder, fluent-but-hesitant,
self-corrector) × ~24 turns each.  For each scripted step we:

    1. build a TutorContext, call ``readcoach.tutor.decide`` (the policy);
    2. call ``verbalize()`` (the utterance layer) to get the line said to the child;
    3. record a TurnRecord (move, hint rung, served reason, utterance, AI-reminder
       flag, skill id).

Outputs (under ``--out-dir``, default ``evals/results``):
    * ``turns_v1.jsonl``           — one JSON turn per line, across all profiles;
    * ``trace_<profile>.json``     — a full SessionTrace per profile.

After generation it runs the policy-compiler audit over every trace and prints
the violation count; the run exits non-zero if ANY invariant is violated (the
hand-scripted profiles are designed to be clean, so a non-zero count is a real
regression in the policy/verbalizer, not in the script).

Transports
----------
``--transport api`` (default)
    Uses ``TutorVerbalizer`` with the Anthropic SDK (forced tool-use).  Requires
    ``ANTHROPIC_API_KEY``.  FAILS LOUD at startup if the key is absent.

``--transport claude-cli``
    Uses ``ClaudeCliTransport`` (strict-JSON prompt via ``claude -p``).  No API
    key required — uses the subscription CLI binary.  The model is pinned to
    ``claude-sonnet-4-6``; transport/model metadata is stamped on every turn record.

Usage
-----
    uv run python scripts/generate_turns.py --transport claude-cli
    uv run python scripts/generate_turns.py --transport api [--out-dir evals/results] [--prompt-version 1.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from readcoach.llm_client import ClaudeCliTransport, TutorVerbalizer, cli_transport  # noqa: E402
from readcoach.miscue import Miscue  # noqa: E402
from readcoach.learner_model import LearnerState  # noqa: E402
from readcoach.policy_compiler import audit, compile_rules, load_policies  # noqa: E402
from readcoach.trace import SessionTrace, TurnRecord, trace_to_dict  # noqa: E402
from readcoach.tutor import POLICY_VERSION, TutorContext, decide  # noqa: E402

_POLICIES_DIR = _PROJECT_ROOT / "policies"
_DEFAULT_OUT = _PROJECT_ROOT / "evals" / "results"

# Reminder cadence must satisfy periodic_ai_reminder (window N=20): place a
# reminder on the FIRST turn of every profile (covers the first window) and again
# well within 20 turns.  Both reminder indices are < 20 and span the profile.
_REMINDER_TURN_INDICES = frozenset({0, 12})


@dataclass(frozen=True)
class Step:
    """One scripted decision context + the serve/skill metadata for the turn.

    The reader-state inputs (miscue, struggles, page position) drive ``decide``;
    ``skill_id`` / ``served_reason`` are quest-planner metadata recorded on the
    TurnRecord for the never_reserves_completed_item check.
    """

    miscue: Miscue | None
    at_page_end: bool
    consecutive_struggles: int = 0
    page_had_struggle: bool = False
    open_comprehension: bool = False
    skill_id: str | None = None
    served_reason: str | None = None


@dataclass(frozen=True)
class Profile:
    name: str
    completed_skills_at_start: tuple[str, ...]
    mastery: dict[str, float]
    steps: tuple[Step, ...]


# ---------------------------------------------------------------------------
# Hand-scripted profiles (plausibility over tuning; documented, deterministic)
# ---------------------------------------------------------------------------

def _sub(word: str) -> Miscue:
    return Miscue(type="substitution", target_word=word, said_word="?", index=0)


def _hes(word: str) -> Miscue:
    return Miscue(type="hesitation", target_word=word, said_word=None, index=0)


def _self_corr(word: str) -> Miscue:
    return Miscue(type="self_correction", target_word=word, said_word="?", index=0, confidence=0.7)


def _struggling_decoder() -> Profile:
    """Sustained decoding struggles -> hints escalate to modeling; page-end coaches.

    24 steps: clusters of mid-page struggle on the same word (escalating to a
    SCAFFOLDED_HINT then MODEL_THE_WORD), with page-end coaching moves.
    """
    steps: list[Step] = []
    words = ["chip", "shut", "thick", "brave", "stump", "clung"]
    # Each page-end serves a DISTINCT new skill — re-serving the same skill as
    # "new" would (correctly) trip never_reserves_completed_item.
    new_skills = [
        "digraph_ch", "digraph_sh", "digraph_th",
        "blend_br", "blend_st", "blend_cl",
    ]
    for w, skill in zip(words, new_skills, strict=True):
        steps.append(Step(miscue=_sub(w), at_page_end=False, consecutive_struggles=1))
        steps.append(Step(miscue=_sub(w), at_page_end=False, consecutive_struggles=2))  # hint
        steps.append(Step(miscue=_sub(w), at_page_end=False, consecutive_struggles=3))  # model
        # Page-end after the struggle: coaching is allowed here.
        steps.append(Step(
            miscue=None, at_page_end=True, page_had_struggle=True,
            skill_id=skill, served_reason="new",
        ))
    return Profile(
        name="struggling-decoder",
        completed_skills_at_start=("cvc_short_a",),
        mastery={"digraph_ch": 0.40, "cvc_blend": 0.55, "vowel_team_ea": 0.30},
        steps=tuple(steps),
    )


def _fluent_but_hesitant() -> Profile:
    """Reads accurately but pauses; hesitation alone never coaches mid-page.

    24 steps: many mid-page hesitations (WAIT — productive struggle protected),
    page-ends mostly ENCOURAGE / COMPREHENSION_PROMPT (no struggle to escalate).
    """
    steps: list[Step] = []
    words = ["meadow", "through", "thought", "island", "answer", "bought"]
    for i, w in enumerate(words):
        steps.append(Step(miscue=_hes(w), at_page_end=False))
        steps.append(Step(miscue=_hes(w), at_page_end=False))
        steps.append(Step(miscue=None, at_page_end=False))  # clean mid-page read -> WAIT
        steps.append(Step(
            miscue=None, at_page_end=True,
            open_comprehension=(i % 2 == 0),  # alternate comprehension / encourage
            skill_id="vowel_team_ea", served_reason="review",
        ))
    return Profile(
        name="fluent-but-hesitant",
        completed_skills_at_start=("cvc_short_a", "digraph_ch"),
        mastery={"vowel_team_ea": 0.88, "r_controlled_ar": 0.72},
        steps=tuple(steps),
    )


def _self_corrector() -> Profile:
    """Catches and fixes own reads; self-corrections are NEVER corrected.

    24 steps: mid-page self-corrections (WAIT, honored), page-ends celebrate.
    """
    steps: list[Step] = []
    words = ["cabin", "wagon", "robin", "lemon", "seven", "melon"]
    # Distinct new skill per page-end (see struggling-decoder note).
    new_skills = [
        "silent_e", "vce_a_e", "vce_i_e",
        "open_syllable", "closed_syllable", "schwa",
    ]
    for w, skill in zip(words, new_skills, strict=True):
        steps.append(Step(miscue=_self_corr(w), at_page_end=False, consecutive_struggles=1))
        steps.append(Step(miscue=_self_corr(w), at_page_end=False, consecutive_struggles=2))
        steps.append(Step(miscue=None, at_page_end=False))  # clean read -> WAIT
        steps.append(Step(
            miscue=None, at_page_end=True,
            skill_id=skill, served_reason="new",
        ))
    return Profile(
        name="self-corrector",
        completed_skills_at_start=("cvc_short_a",),
        mastery={"silent_e": 0.66, "cvc_blend": 0.70},
        steps=tuple(steps),
    )


def build_profiles() -> list[Profile]:
    return [_struggling_decoder(), _fluent_but_hesitant(), _self_corrector()]


# ---------------------------------------------------------------------------
# Generation core (verbalizer injected so tests stub the transport)
# ---------------------------------------------------------------------------

def _turn_for_step(
    *,
    turn_index: int,
    step: Step,
    profile: Profile,
    verbalizer,  # noqa: ANN001
    prompt_version: str,
) -> TurnRecord:
    ctx = TutorContext(
        miscue=step.miscue,
        learner_state=LearnerState(child_id=profile.name, mastery=dict(profile.mastery)),
        at_page_end=step.at_page_end,
        consecutive_struggles=step.consecutive_struggles,
        page_had_struggle=step.page_had_struggle,
        open_comprehension=step.open_comprehension,
    )
    action = decide(ctx)
    is_ai_reminder = turn_index in _REMINDER_TURN_INDICES
    ctx_summary = {
        "at_page_end": step.at_page_end,
        "struggles": step.consecutive_struggles,
        "miscue": step.miscue.type if step.miscue else "none",
    }
    utterance = verbalizer.verbalize(
        action, ctx_summary, prompt_version, is_ai_reminder=is_ai_reminder
    )
    return TurnRecord(
        turn_index=turn_index,
        at_page_end=step.at_page_end,
        miscue_type=(step.miscue.type if step.miscue else None),
        action_move=action.move,
        hint_level=action.hint_level,
        served_reason=step.served_reason,
        utterance=utterance,
        is_ai_reminder=is_ai_reminder,
        skill_id=step.skill_id,
    )


def run(
    *,
    verbalizer,  # noqa: ANN001
    out_dir,  # noqa: ANN001
    prompt_version: str = "1.0",
    transport_meta: dict | None = None,
) -> dict:
    """Generate turns for all profiles; write jsonl + traces; audit; return summary.

    Returns ``{"violations": int, "n_turns": int, "traces": [paths]}``.  Pure of
    any live transport — ``verbalizer`` is injected.

    ``transport_meta`` (optional) is stamped into every JSONL turn record for
    provenance.  Pass ``verbalizer.transport_meta`` when using
    ``ClaudeCliTransport``; omit for the SDK transport.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    profiles = build_profiles()
    checks = compile_rules(load_policies(_POLICIES_DIR))

    jsonl_lines: list[str] = []
    trace_paths: list[Path] = []
    total_violations = 0
    n_turns = 0
    meta = transport_meta or {}

    for profile in profiles:
        records: list[TurnRecord] = []
        for i, step in enumerate(profile.steps):
            rec = _turn_for_step(
                turn_index=i,
                step=step,
                profile=profile,
                verbalizer=verbalizer,
                prompt_version=prompt_version,
            )
            records.append(rec)
            turn_dict = {
                "profile": profile.name,
                "prompt_version": prompt_version,
                **_record_to_dict(rec),
                **meta,
            }
            jsonl_lines.append(json.dumps(turn_dict, sort_keys=True))
            n_turns += 1

        trace = SessionTrace(
            child_id=profile.name,
            policy_version=POLICY_VERSION,
            completed_skills_at_start=profile.completed_skills_at_start,
            turns=tuple(records),
        )
        trace_path = out / f"trace_{profile.name}.json"
        trace_path.write_text(json.dumps(trace_to_dict(trace), indent=2, sort_keys=True))
        trace_paths.append(trace_path)

        report = audit(trace, checks)
        total_violations += report.violations
        if report.violations:
            for f in report.findings:
                if f.severity == "error":
                    print(
                        f"  VIOLATION [{profile.name}] turn {f.turn_index} "
                        f"{f.rule_id}: {f.message}",
                        file=sys.stderr,
                    )

    (out / "turns_v1.jsonl").write_text("\n".join(jsonl_lines) + "\n")

    return {"violations": total_violations, "n_turns": n_turns, "traces": trace_paths}


def _record_to_dict(rec: TurnRecord) -> dict:
    return {
        "turn_index": rec.turn_index,
        "at_page_end": rec.at_page_end,
        "miscue_type": rec.miscue_type,
        "action_move": rec.action_move,
        "hint_level": rec.hint_level,
        "served_reason": rec.served_reason,
        "utterance": rec.utterance,
        "is_ai_reminder": rec.is_ai_reminder,
        "skill_id": rec.skill_id,
    }


# ---------------------------------------------------------------------------
# CLI — builds the REAL verbalizer (fails loud with no key / binary) then runs.
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate verbalized tutoring turns.")
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUT))
    parser.add_argument("--prompt-version", default="1.0")
    parser.add_argument(
        "--transport",
        choices=["api", "claude-cli"],
        default="api",
        help=(
            "Transport to use for verbalization. "
            "'api' uses the Anthropic SDK (requires ANTHROPIC_API_KEY). "
            "'claude-cli' uses the subscription CLI binary (claude -p, no key needed)."
        ),
    )
    args = parser.parse_args(argv)

    transport_meta: dict | None = None

    if args.transport == "claude-cli":
        verbalizer = cli_transport()
        transport_meta = ClaudeCliTransport.transport_meta
        print(
            f"transport: claude-cli (model={ClaudeCliTransport.transport_meta['model']})"
        )
    else:
        # api mode — build the real TutorVerbalizer; FAIL LOUD if key is absent.
        verbalizer = TutorVerbalizer()
        # Force the no-key check up front (the factory is otherwise lazy).
        verbalizer._client_or_build()  # noqa: SLF001 — deliberate eager fail-loud
        print("transport: api (Anthropic SDK)")

    result = run(
        verbalizer=verbalizer,
        out_dir=args.out_dir,
        prompt_version=args.prompt_version,
        transport_meta=transport_meta,
    )

    print(
        f"generated {result['n_turns']} turns across 3 profiles -> {args.out_dir}"
    )
    print(f"invariant violations: {result['violations']}")
    if result["violations"] != 0:
        print("FAIL: invariants violated — see findings above.", file=sys.stderr)
        sys.exit(1)
    print("OK: 0 invariant violations.")
    sys.exit(0)


if __name__ == "__main__":
    main()
