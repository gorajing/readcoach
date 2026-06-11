#!/usr/bin/env python3
"""T5.4 — pre-registered A/B + flywheel + sensitivity (deterministic dev-split slice).

WHAT THIS RUNS TONIGHT (and what it deliberately does NOT)
---------------------------------------------------------
The judge's dimensions are NOT yet human-validated, so JUDGED scoring gates
NOTHING and is DEFERRED.  Tonight's A/B is the DETERMINISTIC comparison on the
DEV split only:

  * v1 (state-blind) vs v2 (mastery-conditioned) -> compare() -> the gate PASSES,
    plus a v1→v2 improvement curve (per-metric deltas).
  * v3 (always-intervene, deliberately worse) -> compare() -> the gate BLOCKS it
    (exit 1) with the invariants breach NAMED.  This is RECEIPT #2: the gate
    catches the bad tutor.
  * promote_failure growth chart: v3's violating turns are promoted into a
    tutor-failures golden; cumulative golden size is charted per batch.
  * Sensitivity (LAST): ±30% persona-rate perturbation -> regenerate ~49
    perturbed sessions (seeded, NOT touching the frozen files, out-of-band ids)
    -> rerun the v1/v2 comparison -> the gate outcome must HOLD.

PRE-REGISTRATION SCOPE (stated in the JSON)
-------------------------------------------
docs/predictions.md Prediction #5 covers JUDGED guidance/actionability on the
HELD-OUT split — adjudicated LATER, after judge validation.  The deterministic
comparison tonight is NOT pre-registered as a prediction, so this artifact
REPORTS deterministic diffs WITHOUT adjudicating any prediction.

THE FROZEN HOLDOUT IS NEVER READ HERE
-------------------------------------
``run_ab`` REFUSES a holdout path (see ``_assert_not_holdout``).  Tonight's code
only ever reads ``persona_sessions_dev.jsonl``; ``scripts/freeze_split.py
--verify`` stays green after this run, proving the frozen files are untouched.

Usage
-----
    uv run python scripts/run_ab.py [--skip-sensitivity]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from evals.harness import EvalReport, GateRule, compare, evaluate, promote_failure  # noqa: E402
from readcoach.inject import load_passages  # noqa: E402
from readcoach.policy_compiler import audit, compile_rules, load_policies  # noqa: E402
from readcoach.persona_gen import (  # noqa: E402
    generate_session,
    load_personas,
    session_item_to_dict,
)
from readcoach.persona_gen import _child_rng  # noqa: E402  (seeded child RNG — reused for parity)
from readcoach.planner import load_curriculum  # noqa: E402
from readcoach.learner_store import InMemoryLearnerStore  # noqa: E402
from readcoach.trace import SessionTrace  # noqa: E402
from readcoach.tutor_versions import is_decision_turn, run_session  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_GOLDEN_DIR = _PROJECT_ROOT / "evals" / "golden"
_DEV_FILE = _GOLDEN_DIR / "persona_sessions_dev.jsonl"
_HOLDOUT_FILE = _GOLDEN_DIR / "persona_sessions_holdout.jsonl"
_TUTOR_FAILURES = _GOLDEN_DIR / "tutor_failures.jsonl"
_RESULTS_DIR = _PROJECT_ROOT / "evals" / "results"
_AB_JSON = _RESULTS_DIR / "ab_dev.json"
_GROWTH_PNG = _RESULTS_DIR / "promote_growth.png"
_CURRICULUM_PATH = _PROJECT_ROOT / "data" / "curriculum" / "scope_sequence.yaml"
_PERSONAS_DIR = _PROJECT_ROOT / "data" / "personas"
_PASSAGES_DIR = _PROJECT_ROOT / "data" / "passages"

# Sensitivity perturbation: ±30% on every persona per-class rate.
_SENSITIVITY_DELTA = 0.30
# Out-of-band seed so perturbed session ids never collide with the frozen corpus.
_SENSITIVITY_SEED = 99_531

# ---------------------------------------------------------------------------
# Pedagogy-only audit (action-level traces have no utterance / no AI-reminder)
# ---------------------------------------------------------------------------
#
# These traces are ACTION-LEVEL (utterance=None, no LLM verbalization), so the
# SAFETY policy set — whose rules live in the verbalization layer
# (``periodic_ai_reminder`` cadence, ``no_emotional_intimacy`` lexicon) — does NOT
# apply: it would flag every turn for a missing AI-identity reminder the action
# policy never emits, drowning the real signal.  The deterministic A/B audits the
# PEDAGOGY policy set (the action-level invariants: never_coaches_mid_page,
# never_corrects_self_correction, never_reserves_completed_item, never_says_wrong).
# The safety/verbalization invariants are exercised by the LIVE turn-generation
# eval (scripts/generate_turns.py), not by this action-level replay.
_POLICIES_DIR = _PROJECT_ROOT / "policies"


def _pedagogy_checks():
    """Compile only the PEDAGOGY policy-set rules (action-level, no utterance)."""
    rules = [r for r in load_policies(_POLICIES_DIR) if r.policy_set == "pedagogy"]
    return compile_rules(rules)


def _serve_check():
    """Compile only the never_reserves_completed_item rule (for serve_violations)."""
    rules = [
        r for r in load_policies(_POLICIES_DIR)
        if r.id == "never_reserves_completed_item"
    ]
    return compile_rules(rules)


# ---------------------------------------------------------------------------
# TUTOR-version gate table (defined HERE, in the A/B runner, per the plan)
# ---------------------------------------------------------------------------
#
# These rules gate a TUTOR-version comparison (distinct from scripts/gate.py's
# miscue-benchmark GATE_RULES).  Encoded:
#
#   invariants.violations         min  threshold 0      — never violate an invariant
#   wait_rate                     max  threshold 0.35   — new >= 0.35 (band FLOOR)
#   wait_rate_ceiling             min  threshold 0.50   — new <= 0.50 (band CEILING)
#   serve_violations              min  threshold 0      — never_reserve violations 0
#   invariants.violations         min  regression-vs-prev — never regress vs prev
#
# The wait_rate BAND [0.35, 0.50] is encoded as TWO threshold rules on the same
# metric value (a min-direction floor and a max-direction ceiling), since a single
# rule cannot express "inside a band".  ``wait_rate_ceiling`` is a SEPARATE metric
# key carrying the SAME pooled wait_rate value (see ``_aggregate``) so both rules
# resolve a real number.
TUTOR_GATE_RULES: list[GateRule] = [
    # Invariants must be zero — the hard safety/pedagogy floor.
    GateRule("invariants.violations", "min", threshold=0),
    # wait_rate band floor: new must be >= 0.35 (direction=max means new >= ref).
    GateRule("wait_rate", "max", threshold=0.35),
    # wait_rate band ceiling: new must be <= 0.50 (direction=min means new <= ref).
    GateRule("wait_rate_ceiling", "min", threshold=0.50),
    # never_reserve violations must be zero.
    GateRule("serve_violations", "min", threshold=0),
    # Regression vs prev: the candidate must not introduce MORE invariant
    # violations than the baseline (smaller-is-better, regression-vs-prev).
    GateRule("invariants.violations", "min", threshold=None),
]


# ---------------------------------------------------------------------------
# Holdout refusal (the freeze stays auditable)
# ---------------------------------------------------------------------------

def _assert_not_holdout(path: Path) -> None:
    """Refuse a holdout path — tonight's deterministic A/B is DEV-ONLY.

    The frozen holdout half may ONLY be evaluated by the LATER judged adjudication
    (Prediction #5, after judge validation).  Reading it here would burn the audit
    anchor, so the runner fails loud if handed it.
    """
    resolved = path.resolve()
    if resolved == _HOLDOUT_FILE.resolve() or "holdout" in resolved.name.lower():
        raise ValueError(
            f"REFUSING to read a holdout path ({path}). Tonight's A/B is the "
            f"DETERMINISTIC DEV-split comparison only. The frozen holdout is "
            f"adjudicated LATER (judged, after validation — docs/predictions.md #5). "
            f"The freeze must stay auditable."
        )


# ---------------------------------------------------------------------------
# Replay + aggregation
# ---------------------------------------------------------------------------

def _load_dev_sessions(dev_path: Path) -> list[dict]:
    """Load the frozen dev split as a list of session-item dicts."""
    _assert_not_holdout(dev_path)
    items = [
        json.loads(line)
        for line in dev_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return items


def replay_version(
    version: str,
    sessions: list[dict],
    *,
    curriculum=None,
) -> list[SessionTrace]:
    """Replay every session under ``version`` -> list of SessionTraces.

    v2 reuses ONE store per persona-child, so memory persists across that
    persona's sessions (cross-session learning).  v1/v3 are memory-free.
    """
    traces: list[SessionTrace] = []
    v2_stores: dict[str, InMemoryLearnerStore] = {}
    for item in sessions:
        store = None
        if version == "v2":
            store = v2_stores.setdefault(item["persona_id"], InMemoryLearnerStore())
        traces.append(
            run_session(item, version, curriculum=curriculum, store=store)
        )
    return traces


def _aggregate(traces: list[SessionTrace]) -> dict:
    """Aggregate per-session traces into the A/B metric dict.

    Metrics
    -------
    invariants.violations : sum of severity-ERROR findings across all traces
                            (the real policy-compiler audit on each trace).
    wait_rate             : pooled WAIT-rate over DECISION turns (miscue turns +
                            page-end turns; see tutor_versions.is_decision_turn) —
                            the denominator the [0.35, 0.50] band is calibrated on.
    wait_rate_ceiling     : the SAME pooled value, carried under a second key so
                            the band-ceiling gate rule resolves a number.
    serve_violations      : never_reserves_completed_item findings (counted in the
                            invariants total too, surfaced separately for its own
                            gate rule).
    targeted_next_items   : count of NEXT_ITEM turns carrying a planner-selected
                            skill_id — v2's mastery-conditioning value-add.  v1 is
                            state-blind (empty learner state -> no gaps -> never
                            NEXT_ITEM), so this is 0 for v1 and positive for v2; it
                            is the behavioural signal of the improvement curve.
    n_sessions / n_turns  : corpus sizes; n_decision_turns is the wait_rate denom.
    """
    pedagogy_checks = _pedagogy_checks()
    serve_checks = _serve_check()
    total_violations = 0
    serve_violations = 0
    n_turns = 0
    n_decision = 0
    n_wait = 0
    targeted_next_items = 0
    for trace in traces:
        total_violations += audit(trace, pedagogy_checks).violations
        serve_violations += audit(trace, serve_checks).violations
        for turn in trace.turns:
            n_turns += 1
            if turn.action_move == "NEXT_ITEM" and turn.skill_id is not None:
                targeted_next_items += 1
            if is_decision_turn(turn):
                n_decision += 1
                if turn.action_move == "WAIT":
                    n_wait += 1

    wait_rate = (n_wait / n_decision) if n_decision else 0.0
    return {
        "invariants": {"violations": total_violations},
        "wait_rate": wait_rate,
        "wait_rate_ceiling": wait_rate,
        "serve_violations": serve_violations,
        "targeted_next_items": targeted_next_items,
        "n_sessions": len(traces),
        "n_turns": n_turns,
        "n_decision_turns": n_decision,
    }


def _to_report(version: str, metrics: dict) -> EvalReport:
    """Write the per-version report via evaluate() and return it.

    evaluate() is immutable-by-content: a re-run with identical metrics is a
    no-op; a differing re-run raises (investigate-don't-rebaseline).
    """
    return evaluate(
        f"tutor-{version}",
        str(_DEV_FILE),
        metrics=metrics,
        results_dir=str(_RESULTS_DIR),
    )


# ---------------------------------------------------------------------------
# Improvement curve (v1 -> v2 per-metric deltas)
# ---------------------------------------------------------------------------

def _improvement_curve(v1_metrics: dict, v2_metrics: dict) -> dict:
    """Per-metric v1->v2 deltas (the improvement curve series)."""
    return {
        "invariants.violations": {
            "v1": v1_metrics["invariants"]["violations"],
            "v2": v2_metrics["invariants"]["violations"],
            "delta": v2_metrics["invariants"]["violations"]
            - v1_metrics["invariants"]["violations"],
        },
        "wait_rate": {
            "v1": v1_metrics["wait_rate"],
            "v2": v2_metrics["wait_rate"],
            "delta": v2_metrics["wait_rate"] - v1_metrics["wait_rate"],
        },
        "targeted_next_items": {
            "v1": v1_metrics["targeted_next_items"],
            "v2": v2_metrics["targeted_next_items"],
            "delta": v2_metrics["targeted_next_items"]
            - v1_metrics["targeted_next_items"],
        },
        "serve_violations": {
            "v1": v1_metrics["serve_violations"],
            "v2": v2_metrics["serve_violations"],
            "delta": v2_metrics["serve_violations"] - v1_metrics["serve_violations"],
        },
        "n_turns": {
            "v1": v1_metrics["n_turns"],
            "v2": v2_metrics["n_turns"],
            "delta": v2_metrics["n_turns"] - v1_metrics["n_turns"],
        },
    }


# ---------------------------------------------------------------------------
# promote_failure — v3's violating turns -> tutor_failures golden
# ---------------------------------------------------------------------------

def _v3_violation_traces(
    sessions: list[dict], v3_traces: list[SessionTrace]
) -> list[dict]:
    """Build one promotable trace dict per v3 violation turn (stable, unique ids).

    A violation turn is a self_correction turn that v3 met with a corrective move
    (the ``never_corrects_self_correction`` breach).  Each becomes a small trace
    dict with a STABLE trace_id derived from the SESSION id + turn index (the
    session id makes it unique across the 49 sessions — child_id alone collides
    because there are only 3 personas), so re-promotion is idempotent.
    """
    out: list[dict] = []
    for item, trace in zip(sessions, v3_traces, strict=True):
        session_id = item["id"]
        for turn in trace.turns:
            if (
                turn.miscue_type == "self_correction"
                and turn.action_move in ("MODEL_THE_WORD", "SCAFFOLDED_HINT")
            ):
                trace_id = f"v3__{session_id}__turn{turn.turn_index}"
                out.append(
                    {
                        "trace_id": trace_id,
                        "version": "v3",
                        "session_id": session_id,
                        "child_id": trace.child_id,
                        "turn_index": turn.turn_index,
                        "miscue_type": turn.miscue_type,
                        "action_move": turn.action_move,
                        "rule_violated": "never_corrects_self_correction",
                        "note": (
                            "DEMO-villain v3 modeled a self-correction mid-page; "
                            "promoted as a regression-guard golden."
                        ),
                    }
                )
    # Stable order so the golden grows deterministically.
    out.sort(key=lambda t: t["trace_id"])
    return out


def _promote_batch(traces_dicts: list[dict], golden_path: Path) -> int:
    """Promote each trace dict; return the golden line count AFTER the batch."""
    for td in traces_dicts:
        promote_failure(td, str(golden_path))
    return _golden_count(golden_path)


def _golden_count(golden_path: Path) -> int:
    if not golden_path.exists():
        return 0
    return sum(
        1 for line in golden_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


# ---------------------------------------------------------------------------
# Sensitivity — ±30% persona-rate perturbation, rerun v1/v2 comparison
# ---------------------------------------------------------------------------

def _perturbed_sessions(scale: float, label: str) -> list[dict]:
    """Regenerate ~49 sessions with every persona rate scaled by ``scale``.

    Seeded, out-of-band ids (``<id>__sens-<label>``), NEVER touches the frozen
    files.  One session per eligible (persona, passage) pair so the corpus size
    tracks the dev split's order of magnitude.
    """
    personas = load_personas(_PERSONAS_DIR)
    passages = load_passages(_PASSAGES_DIR)

    perturbed_personas = [
        dataclasses.replace(
            p, rates={cls: rate * scale for cls, rate in p.rates.items()}
        )
        for p in personas
    ]

    items: list[dict] = []
    for persona in sorted(perturbed_personas, key=lambda p: p.id):
        eligible = [pg for pg in passages if pg.band <= persona.band_ceiling]
        for passage in eligible:
            rng = _child_rng(_SENSITIVITY_SEED, persona.id, passage.id, 0)
            item = generate_session(persona, passage, rng, session_index=0)
            d = session_item_to_dict(item)
            d["id"] = f"{d['id']}__sens-{label}"  # out-of-band id
            items.append(d)
    return items


def _sensitivity_run(curriculum) -> dict:
    """Rerun the v1/v2 deterministic comparison on ±30% perturbed corpora.

    Returns the per-perturbation gate outcomes and whether the v1/v2 conclusion
    (gate PASSES) HOLDS across both perturbations.
    """
    results: dict = {}
    holds = True
    for label, scale in (("minus30", 1.0 - _SENSITIVITY_DELTA), ("plus30", 1.0 + _SENSITIVITY_DELTA)):
        sessions = _perturbed_sessions(scale, label)
        v1_metrics = _aggregate(replay_version("v1", sessions))
        v2_metrics = _aggregate(replay_version("v2", sessions, curriculum=curriculum))
        v1_report = EvalReport(version="sens-v1", metrics=v1_metrics, metadata={})
        v2_report = EvalReport(version="sens-v2", metrics=v2_metrics, metadata={})
        gate = compare(v1_report, v2_report, TUTOR_GATE_RULES)
        passed = gate.passed
        holds = holds and passed
        results[label] = {
            "scale": scale,
            "n_sessions": v1_metrics["n_sessions"],
            "v1_wait_rate": v1_metrics["wait_rate"],
            "v2_wait_rate": v2_metrics["wait_rate"],
            "v1_violations": v1_metrics["invariants"]["violations"],
            "v2_violations": v2_metrics["invariants"]["violations"],
            "gate_passed": passed,
            "gate_exit_code": gate.exit_code,
            "breaches": gate.breaches,
        }
    return {
        "perturbation": f"±{int(_SENSITIVITY_DELTA * 100)}% persona-rate",
        "conclusion_holds": holds,
        "note": (
            "Slip-cut allowed per plan: ±30% only. Out-of-band seeded sessions, "
            "frozen files untouched."
        ),
        "runs": results,
    }


# ---------------------------------------------------------------------------
# Growth chart
# ---------------------------------------------------------------------------

def _write_growth_chart(growth: list[dict], out_png: Path) -> None:
    """Cumulative golden-set size per promotion batch (v1, v2, v3)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [g["batch"] for g in growth]
    cumulative = [g["cumulative_golden_size"] for g in growth]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, cumulative, color=["#9ecae1", "#6baed6", "#d6604d"])
    ax.set_ylabel("cumulative tutor_failures golden size")
    ax.set_xlabel("promotion batch")
    ax.set_title("Eval flywheel — failures promoted into the golden set per batch")
    for bar, val in zip(bars, cumulative, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(val),
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0, max(cumulative) + 2 if cumulative else 1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T5.4 deterministic A/B + flywheel + sensitivity")
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip the ±30% sensitivity analysis (faster smoke run).",
    )
    args = parser.parse_args(argv)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curriculum = load_curriculum(_CURRICULUM_PATH)

    sessions = _load_dev_sessions(_DEV_FILE)
    print(f"Loaded {len(sessions)} DEV sessions from {_DEV_FILE.name} (holdout NOT read).\n")

    # --- Replay all three versions -----------------------------------------
    v1_traces = replay_version("v1", sessions)
    v2_traces = replay_version("v2", sessions, curriculum=curriculum)
    v3_traces = replay_version("v3", sessions)

    v1_metrics = _aggregate(v1_traces)
    v2_metrics = _aggregate(v2_traces)
    v3_metrics = _aggregate(v3_traces)

    # --- Immutable per-version reports -------------------------------------
    v1_report = _to_report("v1", v1_metrics)
    v2_report = _to_report("v2", v2_metrics)
    v3_report = _to_report("v3", v3_metrics)

    # --- Gate comparisons ---------------------------------------------------
    gate_v1_v2 = compare(v1_report, v2_report, TUTOR_GATE_RULES)
    gate_v2_v3 = compare(v2_report, v3_report, TUTOR_GATE_RULES)

    # --- Improvement curve --------------------------------------------------
    curve = _improvement_curve(v1_metrics, v2_metrics)

    # --- promote_failure growth chart --------------------------------------
    # Batches: v1 (clean -> nothing to promote), v2 (clean), v3 (villain breaches).
    growth: list[dict] = []
    # v1 batch: no violations to promote.
    growth.append(
        {"batch": "v1", "promoted": 0, "cumulative_golden_size": _golden_count(_TUTOR_FAILURES)}
    )
    # v2 batch: no violations to promote.
    growth.append(
        {"batch": "v2", "promoted": 0, "cumulative_golden_size": _golden_count(_TUTOR_FAILURES)}
    )
    # v3 batch: promote every villain violation turn.
    v3_violation_dicts = _v3_violation_traces(sessions, v3_traces)
    after_v3 = _promote_batch(v3_violation_dicts, _TUTOR_FAILURES)
    growth.append(
        {
            "batch": "v3",
            "promoted": len(v3_violation_dicts),
            "cumulative_golden_size": after_v3,
        }
    )
    # Idempotency demo: re-promote the SAME v3 batch -> golden unchanged.
    after_v3_again = _promote_batch(v3_violation_dicts, _TUTOR_FAILURES)
    promote_idempotent = after_v3_again == after_v3

    _write_growth_chart(growth, _GROWTH_PNG)

    # --- Sensitivity (LAST) -------------------------------------------------
    if args.skip_sensitivity:
        sensitivity = {"skipped": True}
    else:
        sensitivity = _sensitivity_run(curriculum)

    # --- Assemble + write ab_dev.json --------------------------------------
    receipt_2 = {
        "claim": "the gate BLOCKS the deliberately-worse tutor (v3 always-intervene)",
        "comparison": "v2 (prev) vs v3 (new)",
        "gate_passed": gate_v2_v3.passed,
        "exit_code": gate_v2_v3.exit_code,
        "breaches": gate_v2_v3.breaches,
        "evidence": (
            f"v3 produced {v3_metrics['invariants']['violations']} invariant "
            f"violations (never_corrects_self_correction: modeled self-corrections "
            f"mid-page); v2 produced {v2_metrics['invariants']['violations']}. "
            f"The gate's invariants.violations rule (min/0) is breached -> exit 1."
        ),
    }

    payload = {
        "ticket": "T5.4",
        "what_this_is": (
            "DETERMINISTIC dev-split A/B (v1 state-blind vs v2 mastery-conditioned), "
            "deliberately-worse v3 blocked by the gate, promote_failure flywheel, and "
            "±30% sensitivity."
        ),
        "pre_registration": {
            "adjudicates_a_prediction": False,
            "note": (
                "docs/predictions.md #5 covers JUDGED guidance/actionability on the "
                "HELD-OUT split, adjudicated LATER after judge validation. This "
                "artifact REPORTS deterministic diffs only; it adjudicates NO "
                "prediction. Judged scoring is DEFERRED (judge dims not yet "
                "human-validated)."
            ),
        },
        "split": "dev",
        "dev_file": _DEV_FILE.name,
        "holdout_read": False,
        "n_sessions": len(sessions),
        "date": date.today().isoformat(),
        "tutor_gate_rules": [
            {
                "metric": r.metric,
                "direction": r.direction,
                "threshold": r.threshold,
                "report_only": r.report_only,
            }
            for r in TUTOR_GATE_RULES
        ],
        "metrics": {"v1": v1_metrics, "v2": v2_metrics, "v3": v3_metrics},
        "comparison_table": _comparison_table(v1_metrics, v2_metrics, v3_metrics),
        "gate_outcomes": {
            "v1_vs_v2": {
                "passed": gate_v1_v2.passed,
                "exit_code": gate_v1_v2.exit_code,
                "breaches": gate_v1_v2.breaches,
            },
            "v2_vs_v3": {
                "passed": gate_v2_v3.passed,
                "exit_code": gate_v2_v3.exit_code,
                "breaches": gate_v2_v3.breaches,
            },
        },
        "receipt_2_gate_blocks_bad_tutor": receipt_2,
        "improvement_curve_v1_to_v2": curve,
        "promote_growth": {
            "golden_file": _TUTOR_FAILURES.name,
            "batches": growth,
            "idempotent_on_rerun": promote_idempotent,
            "chart": _GROWTH_PNG.name,
        },
        "sensitivity": sensitivity,
    }

    _AB_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _print_summary(payload)

    # Exit code reflects the PRIMARY A/B claim: v2 must pass the gate vs v1.
    # (v3's expected exit-1 is captured as receipt #2 in the JSON, not the process
    # exit — a CI green here means "v2 improved and the gate accepts it".)
    if not gate_v1_v2.passed:
        print("\nFAIL: v2 did not pass the tutor gate vs v1.", file=sys.stderr)
        return 1
    if not promote_idempotent:
        print("\nFAIL: promote_failure was not idempotent on rerun.", file=sys.stderr)
        return 1
    if not args.skip_sensitivity and not sensitivity.get("conclusion_holds", False):
        print("\nFAIL: sensitivity conclusion did not hold under ±30%.", file=sys.stderr)
        return 1
    return 0


def _comparison_table(v1: dict, v2: dict, v3: dict) -> list[dict]:
    """A row-per-metric comparison table across the three versions."""
    rows = []
    for key, get in (
        ("invariants.violations", lambda m: m["invariants"]["violations"]),
        ("wait_rate", lambda m: round(m["wait_rate"], 4)),
        ("targeted_next_items", lambda m: m["targeted_next_items"]),
        ("serve_violations", lambda m: m["serve_violations"]),
        ("n_sessions", lambda m: m["n_sessions"]),
        ("n_turns", lambda m: m["n_turns"]),
    ):
        rows.append({"metric": key, "v1": get(v1), "v2": get(v2), "v3": get(v3)})
    return rows


def _print_summary(payload: dict) -> None:
    print("=" * 70)
    print("T5.4  DETERMINISTIC A/B — dev split (judged scoring DEFERRED)")
    print("=" * 70)
    print(f"sessions: {payload['n_sessions']}   holdout_read: {payload['holdout_read']}\n")

    print(f"{'metric':<26}{'v1':>10}{'v2':>10}{'v3':>10}")
    print("-" * 56)
    for row in payload["comparison_table"]:
        print(f"{row['metric']:<26}{str(row['v1']):>10}{str(row['v2']):>10}{str(row['v3']):>10}")

    g = payload["gate_outcomes"]
    print("\nGate outcomes (TUTOR_GATE_RULES):")
    print(f"  v1 -> v2 : exit {g['v1_vs_v2']['exit_code']}  "
          f"{'PASS' if g['v1_vs_v2']['passed'] else 'BLOCKED'}")
    print(f"  v2 -> v3 : exit {g['v2_vs_v3']['exit_code']}  "
          f"{'PASS' if g['v2_vs_v3']['passed'] else 'BLOCKED (receipt #2)'}")
    for b in g["v2_vs_v3"]["breaches"]:
        print(f"            breach: {b}")

    print("\nImprovement curve (v1 -> v2):")
    for metric, d in payload["improvement_curve_v1_to_v2"].items():
        print(f"  {metric:<24} v1={d['v1']}  v2={d['v2']}  delta={d['delta']}")

    pg = payload["promote_growth"]
    print(f"\npromote_failure growth ({pg['golden_file']}):")
    for batch in pg["batches"]:
        print(f"  batch {batch['batch']:<3} promoted={batch['promoted']:<3} "
              f"cumulative={batch['cumulative_golden_size']}")
    print(f"  idempotent on rerun: {pg['idempotent_on_rerun']}")
    print(f"  chart: {pg['chart']}")

    sens = payload["sensitivity"]
    if sens.get("skipped"):
        print("\nSensitivity: SKIPPED (--skip-sensitivity)")
    else:
        print(f"\nSensitivity ({sens['perturbation']}): "
              f"conclusion_holds={sens['conclusion_holds']}")
        for label, run in sens["runs"].items():
            print(f"  {label:<8} scale={run['scale']:.2f}  "
                  f"v1_wait={run['v1_wait_rate']:.3f}  v2_wait={run['v2_wait_rate']:.3f}  "
                  f"gate={'PASS' if run['gate_passed'] else 'BLOCKED'}")

    print(f"\nWrote {_AB_JSON.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
