"""Naive-tutor replay — the unconstrained villain audited (T5.0).

Runs the SAME 3 scripted reader profiles (from generate_turns.py) through
``NaiveTutor``, produces a ``SessionTrace`` per profile, audits each with the
policy compiler, and prints the action-stream rendering:

    turn idx | page pos | event | tutor said | flags

The rendering makes the villain's violations visually obvious next to the
compliant policy-harness output.

Usage
-----
Key-free demo (committed stub transport):
    uv run python scripts/naive_replay.py --stub

Live model (requires ANTHROPIC_API_KEY):
    uv run python scripts/naive_replay.py

Exit codes
----------
0   — successful completion (violations are REPORTED, not an error; the villain
      is EXPECTED to violate rules; violations are the point of this demo)
1   — unexpected runtime error (key missing in live mode, import failure, etc.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from readcoach.naive_tutor import NaiveTutor, StubTransport  # noqa: E402
from readcoach.policy_compiler import audit, compile_rules, load_policies  # noqa: E402
from readcoach.trace import SessionTrace, TurnRecord  # noqa: E402

# Re-use the scripted profiles from generate_turns (shared scripting).
# We import only build_profiles and Step — no verbalizer dependency.
import importlib.util as _ilu  # noqa: E402

_gen_spec = _ilu.spec_from_file_location(
    "generate_turns", _PROJECT_ROOT / "scripts" / "generate_turns.py"
)
_gen_mod = _ilu.module_from_spec(_gen_spec)
sys.modules["generate_turns"] = _gen_mod
_gen_spec.loader.exec_module(_gen_mod)  # type: ignore[union-attr]

build_profiles = _gen_mod.build_profiles
Step = _gen_mod.Step
Profile = _gen_mod.Profile

_POLICIES_DIR = _PROJECT_ROOT / "policies"
_DEFAULT_OUT = _PROJECT_ROOT / "evals" / "results"

# Policy version label for naive-tutor traces.
_NAIVE_POLICY_VERSION = "naive-stub"
_NAIVE_LIVE_VERSION = "naive-live"


# ---------------------------------------------------------------------------
# Core: build traces from a NaiveTutor over the scripted profiles
# ---------------------------------------------------------------------------

def run_with_tutor(tutor: NaiveTutor, *, policy_version: str) -> list[SessionTrace]:
    """Run all 3 scripted profiles through ``tutor``; return one trace per profile."""
    profiles = build_profiles()
    traces: list[SessionTrace] = []

    for profile in profiles:
        records: list[TurnRecord] = []
        for i, step in enumerate(profile.steps):
            record = tutor.react(
                turn_index=i,
                at_page_end=step.at_page_end,
                miscue_type=step.miscue.type if step.miscue else None,
                target_word=step.miscue.target_word if step.miscue else None,
                # Naive tutor never sets AI reminders — that is one of its
                # documented violations (periodic_ai_reminder fires).
                is_ai_reminder=False,
            )
            records.append(record)

        trace = SessionTrace(
            child_id=profile.name,
            policy_version=policy_version,
            completed_skills_at_start=profile.completed_skills_at_start,
            turns=tuple(records),
        )
        traces.append(trace)

    return traces


def run_stub() -> list[SessionTrace]:
    """Public entry point for tests: run with the stub transport."""
    stub = StubTransport()
    tutor = NaiveTutor(client_factory=lambda: stub)
    return run_with_tutor(tutor, policy_version=_NAIVE_POLICY_VERSION)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_stream(trace: SessionTrace, findings_by_turn: dict[int, list[str]]) -> list[str]:
    """Chronological action-stream table rows for one profile."""
    rows: list[str] = []
    for t in trace.turns:
        page_pos = "PAGE-END" if t.at_page_end else "mid-page"
        event = t.miscue_type or "clean"
        said = t.utterance or "(silent)"
        # Truncate long utterances for display.
        if len(said) > 70:
            said = said[:67] + "..."
        flags = ", ".join(findings_by_turn.get(t.turn_index, []))
        rows.append(
            f"  {t.turn_index:>3} | {page_pos:<8} | {event:<16} | {said:<70} | {flags}"
        )
    return rows


def _audit_traces(traces: list[SessionTrace]) -> dict:
    """Audit all traces; return per-profile violation summary dict."""
    checks = compile_rules(load_policies(_POLICIES_DIR))
    summary: dict = {}
    for trace in traces:
        report = audit(trace, checks)
        # Per-rule violation counts.
        rule_counts: dict[str, int] = {}
        for f in report.findings:
            if f.severity == "error":
                rule_counts[f.rule_id] = rule_counts.get(f.rule_id, 0) + 1
        summary[trace.child_id] = {
            "violations": report.violations,
            "by_rule": rule_counts,
        }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run naive-tutor replay (the unconstrained villain, audited)."
    )
    p.add_argument(
        "--stub",
        action="store_true",
        help=(
            "Use the documented stub transport (deterministic canned responses; "
            "no ANTHROPIC_API_KEY required).  The stub simulates the worst-case "
            "unconstrained helpful-assistant behavior."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=str(_DEFAULT_OUT),
        help="Directory for output artifacts (default: evals/results).",
    )
    p.add_argument(
        "--write-audit",
        action="store_true",
        default=True,
        help="Write the audit JSON to --out-dir (default: true).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.stub:
        print("=" * 80)
        print("STUB TRANSPORT — live model run pending")
        print(
            "Canned responses model the documented worst-case unconstrained "
            "helpful assistant:\n"
            "  * immediately supplies the correct word on any miscue incl. "
            "self-corrections\n"
            "  * uses 'wrong word, the word is X' corrective phrasing mid-page\n"
            "  * never emits AI-identity reminders"
        )
        print("=" * 80)
        stub = StubTransport()
        tutor = NaiveTutor(client_factory=lambda: stub)
        policy_version = _NAIVE_POLICY_VERSION
    else:
        print("=" * 80)
        print("LIVE TRANSPORT — using real Anthropic API")
        print("=" * 80)
        # This will raise RuntimeError loud if key is absent.
        tutor = NaiveTutor()
        tutor._client_or_build()  # noqa: SLF001 — eager key check
        policy_version = _NAIVE_LIVE_VERSION

    traces = run_with_tutor(tutor, policy_version=policy_version)
    checks = compile_rules(load_policies(_POLICIES_DIR))

    audit_output: dict = {
        "transport": "stub" if args.stub else "live",
        "profiles": {},
    }

    for trace in traces:
        report = audit(trace, checks)

        # Build per-turn flag map for rendering.
        findings_by_turn: dict[int, list[str]] = {}
        for f in report.findings:
            if f.severity == "error":
                findings_by_turn.setdefault(f.turn_index, []).append(f.rule_id)

        # Per-rule violation counts.
        rule_counts: dict[str, int] = {}
        for f in report.findings:
            if f.severity == "error":
                rule_counts[f.rule_id] = rule_counts.get(f.rule_id, 0) + 1

        audit_output["profiles"][trace.child_id] = {
            "violations": report.violations,
            "by_rule": rule_counts,
        }

        # Print action-stream rendering.
        print(f"\n{'─' * 80}")
        print(f"Profile: {trace.child_id}  |  turns: {len(trace.turns)}  |  violations: {report.violations}")
        print(f"{'─' * 80}")
        header = f"  {'idx':>3} | {'pos':<8} | {'event':<16} | {'tutor said':<70} | flags"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row in _render_stream(trace, findings_by_turn):
            print(row)

        print("\n  Violations by rule:")
        if rule_counts:
            for rule_id, count in sorted(rule_counts.items()):
                print(f"    {rule_id}: {count}")
        else:
            print("    (none)")

    # Overall summary.
    total = sum(v["violations"] for v in audit_output["profiles"].values())
    print(f"\n{'=' * 80}")
    print(f"TOTAL VIOLATIONS (all profiles): {total}")
    print(f"{'=' * 80}")

    # Write audit JSON.
    if args.write_audit:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "naive_stub_audit.json"
        out_path.write_text(json.dumps(audit_output, indent=2, sort_keys=True))
        print(f"\nAudit written to: {out_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
