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

from readcoach.naive_tutor import NaiveTutor, NaiveCliTransport, StubTransport, naive_cli_transport  # noqa: E402
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
        "--transport",
        choices=["stub", "claude-cli", "api"],
        default=None,
        help=(
            "Transport to use.  'stub' = deterministic stub (same as --stub); "
            "'claude-cli' = live model via subscription CLI (no API key needed); "
            "'api' = live model via Anthropic SDK (requires ANTHROPIC_API_KEY).  "
            "Defaults to stub if --stub is set, api otherwise."
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

    # Resolve effective transport: --stub / --transport stub → stub;
    # --transport claude-cli → NaiveCliTransport;
    # default (no flags) or --transport api → SDK (requires ANTHROPIC_API_KEY).
    effective_transport = args.transport
    if args.stub and effective_transport is None:
        effective_transport = "stub"
    if effective_transport is None:
        effective_transport = "api"

    if effective_transport == "stub":
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
        out_filename = "naive_stub_audit.json"
        transport_label = "stub"

    elif effective_transport == "claude-cli":
        print("=" * 80)
        print("CLAUDE-CLI TRANSPORT — live model via subscription CLI (claude -p)")
        print(f"  model : {NaiveCliTransport.transport_meta['model']}")
        print("  prompt: unconstrained naive system prompt + format instruction only")
        print("  key   : no ANTHROPIC_API_KEY required (subscription binary)")
        print("=" * 80)
        cli = naive_cli_transport()
        tutor = NaiveTutor(client_factory=lambda: cli)
        policy_version = _NAIVE_LIVE_VERSION
        out_filename = "naive_live_audit.json"
        transport_label = "claude-cli"

    else:  # api
        print("=" * 80)
        print("LIVE TRANSPORT — using real Anthropic API (SDK)")
        print("=" * 80)
        # This will raise RuntimeError loud if key is absent.
        tutor = NaiveTutor()
        tutor._client_or_build()  # noqa: SLF001 — eager key check
        policy_version = _NAIVE_LIVE_VERSION
        out_filename = "naive_live_audit.json"
        transport_label = "api"

    traces = run_with_tutor(tutor, policy_version=policy_version)
    checks = compile_rules(load_policies(_POLICIES_DIR))

    # Include transport metadata so the artifact is self-describing.
    model_meta = (
        NaiveCliTransport.transport_meta["model"]
        if effective_transport == "claude-cli"
        else None
    )
    audit_output: dict = {
        "transport": transport_label,
        "model": model_meta,
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

        # Collect sample utterances per rule for evidence (up to 2 per rule).
        samples_by_rule: dict[str, list[str]] = {}
        for f in report.findings:
            if f.severity == "error" and len(samples_by_rule.get(f.rule_id, [])) < 2:
                turn = next((t for t in trace.turns if t.turn_index == f.turn_index), None)
                if turn and turn.utterance:
                    samples_by_rule.setdefault(f.rule_id, []).append(turn.utterance[:120])

        audit_output["profiles"][trace.child_id] = {
            "violations": report.violations,
            "by_rule": rule_counts,
            "samples": samples_by_rule,
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

    # Write audit JSON — NEVER overwrite the stub artifact with a live result.
    if args.write_audit:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_filename
        out_path.write_text(json.dumps(audit_output, indent=2, sort_keys=True))
        print(f"\nAudit written to: {out_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
