"""CI gate — compare a new eval report against the committed baseline.

Usage
-----
    uv run python scripts/gate.py [PREV_REPORT] NEW_REPORT

    PREV_REPORT  path to the baseline report (default: evals/results/v0.json)
    NEW_REPORT   path to the candidate report (required)

Exit codes
----------
    0   all gating rules pass
    1   at least one gating regression / threshold breach
    2   invalid condition (missing metric path, None value, non-numeric, etc.)

Rule table
----------
The rule table is committed here (not in a config file) so any rule change is
a tracked code change that appears in pull-request diffs.

Metric paths use dot-notation into the EvalReport.metrics dict:
  "miscue.substitution.f1"   → metrics["miscue"]["substitution"]["f1"]
  "invariants.violations"    → metrics["invariants"]["violations"]

Gate rules (direction=max → larger-is-better; direction=min → smaller-is-better):

  Gating rules (breach → exit 1):
    miscue.substitution.f1         max  regression-vs-prev  — recall/precision balance
    miscue.omission.f1             max  regression-vs-prev
    miscue.insertion.f1            max  regression-vs-prev
    miscue.self_correction.f1      max  regression-vs-prev
    miscue.hesitation.f1           max  regression-vs-prev
    miscue.fp_per_100_correct_words  min  regression-vs-prev  — FP rate
    invariants.violations          min  threshold=0           — must be zero always

  Report-only rules (breach → listed but no exit-code effect):
    latency.decision_ms_p95        min  regression-vs-prev  — skipped when either
                                                              report has None value

Latency rule structural skip
----------------------------
If either report has latency.decision_ms_p95 = None (not yet measured), we
print an explicit notice and skip the rule.  This is an explicitly-logged
structural skip, not a silent one.  The rule is report_only=True in any case,
so even when both values are present a breach only prints, never gates.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `evals` is importable when
# this script is invoked directly (e.g. `uv run python scripts/gate.py`).
# ---------------------------------------------------------------------------
_SCRIPT_PROJECT_ROOT = Path(__file__).parent.parent
if str(_SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_PREV = _PROJECT_ROOT / "evals" / "results" / "v0.json"


# ---------------------------------------------------------------------------
# Gate rule table (committed in code — changes appear in PR diffs)
# ---------------------------------------------------------------------------

# Import GateRule here; evals/ is on sys.path when run via `uv run`.
from evals.harness import GateRule, compare, load_report  # noqa: E402


# Gating rules — breach causes exit 1.
# threshold=None means "regression vs prev" (new must not be worse than prev).
GATE_RULES: list[GateRule] = [
    # F1 metrics: higher is better (max), regression-vs-prev
    GateRule("miscue.substitution.f1",        "max", threshold=None),
    GateRule("miscue.omission.f1",            "max", threshold=None),
    GateRule("miscue.insertion.f1",           "max", threshold=None),
    GateRule("miscue.self_correction.f1",     "max", threshold=None),
    GateRule("miscue.hesitation.f1",          "max", threshold=None),
    # FP rate: lower is better (min), regression-vs-prev
    GateRule("miscue.fp_per_100_correct_words", "min", threshold=None),
    # Invariants: must always be zero (hard threshold, not regression-vs-prev)
    GateRule("invariants.violations",         "min", threshold=0),
    # Latency: report-only, skipped when values are None (see main logic below)
    GateRule("latency.decision_ms_p95",       "min", threshold=None, report_only=True),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_report(path: Path):
    """Load an EvalReport from a JSON file.  Exits loudly on missing file."""
    if not path.exists():
        print(f"ERROR: report not found: {path}", file=sys.stderr)
        sys.exit(2)
    return load_report(path)


def _resolve_metric_or_none(metrics: dict, dotted_path: str):
    """Walk a nested dict by dotted path.  Returns the value or None on missing."""
    keys = dotted_path.split(".")
    node = metrics
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = (argv if argv is not None else sys.argv[1:])

    if len(args) == 1:
        prev_path = _DEFAULT_PREV
        new_path = Path(args[0])
    elif len(args) == 2:
        prev_path = Path(args[0])
        new_path = Path(args[1])
    else:
        print(
            "Usage: gate.py [PREV_REPORT] NEW_REPORT\n"
            f"  PREV_REPORT  (default: {_DEFAULT_PREV})\n"
            "  NEW_REPORT   path to candidate report",
            file=sys.stderr,
        )
        sys.exit(2)

    prev = _load_report(prev_path)
    new = _load_report(new_path)

    print(f"gate: prev={prev_path}  new={new_path}")
    print(f"      prev.version={prev.version!r}  new.version={new.version!r}")

    # Handle latency rule structural skip: if either report has None for
    # latency.decision_ms_p95, skip the rule with an explicit notice.
    active_rules: list[GateRule] = []
    for rule in GATE_RULES:
        if rule.metric == "latency.decision_ms_p95":
            prev_lat = _resolve_metric_or_none(prev.metrics, rule.metric)
            new_lat = _resolve_metric_or_none(new.metrics, rule.metric)
            if prev_lat is None or new_lat is None:
                print(
                    "latency: not yet measured, rule skipped "
                    "(report-only anyway; will activate once both reports carry values)"
                )
                continue
        active_rules.append(rule)

    result = compare(prev, new, active_rules)

    if result.report_only_breaches:
        print("\n--- report-only breaches (non-gating) ---")
        for msg in result.report_only_breaches:
            print(f"  {msg}")

    if result.breaches:
        print("\n--- gating breaches ---")
        for msg in result.breaches:
            print(f"  {msg}")
    else:
        print("\ngate: all gating rules passed")

    if result.exit_code == 0:
        print(f"gate: exit 0 (passed)")
    elif result.exit_code == 1:
        print(f"gate: exit 1 (regression)")
    else:
        print(f"gate: exit 2 (invalid)")

    sys.exit(result.exit_code)


if __name__ == "__main__":
    main()
