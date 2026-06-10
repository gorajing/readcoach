"""Invariants metric — the gated bridge between a trace and the eval report (T4.3).

``scripts/gate.py`` already gates ``invariants.violations`` at min/0.  This
module makes a report carry the REAL computed value: load the committed policies,
compile them, audit a ``SessionTrace``, and surface the violation count in the
nested-dict shape ``evaluate(... metrics=...)`` expects.

The default policies directory is the repo's ``policies/`` so callers get the
project's real invariants; tests may pass an explicit dir.
"""
from __future__ import annotations

from pathlib import Path

from readcoach.policy_compiler import audit, compile_rules, load_policies
from readcoach.trace import SessionTrace

_DEFAULT_POLICIES_DIR = Path(__file__).resolve().parent.parent.parent / "policies"


def invariants_metrics(
    trace: SessionTrace,
    *,
    policies_dir: str | Path | None = None,
) -> dict[str, int]:
    """Compute ``{"violations": <int>}`` for ``trace`` against the policies.

    Wire this into a report as ``metrics["invariants"] = invariants_metrics(trace)``
    so the gate's ``invariants.violations`` rule (min/0) sees the real value.
    """
    rules = load_policies(policies_dir or _DEFAULT_POLICIES_DIR)
    checks = compile_rules(rules)
    report = audit(trace, checks)
    return {"violations": report.violations}
