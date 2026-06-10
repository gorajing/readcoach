"""T6.4 — Run LearnerMem v0 memory-consistency probes; publish the artifact.

Runs all six probes (five planted-fact + one over-personalization) over a real
two-session SQLite boundary, prints a results table, and writes
``evals/results/learnermem_v0.json`` (per-probe pass/fail + evidence +
consistency_score + metadata + any honest findings).

The probes are DETERMINISTIC state checks (no LLM, no network, no randomness) —
see ``evals/learnermem.py`` and ``docs/learnermem.md`` for the design and the
PersonaMem / MemoryArena lineage.

Usage
-----
    uv run python scripts/learnermem_probes.py
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Make ``readcoach`` and the top-level ``evals`` package importable when run as a
# plain script (no editable install required).
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from evals.learnermem import (  # noqa: E402
    OVERPERSONALIZATION_FLOOR,
    PROBES,
    LearnerMemReport,
    run_probes,
)
from readcoach.planner import MASTERY_COMPLETED, MASTERY_THRESHOLD  # noqa: E402

_RESULTS_PATH = _PROJECT_ROOT / "evals" / "results" / "learnermem_v0.json"


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _db_path_factory(tmp_dir: Path):
    """Return a factory producing fresh per-probe db paths under tmp_dir."""
    counter = {"n": 0}

    def factory() -> str:
        counter["n"] += 1
        return str(tmp_dir / f"learnermem_{counter['n']}.db")

    return factory


def _print_table(report: LearnerMemReport) -> None:
    descriptions = {p.id: p.description for p in PROBES}
    print("=" * 78)
    print("LearnerMem v0 — memory-consistency probes over the learner store")
    print("=" * 78)
    print(f"{'probe':<6} {'result':<6}  description")
    print("-" * 78)
    for pid in sorted(report.results):
        result = report.results[pid]
        status = "PASS" if result.passed else "FAIL"
        # First clause of the description for a compact one-liner.
        short = descriptions[pid].split(":", 1)[0]
        print(f"{pid:<6} {status:<6}  {short}")
    print("-" * 78)
    print(
        f"consistency_score = {report.n_passed}/{report.n_total} = "
        f"{report.consistency_score:.3f}"
    )
    print("=" * 78)
    print("\nPer-probe evidence:")
    for pid in sorted(report.results):
        print(f"  [{pid}] {report.results[pid].evidence}")
    if report.findings:
        print("\nHonest findings (recorded, not hidden):")
        for f in report.findings:
            print(f"  - {f}")
    print()


def _to_json(report: LearnerMemReport) -> dict:
    descriptions = {p.id: p.description for p in PROBES}
    return {
        "metadata": {
            "eval": "learnermem",
            "version": "v0",
            "scope": (
                "deterministic state probes over the learner store + planner "
                "(no LLM, no network, no randomness); LLM-utterance consistency "
                "is the named v1 extension point"
            ),
            "lineage": [
                "PersonaMem (arXiv:2504.14225)",
                "MemoryArena-style long-horizon consistency evals",
            ],
            "novelty": (
                "first-in-domain: memory-consistency probing applied to a tutor's "
                "learner-model + planner (not LLM-judged dialogue)"
            ),
            "session_boundary": "SQLite close + reopen on the same db per probe",
            "constants": {
                "mastery_threshold": MASTERY_THRESHOLD,
                "mastery_completed": MASTERY_COMPLETED,
                "overpersonalization_floor": OVERPERSONALIZATION_FLOOR,
            },
            "git_commit": _git_head(),
            "date": datetime.date.today().isoformat(),
        },
        "consistency_score": report.consistency_score,
        "n_passed": report.n_passed,
        "n_total": report.n_total,
        "probes": {
            pid: {
                "description": descriptions[pid],
                "passed": result.passed,
                "evidence": result.evidence,
            }
            for pid, result in report.results.items()
        },
        "findings": report.findings,
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="readcoach_learnermem_") as td:
        report = run_probes(_db_path_factory(Path(td)))

    _print_table(report)

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(json.dumps(_to_json(report), indent=2) + "\n")
    print(f"Wrote {_RESULTS_PATH.relative_to(_PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
