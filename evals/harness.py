"""The eval flywheel — prove the tutor improves, gated, with a validated judge.

T2.2: gate machinery — evaluate / compare / promote_failure.

Exit codes: 0 pass / 1 regression / 2 invalid.
Missing metrics are always invalid, never skipped.
Immutable versioned reports: investigate, don't re-baseline.
Latency block present in every report schema from day one; values may be None
until the replay machinery exists. The field name ``rtf_offline_proxy`` bakes
the honest label in.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Dataclasses (all frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateRule:
    metric: str  # dotted path into a report, e.g. "miscue.substitution.f1"
    direction: Literal["min", "max"]
    threshold: float | None = None  # None = regression-vs-prev rule
    report_only: bool = False  # breach listed but never affects exit_code / passed


@dataclass(frozen=True)
class EvalReport:
    version: str
    metrics: dict  # arbitrary nested numeric dict
    metadata: dict  # git commit, date, golden_path sha, etc.


@dataclass(frozen=True)
class GateResult:
    exit_code: int  # 0 pass / 1 regression / 2 invalid
    passed: bool
    breaches: list[str]  # human-readable, gating only
    report_only_breaches: list[str]  # listed but non-gating


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_LATENCY_TEMPLATE: dict[str, Any] = {
    "decision_ms_p50": None,
    "decision_ms_p95": None,
    "rtf_offline_proxy": None,
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _resolve_metric(metrics: dict, dotted_path: str) -> Any:
    """Walk a nested dict by dotted path.  Returns the value or raises KeyError."""
    keys = dotted_path.split(".")
    node: Any = metrics
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"metric path '{dotted_path}' not found (missing key '{k}')")
        node = node[k]
    return node


def _inject_latency_block(metrics: dict) -> dict:
    """Return a shallow-merged copy of metrics that always contains a latency block.

    If a latency key is already present, it is kept as-is (merge keys only if missing).
    We do not mutate the caller's dict.
    """
    result = dict(metrics)
    if "latency" not in result:
        result["latency"] = dict(_LATENCY_TEMPLATE)
    else:
        # Merge: inject any missing keys from the template.
        existing = dict(result["latency"])
        for k, v in _LATENCY_TEMPLATE.items():
            if k not in existing:
                existing[k] = v
        result["latency"] = existing
    return result


def _report_to_dict(report: EvalReport) -> dict:
    return {
        "version": report.version,
        "metrics": report.metrics,
        "metadata": report.metadata,
    }


def _dict_to_report(data: dict) -> EvalReport:
    return EvalReport(
        version=data["version"],
        metrics=data["metrics"],
        metadata=data["metadata"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    version: str,
    golden_path: str,
    *,
    metrics: dict,
    results_dir: str = "evals/results",
) -> EvalReport:
    """Write and return a frozen EvalReport for ``version``.

    Immutability contract
    ---------------------
    * If ``<results_dir>/<version>.json`` already exists AND its content is
      identical to what we would write → no-op, return the report.
    * If it exists AND content differs → raise FileExistsError with a message
      referencing the investigate-don't-rebaseline doctrine.

    The latency block is always present in the metrics; keys missing from the
    caller's ``metrics`` dict are injected as None.

    Atomic write via a temp file in the same directory then os.replace().
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    out_path = results_path / f"{version}.json"
    gp = Path(golden_path)

    # Build metadata.
    golden_sha = _sha256_file(gp) if gp.exists() else "file-not-found"
    metadata: dict = {
        "commit": _git_head_commit(),
        "date": date.today().isoformat(),
        "golden_path": str(golden_path),
        "golden_sha256": golden_sha,
    }

    # Ensure latency block is present.
    full_metrics = _inject_latency_block(metrics)

    report = EvalReport(version=version, metrics=full_metrics, metadata=metadata)
    payload = _report_to_dict(report)
    payload_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if out_path.exists():
        existing_text = out_path.read_text(encoding="utf-8")
        existing_data = json.loads(existing_text)
        # Compare only version and metrics (metadata has date/commit which may differ).
        if (
            existing_data.get("version") == payload["version"]
            and existing_data.get("metrics") == payload["metrics"]
        ):
            # Identical — no-op.
            return _dict_to_report(existing_data)
        raise FileExistsError(
            f"Version '{version}' already has a result at '{out_path}' with different "
            f"metrics. Investigate the discrepancy — do NOT re-baseline. "
            f"(doctrine: investigate-don't-rebaseline)"
        )

    # Atomic write: write to a temp file in the same dir, then replace.
    fd, tmp_name = tempfile.mkstemp(dir=results_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload_text)
        os.replace(tmp_name, out_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    return report


def compare(
    prev: EvalReport,
    new: EvalReport,
    rules: list[GateRule],
) -> GateResult:
    """Evaluate ``rules`` against ``prev`` and ``new`` reports.

    Exit codes
    ----------
    0 — all gating rules pass
    1 — at least one gating regression / threshold breach (no invalids)
    2 — at least one invalid condition (missing path, non-numeric value,
        None value under a needed rule, bogus direction) — OVERRIDES 1.

    report_only rules accumulate into ``report_only_breaches`` but never
    affect exit_code or passed.
    """
    invalids: list[str] = []
    regressions: list[str] = []
    report_only: list[str] = []

    for rule in rules:
        # Validate direction before anything else.
        if rule.direction not in ("min", "max"):
            invalids.append(
                f"INVALID rule for '{rule.metric}': direction={rule.direction!r} "
                f"is not 'min' or 'max'"
            )
            continue

        # Resolve new value (always needed).
        try:
            new_val = _resolve_metric(new.metrics, rule.metric)
        except KeyError as exc:
            invalids.append(f"INVALID: {exc}")
            continue

        if new_val is None:
            invalids.append(
                f"INVALID: metric '{rule.metric}' in new report is None "
                f"(rule requires a numeric value)"
            )
            continue

        if not isinstance(new_val, (int, float)):
            invalids.append(
                f"INVALID: metric '{rule.metric}' in new report is non-numeric "
                f"({type(new_val).__name__})"
            )
            continue

        # Determine reference value (threshold or prev).
        if rule.threshold is not None:
            ref_val = rule.threshold
            ref_label = f"threshold={rule.threshold}"
        else:
            # Regression rule — need prev value.
            try:
                prev_val = _resolve_metric(prev.metrics, rule.metric)
            except KeyError as exc:
                invalids.append(f"INVALID: {exc}")
                continue

            if prev_val is None:
                invalids.append(
                    f"INVALID: metric '{rule.metric}' in prev report is None "
                    f"(regression rule needs a numeric prev value)"
                )
                continue

            if not isinstance(prev_val, (int, float)):
                invalids.append(
                    f"INVALID: metric '{rule.metric}' in prev report is non-numeric "
                    f"({type(prev_val).__name__})"
                )
                continue

            ref_val = prev_val
            ref_label = f"prev={prev_val}"

        # Check rule.
        breached = False
        if rule.direction == "max":
            # larger-is-better; new must be >= ref_val
            if new_val < ref_val:
                breached = True
        else:  # "min"
            # smaller-is-better; new must be <= ref_val
            if new_val > ref_val:
                breached = True

        if breached:
            msg = (
                f"BREACH '{rule.metric}' direction={rule.direction}: "
                f"new={new_val}, {ref_label}"
            )
            if rule.report_only:
                report_only.append(msg)
            else:
                regressions.append(msg)

    # Determine exit code.
    if invalids:
        exit_code = 2
    elif regressions:
        exit_code = 1
    else:
        exit_code = 0

    return GateResult(
        exit_code=exit_code,
        passed=(exit_code == 0),
        breaches=regressions,
        report_only_breaches=report_only,
    )


def promote_failure(trace: dict, golden_path: str) -> str:
    """Append ``trace`` as one JSON line to ``golden_path`` (jsonl).

    Contract
    --------
    * ``trace`` must contain a stable ``trace_id``; raises KeyError if absent.
    * Idempotent: if a line with the same trace_id already exists, do NOT
      append again; return the trace_id either way.
    * Atomic: uses a temp-file + os.replace so a crash cannot leave a
      half-written line.  The approach: read all existing lines, check for
      the id, then write the full file (all old lines + new line) to a temp
      file in the same directory and atomically replace the original.

    Returns
    -------
    The trace_id (str).
    """
    if "trace_id" not in trace:
        raise KeyError("trace must contain a 'trace_id' key")

    trace_id: str = trace["trace_id"]
    gp = Path(golden_path)

    # Load existing lines.
    existing_lines: list[str] = []
    if gp.exists():
        for raw in gp.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                existing_lines.append(raw)

    # Check idempotency.
    for raw in existing_lines:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if obj.get("trace_id") == trace_id:
            return trace_id

    # Append the new trace.
    new_line = json.dumps(trace, ensure_ascii=False, sort_keys=True)
    all_lines = existing_lines + [new_line]
    content = "\n".join(all_lines) + "\n"

    # Atomic write.
    parent = gp.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_name, gp)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    return trace_id


# ---------------------------------------------------------------------------
# Legacy stubs preserved for import compatibility (T1.x modules)
# ---------------------------------------------------------------------------


def validate_judge(hand_labeled_turns: list[dict]) -> dict[str, float]:
    """Report judge-vs-human agreement PER DIMENSION. A judge you haven't measured is
    theater (BEA 2025: judge F1 0.82 vs human 0.91, dimension-dependent)."""
    raise NotImplementedError
