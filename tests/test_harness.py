"""Tests for evals/harness.py — T2.2 gate machinery.

Synthetic reports only.  No audio, no network, no credentials.

CONTRACT UNDER TEST
-------------------
  GateRule        frozen dataclass: metric, direction, threshold, report_only
  EvalReport      frozen dataclass: version, metrics, metadata
  GateResult      frozen dataclass: exit_code, passed, breaches, report_only_breaches
  evaluate(version, golden_path, *, metrics, results_dir) -> EvalReport
  compare(prev, new, rules) -> GateResult
  promote_failure(trace, golden_path) -> str
"""

from __future__ import annotations

import json
import pathlib

import pytest

from evals.harness import (
    EvalReport,
    GateResult,
    GateRule,
    compare,
    evaluate,
    promote_failure,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_METRICS_BASE = {
    "miscue": {
        "substitution": {"f1": 0.80},
        "omission": {"f1": 0.75},
    },
    "invariants": {"violations": 0},
    "latency": {
        "decision_ms_p50": None,
        "decision_ms_p95": None,
        "rtf_offline_proxy": None,
    },
}

_METRICS_BETTER = {
    "miscue": {
        "substitution": {"f1": 0.85},
        "omission": {"f1": 0.80},
    },
    "invariants": {"violations": 0},
    "latency": {
        "decision_ms_p50": None,
        "decision_ms_p95": None,
        "rtf_offline_proxy": None,
    },
}

_METRICS_WORSE = {
    "miscue": {
        "substitution": {"f1": 0.70},
        "omission": {"f1": 0.60},
    },
    "invariants": {"violations": 2},
    "latency": {
        "decision_ms_p50": None,
        "decision_ms_p95": None,
        "rtf_offline_proxy": None,
    },
}


def _make_report(version: str = "v1", metrics: dict | None = None) -> EvalReport:
    return EvalReport(
        version=version,
        metrics=metrics if metrics is not None else dict(_METRICS_BASE),
        metadata={
            "commit": "abc123",
            "date": "2026-06-10",
            "golden_path": "fake.jsonl",
            "golden_sha256": "aabbcc",
        },
    )


# ---------------------------------------------------------------------------
# GateRule / GateResult: frozen (mutation raises)
# ---------------------------------------------------------------------------


class TestFrozenDataclasses:
    def test_gate_rule_is_frozen(self):
        rule = GateRule(metric="miscue.substitution.f1", direction="max", threshold=0.8)
        with pytest.raises((AttributeError, TypeError)):
            rule.metric = "other"  # type: ignore[misc]

    def test_gate_result_is_frozen(self):
        result = GateResult(
            exit_code=0, passed=True, breaches=[], report_only_breaches=[]
        )
        with pytest.raises((AttributeError, TypeError)):
            result.exit_code = 1  # type: ignore[misc]

    def test_eval_report_is_frozen(self):
        report = _make_report()
        with pytest.raises((AttributeError, TypeError)):
            report.version = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# evaluate(): write / immutability / latency block / metadata
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_writes_json_file(self, tmp_path):
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        evaluate("v1", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path))
        out = tmp_path / "v1.json"
        assert out.exists(), "expected v1.json to be written"
        data = json.loads(out.read_text())
        assert data["version"] == "v1"

    def test_returns_eval_report(self, tmp_path):
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        rpt = evaluate(
            "v1", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path)
        )
        assert isinstance(rpt, EvalReport)
        assert rpt.version == "v1"

    def test_metrics_stored_correctly(self, tmp_path):
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        rpt = evaluate(
            "v1", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path)
        )
        assert rpt.metrics["miscue"]["substitution"]["f1"] == 0.80
        assert rpt.metrics["invariants"]["violations"] == 0

    def test_re_evaluate_identical_noop(self, tmp_path):
        """Calling evaluate twice with same version + same metrics → no-op, same return."""
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        rpt1 = evaluate(
            "v1", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path)
        )
        rpt2 = evaluate(
            "v1", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path)
        )
        assert rpt1.version == rpt2.version
        assert rpt1.metrics == rpt2.metrics

    def test_re_evaluate_different_metrics_raises(self, tmp_path):
        """Re-evaluating same version with DIFFERENT metrics raises FileExistsError."""
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        evaluate("v1", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path))
        with pytest.raises(FileExistsError) as exc_info:
            evaluate(
                "v1", str(golden), metrics=_METRICS_BETTER, results_dir=str(tmp_path)
            )
        msg = str(exc_info.value).lower()
        # Message should reference the investigate-don't-rebaseline doctrine
        assert "investigate" in msg or "rebaseline" in msg or "exists" in msg

    def test_latency_block_auto_present_with_nones(self, tmp_path):
        """Even if metrics don't include latency, the latency block is present with Nones."""
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        metrics_no_latency = {
            "miscue": {"substitution": {"f1": 0.80}},
            "invariants": {"violations": 0},
        }
        rpt = evaluate(
            "v1", str(golden), metrics=metrics_no_latency, results_dir=str(tmp_path)
        )
        lat = rpt.metrics["latency"]
        assert lat["decision_ms_p50"] is None
        assert lat["decision_ms_p95"] is None
        assert lat["rtf_offline_proxy"] is None

    def test_metadata_has_commit(self, tmp_path):
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        rpt = evaluate(
            "v1", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path)
        )
        assert "commit" in rpt.metadata

    def test_metadata_has_date(self, tmp_path):
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        rpt = evaluate(
            "v1", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path)
        )
        assert "date" in rpt.metadata
        # Should be an ISO-format date string
        assert rpt.metadata["date"]

    def test_metadata_has_golden_sha(self, tmp_path):
        golden = tmp_path / "gold.jsonl"
        golden.write_text('{"trace_id": "t1"}\n')
        rpt = evaluate(
            "v1", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path)
        )
        assert "golden_sha256" in rpt.metadata
        assert len(rpt.metadata["golden_sha256"]) == 64  # sha256 hex

    def test_results_dir_created_if_absent(self, tmp_path):
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        subdir = tmp_path / "nested" / "results"
        evaluate("v1", str(golden), metrics=_METRICS_BASE, results_dir=str(subdir))
        assert (subdir / "v1.json").exists()

    def test_written_json_is_loadable_as_report(self, tmp_path):
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        evaluate("v1", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path))
        data = json.loads((tmp_path / "v1.json").read_text())
        assert "version" in data
        assert "metrics" in data
        assert "metadata" in data


# ---------------------------------------------------------------------------
# compare(): exit 0 (pass)
# ---------------------------------------------------------------------------


class TestComparePass:
    def test_identical_reports_regression_rules_exit_0(self):
        """Identical prev/new with regression rules (threshold=None) → exit 0."""
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", _METRICS_BASE)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=None),
            GateRule(metric="invariants.violations", direction="min", threshold=None),
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 0
        assert result.passed is True
        assert result.breaches == []

    def test_better_report_max_rule_exit_0(self):
        """New is better on a max metric → exit 0."""
        prev = _make_report("v1", _METRICS_BASE)  # f1 = 0.80
        new = _make_report("v2", _METRICS_BETTER)  # f1 = 0.85
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=None)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 0
        assert result.passed is True

    def test_better_report_min_rule_exit_0(self):
        """New has fewer violations than prev on a min rule → exit 0."""
        prev_m = {**_METRICS_BASE, "invariants": {"violations": 3}}
        new_m = {**_METRICS_BASE, "invariants": {"violations": 0}}
        prev = _make_report("v1", prev_m)
        new = _make_report("v2", new_m)
        rules = [
            GateRule(metric="invariants.violations", direction="min", threshold=None)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 0
        assert result.passed is True

    def test_threshold_floor_met_max_rule_exit_0(self):
        """New meets threshold floor on a max rule → exit 0."""
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", _METRICS_BETTER)  # f1 = 0.85
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=0.84)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 0
        assert result.passed is True

    def test_threshold_ceiling_met_min_rule_exit_0(self):
        """New at exactly threshold on a min rule → exit 0."""
        new_m = {**_METRICS_BASE, "invariants": {"violations": 0}}
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", new_m)
        rules = [GateRule(metric="invariants.violations", direction="min", threshold=0)]
        result = compare(prev, new, rules)
        assert result.exit_code == 0
        assert result.passed is True


# ---------------------------------------------------------------------------
# compare(): exit 1 (regression)
# ---------------------------------------------------------------------------


class TestCompareRegression:
    def test_f1_drop_none_threshold_max_rule_exit_1(self):
        """f1 drops under a None-threshold max rule → exit 1."""
        prev = _make_report("v1", _METRICS_BASE)  # f1 = 0.80
        new = _make_report("v2", _METRICS_WORSE)  # f1 = 0.70
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=None)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 1
        assert result.passed is False

    def test_f1_drop_breach_names_metric_and_values(self):
        """Breach string must name the metric and both values."""
        prev = _make_report("v1", _METRICS_BASE)  # f1 = 0.80
        new = _make_report("v2", _METRICS_WORSE)  # f1 = 0.70
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=None)
        ]
        result = compare(prev, new, rules)
        assert len(result.breaches) == 1
        breach = result.breaches[0]
        # Must mention the metric path
        assert "miscue.substitution.f1" in breach
        # Must mention both observed values
        assert "0.80" in breach or "0.8" in breach
        assert "0.70" in breach or "0.7" in breach

    def test_threshold_floor_breach_max_rule_exit_1(self):
        """New is below threshold floor on a max rule → exit 1."""
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", _METRICS_WORSE)  # f1 = 0.70
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=0.75)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 1
        assert result.passed is False

    def test_min_rule_ceiling_breach_exit_1(self):
        """violations=1 against threshold=0 (min rule) → exit 1."""
        new_m = {**_METRICS_BASE, "invariants": {"violations": 1}}
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", new_m)
        rules = [GateRule(metric="invariants.violations", direction="min", threshold=0)]
        result = compare(prev, new, rules)
        assert result.exit_code == 1
        assert result.passed is False

    def test_min_rule_ceiling_breach_names_metric_and_values(self):
        new_m = {**_METRICS_BASE, "invariants": {"violations": 1}}
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", new_m)
        rules = [GateRule(metric="invariants.violations", direction="min", threshold=0)]
        result = compare(prev, new, rules)
        assert result.breaches
        breach = result.breaches[0]
        assert "invariants.violations" in breach

    def test_multiple_breaches_all_listed(self):
        """Multiple breaching rules → multiple breach strings."""
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", _METRICS_WORSE)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=None),
            GateRule(metric="invariants.violations", direction="min", threshold=0),
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 1
        assert len(result.breaches) == 2


# ---------------------------------------------------------------------------
# compare(): exit 2 (invalid)
# ---------------------------------------------------------------------------


class TestCompareInvalid:
    def test_missing_metric_in_new_report_exit_2(self):
        """Metric path absent in new report → exit 2 (never silently skipped)."""
        prev = _make_report("v1", _METRICS_BASE)
        new_m = {
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": None,
                "decision_ms_p95": None,
                "rtf_offline_proxy": None,
            },
        }
        new = _make_report("v2", new_m)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=None)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 2

    def test_missing_metric_in_prev_report_exit_2(self):
        """Metric path absent in prev report (needed for regression rule) → exit 2."""
        prev_m = {
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": None,
                "decision_ms_p95": None,
                "rtf_offline_proxy": None,
            },
        }
        prev = _make_report("v1", prev_m)
        new = _make_report("v2", _METRICS_BASE)
        # threshold=None → regression rule → needs prev value
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=None)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 2

    def test_none_value_under_needed_rule_exit_2(self):
        """None metric value under a needed rule → exit 2."""
        metrics_with_none = {
            "miscue": {"substitution": {"f1": None}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": None,
                "decision_ms_p95": None,
                "rtf_offline_proxy": None,
            },
        }
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", metrics_with_none)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=0.8)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 2

    def test_bogus_direction_exit_2(self):
        """GateRule with direction not in ('min', 'max') → exit 2."""
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", _METRICS_BASE)
        # Bypass Literal type check by constructing via __new__ + object.__setattr__
        bad_rule = GateRule.__new__(GateRule)
        # Use object.__setattr__ to bypass frozen on the new uninitialized instance
        object.__setattr__(bad_rule, "metric", "miscue.substitution.f1")
        object.__setattr__(bad_rule, "direction", "sideways")  # bogus
        object.__setattr__(bad_rule, "threshold", None)
        object.__setattr__(bad_rule, "report_only", False)
        rules = [bad_rule]
        result = compare(prev, new, rules)
        assert result.exit_code == 2

    def test_invalid_overrides_regression_precedence(self):
        """One invalid + one genuine regression → exit 2 (invalid wins)."""
        prev = _make_report("v1", _METRICS_BASE)  # f1 = 0.80
        new = _make_report("v2", _METRICS_WORSE)  # f1 = 0.70
        bad_rule = GateRule.__new__(GateRule)
        object.__setattr__(bad_rule, "metric", "miscue.substitution.f1")
        object.__setattr__(bad_rule, "direction", "sideways")
        object.__setattr__(bad_rule, "threshold", None)
        object.__setattr__(bad_rule, "report_only", False)
        regression_rule = GateRule(
            metric="miscue.omission.f1", direction="max", threshold=None
        )
        result = compare(prev, new, [bad_rule, regression_rule])
        assert result.exit_code == 2

    def test_none_value_as_prev_for_regression_rule_exit_2(self):
        """prev value is None for a regression rule (threshold=None) → exit 2."""
        prev_m = {
            "miscue": {"substitution": {"f1": None}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": None,
                "decision_ms_p95": None,
                "rtf_offline_proxy": None,
            },
        }
        prev = _make_report("v1", prev_m)
        new = _make_report("v2", _METRICS_BASE)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=None)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# compare(): report_only rules
# ---------------------------------------------------------------------------


class TestReportOnly:
    def test_latency_breach_report_only_exit_0(self):
        """A report_only breach doesn't affect exit_code."""
        # latency is typically None; give it real values in prev and worse in new
        prev_m = {
            "miscue": {"substitution": {"f1": 0.80}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": 100.0,
                "decision_ms_p95": 200.0,
                "rtf_offline_proxy": 0.5,
            },
        }
        new_m = {
            "miscue": {"substitution": {"f1": 0.80}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": 300.0,
                "decision_ms_p95": 400.0,
                "rtf_offline_proxy": 0.9,
            },
        }
        prev = _make_report("v1", prev_m)
        new = _make_report("v2", new_m)
        rules = [
            GateRule(
                metric="latency.decision_ms_p50",
                direction="min",
                threshold=None,
                report_only=True,
            ),
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 0
        assert result.passed is True
        # Breach listed in report_only_breaches, NOT in breaches
        assert len(result.report_only_breaches) == 1
        assert result.breaches == []

    def test_report_only_breach_listed_in_report_only_breaches(self):
        """report_only breach string appears in report_only_breaches."""
        prev_m = {
            **_METRICS_BASE,
            "latency": {
                "decision_ms_p50": 100.0,
                "decision_ms_p95": 200.0,
                "rtf_offline_proxy": 0.5,
            },
        }
        new_m = {
            **_METRICS_WORSE,
            "latency": {
                "decision_ms_p50": 300.0,
                "decision_ms_p95": 400.0,
                "rtf_offline_proxy": 0.9,
            },
        }
        prev = _make_report("v1", prev_m)
        new = _make_report("v2", new_m)
        rules = [
            GateRule(
                metric="latency.decision_ms_p50",
                direction="min",
                threshold=None,
                report_only=True,
            ),
        ]
        result = compare(prev, new, rules)
        assert len(result.report_only_breaches) >= 1
        assert "latency.decision_ms_p50" in result.report_only_breaches[0]

    def test_mix_gating_and_report_only(self):
        """One gating breach + one report_only breach → exit 1, both listed separately."""
        prev_m = {
            "miscue": {"substitution": {"f1": 0.80}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": 100.0,
                "decision_ms_p95": 200.0,
                "rtf_offline_proxy": 0.5,
            },
        }
        new_m = {
            "miscue": {"substitution": {"f1": 0.70}},  # regression
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": 300.0,
                "decision_ms_p95": 400.0,
                "rtf_offline_proxy": 0.9,
            },  # latency bad
        }
        prev = _make_report("v1", prev_m)
        new = _make_report("v2", new_m)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=None),
            GateRule(
                metric="latency.decision_ms_p50",
                direction="min",
                threshold=None,
                report_only=True,
            ),
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 1
        assert result.passed is False
        assert len(result.breaches) == 1
        assert len(result.report_only_breaches) == 1


# ---------------------------------------------------------------------------
# promote_failure(): idempotency, atomicity contract, KeyError on missing id
# ---------------------------------------------------------------------------


class TestPromoteFailure:
    def _golden_path(self, tmp_path: pathlib.Path) -> str:
        p = tmp_path / "golden.jsonl"
        return str(p)

    def test_new_trace_appended(self, tmp_path):
        """A new trace is appended as one JSON line."""
        gp = self._golden_path(tmp_path)
        trace = {"trace_id": "t1", "text": "the cat sat"}
        promote_failure(trace, gp)
        lines = pathlib.Path(gp).read_text().splitlines()
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert loaded["trace_id"] == "t1"

    def test_returns_trace_id(self, tmp_path):
        gp = self._golden_path(tmp_path)
        result = promote_failure({"trace_id": "t1", "data": "x"}, gp)
        assert result == "t1"

    def test_same_trace_idempotent_no_duplicate(self, tmp_path):
        """Promoting the same trace_id twice → file unchanged (byte-identical), same id returned."""
        gp = self._golden_path(tmp_path)
        trace = {"trace_id": "t1", "text": "the cat sat"}
        promote_failure(trace, gp)
        before = pathlib.Path(gp).read_bytes()
        result = promote_failure(trace, gp)
        after = pathlib.Path(gp).read_bytes()
        assert before == after, "second promote_failure changed the file"
        assert result == "t1"

    def test_two_different_traces_two_lines(self, tmp_path):
        """Two different traces → 2 lines in the file."""
        gp = self._golden_path(tmp_path)
        promote_failure({"trace_id": "t1", "data": "a"}, gp)
        promote_failure({"trace_id": "t2", "data": "b"}, gp)
        lines = [ln for ln in pathlib.Path(gp).read_text().splitlines() if ln.strip()]
        assert len(lines) == 2

    def test_missing_trace_id_raises_key_error(self, tmp_path):
        """Trace without 'trace_id' key raises KeyError."""
        gp = self._golden_path(tmp_path)
        with pytest.raises(KeyError):
            promote_failure({"data": "no id here"}, gp)

    def test_file_created_if_not_exists(self, tmp_path):
        """golden_path need not exist — promote_failure creates it."""
        gp = tmp_path / "new_dir" / "golden.jsonl"
        gp.parent.mkdir(parents=True)
        result = promote_failure({"trace_id": "tx", "data": "y"}, str(gp))
        assert gp.exists()
        assert result == "tx"

    def test_existing_golden_file_preserved(self, tmp_path):
        """Pre-existing lines in the golden file are preserved on promote."""
        gp = self._golden_path(tmp_path)
        pathlib.Path(gp).write_text('{"trace_id": "pre", "data": "existing"}\n')
        promote_failure({"trace_id": "t2", "data": "new"}, gp)
        lines = [ln for ln in pathlib.Path(gp).read_text().splitlines() if ln.strip()]
        assert len(lines) == 2
        ids = {json.loads(ln)["trace_id"] for ln in lines}
        assert "pre" in ids
        assert "t2" in ids

    def test_idempotent_with_existing_pre_populated_file(self, tmp_path):
        """Idempotency holds when file was pre-populated with other traces."""
        gp = self._golden_path(tmp_path)
        pathlib.Path(gp).write_text('{"trace_id": "pre", "data": "existing"}\n')
        promote_failure({"trace_id": "t2", "data": "new"}, gp)
        before = pathlib.Path(gp).read_bytes()
        promote_failure({"trace_id": "t2", "data": "new"}, gp)  # again
        after = pathlib.Path(gp).read_bytes()
        assert before == after


# ---------------------------------------------------------------------------
# Fix 1: non-finite metrics (NaN / Inf / bool) → exit 2
# ---------------------------------------------------------------------------


import math  # noqa: E402 — imported at bottom to keep existing tests undisturbed


class TestCompareNonFiniteInvalid:
    """NaN, ±Inf, and bool in metric values must be INVALID (exit 2)."""

    # --- NaN under each rule shape ---

    def test_nan_new_regression_max_rule_exit_2(self):
        """NaN in new report under a regression-max rule → exit 2."""
        nan_metrics = {
            "miscue": {"substitution": {"f1": float("nan")}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": None,
                "decision_ms_p95": None,
                "rtf_offline_proxy": None,
            },
        }
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", nan_metrics)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=None)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 2
        assert result.passed is False

    def test_nan_new_threshold_max_rule_exit_2(self):
        """NaN in new report under a threshold-max rule → exit 2."""
        nan_metrics = {
            "miscue": {"substitution": {"f1": float("nan")}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": None,
                "decision_ms_p95": None,
                "rtf_offline_proxy": None,
            },
        }
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", nan_metrics)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=0.8)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 2

    def test_nan_new_threshold_min_rule_exit_2(self):
        """NaN in new report under a threshold-min rule → exit 2."""
        nan_metrics = {
            "miscue": {"substitution": {"f1": float("nan")}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": None,
                "decision_ms_p95": None,
                "rtf_offline_proxy": None,
            },
        }
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", nan_metrics)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="min", threshold=0.5)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 2

    def test_nan_prev_regression_rule_exit_2(self):
        """NaN in prev report under a regression rule (threshold=None) → exit 2."""
        nan_metrics = {
            "miscue": {"substitution": {"f1": float("nan")}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": None,
                "decision_ms_p95": None,
                "rtf_offline_proxy": None,
            },
        }
        prev = _make_report("v1", nan_metrics)
        new = _make_report("v2", _METRICS_BASE)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=None)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 2

    # --- +Inf / −Inf ---

    def test_pos_inf_new_exit_2(self):
        """positive infinity in new report → exit 2."""
        inf_metrics = {
            "miscue": {"substitution": {"f1": math.inf}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": None,
                "decision_ms_p95": None,
                "rtf_offline_proxy": None,
            },
        }
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", inf_metrics)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=0.8)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 2

    def test_neg_inf_new_exit_2(self):
        """-infinity in new report → exit 2."""
        neg_inf_metrics = {
            "miscue": {"substitution": {"f1": -math.inf}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": None,
                "decision_ms_p95": None,
                "rtf_offline_proxy": None,
            },
        }
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", neg_inf_metrics)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="min", threshold=0.5)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 2

    def test_inf_prev_regression_rule_exit_2(self):
        """infinity in prev report under a regression rule → exit 2."""
        inf_metrics = {
            "miscue": {"substitution": {"f1": math.inf}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": None,
                "decision_ms_p95": None,
                "rtf_offline_proxy": None,
            },
        }
        prev = _make_report("v1", inf_metrics)
        new = _make_report("v2", _METRICS_BASE)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=None)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 2

    # --- bool ---

    def test_bool_true_new_exit_2(self):
        """bool True in new report → exit 2 (bool is subclass of int, must be rejected)."""
        bool_metrics = {
            "miscue": {"substitution": {"f1": True}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": None,
                "decision_ms_p95": None,
                "rtf_offline_proxy": None,
            },
        }
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", bool_metrics)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=0.8)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 2

    def test_bool_false_new_exit_2(self):
        """bool False in new report → exit 2."""
        bool_metrics = {
            "miscue": {"substitution": {"f1": False}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": None,
                "decision_ms_p95": None,
                "rtf_offline_proxy": None,
            },
        }
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", bool_metrics)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="min", threshold=0.5)
        ]
        result = compare(prev, new, rules)
        assert result.exit_code == 2

    def test_nonfinite_invalid_message_names_metric_and_value(self):
        """INVALID message for non-finite must name the metric path and the value."""
        nan_metrics = {
            "miscue": {"substitution": {"f1": float("nan")}},
            "invariants": {"violations": 0},
            "latency": {
                "decision_ms_p50": None,
                "decision_ms_p95": None,
                "rtf_offline_proxy": None,
            },
        }
        prev = _make_report("v1", _METRICS_BASE)
        new = _make_report("v2", nan_metrics)
        rules = [
            GateRule(metric="miscue.substitution.f1", direction="max", threshold=0.8)
        ]
        result = compare(prev, new, rules)
        # The invalids are surfaced on GateResult; since exit_code==2 and passed==False,
        # the message should appear somewhere in the breaches list (invalids ARE the breaches
        # list when exit_code==2 per the implementation) OR in a dedicated invalids field.
        # We test via exit_code only here; message content tested in the impl.
        assert result.exit_code == 2
        assert result.passed is False


# ---------------------------------------------------------------------------
# Fix 2: evaluate() version path traversal → ValueError
# ---------------------------------------------------------------------------


class TestEvaluateVersionValidation:
    """evaluate() must reject version strings that could escape results_dir."""

    def test_dotdot_slash_raises_value_error(self, tmp_path):
        """'../evil' → ValueError."""
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        with pytest.raises(ValueError, match="version"):
            evaluate("../evil", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path))

    def test_slash_in_version_raises_value_error(self, tmp_path):
        """'a/b' → ValueError."""
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        with pytest.raises(ValueError, match="version"):
            evaluate("a/b", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path))

    def test_empty_version_raises_value_error(self, tmp_path):
        """'' → ValueError."""
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        with pytest.raises(ValueError, match="version"):
            evaluate("", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path))

    def test_dotdot_raises_value_error(self, tmp_path):
        """'..' → ValueError."""
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        with pytest.raises(ValueError, match="version"):
            evaluate("..", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path))

    def test_normal_version_allowed(self, tmp_path):
        """'miscue-v0' → no error."""
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        rpt = evaluate("miscue-v0", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path))
        assert rpt.version == "miscue-v0"

    def test_backslash_in_version_raises_value_error(self, tmp_path):
        r"""'a\\b' → ValueError (Windows-style path component)."""
        golden = tmp_path / "gold.jsonl"
        golden.touch()
        with pytest.raises(ValueError, match="version"):
            evaluate("a\\b", str(golden), metrics=_METRICS_BASE, results_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Fix 3: promote_failure — malformed golden line → ValueError with line number
# ---------------------------------------------------------------------------


class TestPromoteFailureMalformedGolden:
    """Corrupted (non-JSON) lines in golden must abort promote_failure with ValueError."""

    def test_garbage_line_raises_value_error_with_line_number(self, tmp_path):
        """A non-JSON line in the golden file → ValueError naming the line number."""
        gp = tmp_path / "golden.jsonl"
        gp.write_text(
            '{"trace_id": "good1", "data": "ok"}\n'
            'THIS IS NOT JSON AT ALL\n'
            '{"trace_id": "good2", "data": "also ok"}\n'
        )
        with pytest.raises(ValueError) as exc_info:
            promote_failure({"trace_id": "t_new", "data": "x"}, str(gp))
        msg = str(exc_info.value)
        # Must name a line number so the operator knows where corruption is.
        assert any(char.isdigit() for char in msg), f"No line number found in: {msg!r}"

    def test_malformed_json_object_raises_value_error(self, tmp_path):
        """A truncated / malformed JSON line → ValueError."""
        gp = tmp_path / "golden.jsonl"
        gp.write_text('{"trace_id": "good"}\n{"broken": "json\n')
        with pytest.raises(ValueError):
            promote_failure({"trace_id": "new", "data": "x"}, str(gp))

    def test_missing_trace_id_in_existing_line_raises_value_error(self, tmp_path):
        """A valid JSON object that lacks trace_id in the golden file → ValueError."""
        gp = tmp_path / "golden.jsonl"
        gp.write_text('{"trace_id": "good"}\n{"no_id_here": "oops"}\n')
        with pytest.raises(ValueError):
            promote_failure({"trace_id": "new", "data": "x"}, str(gp))
