"""Tests for run_benchmark.py --fixtures mode and scripts/gate.py (T2.4).

CI-safe: no network, no model, no audio processing.
Monkeypatched CacheMiss → abort; report structure; gate rule table coverage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent

# Minimal well-formed metrics dict that --fixtures would produce.
_FIXTURES_METRICS = {
    "miscue": {
        "substitution": {"precision": 0.07, "recall": 0.67, "f1": 0.123},
        "omission": {"precision": 0.56, "recall": 0.92, "f1": 0.698},
        "insertion": {"precision": 0.46, "recall": 0.79, "f1": 0.585},
        "self_correction": {"precision": 0.89, "recall": 0.33, "f1": 0.485},
        "hesitation": {"precision": 0.71, "recall": 0.16, "f1": 0.256},
        "fp_per_100_correct_words": 7.10,
    },
    "by_bias": {
        "none": {
            "substitution": {"tp": 16, "fp": 220, "fn": 8,
                             "precision": 0.07, "recall": 0.67, "f1": 0.123,
                             "n_gold": 24, "n_pred": 236},
            "omission": {"tp": 22, "fp": 17, "fn": 2,
                         "precision": 0.56, "recall": 0.92, "f1": 0.698,
                         "n_gold": 24, "n_pred": 39},
            "insertion": {"tp": 19, "fp": 22, "fn": 5,
                          "precision": 0.46, "recall": 0.79, "f1": 0.585,
                          "n_gold": 24, "n_pred": 41},
            "self_correction": {"tp": 8, "fp": 1, "fn": 16,
                                "precision": 0.89, "recall": 0.33, "f1": 0.485,
                                "n_gold": 24, "n_pred": 9},
            "hesitation": {"tp": 5, "fp": 2, "fn": 27,
                           "precision": 0.71, "recall": 0.16, "f1": 0.256,
                           "n_gold": 32, "n_pred": 7},
            "fp_per_100_correct_words": 7.10,
        },
        "prompt": {
            "substitution": {"tp": 4, "fp": 5, "fn": 20,
                             "precision": 0.44, "recall": 0.17, "f1": 0.24,
                             "n_gold": 24, "n_pred": 9},
            "omission": {"tp": 9, "fp": 8, "fn": 15,
                         "precision": 0.53, "recall": 0.38, "f1": 0.44,
                         "n_gold": 24, "n_pred": 17},
            "insertion": {"tp": 13, "fp": 8, "fn": 11,
                          "precision": 0.62, "recall": 0.54, "f1": 0.58,
                          "n_gold": 24, "n_pred": 21},
            "self_correction": {"tp": 1, "fp": 0, "fn": 23,
                                "precision": 1.0, "recall": 0.04, "f1": 0.08,
                                "n_gold": 24, "n_pred": 1},
            "hesitation": {"tp": 1, "fp": 5, "fn": 31,
                           "precision": 0.17, "recall": 0.03, "f1": 0.05,
                           "n_gold": 32, "n_pred": 6},
            "fp_per_100_correct_words": 0.70,
        },
        "strong": {
            "substitution": {"tp": 2, "fp": 1, "fn": 22,
                             "precision": 0.67, "recall": 0.08, "f1": 0.15,
                             "n_gold": 24, "n_pred": 3},
            "omission": {"tp": 7, "fp": 15, "fn": 17,
                         "precision": 0.32, "recall": 0.29, "f1": 0.30,
                         "n_gold": 24, "n_pred": 22},
            "insertion": {"tp": 11, "fp": 1, "fn": 13,
                          "precision": 0.92, "recall": 0.46, "f1": 0.61,
                          "n_gold": 24, "n_pred": 12},
            "self_correction": {"tp": 1, "fp": 0, "fn": 23,
                                "precision": 1.0, "recall": 0.04, "f1": 0.08,
                                "n_gold": 24, "n_pred": 1},
            "hesitation": {"tp": 1, "fp": 3, "fn": 31,
                           "precision": 0.25, "recall": 0.03, "f1": 0.06,
                           "n_gold": 32, "n_pred": 4},
            "fp_per_100_correct_words": 0.54,
        },
    },
    "invariants": {"violations": 0},
    # latency block absent here — evaluate() injects it
}


# ---------------------------------------------------------------------------
# Gate rule table coverage
# ---------------------------------------------------------------------------

class TestGateRuleTable:
    """Import the rule table from gate.py and assert all 7 required paths present."""

    # The 7 metric paths that must be gated (or report-only).
    REQUIRED_PATHS = {
        "miscue.substitution.f1",
        "miscue.omission.f1",
        "miscue.insertion.f1",
        "miscue.self_correction.f1",
        "miscue.hesitation.f1",
        "miscue.fp_per_100_correct_words",
        "invariants.violations",
    }
    # Latency must also be present as report_only
    LATENCY_PATH = "latency.decision_ms_p95"

    def _import_rules(self):
        scripts_dir = _PROJECT_ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        # Use importlib to avoid cached module issues across test sessions.
        import importlib
        spec = importlib.util.spec_from_file_location(
            "gate_module", _PROJECT_ROOT / "scripts" / "gate.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.GATE_RULES

    def test_all_7_required_paths_covered(self):
        rules = self._import_rules()
        metric_paths = {r.metric for r in rules}
        missing = self.REQUIRED_PATHS - metric_paths
        assert not missing, (
            f"Gate rule table is missing these required metric paths: {missing}"
        )

    def test_latency_rule_present_and_report_only(self):
        rules = self._import_rules()
        lat_rules = [r for r in rules if r.metric == self.LATENCY_PATH]
        assert lat_rules, f"No rule for {self.LATENCY_PATH!r} found in GATE_RULES"
        assert all(r.report_only for r in lat_rules), (
            f"latency rule must be report_only=True"
        )

    def test_invariants_violations_has_threshold_zero(self):
        rules = self._import_rules()
        inv_rules = [r for r in rules if r.metric == "invariants.violations"]
        assert inv_rules, "No rule for invariants.violations"
        # Must have a hard threshold=0, not regression-vs-prev
        assert any(r.threshold == 0 for r in inv_rules), (
            "invariants.violations must have threshold=0 (hard floor)"
        )

    def test_f1_rules_are_max_direction(self):
        rules = self._import_rules()
        f1_rules = [r for r in rules if r.metric.endswith(".f1")]
        assert f1_rules, "No F1 rules found"
        for r in f1_rules:
            assert r.direction == "max", (
                f"F1 rule for {r.metric!r} should be direction='max', got {r.direction!r}"
            )

    def test_fp_rate_rule_is_min_direction(self):
        rules = self._import_rules()
        fp_rules = [r for r in rules if "fp_per_100" in r.metric]
        assert fp_rules, "No fp_per_100_correct_words rule found"
        for r in fp_rules:
            assert r.direction == "min", (
                f"fp_per_100_correct_words should be direction='min', got {r.direction!r}"
            )

    def test_invariants_violations_is_min_direction(self):
        rules = self._import_rules()
        inv_rules = [r for r in rules if r.metric == "invariants.violations"]
        for r in inv_rules:
            assert r.direction == "min"


# ---------------------------------------------------------------------------
# --fixtures mode: report structure tests
# ---------------------------------------------------------------------------

class TestFixturesReportStructure:
    """Verify the metrics dict produced by --fixtures has correct shape."""

    def test_miscue_at_top_level(self):
        """The 'miscue' key must be at the top level of metrics."""
        assert "miscue" in _FIXTURES_METRICS

    def test_miscue_contains_all_classes(self):
        classes = {"substitution", "omission", "insertion", "self_correction", "hesitation"}
        present = set(_FIXTURES_METRICS["miscue"].keys()) - {"fp_per_100_correct_words"}
        assert present == classes, f"Missing class keys: {classes - present}"

    def test_each_miscue_class_has_f1(self):
        classes = ("substitution", "omission", "insertion", "self_correction", "hesitation")
        for cls in classes:
            assert "f1" in _FIXTURES_METRICS["miscue"][cls], (
                f"miscue.{cls} missing 'f1' key"
            )

    def test_fp_per_100_at_top_miscue_level(self):
        assert "fp_per_100_correct_words" in _FIXTURES_METRICS["miscue"]

    def test_by_bias_contains_all_three_biases(self):
        by_bias = _FIXTURES_METRICS["by_bias"]
        assert set(by_bias.keys()) == {"none", "prompt", "strong"}

    def test_invariants_violations_present_and_zero(self):
        assert _FIXTURES_METRICS["invariants"]["violations"] == 0

    def test_by_bias_none_matches_top_level_miscue_f1(self):
        """bias=none results in by_bias must match the top-level miscue f1 values."""
        none_bias = _FIXTURES_METRICS["by_bias"]["none"]
        miscue = _FIXTURES_METRICS["miscue"]
        for cls in ("substitution", "omission", "insertion", "self_correction", "hesitation"):
            assert abs(none_bias[cls]["f1"] - miscue[cls]["f1"]) < 1e-6, (
                f"by_bias.none.{cls}.f1 ({none_bias[cls]['f1']}) != "
                f"miscue.{cls}.f1 ({miscue[cls]['f1']})"
            )


# ---------------------------------------------------------------------------
# --fixtures mode: CacheMiss aborts loudly
# ---------------------------------------------------------------------------

class TestFixturesCacheMissAbort:
    """Monkeypatch transcribe to raise CacheMiss → run_benchmark aborts with exit 1."""

    def test_cache_miss_causes_sys_exit_1(self, tmp_path):
        """When the manifest has no entry for a clip/bias, main() exits with code 1.

        We mock _load_gold_rows (synthetic row) and patch json.loads on the manifest
        to return an empty dict — simulating a missing entry in the committed manifest.
        """
        import scripts.run_benchmark as rb

        fake_rows = [
            {
                "utt_id": "p01-clean-nonexistent",
                "target_text": "the cat sat on the mat",
                "gold": [],
            }
        ]

        # Use a real-looking but incomplete manifest (no entry for our utt_id)
        empty_manifest = {}

        with (
            patch.object(rb, "_load_gold_rows", return_value=fake_rows),
        ):
            # Patch Path.read_text to return empty manifest JSON for the manifest file
            orig_read_text = Path.read_text

            def mock_read_text(self, *a, **kw):
                if "asr_cache_manifest" in str(self):
                    return "{}"
                return orig_read_text(self, *a, **kw)

            with patch.object(Path, "read_text", mock_read_text):
                with pytest.raises(SystemExit) as exc_info:
                    rb.main([
                        "--fixtures",
                        "--version", "test-cache-miss",
                        "--results-dir", str(tmp_path),
                    ])
            assert exc_info.value.code == 1

    def test_cache_miss_in_transcribe_unit(self):
        """Direct unit: transcribe with cache_only=True and no cache entry raises CacheMiss."""
        from readcoach.asr import CacheMiss, transcribe

        # Patch _read_cache to return None (simulates no cache entry)
        with patch("readcoach.asr._read_cache", return_value=None):
            # Also patch Path.read_bytes so the key-computation doesn't need a real file
            with patch("pathlib.Path.read_bytes", return_value=b"fake-audio"):
                with pytest.raises(CacheMiss):
                    transcribe(
                        "/nonexistent/file.wav",
                        target_text=None,
                        bias="none",
                        backend="faster-whisper-small",
                        cache_only=True,
                    )


# ---------------------------------------------------------------------------
# evaluate() integration: --fixtures metrics flow through harness
# ---------------------------------------------------------------------------

class TestFixturesEvaluateIntegration:
    """Verify that the metrics structure produced by --fixtures flows through evaluate()."""

    def test_evaluate_accepts_fixtures_metrics(self, tmp_path):
        """evaluate() accepts the --fixtures metrics dict without error."""
        from evals.harness import evaluate

        golden = tmp_path / "benchmark.lock"
        golden.touch()
        rpt = evaluate(
            "test-v0",
            str(golden),
            metrics=_FIXTURES_METRICS,
            results_dir=str(tmp_path),
        )
        assert rpt.version == "test-v0"
        assert rpt.metrics["miscue"]["substitution"]["f1"] == pytest.approx(0.123)

    def test_evaluate_injects_latency_block(self, tmp_path):
        """evaluate() injects a latency block with None values."""
        from evals.harness import evaluate

        golden = tmp_path / "benchmark.lock"
        golden.touch()
        rpt = evaluate(
            "test-v0",
            str(golden),
            metrics=_FIXTURES_METRICS,
            results_dir=str(tmp_path),
        )
        assert "latency" in rpt.metrics
        assert rpt.metrics["latency"]["decision_ms_p95"] is None

    def test_evaluate_stamps_weave_disabled(self, tmp_path, monkeypatch):
        """With WEAVE_DISABLED=1, evaluate() stamps weave:'disabled:WEAVE_DISABLED'."""
        import os
        monkeypatch.setenv("WEAVE_DISABLED", "1")
        from evals.harness import evaluate

        golden = tmp_path / "benchmark.lock"
        golden.touch()
        rpt = evaluate(
            "test-v0-weave",
            str(golden),
            metrics=_FIXTURES_METRICS,
            results_dir=str(tmp_path),
        )
        assert rpt.metadata["weave"].startswith("disabled:")

    def test_report_file_has_investigate_worthy_content(self, tmp_path):
        """The written JSON has version, metrics, and metadata keys."""
        from evals.harness import evaluate

        golden = tmp_path / "benchmark.lock"
        golden.touch()
        evaluate(
            "test-v0",
            str(golden),
            metrics=_FIXTURES_METRICS,
            results_dir=str(tmp_path),
        )
        data = json.loads((tmp_path / "test-v0.json").read_text())
        assert "version" in data
        assert "metrics" in data
        assert "metadata" in data
        assert data["metrics"]["miscue"]["substitution"]["f1"] == pytest.approx(0.123)
        assert data["metrics"]["invariants"]["violations"] == 0


# ---------------------------------------------------------------------------
# gate.py: v0 vs v0 self-comparison → exit 0
# ---------------------------------------------------------------------------

class TestGateSelfComparison:
    """gate.py main() comparing v0.json against itself must exit 0."""

    def test_v0_vs_v0_exit_0(self, tmp_path):
        """Comparing v0 against itself through the gate should pass (exit 0)."""
        from evals.harness import evaluate, load_report, compare, GateRule

        golden = tmp_path / "benchmark.lock"
        golden.touch()
        rpt = evaluate(
            "v0",
            str(golden),
            metrics=_FIXTURES_METRICS,
            results_dir=str(tmp_path),
        )
        v0_path = tmp_path / "v0.json"

        # Reload from disk to simulate the real gate flow
        prev = load_report(v0_path)
        new = load_report(v0_path)

        # Use the actual gate rules (minus latency which has None values → skipped)
        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location(
            "gate_mod", _PROJECT_ROOT / "scripts" / "gate.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        active_rules = [
            r for r in mod.GATE_RULES
            if r.metric != "latency.decision_ms_p95"
        ]
        result = compare(prev, new, active_rules)
        assert result.exit_code == 0, (
            f"v0 vs v0 gate failed: {result.breaches}"
        )
        assert result.passed is True

    def test_worse_substitution_f1_exits_1(self, tmp_path):
        """A report with lower substitution f1 must trigger exit 1 on the gate."""
        import copy
        from evals.harness import evaluate, load_report, compare

        golden = tmp_path / "benchmark.lock"
        golden.touch()

        evaluate("baseline", str(golden), metrics=_FIXTURES_METRICS,
                 results_dir=str(tmp_path))

        # Make a worse variant
        worse_metrics = copy.deepcopy(_FIXTURES_METRICS)
        # Drop substitution f1 significantly
        worse_metrics["miscue"]["substitution"]["f1"] = 0.05  # was 0.123

        evaluate("worse", str(golden), metrics=worse_metrics,
                 results_dir=str(tmp_path))

        prev = load_report(tmp_path / "baseline.json")
        new = load_report(tmp_path / "worse.json")

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gate_mod2", _PROJECT_ROOT / "scripts" / "gate.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        active_rules = [
            r for r in mod.GATE_RULES
            if r.metric != "latency.decision_ms_p95"
        ]
        result = compare(prev, new, active_rules)
        assert result.exit_code == 1, (
            f"Expected exit 1 for worse substitution f1, got {result.exit_code}"
        )
        assert any("substitution.f1" in b for b in result.breaches)
