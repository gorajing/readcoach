"""Tests for evals/tracing.py — T2.3 Weave scoped.

Contract under test
-------------------
weave_status()       -> "disabled:<reason>" | "enabled:<project>"
initialize_weave()   -> str; lazy import; fails loud on enabled-init error
weave_doctor()       -> dict with boolean flags only; never leaks env values
evaluate() metadata  -> contains weave="disabled:..." under test env

All tests run under the hermetic conftest (WANDB_* cleared, WANDB_MODE=disabled).
Laziness proofs assert "weave" absent from sys.modules after disabled-path calls.
"""
from __future__ import annotations

import json
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pop_weave_from_modules() -> None:
    """Remove all weave-related entries from sys.modules (if present).

    Needed to prove that a subsequent call to weave_status/initialize_weave
    does NOT re-import the package.  We restore nothing — the test session
    continues without weave loaded, which is the desired state.

    Note: wandb does NOT register a pytest11 plugin (confirmed), so there is no
    wandb-induced pre-import of weave during test collection.  This helper is
    belt-and-suspenders only.
    """
    to_remove = [k for k in sys.modules if k == "weave" or k.startswith("weave.")]
    for k in to_remove:
        del sys.modules[k]


# ---------------------------------------------------------------------------
# weave_status(): disabled path
# ---------------------------------------------------------------------------

class TestWeaveStatusDisabled:
    def test_disabled_under_hermetic_env(self):
        """Hermetic conftest clears WANDB_API_KEY → status must start with 'disabled'."""
        from evals.tracing import weave_status
        assert weave_status().startswith("disabled")

    def test_disabled_reason_no_key(self, monkeypatch):
        """Without a key, reason is 'no_wandb_api_key'."""
        from evals.tracing import weave_status
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WEAVE_DISABLED", raising=False)
        assert weave_status() == "disabled:no_wandb_api_key"

    def test_weave_disabled_flag_wins_even_with_key(self, monkeypatch):
        """WEAVE_DISABLED=1 wins unconditionally even when WANDB_API_KEY is set."""
        from evals.tracing import weave_status
        monkeypatch.setenv("WANDB_API_KEY", "fake-key-xyz")
        monkeypatch.setenv("WEAVE_DISABLED", "1")
        status = weave_status()
        assert status.startswith("disabled")
        assert "WEAVE_DISABLED" in status

    def test_weave_disabled_flag_wins_with_arbitrary_truthy_value(self, monkeypatch):
        """WEAVE_DISABLED=yes also triggers disabled."""
        from evals.tracing import weave_status
        monkeypatch.setenv("WANDB_API_KEY", "fake-key-xyz")
        monkeypatch.setenv("WEAVE_DISABLED", "yes")
        assert weave_status().startswith("disabled")

    def test_weave_disabled_empty_string_is_not_disabled(self, monkeypatch):
        """WEAVE_DISABLED='' (empty) should NOT trigger disabled (key wins)."""
        from evals.tracing import weave_status
        monkeypatch.setenv("WANDB_API_KEY", "fake-key-xyz")
        monkeypatch.setenv("WEAVE_DISABLED", "")
        # With a key set and WEAVE_DISABLED empty, should be enabled
        status = weave_status()
        assert status.startswith("enabled")


# ---------------------------------------------------------------------------
# weave_status(): enabled path
# ---------------------------------------------------------------------------

class TestWeaveStatusEnabled:
    def test_enabled_with_key_uses_default_project(self, monkeypatch):
        """With WANDB_API_KEY set and no WEAVE_DISABLED, status is 'enabled:<project>'."""
        from evals.tracing import weave_status
        monkeypatch.setenv("WANDB_API_KEY", "fake-key-xyz")
        monkeypatch.delenv("WEAVE_DISABLED", raising=False)
        monkeypatch.delenv("WEAVE_PROJECT", raising=False)
        status = weave_status()
        assert status == "enabled:readcoach-evals"

    def test_enabled_with_custom_project(self, monkeypatch):
        """WEAVE_PROJECT env var sets the project name."""
        from evals.tracing import weave_status
        monkeypatch.setenv("WANDB_API_KEY", "fake-key-xyz")
        monkeypatch.delenv("WEAVE_DISABLED", raising=False)
        monkeypatch.setenv("WEAVE_PROJECT", "my-custom-project")
        status = weave_status()
        assert status == "enabled:my-custom-project"


# ---------------------------------------------------------------------------
# Laziness proof: disabled path must NOT import weave
# ---------------------------------------------------------------------------

class TestLaziness:
    def test_weave_not_imported_after_disabled_status_call(self, monkeypatch):
        """Calling weave_status() on disabled path leaves 'weave' out of sys.modules."""
        _pop_weave_from_modules()
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WEAVE_DISABLED", raising=False)

        from evals.tracing import weave_status
        status = weave_status()
        assert status.startswith("disabled"), f"unexpected: {status}"
        assert "weave" not in sys.modules, (
            f"weave was imported on the disabled path: "
            f"{[k for k in sys.modules if k == 'weave' or k.startswith('weave.')]}"
        )

    def test_weave_not_imported_after_disabled_initialize_call(self, monkeypatch):
        """Calling initialize_weave() on disabled path leaves 'weave' out of sys.modules."""
        _pop_weave_from_modules()
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WEAVE_DISABLED", raising=False)

        from evals.tracing import initialize_weave
        status = initialize_weave()
        assert status.startswith("disabled"), f"unexpected: {status}"
        assert "weave" not in sys.modules, (
            f"weave was imported on the disabled path: "
            f"{[k for k in sys.modules if k == 'weave' or k.startswith('weave.')]}"
        )

    def test_weave_disabled_flag_path_no_import(self, monkeypatch):
        """WEAVE_DISABLED=1 path does not import weave even if key is present."""
        _pop_weave_from_modules()
        monkeypatch.setenv("WANDB_API_KEY", "fake-key-xyz")
        monkeypatch.setenv("WEAVE_DISABLED", "1")

        from evals.tracing import initialize_weave
        status = initialize_weave()
        assert status.startswith("disabled")
        assert "weave" not in sys.modules


# ---------------------------------------------------------------------------
# initialize_weave(): enabled-path failure raises (no silent fallback)
# ---------------------------------------------------------------------------

class TestInitializeWeaveFailLoud:
    def test_enabled_init_failure_raises(self, monkeypatch):
        """When weave.init() raises on the enabled path, the exception propagates."""
        # Step 1: ensure the disabled flag is not set and a key is present.
        monkeypatch.setenv("WANDB_API_KEY", "fake-key-for-test")
        monkeypatch.delenv("WEAVE_DISABLED", raising=False)
        monkeypatch.delenv("WEAVE_PROJECT", raising=False)

        # Step 2: inject a fake weave module whose init() raises.
        fake_weave = types.ModuleType("weave")

        def _fake_init(project: str) -> None:
            raise RuntimeError("simulated weave.init failure")

        fake_weave.init = _fake_init  # type: ignore[attr-defined]

        # Stash any real weave and replace with fake.
        original_weave = sys.modules.get("weave")
        sys.modules["weave"] = fake_weave
        try:
            from evals import tracing as _tracing_module
            # Reload to pick up sys.modules["weave"] correctly in the lazy import.
            import importlib
            importlib.reload(_tracing_module)

            with pytest.raises(RuntimeError, match="simulated weave.init failure"):
                _tracing_module.initialize_weave()
        finally:
            # Restore original state.
            if original_weave is not None:
                sys.modules["weave"] = original_weave
            elif "weave" in sys.modules:
                del sys.modules["weave"]
            # Reload the module again to restore clean state for subsequent tests.
            import importlib
            importlib.reload(_tracing_module)


# ---------------------------------------------------------------------------
# evaluate() metadata stamps weave status
# ---------------------------------------------------------------------------

class TestEvaluateWeaveMetadata:
    def test_evaluate_stamps_weave_in_metadata(self, tmp_path):
        """evaluate() report metadata contains weave='disabled:...' under test env."""
        from evals.harness import evaluate

        golden = tmp_path / "gold.jsonl"
        golden.touch()
        metrics = {
            "miscue": {"substitution": {"f1": 0.80}},
            "invariants": {"violations": 0},
        }
        rpt = evaluate("v_weave_test", str(golden), metrics=metrics, results_dir=str(tmp_path))
        assert "weave" in rpt.metadata, "evaluate() must stamp 'weave' in metadata"
        assert rpt.metadata["weave"].startswith("disabled"), (
            f"Under test env (no creds), expected disabled:..., got: {rpt.metadata['weave']!r}"
        )

    def test_evaluate_weave_metadata_written_to_json(self, tmp_path):
        """The weave stamp must be persisted in the JSON report file."""
        from evals.harness import evaluate

        golden = tmp_path / "gold.jsonl"
        golden.touch()
        metrics = {
            "miscue": {"substitution": {"f1": 0.80}},
            "invariants": {"violations": 0},
        }
        evaluate("v_weave_json", str(golden), metrics=metrics, results_dir=str(tmp_path))
        data = json.loads((tmp_path / "v_weave_json.json").read_text())
        assert "weave" in data["metadata"]
        assert data["metadata"]["weave"].startswith("disabled")


# ---------------------------------------------------------------------------
# weave_doctor(): booleans only, no value leakage
# ---------------------------------------------------------------------------

class TestWeaveDoctorSafety:
    def test_doctor_returns_dict(self):
        from evals.tracing import weave_doctor
        result = weave_doctor()
        assert isinstance(result, dict)

    def test_doctor_has_expected_keys(self):
        from evals.tracing import weave_doctor
        result = weave_doctor()
        for key in (
            "weave_disabled_flag",
            "wandb_api_key_present",
            "weave_project_set",
            "weave_importable",
            "status",
        ):
            assert key in result, f"Expected key {key!r} missing from doctor output"

    def test_doctor_boolean_fields_are_booleans(self):
        from evals.tracing import weave_doctor
        result = weave_doctor()
        bool_keys = ("weave_disabled_flag", "wandb_api_key_present", "weave_project_set", "weave_importable")
        for key in bool_keys:
            assert isinstance(result[key], bool), (
                f"doctor()[{key!r}] should be bool, got {type(result[key]).__name__}"
            )

    def test_doctor_no_value_leakage(self, monkeypatch):
        """Planting a sentinel key value; doctor output must not contain it."""
        sentinel = "sekrit-xyz-do-not-leak"
        monkeypatch.setenv("WANDB_API_KEY", sentinel)

        from evals.tracing import weave_doctor
        result = weave_doctor()
        serialized = json.dumps(result)
        assert sentinel not in serialized, (
            f"Doctor output leaked the WANDB_API_KEY value: {serialized!r}"
        )

    def test_doctor_no_leakage_of_weave_project_value(self, monkeypatch):
        """Custom WEAVE_PROJECT value must not appear in doctor output."""
        sentinel_project = "supersecret-project-name"
        monkeypatch.setenv("WEAVE_PROJECT", sentinel_project)
        # Note: the STATUS string will contain the project name when enabled,
        # but WEAVE_PROJECT itself as a raw value should only appear in 'status'
        # if enabled — for this test we ensure the WANDB_API_KEY is absent
        # so we stay on the disabled path and the project name doesn't leak.
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WEAVE_DISABLED", raising=False)

        from evals.tracing import weave_doctor
        result = weave_doctor()
        # On disabled path, status should be disabled:no_wandb_api_key
        # and the project name should not appear anywhere.
        assert sentinel_project not in json.dumps(result)

    def test_doctor_wandb_api_key_present_is_false_under_hermetic_env(self):
        """Under the hermetic test env (no WANDB_API_KEY), doctor reports False."""
        from evals.tracing import weave_doctor
        result = weave_doctor()
        assert result["wandb_api_key_present"] is False

    def test_doctor_wandb_api_key_present_is_true_when_set(self, monkeypatch):
        """With WANDB_API_KEY set, doctor reports True for wandb_api_key_present."""
        monkeypatch.setenv("WANDB_API_KEY", "any-fake-key")
        from evals.tracing import weave_doctor
        result = weave_doctor()
        assert result["wandb_api_key_present"] is True

    def test_doctor_weave_importable_reflects_package_availability(self):
        """weave_importable reflects whether the weave package is importable."""
        from evals.tracing import weave_doctor
        result = weave_doctor()
        # The package IS installed in this project (pyproject.toml lists it).
        assert result["weave_importable"] is True

    def test_doctor_status_matches_weave_status(self):
        """doctor()['status'] matches the direct weave_status() call."""
        from evals.tracing import weave_doctor, weave_status
        result = weave_doctor()
        assert result["status"] == weave_status()
