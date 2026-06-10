"""Smoke tests: clean imports and hermetic-guard regression.

These tests must pass with zero credentials in the environment.
"""
import os


# ---------------------------------------------------------------------------
# Package import tests
# ---------------------------------------------------------------------------

def test_readcoach_version_is_string():
    import readcoach
    assert isinstance(readcoach.__version__, str)
    assert len(readcoach.__version__) > 0


def test_stub_modules_import():
    import readcoach.asr          # noqa: F401
    import readcoach.miscue       # noqa: F401
    import readcoach.learner_model  # noqa: F401
    import readcoach.tutor        # noqa: F401
    import evals.harness          # noqa: F401


# ---------------------------------------------------------------------------
# Hermetic-guard regression test
# Verifies that conftest.py actually cleared credential env vars before
# any test runs.  If this test fails, a credential has leaked in.
# ---------------------------------------------------------------------------

def test_hermetic_guard_no_credentials():
    assert os.environ.get("ANTHROPIC_API_KEY", "") == "", (
        "ANTHROPIC_API_KEY leaked into test suite — conftest did not clear it"
    )
    assert os.environ.get("WANDB_API_KEY", "") == "", (
        "WANDB_API_KEY leaked into test suite — conftest did not clear it"
    )
    assert os.environ.get("GOOGLE_API_KEY", "") == "", (
        "GOOGLE_API_KEY leaked into test suite — conftest did not clear it"
    )
    assert os.environ.get("WANDB_MODE") == "disabled", (
        "WANDB_MODE should be forced to 'disabled' by conftest"
    )
