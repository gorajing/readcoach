"""T4.1 — deterministic policy replay over the benchmark gold.

Done-when: replay WAIT-rate in [0.35, 0.50] AND the conservative default rule is
NEVER reached (the matrix covers the replay exhaustively; the default is a
backstop, not a load-bearing path).

The replay is imported and run in-process so the test is hermetic and shares the
exact deterministic model used by ``scripts/policy_replay.py``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_replay_module():
    """Import scripts/policy_replay.py as a module (it is not a package)."""
    path = _PROJECT_ROOT / "scripts" / "policy_replay.py"
    spec = importlib.util.spec_from_file_location("policy_replay", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_replay_wait_rate_in_band():
    replay = _load_replay_module()
    result = replay.run_replay(seed=replay.DEFAULT_SEED)
    wait_rate = result["wait_rate"]
    assert 0.35 <= wait_rate <= 0.50, (
        f"WAIT-rate {wait_rate:.3f} outside the [0.35, 0.50] pedagogy band; "
        f"distribution={result['move_distribution']}"
    )


def test_replay_never_hits_default_rule():
    replay = _load_replay_module()
    result = replay.run_replay(seed=replay.DEFAULT_SEED)
    default_hits = result["rule_distribution"].get(replay.tutor.DEFAULT_RULE_ID, 0)
    assert default_hits == 0, (
        f"conservative default rule fired {default_hits}× in the replay — the "
        f"matrix must cover every replay context; default must stay a backstop"
    )


def test_replay_is_deterministic():
    replay = _load_replay_module()
    a = replay.run_replay(seed=replay.DEFAULT_SEED)
    b = replay.run_replay(seed=replay.DEFAULT_SEED)
    assert a["wait_rate"] == b["wait_rate"]
    assert a["move_distribution"] == b["move_distribution"]


def test_replay_stamps_single_policy_version():
    replay = _load_replay_module()
    result = replay.run_replay(seed=replay.DEFAULT_SEED)
    # Every emitted action carries exactly one policy version — the current one.
    assert result["policy_version"] == replay.tutor.POLICY_VERSION


def test_replay_covers_all_gold_items():
    replay = _load_replay_module()
    result = replay.run_replay(seed=replay.DEFAULT_SEED)
    assert result["n_items"] == 88
    # Every item produces at least one action (page-end at minimum).
    assert result["n_actions"] >= result["n_items"]
