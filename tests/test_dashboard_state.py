"""Tests for dashboard_state.build_dashboard_state (T6.2).

All assertions against the committed artifacts in evals/results/.
No Streamlit import permitted in the state module.
"""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

# ---------------------------------------------------------------------------
# Guard: no streamlit import in the state module
# ---------------------------------------------------------------------------

def test_no_streamlit_import_in_state_module() -> None:
    """dashboard_state must not import streamlit (pure logic only)."""
    source = (
        PROJECT_ROOT / "src" / "readcoach" / "dashboard_state.py"
    ).read_text(encoding="utf-8")
    assert "import streamlit" not in source, (
        "dashboard_state.py must NOT import streamlit"
    )
    assert "from streamlit" not in source, (
        "dashboard_state.py must NOT import streamlit"
    )


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

from readcoach.dashboard_state import build_dashboard_state  # noqa: E402


# ---------------------------------------------------------------------------
# Full state from real artifacts
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_state() -> dict:
    return build_dashboard_state(RESULTS_DIR, db_path=None)


# --- detector ---

def test_detector_available(full_state: dict) -> None:
    assert full_state["detector"]["available"] is True


def test_sub_recall_none(full_state: dict) -> None:
    """substitution recall @ none-bias == 0.6667 (2/3)."""
    val = full_state["detector"]["sub_recall_none"]
    assert val == pytest.approx(0.6666666666666666, rel=1e-6)


def test_fp_per_100_strong(full_state: dict) -> None:
    """fp_per_100 @ strong bias == 0.542."""
    val = full_state["detector"]["fp_per_100_strong"]
    assert val == pytest.approx(0.5421523448088913, rel=1e-6)


def test_detector_has_three_biases(full_state: dict) -> None:
    biases = full_state["detector"]["biases"]
    assert set(biases) == {"none", "prompt", "strong"}


def test_detector_all_classes_present(full_state: dict) -> None:
    classes = set(full_state["detector"]["per_class_per_bias"].keys())
    expected = {"substitution", "omission", "insertion", "self_correction", "hesitation"}
    assert classes == expected


# --- masking ---

def test_masking_available(full_state: dict) -> None:
    assert full_state["masking"]["available"] is True


def test_masking_has_three_bias_curves(full_state: dict) -> None:
    curve = full_state["masking"]["curve_points"]
    assert set(curve.keys()) == {"none", "prompt", "strong"}


def test_masking_fp_per_100_none_matches(full_state: dict) -> None:
    """fp_per_100 @ none-bias matches masking_curve.json."""
    val = full_state["masking"]["curve_points"]["none"]["fp_per_100_correct_words"]
    assert val == pytest.approx(7.1021957169964764, rel=1e-6)


def test_masking_ci_none_has_two_elements(full_state: dict) -> None:
    ci = full_state["masking"]["curve_points"]["none"]["ci_fp_per_100"]
    assert ci is not None and len(ci) == 2


# --- learner ---

def test_learner_available(full_state: dict) -> None:
    assert full_state["learner"]["available"] is True


def test_break_even_a(full_state: dict) -> None:
    """break_even_a == 0.9 (exact from file)."""
    val = full_state["learner"]["break_even_a"]
    assert val == pytest.approx(0.9, rel=1e-9)


def test_break_even_has_three_anchors(full_state: dict) -> None:
    anchors = full_state["learner"]["a_eff_anchors"]
    assert len(anchors) == 3


def test_bkt_brier_score(full_state: dict) -> None:
    """Brier score from bkt_recovery.json."""
    val = full_state["learner"]["bkt_brier_score"]
    assert val == pytest.approx(0.13900902846707805, rel=1e-6)


# --- tutor ---

def test_tutor_available(full_state: dict) -> None:
    assert full_state["tutor"]["available"] is True


def test_wait_rate(full_state: dict) -> None:
    """WAIT-rate from policy_replay.json."""
    val = full_state["tutor"]["wait_rate"]
    assert val == pytest.approx(0.4351851851851852, rel=1e-6)


def test_ab_gate_v1_v2_passed(full_state: dict) -> None:
    """v1 vs v2 gate must pass (invariants.violations == 0 both)."""
    outcome = full_state["tutor"]["ab_gate_outcomes"]["v1_vs_v2"]
    assert outcome["passed"] is True


def test_ab_gate_v2_v3_blocked(full_state: dict) -> None:
    """v2 vs v3 gate must be blocked (v3 has 30 violations)."""
    outcome = full_state["tutor"]["ab_gate_outcomes"]["v2_vs_v3"]
    assert outcome["passed"] is False


def test_naive_live_violations(full_state: dict) -> None:
    """Total live naive violations across all profiles."""
    # 23 + 23 + 23 = 69 from the committed naive_live_audit.json
    val = full_state["tutor"]["naive_live_violations"]
    assert val == 69


def test_naive_stub_violations(full_state: dict) -> None:
    """Total stub naive violations across all profiles."""
    # 41 + 53 + 47 = 141 from the committed naive_stub_audit.json
    val = full_state["tutor"]["naive_stub_violations"]
    assert val == 141


def test_move_distribution_has_wait(full_state: dict) -> None:
    dist = full_state["tutor"]["move_distribution"]
    assert "WAIT" in dist
    assert dist["WAIT"] == 94


# --- memory ---

def test_memory_available(full_state: dict) -> None:
    assert full_state["memory"]["available"] is True


def test_learnermem_consistency_score(full_state: dict) -> None:
    """Consistency score == 1.0 (all 6 probes pass)."""
    val = full_state["memory"]["consistency_score"]
    assert val == pytest.approx(1.0, rel=1e-9)


def test_learnermem_n_probes(full_state: dict) -> None:
    assert full_state["memory"]["n_passed"] == 6
    assert full_state["memory"]["n_total"] == 6


def test_learnermem_probe_ids(full_state: dict) -> None:
    probe_ids = set(full_state["memory"]["probes"].keys())
    assert probe_ids == {"P1", "P2", "P3", "P4", "P5", "P6"}


# --- flywheel ---

def test_flywheel_available(full_state: dict) -> None:
    assert full_state["flywheel"]["available"] is True


def test_flywheel_baseline_version(full_state: dict) -> None:
    assert full_state["flywheel"]["baseline_version"] == "v0"


def test_flywheel_promote_cumulative(full_state: dict) -> None:
    """v3 promoted 30 golden failures; cumulative == 30."""
    val = full_state["flywheel"]["promote_cumulative_total"]
    assert val == 30


# ---------------------------------------------------------------------------
# Missing-file / absent-section behavior
# ---------------------------------------------------------------------------

def test_missing_miscue_file_returns_unavailable(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    state = build_dashboard_state(empty_dir)
    assert state["detector"]["available"] is False
    assert "reason" in state["detector"]


def test_missing_masking_file_returns_unavailable(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty2"
    empty_dir.mkdir()
    state = build_dashboard_state(empty_dir)
    assert state["masking"]["available"] is False


def test_missing_break_even_and_bkt_returns_unavailable(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty3"
    empty_dir.mkdir()
    state = build_dashboard_state(empty_dir)
    assert state["learner"]["available"] is False


def test_missing_tutor_files_returns_unavailable(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty4"
    empty_dir.mkdir()
    state = build_dashboard_state(empty_dir)
    assert state["tutor"]["available"] is False


def test_missing_learnermem_returns_unavailable(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty5"
    empty_dir.mkdir()
    state = build_dashboard_state(empty_dir)
    assert state["memory"]["available"] is False


def test_missing_flywheel_files_returns_unavailable(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty6"
    empty_dir.mkdir()
    state = build_dashboard_state(empty_dir)
    assert state["flywheel"]["available"] is False


def test_no_live_db_key_when_not_provided(full_state: dict) -> None:
    assert "live_db" not in full_state


def test_unavailable_section_has_no_fake_numbers(tmp_path: Path) -> None:
    """When a file is missing, no numeric fields appear — only available+reason."""
    empty_dir = tmp_path / "nofake"
    empty_dir.mkdir()
    state = build_dashboard_state(empty_dir)
    for section_name in ("detector", "masking", "memory"):
        sec = state[section_name]
        assert sec["available"] is False
        assert isinstance(sec["reason"], str)
        # Ensure no numeric data sneaks in
        for key, val in sec.items():
            if key in ("available", "reason"):
                continue
            assert not isinstance(val, (int, float)), (
                f"section {section_name!r} key {key!r} has numeric value {val!r} "
                "despite being unavailable"
            )
