"""Integration tests for scripts/label_app.py — CI-safe, no Streamlit import.

Tests:
1. No Streamlit import in shared / pure modules used by the app.
2. The app's submit path (shared writer called the way the app calls it)
   produces rows that validate_judge._load_labels can parse.
"""
from __future__ import annotations

import importlib.util
import pathlib

# ---------------------------------------------------------------------------
# Load modules under test
# ---------------------------------------------------------------------------

_ROOT = pathlib.Path(__file__).parent.parent
_LT_PATH = _ROOT / "scripts" / "label_turns.py"
_VJ_PATH = _ROOT / "scripts" / "validate_judge.py"
_APP_PATH = _ROOT / "scripts" / "label_app.py"


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lt = _load_module("label_turns", _LT_PATH)
vj = _load_module("validate_judge", _VJ_PATH)


# ---------------------------------------------------------------------------
# 1. Streamlit isolation — label_turns.py must not import streamlit
# ---------------------------------------------------------------------------


def test_no_streamlit_in_label_turns() -> None:
    """label_turns.py must not import streamlit (pure logic only)."""
    source = _LT_PATH.read_text(encoding="utf-8")
    assert "import streamlit" not in source, (
        "label_turns.py must NOT import streamlit"
    )
    assert "from streamlit" not in source, (
        "label_turns.py must NOT import streamlit"
    )


# ---------------------------------------------------------------------------
# 2. Submit-path integration: app writer → validate_judge parser
# ---------------------------------------------------------------------------


class TestAppSubmitPath:
    """Simulate exactly what label_app.py does on submit."""

    def _simulate_submit(
        self,
        csv_path: pathlib.Path,
        turn_id: str,
        dim: str,
        score: int,
        borderline: str | None,
        initials: str = "JC",
    ) -> None:
        """Replicate the app's submit logic verbatim."""
        human_passing = lt.derive_passing(score, borderline)
        row = {
            "turn_id": turn_id,
            "dimension": dim,
            "human_score": str(score),
            "human_passing": human_passing,
            "rater_initials": initials,
        }
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        lt._append_label_row(csv_path, row)

    def test_passing_score_4_writes_y(self, tmp_path: pathlib.Path) -> None:
        csv_path = tmp_path / "turn_labels.csv"
        self._simulate_submit(csv_path, "sd-t00", "guidance", 4, None)
        labels = vj._load_labels(csv_path)
        assert labels[("sd-t00", "guidance")] is True

    def test_failing_score_2_writes_n(self, tmp_path: pathlib.Path) -> None:
        csv_path = tmp_path / "turn_labels.csv"
        self._simulate_submit(csv_path, "sd-t00", "actionability", 2, None)
        labels = vj._load_labels(csv_path)
        assert labels[("sd-t00", "actionability")] is False

    def test_borderline_score_3_y_writes_y(self, tmp_path: pathlib.Path) -> None:
        csv_path = tmp_path / "turn_labels.csv"
        self._simulate_submit(csv_path, "fh-t01", "icap", 3, "y")
        labels = vj._load_labels(csv_path)
        assert labels[("fh-t01", "icap")] is True

    def test_borderline_score_3_n_writes_n(self, tmp_path: pathlib.Path) -> None:
        csv_path = tmp_path / "turn_labels.csv"
        self._simulate_submit(csv_path, "fh-t01", "icap", 3, "n")
        labels = vj._load_labels(csv_path)
        assert labels[("fh-t01", "icap")] is False

    def test_three_dimensions_full_turn(self, tmp_path: pathlib.Path) -> None:
        """Submit all 3 dims for one turn; validate_judge parses all 3."""
        csv_path = tmp_path / "turn_labels.csv"
        self._simulate_submit(csv_path, "sc-t05", "guidance", 5, None)
        self._simulate_submit(csv_path, "sc-t05", "actionability", 4, None)
        self._simulate_submit(csv_path, "sc-t05", "icap", 3, "y")
        labels = vj._load_labels(csv_path)
        assert labels[("sc-t05", "guidance")] is True
        assert labels[("sc-t05", "actionability")] is True
        assert labels[("sc-t05", "icap")] is True

    def test_validate_judge_parses_all_five_scores(self, tmp_path: pathlib.Path) -> None:
        """All scores 1-5 produce valid rows parseable by validate_judge."""
        csv_path = tmp_path / "turn_labels.csv"
        expected_passing = {1: False, 2: False, 3: True, 4: True, 5: True}
        for score in range(1, 6):
            borderline = "y" if score == 3 else None
            tid = f"sd-t{score:02d}"
            self._simulate_submit(csv_path, tid, "guidance", score, borderline)

        labels = vj._load_labels(csv_path)
        for score in range(1, 6):
            tid = f"sd-t{score:02d}"
            assert labels[(tid, "guidance")] is expected_passing[score], (
                f"score={score} expected passing={expected_passing[score]}"
            )

    def test_rows_survive_resume(self, tmp_path: pathlib.Path) -> None:
        """Writing two separate turns then reading labeled_pairs sees both."""
        csv_path = tmp_path / "turn_labels.csv"
        for dim in lt.DIMENSIONS:
            self._simulate_submit(csv_path, "sd-t00", dim, 4, None)
        for dim in lt.DIMENSIONS:
            self._simulate_submit(csv_path, "fh-t01", dim, 2, None)

        with csv_path.open(newline="", encoding="utf-8") as f:
            pairs = lt.load_labeled_pairs(f)

        for dim in lt.DIMENSIONS:
            assert ("sd-t00", dim) in pairs
            assert ("fh-t01", dim) in pairs
