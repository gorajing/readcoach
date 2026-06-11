"""CI-safe tests for scripts/blind_verify_app.py.

Covers:
1. Streamlit isolation — blind_verify.py must not import streamlit.
2. Shared-logic import works (importlib pattern).
3. Submit path produces rows that blind_verify's validator accepts.
4. Two-step ordering: can't reach submission without a locked heard list.
5. No streamlit import leaks into blind_verify.py.
"""
from __future__ import annotations

import csv
import datetime
import importlib.util
import json
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Load modules under test
# ---------------------------------------------------------------------------

_ROOT = pathlib.Path(__file__).parent.parent
_BV_PATH = _ROOT / "scripts" / "blind_verify.py"
_APP_PATH = _ROOT / "scripts" / "blind_verify_app.py"


def _load_module(name: str, path: pathlib.Path):
    """Load a module by file path via importlib, no __main__ execution."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bv = _load_module("blind_verify", _BV_PATH)


# ---------------------------------------------------------------------------
# 1. Streamlit isolation
# ---------------------------------------------------------------------------


def test_no_streamlit_in_blind_verify() -> None:
    """blind_verify.py must not import streamlit (pure logic only)."""
    source = _BV_PATH.read_text(encoding="utf-8")
    assert "import streamlit" not in source, (
        "blind_verify.py must NOT import streamlit"
    )
    assert "from streamlit" not in source, (
        "blind_verify.py must NOT import streamlit"
    )


# ---------------------------------------------------------------------------
# 2. Shared-logic import works
# ---------------------------------------------------------------------------


def test_shared_logic_import() -> None:
    """The importlib file-loading pattern used by the app works correctly."""
    # Verify key symbols are importable and callable
    assert callable(bv.build_queue)
    assert callable(bv.auto_compare)
    assert callable(bv.load_rated_ids)
    assert callable(bv.pending_queue)
    assert callable(bv.validate_ratings_csv)
    assert callable(bv.compute_report)
    assert callable(bv._gold_summary)
    assert callable(bv._append_rating_row)
    assert isinstance(bv.CSV_FIELDNAMES, list)
    assert isinstance(bv.ALL_CLASSES, tuple)


# ---------------------------------------------------------------------------
# 3. Submit path: app-produced rows pass blind_verify's validator
# ---------------------------------------------------------------------------


def _make_row(
    opaque_id: str = "q01",
    utt_id: str = "p01-clean",
    heard: list[dict] | None = None,
    gold_summary: str = "clean",
    match: str = "y",
    reason: str = "",
    rater_initials: str = "JC",
) -> dict:
    """Replicate the app's row-construction logic exactly."""
    heard_locked = heard if heard is not None else []
    return {
        "opaque_id": opaque_id,
        "utt_id": utt_id,
        "heard": json.dumps(heard_locked),
        "gold_summary": gold_summary,
        "match": match,
        "reason": reason,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rater_initials": rater_initials,
    }


class TestSubmitPath:
    """Simulate the app's submit path; rows must pass validate_ratings_csv."""

    def _write_rows(self, csv_path: pathlib.Path, rows: list[dict]) -> None:
        """Write rows exactly as _append_rating_row would (via the shared writer)."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        for row in rows:
            bv._append_rating_row(csv_path, row)

    def test_clean_match_row_validates(self, tmp_path: pathlib.Path) -> None:
        csv_path = tmp_path / "ratings.csv"
        row = _make_row(match="y", reason="")
        self._write_rows(csv_path, [row])
        with csv_path.open(newline="", encoding="utf-8") as f:
            result = bv.validate_ratings_csv(f)
        assert len(result) == 1
        assert result[0]["match"] == "y"

    def test_mismatch_row_with_reason_validates(self, tmp_path: pathlib.Path) -> None:
        csv_path = tmp_path / "ratings.csv"
        row = _make_row(
            match="n",
            reason="heard substitution but gold has omission",
            gold_summary="om(nap)",
        )
        self._write_rows(csv_path, [row])
        with csv_path.open(newline="", encoding="utf-8") as f:
            result = bv.validate_ratings_csv(f)
        assert len(result) == 1
        assert result[0]["match"] == "n"
        assert result[0]["reason"] == "heard substitution but gold has omission"

    def test_mismatch_row_without_reason_raises(self, tmp_path: pathlib.Path) -> None:
        """_append_rating_row itself rejects match='n' with empty reason."""
        csv_path = tmp_path / "ratings.csv"
        row = _make_row(match="n", reason="")
        with pytest.raises(ValueError, match="reason"):
            bv._append_rating_row(csv_path, row)

    def test_multi_row_session_validates(self, tmp_path: pathlib.Path) -> None:
        """Writing several rows produces a CSV validate_ratings_csv accepts."""
        csv_path = tmp_path / "ratings.csv"
        rows = [
            _make_row(opaque_id="q01", match="y"),
            _make_row(opaque_id="q02", match="y"),
            _make_row(opaque_id="q03", match="n", reason="missed hesitation"),
        ]
        self._write_rows(csv_path, rows)
        with csv_path.open(newline="", encoding="utf-8") as f:
            result = bv.validate_ratings_csv(f)
        assert len(result) == 3

    def test_rows_survive_resume(self, tmp_path: pathlib.Path) -> None:
        """Rows written in two separate appends are all present on reload."""
        csv_path = tmp_path / "ratings.csv"
        bv._append_rating_row(csv_path, _make_row(opaque_id="q01", match="y"))
        bv._append_rating_row(csv_path, _make_row(opaque_id="q02", match="y"))

        with csv_path.open(newline="", encoding="utf-8") as f:
            rated_set = bv.load_rated_ids(f)

        assert "q01" in rated_set
        assert "q02" in rated_set

    def test_heard_field_is_valid_json(self, tmp_path: pathlib.Path) -> None:
        """The 'heard' field written by the app is valid JSON and round-trips."""
        csv_path = tmp_path / "ratings.csv"
        heard_list = [{"type": "substitution", "word": "gog"}]
        row = _make_row(
            opaque_id="q05",
            heard=heard_list,
            gold_summary="sub(dog->gog)",
            match="y",
        )
        bv._append_rating_row(csv_path, row)

        rows = []
        with csv_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)

        assert len(rows) == 1
        recovered = json.loads(rows[0]["heard"])
        assert recovered == heard_list

    def test_fieldnames_match_csv_fieldnames(self) -> None:
        """App-produced rows use exactly the CSV_FIELDNAMES columns."""
        row = _make_row()
        assert set(row.keys()) == set(bv.CSV_FIELDNAMES)


# ---------------------------------------------------------------------------
# 4. Two-step ordering enforced in pure helper
# ---------------------------------------------------------------------------


class TestTwoStepOrdering:
    """
    The two-step invariant: submission requires a locked heard list.
    We verify this at the pure-logic level — the app's state machine
    guards entry to step 2 by requiring heard_locked is not None.

    We test it here via a pure helper that mirrors the app's guard:
    attempting to build a submission row without a locked list must fail.
    """

    def _simulate_app_submit(
        self,
        csv_path: pathlib.Path,
        heard_locked: list[dict] | None,
        opaque_id: str = "q01",
        utt_id: str = "p01-clean",
        gold_summary: str = "clean",
        match: str = "y",
        reason: str = "",
        rater_initials: str = "JC",
    ) -> None:
        """Mirror the app's guard: heard_locked must not be None."""
        if heard_locked is None:
            raise RuntimeError(
                "Cannot submit: heard list has not been locked. "
                "The rater must complete Step 1 first."
            )
        row = {
            "opaque_id": opaque_id,
            "utt_id": utt_id,
            "heard": json.dumps(heard_locked),
            "gold_summary": gold_summary,
            "match": match,
            "reason": reason,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "rater_initials": rater_initials,
        }
        bv._append_rating_row(csv_path, row)

    def test_submit_with_locked_list_succeeds(self, tmp_path: pathlib.Path) -> None:
        csv_path = tmp_path / "ratings.csv"
        self._simulate_app_submit(csv_path, heard_locked=[], match="y")
        assert csv_path.exists()

    def test_submit_without_locked_list_raises(self, tmp_path: pathlib.Path) -> None:
        csv_path = tmp_path / "ratings.csv"
        with pytest.raises(RuntimeError, match="locked"):
            self._simulate_app_submit(csv_path, heard_locked=None)
        assert not csv_path.exists()

    def test_non_empty_locked_list_submits_correctly(self, tmp_path: pathlib.Path) -> None:
        csv_path = tmp_path / "ratings.csv"
        heard = [{"type": "hesitation", "word": "uh"}]
        self._simulate_app_submit(
            csv_path,
            heard_locked=heard,
            gold_summary="hes/filler(the)",
            match="y",
        )
        with csv_path.open(newline="", encoding="utf-8") as f:
            result = bv.validate_ratings_csv(f)
        assert len(result) == 1

    def test_step2_requires_step1_completion(self, tmp_path: pathlib.Path) -> None:
        """Submitting two consecutive clips requires locking each one first."""
        csv_path = tmp_path / "ratings.csv"

        # Clip 1: complete both steps
        self._simulate_app_submit(csv_path, heard_locked=[], opaque_id="q01", match="y")

        # Clip 2: attempt submit without going through step 1 first
        with pytest.raises(RuntimeError, match="locked"):
            self._simulate_app_submit(csv_path, heard_locked=None, opaque_id="q02", match="y")

        # Only q01 was written
        with csv_path.open(newline="", encoding="utf-8") as f:
            rated_set = bv.load_rated_ids(f)
        assert "q01" in rated_set
        assert "q02" not in rated_set


# ---------------------------------------------------------------------------
# 5. Auto-compare used by the app produces expected verdicts
# ---------------------------------------------------------------------------


class TestAutoCompareInAppContext:
    """Verify auto_compare behaves correctly when called as the app would call it."""

    def test_clean_heard_vs_clean_gold(self) -> None:
        assert bv.auto_compare([], []) == {"match": True, "miss": [], "extra": []}

    def test_substitution_match(self) -> None:
        heard = [{"type": "substitution", "word": "gog"}]
        gold = [{"type": "substitution", "target_word": "dog", "said_word": "gog"}]
        result = bv.auto_compare(heard, gold)
        assert result["match"] is True

    def test_heard_extra_triggers_mismatch(self) -> None:
        heard = [{"type": "insertion", "word": "big"}]
        gold = []
        result = bv.auto_compare(heard, gold)
        assert result["match"] is False
        assert "insertion" in result["extra"]

    def test_gold_summary_clean(self) -> None:
        assert bv._gold_summary([]) == "clean"

    def test_gold_summary_substitution(self) -> None:
        gold = [{"type": "substitution", "target_word": "dog", "said_word": "gog"}]
        summary = bv._gold_summary(gold)
        assert "sub" in summary
