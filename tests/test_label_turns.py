"""Tests for scripts/label_turns.py — CI-safe (no interactivity, no I/O).

Coverage:
  1. Sampling determinism + stratification
  2. CSV format exactly matches validate_judge.py's _load_labels parser
     (integration: write with label_turns writer, parse with validate_judge loader)
  3. Resume skip-logic
  4. Passing-derivation matrix
  5. Malformed-row refusal
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import pathlib
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Load modules under test (file-based import so we don't need an __init__)
# ---------------------------------------------------------------------------

_ROOT = pathlib.Path(__file__).parent.parent
_LT_PATH = _ROOT / "scripts" / "label_turns.py"
_VJ_PATH = _ROOT / "scripts" / "validate_judge.py"


def _load_module(name: str, path: pathlib.Path):
    """Load a script as a module without executing __main__."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lt = _load_module("label_turns", _LT_PATH)
vj = _load_module("validate_judge", _VJ_PATH)

TURNS_PATH = _ROOT / "evals" / "results" / "turns_v1.jsonl"


def _load_turns() -> list[dict]:
    return [json.loads(line) for line in TURNS_PATH.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 1. Sampling: determinism + stratification
# ---------------------------------------------------------------------------


class TestSampling:
    def test_deterministic_for_seed(self):
        """Same seed → identical queue every time."""
        turns = _load_turns()
        q1 = lt.build_queue(turns, n=60, seed=42)
        q2 = lt.build_queue(turns, n=60, seed=42)
        assert [t["turn_id"] for t in q1] == [t["turn_id"] for t in q2]

    def test_different_seeds_differ(self):
        """Different seeds → different orderings (with very high probability)."""
        turns = _load_turns()
        q1 = lt.build_queue(turns, n=60, seed=42)
        q2 = lt.build_queue(turns, n=60, seed=99)
        assert [t["turn_id"] for t in q1] != [t["turn_id"] for t in q2]

    def test_exactly_60_items(self):
        turns = _load_turns()
        q = lt.build_queue(turns, n=60, seed=42)
        assert len(q) == 60

    def test_balanced_profiles(self):
        """All three profiles each appear exactly n//3 times (or ±1 for remainder)."""
        turns = _load_turns()
        n = 60
        q = lt.build_queue(turns, n=n, seed=42)
        from collections import Counter
        counts = Counter(t["profile"] for t in q)
        base = n // 3
        for profile in lt.PROFILES:
            assert counts[profile] in (base, base + 1), (
                f"Profile {profile!r} has {counts[profile]} turns, expected ~{base}"
            )

    def test_all_three_profiles_present(self):
        turns = _load_turns()
        q = lt.build_queue(turns, n=60, seed=42)
        profiles_in_queue = {t["profile"] for t in q}
        assert profiles_in_queue == set(lt.PROFILES)

    def test_mix_page_end_and_mid_page(self):
        """Queue contains both page-end and mid-page turns."""
        turns = _load_turns()
        q = lt.build_queue(turns, n=60, seed=42)
        page_end = sum(1 for t in q if t.get("at_page_end"))
        mid_page = sum(1 for t in q if not t.get("at_page_end"))
        assert page_end >= 3, f"Too few page-end turns: {page_end}"
        assert mid_page >= 3, f"Too few mid-page turns: {mid_page}"

    def test_mix_miscue_and_clean(self):
        """Queue contains both miscue and clean (no miscue) turns."""
        turns = _load_turns()
        q = lt.build_queue(turns, n=60, seed=42)
        miscue = sum(1 for t in q if t.get("miscue_type") is not None)
        clean = sum(1 for t in q if t.get("miscue_type") is None)
        assert miscue >= 5, f"Too few miscue turns: {miscue}"
        assert clean >= 5, f"Too few clean turns: {clean}"

    def test_all_turns_have_turn_id(self):
        """Every sampled turn has a non-empty turn_id field."""
        turns = _load_turns()
        q = lt.build_queue(turns, n=60, seed=42)
        for t in q:
            assert "turn_id" in t
            assert t["turn_id"]

    def test_turn_ids_unique(self):
        """All turn_ids in the queue are unique."""
        turns = _load_turns()
        q = lt.build_queue(turns, n=60, seed=42)
        ids = [t["turn_id"] for t in q]
        assert len(ids) == len(set(ids)), "Duplicate turn_ids found"

    def test_n_less_than_60_respected(self):
        """Smaller n is respected."""
        turns = _load_turns()
        q = lt.build_queue(turns, n=9, seed=42)
        assert len(q) == 9


# ---------------------------------------------------------------------------
# 2. CSV format integration: write via label_turns, parse via validate_judge
# ---------------------------------------------------------------------------


class TestCSVFormatIntegration:
    """Write rows using label_turns' writer logic; parse with validate_judge._load_labels."""

    def _write_rows_to_buffer(self, rows: list[dict]) -> io.StringIO:
        """Simulate writing label rows using the label_turns CSV schema."""
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=lt.CSV_FIELDNAMES)
        w.writeheader()
        for row in rows:
            w.writerow(row)
        buf.seek(0)
        return buf

    def _make_row(
        self,
        turn_id: str = "sd-t00",
        dimension: str = "guidance",
        human_score: int = 4,
        human_passing: str = "y",
        rater_initials: str = "JC",
    ) -> dict:
        return {
            "turn_id": turn_id,
            "dimension": dimension,
            "human_score": str(human_score),
            "human_passing": human_passing,
            "rater_initials": rater_initials,
        }

    def test_fieldnames_match_validate_judge_required_cols(self):
        """label_turns CSV_FIELDNAMES are exactly the columns validate_judge requires."""
        required = {"turn_id", "dimension", "human_score", "human_passing", "rater_initials"}
        assert set(lt.CSV_FIELDNAMES) == required

    def test_written_csv_parseable_by_validate_judge(self):
        """Rows written by label_turns can be loaded by validate_judge._load_labels."""
        rows = [
            self._make_row("sd-t00", "guidance", 5, "y"),
            self._make_row("sd-t00", "actionability", 4, "y"),
            self._make_row("sd-t00", "icap", 2, "n"),
            self._make_row("fh-t01", "guidance", 3, "n"),
        ]
        buf = self._write_rows_to_buffer(rows)
        # Write to a temp path-like object via pathlib so vj._load_labels can open it
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            f.write(buf.read())
            tmp_path = pathlib.Path(f.name)
        try:
            labels = vj._load_labels(tmp_path)
        finally:
            os.unlink(tmp_path)

        assert ("sd-t00", "guidance") in labels
        assert labels[("sd-t00", "guidance")] is True
        assert labels[("sd-t00", "icap")] is False
        assert labels[("fh-t01", "guidance")] is False  # human_passing="n"

    def test_human_passing_y_parses_true(self):
        """human_passing='y' → True via validate_judge._parse_human_passing."""
        assert vj._parse_human_passing("y") is True
        assert vj._parse_human_passing("Y") is True
        assert vj._parse_human_passing("yes") is True

    def test_human_passing_n_parses_false(self):
        """human_passing='n' → False via validate_judge._parse_human_passing."""
        assert vj._parse_human_passing("n") is False
        assert vj._parse_human_passing("N") is False
        assert vj._parse_human_passing("no") is False

    def test_180_rows_for_60_turns_3_dims(self):
        """Full labeling session produces exactly 60 × 3 = 180 rows."""
        turns = _load_turns()
        queue = lt.build_queue(turns, n=60, seed=42)
        rows = []
        for t in queue:
            for dim in lt.DIMENSIONS:
                rows.append(self._make_row(t["turn_id"], dim, 4, "y", "JC"))
        assert len(rows) == 180

    def test_round_trip_preserves_values(self):
        """Values survive a write→parse round trip without mutation."""
        rows_in = [
            self._make_row("sc-t05", "guidance", 3, "n"),
            self._make_row("sc-t05", "actionability", 5, "y"),
        ]
        buf = self._write_rows_to_buffer(rows_in)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            f.write(buf.read())
            tmp_path = pathlib.Path(f.name)
        try:
            labels = vj._load_labels(tmp_path)
        finally:
            os.unlink(tmp_path)
        # ("sc-t05", "guidance") has human_passing="n" → False
        assert labels[("sc-t05", "guidance")] is False
        # ("sc-t05", "actionability") has human_passing="y" → True
        assert labels[("sc-t05", "actionability")] is True


# ---------------------------------------------------------------------------
# 3. Resume skip-logic
# ---------------------------------------------------------------------------


class TestResumability:
    def _labeled_pairs_from_rows(self, rows: list[dict]) -> set[tuple[str, str]]:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=lt.CSV_FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        buf.seek(0)
        return lt.load_labeled_pairs(buf)

    def test_labeled_pairs_detected(self):
        """load_labeled_pairs returns the correct (turn_id, dimension) set."""
        rows = [
            {"turn_id": "sd-t00", "dimension": "guidance", "human_score": "4",
             "human_passing": "y", "rater_initials": "JC"},
            {"turn_id": "sd-t00", "dimension": "actionability", "human_score": "3",
             "human_passing": "n", "rater_initials": "JC"},
        ]
        pairs = self._labeled_pairs_from_rows(rows)
        assert ("sd-t00", "guidance") in pairs
        assert ("sd-t00", "actionability") in pairs
        assert ("sd-t00", "icap") not in pairs

    def test_empty_csv_no_pairs(self):
        """Empty CSV (header only) returns empty set."""
        buf = io.StringIO("turn_id,dimension,human_score,human_passing,rater_initials\n")
        pairs = lt.load_labeled_pairs(buf)
        assert pairs == set()

    def test_all_pairs_labeled_no_pending(self):
        """If all (turn_id, dim) pairs are labeled, no turn is pending."""
        turns = _load_turns()
        queue = lt.build_queue(turns, n=9, seed=42)
        rows = []
        for t in queue:
            for dim in lt.DIMENSIONS:
                rows.append({
                    "turn_id": t["turn_id"], "dimension": dim,
                    "human_score": "4", "human_passing": "y", "rater_initials": "JC",
                })
        labeled = self._labeled_pairs_from_rows(rows)
        pending = [
            t for t in queue
            if not all((t["turn_id"], d) in labeled for d in lt.DIMENSIONS)
        ]
        assert pending == []

    def test_partial_labels_correct_pending(self):
        """Only turns missing at least one dimension appear as pending."""
        turns = _load_turns()
        queue = lt.build_queue(turns, n=9, seed=42)
        # Label only first 3 turns fully
        fully_labeled = queue[:3]
        rows = []
        for t in fully_labeled:
            for dim in lt.DIMENSIONS:
                rows.append({
                    "turn_id": t["turn_id"], "dimension": dim,
                    "human_score": "4", "human_passing": "y", "rater_initials": "JC",
                })
        labeled = self._labeled_pairs_from_rows(rows)
        pending = [
            t for t in queue
            if not all((t["turn_id"], d) in labeled for d in lt.DIMENSIONS)
        ]
        # 9 total - 3 fully labeled = 6 pending
        assert len(pending) == 6
        pending_ids = {t["turn_id"] for t in pending}
        for t in fully_labeled:
            assert t["turn_id"] not in pending_ids

    def test_partial_dim_labeled_turn_stays_pending(self):
        """A turn with only 1/3 dims labeled still appears in pending."""
        turns = _load_turns()
        queue = lt.build_queue(turns, n=9, seed=42)
        t = queue[0]
        rows = [
            {"turn_id": t["turn_id"], "dimension": "guidance",
             "human_score": "4", "human_passing": "y", "rater_initials": "JC"},
        ]
        labeled = self._labeled_pairs_from_rows(rows)
        pending = [
            q for q in queue
            if not all((q["turn_id"], d) in labeled for d in lt.DIMENSIONS)
        ]
        assert t["turn_id"] in {q["turn_id"] for q in pending}


# ---------------------------------------------------------------------------
# 4. Passing-derivation matrix
# ---------------------------------------------------------------------------


class TestPassingDerivation:
    def test_score_5_passes(self):
        assert lt.derive_passing(5) == "y"

    def test_score_4_passes(self):
        assert lt.derive_passing(4) == "y"

    def test_score_3_with_y_passes(self):
        assert lt.derive_passing(3, "y") == "y"

    def test_score_3_with_n_fails(self):
        assert lt.derive_passing(3, "n") == "n"

    def test_score_3_without_answer_raises(self):
        with pytest.raises(ValueError):
            lt.derive_passing(3)

    def test_score_2_fails(self):
        assert lt.derive_passing(2) == "n"

    def test_score_1_fails(self):
        assert lt.derive_passing(1) == "n"

    def test_invalid_borderline_answer_raises(self):
        with pytest.raises(ValueError):
            lt.derive_passing(3, "maybe")

    def test_borderline_y_case_insensitive(self):
        assert lt.derive_passing(3, "Y") == "y"

    def test_borderline_n_case_insensitive(self):
        assert lt.derive_passing(3, "N") == "n"


# ---------------------------------------------------------------------------
# 5. Malformed-row refusal
# ---------------------------------------------------------------------------


class TestMalformedRowRefusal:
    def _good_row(self, **overrides) -> dict:
        row = {
            "turn_id": "sd-t00",
            "dimension": "guidance",
            "human_score": "4",
            "human_passing": "y",
            "rater_initials": "JC",
        }
        row.update(overrides)
        return row

    def test_valid_row_passes(self):
        lt.validate_label_row(self._good_row())  # must not raise

    def test_missing_field_raises(self):
        row = self._good_row()
        del row["human_score"]
        with pytest.raises(ValueError, match="human_score"):
            lt.validate_label_row(row)

    def test_empty_turn_id_raises(self):
        with pytest.raises(ValueError, match="turn_id"):
            lt.validate_label_row(self._good_row(turn_id=""))

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError, match="dimension"):
            lt.validate_label_row(self._good_row(dimension="spelling"))

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            lt.validate_label_row(self._good_row(human_score="6"))

    def test_score_zero_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            lt.validate_label_row(self._good_row(human_score="0"))

    def test_non_integer_score_raises(self):
        with pytest.raises(ValueError):
            lt.validate_label_row(self._good_row(human_score="3.5"))

    def test_invalid_human_passing_raises(self):
        with pytest.raises(ValueError, match="human_passing"):
            lt.validate_label_row(self._good_row(human_passing="maybe"))

    def test_empty_human_passing_raises(self):
        with pytest.raises(ValueError, match="human_passing"):
            lt.validate_label_row(self._good_row(human_passing=""))

    def test_empty_rater_initials_raises(self):
        with pytest.raises(ValueError, match="rater_initials"):
            lt.validate_label_row(self._good_row(rater_initials=""))

    def test_all_valid_dimensions_accepted(self):
        for dim in lt.DIMENSIONS:
            lt.validate_label_row(self._good_row(dimension=dim))  # must not raise

    def test_all_valid_scores_accepted(self):
        for score in range(1, 6):
            lt.validate_label_row(self._good_row(human_score=str(score)))  # must not raise

    def test_valid_human_passing_values(self):
        for val in ("y", "n"):
            lt.validate_label_row(self._good_row(human_passing=val))  # must not raise
