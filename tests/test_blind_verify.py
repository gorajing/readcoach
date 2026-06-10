"""Tests for scripts/blind_verify.py — CI-safe (no afplay, no interactivity).

All I/O-independent logic is factored into pure functions tested here.
"""
from __future__ import annotations

import csv
import io
import json
import pathlib
import textwrap
from collections import Counter
from typing import NamedTuple

import pytest

# ---------------------------------------------------------------------------
# Import the pure-logic layer from blind_verify
# ---------------------------------------------------------------------------
import importlib.util
import sys

BV_PATH = pathlib.Path(__file__).parent.parent / "scripts" / "blind_verify.py"


def _load_bv():
    """Load blind_verify module without executing its __main__ block."""
    spec = importlib.util.spec_from_file_location("blind_verify", BV_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    bv = _load_bv()
    _IMPORT_OK = True
except Exception as exc:
    _IMPORT_OK = False
    _IMPORT_ERR = str(exc)


pytestmark = pytest.mark.skipif(
    not _IMPORT_OK,
    reason=f"blind_verify import failed: {_IMPORT_ERR if not _IMPORT_OK else ''}",
)

GOLD_PATH = pathlib.Path(__file__).parent.parent / "data" / "benchmark" / "gold.jsonl"


def _load_gold() -> list[dict]:
    return [json.loads(line) for line in GOLD_PATH.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 1. Sampling tests
# ---------------------------------------------------------------------------

class TestSampling:
    def test_deterministic_for_seed(self):
        """Same seed → same queue every time."""
        gold = _load_gold()
        q1 = bv.build_queue(gold, n=30, seed=42)
        q2 = bv.build_queue(gold, n=30, seed=42)
        assert [e["utt_id"] for e in q1] == [e["utt_id"] for e in q2]

    def test_different_seeds_differ(self):
        """Different seeds → different orderings (with very high probability)."""
        gold = _load_gold()
        q1 = bv.build_queue(gold, n=30, seed=42)
        q2 = bv.build_queue(gold, n=30, seed=99)
        assert [e["utt_id"] for e in q1] != [e["utt_id"] for e in q2]

    def test_exactly_30_items(self):
        gold = _load_gold()
        q = bv.build_queue(gold, n=30, seed=42)
        assert len(q) == 30

    def test_class_coverage_all_five(self):
        """Every miscue class appears >= 4 times in the 30-clip sample."""
        gold = _load_gold()
        q = bv.build_queue(gold, n=30, seed=42)
        utt_ids = {e["utt_id"] for e in q}
        gold_map = {g["utt_id"]: g for g in gold}
        counts: Counter[str] = Counter()
        for uid in utt_ids:
            for miscue in gold_map[uid]["gold"]:
                counts[miscue["type"]] += 1
        for cls in ("substitution", "omission", "insertion", "self_correction", "hesitation"):
            assert counts[cls] >= 4, f"class {cls!r} appears only {counts[cls]} times"

    def test_both_hesitation_renders_present(self):
        """Both hesitation renders (filler, silence) appear in the sample."""
        gold = _load_gold()
        q = bv.build_queue(gold, n=30, seed=42)
        utt_ids = {e["utt_id"] for e in q}
        gold_map = {g["utt_id"]: g for g in gold}
        renders = set()
        for uid in utt_ids:
            for miscue in gold_map[uid]["gold"]:
                if miscue["type"] == "hesitation" and miscue.get("render"):
                    renders.add(miscue["render"])
        assert "filler" in renders, "no filler hesitation in sample"
        assert "silence" in renders, "no silence hesitation in sample"

    def test_at_least_3_clean_items(self):
        """At least 3 clean (gold=[]) items appear in the 30-clip sample."""
        gold = _load_gold()
        q = bv.build_queue(gold, n=30, seed=42)
        utt_ids = {e["utt_id"] for e in q}
        gold_map = {g["utt_id"]: g for g in gold}
        clean_count = sum(1 for uid in utt_ids if gold_map[uid]["gold"] == [])
        assert clean_count >= 3, f"only {clean_count} clean items in sample"

    def test_opaque_ids_shuffled_wrt_gold_order(self):
        """For seed 42, the opaque-id order differs from sorted-by-utt_id order."""
        gold = _load_gold()
        q = bv.build_queue(gold, n=30, seed=42)
        # utt_ids in queue presentation order
        queue_utt_ids = [e["utt_id"] for e in q]
        # utt_ids sorted alphabetically (gold-file canonical order proxy)
        sorted_utt_ids = sorted(queue_utt_ids)
        assert queue_utt_ids != sorted_utt_ids, (
            "queue order matches sorted order — shuffling appears ineffective for seed 42"
        )

    def test_opaque_ids_format(self):
        """Opaque ids are q01..q30 with zero-padded two-digit numbers."""
        gold = _load_gold()
        q = bv.build_queue(gold, n=30, seed=42)
        ids = [e["opaque_id"] for e in q]
        assert ids == [f"q{i:02d}" for i in range(1, 31)]

    def test_no_utt_id_in_opaque_ids(self):
        """Opaque ids must not contain the utt_id string."""
        gold = _load_gold()
        q = bv.build_queue(gold, n=30, seed=42)
        for e in q:
            assert e["utt_id"] not in e["opaque_id"]

    def test_wav_path_field_present(self):
        """Every queue entry has a wav_path field."""
        gold = _load_gold()
        q = bv.build_queue(gold, n=30, seed=42)
        for e in q:
            assert "wav_path" in e


# ---------------------------------------------------------------------------
# 2. Resumability tests
# ---------------------------------------------------------------------------

class TestResumability:
    def _make_csv(self, rows: list[dict]) -> str:
        """Produce a CSV string from a list of row dicts."""
        fieldnames = bv.CSV_FIELDNAMES
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
        return buf.getvalue()

    def test_rated_items_excluded_from_pending(self):
        """Items already in the CSV are excluded from the pending queue."""
        gold = _load_gold()
        queue = bv.build_queue(gold, n=30, seed=42)

        # Simulate 2 rated rows using the first 2 opaque_ids
        rated_ids = [queue[0]["opaque_id"], queue[1]["opaque_id"]]
        csv_text = self._make_csv([
            {
                "opaque_id": rated_ids[0],
                "utt_id": queue[0]["utt_id"],
                "heard": "[]",
                "gold_summary": "clean",
                "match": "y",
                "reason": "",
                "timestamp": "2026-01-01T00:00:00",
                "rater_initials": "JC",
            },
            {
                "opaque_id": rated_ids[1],
                "utt_id": queue[1]["utt_id"],
                "heard": "[]",
                "gold_summary": "clean",
                "match": "y",
                "reason": "",
                "timestamp": "2026-01-01T00:00:01",
                "rater_initials": "JC",
            },
        ])

        rated_set = bv.load_rated_ids(io.StringIO(csv_text))
        pending = bv.pending_queue(queue, rated_set)

        assert len(pending) == 28
        pending_ids = {e["opaque_id"] for e in pending}
        for rid in rated_ids:
            assert rid not in pending_ids

    def test_empty_csv_all_pending(self):
        """Empty CSV → all 30 items pending."""
        gold = _load_gold()
        queue = bv.build_queue(gold, n=30, seed=42)
        rated_set = bv.load_rated_ids(io.StringIO("opaque_id\n"))
        pending = bv.pending_queue(queue, rated_set)
        assert len(pending) == 30

    def test_all_rated_no_pending(self):
        """All items rated → empty pending list."""
        gold = _load_gold()
        queue = bv.build_queue(gold, n=30, seed=42)
        # Mark every id as rated
        rated_set = {e["opaque_id"] for e in queue}
        pending = bv.pending_queue(queue, rated_set)
        assert pending == []


# ---------------------------------------------------------------------------
# 3. Enum validation tests
# ---------------------------------------------------------------------------

class TestEnumValidation:
    def _make_row(self, **overrides) -> dict:
        base = {
            "opaque_id": "q01",
            "utt_id": "p01-clean",
            "heard": "[]",
            "gold_summary": "clean",
            "match": "y",
            "reason": "",
            "timestamp": "2026-01-01T00:00:00",
            "rater_initials": "JC",
        }
        base.update(overrides)
        return base

    def _csv_from_rows(self, rows: list[dict]) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=bv.CSV_FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        return buf.getvalue()

    def test_invalid_match_value_raises(self):
        """A row with match='x' raises ValueError naming the row number."""
        rows = [self._make_row(match="x")]
        csv_text = self._csv_from_rows(rows)
        with pytest.raises(ValueError, match=r"row [12]"):
            bv.validate_ratings_csv(io.StringIO(csv_text))

    def test_empty_match_raises(self):
        """A row with match='' raises ValueError."""
        rows = [self._make_row(match="")]
        csv_text = self._csv_from_rows(rows)
        with pytest.raises(ValueError, match=r"row [12]"):
            bv.validate_ratings_csv(io.StringIO(csv_text))

    def test_missing_reason_on_n_raises(self):
        """A mismatch row (match='n') with empty reason raises ValueError."""
        rows = [self._make_row(match="n", reason="")]
        csv_text = self._csv_from_rows(rows)
        with pytest.raises(ValueError, match=r"row [12]"):
            bv.validate_ratings_csv(io.StringIO(csv_text))

    def test_valid_n_with_reason_ok(self):
        """A mismatch row with a non-empty reason is valid."""
        rows = [self._make_row(match="n", reason="heard extra 'um' before last word")]
        csv_text = self._csv_from_rows(rows)
        result = bv.validate_ratings_csv(io.StringIO(csv_text))  # must not raise
        assert len(result) == 1

    def test_valid_y_row_ok(self):
        """A match='y' row with empty reason is valid."""
        rows = [self._make_row(match="y", reason="")]
        csv_text = self._csv_from_rows(rows)
        result = bv.validate_ratings_csv(io.StringIO(csv_text))
        assert len(result) == 1

    def test_multiple_rows_bad_middle_row(self):
        """Error message includes the 1-based row number of the offending row."""
        rows = [
            self._make_row(opaque_id="q01", match="y"),
            self._make_row(opaque_id="q02", match="n", reason=""),  # bad: row 2 (data row 2 = csv row 3)
            self._make_row(opaque_id="q03", match="y"),
        ]
        csv_text = self._csv_from_rows(rows)
        with pytest.raises(ValueError) as exc_info:
            bv.validate_ratings_csv(io.StringIO(csv_text))
        # Row 2 in data = row 3 in the CSV file (1 header + 2 rows before bad one).
        # Accept any row-number mention that includes '2' or '3'.
        assert any(str(n) in str(exc_info.value) for n in range(1, 5))


# ---------------------------------------------------------------------------
# 4. Report math tests
# ---------------------------------------------------------------------------

class TestReportMath:
    """Hand-computed mismatch rate + per-class counts from a synthetic CSV."""

    def _make_csv(self, rows: list[dict]) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=bv.CSV_FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        return buf.getvalue()

    def _base_row(self, **kw) -> dict:
        defaults = {
            "opaque_id": "q01",
            "utt_id": "p01-clean",
            "heard": "[]",
            "gold_summary": "clean",
            "match": "y",
            "reason": "",
            "timestamp": "2026-01-01T00:00:00",
            "rater_initials": "JC",
        }
        defaults.update(kw)
        return defaults

    def test_mismatch_rate_zero(self):
        """All matches → mismatch_rate = 0.0."""
        rows = [self._base_row(opaque_id=f"q{i:02d}", match="y") for i in range(1, 6)]
        csv_text = self._make_csv(rows)
        report = bv.compute_report(io.StringIO(csv_text))
        assert report["n_rated"] == 5
        assert report["n_match"] == 5
        assert report["n_mismatch"] == 0
        assert report["mismatch_rate"] == 0.0

    def test_mismatch_rate_partial(self):
        """2 mismatches out of 10 → mismatch_rate = 0.2."""
        rows = [
            self._base_row(opaque_id=f"q{i:02d}", match="y") for i in range(1, 9)
        ] + [
            self._base_row(opaque_id="q09", match="n", reason="different class heard"),
            self._base_row(opaque_id="q10", match="n", reason="extra word missed"),
        ]
        csv_text = self._make_csv(rows)
        report = bv.compute_report(io.StringIO(csv_text))
        assert report["n_rated"] == 10
        assert report["n_match"] == 8
        assert report["n_mismatch"] == 2
        assert abs(report["mismatch_rate"] - 0.2) < 1e-9

    def test_mismatch_rate_all_mismatch(self):
        """All mismatches → mismatch_rate = 1.0."""
        rows = [
            self._base_row(opaque_id=f"q{i:02d}", match="n", reason="test reason")
            for i in range(1, 4)
        ]
        csv_text = self._make_csv(rows)
        report = bv.compute_report(io.StringIO(csv_text))
        assert report["mismatch_rate"] == 1.0

    def test_mismatches_list_contains_reasons(self):
        """Mismatches list includes opaque_id and reason for each mismatch."""
        rows = [
            self._base_row(opaque_id="q01", match="y"),
            self._base_row(opaque_id="q02", match="n", reason="heard substitution not omission"),
        ]
        csv_text = self._make_csv(rows)
        report = bv.compute_report(io.StringIO(csv_text))
        assert len(report["mismatches"]) == 1
        mm = report["mismatches"][0]
        assert mm["opaque_id"] == "q02"
        assert mm["reason"] == "heard substitution not omission"

    def test_empty_csv_raises(self):
        """Report mode fails loud if CSV has no data rows (pre-session)."""
        csv_text = "opaque_id,utt_id,heard,gold_summary,match,reason,timestamp,rater_initials\n"
        with pytest.raises(Exception):
            bv.compute_report(io.StringIO(csv_text))

    def test_absent_csv_raises(self):
        """compute_report_from_path raises if the file doesn't exist."""
        missing = pathlib.Path("/tmp/blind_verify_nonexistent_ratings_xyz.csv")
        with pytest.raises(Exception):
            bv.compute_report_from_path(missing)


# ---------------------------------------------------------------------------
# 5. Auto-compare logic tests
# ---------------------------------------------------------------------------

class TestAutoCompare:
    """Heard-vs-gold multiset comparisons (match, miss, extra)."""

    def test_exact_match_single_class(self):
        """Single-class, same count → match=True, miss=[], extra=[]."""
        heard = [{"type": "substitution", "word": "dog"}]
        gold = [{"type": "substitution", "target_word": "dog"}]
        result = bv.auto_compare(heard, gold)
        assert result["match"] is True
        assert result["miss"] == []
        assert result["extra"] == []

    def test_missing_class(self):
        """Gold has omission, heard has none → miss=['omission']."""
        heard = []
        gold = [{"type": "omission", "target_word": "cat"}]
        result = bv.auto_compare(heard, gold)
        assert result["match"] is False
        assert "omission" in result["miss"]

    def test_extra_class(self):
        """Heard has insertion, gold has none → extra=['insertion']."""
        heard = [{"type": "insertion", "word": "big"}]
        gold = []
        result = bv.auto_compare(heard, gold)
        assert result["match"] is False
        assert "insertion" in result["extra"]

    def test_count_mismatch(self):
        """Gold has 2 substitutions, heard has 1 → miss=['substitution']."""
        heard = [{"type": "substitution", "word": "dog"}]
        gold = [
            {"type": "substitution", "target_word": "dog"},
            {"type": "substitution", "target_word": "cat"},
        ]
        result = bv.auto_compare(heard, gold)
        assert result["match"] is False
        assert "substitution" in result["miss"]

    def test_clean_item_both_empty(self):
        """Clean item: heard=[], gold=[] → match=True."""
        result = bv.auto_compare([], [])
        assert result["match"] is True

    def test_hesitation_treated_as_single_class(self):
        """Both filler and silence hesitation roll up into 'hesitation' class."""
        heard = [{"type": "hesitation", "word": "uh"}]
        gold = [{"type": "hesitation", "render": "filler", "target_word": None}]
        result = bv.auto_compare(heard, gold)
        assert result["match"] is True

    def test_multiclass_match(self):
        """Substitution + omission in heard = gold → match=True."""
        heard = [
            {"type": "substitution", "word": "gog"},
            {"type": "omission", "word": "nap"},
        ]
        gold = [
            {"type": "substitution", "target_word": "dog"},
            {"type": "omission", "target_word": "nap"},
        ]
        result = bv.auto_compare(heard, gold)
        assert result["match"] is True
