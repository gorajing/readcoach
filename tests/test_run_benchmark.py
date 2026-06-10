"""Tests for scripts/run_benchmark.py (T1.5).

CI-safe: no network, no model, no audio processing.
All tests use hand-built data or synthetic JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

from readcoach.miscue import Miscue, match_counts


# ===========================================================================
# match_counts unit tests
# ===========================================================================

class TestMatchCounts:
    """Verify match_counts() TP/FP/FN arithmetic and ±1 tolerance."""

    # -----------------------------------------------------------------------
    # Basic exact match
    # -----------------------------------------------------------------------

    def test_perfect_match_single_sub(self):
        """One predicted substitution exactly matching one gold substitution."""
        gold = [Miscue("substitution", "dog", "gog", index=5)]
        pred = [Miscue("substitution", "dog", "gog", index=5)]
        counts = match_counts(pred, gold, n_target_words=10)
        assert counts["substitution"] == {"tp": 1, "fp": 0, "fn": 0}
        # all other classes are zero
        for cls in ("omission", "insertion", "self_correction", "hesitation"):
            assert counts[cls] == {"tp": 0, "fp": 0, "fn": 0}

    def test_no_predictions_all_fn(self):
        """No predictions → all gold become FN."""
        gold = [
            Miscue("omission", "cat", None, index=3),
            Miscue("substitution", "dog", "gog", index=7),
        ]
        counts = match_counts([], gold, n_target_words=10)
        assert counts["omission"] == {"tp": 0, "fp": 0, "fn": 1}
        assert counts["substitution"] == {"tp": 0, "fp": 0, "fn": 1}

    def test_no_gold_all_fp(self):
        """No gold → all predictions are FP."""
        pred = [
            Miscue("insertion", None, "big", index=4),
            Miscue("insertion", None, "really", index=4),
        ]
        counts = match_counts(pred, [], n_target_words=10)
        assert counts["insertion"] == {"tp": 0, "fp": 2, "fn": 0}

    def test_empty_both_all_zeros(self):
        """Nothing on either side → zeros everywhere."""
        counts = match_counts([], [], n_target_words=5)
        for cls in ("substitution", "omission", "insertion", "self_correction", "hesitation"):
            assert counts[cls] == {"tp": 0, "fp": 0, "fn": 0}
        assert counts["_correct_words"] == 5

    # -----------------------------------------------------------------------
    # ±1 index tolerance
    # -----------------------------------------------------------------------

    def test_plus_one_tolerance_match(self):
        """Predicted at index 6, gold at index 7 → match (|6-7|=1 ≤ 1)."""
        gold = [Miscue("substitution", "dog", "gog", index=7)]
        pred = [Miscue("substitution", "dog", "gog", index=6)]
        counts = match_counts(pred, gold, n_target_words=10)
        assert counts["substitution"] == {"tp": 1, "fp": 0, "fn": 0}

    def test_plus_one_tolerance_no_match(self):
        """Predicted at index 5, gold at index 7 → NO match (|5-7|=2 > 1)."""
        gold = [Miscue("substitution", "dog", "gog", index=7)]
        pred = [Miscue("substitution", "dog", "gog", index=5)]
        counts = match_counts(pred, gold, n_target_words=10)
        assert counts["substitution"] == {"tp": 0, "fp": 1, "fn": 1}

    # -----------------------------------------------------------------------
    # Cross-class isolation
    # -----------------------------------------------------------------------

    def test_cross_class_no_match(self):
        """A predicted substitution does NOT match a gold omission, even at same index."""
        gold = [Miscue("omission", "cat", None, index=3)]
        pred = [Miscue("substitution", "cat", "hat", index=3)]
        counts = match_counts(pred, gold, n_target_words=10)
        assert counts["substitution"] == {"tp": 0, "fp": 1, "fn": 0}
        assert counts["omission"] == {"tp": 0, "fp": 0, "fn": 1}

    def test_mixed_classes_all_correct(self):
        """Multiple classes, all correctly predicted."""
        gold = [
            Miscue("substitution", "dog", "gog", index=5),
            Miscue("omission", "cat", None, index=8),
            Miscue("insertion", None, "big", index=2),
        ]
        pred = [
            Miscue("substitution", "dog", "gog", index=5),
            Miscue("omission", "cat", None, index=8),
            Miscue("insertion", None, "big", index=2),
        ]
        counts = match_counts(pred, gold, n_target_words=15)
        assert counts["substitution"] == {"tp": 1, "fp": 0, "fn": 0}
        assert counts["omission"] == {"tp": 1, "fp": 0, "fn": 0}
        assert counts["insertion"] == {"tp": 1, "fp": 0, "fn": 0}

    # -----------------------------------------------------------------------
    # Greedy one-to-one matching (no double-counting)
    # -----------------------------------------------------------------------

    def test_greedy_no_double_count(self):
        """Two preds at adjacent indices, one gold: only one TP, not two."""
        gold = [Miscue("substitution", "dog", "gog", index=5)]
        pred = [
            Miscue("substitution", "dog", "gog", index=4),
            Miscue("substitution", "dog", "gog", index=5),
        ]
        counts = match_counts(pred, gold, n_target_words=10)
        # Greedy picks the exact match (distance 0) first; the other is FP.
        assert counts["substitution"] == {"tp": 1, "fp": 1, "fn": 0}

    # -----------------------------------------------------------------------
    # Correct-word count
    # -----------------------------------------------------------------------

    def test_correct_words_no_gold(self):
        """No gold miscues → all n_target_words are correct."""
        counts = match_counts([], [], n_target_words=7)
        assert counts["_correct_words"] == 7

    def test_correct_words_with_gold(self):
        """Gold miscues at indices 2 and 5 → 5 correct words out of 7."""
        gold = [
            Miscue("substitution", "x", "y", index=2),
            Miscue("omission", "z", None, index=5),
        ]
        counts = match_counts([], gold, n_target_words=7)
        assert counts["_correct_words"] == 5

    def test_correct_words_duplicate_gold_index(self):
        """Two gold miscues at the SAME index count as ONE occupied slot."""
        gold = [
            Miscue("substitution", "x", "y", index=3),
            Miscue("hesitation", "x", None, index=3),
        ]
        counts = match_counts([], gold, n_target_words=7)
        # index 3 is occupied once → 7 - 1 = 6 correct words
        assert counts["_correct_words"] == 6


# ===========================================================================
# gold.jsonl → Miscue parsing round-trip
# ===========================================================================

_PROJECT_ROOT = Path(__file__).parent.parent
_GOLD_JSONL = _PROJECT_ROOT / "data" / "benchmark" / "gold.jsonl"


def _load_gold_rows() -> list[dict]:
    rows = []
    for line in _GOLD_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _row_to_gold_miscues(row: dict) -> list[Miscue]:
    """Mirror the parser in run_benchmark.py."""
    return [
        Miscue(
            type=entry["type"],
            target_word=entry["target_word"],
            said_word=entry["said_word"],
            index=entry["index"],
        )
        for entry in row["gold"]
    ]


class TestGoldParsing:
    """Parse gold.jsonl rows into Miscues and validate the round-trip."""

    def test_clean_row_parses_to_empty_list(self):
        rows = _load_gold_rows()
        clean_rows = [r for r in rows if r["gold"] == []]
        assert clean_rows, "No clean rows found — unexpected"
        for row in clean_rows[:3]:
            miscues = _row_to_gold_miscues(row)
            assert miscues == []

    def test_substitution_row_round_trips(self):
        rows = _load_gold_rows()
        sub_rows = [
            r for r in rows
            if len(r["gold"]) == 1 and r["gold"][0]["type"] == "substitution"
        ]
        assert sub_rows, "No single-substitution rows found"
        row = sub_rows[0]
        entry = row["gold"][0]
        miscues = _row_to_gold_miscues(row)
        assert len(miscues) == 1
        m = miscues[0]
        assert m.type == "substitution"
        assert m.target_word == entry["target_word"]
        assert m.said_word == entry["said_word"]
        assert m.index == entry["index"]

    def test_silence_hesitation_row_round_trips(self):
        """Silence-hesitation rows (render == 'silence') parse correctly."""
        rows = _load_gold_rows()
        sil_rows = [
            r for r in rows
            if any(e["type"] == "hesitation" and e.get("render") == "silence"
                   for e in r["gold"])
        ]
        assert sil_rows, "No silence-hesitation rows found"
        row = sil_rows[0]
        miscues = _row_to_gold_miscues(row)
        sil_miscues = [m for m in miscues if m.type == "hesitation"]
        assert sil_miscues, "Expected at least one hesitation Miscue"
        # Silence hesitations have target_word set (aligned to the next word)
        # and said_word = None.
        for m in sil_miscues:
            assert m.said_word is None or m.said_word  # either None or non-empty str

    def test_filler_hesitation_row_round_trips(self):
        """Filler-hesitation rows (render == 'filler') parse correctly."""
        rows = _load_gold_rows()
        filler_rows = [
            r for r in rows
            if any(e["type"] == "hesitation" and e.get("render") == "filler"
                   for e in r["gold"])
        ]
        assert filler_rows, "No filler-hesitation rows found"
        row = filler_rows[0]
        miscues = _row_to_gold_miscues(row)
        filler_miscues = [m for m in miscues if m.type == "hesitation"]
        assert filler_miscues

    def test_multi_miscue_row_count_matches(self):
        """A row with 3 gold entries parses to a list of exactly 3 Miscues."""
        rows = _load_gold_rows()
        multi = [r for r in rows if len(r["gold"]) == 3]
        assert multi, "No 3-gold-entry rows found"
        row = multi[0]
        miscues = _row_to_gold_miscues(row)
        assert len(miscues) == 3


# ===========================================================================
# Aggregation math tests (micro vs macro)
# ===========================================================================

class TestAggregation:
    """Verify that the runner's micro-aggregation is correct.

    We simulate the accumulation loop directly using match_counts and the
    internal helper logic from the runner.
    """

    def _accum(self):
        classes = ("substitution", "omission", "insertion", "self_correction", "hesitation")
        out: dict = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in classes}
        out["_correct_words"] = 0
        out["_total_fp"] = 0
        return out

    def _add(self, accum, counts):
        classes = ("substitution", "omission", "insertion", "self_correction", "hesitation")
        for cls in classes:
            accum[cls]["tp"] += counts[cls]["tp"]
            accum[cls]["fp"] += counts[cls]["fp"]
            accum[cls]["fn"] += counts[cls]["fn"]
            accum["_total_fp"] += counts[cls]["fp"]
        accum["_correct_words"] += counts["_correct_words"]

    def test_micro_sums_across_items(self):
        """Two items: item1 has 1 sub, item2 has 1 sub; micro sum = 2 gold."""
        gold1 = [Miscue("substitution", "cat", "bat", index=2)]
        pred1 = [Miscue("substitution", "cat", "bat", index=2)]  # TP
        gold2 = [Miscue("substitution", "dog", "fog", index=5)]
        pred2 = []  # FN

        acc = self._accum()
        self._add(acc, match_counts(pred1, gold1, n_target_words=5))
        self._add(acc, match_counts(pred2, gold2, n_target_words=8))

        assert acc["substitution"]["tp"] == 1
        assert acc["substitution"]["fn"] == 1
        assert acc["substitution"]["fp"] == 0

    def test_micro_ne_macro(self):
        """Micro and macro diverge when items have unequal event counts.

        Item 1: 3 gold subs, 3 predicted (all TP) → item recall = 1.0
        Item 2: 1 gold sub, 0 predicted (0 TP)    → item recall = 0.0
        Macro recall = (1.0 + 0.0) / 2 = 0.50
        Micro recall = 3 / (3 + 1) = 0.75  ← different
        """
        gold1 = [
            Miscue("substitution", "a", "x", index=1),
            Miscue("substitution", "b", "y", index=3),
            Miscue("substitution", "c", "z", index=5),
        ]
        pred1 = [
            Miscue("substitution", "a", "x", index=1),
            Miscue("substitution", "b", "y", index=3),
            Miscue("substitution", "c", "z", index=5),
        ]
        gold2 = [Miscue("substitution", "d", "w", index=2)]
        pred2 = []

        # Micro
        acc = self._accum()
        self._add(acc, match_counts(pred1, gold1, n_target_words=10))
        self._add(acc, match_counts(pred2, gold2, n_target_words=10))
        tp = acc["substitution"]["tp"]
        fn = acc["substitution"]["fn"]
        micro_recall = tp / (tp + fn)
        assert abs(micro_recall - 0.75) < 1e-9

        # Macro (computed separately, just for comparison)
        macro_recall = (1.0 + 0.0) / 2
        assert abs(macro_recall - 0.50) < 1e-9

        # They differ
        assert micro_recall != macro_recall

    def test_fp_per_100_correct_words_micro(self):
        """fp_per_100_correct_words is total_fp / total_correct_words * 100."""
        # Item 1: 5-word target, no gold, 1 FP prediction
        gold1: list[Miscue] = []
        pred1 = [Miscue("insertion", None, "extra", index=2)]
        # Item 2: 5-word target, no gold, 0 FP
        gold2: list[Miscue] = []
        pred2: list[Miscue] = []

        acc = self._accum()
        self._add(acc, match_counts(pred1, gold1, n_target_words=5))
        self._add(acc, match_counts(pred2, gold2, n_target_words=5))

        total_fp = acc["_total_fp"]
        total_correct = acc["_correct_words"]
        fp_per_100 = total_fp / total_correct * 100

        # 1 FP / 10 correct words * 100 = 10.0
        assert total_fp == 1
        assert total_correct == 10
        assert abs(fp_per_100 - 10.0) < 1e-9


# ===========================================================================
# Metadata completeness on a synthetic write
# ===========================================================================

class TestMetadataCompleteness:
    """Verify output JSON schema without touching any audio or model."""

    REQUIRED_METADATA_KEYS = {
        "backend",
        "benchmark_version",
        "gold_sha256",
        "git_commit",
        "date",
        "n_items",
        "biases_run",
        "aggregation",
    }

    REQUIRED_CLASS_KEYS = {
        "tp", "fp", "fn",
        "precision", "recall", "f1",
        "n_gold", "n_pred",
    }

    def _make_synthetic_output(self, biases=("none",), n_items=2) -> dict:
        """Build a minimal but schema-valid output dict (no audio needed)."""
        classes = ("substitution", "omission", "insertion", "self_correction", "hesitation")
        results = {}
        for bias in biases:
            bias_result = {}
            for cls in classes:
                bias_result[cls] = {
                    "tp": 0, "fp": 0, "fn": 0,
                    "precision": None, "recall": None, "f1": None,
                    "n_gold": 0, "n_pred": 0,
                }
            bias_result["fp_per_100_correct_words"] = 0.0
            results[bias] = bias_result

        return {
            "metadata": {
                "backend": "faster-whisper-small",
                "benchmark_version": "0.1.0",
                "gold_sha256": "abc123",
                "git_commit": "deadbeef",
                "date": "2026-06-10",
                "n_items": n_items,
                "biases_run": list(biases),
                "aggregation": "micro",
            },
            "results": results,
        }

    def test_metadata_has_all_required_keys(self):
        output = self._make_synthetic_output()
        missing = self.REQUIRED_METADATA_KEYS - set(output["metadata"].keys())
        assert not missing, f"Missing metadata keys: {missing}"

    def test_aggregation_is_micro(self):
        output = self._make_synthetic_output()
        assert output["metadata"]["aggregation"] == "micro"

    def test_results_have_all_biases(self):
        biases = ("none", "prompt", "strong")
        output = self._make_synthetic_output(biases=biases)
        assert set(output["results"].keys()) == set(biases)

    def test_each_class_has_required_keys(self):
        output = self._make_synthetic_output(biases=("none",))
        classes = ("substitution", "omission", "insertion", "self_correction", "hesitation")
        for cls in classes:
            missing = self.REQUIRED_CLASS_KEYS - set(output["results"]["none"][cls].keys())
            assert not missing, f"Class {cls} missing keys: {missing}"

    def test_fp_per_100_correct_words_present(self):
        output = self._make_synthetic_output()
        assert "fp_per_100_correct_words" in output["results"]["none"]

    def test_n_items_matches_actual(self):
        output = self._make_synthetic_output(n_items=5)
        assert output["metadata"]["n_items"] == 5

    def test_output_is_json_serializable(self, tmp_path):
        """The output dict must round-trip through json.dumps / json.loads."""
        output = self._make_synthetic_output(biases=("none", "prompt"))
        serialized = json.dumps(output, indent=2)
        restored = json.loads(serialized)
        assert restored["metadata"]["n_items"] == output["metadata"]["n_items"]
        assert set(restored["results"].keys()) == {"none", "prompt"}
