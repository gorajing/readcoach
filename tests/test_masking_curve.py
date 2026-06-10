"""Tests for scripts/masking_curve.py (T2.1).

CI-safe: no network, no model, no audio.
All tests use synthetic per-item counts and hand-crafted data.
"""
from __future__ import annotations


import numpy as np
import pytest

from scripts.masking_curve import (
    bootstrap_micro_stats,
    percentile_ci,
    _check_agreement,
    _micro_aggregate,
    _normalize_sentence,
    _wer_normalized,
    _CLASSES,
)


# ---------------------------------------------------------------------------
# _normalize_sentence
# ---------------------------------------------------------------------------

class TestNormalizeSentence:
    def test_strips_punctuation_and_casefolds(self):
        assert _normalize_sentence("The cat.") == "the cat"

    def test_empty_string(self):
        assert _normalize_sentence("") == ""

    def test_all_punctuation_tokens(self):
        # "..." and "," are pure punctuation — filtered out
        assert _normalize_sentence("... ,") == ""

    def test_preserves_internal_apostrophe(self):
        # "don't" — internal apostrophe survives
        assert "don't" in _normalize_sentence("Don't run.")


# ---------------------------------------------------------------------------
# _wer_normalized
# ---------------------------------------------------------------------------

class TestWerNormalized:
    def test_identical(self):
        assert _wer_normalized("the cat sat", "the cat sat") == pytest.approx(0.0)

    def test_one_substitution(self):
        # 1 error / 4 words = 0.25
        assert _wer_normalized("the cat sat on", "the dog sat on") == pytest.approx(0.25)

    def test_fully_wrong(self):
        # 3 errors / 3 reference words = 1.0
        assert _wer_normalized("a b c", "x y z") == pytest.approx(1.0)

    def test_empty_hypothesis(self):
        # 3 deletions / 3 words = 1.0
        assert _wer_normalized("the cat sat", "") == pytest.approx(1.0)

    def test_empty_reference(self):
        # Degenerate: returns 0.0 (no reference words to evaluate)
        assert _wer_normalized("", "any word") == pytest.approx(0.0)

    def test_extra_hypothesis_words(self):
        # ref="a b", hyp="a b c d" → 2 insertions / 2 ref words = 1.0
        assert _wer_normalized("a b", "a b c d") == pytest.approx(1.0)

    def test_one_deletion(self):
        # ref="a b c", hyp="a c" → 1 deletion / 3 = 0.333...
        assert _wer_normalized("a b c", "a c") == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# _micro_aggregate
# ---------------------------------------------------------------------------

def _make_counts(
    *,
    sub_tp=0, sub_fp=0, sub_fn=0,
    om_tp=0, om_fp=0, om_fn=0,
    ins_tp=0, ins_fp=0, ins_fn=0,
    sc_tp=0, sc_fp=0, sc_fn=0,
    hes_tp=0, hes_fp=0, hes_fn=0,
    correct_words=10,
) -> dict:
    return {
        "substitution": {"tp": sub_tp, "fp": sub_fp, "fn": sub_fn},
        "omission": {"tp": om_tp, "fp": om_fp, "fn": om_fn},
        "insertion": {"tp": ins_tp, "fp": ins_fp, "fn": ins_fn},
        "self_correction": {"tp": sc_tp, "fp": sc_fp, "fn": sc_fn},
        "hesitation": {"tp": hes_tp, "fp": hes_fp, "fn": hes_fn},
        "_correct_words": correct_words,
    }


def _make_item(counts: dict, wer_spoken=0.1, wer_target=0.05) -> dict:
    return {"counts": counts, "wer_vs_spoken": wer_spoken, "wer_vs_target": wer_target}


class TestMicroAggregate:
    def test_single_perfect_substitution(self):
        c = _make_counts(sub_tp=1, correct_words=9)
        result = _micro_aggregate([_make_item(c)])
        assert result["substitution_precision"] == pytest.approx(1.0)
        assert result["substitution_recall"] == pytest.approx(1.0)

    def test_all_fn_zero_pred(self):
        c = _make_counts(sub_fn=3, correct_words=7)
        result = _micro_aggregate([_make_item(c)])
        assert result["substitution_recall"] == pytest.approx(0.0)
        assert result["substitution_precision"] == pytest.approx(0.0)

    def test_all_fp_zero_gold(self):
        c = _make_counts(sub_fp=2, correct_words=10)
        result = _micro_aggregate([_make_item(c)])
        assert result["substitution_precision"] == pytest.approx(0.0)
        # recall: 0 gold → 0 denominator. Our code returns 0.0 (n_gold=0, not None
        # because n_pred > 0).
        assert result["substitution_recall"] == pytest.approx(0.0)

    def test_both_absent_is_none(self):
        c = _make_counts(correct_words=5)  # all zeros
        result = _micro_aggregate([_make_item(c)])
        assert result["substitution_precision"] is None
        assert result["substitution_recall"] is None

    def test_fp_per_100_correct_words(self):
        # 2 FPs (both from insertion), 10 correct words → 20.0
        c = _make_counts(ins_fp=2, correct_words=10)
        result = _micro_aggregate([_make_item(c)])
        assert result["fp_per_100_correct_words"] == pytest.approx(20.0)

    def test_wer_means_across_items(self):
        items = [
            _make_item(_make_counts(), wer_spoken=0.1, wer_target=0.2),
            _make_item(_make_counts(), wer_spoken=0.3, wer_target=0.4),
        ]
        result = _micro_aggregate(items)
        assert result["wer_vs_spoken_mean"] == pytest.approx(0.2)
        assert result["wer_vs_target_mean"] == pytest.approx(0.3)

    def test_micro_sums_across_items(self):
        # Item 1: 2 TP substitutions; Item 2: 1 FN substitution
        c1 = _make_counts(sub_tp=2, correct_words=8)
        c2 = _make_counts(sub_fn=1, correct_words=9)
        result = _micro_aggregate([_make_item(c1), _make_item(c2)])
        # Micro: tp=2, fn=1, fp=0 → precision=1.0, recall=2/3
        assert result["substitution_tp"] == 2
        assert result["substitution_fn"] == 1
        assert result["substitution_precision"] == pytest.approx(1.0)
        assert result["substitution_recall"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# bootstrap_micro_stats — determinism and CI correctness
# ---------------------------------------------------------------------------

def _make_counts_list(n: int, tp: int = 1, fp: int = 0, fn: int = 0) -> list[dict]:
    """Return n identical count dicts (substitution-only)."""
    return [_make_counts(sub_tp=tp, sub_fp=fp, sub_fn=fn, correct_words=10) for _ in range(n)]


class TestBootstrapMicroStats:
    def test_same_seed_same_result(self):
        counts = _make_counts_list(10, tp=1, fp=0, fn=0)
        wers_s = [0.1] * 10
        wers_t = [0.05] * 10

        rng1 = np.random.default_rng(42)
        result1 = bootstrap_micro_stats(counts, wers_s, wers_t, n_boot=50, rng=rng1)

        rng2 = np.random.default_rng(42)
        result2 = bootstrap_micro_stats(counts, wers_s, wers_t, n_boot=50, rng=rng2)

        for key in result1:
            np.testing.assert_array_equal(result1[key], result2[key])

    def test_different_seeds_different_result(self):
        # Use heterogeneous items so resampled statistics vary across bootstrap draws.
        # Items have varying TP/FP/FN so different resamples give different precisions.
        rng_data = np.random.default_rng(42)
        counts = []
        wers_s = []
        wers_t = []
        for i in range(20):
            tp = int(rng_data.integers(0, 4))
            fp = int(rng_data.integers(0, 4))
            fn = int(rng_data.integers(0, 4))
            counts.append(_make_counts(sub_tp=tp, sub_fp=fp, sub_fn=fn, correct_words=10))
            wers_s.append(float(rng_data.uniform(0.0, 0.5)))
            wers_t.append(float(rng_data.uniform(0.0, 0.3)))

        rng_a = np.random.default_rng(1)
        r_a = bootstrap_micro_stats(counts, wers_s, wers_t, n_boot=100, rng=rng_a)
        rng_b = np.random.default_rng(999)
        r_b = bootstrap_micro_stats(counts, wers_s, wers_t, n_boot=100, rng=rng_b)

        # Samples should differ with different seeds (virtually certain with 100 draws
        # over heterogeneous data; probability of equal arrays is negligible).
        assert not np.array_equal(r_a["substitution_precision"], r_b["substitution_precision"])

    def test_all_correct_precision_one(self):
        """When every resample has tp=N, fp=0, micro precision = 1.0 always."""
        counts = _make_counts_list(15, tp=2, fp=0, fn=0)
        wers_s = [0.0] * 15
        wers_t = [0.0] * 15
        rng = np.random.default_rng(0)
        result = bootstrap_micro_stats(counts, wers_s, wers_t, n_boot=100, rng=rng)
        np.testing.assert_allclose(result["substitution_precision"], 1.0)

    def test_output_length_matches_n_boot(self):
        counts = _make_counts_list(5, tp=1, fp=1, fn=0)
        wers_s = [0.1] * 5
        wers_t = [0.2] * 5
        rng = np.random.default_rng(7)
        result = bootstrap_micro_stats(counts, wers_s, wers_t, n_boot=77, rng=rng)
        for arr in result.values():
            assert len(arr) == 77

    def test_expected_keys_present(self):
        counts = _make_counts_list(5)
        rng = np.random.default_rng(0)
        result = bootstrap_micro_stats(counts, [0.1] * 5, [0.05] * 5, n_boot=10, rng=rng)
        expected_keys = (
            [f"{c}_precision" for c in _CLASSES] +
            [f"{c}_recall" for c in _CLASSES] +
            ["fp_per_100", "wer_vs_spoken_mean", "wer_vs_target_mean"]
        )
        for k in expected_keys:
            assert k in result, f"Missing key: {k}"


# ---------------------------------------------------------------------------
# percentile_ci — math hand-check
# ---------------------------------------------------------------------------

class TestPercentileCi:
    def test_symmetric_normal_like(self):
        rng = np.random.default_rng(123)
        samples = rng.normal(0.5, 0.1, size=10_000)
        lo, hi = percentile_ci(samples, lo=2.5, hi=97.5)
        # 95% CI for N(0.5, 0.1): roughly [0.30, 0.70].
        # Assert: lo < mean < hi, lo and hi close to ±1.96σ from mean.
        assert lo < 0.5 < hi
        assert lo == pytest.approx(0.5 - 1.96 * 0.1, abs=0.02)
        assert hi == pytest.approx(0.5 + 1.96 * 0.1, abs=0.02)

    def test_constant_array(self):
        arr = np.full(100, 0.75)
        lo, hi = percentile_ci(arr)
        assert lo == pytest.approx(0.75)
        assert hi == pytest.approx(0.75)

    def test_wide_ci(self):
        arr = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        lo, hi = percentile_ci(arr, lo=0.0, hi=100.0)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(1.0)

    def test_returns_floats(self):
        arr = np.array([0.1, 0.5, 0.9])
        lo, hi = percentile_ci(arr)
        assert isinstance(lo, float)
        assert isinstance(hi, float)


# ---------------------------------------------------------------------------
# Agreement check logic — synthetic mismatch must raise
# ---------------------------------------------------------------------------

class TestAgreementCheck:
    """Verify the agreement-check guard raises on a synthetic mismatch."""

    def _make_per_item_for_bias(
        self,
        tp: int,
        fp: int,
        fn: int,
        n: int = 5,
    ) -> list[dict]:
        """Build n identical items for one bias, substitution class only."""
        counts = {
            "substitution": {"tp": tp, "fp": fp, "fn": fn},
            "omission": {"tp": 0, "fp": 0, "fn": 0},
            "insertion": {"tp": 0, "fp": 0, "fn": 0},
            "self_correction": {"tp": 0, "fp": 0, "fn": 0},
            "hesitation": {"tp": 0, "fp": 0, "fn": 0},
            "_correct_words": 10,
        }
        return [
            {"counts": counts, "wer_vs_spoken": 0.1, "wer_vs_target": 0.05}
            for _ in range(n)
        ]

    def test_mismatch_raises(self, tmp_path, monkeypatch):
        """Mismatch between recomputed and baseline P/R → ValueError."""
        import json
        import scripts.masking_curve as mc

        # Build a fake baseline with precision=0.9 for substitution/none
        # But per-item stats give precision=0.5 → mismatch should raise.
        fake_baseline = {
            "metadata": {},
            "results": {
                "none": {
                    "substitution": {
                        "tp": 1, "fp": 1, "fn": 1,
                        "precision": 0.9,  # WRONG — real value is 0.5
                        "recall": 0.5,
                        "f1": 0.4,
                        "n_gold": 2, "n_pred": 2,
                    },
                    "omission": {"tp": 0, "fp": 0, "fn": 0,
                                 "precision": None, "recall": None, "f1": None,
                                 "n_gold": 0, "n_pred": 0},
                    "insertion": {"tp": 0, "fp": 0, "fn": 0,
                                  "precision": None, "recall": None, "f1": None,
                                  "n_gold": 0, "n_pred": 0},
                    "self_correction": {"tp": 0, "fp": 0, "fn": 0,
                                        "precision": None, "recall": None, "f1": None,
                                        "n_gold": 0, "n_pred": 0},
                    "hesitation": {"tp": 0, "fp": 0, "fn": 0,
                                   "precision": None, "recall": None, "f1": None,
                                   "n_gold": 0, "n_pred": 0},
                    "fp_per_100_correct_words": 0.0,
                }
            },
        }
        baseline_path = tmp_path / "miscue-v0.json"
        baseline_path.write_text(json.dumps(fake_baseline), encoding="utf-8")

        # Monkeypatch the module-level path so _check_agreement reads our fake file.
        monkeypatch.setattr(mc, "_BASELINE_JSON", baseline_path)

        # per_item has tp=1, fp=1, fn=1 → precision = 1/2 = 0.5 (not 0.9)
        per_item = {"none": self._make_per_item_for_bias(tp=1, fp=1, fn=1)}

        with pytest.raises(ValueError, match="Agreement mismatch"):
            _check_agreement(per_item, ("none",))

    def test_agreement_passes_when_matching(self, tmp_path, monkeypatch):
        """Exact match (within tolerance) should not raise."""
        import json
        import scripts.masking_curve as mc

        # per_item has tp=2, fp=2, fn=2 per item × 5 items
        # micro precision = (5*2) / (5*2 + 5*2) = 10/20 = 0.5
        # micro recall    = (5*2) / (5*2 + 5*2) = 10/20 = 0.5
        # fp_per_100_correct_words = (5*10) / (5*10) * 100 = 100.0  ← wait, let's check:
        # total_all_fp = sum of ALL class FPs = 5*2 = 10 (only sub has fp=2)
        # total_correct_words = 5*10 = 50
        # fp_per_100 = 10/50 * 100 = 20.0

        fake_baseline = {
            "metadata": {},
            "results": {
                "none": {
                    "substitution": {
                        "tp": 0, "fp": 0, "fn": 0,  # counts don't matter for agreement
                        "precision": 0.5,
                        "recall": 0.5,
                        "f1": 0.5,
                        "n_gold": 10, "n_pred": 10,
                    },
                    "omission": {"tp": 0, "fp": 0, "fn": 0,
                                 "precision": None, "recall": None, "f1": None,
                                 "n_gold": 0, "n_pred": 0},
                    "insertion": {"tp": 0, "fp": 0, "fn": 0,
                                  "precision": None, "recall": None, "f1": None,
                                  "n_gold": 0, "n_pred": 0},
                    "self_correction": {"tp": 0, "fp": 0, "fn": 0,
                                        "precision": None, "recall": None, "f1": None,
                                        "n_gold": 0, "n_pred": 0},
                    "hesitation": {"tp": 0, "fp": 0, "fn": 0,
                                   "precision": None, "recall": None, "f1": None,
                                   "n_gold": 0, "n_pred": 0},
                    "fp_per_100_correct_words": 20.0,
                }
            },
        }
        baseline_path = tmp_path / "miscue-v0.json"
        baseline_path.write_text(json.dumps(fake_baseline), encoding="utf-8")
        monkeypatch.setattr(mc, "_BASELINE_JSON", baseline_path)

        per_item = {"none": self._make_per_item_for_bias(tp=2, fp=2, fn=2)}

        # Should not raise
        _check_agreement(per_item, ("none",))
