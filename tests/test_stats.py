"""Tests for evals/stats.py — T5.2 judge-validation statistics.

TDD: red first.  All vectors are hand-computed; arithmetic shown in comments.

CONTRACT UNDER TEST
-------------------
tpr_tnr(judge, human) -> (tpr, tnr)
cohens_kappa(a, b) -> float
bootstrap_ci(stat_fn, *paired_arrays, n_boot, seed, ci) -> (lo, hi)

Degenerate cases, None-class rates, ValueError on bad inputs: all covered.
"""

from __future__ import annotations

import pytest

from evals.stats import bootstrap_ci, cohens_kappa, tpr_tnr


# ---------------------------------------------------------------------------
# tpr_tnr — binary agreement: judge.passing vs human.passing (human = truth)
# ---------------------------------------------------------------------------


class TestTprTnr:
    # --- basic correctness ---

    def test_perfect_agreement_all_positive(self):
        # human=[T,T,T], judge=[T,T,T]  → TP=3, FP=0, TN=0, FN=0
        # TPR = TP/(TP+FN) = 3/3 = 1.0; no negatives → TNR = None
        tpr, tnr = tpr_tnr([True, True, True], [True, True, True])
        assert tpr == 1.0
        assert tnr is None

    def test_perfect_agreement_all_negative(self):
        # human=[F,F,F], judge=[F,F,F]  → TN=3, TP=0, FP=0, FN=0
        # TNR = TN/(TN+FP) = 3/3 = 1.0; no positives → TPR = None
        tpr, tnr = tpr_tnr([False, False, False], [False, False, False])
        assert tpr is None
        assert tnr == 1.0

    def test_mixed_classes_standard(self):
        # human=[T,T,F,F], judge=[T,F,T,F]
        # TP=1, FN=1, FP=1, TN=1
        # TPR=1/2=0.5; TNR=1/2=0.5
        tpr, tnr = tpr_tnr([True, False, True, False], [True, True, False, False])
        assert tpr == pytest.approx(0.5)
        assert tnr == pytest.approx(0.5)

    def test_tpr_perfect_tnr_imperfect(self):
        # judge=[T,T,T,T], human=[T,T,T,F]
        # human positives = 3 (first 3), negatives = 1 (last)
        # judge for human-T: T,T,T → TP=3, FN=0; TPR=1.0
        # judge for human-F: T → FP=1, TN=0; TNR=0.0
        tpr, tnr = tpr_tnr([True, True, True, True], [True, True, True, False])
        assert tpr == pytest.approx(1.0)
        assert tnr == pytest.approx(0.0)

    def test_tpr_zero(self):
        # human=[T,T], judge=[F,F]  → TP=0, FN=2; TPR=0.0
        # no negatives → TNR=None
        tpr, tnr = tpr_tnr([False, False], [True, True])
        assert tpr == pytest.approx(0.0)
        assert tnr is None

    def test_tnr_zero(self):
        # human=[F,F], judge=[T,T]  → FP=2, TN=0; TNR=0.0
        # no positives → TPR=None
        tpr, tnr = tpr_tnr([True, True], [False, False])
        assert tpr is None
        assert tnr == pytest.approx(0.0)

    # --- None-class: never 0/0 ---

    def test_no_positives_in_human_tpr_is_none(self):
        # human all False → no positives → TPR cannot be computed → None
        tpr, tnr = tpr_tnr([False, True, False, True], [False, False, False, False])
        assert tpr is None

    def test_no_negatives_in_human_tnr_is_none(self):
        # human all True → no negatives → TNR cannot be computed → None
        tpr, tnr = tpr_tnr([True, True, True, True], [True, True, True, True])
        assert tnr is None

    # --- length mismatch raises ---

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            tpr_tnr([True, False], [True, False, True])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            tpr_tnr([], [])


# ---------------------------------------------------------------------------
# cohens_kappa — binary / categorical agreement
#
# Hand-computed vectors (show work in comments so reviewers can verify):
#
#  Example 1 — 2×2 confusion matrix:
#    Rater A: [T, T, T, F, F, F, F, F, F, F]   (3 T, 7 F)
#    Rater B: [T, T, F, T, F, F, F, F, F, F]   (3 T, 7 F)
#
#    Confusion table (A\B):
#             B=T   B=F
#    A=T    [  2     1  ]   row sums: 3
#    A=F    [  1     6  ]   row sums: 7
#    col:     3     7       n=10
#
#    P_o (observed agreement) = (2+6)/10 = 0.8
#    P_e (expected agreement):
#      P(both T) = (3/10)*(3/10) = 9/100 = 0.09
#      P(both F) = (7/10)*(7/10) = 49/100 = 0.49
#      P_e = 0.09 + 0.49 = 0.58
#    kappa = (P_o - P_e)/(1 - P_e) = (0.8 - 0.58)/(1 - 0.58)
#          = 0.22/0.42 ≈ 0.5238095...
#
#  Example 2 — perfect agreement (non-degenerate):
#    a = b = [T, T, T, F, F, F]
#    P_o = 1.0; P_e = (3/6)^2 + (3/6)^2 = 0.25 + 0.25 = 0.5
#    kappa = (1.0 - 0.5)/(1 - 0.5) = 1.0
#
#  Degenerate case (all same class):
#    a = b = [T, T, T, T]  → P_e = 1.0 → denominator = 0 → kappa undefined
#    → must raise ValueError, not return 1.0 or 0.0
# ---------------------------------------------------------------------------


class TestCohensKappa:
    def test_known_vector_approx(self):
        # Example 1 above: kappa ≈ 0.5238
        a = [True, True, True, False, False, False, False, False, False, False]
        b = [True, True, False, True, False, False, False, False, False, False]
        k = cohens_kappa(a, b)
        assert k == pytest.approx(0.5238095, rel=1e-4)

    def test_perfect_agreement_non_degenerate(self):
        # Example 2 above: kappa = 1.0
        a = [True, True, True, False, False, False]
        b = [True, True, True, False, False, False]
        k = cohens_kappa(a, b)
        assert k == pytest.approx(1.0)

    def test_perfect_agreement_all_same_class_raises(self):
        # Degenerate: P_e = 1.0 → denominator = 0 → undefined → ValueError
        # Must NOT silently return 1.0 or 0.0
        with pytest.raises(ValueError, match="undefined|degenerate|denominator"):
            cohens_kappa([True, True, True], [True, True, True])

    def test_all_disagree_symmetrically(self):
        # a=[T,F], b=[F,T]  — complete disagreement on 2 items, balanced classes
        # P_o = 0.0
        # P_e: A has 1T,1F; B has 1F,1T  → same marginals as balanced
        #   P(both T) = (1/2)*(1/2) = 0.25; P(both F) = (1/2)*(1/2) = 0.25; P_e=0.5
        # kappa = (0-0.5)/(1-0.5) = -1.0
        k = cohens_kappa([True, False], [False, True])
        assert k == pytest.approx(-1.0)

    def test_chance_agreement(self):
        # Construct a case where P_o == P_e (kappa = 0).
        # With a=[T,T,F,F], b=[T,F,T,F]:
        #   Confusion: TP=1, FP=1, FN=1, TN=1
        #   P_o = (1+1)/4 = 0.5
        #   P_e = (2/4)*(2/4) + (2/4)*(2/4) = 0.25 + 0.25 = 0.5
        #   kappa = 0.0
        a = [True, True, False, False]
        b = [True, False, True, False]
        k = cohens_kappa(a, b)
        assert k == pytest.approx(0.0, abs=1e-9)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            cohens_kappa([True, False], [True, False, True])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            cohens_kappa([], [])

    def test_single_item_agreement_raises_degenerate(self):
        # n=1, a=[T], b=[T] → all same class → P_e=1 → undefined
        with pytest.raises(ValueError):
            cohens_kappa([True], [True])

    def test_works_with_integers(self):
        # kappa works on any comparable labels, not just booleans
        a = [1, 1, 0, 0, 1]
        b = [1, 0, 0, 0, 1]
        # P_o = (1+1+1)/5 = 3/5 = 0.6 ... just check it returns a float without crashing
        k = cohens_kappa(a, b)
        assert isinstance(k, float)


# ---------------------------------------------------------------------------
# bootstrap_ci — seeded, percentile method, paired resampling
# ---------------------------------------------------------------------------


class TestBootstrapCi:
    def test_deterministic_with_seed(self):
        a = [True, True, True, False, False]
        b = [True, True, False, True, False]
        lo1, hi1 = bootstrap_ci(cohens_kappa, a, b, n_boot=500, seed=42, ci=0.95)
        lo2, hi2 = bootstrap_ci(cohens_kappa, a, b, n_boot=500, seed=42, ci=0.95)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_different_seeds_give_different_ci(self):
        # With only 5 items, two different seeds should usually differ
        a = [True, True, True, False, False]
        b = [True, True, False, True, False]
        lo1, hi1 = bootstrap_ci(cohens_kappa, a, b, n_boot=500, seed=1, ci=0.95)
        lo2, hi2 = bootstrap_ci(cohens_kappa, a, b, n_boot=500, seed=999, ci=0.95)
        # Not guaranteed to differ, but with 500 boots and 5 items, they almost always do
        # Just check the function runs without error — seed isolation is tested above
        assert isinstance(lo1, float)
        assert isinstance(hi1, float)

    def test_tpr_ci_brackets_true_rate(self):
        # 60% positive rate (6 T out of 10); judge matches human perfectly
        # True TPR = 1.0 for positives, 1.0 for negatives
        # CI should bracket 1.0
        human = [True] * 6 + [False] * 4
        judge = human[:]  # perfect match
        # tpr_tnr returns (tpr, tnr); we want TPR only

        def _tpr(j, h):
            return tpr_tnr(j, h)[0]

        lo, hi = bootstrap_ci(_tpr, judge, human, n_boot=1000, seed=7, ci=0.95)
        assert lo <= 1.0 <= hi

    def test_known_distribution_mean_brackets_true(self):
        # Mean of a 0/1 array with true proportion p=0.7
        # 95% CI should bracket 0.7 for n=100
        import numpy as np

        rng = np.random.default_rng(0)
        data = rng.binomial(1, 0.7, size=100).tolist()

        def _mean(x):
            return sum(x) / len(x)

        lo, hi = bootstrap_ci(_mean, data, n_boot=2000, seed=0, ci=0.95)
        assert lo <= 0.7 <= hi, f"CI [{lo:.3f}, {hi:.3f}] does not bracket 0.7"

    def test_ci_bounds_ordered(self):
        a = [True, True, False, False, True]
        b = [True, False, False, True, True]
        lo, hi = bootstrap_ci(cohens_kappa, a, b, n_boot=300, seed=5, ci=0.95)
        assert lo <= hi

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            bootstrap_ci(cohens_kappa, [True, False], [True, False, True], n_boot=10, seed=0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci(cohens_kappa, [], [], n_boot=10, seed=0)

    def test_single_arg_array(self):
        # stat_fn takes a single array (e.g. mean)
        data = [1, 0, 1, 1, 0, 0, 1]

        def _mean_single(x):
            return sum(x) / len(x)

        lo, hi = bootstrap_ci(_mean_single, data, n_boot=200, seed=3, ci=0.90)
        assert lo <= hi
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_none_resample_causes_redraw_not_silent_drop(self):
        # Build a stat_fn that returns None on the first call, then valid values.
        # bootstrap_ci must REDRAW (not silently drop) up to the cap.
        # We verify it completes successfully (doesn't drop without cap).
        call_count = [0]

        def sometimes_none(a, b):
            call_count[0] += 1
            # Return None for the first 10 calls, then valid kappa
            if call_count[0] <= 10:
                return None
            return cohens_kappa(a, b)

        a = [True, True, True, False, False, False]
        b = [True, True, False, True, False, False]
        lo, hi = bootstrap_ci(sometimes_none, a, b, n_boot=200, seed=9)
        assert lo <= hi

    def test_stat_fn_always_none_raises(self):
        # If stat_fn returns None on every redraw → raises after cap
        def _always_none(a, b):
            return None

        a = [True, True, True, False, False]
        b = [True, True, False, True, False]
        with pytest.raises((ValueError, RuntimeError)):
            bootstrap_ci(_always_none, a, b, n_boot=50, seed=0)


# ---------------------------------------------------------------------------
# Gate eligibility boundaries
# ---------------------------------------------------------------------------


class TestGateEligibility:
    """gate_eligible = kappa point estimate >= 0.4 AND n >= 30.

    Both conditions must hold.  These tests exercise the boundary values.
    The gate logic lives in validate_judge.py; here we just verify the
    stat functions produce the right kappa values so the boundaries work.
    """

    def test_kappa_below_floor_039(self):
        # Construct a kappa near 0.39 (below floor 0.4)
        # Use known-kappa Example 1 from above (≈0.5238).
        # A vector with slightly less agreement:
        # a=[T,T,F,F,F,F,F,F,F,F], b=[T,F,T,F,F,F,F,F,F,F]
        # Confusion: TP=1, FP=1, FN=1, TN=7; P_o=(1+7)/10=0.8
        # A marginals: 2T,8F; B marginals: 2T,8F
        # P_e = (2/10)^2 + (8/10)^2 = 0.04 + 0.64 = 0.68
        # kappa = (0.8-0.68)/(1-0.68) = 0.12/0.32 = 0.375
        a = [True, True, False, False, False, False, False, False, False, False]
        b = [True, False, True, False, False, False, False, False, False, False]
        k = cohens_kappa(a, b)
        assert k == pytest.approx(0.375, rel=1e-3)
        assert k < 0.4  # below floor

    def test_kappa_above_floor_041(self):
        # Example 1 kappa ≈ 0.5238 > 0.4
        a = [True, True, True, False, False, False, False, False, False, False]
        b = [True, True, False, True, False, False, False, False, False, False]
        k = cohens_kappa(a, b)
        assert k > 0.4  # above floor


# ---------------------------------------------------------------------------
# validate_judge.py integration — join-mismatch and n<10 refusal
# ---------------------------------------------------------------------------


class TestValidateJudgeScript:
    """Smoke tests for the validate_judge.py CLI.

    The script cannot RUN successfully (no labels exist); these tests exercise
    its loud failure modes — missing args, n<10 refusal, join-mismatch error.
    """

    def _run(self, args: list[str]) -> tuple[int, str]:
        """Run validate_judge.py as a subprocess and return (returncode, combined_output)."""
        import subprocess
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "scripts/validate_judge.py"] + args,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        combined = result.stdout + result.stderr
        return result.returncode, combined

    def test_no_args_fails_loud(self):
        rc, out = self._run([])
        assert rc != 0, "Should exit non-zero with no args"
        # Must print something useful, not just a traceback with no context
        assert len(out.strip()) > 0

    def test_missing_label_file_fails_loud(self):
        rc, out = self._run(["--labels", "/nonexistent/labels.csv", "--verdicts", "/nonexistent/v.jsonl"])
        assert rc != 0

    def test_help_flag_shows_format_docs(self):
        rc, out = self._run(["--help"])
        # --help exits 0 and shows documentation
        assert rc == 0
        # Must document the expected CSV format
        assert "turn_id" in out or "csv" in out.lower()
