"""Tests for the soft-evidence BKT core (T3.1).

TDD: tests written before implementation.  Run `uv run pytest tests/test_bkt.py -v`
to see the initial red → green after src/readcoach/bkt.py exists.
"""

import math
import pytest

from readcoach.bkt import BktParams, bkt_update, update_sequence


# ---------------------------------------------------------------------------
# 1. conf=1.0 — textbook BKT equivalence (hand-computed)
# ---------------------------------------------------------------------------

class TestTextbookEquivalence:
    """conf=1.0 collapses virtual evidence to standard BKT.

    Worked example 1
    ----------------
    p_L=0.3, correct=True, s=0.1, g=0.2, t=0.1, conf=1.0
      p_obs_L  = 1 - s     = 0.9      (P(correct | knows)   = 1-slip)
      p_obs_nL = g         = 0.2      (P(correct | !knows)  = guess)
      like_L   = 1.0*0.9 + 0.0*0.1   = 0.9
      like_nL  = 1.0*0.2 + 0.0*0.8   = 0.2
      numer    = p_L * like_L         = 0.3 * 0.9  = 0.27
      denom    = 0.27 + 0.7*0.2       = 0.27 + 0.14 = 0.41
      post     = 0.27 / 0.41
      result   = post + (1 - post) * 0.1

    Worked example 2
    ----------------
    p_L=0.5, correct=False, s=0.15, g=0.25, t=0.05, conf=1.0
      p_obs_L  = s          = 0.15    (P(incorrect | knows)  = slip)
      p_obs_nL = 1 - g      = 0.75    (P(incorrect | !knows) = 1-guess)
      like_L   = 1.0*0.15 + 0.0*0.85 = 0.15
      like_nL  = 1.0*0.75 + 0.0*0.25 = 0.75
      numer    = 0.5 * 0.15           = 0.075
      denom    = 0.075 + 0.5*0.75     = 0.075 + 0.375 = 0.45
      post     = 0.075 / 0.45 = 1/6
      result   = (1/6) + (5/6) * 0.05
    """

    def test_example1_correct_observation(self):
        post = 0.27 / 0.41
        expected = post + (1 - post) * 0.1
        result = bkt_update(p_L=0.3, correct=True, conf=1.0, s=0.1, g=0.2, t=0.1)
        assert math.isclose(result, expected, rel_tol=1e-12)

    def test_example2_incorrect_observation(self):
        post = 0.075 / 0.45  # = 1/6
        expected = post + (1 - post) * 0.05
        result = bkt_update(p_L=0.5, correct=False, conf=1.0, s=0.15, g=0.25, t=0.05)
        assert math.isclose(result, expected, rel_tol=1e-12)

    def test_example1_exact_formula_value(self):
        """Second assertion: formula-computed float matches observed float."""
        post = 0.27 / 0.41
        expected = post + (1 - post) * 0.1
        # ~0.6927 — assert range as sanity check
        assert 0.69 < expected < 0.70

    def test_example2_exact_formula_value(self):
        post = 0.075 / 0.45
        expected = post + (1 - post) * 0.05
        # post = 1/6 ≈ 0.1667, result ≈ 0.2083
        assert math.isclose(expected, 1 / 6 + (5 / 6) * 0.05, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 2. conf=0.5 — information-free: posterior == prior, transit only
# ---------------------------------------------------------------------------

class TestInformationFreeConf:
    """When conf=0.5 (max uncertainty about the label), the likelihoods are:
        like_L  = 0.5*p_obs + 0.5*(1-p_obs) = 0.5  (regardless of p_obs)
        like_nL = 0.5
    The Bayes update is the identity (posterior = prior).
    The only movement is from the transition term:
        result = p_L + (1 - p_L) * t
    """

    @pytest.mark.parametrize("p_L", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    @pytest.mark.parametrize("correct", [True, False])
    def test_conf_half_is_identity(self, p_L, correct):
        t = 0.1
        expected = p_L + (1 - p_L) * t
        result = bkt_update(p_L=p_L, correct=correct, conf=0.5, s=0.1, g=0.2, t=t)
        assert math.isclose(result, expected, rel_tol=1e-12, abs_tol=1e-15)

    @pytest.mark.parametrize("t", [0.0, 0.05, 0.2])
    def test_conf_half_varies_t(self, t):
        p_L = 0.4
        expected = p_L + (1 - p_L) * t
        assert math.isclose(
            bkt_update(p_L, True, 0.5, 0.1, 0.2, t), expected, rel_tol=1e-12
        )


# ---------------------------------------------------------------------------
# 3. Monotonicity in conf
# ---------------------------------------------------------------------------

class TestMonotonicityInConf:
    """When (1-s) > g — i.e., a correct response is genuinely evidence of mastery
    (the non-degenerate regime enforced by s<0.5, g<0.5) — the posterior after a
    CORRECT response strictly increases with conf over (0.5, 1.0].
    After an INCORRECT response it strictly decreases with conf over (0.5, 1.0].

    Intuition: higher conf means the label is more trustworthy, so a correct
    response pushes the posterior up more, and an incorrect one pushes it down more.
    """

    def _grid(self):
        """51 evenly-spaced conf values from 0.5 to 1.0."""
        n = 51
        return [0.5 + 0.5 * i / (n - 1) for i in range(n)]

    def test_correct_obs_strictly_increases_with_conf(self):
        p_L, s, g, t = 0.4, 0.1, 0.2, 0.05
        vals = [bkt_update(p_L, True, c, s, g, t) for c in self._grid()]
        for a, b in zip(vals, vals[1:]):
            assert b > a, f"Non-monotone at transition: {a} -> {b}"

    def test_incorrect_obs_strictly_decreases_with_conf(self):
        p_L, s, g, t = 0.6, 0.1, 0.2, 0.05
        vals = [bkt_update(p_L, False, c, s, g, t) for c in self._grid()]
        for a, b in zip(vals, vals[1:]):
            assert b < a, f"Non-monotone at transition: {a} -> {b}"

    def test_monotone_for_multiple_p_L_values(self):
        s, g, t = 0.05, 0.15, 0.03
        grid = self._grid()
        for p_L in [0.1, 0.3, 0.5, 0.7, 0.9]:
            vals_correct = [bkt_update(p_L, True, c, s, g, t) for c in grid]
            vals_wrong = [bkt_update(p_L, False, c, s, g, t) for c in grid]
            for a, b in zip(vals_correct, vals_correct[1:]):
                assert b > a
            for a, b in zip(vals_wrong, vals_wrong[1:]):
                assert b < a


# ---------------------------------------------------------------------------
# 4. conf < 0.5 — label-flip regime; symmetry identity
# ---------------------------------------------------------------------------

class TestSubHalfConf:
    """conf < 0.5 means P(the reported label is correct) < 0.5 — the caller
    believes the label is more likely WRONG than right.  The formula remains
    well-defined.

    Algebraic identity (holds exactly):
        bkt_update(p, True, 1-c, s, g, t) == bkt_update(p, False, c, s, g, t)

    Proof sketch: for correct=True the likelihoods are
        like_L  = (1-c)*(1-s) + c*s
        like_nL = (1-c)*g + c*(1-g)
    For correct=False with conf=c:
        like_L  = c*s + (1-c)*(1-s)    ← identical to above
        like_nL = c*(1-g) + (1-c)*g    ← identical to above
    Equality of likelihood ratios implies equality of posteriors.
    """

    @pytest.mark.parametrize("c", [0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0])
    @pytest.mark.parametrize("p_L", [0.1, 0.4, 0.7])
    def test_symmetry_identity(self, p_L, c):
        """bkt_update(p, True, 1-c, ...) == bkt_update(p, False, c, ...)"""
        s, g, t = 0.1, 0.2, 0.1
        a = bkt_update(p_L, True, 1 - c, s, g, t)
        b = bkt_update(p_L, False, c, s, g, t)
        assert math.isclose(a, b, rel_tol=1e-12), f"p={p_L}, c={c}: {a} != {b}"

    def test_conf_below_half_correct_pushes_posterior_down_vs_conf_half(self):
        """conf=0.3 on correct=True: caller says label is likely wrong, so
        posterior should be LOWER than the pure-transit value (conf=0.5)."""
        p_L, s, g, t = 0.4, 0.1, 0.2, 0.05
        result_low_conf = bkt_update(p_L, True, 0.3, s, g, t)
        result_info_free = bkt_update(p_L, True, 0.5, s, g, t)
        assert result_low_conf < result_info_free

    def test_conf_below_half_incorrect_pushes_posterior_up_vs_conf_half(self):
        """conf=0.3 on correct=False: a 'wrong' label becomes weak evidence OF mastery."""
        p_L, s, g, t = 0.6, 0.1, 0.2, 0.05
        result_low_conf = bkt_update(p_L, False, 0.3, s, g, t)
        result_info_free = bkt_update(p_L, False, 0.5, s, g, t)
        assert result_low_conf > result_info_free


# ---------------------------------------------------------------------------
# 5. Boundary conditions and transit-order assertion
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    """Asserts exact behaviour at extreme inputs and the evidence-then-transit
    ordering mandated by the frozen formula."""

    def test_p_L_zero_gives_t(self):
        """p_L=0: post=0 (zero numerator), result = 0 + 1*t = t."""
        t = 0.1
        result = bkt_update(p_L=0.0, correct=True, conf=1.0, s=0.1, g=0.2, t=t)
        assert math.isclose(result, t, rel_tol=1e-12)

    def test_p_L_zero_correct_false_gives_t(self):
        t = 0.07
        result = bkt_update(p_L=0.0, correct=False, conf=1.0, s=0.1, g=0.2, t=t)
        assert math.isclose(result, t, rel_tol=1e-12)

    def test_p_L_one_stays_one(self):
        """p_L=1: post=1 (full numerator), result = 1 + 0*t = 1."""
        result = bkt_update(p_L=1.0, correct=True, conf=1.0, s=0.1, g=0.2, t=0.1)
        assert math.isclose(result, 1.0, rel_tol=1e-12)

    def test_p_L_one_correct_false_stays_one(self):
        result = bkt_update(p_L=1.0, correct=False, conf=1.0, s=0.1, g=0.2, t=0.1)
        assert math.isclose(result, 1.0, rel_tol=1e-12)

    def test_t_zero_returns_pure_posterior(self):
        """t=0: transit term vanishes; result == post."""
        p_L, s, g = 0.3, 0.1, 0.2
        post = 0.27 / 0.41  # from hand-worked example 1
        result = bkt_update(p_L=p_L, correct=True, conf=1.0, s=s, g=g, t=0.0)
        assert math.isclose(result, post, rel_tol=1e-12)

    def test_evidence_before_transit_order(self):
        """The frozen formula applies Bayes update FIRST, then adds transit.
        If transit were applied before evidence, the result would differ.
        Verify by checking result == post + (1-post)*t (not transit-first form)."""
        p_L, s, g, t = 0.3, 0.1, 0.2, 0.1
        post = 0.27 / 0.41
        expected_evidence_then_transit = post + (1 - post) * t

        # If transit were FIRST: p_L' = p_L + (1-p_L)*t = 0.3+0.07=0.37
        # Then evidence: numer=0.37*0.9=0.333, denom=0.333+0.63*0.2=0.459
        p_L_after_transit = p_L + (1 - p_L) * t
        post_transit_first = (p_L_after_transit * 0.9) / (
            p_L_after_transit * 0.9 + (1 - p_L_after_transit) * 0.2
        )

        result = bkt_update(p_L=p_L, correct=True, conf=1.0, s=s, g=g, t=t)
        assert math.isclose(result, expected_evidence_then_transit, rel_tol=1e-12)
        # The two orderings must yield different values (confirms test is discriminating)
        assert not math.isclose(result, post_transit_first, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 6. update_sequence
# ---------------------------------------------------------------------------

class TestUpdateSequence:
    """update_sequence must return a trajectory of length len(observations)+1,
    starting from p_L0, and each element must match the result of calling
    bkt_update manually in sequence."""

    def test_trajectory_length(self):
        params = BktParams(s=0.1, g=0.2, t=0.05, L0=0.3)
        obs = [(True, 1.0), (False, 0.8), (True, 0.9)]
        traj = update_sequence(0.3, obs, params)
        assert len(traj) == len(obs) + 1  # initial + one per observation

    def test_trajectory_starts_at_p_L0(self):
        params = BktParams(s=0.1, g=0.2, t=0.05, L0=0.3)
        traj = update_sequence(0.4, [(True, 1.0)], params)
        assert math.isclose(traj[0], 0.4)

    def test_trajectory_matches_manual_sequence(self):
        params = BktParams(s=0.1, g=0.2, t=0.05, L0=0.3)
        obs = [(True, 1.0), (False, 0.8), (True, 0.6), (False, 0.9)]
        traj = update_sequence(0.3, obs, params)

        p = 0.3
        assert math.isclose(traj[0], p, rel_tol=1e-12)
        for i, (correct, conf) in enumerate(obs):
            p = bkt_update(p, correct, conf, params.s, params.g, params.t)
            assert math.isclose(traj[i + 1], p, rel_tol=1e-12), (
                f"Mismatch at step {i + 1}: traj={traj[i + 1]}, manual={p}"
            )

    def test_empty_observations(self):
        params = BktParams(s=0.1, g=0.2, t=0.05, L0=0.3)
        traj = update_sequence(0.5, [], params)
        assert traj == [0.5]

    def test_composition_equals_single_update(self):
        """Two-step composition must equal two direct calls."""
        params = BktParams(s=0.1, g=0.2, t=0.1, L0=0.3)
        p0 = 0.3
        traj = update_sequence(p0, [(True, 1.0), (False, 0.7)], params)

        p1 = bkt_update(p0, True, 1.0, params.s, params.g, params.t)
        p2 = bkt_update(p1, False, 0.7, params.s, params.g, params.t)
        assert math.isclose(traj[1], p1, rel_tol=1e-12)
        assert math.isclose(traj[2], p2, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 7. Validation — out-of-range params raise ValueError
# ---------------------------------------------------------------------------

class TestValidation:
    """BktParams and bkt_update must reject degenerate or out-of-range inputs."""

    # --- BktParams structural validation ---

    @pytest.mark.parametrize("s", [-0.01, 1.01, 0.5, 0.6])
    def test_invalid_s_raises(self, s):
        with pytest.raises(ValueError, match="s"):
            BktParams(s=s, g=0.2, t=0.1, L0=0.3)

    @pytest.mark.parametrize("g", [-0.01, 1.01, 0.5, 0.7])
    def test_invalid_g_raises(self, g):
        with pytest.raises(ValueError, match="g"):
            BktParams(s=0.1, g=g, t=0.1, L0=0.3)

    def test_s_exactly_half_raises(self):
        with pytest.raises(ValueError):
            BktParams(s=0.5, g=0.2, t=0.1, L0=0.3)

    def test_g_exactly_half_raises(self):
        with pytest.raises(ValueError):
            BktParams(s=0.1, g=0.5, t=0.1, L0=0.3)

    @pytest.mark.parametrize("t", [-0.01, 1.01])
    def test_invalid_t_raises(self, t):
        with pytest.raises(ValueError, match="t"):
            BktParams(s=0.1, g=0.2, t=t, L0=0.3)

    @pytest.mark.parametrize("L0", [-0.01, 1.01])
    def test_invalid_L0_raises(self, L0):
        with pytest.raises(ValueError, match="L0"):
            BktParams(s=0.1, g=0.2, t=0.1, L0=L0)

    def test_valid_params_do_not_raise(self):
        p = BktParams(s=0.1, g=0.2, t=0.05, L0=0.3)
        assert p.s == 0.1

    def test_boundary_valid_s_near_zero(self):
        BktParams(s=0.0, g=0.2, t=0.1, L0=0.3)  # should not raise

    def test_boundary_valid_g_near_zero(self):
        BktParams(s=0.1, g=0.0, t=0.1, L0=0.3)  # should not raise

    # --- bkt_update input validation ---

    @pytest.mark.parametrize("p_L", [-0.01, 1.01])
    def test_invalid_p_L_raises(self, p_L):
        with pytest.raises(ValueError, match="p_L"):
            bkt_update(p_L=p_L, correct=True, conf=1.0, s=0.1, g=0.2, t=0.1)

    @pytest.mark.parametrize("conf", [-0.01, 1.01])
    def test_invalid_conf_raises(self, conf):
        with pytest.raises(ValueError, match="conf"):
            bkt_update(p_L=0.3, correct=True, conf=conf, s=0.1, g=0.2, t=0.1)

    def test_p_L_zero_and_one_are_valid(self):
        """Boundary values 0 and 1 for p_L must be accepted."""
        bkt_update(0.0, True, 1.0, 0.1, 0.2, 0.1)
        bkt_update(1.0, True, 1.0, 0.1, 0.2, 0.1)

    def test_conf_zero_and_one_are_valid(self):
        """Boundary values 0 and 1 for conf must be accepted."""
        bkt_update(0.3, True, 0.0, 0.1, 0.2, 0.1)
        bkt_update(0.3, True, 1.0, 0.1, 0.2, 0.1)
