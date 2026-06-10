"""Tests for the BKT fitting / parameter-recovery layer (T3.2).

TDD: written before ``src/readcoach/bkt_fit.py`` exists.  Run
``uv run pytest tests/test_bkt_fit.py -v`` to see the initial red, then green
once the module lands.

The central correctness concern this file guards is the **event-order
agreement** between the generative process (:func:`simulate`) and the model
that scores / fits it (:func:`log_likelihood`, :func:`posterior_trajectory`,
which reuse the frozen :func:`readcoach.bkt.bkt_update`).  Both use
*evidence-then-transit* semantics: the observation at opportunity ``i`` is
emitted from the mastery state BEFORE that opportunity's transit, and the
transit is applied afterwards.  A mismatch here is the classic silent killer
in knowledge-tracing experiments, so it is asserted explicitly.
"""

import math

import numpy as np
import pytest

from readcoach.bkt import BktParams, bkt_update
from readcoach.bkt_fit import (
    fit_grid,
    log_likelihood,
    posterior_trajectory,
    predict_correct,
    simulate,
)


# ---------------------------------------------------------------------------
# 1. simulate — generative process
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_shapes_and_dtype(self):
        params = BktParams(s=0.1, g=0.2, t=0.15, L0=0.4)
        rng = np.random.default_rng(0)
        obs, latent = simulate(params, n_students=37, n_opportunities=11, rng=rng)
        assert obs.shape == (37, 11)
        assert latent.shape == (37, 11)
        assert obs.dtype == np.bool_
        assert latent.dtype == np.bool_

    def test_determinism_with_seed(self):
        params = BktParams(s=0.1, g=0.2, t=0.15, L0=0.4)
        obs1, lat1 = simulate(params, 50, 12, np.random.default_rng(2026))
        obs2, lat2 = simulate(params, 50, 12, np.random.default_rng(2026))
        assert np.array_equal(obs1, obs2)
        assert np.array_equal(lat1, lat2)

    def test_different_seeds_differ(self):
        params = BktParams(s=0.1, g=0.2, t=0.15, L0=0.4)
        obs1, _ = simulate(params, 50, 12, np.random.default_rng(1))
        obs2, _ = simulate(params, 50, 12, np.random.default_rng(2))
        assert not np.array_equal(obs1, obs2)

    def test_mastery_is_absorbing(self):
        """Once a student's latent state is True it must never return to False."""
        params = BktParams(s=0.1, g=0.2, t=0.3, L0=0.3)
        _, latent = simulate(params, 500, 20, np.random.default_rng(7))
        # For every row, the cumulative-max must equal the row itself: monotone.
        cummax = np.maximum.accumulate(latent, axis=1)
        assert np.array_equal(cummax, latent)

    def test_mastered_correct_rate_approx_one_minus_s(self):
        """Among observations emitted from a MASTERED state, the empirical
        correct-rate must be ≈ 1 - s (slip)."""
        s = 0.12
        params = BktParams(s=s, g=0.2, t=0.25, L0=0.5)
        obs, latent = simulate(params, 4000, 20, np.random.default_rng(11))
        mastered_obs = obs[latent]
        rate = mastered_obs.mean()
        assert abs(rate - (1 - s)) < 0.01, f"mastered correct-rate {rate} vs {1 - s}"

    def test_unmastered_correct_rate_approx_g(self):
        """Among observations emitted from an UNMASTERED state, the empirical
        correct-rate must be ≈ g (guess)."""
        g = 0.22
        params = BktParams(s=0.1, g=g, t=0.05, L0=0.1)
        obs, latent = simulate(params, 4000, 20, np.random.default_rng(13))
        unmastered_obs = obs[~latent]
        rate = unmastered_obs.mean()
        assert abs(rate - g) < 0.01, f"unmastered correct-rate {rate} vs {g}"

    def test_initial_mastery_rate_approx_L0(self):
        L0 = 0.4
        params = BktParams(s=0.1, g=0.2, t=0.2, L0=L0)
        _, latent = simulate(params, 5000, 8, np.random.default_rng(17))
        assert abs(latent[:, 0].mean() - L0) < 0.02

    def test_event_order_observation_from_pre_transit_state(self):
        """Evidence-then-transit edge: with t=1.0 the second opportunity is
        always emitted from a MASTERED state (everyone transits after obs 0),
        yet the FIRST observation must still reflect the L0 (pre-transit) state.

        Concretely with L0=0 and t=1.0:
          * obs 0 is emitted from the unmastered state for EVERY student
            (latent[:,0] all False),
          * obs 1 onward is emitted from the mastered state (latent all True),
        proving the emission at opportunity i uses the state BEFORE that
        opportunity's transit, not after.
        """
        params = BktParams(s=0.05, g=0.2, t=1.0, L0=0.0)
        obs, latent = simulate(params, 3000, 3, np.random.default_rng(19))
        # Column 0: nobody mastered yet (L0=0 and transit happens AFTER obs 0).
        assert not latent[:, 0].any()
        # Columns 1,2: everyone mastered (t=1 transits all unmastered after obs0).
        assert latent[:, 1].all()
        assert latent[:, 2].all()
        # Emission consistency: col-0 correct-rate ≈ g (unmastered),
        # col-1 correct-rate ≈ 1-s (mastered).
        assert abs(obs[:, 0].mean() - params.g) < 0.03
        assert abs(obs[:, 1].mean() - (1 - params.s)) < 0.02


# ---------------------------------------------------------------------------
# 2. log_likelihood — exact forward computation
# ---------------------------------------------------------------------------

class TestLogLikelihood:
    """Forward (HMM marginal) likelihood for the 2-state absorbing BKT chain.

    Hand-computable case — 1 student, 2 opportunities, obs=[correct, incorrect],
    params s=0.1, g=0.2, t=0.3, L0=0.4.

    Forward algebra (evidence-then-transit):
      Before obs 0:  P(M)=L0=0.4,  P(¬M)=0.6
      Obs 0 = correct:  P(c|M)=1-s=0.9,  P(c|¬M)=g=0.2
        joint_M = 0.4*0.9 = 0.36 ; joint_¬M = 0.6*0.2 = 0.12
        P(obs0)  = 0.48
        post P(M | obs0) = 0.36/0.48 = 0.75
      Transit (¬M→M w.p. t):  P(M) = 0.75 + 0.25*0.3 = 0.825
      Obs 1 = incorrect:  P(i|M)=s=0.1,  P(i|¬M)=1-g=0.8
        joint_M = 0.825*0.1 = 0.0825 ; joint_¬M = 0.175*0.8 = 0.14
        P(obs1|obs0) = 0.2225
      L = 0.48 * 0.2225 = 0.1068
      log L = ln(0.1068) = -2.2367973524560423
    """

    def test_hand_computable_single_student(self):
        obs = np.array([[True, False]], dtype=bool)
        ll = log_likelihood(obs, s=0.1, g=0.2, t=0.3, L0=0.4)
        assert math.isclose(ll, math.log(0.1068), rel_tol=1e-12)
        assert math.isclose(ll, -2.2367973524560423, rel_tol=1e-12)

    def test_additive_over_independent_students(self):
        """Two identical independent students → exactly 2× the single-student ll."""
        obs1 = np.array([[True, False]], dtype=bool)
        obs2 = np.array([[True, False], [True, False]], dtype=bool)
        ll1 = log_likelihood(obs1, s=0.1, g=0.2, t=0.3, L0=0.4)
        ll2 = log_likelihood(obs2, s=0.1, g=0.2, t=0.3, L0=0.4)
        assert math.isclose(ll2, 2 * ll1, rel_tol=1e-12)

    def test_higher_at_true_than_wrong_params(self):
        """On data simulated from a known truth, the true params score higher
        log-likelihood than a clearly-wrong point (statistical, seeded)."""
        true = BktParams(s=0.1, g=0.2, t=0.2, L0=0.4)
        obs, _ = simulate(true, 400, 15, np.random.default_rng(2026))
        ll_true = log_likelihood(obs, true.s, true.g, true.t, true.L0)
        # A clearly-wrong point: swap the learning dynamics and priors.
        ll_wrong = log_likelihood(obs, s=0.4, g=0.45, t=0.9, L0=0.05)
        assert ll_true > ll_wrong

    def test_returns_python_float(self):
        obs = np.array([[True, False]], dtype=bool)
        ll = log_likelihood(obs, s=0.1, g=0.2, t=0.3, L0=0.4)
        assert isinstance(ll, float)


# ---------------------------------------------------------------------------
# 3. fit_grid — two-stage grid MLE
# ---------------------------------------------------------------------------

class TestFitGrid:
    def test_recovers_true_params_well_conditioned(self):
        """On a modest dataset from a well-conditioned regime, the fit recovers
        each parameter within grid resolution + sampling tolerance."""
        true = BktParams(s=0.08, g=0.15, t=0.15, L0=0.4)
        obs, _ = simulate(true, 100, 15, np.random.default_rng(2026))
        fit = fit_grid(obs)
        for name, true_v, fit_v in (
            ("s", true.s, fit.s),
            ("g", true.g, fit.g),
            ("t", true.t, fit.t),
            ("L0", true.L0, fit.L0),
        ):
            assert abs(fit_v - true_v) <= 0.07, (
                f"{name}: fit {fit_v} vs true {true_v}"
            )

    def test_determinism_same_data_same_fit(self):
        true = BktParams(s=0.08, g=0.15, t=0.15, L0=0.4)
        obs, _ = simulate(true, 100, 15, np.random.default_rng(2026))
        fit1 = fit_grid(obs)
        fit2 = fit_grid(obs)
        assert (fit1.s, fit1.g, fit1.t, fit1.L0) == (fit2.s, fit2.g, fit2.t, fit2.L0)

    def test_returns_bktparams(self):
        true = BktParams(s=0.08, g=0.15, t=0.15, L0=0.4)
        obs, _ = simulate(true, 60, 10, np.random.default_rng(3))
        fit = fit_grid(obs)
        assert isinstance(fit, BktParams)


# ---------------------------------------------------------------------------
# 4. posterior_trajectory + predict_correct
# ---------------------------------------------------------------------------

class TestPosteriorTrajectory:
    def test_first_value_is_L0(self):
        """The trajectory is P(L) BEFORE each observation; the first entry is the
        prior L0 for every student."""
        params = BktParams(s=0.1, g=0.2, t=0.3, L0=0.4)
        obs = np.array([[True, False, True]], dtype=bool)
        traj = posterior_trajectory(obs, params)
        assert traj.shape == (1, 3)
        assert math.isclose(traj[0, 0], 0.4, rel_tol=1e-12)

    def test_matches_manual_two_step_update(self):
        """Pre-observation P(L) at each step equals repeated bkt_update(conf=1)."""
        params = BktParams(s=0.1, g=0.2, t=0.3, L0=0.4)
        obs = np.array([[True, False]], dtype=bool)
        traj = posterior_trajectory(obs, params)
        # step 0: prior
        assert math.isclose(traj[0, 0], 0.4, rel_tol=1e-12)
        # step 1: after observing obs0=correct (evidence+transit) → 0.825
        manual1 = bkt_update(0.4, True, 1.0, params.s, params.g, params.t)
        assert math.isclose(traj[0, 1], manual1, rel_tol=1e-12)
        assert math.isclose(traj[0, 1], 0.825, rel_tol=1e-12)

    def test_vectorized_matches_per_student_loop(self):
        params = BktParams(s=0.12, g=0.18, t=0.2, L0=0.35)
        obs, _ = simulate(params, 25, 14, np.random.default_rng(5))
        traj = posterior_trajectory(obs, params)
        assert traj.shape == obs.shape
        # Reconstruct row 0 manually with the frozen scalar update.
        p = params.L0
        for k in range(obs.shape[1]):
            assert math.isclose(traj[0, k], p, rel_tol=1e-12)
            p = bkt_update(p, bool(obs[0, k]), 1.0, params.s, params.g, params.t)

    def test_predict_correct_formula(self):
        # predict = p*(1-s) + (1-p)*g
        assert math.isclose(predict_correct(0.4, 0.1, 0.2), 0.4 * 0.9 + 0.6 * 0.2)
        assert math.isclose(predict_correct(1.0, 0.1, 0.2), 0.9)
        assert math.isclose(predict_correct(0.0, 0.1, 0.2), 0.2)

    def test_predict_correct_vectorized(self):
        p = np.array([0.0, 0.5, 1.0])
        out = predict_correct(p, 0.1, 0.2)
        expected = p * 0.9 + (1 - p) * 0.2
        assert np.allclose(out, expected)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
