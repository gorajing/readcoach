"""Tests for the break-even experiment (T3.3).

TDD: written before ``scripts/break_even.py`` exists.  Run
``uv run pytest tests/test_break_even.py -v`` to see the initial red, then green
once the script lands.

What these guard (the load-bearing correctness concerns for this experiment):

* **Channel**: the noise channel flips a TRUE correctness label to its opposite
  with probability ``1-a`` (channel accuracy ``a``).  Empirically, over many
  draws, the flip rate must be ≈ ``1-a``.
* **Paired streams**: naive and soft updaters must run on the *identical*
  observed-label stream.  This is the whole point of a paired comparison — if
  the two updaters consumed different RNG draws, any RMSE difference would be
  confounded by sampling noise.  We assert they see byte-identical observations.
* **Selector gating**: the proxy item-selector must never pick skill ``k+1``
  while its prerequisite skill ``k`` has P(L) < 0.8 (the first skill is always
  available).
* **Regret property**: oracle (fed true latent) total true-mastery ≥ policy
  (fed posterior) total true-mastery on the same stream → regret ≥ 0.  A seed
  violating this is a finding about the proxy, investigated before asserting.
* **a_eff formula** hand-check against the documented closed form.
* **never-detected counting**: a student whose latent never reaches mastery, or
  whose posterior never crosses 0.95, is counted as never-detected (not silently
  dropped or assigned a bogus latency).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from readcoach.bkt import BktParams

# Import the experiment module from scripts/.  conftest.py adds the repo root to
# sys.path (pythonpath=["."] in pyproject), and scripts/ has no __init__, so we
# load by file path to keep the import robust regardless of packaging.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "break_even.py"
_spec = importlib.util.spec_from_file_location("break_even", _SCRIPT)
be = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(be)


# ---------------------------------------------------------------------------
# 1. Noise channel — flip rate ≈ 1 - a
# ---------------------------------------------------------------------------

class TestChannel:
    def test_flip_rate_matches_one_minus_a(self):
        rng = np.random.default_rng(0)
        true = rng.integers(0, 2, size=200_000).astype(bool)
        for a in (0.55, 0.7, 0.9, 0.99):
            obs = be.apply_channel(true, a, np.random.default_rng(1))
            flip_rate = float(np.mean(obs != true))
            assert abs(flip_rate - (1 - a)) < 0.005, (
                f"a={a}: flip rate {flip_rate} vs expected {1 - a}"
            )

    def test_a_equals_one_is_identity(self):
        rng = np.random.default_rng(2)
        true = rng.integers(0, 2, size=5000).astype(bool)
        obs = be.apply_channel(true, 1.0, np.random.default_rng(3))
        assert np.array_equal(obs, true)

    def test_channel_is_deterministic_under_seed(self):
        true = np.array([True, False, True, True, False] * 100, dtype=bool)
        o1 = be.apply_channel(true, 0.8, np.random.default_rng(99))
        o2 = be.apply_channel(true, 0.8, np.random.default_rng(99))
        assert np.array_equal(o1, o2)

    def test_returns_bool_same_shape(self):
        true = np.zeros((7, 13), dtype=bool)
        obs = be.apply_channel(true, 0.75, np.random.default_rng(4))
        assert obs.shape == (7, 13)
        assert obs.dtype == np.bool_


# ---------------------------------------------------------------------------
# 2. Paired streams — naive and soft consume the SAME observations
# ---------------------------------------------------------------------------

class TestPairing:
    def test_both_updaters_see_identical_observations(self):
        """run_paired must return the observation array it scored with both
        updaters; that array (and the latent) is shared, not regenerated."""
        params = BktParams(s=0.1, g=0.2, t=0.15, L0=0.3)
        res = be.run_paired(
            params, a=0.7, n_students=50, n_opps=15,
            rng=np.random.default_rng(123),
        )
        # The observed labels fed to BOTH updaters are returned for inspection.
        assert res["obs"].shape == (50, 15)
        # Re-scoring the SAME obs with each updater must reproduce the same
        # trajectories the run already computed (i.e. updaters are pure
        # functions of the shared obs array, no hidden RNG divergence).
        naive2 = be.posterior_trajectory_conf(res["obs"], params, conf=1.0)
        soft2 = be.posterior_trajectory_conf(res["obs"], params, conf=0.7)
        assert np.allclose(res["post_naive"], naive2)
        assert np.allclose(res["post_soft"], soft2)
        # And the two updaters genuinely differ at a<1 (else the test is vacuous).
        assert not np.allclose(res["post_naive"], res["post_soft"])

    def test_naive_equals_soft_when_a_is_one(self):
        """At a=1.0 the channel is perfect and conf=1.0; soft (conf=1.0) and
        naive (conf=1.0) coincide exactly."""
        params = BktParams(s=0.1, g=0.2, t=0.15, L0=0.3)
        res = be.run_paired(
            params, a=1.0, n_students=40, n_opps=12,
            rng=np.random.default_rng(7),
        )
        assert np.allclose(res["post_naive"], res["post_soft"])


# ---------------------------------------------------------------------------
# 3. posterior_trajectory_conf — the conf-parameterized vectorized updater
# ---------------------------------------------------------------------------

class TestPosteriorConf:
    def test_conf_one_matches_hard_trajectory(self):
        """conf=1.0 must equal the frozen hard-label posterior_trajectory."""
        from readcoach.bkt_fit import posterior_trajectory
        params = BktParams(s=0.12, g=0.18, t=0.2, L0=0.35)
        obs, _ = be.simulate(params, 30, 14, np.random.default_rng(5))
        hard = posterior_trajectory(obs, params)
        confd = be.posterior_trajectory_conf(obs, params, conf=1.0)
        assert np.allclose(hard, confd)

    def test_matches_scalar_bkt_update(self):
        from readcoach.bkt import bkt_update
        params = BktParams(s=0.1, g=0.2, t=0.3, L0=0.4)
        obs = np.array([[True, False, True]], dtype=bool)
        traj = be.posterior_trajectory_conf(obs, params, conf=0.7)
        p = params.L0
        for k in range(obs.shape[1]):
            assert abs(traj[0, k] - p) < 1e-12
            p = bkt_update(p, bool(obs[0, k]), 0.7, params.s, params.g, params.t)


# ---------------------------------------------------------------------------
# 4. Item-selection proxy selector — prerequisite gating
# ---------------------------------------------------------------------------

class TestSelector:
    def test_prereq_gating_respected(self):
        """Skill k+1 is never selected while P(L)_k < 0.8.  Construct beliefs
        where only skill 0 is unlocked and verify the selector picks within the
        unlocked prefix."""
        # 5 skills; only skill 0 mastered-enough to unlock skill 1? No: skill 0
        # at 0.5 (<0.8) → only skill 0 is available (first skill always avail).
        beliefs = np.array([[0.5, 0.1, 0.1, 0.1, 0.1]])
        choice = be.select_item(beliefs)
        assert choice[0] == 0, "only skill 0 unlocked → must select skill 0"

        # skill 0 at 0.9 (≥0.8) unlocks skill 1; skill 1 at 0.2 (<0.8) keeps
        # skills 2..4 locked.  Available = {0, 1}; argmax gap (1-P) → skill 1
        # (gap 0.8) beats skill 0 (gap 0.1).
        beliefs = np.array([[0.9, 0.2, 0.05, 0.05, 0.05]])
        choice = be.select_item(beliefs)
        assert choice[0] == 1

    def test_locked_skill_never_chosen(self):
        """Even if a locked deep skill has the largest gap, it must not be
        picked while its prerequisite is below threshold."""
        # skill 0 = 0.95 unlocks 1; skill 1 = 0.5 (<0.8) → skills 2,3,4 LOCKED.
        # skill 4 has the biggest gap (0.99) but is locked.
        beliefs = np.array([[0.95, 0.5, 0.01, 0.01, 0.01]])
        choice = be.select_item(beliefs)
        assert choice[0] in (0, 1), f"locked skill chosen: {choice[0]}"
        assert choice[0] == 1  # gap(1)=0.5 > gap(0)=0.05

    def test_full_chain_unlocked_picks_global_max_gap(self):
        beliefs = np.array([[0.9, 0.85, 0.82, 0.81, 0.3]])
        # all prereqs ≥0.8 → skill 4 unlocked, gap 0.7 is largest.
        choice = be.select_item(beliefs)
        assert choice[0] == 4


# ---------------------------------------------------------------------------
# 5. Regret property — oracle ≥ policy → regret ≥ 0
# ---------------------------------------------------------------------------

class TestRegret:
    def test_regret_nonnegative_across_seeds(self):
        """The oracle selector (fed TRUE latent) should achieve at least as much
        true mastery at the horizon as the policy (fed posterior).  Regret ≥ 0
        is a property of the proxy; a violating seed is a finding to investigate,
        not to silently clamp.
        """
        params = BktParams(s=0.1, g=0.2, t=0.2, L0=0.2)
        for seed in (1, 2, 3, 11, 42):
            out = be.run_regret(
                params, a=0.75, n_students=120, n_opps=25,
                rng=np.random.default_rng(seed),
            )
            assert out["regret_naive"] >= -1e-9, (
                f"seed {seed}: naive regret {out['regret_naive']} < 0 — "
                "investigate the proxy before relaxing this."
            )
            assert out["regret_soft"] >= -1e-9, (
                f"seed {seed}: soft regret {out['regret_soft']} < 0 — "
                "investigate the proxy before relaxing this."
            )

    def test_regret_strictly_positive_at_low_a(self):
        """Non-vacuity: with a noisy channel (low a) the policy mis-selects and
        achieves strictly less true mastery than the oracle → positive regret.
        If this were ~0 everywhere the selector/metric would be inert."""
        params = BktParams(s=0.1, g=0.2, t=0.2, L0=0.2)
        out = be.run_regret(
            params, a=0.55, n_students=400, n_opps=25,
            rng=np.random.default_rng(2026),
        )
        assert out["regret_naive"] > 0.01, out["regret_naive"]


# ---------------------------------------------------------------------------
# 6. a_eff formula hand-check
# ---------------------------------------------------------------------------

class TestAEff:
    def test_a_eff_closed_form(self):
        """a_eff = (1-d)*(1 - fp_rate) + d*pooled_recall."""
        d = 0.0335
        fp_rate = 0.071
        pooled_recall = 0.792
        expected = (1 - d) * (1 - fp_rate) + d * pooled_recall
        got = be.effective_accuracy(
            miscue_density=d, fp_per_100=7.1, pooled_recall=pooled_recall
        )
        assert abs(got - expected) < 1e-9

    def test_a_eff_perfect_detector_is_one(self):
        """Zero false alarms + perfect recall → a_eff = 1.0 regardless of
        density."""
        got = be.effective_accuracy(
            miscue_density=0.1, fp_per_100=0.0, pooled_recall=1.0
        )
        assert abs(got - 1.0) < 1e-12

    def test_a_eff_in_unit_interval(self):
        got = be.effective_accuracy(
            miscue_density=0.05, fp_per_100=12.0, pooled_recall=0.3
        )
        assert 0.0 <= got <= 1.0


# ---------------------------------------------------------------------------
# 7. never-detected counting + detection latency
# ---------------------------------------------------------------------------

class TestDetection:
    def test_never_detected_when_posterior_never_crosses(self):
        """A flat posterior that never reaches 0.95 → detected index is None;
        a latent that never masters → true-mastery index is None.  Both feed the
        never-detected bookkeeping rather than a fabricated latency."""
        # posterior never crosses 0.95
        post = np.array([[0.1, 0.2, 0.3, 0.4]])
        latent = np.array([[False, False, True, True]])  # masters at k=2
        out = be.detection_latency(post, latent, threshold=0.95, horizon=4)
        assert out["n_never_detected"] == 1
        assert out["n_never_mastered"] == 0

    def test_never_mastered_counted(self):
        post = np.array([[0.96, 0.97, 0.98, 0.99]])  # crosses immediately
        latent = np.array([[False, False, False, False]])  # never masters
        out = be.detection_latency(post, latent, threshold=0.95, horizon=4)
        assert out["n_never_mastered"] == 1

    def test_signed_error_early_vs_late(self):
        """detect index minus true-mastered index: positive = late, negative =
        early.  A student detected at k=3 who truly mastered at k=1 has signed
        error +2 (detected 2 steps late)."""
        post = np.array([[0.1, 0.5, 0.9, 0.96]])  # crosses at k=3
        latent = np.array([[False, True, True, True]])  # masters at k=1
        out = be.detection_latency(post, latent, threshold=0.95, horizon=4)
        assert out["mean_signed_error"] == pytest.approx(2.0)
        assert out["mean_abs_error"] == pytest.approx(2.0)

    def test_both_never_excluded_from_latency_means(self):
        """A student who is never-mastered AND never-detected contributes to
        neither latency mean (no defined latency) but is counted in the
        never-* tallies."""
        post = np.array([[0.1, 0.2, 0.3, 0.4]])
        latent = np.array([[False, False, False, False]])
        out = be.detection_latency(post, latent, threshold=0.95, horizon=4)
        assert out["n_never_detected"] == 1
        assert out["n_never_mastered"] == 1
        assert np.isnan(out["mean_signed_error"])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
