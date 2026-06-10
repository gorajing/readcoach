"""BKT fitting & parameter recovery — T3.2 (hand-rolled numpy).

Block-0 fallback: pyBKT was dropped (import-dead in this environment), so the
fitting layer is a hand-rolled numpy implementation:

  * :func:`simulate`            — the standard BKT generative process.
  * :func:`log_likelihood`      — exact forward (HMM marginal) likelihood,
                                  vectorized over students.
  * :func:`fit_grid`            — two-stage (coarse → refine) grid MLE over
                                  the four parameters ``(s, g, t, L0)``.
  * :func:`posterior_trajectory`— pre-observation P(L) per opportunity, the
                                  quantity a tutor acts on, via the frozen
                                  :func:`readcoach.bkt.bkt_update`.
  * :func:`predict_correct`     — P(correct) given mastery prob, for calibration.

Event order — the silent killer in KT experiments
-------------------------------------------------
The generator and every model routine here share ONE convention, matching the
frozen :func:`readcoach.bkt.bkt_update` ("evidence first, then transit"):

    At opportunity ``i``:
      1. EMIT the observation from the mastery state that holds *before* this
         opportunity's transit (correct w.p. ``1-s`` if mastered else ``g``).
      2. TRANSIT: an unmastered student becomes mastered w.p. ``t``; mastery is
         absorbing (a mastered student stays mastered).

So the observation at opportunity ``i`` reflects the PRE-transit state, and the
posterior that :func:`bkt_update` returns after observation ``i`` is exactly the
prior P(L) going into observation ``i+1``.  The generator and the fitted model
therefore agree by construction; ``test_event_order_observation_from_pre_transit_state``
pins this down with a ``t=1.0`` edge case.

All randomness flows through a caller-supplied ``numpy.random.Generator`` so
experiments are fully seeded and deterministic.
"""

from __future__ import annotations

import numpy as np

from readcoach.bkt import BktParams, bkt_update

# ---------------------------------------------------------------------------
# Generative process
# ---------------------------------------------------------------------------


def simulate(
    params: BktParams,
    n_students: int,
    n_opportunities: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a synthetic BKT population.

    Standard BKT generative process with **evidence-then-transit** semantics
    (see module docstring): latent mastery starts ``Bernoulli(L0)``; at each
    opportunity the observation is emitted from the current (pre-transit) state,
    then an unmastered student transits to mastered with probability ``t``
    (mastery is absorbing).

    Parameters
    ----------
    params          : true :class:`BktParams`.
    n_students      : number of students (rows).
    n_opportunities : number of practice opportunities per student (columns).
    rng             : seeded ``numpy.random.Generator``; the sole entropy source.

    Returns
    -------
    (observations, latent) : both ``bool`` ndarrays of shape
        ``(n_students, n_opportunities)``.  ``observations[i, k]`` is whether
        student ``i``'s response at opportunity ``k`` was correct;
        ``latent[i, k]`` is the (hidden) mastery state from which that response
        was emitted (the pre-transit state at opportunity ``k``).
    """
    s, g, t, L0 = params.s, params.g, params.t, params.L0

    observations = np.empty((n_students, n_opportunities), dtype=bool)
    latent = np.empty((n_students, n_opportunities), dtype=bool)

    # Initial mastery ~ Bernoulli(L0).
    mastered = rng.random(n_students) < L0

    for k in range(n_opportunities):
        latent[:, k] = mastered

        # Emit from the PRE-transit state: correct w.p. 1-s if mastered else g.
        p_correct = np.where(mastered, 1.0 - s, g)
        observations[:, k] = rng.random(n_students) < p_correct

        # Transit AFTER the observation: unmastered → mastered w.p. t (absorbing).
        newly = (~mastered) & (rng.random(n_students) < t)
        mastered = mastered | newly

    return observations, latent


# ---------------------------------------------------------------------------
# Exact forward likelihood (vectorized over students)
# ---------------------------------------------------------------------------


def log_likelihood(
    observations: np.ndarray,
    s: float,
    g: float,
    t: float,
    L0: float,
) -> float:
    """Exact log-likelihood of an observation matrix under BKT params.

    Runs the forward algorithm for the 2-state absorbing BKT HMM, vectorized
    over students (the only Python loop is over opportunities — never over
    students).  Fitting uses HARD labels, so every observation is treated as a
    fully-confident correct/incorrect outcome.

    Parameters
    ----------
    observations : ``bool`` ndarray ``(n_students, n_opportunities)``.
    s, g, t, L0  : BKT parameters.

    Returns
    -------
    float
        Sum over students of the per-student forward log-likelihood.
    """
    obs = np.asarray(observations, dtype=bool)
    n_students, n_opps = obs.shape

    # Per-state emission probabilities for a correct response.
    #   P(correct | mastered)   = 1 - s
    #   P(correct | unmastered) = g
    # Prior over the mastered state, per student (all start at L0).
    p_mastered = np.full(n_students, L0, dtype=np.float64)
    total_log_lik = 0.0

    for k in range(n_opps):
        correct = obs[:, k]
        # Emission likelihood of THIS observation under each latent state.
        like_m = np.where(correct, 1.0 - s, s)        # P(obs | mastered)
        like_u = np.where(correct, g, 1.0 - g)        # P(obs | unmastered)

        # Marginal P(obs at k | history) — the forward normalizer.
        p_obs = p_mastered * like_m + (1.0 - p_mastered) * like_u
        # Valid params (s<0.5, g<0.5) keep both like_* > 0, so p_obs > 0.
        total_log_lik += float(np.sum(np.log(p_obs)))

        # Posterior over mastered AFTER this observation.
        post_mastered = (p_mastered * like_m) / p_obs
        # Transit: unmastered → mastered w.p. t (mastered absorbing).
        p_mastered = post_mastered + (1.0 - post_mastered) * t

    return total_log_lik


# ---------------------------------------------------------------------------
# Two-stage grid MLE
# ---------------------------------------------------------------------------

# Search bounds.  s and g are capped strictly below 0.5 (BktParams rejects ≥0.5
# as identifiability-breaking) and floored strictly above 0: s=0 or g=0 makes
# an emission probability exactly 0, so any observation contradicting it yields
# P(obs)=0 → log(0)=-inf and a 0/0 NaN in the posterior that would propagate
# forward.  Excluding the degenerate edges keeps every grid point inside the
# interior where the forward pass is finite (zero slip / zero guess are not
# realistic regimes anyway).  t and L0 span the full [0, 1] safely, since the
# emission likelihoods depend only on s and g.
_S_MIN = 0.01
_S_MAX = 0.48
_G_MIN = 0.01
_G_MAX = 0.48


def _best_on_grid(
    observations: np.ndarray,
    s_vals: np.ndarray,
    g_vals: np.ndarray,
    t_vals: np.ndarray,
    l0_vals: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return the (s, g, t, L0) maximizing log-likelihood over the product grid.

    Ties are broken deterministically: candidates are evaluated in a fixed
    nested order (s outer, then g, then t, then L0, each ascending), and a
    candidate replaces the incumbent only on a STRICT improvement.  The first
    grid point reaching the maximal log-likelihood therefore wins — i.e. the
    lexicographically-smallest ``(s, g, t, L0)`` among the argmax set.
    """
    best_ll = -np.inf
    best = (float(s_vals[0]), float(g_vals[0]), float(t_vals[0]), float(l0_vals[0]))
    for s in s_vals:
        for g in g_vals:
            for t in t_vals:
                for l0 in l0_vals:
                    ll = log_likelihood(observations, float(s), float(g), float(t), float(l0))
                    if ll > best_ll:
                        best_ll = ll
                        best = (float(s), float(g), float(t), float(l0))
    return best


def _clip_axis(center: float, lo: float, hi: float, step: float) -> np.ndarray:
    """Refine grid: ``center ± step`` at resolution ``step``, clipped to [lo, hi]."""
    vals = np.round(np.arange(center - step, center + step + 1e-9, step), 6)
    return np.unique(np.clip(vals, lo, hi))


def fit_grid(observations: np.ndarray) -> BktParams:
    """Two-stage grid-search maximum-likelihood fit of ``(s, g, t, L0)``.

    Stage 1 (coarse) — step 0.05 over::

        s  ∈ {0.01, 0.06, …, 0.46}   (10 values, 0 < s < 0.5)
        g  ∈ {0.01, 0.06, …, 0.46}   (10 values, 0 < g < 0.5)
        t  ∈ {0.00, 0.05, …, 1.00}   (21 values)
        L0 ∈ {0.00, 0.05, …, 1.00}   (21 values)

      → 10·10·21·21 = 44 100 forward-likelihood evaluations.  ``s`` and ``g``
      start at 0.01 (not 0) so every emission probability stays strictly
      positive and the forward log-likelihood is always finite.

    Stage 2 (refine) — step 0.01 over ``argmax ± 0.05`` on each axis (≤ 11 values
    per axis, clipped to the valid range) → at most 11⁴ = 14 641 evaluations,
    typically fewer after clipping.  Total ≲ 59 000 evaluations per dataset.

    Each evaluation is a vectorized forward pass (one numpy reduction per
    opportunity), so a 200×20 dataset fits in well under a second.

    Determinism
    -----------
    The grids are fixed and ties are broken deterministically (see
    :func:`_best_on_grid`): identical input always yields an identical fit.

    Returns
    -------
    BktParams
        The maximum-likelihood parameter bundle (validated by ``BktParams``).
    """
    obs = np.asarray(observations, dtype=bool)

    # ---- Stage 1: coarse grid (step 0.05) ----
    coarse_s = np.round(np.arange(_S_MIN, _S_MAX + 1e-9, 0.05), 6)  # 0.01 … 0.46
    coarse_g = np.round(np.arange(_G_MIN, _G_MAX + 1e-9, 0.05), 6)  # 0.01 … 0.46
    coarse_t = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 6)        # 0.00 … 1.00
    coarse_l0 = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 6)       # 0.00 … 1.00

    s0, g0, t0, l00 = _best_on_grid(obs, coarse_s, coarse_g, coarse_t, coarse_l0)

    # ---- Stage 2: refine ±0.05 at step 0.01 around the coarse argmax ----
    fine_s = _clip_axis(s0, _S_MIN, _S_MAX, 0.01)
    fine_g = _clip_axis(g0, _G_MIN, _G_MAX, 0.01)
    fine_t = _clip_axis(t0, 0.0, 1.0, 0.01)
    fine_l0 = _clip_axis(l00, 0.0, 1.0, 0.01)

    s1, g1, t1, l01 = _best_on_grid(obs, fine_s, fine_g, fine_t, fine_l0)

    return BktParams(s=s1, g=g1, t=t1, L0=l01)


# ---------------------------------------------------------------------------
# Posterior trajectory & calibration helper
# ---------------------------------------------------------------------------


def posterior_trajectory(observations: np.ndarray, params: BktParams) -> np.ndarray:
    """Pre-observation mastery probability P(L) for every opportunity.

    Returns, for each student and opportunity ``k``, the probability of mastery
    BEFORE observing response ``k`` — i.e. the quantity a tutor would act on at
    that moment.  Computed by repeated application of the frozen
    :func:`readcoach.bkt.bkt_update` with ``conf=1.0`` (hard labels).

    The first column is the prior ``L0`` for every student; column ``k`` for
    ``k ≥ 1`` is the posterior-after-transit from observation ``k-1`` (which is
    exactly the prior going into observation ``k``).

    Vectorized over students (the only loop is over opportunities).

    Returns
    -------
    ndarray of float, shape == ``observations.shape``.
    """
    obs = np.asarray(observations, dtype=bool)
    n_students, n_opps = obs.shape
    s, g, t = params.s, params.g, params.t

    traj = np.empty((n_students, n_opps), dtype=np.float64)
    p = np.full(n_students, params.L0, dtype=np.float64)

    for k in range(n_opps):
        traj[:, k] = p  # P(L) BEFORE observing response k.

        # Vectorized hard-label bkt_update (conf=1.0): evidence then transit.
        # Mirrors readcoach.bkt.bkt_update exactly.
        correct = obs[:, k]
        p_obs_L = np.where(correct, 1.0 - s, s)       # P(obs | knows)
        p_obs_nL = np.where(correct, g, 1.0 - g)      # P(obs | !knows)
        post = (p * p_obs_L) / (p * p_obs_L + (1.0 - p) * p_obs_nL)
        p = post + (1.0 - post) * t

    return traj


def predict_correct(p_L, s: float, g: float):
    """P(next response correct) given mastery probability ``p_L``.

    ``predict = p_L*(1-s) + (1-p_L)*g`` — the marginal over the (unknown) latent
    mastery state.  Works on scalars or numpy arrays.  Used for calibration
    (reliability / Brier): compare ``predict_correct`` against actual
    correctness.
    """
    return p_L * (1.0 - s) + (1.0 - p_L) * g


# Sanity check used by callers: bkt_update is the per-step engine the vectorized
# trajectory reproduces.  Imported here to keep the dependency explicit.
__all__ = [
    "simulate",
    "log_likelihood",
    "fit_grid",
    "posterior_trajectory",
    "predict_correct",
    "bkt_update",
]
