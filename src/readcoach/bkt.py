"""Soft-evidence Bayesian Knowledge Tracing (BKT) core — T3.1.

Implements a 2-state (mastered / not-mastered) BKT update extended with
*virtual evidence* (Pearl 1988) so that the confidence in an observation
label can be taken into account.  When ``conf=1.0`` the update reduces
exactly to textbook BKT (Corbett & Anderson 1994).  When ``conf=0.5`` the
label carries no information and the posterior is unchanged before the
transit step.

References
----------
* Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*.
  Morgan Kaufmann. — virtual evidence / likelihood weighting.
* Beck, J. E., & Sison, J. (2004-06). Project LISTEN / observation-
  confidence KT lineage. — applying soft labels to knowledge tracing.

Design notes
------------
Parameter constraints
~~~~~~~~~~~~~~~~~~~~~
* ``s`` (slip)  must be in [0, 0.5).  Values ≥ 0.5 are rejected because they
  imply P(correct | knows) ≤ P(correct | !knows), which flips the meaning of
  the observation and destroys BKT identifiability: you can no longer tell
  mastery from non-mastery from the data.
* ``g`` (guess) must be in [0, 0.5).  Same reason — if the guess rate equals
  or exceeds 0.5 the signal direction reverses; the skill is unidentifiable
  from observations.

Denominator safety
~~~~~~~~~~~~~~~~~~
With fully validated parameters (``s`` < 0.5, ``g`` < 0.5, ``conf`` ∈ [0,1],
``p_L`` ∈ [0,1]):

  For correct=True:
    like_L  = conf*(1-2s)+s ≥ s ≥ 0   (= 0 only if s=0 and conf=0)
    like_nL = conf*(2g-1)+(1-g): since (2g-1)<0 and (1-g)>0,
              requires conf=(1-g)/(1-2g) > 1 to be 0 — impossible.
  For correct=False:
    like_L  = (1-s)+conf*(2s-1): requires conf=(1-s)/(1-2s) > 1 — impossible.
    like_nL = conf*(1-2g)+g ≥ g ≥ 0   (= 0 only if g=0 and conf=0)

  Denominator = p_L*like_L + (1-p_L)*like_nL.  It is zero only if BOTH
  like_L=0 AND like_nL=0 simultaneously.  From above, for correct=True
  like_nL is never 0 with valid params, so the denominator is always > 0.
  For correct=False like_L is never 0, same conclusion.
  Therefore ZeroDivisionError is unreachable for any fully validated input.
  No silent clamping is applied; the formula is evaluated as-is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------------------
# Frozen parameters dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BktParams:
    """Immutable BKT parameter bundle.

    Attributes
    ----------
    s  : slip probability — P(incorrect response | student knows skill).
         Must be in [0, 0.5).  Values ≥ 0.5 are identifiability-breaking
         (see module docstring).
    g  : guess probability — P(correct response | student does NOT know skill).
         Must be in [0, 0.5).  Same constraint.
    t  : transit probability — P(learns skill on this opportunity | did not know).
         Must be in [0, 1].
    L0 : prior probability of mastery at the start of practice.
         Must be in [0, 1].
    """

    s: float
    g: float
    t: float
    L0: float

    def __post_init__(self) -> None:
        # Every param must be in [0, 1].
        for name, val in (("s", self.s), ("g", self.g), ("t", self.t), ("L0", self.L0)):
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"BktParams.{name}={val!r} is out of range [0, 1]."
                )
        # Reject degenerate / label-flipping regimes.
        if self.s >= 0.5:
            raise ValueError(
                f"BktParams.s={self.s!r} >= 0.5: identifiability broken — "
                "P(correct | knows) would be ≤ P(correct | !knows), making "
                "mastery indistinguishable from non-mastery."
            )
        if self.g >= 0.5:
            raise ValueError(
                f"BktParams.g={self.g!r} >= 0.5: identifiability broken — "
                "guess rate ≥ 0.5 flips the evidential direction of observations."
            )


# ---------------------------------------------------------------------------
# Core update (FROZEN — do not alter without updating the pre-registration)
# ---------------------------------------------------------------------------

def bkt_update(
    p_L: float,
    correct: bool,
    conf: float,
    s: float,
    g: float,
    t: float,
) -> float:
    """One-step soft-evidence BKT posterior update.

    Uses virtual evidence (Pearl 1988): ``conf`` is P(the observation label is
    correct).  ``conf=1.0`` recovers textbook BKT; ``conf=0.5`` is
    information-free (posterior = prior before transit).

    The update order is **evidence first, then transit**:
        post   = Bayes(p_L | observation, conf)
        result = post + (1 - post) * t

    Parameters
    ----------
    p_L     : Prior probability of mastery, in [0, 1].
    correct : Whether the observed response was correct.
    conf    : P(observation label is right), in [0, 1].
    s       : Slip probability.  Must satisfy 0 ≤ s < 0.5.
    g       : Guess probability.  Must satisfy 0 ≤ g < 0.5.
    t       : Transit probability, in [0, 1].

    Returns
    -------
    float
        Updated probability of mastery after evidence and transit.

    Raises
    ------
    ValueError
        If ``p_L`` or ``conf`` is outside [0, 1].
    """
    if not (0.0 <= p_L <= 1.0):
        raise ValueError(f"p_L={p_L!r} is out of range [0, 1].")
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"conf={conf!r} is out of range [0, 1].")

    # Textbook BKT observation likelihoods.
    p_obs_L  = (1 - s) if correct else s        # P(obs | knows)
    p_obs_nL = g        if correct else (1 - g)  # P(obs | !knows)

    # Soft-evidence (virtual) blending.
    # conf=1 → like = p_obs; conf=0.5 → like = 0.5 (information-free).
    like_L   = conf * p_obs_L  + (1 - conf) * (1 - p_obs_L)
    like_nL  = conf * p_obs_nL + (1 - conf) * (1 - p_obs_nL)

    # Bayesian posterior (denominator cannot be 0 for valid params; see module doc).
    post = p_L * like_L / (p_L * like_L + (1 - p_L) * like_nL)

    # Transit: P(learns | did not know) = t.
    return post + (1 - post) * t


# ---------------------------------------------------------------------------
# Convenience: trajectory over a sequence of observations
# ---------------------------------------------------------------------------

def update_sequence(
    p_L0: float,
    observations: Sequence[tuple[bool, float]],
    params: BktParams,
) -> list[float]:
    """Return the full posterior trajectory for a list of observations.

    Parameters
    ----------
    p_L0         : Initial mastery probability (prior).
    observations : Sequence of ``(correct, conf)`` tuples.
    params       : :class:`BktParams` bundle.

    Returns
    -------
    list[float]
        List of length ``len(observations) + 1``.  The first element is
        ``p_L0``; element ``i+1`` is the posterior after observation ``i``.
    """
    trajectory: list[float] = [p_L0]
    p = p_L0
    for correct, conf in observations:
        p = bkt_update(p, correct, conf, params.s, params.g, params.t)
        trajectory.append(p)
    return trajectory
