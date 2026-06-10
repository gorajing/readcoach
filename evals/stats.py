"""Judge-validation statistics — T5.2.

Pure statistical functions for computing agreement metrics between the LLM
judge and human labelers.  No I/O; numpy is the only non-stdlib dependency.

## Binary-only scope

All functions operate on BINARY (bool) passing labels — judge.passing vs
human.passing.  Score-level agreement (1–5 Likert scale) would require
weighted Cohen's kappa; that is a stretch goal NOT ticketed for T5.2.
The cohens_kappa function accepts any comparable labels but is validated
and documented only for the binary case.

## Degenerate-case policy

- Empty inputs or length mismatches → ValueError (loud).
- tpr_tnr: if the human labels contain NO positives, TPR is undefined and
  returned as None (not 0 — that would be a false claim).  Same for TNR
  when human labels contain NO negatives.  The caller must handle None.
  NEVER silently returns 0.0 for an undefined rate.
- cohens_kappa: when all raters agree on a single class, P_e = 1.0 and
  the denominator (1 - P_e) = 0.  kappa is mathematically undefined in
  this case.  We raise ValueError rather than returning 1.0 (flattering
  and wrong) or 0.0 (opposite direction).
- bootstrap_ci: if stat_fn returns None on a resample (e.g. because the
  resample is degenerate), that resample is REDRAWN up to _MAX_REDRAW
  attempts before raising ValueError.  Silent dropping would bias the CI
  by excluding the tails most likely to produce degenerate samples.

## Cohen's kappa reference

Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement
for categorical data. *Biometrics*, 33(1), 159–174.

Banding (ibid):
  κ < 0.00 — poor (worse than chance)
  0.00–0.20 — slight
  0.21–0.40 — fair
  0.41–0.60 — moderate   ← KAPPA_FLOOR = 0.4 (gate threshold)
  0.61–0.80 — substantial
  0.81–1.00 — almost perfect

KAPPA_FLOOR = 0.4 is the minimum acceptable point estimate for a dimension
to be gate-eligible.  A dimension below floor is reported as untrusted and
excluded from gating — a finding, not a failure.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import numpy as np

# Landis & Koch moderate-agreement boundary (documented above).
KAPPA_FLOOR: float = 0.4

# Maximum redraws when stat_fn returns None on a degenerate resample.
_MAX_REDRAW: int = 50


# ---------------------------------------------------------------------------
# tpr_tnr
# ---------------------------------------------------------------------------


def tpr_tnr(
    judge: list[bool],
    human: list[bool],
) -> tuple[float | None, float | None]:
    """True-positive rate and true-negative rate (human is the truth).

    Parameters
    ----------
    judge:
        LLM judge's binary passing labels.
    human:
        Human labeler's binary passing labels (ground truth).

    Returns
    -------
    (tpr, tnr)
        tpr = TP / (TP + FN).  None when human has NO positive examples
              (the denominator is zero; returning 0.0 would be a false claim).
        tnr = TN / (TN + FP).  None when human has NO negative examples
              (same reasoning).

    Raises
    ------
    ValueError
        If inputs are empty or have different lengths.
    """
    if len(judge) == 0 or len(human) == 0:
        raise ValueError("tpr_tnr: inputs must not be empty")
    if len(judge) != len(human):
        raise ValueError(
            f"tpr_tnr: length mismatch — judge has {len(judge)} items, "
            f"human has {len(human)} items"
        )

    tp = tn = fp = fn = 0
    for j, h in zip(judge, human):
        if h and j:
            tp += 1
        elif h and not j:
            fn += 1
        elif not h and j:
            fp += 1
        else:
            tn += 1

    # TPR: if no positives in human, denominator = 0 → None.
    tpr: float | None
    if tp + fn == 0:
        tpr = None
    else:
        tpr = tp / (tp + fn)

    # TNR: if no negatives in human, denominator = 0 → None.
    tnr: float | None
    if tn + fp == 0:
        tnr = None
    else:
        tnr = tn / (tn + fp)

    return tpr, tnr


# ---------------------------------------------------------------------------
# cohens_kappa
# ---------------------------------------------------------------------------


def cohens_kappa(a: list[Any], b: list[Any]) -> float:
    """Cohen's kappa for inter-rater agreement.

    Standard formula:
        P_o = observed proportion of agreement
        P_e = expected proportion of agreement under independence
        kappa = (P_o - P_e) / (1 - P_e)

    Hand-computed test vectors (see tests/test_stats.py for full arithmetic):

      Vector 1 — partial agreement:
        a = [T,T,T,F,F,F,F,F,F,F]  (3T,7F)
        b = [T,T,F,T,F,F,F,F,F,F]  (3T,7F)
        Confusion: TP=2, FP=1, FN=1, TN=6
        P_o = (2+6)/10 = 0.8
        P_e = (3/10)*(3/10) + (7/10)*(7/10) = 0.09 + 0.49 = 0.58
        kappa = (0.8-0.58)/(1-0.58) = 0.22/0.42 ≈ 0.5238

      Vector 2 — perfect agreement:
        a = b = [T,T,T,F,F,F]
        P_o = 1.0; P_e = (3/6)^2 + (3/6)^2 = 0.5
        kappa = (1.0-0.5)/(1-0.5) = 1.0

      Degenerate — all same class:
        a = b = [T,T,T,T]
        P_e = (4/4)^2 + (0/4)^2 = 1.0
        denominator = 1 - 1.0 = 0 → kappa undefined → raise ValueError

    NOTE: weighted kappa for ordinal (1–5) scores is NOT implemented here;
    this function is validated for binary labels only.  Score-level agreement
    is a stretch goal not ticketed for T5.2.

    Parameters
    ----------
    a, b:
        Lists of comparable labels (typically bool, but any hashable type works).

    Returns
    -------
    float
        Cohen's kappa in [-1, 1].

    Raises
    ------
    ValueError
        If inputs are empty, have different lengths, or if kappa is undefined
        (P_e == 1.0, i.e. both raters chose the same single class unanimously).
    """
    if len(a) == 0 or len(b) == 0:
        raise ValueError("cohens_kappa: inputs must not be empty")
    if len(a) != len(b):
        raise ValueError(
            f"cohens_kappa: length mismatch — a has {len(a)} items, "
            f"b has {len(b)} items"
        )

    n = len(a)

    # Marginal counts.
    a_counts: Counter = Counter(a)
    b_counts: Counter = Counter(b)

    # Observed agreement.
    agree = sum(1 for ai, bi in zip(a, b) if ai == bi)
    p_o = agree / n

    # Expected agreement under independence.
    all_labels = set(a_counts) | set(b_counts)
    p_e = sum((a_counts[lbl] / n) * (b_counts[lbl] / n) for lbl in all_labels)

    denom = 1.0 - p_e
    if abs(denom) < 1e-12:
        raise ValueError(
            "cohens_kappa: kappa is undefined (degenerate case) — "
            "denominator (1 - P_e) is zero, which happens when both raters "
            "chose the same single class unanimously.  "
            "Returning 1.0 (flattering) or 0.0 (opposite) would both be wrong."
        )

    return float((p_o - p_e) / denom)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def bootstrap_ci(
    stat_fn: Callable[..., float | None],
    *paired_arrays: list[Any],
    n_boot: int = 2000,
    seed: int,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Seeded bootstrap confidence interval using the percentile method.

    Paired resampling: items are resampled by index, preserving alignment
    across all input arrays (judge labels paired with human labels, etc.).

    Parameters
    ----------
    stat_fn:
        Callable that accepts *paired_arrays (unzipped by column) and returns
        a float.  May return None if the resample is degenerate.
    *paired_arrays:
        One or more equal-length lists.  All are resampled identically.
    n_boot:
        Number of bootstrap resamples.  Default 2000.
    seed:
        RNG seed for reproducibility.  Required — no default (caller must
        be explicit to prevent accidental non-determinism).
    ci:
        Confidence level in (0, 1).  Default 0.95 (95% CI).

    Returns
    -------
    (lo, hi)
        Percentile-method CI bounds.

    Raises
    ------
    ValueError
        - If any input array is empty.
        - If arrays have different lengths.
        - If stat_fn returns None on more than _MAX_REDRAW consecutive
          redraws for a single bootstrap replicate.  Silent dropping would
          bias the CI by under-representing tail behavior.

    Notes
    -----
    The percentile method is used (not BCa) because it is unbiased for
    symmetric distributions and simple to audit.  For the n=60 human-label
    dataset, BCa would add complexity without meaningful gain.
    """
    if not paired_arrays:
        raise ValueError("bootstrap_ci: at least one array is required")

    # Validate lengths.
    lengths = [len(arr) for arr in paired_arrays]
    if any(ln == 0 for ln in lengths):
        raise ValueError("bootstrap_ci: input arrays must not be empty")
    if len(set(lengths)) > 1:
        raise ValueError(
            f"bootstrap_ci: length mismatch among arrays — lengths: {lengths}"
        )

    n = lengths[0]
    rng = np.random.default_rng(seed)
    boot_stats: list[float] = []

    for _ in range(n_boot):
        # Paired resample with replacement.
        # On degenerate resamples (stat_fn returns None OR raises ValueError),
        # redraw up to cap.  Silent dropping would bias the CI by excluding
        # the tail resamples most likely to produce degenerate samples.
        stat_val: float | None = None
        for redraw in range(_MAX_REDRAW + 1):
            idx = rng.integers(0, n, size=n)
            resampled = [[arr[i] for i in idx] for arr in paired_arrays]
            try:
                stat_val = stat_fn(*resampled)
            except (ValueError, ZeroDivisionError):
                # Degenerate resample — treat as None and redraw.
                stat_val = None
            if stat_val is not None:
                break
        else:
            raise ValueError(
                f"bootstrap_ci: stat_fn returned None (or raised) on every one "
                f"of {_MAX_REDRAW} consecutive redraws.  This suggests the "
                f"statistic is always undefined for this dataset.  "
                f"Silent dropping would bias the CI — raising instead."
            )

        boot_stats.append(float(stat_val))  # type: ignore[arg-type]

    alpha = 1.0 - ci
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lo, hi
