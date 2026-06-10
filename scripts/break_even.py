"""Break-even experiment — where ASR confidence must enter ANY mastery update (T3.3).

This is the project's headline soft-evidence artifact: it adjudicates
pre-registered Prediction 4 by sweeping detector reliability and comparing a
naive (hard-binary) BKT update against a confidence-weighted (virtual-evidence)
update, on the *same* observation streams.

Lineage
-------
Confidence-weighted knowledge tracing in a reading tutor is the Project LISTEN
line of work: Beck, J. E., & Sison, J. (2004-06) used speech-recognizer
confidence as soft evidence in a knowledge-tracing student model.  The
virtual-evidence machinery is Pearl (1988).  This experiment asks the same
question with the project's frozen ``readcoach.bkt.bkt_update``: once the
detector is noisy, does discounting each observation by the channel's known
reliability beat treating every label as certain?

------------------------------------------------------------------------------
DESIGN (all controller decisions documented here, in the script that runs them)
------------------------------------------------------------------------------

1. THE NOISE CHANNEL.
   Per opportunity the student's TRUE correctness ``c ∈ {0,1}`` comes from the
   BKT generative simulation (``simulate``).  The detector reports an observed
   label ``o`` with
        P(o = c) = a        (channel accuracy)
        P(o ≠ c) = 1 - a     (a symmetric bit-flip).
   ``a`` is the PRIMARY x-axis, swept over a clean, assumption-light grid
        a ∈ {0.55, 0.60, …, 0.95, 0.99}.
   We deliberately present the sweep in channel-accuracy space rather than in
   miscue-detection-F1 space, because for an imbalanced miscue-detection task
   the relationship F1 ↔ per-word label accuracy is only loose.  Instead we
   ANCHOR reality as a secondary annotation: each measured bias setting
   (none / prompt / strong) is converted to an *effective per-word channel
   accuracy* ``a_eff`` and marked on the curve.

   a_eff formula (closed form, ``effective_accuracy`` below).  Pool over the
   benchmark's real miscue density ``d`` = (gold word-level miscue events) /
   (total target words), computed from data/benchmark/gold.jsonl:

        a_eff = (1 - d) · (1 - fp_rate)  +  d · pooled_recall

   where
     * ``fp_rate`` = fp_per_100_correct_words / 100   — the probability a TRULY
       CORRECT word is flagged (false alarm), i.e. P(label wrong | word correct);
       so P(label right | word correct) = 1 - fp_rate.
     * ``pooled_recall`` = ΣTP / (ΣTP + ΣFN) over the word-level miscue classes
       {substitution, omission, insertion} — the probability a TRULY MISCUED
       word is flagged, i.e. P(label right | word miscued).
   self_correction and hesitation are excluded from the recall pool: they are
   not clean "this single word was misread" word-substitution events (they are
   span/disfluency phenomena), and folding them in would muddy a per-word flip
   model.  This choice is documented and the excluded classes are reported in
   the JSON for transparency.

   NOTE ON DENSITY.  The benchmark injects ONE class per clip, so its per-item
   miscue rate (~1.46 events / ~43.4 words ≈ 0.0335) is sparse by construction.
   A real running read is denser; with a sparse channel the correct-word term
   dominates, which (honestly) pushes every a_eff anchor high (~0.92-0.97).  We
   report the density we actually measured and let the anchors fall where they
   fall — that itself is a finding (the measured operating points sit ABOVE the
   break-even band, in the converged regime).

2. TWO UPDATERS ON IDENTICAL STREAMS.
   For a given (regime, a, seed) we simulate (true latent, true correctness),
   pass true correctness through the channel ONCE to get the observed labels
   ``o``, then score that SAME ``o`` array twice:
     * naive : ``bkt_update(conf=1.0)``  — textbook BKT, treats o as certain.
     * soft  : ``bkt_update(conf=a)``    — well-calibrated virtual evidence; the
               tutor knows the channel quality and discounts by it.
   Because both updaters are pure functions of the one shared ``o`` array, the
   comparison is exactly paired: no RNG divergence between arms.  (Guarded by
   ``test_both_updaters_see_identical_observations``.)

3. METRICS (per a; averaged over n_students × n_opps × 3 regimes).
   Regimes: three of the four T3.2 regimes — ``easy_skill``, ``hard_skill``,
   ``high_guess`` — chosen to span low/high guess and slow/fast transit (the
   axes that most stress a noisy channel); ``low_noise`` is omitted as the
   easiest case where any method works.  Same regime defs as T3.2.

   a) mastery RMSE vs the true latent 0/1 state (as T3.2), pooled over
      student × opportunity × regime, per updater.
   b) time-to-mastery-detection error: first k where P(L) ≥ 0.95 MINUS first k
      where the latent first becomes mastered.  Reported as mean SIGNED error
      (positive = detected late, negative = early) and mean |error|, over
      students who BOTH truly master AND are detected.  Students who never
      master, or are never detected within the horizon, are excluded from the
      latency means and counted separately; the never-detected RATE is
      reported, not hidden.
   c) item-selection regret (proxy, ~10 lines — see ``select_item`` /
      ``run_regret``): a 5-skill toy with a linear prerequisite chain.  The
      policy selects argmax mastery-GAP (1 - P(L)) among skills whose
      prerequisite has P(L) ≥ 0.8 (skill 0 always available); the oracle runs
      the SAME selector on TRUE latent (gap = 1 - latent).  Each selects 25
      items; "true mastery achieved" = the true latent state of the selected
      skill (the oracle only ever practises genuinely-unmastered skills, so it
      bounds the achievable).  Regret = oracle_total − policy_total, averaged
      over students.  Same selector logic drives naive and soft posteriors.

4. BREAK-EVEN EXTRACTION.
   Break-even = the smallest ``a`` at which |RMSE_soft − RMSE_naive| < 0.005
   (a convergence threshold: below 0.005 RMSE the two methods are practically
   indistinguishable on this metric).  Below the break-even we report the soft
   advantage Δ = RMSE_naive − RMSE_soft at each grid point with a seeded paired
   bootstrap 95% CI (500 resamples over students, same resample index applied to
   both arms so the pairing is preserved).

5. OUTPUTS (both committed).
   evals/results/break_even.json — full grid (RMSE both arms + Δ + paired CI,
     detection error, regret), a_eff anchors, break-even point, and metadata.
     Volatile fields (runtime, git commit) live in a ``volatile`` sub-object,
     documented as non-deterministic so the rest of the file is reproducible.
   evals/results/break_even.png — 3 panels (RMSE vs a with soft−naive Δ band;
     detection-error vs a; regret vs a), captioned with the Beck & Sison
     lineage line, n, and seed.

6. PREDICTION 4 VERDICT goes in docs/results_vs_predictions.md (this script
   only prints what's needed; predictions.md is never touched).

7-8. TDD in tests/test_break_even.py; run with
     ``uv run python scripts/break_even.py --seed 2026`` (< ~3 min target).

Usage
-----
  uv run python scripts/break_even.py [--seed 2026] [--n-students 500]
                                      [--n-opportunities 25] [--n-boot 500]
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; before pyplot import.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Make ``readcoach`` importable when run as a plain script (no editable install).
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from readcoach.bkt import BktParams  # noqa: E402
from readcoach.bkt_fit import simulate  # noqa: E402

_GOLD_JSONL = _PROJECT_ROOT / "data" / "benchmark" / "gold.jsonl"
_MISCUE_JSON = _PROJECT_ROOT / "evals" / "results" / "miscue-v0.json"
_OUT_JSON = _PROJECT_ROOT / "evals" / "results" / "break_even.json"
_OUT_PNG = _PROJECT_ROOT / "evals" / "results" / "break_even.png"

# Channel-accuracy grid — the primary, assumption-light x-axis.
_A_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]

# Three of the four T3.2 regimes (low_noise omitted — easiest, least stressing).
_REGIMES: dict[str, BktParams] = {
    "easy_skill": BktParams(s=0.08, g=0.15, t=0.15, L0=0.4),
    "hard_skill": BktParams(s=0.15, g=0.25, t=0.05, L0=0.15),
    "high_guess": BktParams(s=0.10, g=0.35, t=0.10, L0=0.25),
}

# Word-level miscue classes pooled for a_eff recall (see module docstring).
_RECALL_CLASSES = ("substitution", "omission", "insertion")
_EXCLUDED_CLASSES = ("self_correction", "hesitation")

_DETECT_THRESHOLD = 0.95   # P(L) ≥ this counts as "detected mastered".
_PREREQ_THRESHOLD = 0.80   # prerequisite P(L) ≥ this unlocks the next skill.
_CONVERGE_EPS = 0.005      # |RMSE_soft - RMSE_naive| < this ⇒ converged.
_N_SKILLS = 5              # prerequisite-chain length for the regret proxy.


# ---------------------------------------------------------------------------
# Channel & conf-parameterized updater
# ---------------------------------------------------------------------------


def apply_channel(true_correct: np.ndarray, a: float, rng: np.random.Generator) -> np.ndarray:
    """Symmetric bit-flip channel: observe the true label w.p. ``a``, else flip.

    Parameters
    ----------
    true_correct : bool ndarray of TRUE correctness from the generative sim.
    a            : channel accuracy P(observed == true), in [0, 1].
    rng          : seeded generator (the only entropy source).

    Returns
    -------
    bool ndarray, same shape as ``true_correct``: the OBSERVED labels.
    """
    keep = rng.random(true_correct.shape) < a
    return np.where(keep, true_correct, ~true_correct)


def posterior_trajectory_conf(
    observations: np.ndarray, params: BktParams, conf: float
) -> np.ndarray:
    """Pre-observation P(L) per opportunity under a FIXED confidence ``conf``.

    Vectorized mirror of the frozen ``readcoach.bkt.bkt_update`` (evidence then
    transit) with the virtual-evidence blend applied at constant ``conf`` for
    every observation.  ``conf=1.0`` reproduces the hard-label trajectory; the
    soft arm passes ``conf=a`` (the known channel accuracy).

    Returns float ndarray of shape == ``observations.shape``; column k is P(L)
    BEFORE observing response k (the quantity a tutor acts on).
    """
    obs = np.asarray(observations, dtype=bool)
    n_students, n_opps = obs.shape
    s, g, t = params.s, params.g, params.t

    traj = np.empty((n_students, n_opps), dtype=np.float64)
    p = np.full(n_students, params.L0, dtype=np.float64)

    for k in range(n_opps):
        traj[:, k] = p
        correct = obs[:, k]
        # Textbook emission likelihoods, then virtual-evidence blend by conf.
        p_obs_L = np.where(correct, 1.0 - s, s)        # P(obs | knows)
        p_obs_nL = np.where(correct, g, 1.0 - g)       # P(obs | !knows)
        like_L = conf * p_obs_L + (1.0 - conf) * (1.0 - p_obs_L)
        like_nL = conf * p_obs_nL + (1.0 - conf) * (1.0 - p_obs_nL)
        post = (p * like_L) / (p * like_L + (1.0 - p) * like_nL)
        p = post + (1.0 - post) * t

    return traj


# ---------------------------------------------------------------------------
# a_eff (measured operating-point anchor)
# ---------------------------------------------------------------------------


def effective_accuracy(miscue_density: float, fp_per_100: float, pooled_recall: float) -> float:
    """Effective per-word channel accuracy for a measured detector setting.

        a_eff = (1 - d)·(1 - fp_rate) + d·pooled_recall

    where d = miscue_density, fp_rate = fp_per_100 / 100 (false-alarm rate on
    correct words), pooled_recall = recall on truly-miscued words.  See the
    module docstring for the derivation.
    """
    fp_rate = fp_per_100 / 100.0
    return (1.0 - miscue_density) * (1.0 - fp_rate) + miscue_density * pooled_recall


def _compute_miscue_density() -> float:
    """(gold word-level miscue events) / (total target words) from gold.jsonl.

    Counts EVERY gold event (all classes) as a miscued word position and divides
    by total target-passage words.  This is the fraction of word slots that are
    miscued in the benchmark — the density at which the channel operates.
    """
    total_events = 0
    total_words = 0
    with _GOLD_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            total_events += len(d["gold"])
            total_words += len(d["target_text"].split())
    return total_events / total_words


def _a_eff_anchors() -> list[dict]:
    """Convert each measured bias setting to an a_eff anchor."""
    density = _compute_miscue_density()
    miscue = json.loads(_MISCUE_JSON.read_text(encoding="utf-8"))["results"]
    anchors: list[dict] = []
    for bias in ("none", "prompt", "strong"):
        r = miscue[bias]
        tp = sum(r[c]["tp"] for c in _RECALL_CLASSES)
        fn = sum(r[c]["fn"] for c in _RECALL_CLASSES)
        pooled_recall = tp / (tp + fn)
        fp_per_100 = r["fp_per_100_correct_words"]
        a_eff = effective_accuracy(density, fp_per_100, pooled_recall)
        anchors.append({
            "bias": bias,
            "a_eff": a_eff,
            "fp_per_100_correct_words": fp_per_100,
            "pooled_recall": pooled_recall,
            "pooled_recall_classes": list(_RECALL_CLASSES),
        })
    return anchors, density


# ---------------------------------------------------------------------------
# Metric 3a — paired RMSE
# ---------------------------------------------------------------------------


def run_paired(
    params: BktParams, a: float, n_students: int, n_opps: int, rng: np.random.Generator
) -> dict:
    """Simulate one regime, pass true correctness through the channel ONCE, and
    score the resulting observed labels with BOTH updaters on the shared array.

    Returns a dict with the shared ``obs``, the ``latent`` truth, and the two
    posterior trajectories ``post_naive`` (conf=1) and ``post_soft`` (conf=a).
    """
    true_obs, latent = simulate(params, n_students, n_opps, rng)
    obs = apply_channel(true_obs, a, rng)  # ONE channel pass → shared stream.
    post_naive = posterior_trajectory_conf(obs, params, conf=1.0)
    post_soft = posterior_trajectory_conf(obs, params, conf=a)
    return {"obs": obs, "latent": latent, "post_naive": post_naive, "post_soft": post_soft}


# ---------------------------------------------------------------------------
# Metric 3b — time-to-mastery-detection error
# ---------------------------------------------------------------------------


def detection_latency(
    post: np.ndarray, latent: np.ndarray, threshold: float, horizon: int
) -> dict:
    """First-crossing latency of P(L)≥threshold vs first true-mastery opportunity.

    For each student: detect_k = first column where post ≥ threshold (or None);
    master_k = first column where latent is True (or None).  Signed latency =
    detect_k − master_k (positive = detected late).  Students with either index
    None are excluded from the latency means and counted in the never-* tallies.
    """
    n_students = post.shape[0]
    detect_idx = _first_true_index(post >= threshold, horizon)
    master_idx = _first_true_index(latent, horizon)

    detected = detect_idx >= 0
    mastered = master_idx >= 0
    both = detected & mastered

    signed = (detect_idx - master_idx).astype(np.float64)
    if both.any():
        mean_signed = float(np.mean(signed[both]))
        mean_abs = float(np.mean(np.abs(signed[both])))
    else:
        mean_signed = float("nan")
        mean_abs = float("nan")

    return {
        "mean_signed_error": mean_signed,
        "mean_abs_error": mean_abs,
        "n_never_detected": int(np.sum(~detected)),
        "n_never_mastered": int(np.sum(~mastered)),
        "n_students": int(n_students),
    }


def _first_true_index(mask: np.ndarray, horizon: int) -> np.ndarray:
    """Per-row index of the first True, or -1 if a row is all-False."""
    any_true = mask.any(axis=1)
    first = np.argmax(mask, axis=1)  # 0 for all-False rows; guarded by any_true.
    return np.where(any_true, first, -1).astype(np.int64)


# ---------------------------------------------------------------------------
# Metric 3c — item-selection regret (PROXY selector, ~10 lines)
# ---------------------------------------------------------------------------


def select_item(beliefs: np.ndarray) -> np.ndarray:
    """PROXY selector: argmax mastery-gap among prerequisite-satisfied skills.

    ``beliefs`` is (n_students, n_skills) of P(L) (or true latent for the
    oracle).  A skill is AVAILABLE if it is skill 0 or its predecessor's belief
    ≥ _PREREQ_THRESHOLD.  Among available skills pick argmax gap = 1 - belief.
    Returns the chosen skill index per student.  (This is the ~10-line inline
    proxy; "re-run with the real planner" is a later stretch.)
    """
    n_students, n_skills = beliefs.shape
    unlocked = np.zeros_like(beliefs, dtype=bool)
    unlocked[:, 0] = True                                   # skill 0 always open
    unlocked[:, 1:] = beliefs[:, :-1] >= _PREREQ_THRESHOLD  # prereq satisfied
    gap = np.where(unlocked, 1.0 - beliefs, -np.inf)        # locked skills excluded
    return np.argmax(gap, axis=1)


def _practice_round(
    state: dict, conf: float | None, params: BktParams, a: float, rng: np.random.Generator
) -> None:
    """One practice round of the adaptive-tutoring loop, in place on ``state``.

    The selector (``select_item``) picks one skill per student from the
    beliefs this arm acts on (true latent for the oracle, posterior otherwise).
    Practising a skill ADVANCES learning: the chosen skill gets one BKT transit
    opportunity (unmastered → mastered w.p. t), and the arm observes a channeled
    response from the PRE-transit state, which updates that skill's belief via
    the frozen virtual-evidence update.  Letting practice drive mastery is what
    makes a better-targeted selector accumulate more true mastery — and makes
    regret a meaningful quantity.
    """
    n_students = state["latent"].shape[0]
    s, g, t = params.s, params.g, params.t
    rows = np.arange(n_students)

    beliefs = state["latent"].astype(np.float64) if conf is None else state["belief"]
    chosen = select_item(beliefs)                       # PROXY selector (shared)
    mastered = state["latent"][rows, chosen]

    # Emit a channeled observation from the PRE-transit state of the chosen skill.
    p_correct = np.where(mastered, 1.0 - s, g)
    true_correct = rng.random(n_students) < p_correct
    obs = np.where(rng.random(n_students) < a, true_correct, ~true_correct)

    # Transit the chosen skill (mastery absorbing); learning happens on practice.
    newly = (~mastered) & (rng.random(n_students) < t)
    state["latent"][rows, chosen] = mastered | newly

    if conf is not None:  # update the posterior for the practised skill only.
        prior = state["belief"][rows, chosen]
        p_obs_L = np.where(obs, 1.0 - s, s)
        p_obs_nL = np.where(obs, g, 1.0 - g)
        like_L = conf * p_obs_L + (1.0 - conf) * (1.0 - p_obs_L)
        like_nL = conf * p_obs_nL + (1.0 - conf) * (1.0 - p_obs_nL)
        post = (prior * like_L) / (prior * like_L + (1.0 - prior) * like_nL)
        state["belief"][rows, chosen] = post + (1.0 - post) * t


def run_regret(
    params: BktParams, a: float, n_students: int, n_opps: int, rng: np.random.Generator
) -> dict:
    """5-skill linear-prerequisite-chain regret for naive vs soft posteriors.

    Each skill shares the (one-regime) BKT params and the channel ``a``.  Each
    arm runs an adaptive loop of ``n_opps`` rounds: pick a skill (``select_item``
    on this arm's beliefs), practise it (which advances true mastery and yields a
    channeled belief update — see ``_practice_round``).  "True mastery achieved"
    at the horizon = mean over students of the count of TRULY-mastered skills.
    The oracle acts on the true latent, so it never wastes a round on an
    already-mastered skill and bounds the achievable mastery; regret =
    oracle_total − arm_total ≥ 0 in expectation.  All three arms start from the
    SAME initial latent draw (paired), so the difference isolates selection
    quality, not initial luck.
    """
    # Shared initial latent state ~ Bernoulli(L0) for all three arms (paired).
    init_latent = rng.random((n_students, _N_SKILLS)) < params.L0

    def _achieved(conf: float | None, arm_rng: np.random.Generator) -> float:
        state = {
            "latent": init_latent.copy(),
            "belief": np.full((n_students, _N_SKILLS), params.L0, dtype=np.float64),
        }
        for _ in range(n_opps):
            _practice_round(state, conf, params, a, arm_rng)
        return float(np.mean(state["latent"].sum(axis=1)))

    # Independent practice RNG per arm so transit/channel noise doesn't couple
    # arms; pairing is via the shared init_latent above.
    arm_seeds = rng.spawn(3)
    oracle = _achieved(None, arm_seeds[0])
    naive = _achieved(1.0, arm_seeds[1])
    soft = _achieved(a, arm_seeds[2])
    return {
        "oracle_achieved": oracle,
        "naive_achieved": naive,
        "soft_achieved": soft,
        "regret_naive": oracle - naive,
        "regret_soft": oracle - soft,
    }


# ---------------------------------------------------------------------------
# Paired bootstrap CI for the RMSE advantage
# ---------------------------------------------------------------------------


def _paired_bootstrap_ci(
    sq_naive: np.ndarray, sq_soft: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """95% CI on Δ = RMSE_naive − RMSE_soft via a PAIRED student bootstrap.

    ``sq_naive`` / ``sq_soft`` are per-student mean squared errors (length
    n_students), aligned by student.  Each resample draws the SAME student
    indices for both arms (preserving pairing), recomputes pooled RMSE for each
    arm, and takes the difference.  Returns (lo, hi) at the 2.5/97.5 pctiles.
    """
    n = sq_naive.shape[0]
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rmse_n = np.sqrt(np.mean(sq_naive[idx]))
        rmse_s = np.sqrt(np.mean(sq_soft[idx]))
        deltas[b] = rmse_n - rmse_s
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Per-a aggregation over the three regimes
# ---------------------------------------------------------------------------


def _run_grid_point(
    a: float, n_students: int, n_opps: int, n_boot: int, rng: np.random.Generator
) -> dict:
    """All metrics at one channel-accuracy ``a``, pooled over the 3 regimes."""
    # Per-student mean squared error, concatenated across regimes (for pooled
    # RMSE and a student-level paired bootstrap).
    sq_naive_all: list[np.ndarray] = []
    sq_soft_all: list[np.ndarray] = []
    # Detection bookkeeping accumulated across regimes.
    signed_n, signed_s, abs_n, abs_s = [], [], [], []
    never_det_n = never_det_s = never_master = det_total = 0
    # Regret accumulated across regimes (then averaged).
    regret_naive_acc, regret_soft_acc = [], []

    for params in _REGIMES.values():
        paired = run_paired(params, a, n_students, n_opps, rng)
        latent_f = paired["latent"].astype(np.float64)

        sq_n = np.mean((paired["post_naive"] - latent_f) ** 2, axis=1)  # per student
        sq_s = np.mean((paired["post_soft"] - latent_f) ** 2, axis=1)
        sq_naive_all.append(sq_n)
        sq_soft_all.append(sq_s)

        det_n = detection_latency(paired["post_naive"], paired["latent"], _DETECT_THRESHOLD, n_opps)
        det_s = detection_latency(paired["post_soft"], paired["latent"], _DETECT_THRESHOLD, n_opps)
        if not np.isnan(det_n["mean_signed_error"]):
            signed_n.append(det_n["mean_signed_error"])
            abs_n.append(det_n["mean_abs_error"])
        if not np.isnan(det_s["mean_signed_error"]):
            signed_s.append(det_s["mean_signed_error"])
            abs_s.append(det_s["mean_abs_error"])
        never_det_n += det_n["n_never_detected"]
        never_det_s += det_s["n_never_detected"]
        never_master += det_n["n_never_mastered"]
        det_total += det_n["n_students"]

        reg = run_regret(params, a, n_students, n_opps, rng)
        regret_naive_acc.append(reg["regret_naive"])
        regret_soft_acc.append(reg["regret_soft"])

    sq_naive = np.concatenate(sq_naive_all)
    sq_soft = np.concatenate(sq_soft_all)
    rmse_naive = float(np.sqrt(np.mean(sq_naive)))
    rmse_soft = float(np.sqrt(np.mean(sq_soft)))
    delta = rmse_naive - rmse_soft
    ci_lo, ci_hi = _paired_bootstrap_ci(sq_naive, sq_soft, n_boot, rng)

    return {
        "a": a,
        "rmse_naive": rmse_naive,
        "rmse_soft": rmse_soft,
        "delta_rmse": delta,            # naive − soft; >0 ⇒ soft wins.
        "delta_ci95": [ci_lo, ci_hi],
        "converged": abs(delta) < _CONVERGE_EPS,
        "detection": {
            "mean_signed_error_naive": float(np.mean(signed_n)) if signed_n else float("nan"),
            "mean_signed_error_soft": float(np.mean(signed_s)) if signed_s else float("nan"),
            "mean_abs_error_naive": float(np.mean(abs_n)) if abs_n else float("nan"),
            "mean_abs_error_soft": float(np.mean(abs_s)) if abs_s else float("nan"),
            "never_detected_rate_naive": never_det_n / det_total,
            "never_detected_rate_soft": never_det_s / det_total,
            "never_mastered_rate": never_master / det_total,
        },
        "regret": {
            "regret_naive": float(np.mean(regret_naive_acc)),
            "regret_soft": float(np.mean(regret_soft_acc)),
        },
    }


def _break_even_a(grid: list[dict]) -> float | None:
    """Smallest ``a`` at which the two methods have converged (|Δ| < eps)."""
    for point in grid:  # grid is ascending in a.
        if point["converged"]:
            return point["a"]
    return None


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _make_figure(
    grid: list[dict],
    anchors: list[dict],
    break_even: float | None,
    n_students: int,
    n_opps: int,
    seed: int,
    out_path: Path,
) -> None:
    a_vals = [g["a"] for g in grid]
    rmse_naive = [g["rmse_naive"] for g in grid]
    rmse_soft = [g["rmse_soft"] for g in grid]
    ci_lo = [g["delta_ci95"][0] for g in grid]
    ci_hi = [g["delta_ci95"][1] for g in grid]
    delta = [g["delta_rmse"] for g in grid]

    fig, (ax_rmse, ax_det, ax_reg) = plt.subplots(1, 3, figsize=(16, 5.2))

    # --- Panel A: RMSE vs a, both arms, with soft−naive Δ band on a twin axis.
    ax_rmse.plot(a_vals, rmse_naive, "o-", color="#d62728", label="naive (conf=1)")
    ax_rmse.plot(a_vals, rmse_soft, "s-", color="#1f77b4", label="soft (conf=a)")
    ax_rmse.set_xlabel("channel accuracy a")
    ax_rmse.set_ylabel("mastery RMSE vs true latent")
    ax_rmse.set_title("Mastery RMSE: naive vs soft-evidence")
    ax_rmse.grid(alpha=0.3)
    if break_even is not None:
        ax_rmse.axvline(break_even, color="gray", ls=":", lw=1.5,
                        label=f"break-even a={break_even:.2f}")
    ax_twin = ax_rmse.twinx()
    ax_twin.fill_between(a_vals, ci_lo, ci_hi, color="#2ca02c", alpha=0.18)
    ax_twin.plot(a_vals, delta, "^--", color="#2ca02c", lw=1.2, label="Δ=naive−soft")
    ax_twin.axhline(0.0, color="#2ca02c", lw=0.6, alpha=0.5)
    ax_twin.set_ylabel("Δ RMSE (naive − soft), 95% paired CI", color="#2ca02c")
    ax_twin.tick_params(axis="y", labelcolor="#2ca02c")
    lines, labels = ax_rmse.get_legend_handles_labels()
    l2, lb2 = ax_twin.get_legend_handles_labels()
    ax_rmse.legend(lines + l2, labels + lb2, loc="upper right", fontsize=7)

    # --- Panel B: detection-error vs a (mean |error|), both arms + never-det.
    ax_det.plot(a_vals, [g["detection"]["mean_abs_error_naive"] for g in grid],
                "o-", color="#d62728", label="naive |latency err|")
    ax_det.plot(a_vals, [g["detection"]["mean_abs_error_soft"] for g in grid],
                "s-", color="#1f77b4", label="soft |latency err|")
    ax_det.set_xlabel("channel accuracy a")
    ax_det.set_ylabel("mean |detection-latency error| (opportunities)")
    ax_det.set_title("Time-to-mastery-detection error")
    ax_det.grid(alpha=0.3)
    ax_det_t = ax_det.twinx()
    ax_det_t.plot(a_vals, [g["detection"]["never_detected_rate_naive"] for g in grid],
                  "o:", color="#d62728", alpha=0.5, label="naive never-det rate")
    ax_det_t.plot(a_vals, [g["detection"]["never_detected_rate_soft"] for g in grid],
                  "s:", color="#1f77b4", alpha=0.5, label="soft never-det rate")
    ax_det_t.set_ylabel("never-detected rate", color="gray")
    ax_det_t.set_ylim(0, 1)
    ld, lbd = ax_det.get_legend_handles_labels()
    ld2, lbd2 = ax_det_t.get_legend_handles_labels()
    ax_det.legend(ld + ld2, lbd + lbd2, loc="upper right", fontsize=7)

    # --- Panel C: regret vs a, both arms.
    ax_reg.plot(a_vals, [g["regret"]["regret_naive"] for g in grid],
                "o-", color="#d62728", label="naive regret")
    ax_reg.plot(a_vals, [g["regret"]["regret_soft"] for g in grid],
                "s-", color="#1f77b4", label="soft regret")
    ax_reg.set_xlabel("channel accuracy a")
    ax_reg.set_ylabel("item-selection regret (oracle − policy)")
    ax_reg.set_title("Selection regret (proxy selector)")
    ax_reg.grid(alpha=0.3)
    ax_reg.legend(loc="upper right", fontsize=8)

    # a_eff anchors on every panel (vertical lines).
    for ax in (ax_rmse, ax_det, ax_reg):
        for anc in anchors:
            ax.axvline(anc["a_eff"], color="black", ls="--", lw=0.8, alpha=0.45)
            ax.text(anc["a_eff"], ax.get_ylim()[1], f" {anc['bias']}",
                    rotation=90, va="top", ha="right", fontsize=6, alpha=0.7)

    fig.suptitle(
        "Break-even: where ASR confidence must enter the mastery update\n"
        "Confidence-weighted KT lineage: Beck & Sison, Project LISTEN (2004-06)  ·  "
        f"n={n_students} students × {n_opps} opps × {len(_REGIMES)} regimes · seed={seed}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_ROOT), capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026, help="RNG seed (default 2026)")
    parser.add_argument("--n-students", type=int, default=500, help="students per regime")
    parser.add_argument("--n-opportunities", type=int, default=25, help="opps per student")
    parser.add_argument("--n-boot", type=int, default=500, help="paired bootstrap resamples")
    args = parser.parse_args(argv)

    t_start = time.time()

    # One root RNG; spawn one independent child stream per grid point so the
    # whole sweep is deterministic and order-independent within each point.
    root_rng = np.random.default_rng(args.seed)
    children = root_rng.spawn(len(_A_GRID))

    grid = [
        _run_grid_point(a, args.n_students, args.n_opportunities, args.n_boot, child)
        for a, child in zip(_A_GRID, children)
    ]

    anchors, density = _a_eff_anchors()
    break_even = _break_even_a(grid)
    runtime_s = time.time() - t_start

    _make_figure(
        grid, anchors, break_even,
        args.n_students, args.n_opportunities, args.seed, _OUT_PNG,
    )

    output = {
        "metadata": {
            "experiment": "break-even: soft-evidence vs naive BKT under a noisy channel",
            "lineage": "Beck & Sison, Project LISTEN (2004-06); virtual evidence: Pearl (1988)",
            "prediction": "pre-registered #4 (docs/predictions.md)",
            "seed": args.seed,
            "n_students": args.n_students,
            "n_opportunities": args.n_opportunities,
            "n_regimes": len(_REGIMES),
            "regimes": {n: {"s": p.s, "g": p.g, "t": p.t, "L0": p.L0}
                        for n, p in _REGIMES.items()},
            "channel": "symmetric bit-flip, P(observed==true)=a",
            "updaters": "naive=bkt_update(conf=1.0); soft=bkt_update(conf=a) on the SAME observed stream",
            "a_grid": _A_GRID,
            "n_boot": args.n_boot,
            "converge_eps": _CONVERGE_EPS,
            "detect_threshold": _DETECT_THRESHOLD,
            "prereq_threshold": _PREREQ_THRESHOLD,
            "miscue_density": density,
            "a_eff_formula": "(1-d)*(1-fp_per_100/100) + d*pooled_recall over {substitution,omission,insertion}",
            "a_eff_excluded_classes": list(_EXCLUDED_CLASSES),
            "date": datetime.date.today().isoformat(),
        },
        "break_even_a": break_even,
        "a_eff_anchors": anchors,
        "grid": grid,
        "volatile": {
            "_note": "non-deterministic fields; excluded from reproducibility comparisons",
            "git_commit": _git_head(),
            "runtime_seconds": round(runtime_s, 2),
        },
    }
    _OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Console summary ----
    print(f"\nBreak-even experiment — seed={args.seed}, "
          f"n={args.n_students}×{args.n_opportunities}×{len(_REGIMES)} regimes, "
          f"runtime={runtime_s:.1f}s\n")
    header = (f"{'a':>5} {'RMSE_naive':>11} {'RMSE_soft':>10} {'Δ(n-s)':>9} "
              f"{'Δ95%CI':>20} {'conv':>5} {'reg_naive':>10} {'reg_soft':>9}")
    print(header)
    print("-" * len(header))
    for g in grid:
        ci = g["delta_ci95"]
        print(f"{g['a']:>5.2f} {g['rmse_naive']:>11.4f} {g['rmse_soft']:>10.4f} "
              f"{g['delta_rmse']:>+9.4f} [{ci[0]:>+7.4f},{ci[1]:>+7.4f}] "
              f"{'yes' if g['converged'] else 'no':>5} "
              f"{g['regret']['regret_naive']:>10.4f} {g['regret']['regret_soft']:>9.4f}")
    print(f"\nBreak-even a (smallest a with |Δ|<{_CONVERGE_EPS}): {break_even}")
    print(f"\nMeasured miscue density d = {density:.4f}")
    print("a_eff anchors (measured operating points → effective channel accuracy):")
    for anc in anchors:
        print(f"  {anc['bias']:>6}: a_eff={anc['a_eff']:.4f} "
              f"(fp/100={anc['fp_per_100_correct_words']:.2f}, "
              f"pooled_recall={anc['pooled_recall']:.3f})")
    print(f"\nWrote {_OUT_JSON.relative_to(_PROJECT_ROOT)}")
    print(f"Wrote {_OUT_PNG.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
