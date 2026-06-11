# Results vs Pre-registered Predictions

**Source run:** `uv run python scripts/masking_curve.py`
**Repro command:** `uv run python scripts/masking_curve.py --seed 1337 --n-boot 1000`
**Date:** 2026-06-10
**Benchmark:** 88 synthetic TTS clips (n=88), faster-whisper-small backend
**Note:** These numbers come from a synthetic TTS benchmark; TTS ≠ child speech. All figures are upper-bound estimates.

---

## Prediction 1 — Masking direction

**Verbatim prediction:**

> As bias strength rises (none → prompt → strong), WER on clean reads falls, but RECALL for substitution and omission classes also falls. Insertion recall is less affected. Precision may rise with stronger bias.

**Observed numbers (point estimates with 95% bootstrap CIs, 1000 resamples, seed=1337):**

| Bias | Class | Recall | 95% CI | Precision | 95% CI |
|------|-------|--------|--------|-----------|--------|
| none | substitution | 0.667 | [0.478, 0.850] | 0.068 | [0.042, 0.102] |
| prompt | substitution | 0.167 | [0.040, 0.320] | 0.444 | [0.125, 0.778] |
| strong | substitution | 0.083 | [0.000, 0.208] | 0.667 | [0.000, 1.000] |
| none | omission | 0.917 | [0.786, 1.000] | 0.564 | [0.426, 0.724] |
| prompt | omission | 0.375 | [0.200, 0.565] | 0.529 | [0.200, 1.000] |
| strong | omission | 0.292 | [0.118, 0.478] | 0.318 | [0.125, 1.000] |
| none | insertion | 0.792 | [0.621, 0.952] | 0.463 | [0.333, 0.605] |
| prompt | insertion | 0.542 | [0.343, 0.739] | 0.619 | [0.389, 0.840] |
| strong | insertion | 0.458 | [0.272, 0.667] | 0.917 | [0.727, 1.000] |

**WER table (mean across 88 items ± 95% CI):**

| Bias | WER vs spoken | 95% CI | WER vs target | 95% CI |
|------|---------------|--------|---------------|--------|
| none | 0.079 | [0.067, 0.091] | 0.085 | [0.074, 0.098] |
| prompt | 0.027 | [0.021, 0.035] | 0.014 | [0.009, 0.021] |
| strong | 0.028 | [0.021, 0.036] | 0.011 | [0.006, 0.017] |

**Verdict: CONFIRMED (substitution + omission masking) / PARTIAL (insertion "less affected")**

The substitution-recall drop from none→prompt→strong (0.667 → 0.167 → 0.083) and omission-recall drop (0.917 → 0.375 → 0.292) confirm the predicted masking direction strongly. WER vs target also falls sharply (0.085 → 0.014 → 0.011), reflecting the prompt-echo effect: the biased model emits the expected passage rather than what was actually said. Insertion recall does fall with bias (0.792 → 0.542 → 0.458), but substantially less than substitution or omission, consistent with the prediction's rationale that the model cannot "fill in" extra words the reader added. The "less affected" qualifier is confirmed in direction, though the absolute drop (~0.33) is larger than the phrase "less affected" might imply. Precision rises with bias exactly as predicted: substitution precision goes from 0.068 (none) → 0.444 (prompt) → 0.667 (strong), and insertion precision rises from 0.463 → 0.619 → 0.917. The tradeoff is sharp: strong bias buys near-zero false-positive rate (fp_per_100: 7.1 → 0.71 → 0.54) at the cost of missing almost all real miscues.

---

## Prediction 2 — Hesitation-recall confound

**Verbatim prediction:**

> Hesitation recall will be approximately zero at all bias settings because Whisper-family models normalize disfluencies out of transcripts — a transcript-style effect that is distinct from the masking effect in Prediction 1. The timing-based rule may partially recover silence-hesitations if word timestamps remain accurate.

**Observed numbers:**

| Bias | Hesitation recall | 95% CI | Hesitation precision | 95% CI |
|------|-------------------|--------|----------------------|--------|
| none | 0.156 | [0.038, 0.292] | 0.714 | [0.333, 1.000] |
| prompt | 0.031 | [0.000, 0.107] | 0.167 | [0.000, 0.501] |
| strong | 0.031 | [0.000, 0.105] | 0.250 | [0.000, 1.000] |

**Verdict: CONFIRMED**

Hesitation recall is near-zero at all bias settings (0.156 / 0.031 / 0.031). At bias=none, recall reaches only 0.156 despite having the timing-based silence rule active — and the 95% CI lower bound of 0.038 indicates this is reliably low, not noise. The partial recovery via silence hesitations does occur (recall is higher at none than prompt/strong), consistent with the prediction that the timing-based rule recovers some signal when the model is not biased. However, the recovery is weak. At prompt/strong, recall collapses to ~3%, confirming that bias additionally suppresses whatever weak timing signals existed. The wide CIs on precision at prompt/strong (including 0.0) reflect the very small number of predicted hesitations (6 and 4 respectively), making the precision estimate unreliable at those settings. The core prediction — near-zero hesitation recall at all biases due to Whisper's disfluency normalization — is confirmed.

---

## Prediction 3 — TTS-vs-real-kid gap

**Status: pending (experiment not yet run)**

Real-child audio comparison not yet available. The TTS benchmark establishes the upper bound; actual child-speech performance requires a held-out real-reader sample.

---

## Prediction 4 — Soft-evidence break-even

**Source run:** `uv run python scripts/break_even.py --seed 2026`
**Date:** 2026-06-10
**Setup:** n=500 students × 25 opportunities × 3 BKT regimes (easy_skill, hard_skill,
high_guess); paired naive (`conf=1.0`) vs soft (`conf=a`) updates on identical
observed-label streams; symmetric bit-flip channel P(observed==true)=a.

**Verbatim prediction:**

> Confidence-weighted (virtual-evidence) BKT updates will beat naive Bernoulli updates on mastery RMSE once detector reliability degrades. The break-even point is expected somewhere in the detector F1 = 0.6–0.85 range; above that range the two methods converge.

**Observed (Δ = RMSE_naive − RMSE_soft; >0 ⇒ soft wins; 95% paired student bootstrap, 500 resamples, seed=2026):**

| channel accuracy a | RMSE naive | RMSE soft | Δ (naive−soft) | Δ 95% CI | converged? |
|------|------------|-----------|----------------|----------|------------|
| 0.55 | 0.5049 | 0.3944 | +0.1105 | [+0.1032, +0.1185] | no |
| 0.60 | 0.4690 | 0.3898 | +0.0791 | [+0.0728, +0.0858] | no |
| 0.65 | 0.4341 | 0.3823 | +0.0518 | [+0.0456, +0.0579] | no |
| 0.70 | 0.4019 | 0.3596 | +0.0423 | [+0.0377, +0.0470] | no |
| 0.75 | 0.3685 | 0.3446 | +0.0238 | [+0.0202, +0.0272] | no |
| 0.80 | 0.3479 | 0.3332 | +0.0147 | [+0.0116, +0.0174] | no |
| 0.85 | 0.3217 | 0.3130 | +0.0087 | [+0.0069, +0.0105] | no |
| 0.90 | 0.2954 | 0.2917 | +0.0037 | [+0.0027, +0.0049] | **yes** |
| 0.95 | 0.2785 | 0.2780 | +0.0005 | [−0.0001, +0.0011] | yes |
| 0.99 | 0.2703 | 0.2703 | +0.0000 | [−0.0001, +0.0001] | yes |

**Break-even (smallest a with |Δ| < 0.005): a = 0.90.**

**Axis note (important for reading the prediction honestly).** The prediction is phrased
in *detector-F1* space; the experiment's primary, assumption-light axis is *channel
accuracy a* (per-opportunity P(observed label == true correctness)), because for an
imbalanced miscue-detection task F1 ↔ per-word label accuracy is only a loose mapping.
To anchor the prediction in reality we convert each measured operating point to an
*effective per-word channel accuracy* a_eff = (1−d)·(1−fp_rate) + d·pooled_recall, using
the benchmark's real miscue density d = 0.0335 (128 gold word-level events / 3 817 target
words) and pooled recall over {substitution, omission, insertion}:

| measured bias | fp/100 correct words | pooled recall | a_eff |
|---------------|----------------------|---------------|-------|
| none | 7.10 | 0.792 | 0.9244 |
| prompt | 0.70 | 0.361 | 0.9718 |
| strong | 0.54 | 0.278 | 0.9705 |

**Verdict: CONFIRMED (direction + convergence) / PARTIAL (break-even location).**

The directional claim holds cleanly: soft-evidence updates beat naive on mastery RMSE at
every degraded operating point, the advantage Δ rises monotonically as the channel worsens
(from +0.0000 at a=0.99 to +0.1105 at a=0.55, every below-break-even CI strictly above 0),
and the two methods converge above the break-even exactly as predicted. The break-even
itself lands at channel accuracy **a = 0.90** — slightly *above* the predicted band's upper
edge of 0.85 when the two axes are read as if interchangeable, which is why this is PARTIAL
rather than CONFIRMED: the F1↔a mapping is loose, so a numeric match was never guaranteed,
and the data put convergence a touch higher than the pre-registered 0.6–0.85 window. The
honest reality check is the a_eff anchors: this benchmark's *measured* operating points sit
at a_eff ≈ 0.92–0.97, i.e. at or just above the a=0.90 break-even, in the converged regime.
On the benchmark as it stands, soft evidence buys little (the sparse one-miscue-per-clip
density keeps the channel clean); its payoff is real and growing precisely where this
prediction said it would be — at lower reliability than the current TTS benchmark exhibits,
the regime a denser real-child read (Prediction 3) is expected to enter.

The selection-regret proxy (oracle − policy true mastery achieved at the horizon) is
positive at every a and shrinks as the channel improves (naive 0.782 → 0.352, soft 0.900 →
0.339 from a=0.55 to a=0.99), confirming noise degrades adaptive item selection too.

Detection latency reveals an honest trade-off that runs *against* soft on this metric: at
low a, soft's mean |latency error| is **larger** than naive's (14.4 vs 7.8 opportunities at
a=0.55) and its never-detected rate is higher (0.537 vs 0.291), converging only by a≈0.95.
This is the price of soft's calibration win: by discounting each noisy observation, soft's
P(L) climbs toward the 0.95 detection threshold more slowly, so it declares mastery later
and more often fails to cross the threshold within the 25-opportunity horizon. Naive's hard
updates over-trust the (noisy) evidence and cross 0.95 sooner — which looks like lower
latency but is the same over-confidence that inflates its RMSE. The two metrics tell a
consistent story: soft trades a slower, more conservative mastery declaration for a more
accurate mastery *estimate*. Never-detected and never-mastered rates (the latter ≈0.09 at
a=0.55, the irreducible floor from students who never truly master in-horizon) are reported
per grid point in the JSON, not hidden.

**Status: resolved.**

---

## Prediction 5 — v1 → v2 tutor diff

**Status: pending (experiment not yet run)**

Requires both tutor variants and the held-out evaluation split (T4+ scope).

<!-- holdout-prediction-5-verdict -->
### Holdout adjudication (run via scripts/run_ab_holdout.py --live)

**Verdict: UNADJUDICABLE**

dimension(s) ['guidance', 'actionability'] failed judge validation (not gate_eligible), so prediction #5's claim on them cannot be adjudicated. Reported as a finding, not silently dropped.

**Verbatim prediction #5:**
> v2 (mastery-conditioned) beats v1 (state-blind) on judged guidance and actionability on the held-out split.

**Why UNADJUDICABLE:** All judged dimensions failed the kappa ≥ 0.4 gate: guidance kappa=0.000, actionability kappa=0.302, icap kappa=0.207. Phases 3–4 (verbalize/judge) did not run.

The kappa ≥ 0.4 floor (Landis & Koch moderate-agreement threshold, pre-registered in scripts/validate_judge.py) was not met by any dimension. Without a validated judge, the LLM-judged scores on the holdout carry no agreed-upon reliability guarantee and cannot adjudicate the prediction.

**Deterministic results (DID run — 49 holdout sessions):**

- Invariant-gate v1→v2: exit 0 (PASS)
  - no breaches

**Path to future adjudication:**

A better judge must be validated on FRESH labels before any judge iteration can adjudicate prediction #5. The 60 labels used in judge_validation.json must NOT be reused for a new judge — doing so would overfit the judge to its own validation set, undermining the independence of the kappa estimate. Collect a new annotation batch (≥ 30 per dimension), run scripts/validate_judge.py on those fresh labels, and re-run this holdout runner only if ≥ 1 dimension clears the kappa ≥ 0.4 gate.

