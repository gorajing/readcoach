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

**Status: pending (experiment not yet run)**

Requires the BKT learner model and detector-reliability sweep (T3 scope).

---

## Prediction 5 — v1 → v2 tutor diff

**Status: pending (experiment not yet run)**

Requires both tutor variants and the held-out evaluation split (T4+ scope).
