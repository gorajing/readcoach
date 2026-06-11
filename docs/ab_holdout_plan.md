# Holdout A/B run plan (pre-registered)

Committed 2026-06-11, before `evals/results/judge_validation.json` exists and before
any tutor version has been run against the held-out split. This file is never edited
after its initial commit; deviations would be visible in git history.

## What runs

- **Versions:** v1 (state-blind) and v2 (mastery-conditioned), exactly as committed in
  `src/readcoach/tutor_versions.py` at the time of the run.
- **Data:** all 49 sessions in `evals/golden/persona_sessions_holdout.jsonl`
  (hash-locked by `evals/golden/holdout.lock`; verified before and after the run).
- **Deterministic metrics** (invariant violations, WAIT-rate, targeted next-items,
  serve violations) are computed over ALL holdout sessions for both versions, same
  aggregation as the dev A/B (`scripts/run_ab.py`).

## Judged-dimension sampling (fixed before any results)

- From each version's holdout replay, sample **60 decision turns** (miscue or
  page-end contexts), stratified by persona (20/20/20), seed **7331**, sampling code
  committed alongside the runner.
- Each sampled turn is verbalized live (claude-cli transport, model pinned, prompt
  version recorded) and judged by the cross-family judge on **only the dimensions
  that `judge_validation.json` marks `gate_eligible`** under the pre-committed rule
  (Cohen's kappa ≥ 0.4 AND n ≥ 30 — fixed in `scripts/validate_judge.py` before
  validation ran). Ineligible dimensions are reported as untrusted and are neither
  gated nor adjudicated.

## Adjudication rule for pre-registered prediction #5

Prediction #5 (frozen in `docs/predictions.md`) claims v2 beats v1 on judged guidance
and actionability on the held-out split.

- For each eligible dimension among {guidance, actionability}: v2 "beats" v1 iff the
  mean judged score difference (v2 − v1) is positive AND its seeded paired bootstrap
  95% CI over sampled turns excludes zero.
- Verdict mapping: both eligible dimensions positive-and-significant → CONFIRMED;
  one of two → PARTIAL; neither → MISSED; any dimension ineligible (judge failed
  validation) → that part is UNADJUDICABLE and reported as a finding, not silently
  dropped. The icap dimension is reported but is not part of prediction #5.

## What would invalidate the run

Any holdout-lock hash mismatch, any judge parse failure surviving retries, any
sampled turn skipped, or any post-hoc change to the sampling seed/size. All of these
abort rather than degrade.
