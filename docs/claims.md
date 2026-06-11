# Claim ledger

The honesty mechanism that precedes any writeup or video. Every claimable
number in ReadCoach appears here exactly once, with its **precise meaning**
(what was measured, on what data), its **re-run command**, its **observed
value**, and the **source artifact** it lives in. Plus an explicit
[DO-NOT-CLAIM](#do-not-claim) list of everything that is *not* yet a claim.

**Discipline:** a writeup or a video may state a number only if that number
appears in the table below. If it is not here, it is not claimable.

## How to read this ledger

- **Observed value** is from a re-run performed while building this ledger
  (date 2026-06-10), not copied from another doc.
- Each row's **mode** is one of:
  - **RERUN** — the command was executed in full while building the ledger and
    the observed value comes from that fresh run. Fast and hermetic
    (fixtures / cache / pure-python).
  - **ARTIFACT-VERIFIED** — re-running requires a slow or external resource (a
    benchmark fetch, a wav2vec2 forward pass, a live LLM). Instead the committed
    artifact was confirmed to exist and to hash-match its lock / metadata. The
    row says so.
- `scripts/verify_claims.py` re-runs the **RERUN** rows mechanically and fails on
  any numeric drift beyond tolerance `1e-6`. ARTIFACT-VERIFIED rows are checked
  for artifact presence / hash, not re-computed.

---

## Benchmark

| claim | precise meaning | re-run command | observed value | source artifact | mode |
|-------|-----------------|----------------|----------------|-----------------|------|
| The miscue benchmark contains 88 clips. | Count of lines in committed `gold.jsonl`; one item per WAV stem. | `wc -l data/benchmark/gold.jsonl` | 88 | `data/benchmark/gold.jsonl` | RERUN |
| The benchmark spans 8 original passages. | Distinct `passage_id` values in gold. | `python3 scripts/verify_claims.py` (benchmark block) | 8 (p01–p08) | `data/benchmark/gold.jsonl` | RERUN |
| Coverage is at least 3 items per (passage × class) cell. | Min over the per-passage per-class counts in the manifest coverage matrix. | `python3 scripts/verify_claims.py` (benchmark block) | min cell = 3 (sub/om/ins/self_correction = 3 each; hesitation = 4) per passage | `data/benchmark/manifest.json` | RERUN |
| Gold totals 24 substitution / 24 omission / 24 insertion / 24 self_correction / 32 hesitation. | Class histogram over every `gold[].type` in gold.jsonl. | `python3 scripts/verify_claims.py` (benchmark block) | sub 24, om 24, ins 24, sc 24, hes 32 (128 total) | `data/benchmark/gold.jsonl` | RERUN |
| The 32 hesitations split 16 filler / 16 silence. | Histogram of `render` over hesitation gold entries. | `python3 scripts/verify_claims.py` (benchmark block) | filler 16, silence 16 | `data/benchmark/gold.jsonl` | RERUN |
| The release tarball hash-matches its lock and is fetchable from a committed command. | sha256 of `dist/readcoach-benchmark-0.1.0.tar.gz` equals `tarball.sha256` in the lock; `scripts/fetch_benchmark.py` verifies the same lock on download. | `shasum -a 256 dist/readcoach-benchmark-0.1.0.tar.gz` (compare to lock) | `d89a15431b788f136935ba8e4ef4c949dd279a599838a0e00706abb53eeacc48` (MATCH) | `dist/readcoach-benchmark-0.1.0.tar.gz`, `evals/golden/benchmark.lock` | ARTIFACT-VERIFIED |
| The committed baseline reproduces from fixtures with no model load. | The `--fixtures` sweep over the committed ASR cache reproduces `metrics.miscue` of `v0.json` bit-for-bit. | `uv run python scripts/run_benchmark.py --fixtures --version v0check --results-dir /tmp/claims-ci` then diff `metrics.miscue` vs `evals/results/v0.json` | miscue metrics identical | `evals/results/v0.json` | RERUN |

---

## Detector (per-class P/R/F1 across 3 biases, masking, false-positive collapse)

The detector numbers are the bias=none / prompt / strong sweep over the 88-clip
benchmark, micro-aggregated, with 95% bootstrap CIs (seed 1337, 1000 resamples).

| claim | precise meaning | re-run command | observed value | source artifact | mode |
|-------|-----------------|----------------|----------------|-----------------|------|
| Substitution recall collapses as ASR bias rises (none→prompt→strong). | Per-class substitution recall, micro-aggregated, at each bias. | `uv run python scripts/masking_curve.py --seed 1337 --n-boot 1000` | 0.667 → 0.167 → 0.083 | `evals/results/masking_curve.json` | RERUN |
| Omission recall also collapses with bias. | Omission recall at each bias. | `uv run python scripts/masking_curve.py --seed 1337 --n-boot 1000` | 0.917 → 0.375 → 0.292 | `evals/results/masking_curve.json` | RERUN |
| Insertion recall is masked less than substitution/omission. | Insertion recall at each bias. | `uv run python scripts/masking_curve.py --seed 1337 --n-boot 1000` | 0.792 → 0.542 → 0.458 | `evals/results/masking_curve.json` | RERUN |
| Substitution precision rises with bias. | Substitution precision at each bias. | `uv run python scripts/masking_curve.py --seed 1337 --n-boot 1000` | 0.068 → 0.444 → 0.667 | `evals/results/masking_curve.json` | RERUN |
| Insertion precision rises with bias. | Insertion precision at each bias. | `uv run python scripts/masking_curve.py --seed 1337 --n-boot 1000` | 0.463 → 0.619 → 0.917 | `evals/results/masking_curve.json` | RERUN |
| False positives per 100 correct words fall from 7.10 to 0.54 as bias goes none→strong. | `fp_per_100_correct_words`, micro-aggregated, at bias=none vs strong (intermediate prompt=0.70). | `uv run python scripts/masking_curve.py --seed 1337 --n-boot 1000` | 7.102 → 0.705 → 0.542 | `evals/results/masking_curve.json` | RERUN |
| Substitution-recall masking has a 95% bootstrap CI at bias=none. | Bootstrap 95% CI on substitution recall (none). | `uv run python scripts/masking_curve.py --seed 1337 --n-boot 1000` | [0.478, 0.850] | `evals/results/masking_curve.json` | RERUN |
| Omission-recall masking has a 95% bootstrap CI at bias=none. | Bootstrap 95% CI on omission recall (none). | `uv run python scripts/masking_curve.py --seed 1337 --n-boot 1000` | [0.786, 1.000] | `evals/results/masking_curve.json` | RERUN |
| The two-WER split: WER-vs-spoken vs WER-vs-target diverge under bias. | Mean WER of the ASR hypothesis against what was actually said vs against the target passage, at each bias. | `uv run python scripts/masking_curve.py --seed 1337 --n-boot 1000` | vs-spoken 0.079 → 0.027 → 0.028; vs-target 0.085 → 0.014 → 0.011 | `evals/results/masking_curve.json` | RERUN |

The masking *direction* (substitution + omission recall down, insertion less
affected, precision up) is the CONFIRMED verdict on pre-registered Prediction 1
(`docs/results_vs_predictions.md`). The number that matters for the
false-positive story is the **7.10 → 0.54** collapse, bought at the cost of
recall.

---

## BYO-ASR (second public ASR, scored unchanged)

| claim | precise meaning | re-run command | observed value | source artifact | mode |
|-------|-----------------|----------------|----------------|-----------------|------|
| A second public ASR (wav2vec2) scores through the exact same detector and metrics. | The committed wav2vec2 hypotheses score against gold via `readcoach-bench score`, covering all 88 items, producing its own per-class table. | `uv run readcoach-bench score --hypotheses wav2vec2=examples/wav2vec2_adapter/hypotheses_wav2vec2.jsonl` | n_covered = 88/88; renders a full per-class column | `examples/wav2vec2_adapter/hypotheses_wav2vec2.jsonl` | ARTIFACT-VERIFIED (committed hypotheses; wav2vec2 forward pass not re-run, scoring re-run) |
| The wav2vec2 column: alignment-class recall is comparable to baseline. | wav2vec2 sub/om/ins recall vs the bias=none baseline column. | (same score command) | wav2vec2 recall sub 0.833 / om 1.000 / ins 0.625 (baseline 0.667 / 0.917 / 0.792) | `examples/wav2vec2_adapter/hypotheses_wav2vec2.jsonl` | ARTIFACT-VERIFIED |
| Hesitation recall via timings: the timestamp-emitting ASR recovers the silence-hesitation subset the baseline cannot. | Hesitation recall for wav2vec2 (emits word timings) vs the committed faster-whisper baseline column. | (same score command) | wav2vec2 hesitation recall 0.500 vs baseline 0.156 | `examples/wav2vec2_adapter/hypotheses_wav2vec2.jsonl`, `evals/results/v0.json` | ARTIFACT-VERIFIED |
| The wav2vec2 path is noisier on alignment precision (its own fp/100 is high). | wav2vec2 `fp_per_100_correct_words` in the same scoring run. | (same score command) | 28.0 (baseline 7.10) | `examples/wav2vec2_adapter/hypotheses_wav2vec2.jsonl` | ARTIFACT-VERIFIED |

The point of the BYO row is the *contract*, not the wav2vec2 score: a different
ASR runs through the identical detector + metrics, and because it emits word
timings it picks up the timing-gated silence-hesitations (0.500 recall) that the
baseline, scored the same way, misses (0.156). That contrast is a property of the
**transcript style / timing emission**, not of the detector.

---

## BKT (parameter recovery, mastery floor, calibration, cold-start, break-even)

| claim | precise meaning | re-run command | observed value | source artifact | mode |
|-------|-----------------|----------------|----------------|-----------------|------|
| BKT parameter recovery error is at or below 0.06 in every regime. | Max `|fit − true|` over (s, g, t, L0) across 4 regimes; worst single residual. | `uv run python scripts/bkt_recovery.py` | worst residual = 0.06 (L0, high_guess); all others ≤ 0.04 | `evals/results/bkt_recovery.json` | RERUN |
| Mastery RMSE has a floor of 0.20–0.31 even with near-perfect parameter recovery. | Pooled mastery RMSE against the true latent state, per regime, min and max. | `uv run python scripts/bkt_recovery.py` | 0.196 (low_noise) … 0.310 (high_guess) | `evals/results/bkt_recovery.json` | RERUN |
| Calibration has a coverage hole below ~0.28: the model cannot predict a likely failure. | The two lowest reliability bins are empty because `predict_correct` is floored at the guess rate; minimum populated predicted probability. | `uv run python scripts/bkt_recovery.py` | lowest populated bin mean ≈ 0.281; bins below empty; Brier 0.139 | `evals/results/bkt_recovery.json` | RERUN |
| Cold-start: mastery estimates plateau by about k≈11 responses. | Pooled mastery RMSE by opportunity index; the index where the curve plateaus near its asymptote (the further k=20 dip is a horizon boundary effect). | `uv run python scripts/bkt_recovery.py` | RMSE 0.445 (k=1) → plateau ≈0.21 by k≈11 → 0.162 (k=20) | `evals/results/bkt_recovery.json` | RERUN |
| Soft-evidence BKT beats naive on mastery RMSE; the break-even is channel accuracy a = 0.90. | Smallest channel accuracy `a` with `|RMSE_naive − RMSE_soft| < 0.005` (paired streams, 500-student bootstrap, seed 2026). | `uv run python scripts/break_even.py --seed 2026` | break-even a = 0.90 | `evals/results/break_even.json` | RERUN |
| The soft-vs-naive RMSE advantage grows monotonically as the channel worsens. | Δ = RMSE_naive − RMSE_soft across the `a` grid; sign and monotonicity. | `uv run python scripts/break_even.py --seed 2026` | Δ from +0.0000 (a=0.99) to +0.1105 (a=0.55), monotone | `evals/results/break_even.json` | RERUN |
| Soft evidence trades slower detection for a more accurate estimate. | At a=0.55, soft's mean |latency error| and never-detected rate exceed naive's. | `uv run python scripts/break_even.py --seed 2026` | a=0.55: |latency err| soft 14.4 vs naive 7.8; never-detected soft 0.537 vs naive 0.291 | `evals/results/break_even.json` | RERUN |
| The benchmark's measured operating points sit at or above the a=0.90 break-even (TTS is "above break-even"). | Effective per-word channel accuracy `a_eff` at each measured bias, using d=0.0335 and pooled recall. | `uv run python scripts/break_even.py --seed 2026` | a_eff = 0.924 (none) / 0.972 (prompt) / 0.971 (strong) | `evals/results/break_even.json` | RERUN |

The honest nuance for any writeup: on **this TTS benchmark** soft evidence buys
little, because the measured operating points (a_eff ≈ 0.92–0.97) are at or above
the a=0.90 break-even — the payoff is real but lives at lower reliability than the
current benchmark exhibits.

---

## Policy (deterministic move selection)

| claim | precise meaning | re-run command | observed value | source artifact | mode |
|-------|-----------------|----------------|----------------|-----------------|------|
| The policy's WAIT rate is 0.435, inside the pre-set [0.35, 0.50] band. | Fraction of decision actions that are WAIT over the 88-item replay (seed 4101, policy 1.0.0). | `uv run python scripts/policy_replay.py` | 0.4352 (in band) | `evals/results/policy_replay.json` | RERUN |
| Self-correction immunity: self-corrections never trigger an escalation or a corrective move. | Self_correction events route to R-MID-SELF-CORRECTION (a non-corrective rule); they never escalate. | `uv run python scripts/policy_replay.py` | R-MID-SELF-CORRECTION fires 24×; 0 corrective moves on those turns | `evals/results/policy_replay.json` | RERUN |
| The default rule is never reached (every action is explained by a named rule). | Count of R-DEFAULT hits in the replay. | `uv run python scripts/policy_replay.py` | R-DEFAULT hits = 0 | `evals/results/policy_replay.json` | RERUN |

---

## Compiler / safety (policy harness vs the unconstrained model)

| claim | precise meaning | re-run command | observed value | source artifact | mode |
|-------|-----------------|----------------|----------------|-----------------|------|
| The live unconstrained model commits 69 policy violations across 3 reader profiles; the policy harness commits 0. | Total compiler-audited violations of the live `claude-sonnet-4-6` naive tutor over 3 scripted profiles, vs the gated tutor's invariants.violations. | live tutor: ARTIFACT-VERIFIED from `naive_live_audit.json`; harness 0: `uv run python scripts/run_ab.py` (v1 invariants.violations) | naive live 69; harness 0 | `evals/results/naive_live_audit.json`, `evals/results/ab_dev.json` | ARTIFACT-VERIFIED (live) + RERUN (harness 0) |
| Of the 69 live violations, 54 are mid-page coaching and 15 are missing AI reminders. | Per-rule split of the live naive audit: `never_coaches_mid_page` × 3 profiles + `periodic_ai_reminder` × 3 profiles. | ARTIFACT-VERIFIED from `naive_live_audit.json` | mid-page 54 (18×3); reminders 15 (5×3) | `evals/results/naive_live_audit.json` | ARTIFACT-VERIFIED |
| Every compiled safety/pedagogy rule cites the verbatim published sentence it operationalizes. | Each YAML rule carries a `verbatim_sentence` + `source.url`; safety rules cite Common Sense Media, pedagogy rules cite the architecture doc; operationalizations the source does not state are explicitly marked OURS. | `uv run python scripts/verify_claims.py` (policy-citation block) reads the YAML and asserts every non-deferred rule has a verbatim sentence + source | all rules carry verbatim_sentence + source.url | `policies/safety.yaml`, `policies/pedagogy.yaml` | RERUN |
| Receipt #1: a deliberately-broken detector turns CI red. | A pushed branch that suppresses all substitution detection fails the CI gate. | `gh run view 27299782480 --repo gorajing/readcoach --json conclusion,displayTitle` | conclusion=failure; title "DEMO: suppress all substitution detection on AsrResult (DO NOT MERGE)" | GitHub Actions run 27299782480 | ARTIFACT-VERIFIED (external CI run) |
| Receipt #2: the gate blocks a deliberately-worse tutor (v3) and names the breach, exit 1. | The v2→v3 gate comparison returns exit 1 with the `invariants.violations` breach named (v3 produces 30 violations vs v2's 0). | `uv run python scripts/run_ab.py` | v2→v3 exit 1, BLOCKED; breach "invariants.violations … new=30, threshold=0" | `evals/results/ab_dev.json` | RERUN |

---

## Judge (cross-family) — VALIDATION PENDING

The judge is a cross-family LLM grader: the tutor runs on Claude, the judge runs
on a GPT-family model via the `codex` CLI. The harness exists and the structured
verdict contract is enforced. **But no human labels have been collected yet** —
so every judge-validity claim and every judged A/B claim is **NOT YET
CLAIMABLE**. See [DO-NOT-CLAIM](#do-not-claim).

| claim | precise meaning | re-run command | observed value | source artifact | mode |
|-------|-----------------|----------------|----------------|-----------------|------|
| The judge harness exists and is cross-family (codex grades Claude). | Judge module invokes the `codex` CLI, enforces the score/passing/issues consistency matrix, fails loud on malformed output. | (code present) | harness present; `tests/test_judge.py` green | `evals/judge.py` | ARTIFACT-VERIFIED (code + tests) |
| Judge accuracy / kappa per dimension. | Cohen's kappa, TPR, TNR per judged dimension against ≥30 human labels. | (blocked — no labels) | **NOT YET CLAIMABLE — 0 labels collected** | (none yet) | NOT CLAIMABLE |
| The judged v1-vs-v2 guidance/actionability comparison. | Judged guidance + actionability on the held-out split (pre-registered Prediction 5). | (blocked — judge not validated, holdout not read) | **NOT YET CLAIMABLE** | (none yet) | NOT CLAIMABLE |

---

## Memory (LearnerMem)

| claim | precise meaning | re-run command | observed value | source artifact | mode |
|-------|-----------------|----------------|----------------|-----------------|------|
| LearnerMem v0 passes 6 of 6 memory-consistency probes (consistency_score 1.000). | Fraction of deterministic state probes that pass across a real SQLite close+reopen session boundary. | `uv run python scripts/learnermem_probes.py` | 6/6 = 1.000 | `evals/results/learnermem_v0.json` | RERUN |
| Completion-fragility finding: one generic failure un-completes a mastered skill (drops below 0.95) while staying servable (≥0.80). | P6 records that a single untagged failure on a mastered root skill drops mastery from ~0.98 to ~0.90 — below MASTERY_COMPLETED 0.95 but above the 0.80 floor. | `uv run python scripts/learnermem_probes.py` | mastery → 0.9042: ≥0.80 (invariant holds), <0.95 (completed-status lost) | `evals/results/learnermem_v0.json` | RERUN |
| Two-session continuity: mastery + due-reviews survive a store close and reopen. | The two-session demo plants S1 facts, closes + reopens SQLite, and the S2 store reproduces mastery bit-exact and orders due-reviews correctly. | `uv run python scripts/two_session_demo.py` | exits 0; S1 mastery reappears bit-exact in S2, class gate fires | (printed; demonstrated by P2/P3 in `learnermem_v0.json`) | RERUN |

---

## Flywheel (frozen split, pre-registration, gate, idempotent promotion)

| claim | precise meaning | re-run command | observed value | source artifact | mode |
|-------|-----------------|----------------|----------------|-----------------|------|
| The dev/held-out split is frozen one-way and hash-locked. | Both frozen persona files re-hash to the sha256 recorded in the lock; re-running freeze refuses if a fresh generation differs. | `uv run python scripts/freeze_split.py --verify` | VERIFY PASSED — dev (49) + holdout (49) match lock | `evals/golden/holdout.lock` | RERUN |
| The split lock records an auditable commit timestamp. | The lock carries `created_utc`; the freeze commit is in git history. | (read lock) | `created_utc = 2026-06-10T22:30:40Z` | `evals/golden/holdout.lock` | ARTIFACT-VERIFIED |
| Predictions were pre-registered before the runs that adjudicate them. | The predictions commit is an ancestor of the first benchmark-runner commit. | `git merge-base --is-ancestor 56624ff ed1761c && echo PRE-REGISTERED` | 56624ff (predictions, 11:10:51) is ancestor of ed1761c (runner, 11:15:00) | git history | RERUN |
| The gate emits exit codes that distinguish pass from breach. | v1→v2 passes (exit 0); v2→v3 breaches (exit 1). | `uv run python scripts/run_ab.py` | v1→v2 exit 0; v2→v3 exit 1 | `evals/results/ab_dev.json` | RERUN |
| promote_failure is idempotent on re-run (no double-promotion). | Re-running the A/B promotes v3's 30 failures once; a second run promotes 0 more (golden size stays 30). | `uv run python scripts/run_ab.py` | promoted v3=30, idempotent_on_rerun = True | `evals/results/ab_dev.json` | RERUN |
| The conclusion holds under ±30% persona-rate perturbation. | With persona rates scaled 0.70 and 1.30, both gates still pass and wait-rate stays sane. | `uv run python scripts/run_ab.py` | minus30 wait 0.351 / plus30 wait 0.457; both gates PASS | `evals/results/ab_dev.json` | RERUN |

---

## Turns (live generation)

| claim | precise meaning | re-run command | observed value | source artifact | mode |
|-------|-----------------|----------------|----------------|-----------------|------|
| 72 live tutor turns were generated with 0 invariant violations. | 24 turns × 3 reader profiles produced by the live model; the committed traces re-audit through the policy compiler to 0 violations. | `uv run python scripts/verify_claims.py` (turns-audit block re-audits the committed traces deterministically) | 72 turns, 0 invariant violations | `evals/results/turns_v1.jsonl`, `evals/results/trace_*.json` | RERUN (audit of committed traces) |
| The live turns ran on claude-sonnet-4-6 via the subscription CLI, prompt 1.0. | The model id, transport, and prompt version recorded on every turn row. | `python3 scripts/verify_claims.py` (turns metadata block) | model claude-sonnet-4-6; transport claude-cli; prompt 1.0 | `evals/results/turns_v1.jsonl` | RERUN (metadata) / ARTIFACT-VERIFIED (the live generation itself) |

The *generation* of the 72 turns required the live model (ARTIFACT-VERIFIED for
model/transport identity); the **0-violations property** is re-derivable
deterministically from the committed traces (RERUN), which is why a regression in
the policy compiler would surface here without re-calling the model.

---

## DO-NOT-CLAIM

These are stated so a writeup or a video can be checked against them. Each is a
boundary on external validity or a not-yet-collected measurement. **Do not claim
any of these. Do not imply them.**

### From the plan (standing limitations)

1. **TTS ≠ child speech.** Every benchmark number is on synthetic macOS `say`
   clips. No real-child numbers exist. The masking curve has **no
   external-validity anchor** until a consented real-child recording happens. All
   per-class recall is an **upper bound** on real-world performance, not an
   estimate of deployed performance. Do not present any benchmark number as a
   child-performance number.

2. **Hesitation-recall confound.** Hesitation and self-correction recall measure
   whether the ASR *preserves disfluencies* as much as they measure the detector
   (a transcript-style effect). Silence-hesitations are **undetectable without
   word timings** by construction. Do not attribute the hesitation numbers purely
   to detector quality, and do not compare hesitation recall across ASRs without
   stating the timing-emission difference.

3. **Judge labels: n=60 planned, ZERO collected as of now.** The validation
   protocol targets n=60 human-labeled turns (already below the 100+ community
   norm) — and **no labels have been collected yet**. Therefore **every
   judge-validity claim and every judged A/B claim is unclaimable.** No kappa, no
   judged-quality delta, no "the judge agrees with humans" — none of it exists.

4. **Laptop-CPU latency is not a latency claim.** Any real-time-factor (RTF)
   number is an **offline proxy** measured on a developer laptop. Do not make
   latency or responsiveness claims; the latency gate rule is report-only and
   currently unmeasured (`null`).

5. **speechocean762 is excluded from headline claims.** It is a *different task*
   (mispronunciation detection, not the 5 miscue classes). A loader exists and
   the corpus's mispronunciation prevalence was measured at 5.41%, and nothing
   more. Do not fold it into any miscue-detection headline number.

### Added by tonight's ledger work

6. **The A/B is deterministic-only.** `ab_dev.json` reports deterministic diffs
   (invariants, wait-rate, targeted next-items, serve-violations). The **judged**
   v1-vs-v2 comparison is **pending judge validation** — `ab_dev.json` itself
   states it adjudicates **no** prediction. Do not present the A/B as a judged
   quality win.

7. **The naive-villain numbers are action-level, not judged dimensions.** The 69
   live violations are compiler-audited rule hits (mid-page coaching, missing
   reminders). The **utterance-level judged dimensions** (guidance, actionability,
   warmth) for the villain are **pending** the same judge validation. Do not
   claim the villain is "worse on guidance quality" — only that it breaks
   action-level invariants.

8. **Subscription-CLI model pinning is coarser than API pinning.** The live
   model id (`claude-sonnet-4-6`) is recorded, but the subscription CLI does not
   guarantee an exact model snapshot the way a pinned API version would. State the
   model id; do not claim snapshot-exact reproducibility for the live runs.
