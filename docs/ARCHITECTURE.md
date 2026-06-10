# ReadCoach — Architecture

> The throughline: a **harness** (an inspectable decision + measurement loop), not a
> chatbot wrapper. A raw LLM is RLHF'd to be helpful and *give answers* — the opposite
> of teaching — so the policy decides and the model only verbalizes, and everything is
> measured.

## Data flow

```
 (audio, target_text)                          per-child state (SQLite; Redis optional)
        │                                             ▲   │
        ▼                                             │   ▼
   ┌─────────┐   words+timings  ┌──────────┐  miscues  ┌───────────────┐  action  ┌──────────────┐
   │  ASR    │────────────────▶ │ Miscue   │─────────▶ │ Learner model │────────▶ │ Tutor harness│
   │ (swap,  │  +confidences    │ detector │  +conf    │ (soft-evidence│          │ (move policy │
   │  bias   │                  │ (5-class)│           │  BKT + FSRS)  │          │  + planner)  │
   │  knob)  │                  └──────────┘           └───────────────┘          └──────────────┘
   └─────────┘                        │                       │                          │
                                      └───────────────────────┴──────────────┬───────────┘
                                                                              ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │  EVAL FLYWHEEL                                          │
                                  │  deterministic scorers (injected gold) · cross-family   │
                                  │  validated judge · policy-compiler invariants ·         │
                                  │  hermetic CI regression gate · promote_failure          │
                                  └────────────────────────────────────────────────────────┘
```

## Components

### `asr.py` — ASR layer (swappable)
- `transcribe(audio, target_text=None, bias="none"|"prompt"|"strong", cache_only=False)
  -> AsrResult` with **required** word timings + confidences.
- The bias-strength knob exists to *measure* the target-text tradeoff: prompting lowers
  WER (arXiv:2505.23627, 2506.11079) but can suppress detection of deviations from the
  expected text. `cache_only` raises on cache miss — CI never loads a model.
- The benchmark also accepts external hypothesis files, so any ASR can be scored
  without this module (see `docs/BENCHMARK.md`).

### `miscue.py` — Miscue detector
- Align hypothesis to `target_text` (jiwer ops); classify
  `substitution | omission | insertion | self_correction | hesitation`, each with a
  detector confidence consumed downstream.
- Hesitation uses inter-word timing gaps as well as a filler lexicon, because
  Whisper-family models often normalize disfluencies out of transcripts entirely.
- Scored on the synthetic-injection benchmark (gold by construction). Headline product
  metric: **false-positive interventions per 100 correctly-read words** — falsely
  "correcting" a child who read correctly is the product-killing failure mode.

### `learner_model.py` — the tutor's memory of the child
- Per-skill mastery via BKT, with **virtual-evidence (confidence-weighted) updates**:
  an observation from a noisy detector moves the posterior less than a certain one
  (conf=1.0 reduces to textbook BKT; conf=0.5 is information-free). Only prior art for
  observation-confidence KT is Beck & Sison's Project LISTEN work (2004–06).
- Honesty is demonstrated, not asserted: parameter-recovery study, calibration plots,
  cold-start curves, and a break-even analysis of naive vs confidence-weighted updates
  under controlled observation noise.
- Pace (WCPM per session) and engagement (hesitation-rate, session-length trends) live
  in the same state. FSRS schedules reviews. SQLite by default.

### `tutor.py` + `planner.py` — the decision loop, two timescales
1. **In-the-moment move** (pure, versioned rule matrix; LLM only verbalizes):
   `WAIT | ENCOURAGE | SCAFFOLDED_HINT | MODEL_THE_WORD | COMPREHENSION_PROMPT |
   NEXT_ITEM`, with SCAFFOLDED_HINT structured as an escalation ladder
   (bounce → highlight → phonetic help). Coaching happens at page-end, not mid-page;
   productive struggle is protected (WAIT-rate target 35–50%, per MetaCLASS
   arXiv:2602.02457); self-corrections are never "corrected."
2. **Quest planner**: mastery-gated traversal of a phonics scope-and-sequence graph
   whose prerequisite edges are **typed by miscue class** — what to re-teach depends on
   *which kind* of failure occurred. Completed items are never re-served without
   review intent.
- Cross-session continuity is a first-class behavior: session 2 opens from session 1's
  mastery state and due reviews.

### `policies/` — the invariant compiler
- Safety and pedagogy rules are **data**: each YAML rule carries the verbatim sentence
  of the policy it implements (our pedagogy rules, plus one published kids-tutor safety
  policy, cited by URL) and compiles to a named executable check with severity.
- Checks include: never says "wrong"; never coaches mid-page; never corrects a
  self-correction; no emotional-intimacy language (lexicon check, documented as
  necessary-but-not-sufficient); periodic AI-disclosure reminders (action-stream
  cadence); never re-serve completed content.
- Violations are a gated metric with threshold zero.

### `evals/` — the flywheel
- **Deterministic scorers first**: the benchmark's miscues are injected, so
  mistake-identification and location are scored against ground truth — no judge.
- **LLM judge from a different model family than the tutor**, only for dimensions
  ground truth can't reach (guidance quality, actionability, ICAP engagement), with
  hard verdict rules, schema-enforced output, and fail-loud parsing.
- **Judge validation**: TPR/TNR + Cohen's kappa with bootstrap CIs per dimension
  against hand-labeled turns, on a held-out split; dimensions below the kappa floor
  are excluded from gating and reported as untrusted.
- **Frozen held-out split** (content-hash lockfile committed before any A/B) and
  **pre-registered comparisons** (predictions committed before experiments run).
- **Hermetic CI gate**: committed ASR outputs make the full detector+scorer path
  deterministic in CI; a regression fails the build; latency is report-only in CI and
  threshold-enforced locally. Baseline updates are deliberate, separate commits.
- **`promote_failure(trace)`**: gate violations and low-scoring turns grow the golden
  set, with provenance; golden-set growth is charted per batch.
