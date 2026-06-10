# ReadCoach

![CI](https://github.com/gorajing/readcoach/actions/workflows/ci.yml/badge.svg)

**An eval-first agent harness for an AI reading tutor.**

ReadCoach listens to a child read a passage aloud, detects *miscues* (the reading
errors that drive instruction — not just transcription errors), maintains a per-child
model of what they've mastered, and runs a tutoring harness that decides *when to
intervene* and *how to help*. The point of the project is the layer most demos skip:
a measurement flywheel that proves the tutor is good and provably improves — without
ever experimenting on a real child.

## Status

Day 0 — scaffolding. The build is eval-first: every claim that lands in this README
will sit next to the command that reproduces it.

## Quickstart (will be kept true as the build progresses)

```bash
uv sync
uv run pytest
# benchmark + first numbers land here as they ship
```

## Layout

- `src/readcoach/` — ASR layer, miscue detector, learner model, tutor policy
- `evals/` — the eval harness: golden sets, scorers, regression gate
- `data/` — benchmark protocol and passages (see `data/README.md`)
- `docs/ARCHITECTURE.md` — component responsibilities and interfaces

## Judge validation

The LLM judge (GPT-family via codex CLI) is validated against human labels before
its verdicts are trusted to gate builds. Protocol: n=60 human-labeled turns across
3 dimensions (guidance, actionability, icap) — below the 100+ community norm; the
CIs reflect it. Each turn is labeled by a human rater using the same 1–5 rubric
anchors in `docs/labeling_rubric.md`, producing a binary passing verdict per
(turn, dimension) pair.

Agreement is reported as Cohen's kappa per dimension (Landis & Koch 1977). Floor:
**κ ≥ 0.4** (moderate agreement). Both κ ≥ 0.4 AND n ≥ 30 must hold for a dimension
to be gate-eligible. Dimensions below floor are **excluded from gating and reported
as untrusted** — this is a finding, not a failure.

Numbers land when the labeling session runs:
`uv run python scripts/validate_judge.py --labels evals/human_labels.csv --verdicts evals/results/judged_turns.jsonl`

## License

MIT
