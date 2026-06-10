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

## License

MIT
