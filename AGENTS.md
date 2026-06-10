# Agent Coordination Rules

These rules apply to every automated agent (and apply just as well to humans)
working in this repository.

## Operating model

- Work is ticketed. One task at a time; every changed line must trace to the
  assigned task. Side-ideas are reported back, not folded into the diff.
- Small, reviewed, sequential changes land directly on `main`, one commit per
  green step. If multiple agents ever work in parallel, each takes its own
  branch or git worktree (`agent/<short-task-name>`) and `main` is integrated
  by a single controller.
- Do not overwrite, revert, reformat, or "clean up" another agent's work
  unless the controller explicitly asks for it.

## Non-negotiable project rules

1. **TDD**: failing test (show the failure) → minimal code → green (show it)
   → commit. A passing test that would not have failed before the change is
   decorative, not proof.
2. **Fail loud.** No `except → return default`, no fallback values in eval or
   judge paths, no silently skipped samples. A parse failure fails the run.
3. **Hermetic tests.** The suite runs with zero credentials —
   `tests/conftest.py` force-clears `WANDB_*` / `ANTHROPIC_*` / `GOOGLE_*`
   before any import. Tests that need the network carry the `network` marker
   and are deselected by default (and in CI).
4. **The language-boundary test (`tests/test_language_boundary.py`) is never
   weakened.** Its pattern list may grow; it never shrinks. The per-file
   allowlist is reserved for one documented future change.
5. **Every published number must reproduce from a committed command.** If a
   number appears in a doc or commit message, the command that produced it is
   in the tree.
6. Baselines are updated deliberately, in their own commit — investigate
   regressions, do not re-baseline to make CI green.

## Start of work

Before editing, report: assigned task, current branch, `git status -sb`,
files you expect to touch, and the checks you expect to run.

## Handoff

When finished, report: files changed, behavior changed, tests run with
literal pass/fail output, commit SHA(s), and known risks or assumptions.
Never report "it should work" — show the command output.
