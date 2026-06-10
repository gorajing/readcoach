# LearnerMem v0 — memory-consistency probes for a tutor agent

**Status:** v0 shipped. `consistency_score = 6/6 = 1.000`. One honest finding (P6).
**Artifact:** [`evals/results/learnermem_v0.json`](../evals/results/learnermem_v0.json)
**Code:** [`evals/learnermem.py`](../evals/learnermem.py) · **Runner:** [`scripts/learnermem_probes.py`](../scripts/learnermem_probes.py) · **Tests:** [`tests/test_learnermem.py`](../tests/test_learnermem.py)

## What memory-consistency probing is

A tutor agent's "memory" is **not** a chat transcript — it is the *learner model*
(BKT mastery + FSRS review cards + miscue-tagged observations) plus the *planner*
that reads it. When a child returns for session 2, the question this eval asks is:

> Is the system's session-2 behaviour consistent with what it learned about this
> child in session 1?

A tutor that forgets a planted struggle, drifts a mastery value on reopen,
re-teaches a finished skill, leaks one skill's struggles into another, or
over-reacts to a single bad data point has a **memory bug** — and that bug is
invisible to any single-session eval. LearnerMem makes it a first-class,
machine-checkable property.

## Lineage

This ports a known LLM-eval idea into a new domain:

- **PersonaMem** (arXiv:2504.14225) — probes whether an LLM agent's later turns
  stay faithful to a persona / facts established in earlier turns.
- **MemoryArena-style long-horizon consistency evals** — long-horizon checks that
  later behaviour remains consistent with earlier-established state.

**In-domain novelty (first-in-domain):** we apply memory-consistency probing to a
*tutor's learner-model + planner* rather than to LLM dialogue. The "persona" is
the learner model; the "later turns" are the planner's session-2 decisions; the
"planted facts" are reading struggles, mastery levels, completion state, and
review schedules.

## v0 scope (stated plainly)

These are **deterministic state probes**, not LLM-judged dialogue checks. The
planner + store *are* the memory; the utterance layer merely verbalises their
decisions downstream. So each probe is a machine-checkable assertion over
`(store state, planner decision)` — **no LLM, no network, no randomness, no
flakiness**. Each probe runs across a real session boundary: setup writes
session-1 facts, the SQLite store is **closed and reopened** on the same db, and
the check runs against that reopened session-2 store. The reopen is what makes
"did the fact survive the session boundary?" the thing actually under test.

**Named v1 extension point:** probing the *LLM utterance layer* for verbal
consistency — e.g. does the opener say *"you worked hard on silent-e last time,
let's keep going there"* in agreement with the planner's gated decision? That
requires an LLM judge and is explicitly out of v0 scope.

## The six probes

| ID | Probe | One-line claim |
|----|-------|----------------|
| **P1** | struggled-fact persistence | A silent-e **substitution** struggle planted in S1 keeps the silent-e-gated successor (`vowel_team_ai_ay`) **LOCKED** in S2 — mastery gate passes, **class gate fires**. |
| **P2** | mastery continuity | Every S1 mastery value reappears **bit-exact** in S2 (no drift on reopen). |
| **P3** | due-review consistency | A skill **failed** in S1 is due for review **before** a skill **passed** in S1 (FSRS ordering survives reopen). |
| **P4** | completed-skill memory | A skill completed (≥0.95) in S1 is **never served as `new`** in S2 (served-log + planner agree). |
| **P5** | cross-skill inference guard | Struggles planted **only on skill A** do not alter skill B's mastery or B's gating — memory is **skill-scoped**. |
| **P6** | over-personalization guard | **One generic (untagged) failure** on a mastered root skill keeps it **≥ threshold** (servable) with unlock status unchanged — it must **not** over-react to a single data point. |

Each probe has a `setup(store)` that plants the S1 facts and a
`check(store, ctx) -> ProbeResult(passed, evidence)` that asserts S2 consistency.
The aggregate `consistency_score` is the **fraction of probes passed**.

### Falsifiability (a probe that cannot fail is decoration)

Every probe ships with a **sabotage test**: between S1 and S2 the store is
corrupted (delete the planted struggle / drift a mastery value / invert a due
date / un-complete a skill / leak a struggle into B / drop below the floor) and
the probe is asserted to return `passed=False`. This proves each probe has teeth
— see `tests/test_learnermem.py`.

## Honest finding (P6 — over-personalization sensitivity)

P6 probes the **real** over-personalization risk. Note the class-gate semantics
are *correct by design*: one *tagged* failure in the last-5 window **should** lock
the gated successor (that is P1, and it is the system working). So P6 instead uses
a **generic (untagged)** failure on a mastered **root** skill (no class gate of
its own) and asserts the invariant the system *promises*: one bad observation
leaves the skill `mastery ≥ 0.80` (still satisfies any downstream prerequisite —
stays servable/unlocked) and its own unlock status unchanged. **That invariant
holds** (P6 passes).

But the same single generic failure **does** drop mastery from ~0.98 to ~0.90 —
**below `MASTERY_COMPLETED` (0.95)** — so the skill **loses "completed" status
from one data point**. This is a genuine over-personalization sensitivity in the
*completion* semantics (not the servability semantics). Per the project's
missed-prediction-is-content doctrine, the probe **passes on the promised
invariant and separately records the completed-status flip as a finding** rather
than redefining the probe to make the flip "pass". The finding is published in
`learnermem_v0.json` under `findings`.

**Suggested mitigation (v1 planner item):** completion *hysteresis* — require N
consecutive sub-threshold observations before un-completing a skill, so a single
bad read does not un-finish mastered material.

## Reproduce

```bash
uv run python scripts/learnermem_probes.py   # prints the table, writes the JSON
uv run pytest tests/test_learnermem.py -q     # 19 tests: pass + sabotage per probe
```
