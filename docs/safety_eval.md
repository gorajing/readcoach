# Safety eval — what exists and what's designed

What exists today: the policy compiler (`policies/*.yaml` → named executable checks,
severity-bucketed) with `invariants.violations` gated at zero in CI. Each rule carries
the verbatim published sentence it operationalizes and its source URL; where the
operationalization adds specifics the source doesn't state (e.g. the AI-reminder
cadence of one reminder per 20 turns), the YAML says so explicitly.

Designed but deliberately not built yet: an `discloses_ai_when_asked` check (blocked
on a child-initiated input channel — the trace schema has no child utterances yet),
judge-graded extensions of the lexicon checks (which are necessary-but-not-sufficient
by construction; they gate only after passing the same human-agreement validation bar
as the pedagogy dimensions), red-team trace categories, and age-banded strictness
with fail-closed defaults when age is unknown. The lexicon checks should be read as
tripwires, not as a safety argument.
