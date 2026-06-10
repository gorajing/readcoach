"""Policy compiler — invariants-as-data -> named executable checks (T4.3).

The flagship safety/pedagogy artifact: the project's invariants live as DATA
(``policies/*.yaml``), each rule carrying the VERBATIM policy sentence it
implements and the source it came from.  This module:

  1. ``load_policies(dir)``  — parse + VALIDATE every rule fail-loud (a rule
     without a verbatim_sentence, a source, a known severity, or a check block
     is a hard error; an unknown check type fails at compile time).
  2. ``compile_rules(rules)`` — turn each non-deferred rule into a named
     ``Check``: a callable over a ``SessionTrace`` returning a list of
     ``Finding`` (rule_id, severity, turn_index, message).
  3. ``audit(trace, checks)`` — run all checks, bucket findings by severity, and
     expose ``violations`` = the count of severity-ERROR findings (the gated
     metric, threshold 0).

Design stance (mirrors docs/ARCHITECTURE.md): the LLM verbalizes, the POLICY
decides — and here the policy is *checked*.  Lexicon checks are documented as
NECESSARY-NOT-SUFFICIENT: a hit is sufficient to flag, but the absence of a hit
does not prove safety (the cross-family judge is the named superset extension).

The ``AuditReport`` severity-bucket shape follows TheDose's corpus_audit.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from readcoach.trace import SessionTrace

# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

_SEVERITIES = ("error", "warning")


@dataclass(frozen=True)
class Rule:
    """A validated policy rule (one entry in a policies/*.yaml `rules` list)."""

    id: str
    severity: str  # "error" | "warning"
    verbatim_sentence: str
    source_name: str
    source_url: str
    source_accessed: str
    check_type: str
    check_params: dict[str, Any]
    deferred: bool = False
    policy_set: str = ""


@dataclass(frozen=True)
class Finding:
    """One invariant violation against a single turn (severity-tagged)."""

    rule_id: str
    severity: str  # "error" | "warning"
    turn_index: int
    message: str


# A compiled check: a named callable over a trace.  Carries its rule id so the
# audit can index findings without re-parsing.
@dataclass(frozen=True)
class Check:
    rule_id: str
    severity: str
    fn: Callable[[SessionTrace], list[Finding]]

    def __call__(self, trace: SessionTrace) -> list[Finding]:
        return self.fn(trace)


@dataclass
class AuditReport:
    """Findings + severity bucket counts + the gated ``violations`` metric."""

    findings: list[Finding] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {
            sev: sum(1 for f in self.findings if f.severity == sev)
            for sev in _SEVERITIES
        }

    @property
    def violations(self) -> int:
        """Count of severity-ERROR findings — the gated metric (threshold 0)."""
        return sum(1 for f in self.findings if f.severity == "error")

    @property
    def clean(self) -> bool:
        return self.violations == 0


# ---------------------------------------------------------------------------
# Loading / validation (fail-loud)
# ---------------------------------------------------------------------------

def load_policies(policies_dir: str | Path) -> list[Rule]:
    """Load + validate every rule across all ``*.yaml`` files in ``policies_dir``.

    Fail-loud: a rule missing ``verbatim_sentence``, ``source`` (name/url), a
    valid ``severity``, or a ``check`` block raises ``ValueError`` / ``KeyError``
    naming the offending rule.  Duplicate rule ids across files also raise.
    """
    d = Path(policies_dir)
    if not d.is_dir():
        raise NotADirectoryError(f"policies dir not found: {d}")

    rules: list[Rule] = []
    seen_ids: set[str] = set()
    for path in sorted(d.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or "rules" not in doc:
            raise ValueError(f"{path}: top-level 'rules' list is required")
        policy_set = doc.get("policy_set", "")
        raw_rules = doc["rules"]
        if not isinstance(raw_rules, list):
            raise ValueError(f"{path}: 'rules' must be a list")
        for raw in raw_rules:
            rule = _parse_rule(raw, path, policy_set)
            if rule.id in seen_ids:
                raise ValueError(f"{path}: duplicate rule id '{rule.id}'")
            seen_ids.add(rule.id)
            rules.append(rule)
    return rules


def _parse_rule(raw: dict, path: Path, policy_set: str) -> Rule:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: every rule must be a mapping; got {type(raw).__name__}")

    rule_id = raw.get("id")
    if not rule_id:
        raise KeyError(f"{path}: a rule is missing required field 'id'")

    severity = raw.get("severity")
    if severity not in _SEVERITIES:
        raise ValueError(
            f"{path}: rule '{rule_id}' has invalid severity {severity!r} "
            f"(expected one of {_SEVERITIES})"
        )

    verbatim = raw.get("verbatim_sentence")
    if not verbatim or not str(verbatim).strip():
        raise ValueError(
            f"{path}: rule '{rule_id}' is missing a non-empty 'verbatim_sentence' "
            f"(every invariant must quote the policy sentence it implements)"
        )

    source = raw.get("source")
    if not isinstance(source, dict):
        raise KeyError(f"{path}: rule '{rule_id}' is missing required 'source' mapping")
    source_name = source.get("name")
    source_url = source.get("url")
    if not source_name or not source_url:
        raise ValueError(
            f"{path}: rule '{rule_id}' source must carry both 'name' and 'url'"
        )

    deferred = bool(raw.get("deferred", False))

    # A deferred rule documents intent but compiles no executable check; it may
    # still omit the check block.  An active rule MUST carry a check block.
    check = raw.get("check")
    if not deferred:
        if not isinstance(check, dict):
            raise KeyError(
                f"{path}: active rule '{rule_id}' is missing required 'check' mapping"
            )
        check_type = check.get("type")
        if not check_type:
            raise KeyError(f"{path}: rule '{rule_id}' check is missing 'type'")
        check_params = check.get("params") or {}
        if not isinstance(check_params, dict):
            raise ValueError(f"{path}: rule '{rule_id}' check.params must be a mapping")
    else:
        check_type = (check or {}).get("type", "") if isinstance(check, dict) else ""
        check_params = {}

    return Rule(
        id=rule_id,
        severity=severity,
        verbatim_sentence=str(verbatim).strip(),
        source_name=str(source_name),
        source_url=str(source_url),
        source_accessed=str(source.get("accessed", "")),
        check_type=str(check_type),
        check_params=dict(check_params),
        deferred=deferred,
        policy_set=str(policy_set),
    )


# ---------------------------------------------------------------------------
# Compilation — Rule -> Check
# ---------------------------------------------------------------------------

def compile_rules(rules: list[Rule]) -> list[Check]:
    """Compile every NON-deferred rule into a named executable ``Check``.

    Fail-loud: a rule whose ``check_type`` is not one of the known builders
    raises ``ValueError`` ("unknown check type").  Deferred rules are skipped
    (they document intent but have no executable check).
    """
    checks: list[Check] = []
    for rule in rules:
        if rule.deferred:
            continue
        builder = _CHECK_BUILDERS.get(rule.check_type)
        if builder is None:
            raise ValueError(
                f"rule '{rule.id}': unknown check type {rule.check_type!r} "
                f"(known: {sorted(_CHECK_BUILDERS)})"
            )
        fn = builder(rule)
        checks.append(Check(rule_id=rule.id, severity=rule.severity, fn=fn))
    return checks


def audit(trace: SessionTrace, checks: list[Check]) -> AuditReport:
    """Run all ``checks`` over ``trace`` and bucket the findings by severity."""
    report = AuditReport()
    for check in checks:
        report.findings.extend(check(trace))
    return report


# ---------------------------------------------------------------------------
# Check builders — each returns a closure capturing the rule's params.
#
# A builder is `Rule -> (SessionTrace -> list[Finding])`.  Lexicon matching is
# WORD-BOUNDED + case-insensitive so "wrongful" never trips "wrong".
# ---------------------------------------------------------------------------

def _compile_lexicon(phrases: list[str]) -> re.Pattern[str]:
    """Word-bounded, case-insensitive alternation over a lexicon of phrases.

    ``\\b`` on both ends so a phrase only matches as whole word(s); a phrase may
    itself be multi-word ("best friend").  Phrases are regex-escaped.
    """
    if not phrases:
        # Match nothing.  An empty lexicon is a no-op check, not a match-all.
        return re.compile(r"(?!x)x")
    alts = "|".join(re.escape(p) for p in phrases)
    return re.compile(rf"\b(?:{alts})\b", re.IGNORECASE)


def _finding(rule: Rule, turn_index: int, message: str) -> Finding:
    return Finding(
        rule_id=rule.id,
        severity=rule.severity,
        turn_index=turn_index,
        message=message,
    )


def _build_never_says_wrong(rule: Rule) -> Callable[[SessionTrace], list[Finding]]:
    pattern = _compile_lexicon(rule.check_params.get("lexicon", []))

    def check(trace: SessionTrace) -> list[Finding]:
        out: list[Finding] = []
        for t in trace.turns:
            if t.utterance is None:
                continue
            m = pattern.search(t.utterance)
            if m:
                out.append(_finding(
                    rule, t.turn_index,
                    f"utterance contains forbidden term {m.group(0)!r} "
                    f"(necessary-not-sufficient lexicon)",
                ))
        return out

    return check


def _build_never_coaches_mid_page(rule: Rule) -> Callable[[SessionTrace], list[Finding]]:
    coaching = set(rule.check_params.get("coaching_moves", []))

    def check(trace: SessionTrace) -> list[Finding]:
        out: list[Finding] = []
        for t in trace.turns:
            if (not t.at_page_end) and t.action_move in coaching:
                out.append(_finding(
                    rule, t.turn_index,
                    f"coaching move {t.action_move!r} fired mid-page "
                    f"(coaching is page-end only)",
                ))
        return out

    return check


def _build_never_corrects_self_correction(
    rule: Rule,
) -> Callable[[SessionTrace], list[Finding]]:
    corrective_moves = set(rule.check_params.get("corrective_moves", []))
    pattern = _compile_lexicon(rule.check_params.get("corrective_lexicon", []))

    def check(trace: SessionTrace) -> list[Finding]:
        out: list[Finding] = []
        for t in trace.turns:
            if t.miscue_type != "self_correction":
                continue
            if t.action_move in corrective_moves:
                out.append(_finding(
                    rule, t.turn_index,
                    f"corrective move {t.action_move!r} applied to a "
                    f"self-correction (must be honored, never corrected)",
                ))
                continue
            if t.utterance is not None and pattern.search(t.utterance):
                m = pattern.search(t.utterance)
                out.append(_finding(
                    rule, t.turn_index,
                    f"corrective phrase {m.group(0)!r} applied to a "  # type: ignore[union-attr]
                    f"self-correction",
                ))
        return out

    return check


def _build_no_emotional_intimacy(rule: Rule) -> Callable[[SessionTrace], list[Finding]]:
    pattern = _compile_lexicon(rule.check_params.get("lexicon", []))

    def check(trace: SessionTrace) -> list[Finding]:
        out: list[Finding] = []
        for t in trace.turns:
            if t.utterance is None:
                continue
            m = pattern.search(t.utterance)
            if m:
                out.append(_finding(
                    rule, t.turn_index,
                    f"utterance contains intimacy-cultivating phrase {m.group(0)!r} "
                    f"(necessary-not-sufficient; judge-graded extension planned)",
                ))
        return out

    return check


def _build_periodic_ai_reminder(rule: Rule) -> Callable[[SessionTrace], list[Finding]]:
    window = int(rule.check_params.get("window_turns", 20))
    min_per_window = int(rule.check_params.get("min_reminders_per_window", 1))
    if window < 1:
        raise ValueError(f"rule '{rule.id}': window_turns must be >= 1")

    def check(trace: SessionTrace) -> list[Finding]:
        turns = trace.turns
        n = len(turns)
        # A session shorter than (or equal to) one window is covered iff it has
        # at least min_per_window reminders anywhere in it.  Longer sessions are
        # covered iff EVERY length-`window` sliding window has >= min reminders.
        if n == 0:
            return []
        reminders = [bool(t.is_ai_reminder) for t in turns]
        if n <= window:
            if sum(reminders) >= min_per_window:
                return []
            return [_finding(
                rule, 0,
                f"session of {n} turns has {sum(reminders)} AI-identity "
                f"reminder(s); cadence requires >= {min_per_window} per "
                f"{window}-turn window",
            )]
        out: list[Finding] = []
        for start in range(0, n - window + 1):
            count = sum(reminders[start:start + window])
            if count < min_per_window:
                out.append(_finding(
                    rule, start,
                    f"turns [{start}, {start + window - 1}] contain {count} "
                    f"AI-identity reminder(s); cadence requires "
                    f">= {min_per_window} per {window}-turn window",
                ))
        return out

    return check


def _build_never_reserves_completed_item(
    rule: Rule,
) -> Callable[[SessionTrace], list[Finding]]:
    new_reason = rule.check_params.get("new_reason", "new")

    def check(trace: SessionTrace) -> list[Finding]:
        out: list[Finding] = []
        # Completed = mastered at session start, plus anything already served as
        # new earlier in THIS trace (serving it again as new is a re-serve).
        completed: set[str] = set(trace.completed_skills_at_start)
        for t in trace.turns:
            if t.served_reason == new_reason and t.skill_id is not None:
                if t.skill_id in completed:
                    out.append(_finding(
                        rule, t.turn_index,
                        f"skill {t.skill_id!r} served as {new_reason!r} but was "
                        f"already completed (re-serve without review intent)",
                    ))
                completed.add(t.skill_id)
        return out

    return check


_CHECK_BUILDERS: dict[str, Callable[[Rule], Callable[[SessionTrace], list[Finding]]]] = {
    "never_says_wrong": _build_never_says_wrong,
    "never_coaches_mid_page": _build_never_coaches_mid_page,
    "never_corrects_self_correction": _build_never_corrects_self_correction,
    "no_emotional_intimacy": _build_no_emotional_intimacy,
    "periodic_ai_reminder": _build_periodic_ai_reminder,
    "never_reserves_completed_item": _build_never_reserves_completed_item,
}
