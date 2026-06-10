"""Cross-family judge — OpenAI/Codex via subscription CLI (tutor is Claude).

Family separation: the tutor runs on Claude (Anthropic); the judge runs on
GPT-family models via the ``codex`` CLI subscription.  This preserves the
cross-family intent: an OpenAI model judges Claude outputs.

## codex exec output contract (probed 2026-06-10, codex-cli 0.130.0)

Invocation::

    codex exec --skip-git-repo-check -s read-only \\
        -c 'service_tier="fast"' "<prompt>" 2>/dev/null

Stdout (with stderr suppressed via ``2>/dev/null``):
    The model's final response text and nothing else.
    No "tokens used" line, no session header, no harness noise.

Stdout (without stderr suppression, for diagnostics):
    Lines on stdout:  ``codex`` header, model response, ``tokens used``,
    token count, model response (repeated).  The last non-empty line is the
    model response.

Contract chosen: **pipe stderr to /dev/null; treat entire stdout as the
model's response text** (strip leading/trailing whitespace).  This is the
simplest, most stable contract.

Model identity: the active model is printed in the session header (stderr),
e.g. ``model: gpt-5.5``.  We record it in model_meta as ``"codex-default"``
because we do not parse stderr; the CLI version is captured from
``codex --version``.

## Judged dimensions

``JUDGED_DIMENSIONS = ("guidance", "actionability", "icap")`` — these are
the three dimensions human labelers score in docs/labeling_rubric.md.
Deterministic dimensions (mistake-ID, location) are NEVER judged here;
passing a deterministic dim to judge_turn raises ValueError.

## Verdict dataclass

``Verdict`` is a frozen dataclass::

    dimension: str
    score: int           # 1–5
    passing: bool        # true iff score >= 4
    issues: list[str]    # non-empty iff score <= 2 (blocking issues)
    rationale: str
    model_meta: dict     # {"model": ..., "cli_version": ...}

## Consistency matrix (enforced after parse)

- score >= 4  →  passing MUST be True
- score <= 2  →  passing MUST be False AND issues MUST be non-empty
- passing=True with non-empty issues   → InconsistentVerdictError
- passing=False with empty issues      → InconsistentVerdictError

## Retry policy

| Error class          | Max retries | Retry prompt                              |
|----------------------|-------------|-------------------------------------------|
| parse_error          | 2           | includes original output + error message  |
| inconsistent_verdict | 2           | includes original JSON + inconsistency    |
| transport_error      | 2           | bare re-attempt (no context injected)     |

After retries exhausted → raises ``JudgeError`` naming class + attempts.
NEVER returns a default verdict.  NEVER skips a sample.

## Thread-safety / partial results

``judge_trace`` is sequential.  If any turn's judge exhausts retries, the
exception propagates immediately; NO partial results are written.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

JUDGED_DIMENSIONS: tuple[str, ...] = ("guidance", "actionability", "icap")

# Dimensions that are computed deterministically and must NOT be judged here.
_DETERMINISTIC_DIMS: tuple[str, ...] = (
    "mistake_id",
    "location",
    "miscue_id",
    "miscue_location",
)

# Retry budget per error class.
_MAX_RETRIES = 2

# subprocess timeout in seconds.
_SUBPROCESS_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class JudgeError(RuntimeError):
    """Raised when a dimension judgment exhausts all retries.

    Attributes
    ----------
    error_class : str
        One of "parse_error", "inconsistent_verdict", "transport_error".
    attempts : int
        Total attempts made (1 original + retries).
    """

    def __init__(self, error_class: str, attempts: int, detail: str = "") -> None:
        self.error_class = error_class
        self.attempts = attempts
        msg = (
            f"JudgeError: exhausted {attempts} attempt(s) "
            f"(error_class={error_class!r})"
        )
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)


class ParseError(ValueError):
    """Raised when the model output cannot be parsed as a valid Verdict JSON."""


class InconsistentVerdictError(ValueError):
    """Raised when a parsed Verdict violates the consistency matrix.

    Consistency rules
    -----------------
    1. score >= 4  →  passing must be True
    2. score <= 2  →  passing must be False AND issues must be non-empty
    3. passing=True with non-empty issues is always inconsistent
    4. passing=False with empty issues is always inconsistent
    """


# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """Structured judgment for one (turn, dimension) pair."""

    dimension: str
    score: int  # 1–5
    passing: bool
    issues: list[str]
    rationale: str
    model_meta: dict

    def __post_init__(self) -> None:
        if not isinstance(self.issues, list):
            # Thaw the frozen field safely for the post_init check.
            object.__setattr__(self, "issues", list(self.issues))


# ---------------------------------------------------------------------------
# Rubric anchors (quoted verbatim from docs/labeling_rubric.md)
# ---------------------------------------------------------------------------

_RUBRIC_GUIDANCE = """
Dimension: guidance — "Is this a warm, pedagogically sound thing to say to a 5–8-year-old at this moment?"

Score anchors:
5 — Warm, age-perfect, motivation-protective, and exactly fits the move. A skilled reading teacher would say this.
4 — Good and appropriate; minor stiffness or a slightly-too-adult word, but lands well.
3 — Serviceable but flat or generic ("Good job.") — not harmful, not memorable, doesn't quite fit the moment.
2 — Off-tone for a young child (clinical, condescending, or over-effusive), or only loosely related to the move.
1 — Cold, discouraging, confusing, or contradicts the move (e.g. lectures during a WAIT).
""".strip()

_RUBRIC_ACTIONABILITY = """
Dimension: actionability — "Can the child actually DO something with this line right now?"

Score anchors:
5 — Gives one clear, concrete, child-doable next action ("look at the first two letters and blend them"). The child knows exactly what to try.
4 — Actionable, but the step is slightly broad or assumes a skill not yet cued ("sound it out").
3 — Points in a direction but leaves the how unspecified ("try again").
2 — Mostly affect with a faint nudge ("you can do it, keep looking!") — little to act on.
1 — No actionable content, or the action is wrong for the move (asks the child to re-decode a word the move just modeled).
""".strip()

_RUBRIC_ICAP = """
Dimension: icap — ICAP engagement level invited (Chi & Wylie, 2014): Passive < Active < Constructive < Interactive.

Score anchors:
5 — Interactive / strongly Constructive: Invites the child to generate, explain, or reason ("Why do you think the wolf did that?", "What sound does this part make, and how do you know?").
4 — Constructive: Prompts the child to produce something new — predict, infer, build the word from parts.
3 — Active: Asks the child to do the focused thing (blend these sounds, reread this line) without generating new reasoning.
2 — Passive-leaning: Mostly tells; the child receives rather than acts (a bare model with no invitation to try).
1 — Passive / disengaging: No cognitive invitation at all, or shuts engagement down.

Note: a correct MODEL_THE_WORD line is expected to be lower on ICAP — the move's job is to hand over a stuck word. Score it on whether it models cleanly and re-invites reading, not penalize it for not being Interactive.
""".strip()

_RUBRICS: dict[str, str] = {
    "guidance": _RUBRIC_GUIDANCE,
    "actionability": _RUBRIC_ACTIONABILITY,
    "icap": _RUBRIC_ICAP,
}

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """You are a reading-tutor quality judge.
You will be given a tutor turn (context + utterance) and asked to score ONE dimension.
You MUST output ONLY valid JSON — no prose, no markdown fences, no explanation outside the JSON.

HARD VERDICT RULES (you must follow these exactly):
- score >= 4 → "passing" MUST be true
- score <= 2 → "passing" MUST be false AND "issues" MUST be non-empty (list at least one blocking issue)
- score == 3 → "passing" may be true or false at your discretion, but be consistent with issues

Output schema (strict JSON, all fields required):
{
  "dimension": "<dimension name>",
  "score": <integer 1-5>,
  "passing": <true|false>,
  "issues": [<string>, ...],
  "rationale": "<one or two sentence explanation>"
}"""


def _build_prompt(turn: dict, dimension: str) -> str:
    """Build the judge prompt for a single (turn, dimension) pair."""
    rubric = _RUBRICS[dimension]
    utterance = turn.get("utterance", turn.get("tutor_utterance", ""))
    move = turn.get("move", turn.get("policy_move", "UNKNOWN"))
    miscue = turn.get("miscue", turn.get("miscue_context", ""))
    context_parts = [f"Move: {move}"]
    if miscue:
        context_parts.append(f"Miscue context: {miscue}")
    context_str = "\n".join(context_parts)

    return (
        f"{_JUDGE_SYSTEM}\n\n"
        f"--- RUBRIC ---\n{rubric}\n\n"
        f"--- TURN TO JUDGE ---\n{context_str}\n"
        f"Utterance: {utterance!r}\n\n"
        f"Score the dimension '{dimension}' for this utterance. "
        f"Output ONLY the JSON object."
    )


def _build_retry_parse_prompt(
    original_prompt: str, bad_output: str, error_msg: str
) -> str:
    """Retry prompt for parse errors — injects the failure reason."""
    return (
        f"{original_prompt}\n\n"
        f"--- PREVIOUS ATTEMPT FAILED ---\n"
        f"Your previous output failed JSON parsing with error: {error_msg}\n"
        f"Your previous output was:\n{bad_output!r}\n\n"
        f"Output ONLY valid JSON matching the schema. No prose, no fences."
    )


def _build_retry_inconsistency_prompt(
    original_prompt: str, bad_json: str, inconsistency_msg: str
) -> str:
    """Retry prompt for consistency violations — injects the rule that was broken."""
    return (
        f"{original_prompt}\n\n"
        f"--- PREVIOUS ATTEMPT FAILED ---\n"
        f"Your previous JSON output was inconsistent: {inconsistency_msg}\n"
        f"Your previous JSON was:\n{bad_json}\n\n"
        f"Fix the inconsistency and output ONLY valid JSON. "
        f"Remember: score>=4 → passing=true; score<=2 → passing=false AND issues non-empty."
    )


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


@dataclass
class CodexCliTransport:
    """Subprocess wrapper for codex exec.

    Contracts
    ---------
    - Always passes ``-c service_tier="fast"`` to override the global config.
    - Always passes ``--skip-git-repo-check`` (judge runs outside any git repo).
    - Captures stderr; raises TransportError on nonzero exit or timeout.
    - model_meta captures the CLI version; model identity is "codex-default"
      because we do not parse stderr for the model line.
    """

    timeout: int = _SUBPROCESS_TIMEOUT
    _cli_version: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        self._cli_version = _get_codex_cli_version()

    def run(self, prompt: str) -> str:
        """Run the prompt through codex exec and return the model response text.

        Raises
        ------
        TransportError
            On nonzero exit code or subprocess timeout.
        """
        cmd = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-c",
            'service_tier="fast"',
            prompt,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TransportError(
                f"codex exec timed out after {self.timeout}s"
            ) from exc

        if result.returncode != 0:
            raise TransportError(
                f"codex exec exited with code {result.returncode}; "
                f"stderr: {result.stderr[:500]!r}"
            )

        return result.stdout.strip()

    @property
    def model_meta(self) -> dict:
        return {"model": "codex-default", "cli_version": self._cli_version}


class TransportError(RuntimeError):
    """Raised when the subprocess transport fails (timeout or nonzero exit)."""


def _get_codex_cli_version() -> str:
    """Return ``codex --version`` output, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or result.stderr.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Parse + validate
# ---------------------------------------------------------------------------


def _parse_verdict(raw: str, dimension: str, model_meta: dict) -> Verdict:
    """Parse raw model output into a Verdict.

    Raises ParseError on any JSON/schema problem.
    Raises InconsistentVerdictError on consistency matrix violations.
    """
    # Strip markdown code fences if the model wrapped the output.
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first and last fence line.
        inner = [ln for ln in lines if not ln.startswith("```")]
        text = "\n".join(inner).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"JSON decode failed: {exc}") from exc

    if not isinstance(data, dict):
        raise ParseError(f"Expected JSON object, got {type(data).__name__}")

    # Required fields.
    for field_name in ("dimension", "score", "passing", "issues", "rationale"):
        if field_name not in data:
            raise ParseError(f"Missing required field: {field_name!r}")

    score = data["score"]
    passing = data["passing"]
    issues = data["issues"]

    if not isinstance(score, int) or isinstance(score, bool):
        raise ParseError(f"'score' must be an integer, got {type(score).__name__}")
    if score < 1 or score > 5:
        raise ParseError(f"'score' must be 1–5, got {score}")
    if not isinstance(passing, bool):
        raise ParseError(f"'passing' must be a boolean, got {type(passing).__name__}")
    if not isinstance(issues, list):
        raise ParseError(f"'issues' must be a list, got {type(issues).__name__}")
    if not isinstance(data["rationale"], str):
        raise ParseError(
            f"'rationale' must be a string, got {type(data['rationale']).__name__}"
        )

    issues_list: list[str] = [str(i) for i in issues]

    # --- Consistency matrix ---
    _check_consistency(score, passing, issues_list)

    return Verdict(
        dimension=dimension,
        score=score,
        passing=passing,
        issues=issues_list,
        rationale=data["rationale"],
        model_meta=model_meta,
    )


def _check_consistency(score: int, passing: bool, issues: list[str]) -> None:
    """Apply the consistency matrix rules. Raises InconsistentVerdictError on violation."""
    # Rule 1: score >= 4 → passing must be True
    if score >= 4 and not passing:
        raise InconsistentVerdictError(
            f"score={score} (>=4) requires passing=true, but passing=false"
        )
    # Rule 2: score <= 2 → passing must be False AND issues must be non-empty
    if score <= 2 and passing:
        raise InconsistentVerdictError(
            f"score={score} (<=2) requires passing=false, but passing=true"
        )
    if score <= 2 and not issues:
        raise InconsistentVerdictError(
            f"score={score} (<=2) requires non-empty issues, but issues=[]"
        )
    # Rule 3: passing=True with non-empty issues → inconsistent
    if passing and issues:
        raise InconsistentVerdictError(
            f"passing=true but issues is non-empty: {issues!r}"
        )
    # Rule 4: passing=False with empty issues → inconsistent
    if not passing and not issues:
        raise InconsistentVerdictError(
            "passing=false but issues is empty"
        )


# ---------------------------------------------------------------------------
# Core judge_turn
# ---------------------------------------------------------------------------


def judge_turn(
    turn: dict,
    dimension: str,
    transport: CodexCliTransport | None = None,
) -> Verdict:
    """Judge one turn on one dimension.

    Parameters
    ----------
    turn:
        Dict with at least ``utterance`` (or ``tutor_utterance``), ``move``
        (or ``policy_move``), and optionally ``miscue`` (or ``miscue_context``).
    dimension:
        One of JUDGED_DIMENSIONS.  Raises ValueError for deterministic dims.
    transport:
        CodexCliTransport instance.  If None, a fresh default is created.

    Returns
    -------
    Verdict

    Raises
    ------
    ValueError
        If ``dimension`` is not in JUDGED_DIMENSIONS.
    JudgeError
        If all retries are exhausted.
    """
    if dimension in _DETERMINISTIC_DIMS:
        raise ValueError(
            f"Dimension {dimension!r} is deterministic and must not be judged "
            f"by the LLM judge (deterministic-dims doctrine). "
            f"Judged dimensions are: {JUDGED_DIMENSIONS}"
        )
    if dimension not in JUDGED_DIMENSIONS:
        raise ValueError(
            f"Unknown dimension {dimension!r}. "
            f"Judged dimensions are: {JUDGED_DIMENSIONS}. "
            f"Deterministic dimensions must not be passed to the judge."
        )

    if transport is None:
        transport = CodexCliTransport()

    base_prompt = _build_prompt(turn, dimension)

    # We track the current prompt (may be modified for retry).
    current_prompt = base_prompt
    last_raw: str = ""
    last_error_msg: str = ""
    error_class: str = ""
    total_attempts = 0

    for attempt in range(1 + _MAX_RETRIES):
        total_attempts = attempt + 1

        # --- Transport call ---
        try:
            raw = transport.run(current_prompt)
        except TransportError as exc:
            error_class = "transport_error"
            last_error_msg = str(exc)
            # For transport errors, retry with bare original prompt (no context).
            current_prompt = base_prompt
            continue

        last_raw = raw

        # --- Parse ---
        try:
            verdict = _parse_verdict(raw, dimension, transport.model_meta)
            return verdict
        except InconsistentVerdictError as exc:
            error_class = "inconsistent_verdict"
            last_error_msg = str(exc)
            current_prompt = _build_retry_inconsistency_prompt(
                base_prompt, raw, last_error_msg
            )
            continue
        except ParseError as exc:
            error_class = "parse_error"
            last_error_msg = str(exc)
            current_prompt = _build_retry_parse_prompt(
                base_prompt, raw, last_error_msg
            )
            continue

    # All attempts exhausted.
    detail = f"last error: {last_error_msg}"
    if last_raw and error_class != "transport_error":
        detail += f"; last raw output: {last_raw[:200]!r}"
    raise JudgeError(error_class=error_class, attempts=total_attempts, detail=detail)


# ---------------------------------------------------------------------------
# judge_trace
# ---------------------------------------------------------------------------


def judge_trace(
    trace_or_turns: dict | list[dict],
    dimensions: Sequence[str],
    transport: CodexCliTransport | None = None,
) -> list[Verdict]:
    """Judge all turns in a trace across all requested dimensions.

    Parameters
    ----------
    trace_or_turns:
        Either a trace dict with a ``"turns"`` key, or a bare list of turn dicts.
    dimensions:
        Sequence of dimension names (must all be in JUDGED_DIMENSIONS).
    transport:
        Transport to use; if None, a single shared CodexCliTransport is created.

    Returns
    -------
    list[Verdict]
        All verdicts, in turn-major order (all dims for turn[0], then turn[1], …).

    Raises
    ------
    JudgeError
        If any (turn, dimension) pair exhausts retries.  Partial results are
        NEVER returned — the exception propagates immediately.
    ValueError
        If any dimension is invalid.
    """
    if isinstance(trace_or_turns, dict):
        turns = trace_or_turns["turns"]
    else:
        turns = list(trace_or_turns)

    # Validate all dimensions up front (fail fast, not mid-run).
    for dim in dimensions:
        if dim in _DETERMINISTIC_DIMS:
            raise ValueError(
                f"Dimension {dim!r} is deterministic and must not be judged "
                f"(deterministic-dims doctrine)."
            )
        if dim not in JUDGED_DIMENSIONS:
            raise ValueError(
                f"Unknown dimension {dim!r}. "
                f"Judged dimensions are: {JUDGED_DIMENSIONS}."
            )

    if transport is None:
        transport = CodexCliTransport()

    verdicts: list[Verdict] = []
    for turn in turns:
        for dim in dimensions:
            # Raises JudgeError on exhaustion — partial results never written.
            v = judge_turn(turn, dim, transport=transport)
            verdicts.append(v)

    return verdicts
