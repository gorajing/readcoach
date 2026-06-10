"""Utterance layer — the LLM VERBALIZES, the policy DECIDES (T4.3).

``readcoach.tutor.decide`` chooses a discrete pedagogical move; this module turns
that move into ONE warm, age-appropriate line said to the child.  The model is
boxed in so it cannot free-form a different action:

  * **Forced single tool (SDK / api transport).**  The request pins
    ``tool_choice`` to one tool, ``say_to_child(text)`` — the model's only legal
    output is a structured utterance.  No prose channel, no second move.
  * **Strict-JSON prompt (CLI transport).**  ``ClaudeCliTransport`` shells out via
    ``claude -p ... --output-format json`` and instructs the model to emit
    ``{"text": "..."}`` as its *entire* response.  The transport parses that field
    fail-loud: a JSON parse failure or a missing ``text`` key raises
    ``RuntimeError``; it **never** falls back to returning raw text.  This is a
    semantic difference from the SDK transport: the SDK *forces* structured output
    via tool-use; the CLI transport *instructs* it.  A misbehaving model can still
    return prose on the CLI path — the parse error will surface it rather than
    silently embedding prose in a turn record.
  * **Versioned prompt.**  The system prompt is the file ``prompts/tutor/<v>.md``
    (role + the move it must verbalize + the hard constraints mirrored from the
    policies).  Bumping the prompt is a tracked file change.
  * **Pinned model + explicit timeout + truncation check.**  The model id is a
    constant; an explicit timeout is passed per call; a response whose
    ``stop_reason`` is not ``tool_use``/``end_turn`` (e.g. ``max_tokens`` /
    ``refusal``) raises rather than returning a half-line.
  * **Injected transport.**  ``client_factory`` is injected so tests stub it; the
    default factory builds an ``anthropic.Anthropic`` client and FAILS LOUD with a
    clear message if ``ANTHROPIC_API_KEY`` is absent at real-call time.

Mirrors the gtm-ops-router claude-client shape (forced tool-use, explicit
timeout, truncation check) in Python via the official Anthropic SDK.

Transport inventory
-------------------
``TutorVerbalizer``          — SDK transport (forced tool-use; requires ANTHROPIC_API_KEY).
``ClaudeCliTransport``       — CLI transport (strict-JSON prompt; uses ``claude -p``
                               subscription binary; no API key needed).
``cli_transport()``          — factory returning a ``ClaudeCliTransport`` with defaults.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from readcoach.tutor import TutorAction

# Pinned model id (claude-api skill, 2026-06: default to Opus 4.8).  A model bump
# is a tracked code change, like the prompt version.
MODEL_ID = "claude-opus-4-8"

# Model pinned for the CLI / subscription transport.
CLI_MODEL_ID = "claude-sonnet-4-6"

# The single tool the model is forced to call — its only output channel.
SAY_TOOL_NAME = "say_to_child"

# Caps the verbalized line; a child-facing utterance is one or two short
# sentences, so this is generous headroom (and bounds truncation risk).
_MAX_TOKENS = 256

_DEFAULT_TIMEOUT_S = 30.0

# CLI transport uses a longer per-call timeout (subprocess spin-up + network).
_CLI_DEFAULT_TIMEOUT_S = 120.0

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "tutor"

# Stop reasons that mean "we have a complete response to read".  Anything else
# (max_tokens, refusal, pause_turn, ...) is a failure for this single-shot call.
_OK_STOP_REASONS = frozenset({"tool_use", "end_turn"})

# Suffix appended to the user message on the CLI path to instruct strict-JSON output.
_CLI_JSON_INSTRUCTION = (
    "\n\n"
    "IMPORTANT — OUTPUT FORMAT: You MUST respond with ONLY a single JSON object "
    "on one line, exactly like this: {\"text\": \"<your one line>\"}\n"
    "No markdown, no code fences, no extra text, no explanation — ONLY the JSON object."
)


def load_prompt(version: str) -> str:
    """Load the versioned tutor system prompt ``prompts/tutor/<version>.md``.

    Fail-loud: an unknown version raises ``FileNotFoundError``.
    """
    path = _PROMPTS_DIR / f"{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"tutor prompt version {version!r} not found at {path}")
    return path.read_text(encoding="utf-8")


_SAY_TOOL = {
    "name": SAY_TOOL_NAME,
    "description": (
        "Say exactly one short, warm, age-appropriate line out loud to the child, "
        "verbalizing the move you were given. Call this tool once with that line."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The one line to say to the child.",
            }
        },
        "required": ["text"],
        "additionalProperties": False,
    },
}


class ClaudeCliTransport:
    """Verbalize a ``TutorAction`` via the ``claude -p`` subscription CLI.

    Semantic difference vs. the SDK transport
    ------------------------------------------
    The SDK ``TutorVerbalizer`` forces structured output through tool-use: the
    model's *only* legal output channel is the ``say_to_child`` tool, making
    free-form prose structurally impossible.

    This transport *cannot* use tool forcing — the ``claude`` CLI has no
    ``--tool-choice`` flag.  Instead the user message is extended with a strict
    instruction to respond with ONLY ``{"text": "..."}`` JSON.  A compliant model
    will obey; a misbehaving model might return prose.  The transport parses
    fail-loud: any non-parseable or ``text``-less response raises ``RuntimeError``
    rather than silently returning raw text.  This surfaces the failure rather
    than embedding malformed data in the turn record.

    Metadata
    --------
    ``transport_meta`` is a ``dict`` recording ``transport`` and ``model``; it is
    attached per-call to enable provenance in the turn records / report.

    Parameters
    ----------
    timeout_s:
        Per-call subprocess timeout in seconds.  Subprocess timeout raises
        ``RuntimeError`` (not ``subprocess.TimeoutExpired``) so callers see a
        uniform error type.
    """

    transport_meta: dict = {
        "transport": "claude-cli",
        "model": CLI_MODEL_ID,
    }

    def __init__(self, *, timeout_s: float = _CLI_DEFAULT_TIMEOUT_S) -> None:
        self._timeout_s = timeout_s

    def verbalize(
        self,
        action: TutorAction,
        ctx_summary: dict,
        prompt_version: str,
        *,
        is_ai_reminder: bool = False,
    ) -> str:
        """Return the one line to say for ``action`` via the ``claude`` CLI.

        Builds the same user message as ``TutorVerbalizer._render_user_message``
        and prepends the versioned system prompt inline (the CLI ``-p`` flag
        takes a single prompt; we merge system + user).  Appends the strict-JSON
        output instruction.

        Raises
        ------
        RuntimeError
            Non-zero exit code, subprocess timeout, unparseable JSON, or missing
            ``text`` key in the response.
        """
        system = load_prompt(prompt_version)
        user_text = _render_user_message(action, ctx_summary, is_ai_reminder)

        # Merge system prompt + user message for the -p flag (CLI has no
        # separate system-prompt flag in basic usage); append strict-JSON instruction.
        full_prompt = (
            f"{system}\n\n---\n\n{user_text}{_CLI_JSON_INSTRUCTION}"
        )

        cmd = [
            "claude",
            "-p", full_prompt,
            "--model", CLI_MODEL_ID,
            "--output-format", "json",
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ClaudeCliTransport: subprocess timed out after {self._timeout_s}s "
                f"for move={action.move!r}"
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"ClaudeCliTransport: claude exited {proc.returncode} for "
                f"move={action.move!r}; stderr={proc.stderr!r}"
            )

        # Parse the outer JSON envelope from --output-format json.
        raw_output = proc.stdout.strip()
        try:
            envelope = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"ClaudeCliTransport: could not parse outer JSON envelope; "
                f"raw={raw_output[:200]!r}"
            ) from exc

        # Extract the model's text from the envelope's ``result`` field.
        result_text = envelope.get("result", "")
        if not result_text:
            raise RuntimeError(
                f"ClaudeCliTransport: empty or missing 'result' in envelope; "
                f"envelope keys={list(envelope)}"
            )

        # The model's response should be ONLY {"text": "..."} — parse it.
        result_text = result_text.strip()
        # Strip markdown code fences if the model disobeyed the format instruction.
        if result_text.startswith("```"):
            lines = result_text.splitlines()
            inner = [ln for ln in lines if not ln.startswith("```")]
            result_text = "\n".join(inner).strip()

        try:
            payload = json.loads(result_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"ClaudeCliTransport: model did not return strict JSON; "
                f"result={result_text[:200]!r} — "
                f"check prompt for move={action.move!r}"
            ) from exc

        text = payload.get("text") if isinstance(payload, dict) else None
        if not text or not str(text).strip():
            raise RuntimeError(
                f"ClaudeCliTransport: parsed JSON missing 'text' key or empty; "
                f"payload={payload!r}"
            )
        return str(text).strip()


def cli_transport(*, timeout_s: float = _CLI_DEFAULT_TIMEOUT_S) -> ClaudeCliTransport:
    """Factory: return a ``ClaudeCliTransport`` with the given timeout."""
    return ClaudeCliTransport(timeout_s=timeout_s)


def _default_client_factory():  # noqa: ANN202
    """Build a real Anthropic client; FAIL LOUD if the key is missing.

    The suite never reaches here (transport is stubbed); a real verbalize() call
    on a machine with no ``ANTHROPIC_API_KEY`` raises a clear RuntimeError rather
    than letting the SDK surface an opaque auth error later.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set — the utterance layer requires a live "
            "Anthropic key to verbalize a move. Set ANTHROPIC_API_KEY (the eval "
            "scoring path needs no key, but verbalization does)."
        )
    import anthropic  # local import: tests never construct a real client

    return anthropic.Anthropic()


class TutorVerbalizer:
    """Turns a chosen ``TutorAction`` into one line said to the child.

    ``client_factory`` returns an object exposing ``messages.create(**kwargs)``
    (the Anthropic SDK shape).  Injected so tests stub it; the client is built
    lazily on first use so constructing a ``TutorVerbalizer`` never needs a key.
    """

    def __init__(
        self,
        client_factory: Callable[[], Any] | None = None,
        *,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._client_factory = client_factory or _default_client_factory
        self._timeout_s = timeout_s
        self._client: Any | None = None

    def _client_or_build(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def verbalize(
        self,
        action: TutorAction,
        ctx_summary: dict,
        prompt_version: str,
        *,
        is_ai_reminder: bool = False,
    ) -> str:
        """Return the one line to say for ``action``.

        Forces the ``say_to_child`` tool, pins the model, passes an explicit
        timeout, and raises on any non-terminal stop reason or a missing tool
        block (no silent half-line).
        """
        system = load_prompt(prompt_version)
        user_text = _render_user_message(action, ctx_summary, is_ai_reminder)

        client = self._client_or_build()
        resp = client.messages.create(
            model=MODEL_ID,
            max_tokens=_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_text}],
            tools=[_SAY_TOOL],
            tool_choice={"type": "tool", "name": SAY_TOOL_NAME},
            timeout=self._timeout_s,
        )

        stop_reason = getattr(resp, "stop_reason", None)
        if stop_reason not in _OK_STOP_REASONS:
            raise RuntimeError(
                f"verbalize: response did not complete cleanly "
                f"(stop_reason={stop_reason!r}); expected one of "
                f"{sorted(_OK_STOP_REASONS)} — refusing to emit a partial line"
            )

        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == SAY_TOOL_NAME:
                text = block.input.get("text") if isinstance(block.input, dict) else None
                if not text or not str(text).strip():
                    raise RuntimeError(
                        f"verbalize: {SAY_TOOL_NAME} returned no 'text'"
                    )
                return str(text)

        raise RuntimeError(
            f"verbalize: no {SAY_TOOL_NAME} tool_use block in response "
            f"(stop_reason={stop_reason!r})"
        )


def _render_user_message(
    action: TutorAction,
    ctx_summary: dict,
    is_ai_reminder: bool,
) -> str:
    """Build the user turn: the chosen move + scaffold rung + context + reminder flag."""
    lines = [
        "Verbalize this already-decided move as one short line to the child.",
        f"move: {action.move}",
    ]
    if action.target_word:
        lines.append(f"target_word: {action.target_word}")
    if action.hint_level:
        lines.append(f"hint_level: {action.hint_level}")
    if action.error_type:
        lines.append(f"miscue_type: {action.error_type}")
    if ctx_summary:
        ctx = ", ".join(f"{k}={v}" for k, v in sorted(ctx_summary.items()))
        lines.append(f"context: {ctx}")
    if is_ai_reminder:
        lines.append(
            "AI reminder turn: yes — also include one short, friendly clause "
            "reminding the child you are a computer/AI helper, not a real person."
        )
    return "\n".join(lines)
