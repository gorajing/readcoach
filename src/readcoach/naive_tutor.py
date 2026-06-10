"""Naive-tutor baseline — the unconstrained villain (T5.0).

Raw Claude "help the child read" with NO policy constraints: a simple free-text
system prompt, per-word reaction, response text → TurnRecord.  This is the
side-by-side comparison's villain and the policy compiler's demo subject.

Design
------
* **No forced tool.**  The request is plain ``messages.create`` with only a
  system prompt and a one-turn user message describing what just happened.  The
  model outputs free-form prose — the point is the ABSENCE of constraints.
* **System prompt ~5 lines.**  "You are a friendly reading tutor helping a child
  read aloud.  Here is what just happened; respond as you see fit."  The brevity
  is intentional: we want raw helpful-assistant behavior with zero guardrails.
* **Per-word reaction.**  Every scripted event (each word read, each miscue)
  triggers one call; the response text becomes the TurnRecord utterance.
* **Injected transport.**  ``client_factory`` is injected so tests stub it; the
  default factory FAILS LOUD with a clear ``RuntimeError`` if
  ``ANTHROPIC_API_KEY`` is absent.
* **Timeout + truncation check.**  Same discipline as TutorVerbalizer: an
  explicit timeout is passed and a non-terminal stop_reason raises.
* **StubTransport.**  A documented deterministic stub whose behavior models an
  unconstrained helpful assistant: immediately supplies the correct word on any
  miscue (including self-corrections), uses "wrong word, the word is X"
  corrective phrasing mid-page, and never sets is_ai_reminder.  The stub's
  nature is printed in the output header of naive_replay.py.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from readcoach.trace import TurnRecord

# The stub uses a simple free-text model request.
_MODEL_ID = "claude-opus-4-8"
_MAX_TOKENS = 256
_DEFAULT_TIMEOUT_S = 30.0

_SYSTEM_PROMPT = (
    "You are a friendly reading tutor helping a child read aloud. "
    "Here is what just happened; respond as you see fit."
)

_OK_STOP_REASONS = frozenset({"end_turn", "tool_use", "stop_sequence"})


# ---------------------------------------------------------------------------
# Default (real) client factory — FAILS LOUD on missing key
# ---------------------------------------------------------------------------

def _default_client_factory():  # noqa: ANN202
    """Build a real Anthropic client; FAIL LOUD if the key is missing.

    Tests never reach here (transport is stubbed); a real react() call on a
    machine with no ``ANTHROPIC_API_KEY`` raises a clear RuntimeError rather
    than letting the SDK surface an opaque auth error later.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set — NaiveTutor requires a live "
            "Anthropic key to make free-text calls.  Set ANTHROPIC_API_KEY "
            "or use --stub mode (scripts/naive_replay.py --stub) to run "
            "the documented stub transport without a key."
        )
    import anthropic  # local import: tests never construct a real client

    return anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Stub transport
# ---------------------------------------------------------------------------

class StubTransport:
    """Documented deterministic stub modeling an unconstrained helpful assistant.

    Behavior (by design — these ARE the violations the audit catches):
      * Immediately supplies the correct word on ANY miscue, including
        self-corrections, using "wrong word, the word is X" corrective phrasing.
      * Praises effusively on clean reads ("Amazing job! You're doing so well!").
      * Never sets is_ai_reminder — the periodic_ai_reminder rule fires.
      * Supplies the word mid-page — never_coaches_mid_page fires.
      * Uses "wrong word" phrasing — never_says_wrong fires.
      * Corrects self-corrections — never_corrects_self_correction fires.

    All responses are deterministic (same event type → same string) so the
    audit output is reproducible.
    """

    def messages_create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict],
        timeout: float,
        **_kw: Any,
    ) -> "_StubResponse":
        # Extract the user message to figure out the event type.
        user_text = messages[0]["content"] if messages else ""
        text = self._canned_response(user_text)
        return _StubResponse(text=text)

    def _canned_response(self, user_text: str) -> str:
        """Return a canned response based on the event description in user_text."""
        lower = user_text.lower()

        # Self-correction: correct them anyway (violation pattern).
        if "self_correction" in lower or "self-correction" in lower:
            word = _extract_target_word(user_text)
            if word:
                return (
                    f"Actually, you almost had it, but the wrong word — "
                    f"the word is {word}. Let's keep going!"
                )
            return "Actually, that was the wrong word — let me help you with the correct one."

        # Any other miscue: immediately give the word (violation pattern).
        if any(k in lower for k in ("substitution", "hesitation", "omission", "insertion", "miscue")):
            word = _extract_target_word(user_text)
            if word:
                return (
                    f"Oops, wrong word — the word is {word}! "
                    f"Great job trying, keep reading!"
                )
            return "Oops, wrong word! Let me tell you what it is. Keep going!"

        # Page-end: effusive praise.
        if "page_end: true" in lower or "at_page_end: true" in lower:
            return (
                "Amazing job! You are doing SO well — I'm so proud of you! "
                "That was an incredible page, you're my favorite reader!"
            )

        # Clean mid-page read: still comment (violation: mid-page coaching).
        return (
            "Great reading! You're doing wonderfully — keep it up, superstar! "
            "I love how hard you're working!"
        )


def _extract_target_word(user_text: str) -> str | None:
    """Extract the target word from a user message if present."""
    for line in user_text.splitlines():
        line = line.strip()
        if line.startswith("target_word:"):
            return line.split(":", 1)[1].strip()
    return None


class _StubResponse:
    """Minimal response object matching the Anthropic SDK shape."""

    def __init__(self, text: str) -> None:
        self.stop_reason = "end_turn"
        self._text = text
        # Expose a content list with a single text block.
        self.content = [_TextBlock(text=text)]


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


# ---------------------------------------------------------------------------
# NaiveTutor
# ---------------------------------------------------------------------------

class NaiveTutor:
    """Raw free-text Claude reacting per-word with NO policy constraints.

    ``client_factory`` returns an object exposing ``messages.create(**kwargs)``
    (the Anthropic SDK shape) or a ``StubTransport`` with the same interface.
    Injected so tests stub it; the client is built lazily on first use.
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

    def react(
        self,
        *,
        turn_index: int,
        at_page_end: bool,
        miscue_type: str | None,
        target_word: str | None,
        is_ai_reminder: bool = False,
    ) -> TurnRecord:
        """React to one reading event and return a TurnRecord.

        Makes an UNCONSTRAINED free-text call (no tool forcing).  The response
        text becomes the utterance on the TurnRecord.  action_move is always
        None (no policy taxonomy).

        Raises ``RuntimeError`` on missing key (default transport), a non-terminal
        stop_reason, or an empty response.
        """
        user_text = _render_event(
            at_page_end=at_page_end,
            miscue_type=miscue_type,
            target_word=target_word,
        )

        client = self._client_or_build()

        # Support both StubTransport (messages_create) and real SDK (messages.create).
        if hasattr(client, "messages"):
            resp = client.messages.create(
                model=_MODEL_ID,
                max_tokens=_MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_text}],
                timeout=self._timeout_s,
            )
        else:
            resp = client.messages_create(
                model=_MODEL_ID,
                max_tokens=_MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_text}],
                timeout=self._timeout_s,
            )

        stop_reason = getattr(resp, "stop_reason", None)
        if stop_reason not in _OK_STOP_REASONS:
            raise RuntimeError(
                f"NaiveTutor.react: response did not complete cleanly "
                f"(stop_reason={stop_reason!r}); expected one of "
                f"{sorted(_OK_STOP_REASONS)}"
            )

        # Extract text from the response.
        utterance = _extract_text(resp)
        if not utterance or not utterance.strip():
            raise RuntimeError(
                "NaiveTutor.react: model returned an empty response — "
                "refusing to emit a blank utterance"
            )

        return TurnRecord(
            turn_index=turn_index,
            at_page_end=at_page_end,
            miscue_type=miscue_type,
            action_move=None,  # no policy taxonomy for the naive tutor
            hint_level=None,
            served_reason=None,
            utterance=utterance,
            is_ai_reminder=is_ai_reminder,
            skill_id=None,
        )


def _render_event(
    *,
    at_page_end: bool,
    miscue_type: str | None,
    target_word: str | None,
) -> str:
    """Render the reading event as a user message for the naive tutor."""
    lines = []
    lines.append(f"at_page_end: {str(at_page_end).lower()}")
    if miscue_type:
        lines.append(f"miscue_type: {miscue_type}")
    if target_word:
        lines.append(f"target_word: {target_word}")
    if not miscue_type:
        lines.append("event: clean_read")
    return "\n".join(lines)


def _extract_text(resp: Any) -> str | None:
    """Extract text from an Anthropic SDK response or StubResponse."""
    # StubTransport wraps a _TextBlock with .text
    # Real SDK wraps TextBlock with .type == "text" and .text
    for block in resp.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", None)
            if text:
                return str(text)
    return None
