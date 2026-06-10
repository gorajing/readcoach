"""T4.3 — utterance layer: forced-tool-use Anthropic client (red first).

The LLM VERBALIZES; the policy DECIDES.  This client takes a chosen TutorAction +
context and returns the one line to say to the child, via a FORCED single-tool
call (`say_to_child`) so the output is structured.  Transport is injected so the
suite stubs it; a real call with no ANTHROPIC_API_KEY raises loudly.

Tests assert the request shape (forced tool choice, model pinned, system prompt
from the versioned file, timeout present), truncation/stop-reason raises, the
AI-reminder requirement reaches the model, and the no-key RuntimeError.
"""
from __future__ import annotations

import pytest

from readcoach.llm_client import (
    MODEL_ID,
    SAY_TOOL_NAME,
    TutorVerbalizer,
    load_prompt,
)
from readcoach.tutor import TutorAction


# ---------------------------------------------------------------------------
# Stub transport — records the kwargs of the single messages.create call.
# ---------------------------------------------------------------------------

class _Block:
    def __init__(self, type_, **kw):
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class _Resp:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class _StubMessages:
    def __init__(self, resp):
        self._resp = resp
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


class _StubClient:
    def __init__(self, resp):
        self.messages = _StubMessages(resp)


def _tool_use_resp(text: str) -> _Resp:
    return _Resp(
        stop_reason="tool_use",
        content=[_Block("tool_use", name=SAY_TOOL_NAME, input={"text": text})],
    )


def _action(move="ENCOURAGE", **kw) -> TutorAction:
    base = dict(target_word=None, rationale="[R-TEST] test")
    base.update(kw)
    return TutorAction(move=move, **base)


# ---------------------------------------------------------------------------
# load_prompt — versioned file
# ---------------------------------------------------------------------------

def test_load_prompt_reads_versioned_file():
    text = load_prompt("1.0")
    assert "ReadCoach" in text
    assert "say_to_child" in text


def test_load_prompt_unknown_version_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("9.9")


# ---------------------------------------------------------------------------
# Happy path + request shape
# ---------------------------------------------------------------------------

def test_verbalize_returns_tool_text():
    client = _StubClient(_tool_use_resp("Nice work on that page!"))
    v = TutorVerbalizer(client_factory=lambda: client)
    out = v.verbalize(_action("ENCOURAGE"), {"page": "done"}, "1.0")
    assert out == "Nice work on that page!"


def test_request_forces_say_tool_and_pins_model():
    client = _StubClient(_tool_use_resp("ok"))
    v = TutorVerbalizer(client_factory=lambda: client)
    v.verbalize(_action("WAIT"), {}, "1.0")
    (kwargs,) = client.messages.calls
    assert kwargs["model"] == MODEL_ID == "claude-opus-4-8"
    assert kwargs["tool_choice"] == {"type": "tool", "name": SAY_TOOL_NAME}
    tool_names = [t["name"] for t in kwargs["tools"]]
    assert tool_names == [SAY_TOOL_NAME]
    # System prompt is the versioned file.
    assert "ReadCoach" in kwargs["system"]


def test_request_carries_a_timeout():
    """An explicit timeout must be set on the call (not the SDK default)."""
    client = _StubClient(_tool_use_resp("ok"))
    v = TutorVerbalizer(client_factory=lambda: client, timeout_s=12.5)
    v.verbalize(_action("WAIT"), {}, "1.0")
    (kwargs,) = client.messages.calls
    assert kwargs["timeout"] == 12.5


def test_ai_reminder_requirement_reaches_the_model():
    client = _StubClient(_tool_use_resp("I'm your computer buddy — nice reading!"))
    v = TutorVerbalizer(client_factory=lambda: client)
    v.verbalize(_action("ENCOURAGE"), {}, "1.0", is_ai_reminder=True)
    (kwargs,) = client.messages.calls
    user_text = _user_text(kwargs)
    assert "AI reminder" in user_text or "ai reminder" in user_text.lower()


def test_no_ai_reminder_by_default():
    client = _StubClient(_tool_use_resp("ok"))
    v = TutorVerbalizer(client_factory=lambda: client)
    v.verbalize(_action("WAIT"), {}, "1.0")
    (kwargs,) = client.messages.calls
    assert "reminder turn: yes" not in _user_text(kwargs).lower()


def _user_text(kwargs: dict) -> str:
    """Flatten the user message content into one string."""
    msgs = kwargs["messages"]
    parts: list[str] = []
    for m in msgs:
        c = m["content"]
        if isinstance(c, str):
            parts.append(c)
        else:
            for blk in c:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(blk["text"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Fail-loud: truncation / wrong stop reason / missing tool block
# ---------------------------------------------------------------------------

def test_truncated_response_raises():
    client = _StubClient(_Resp(stop_reason="max_tokens", content=[]))
    v = TutorVerbalizer(client_factory=lambda: client)
    with pytest.raises(RuntimeError, match="max_tokens"):
        v.verbalize(_action("WAIT"), {}, "1.0")


def test_refusal_stop_reason_raises():
    client = _StubClient(_Resp(stop_reason="refusal", content=[]))
    v = TutorVerbalizer(client_factory=lambda: client)
    with pytest.raises(RuntimeError, match="refusal"):
        v.verbalize(_action("WAIT"), {}, "1.0")


def test_missing_tool_block_raises():
    # stop_reason ok but no say_to_child block present.
    client = _StubClient(_Resp(stop_reason="end_turn", content=[_Block("text", text="hi")]))
    v = TutorVerbalizer(client_factory=lambda: client)
    with pytest.raises(RuntimeError, match="say_to_child"):
        v.verbalize(_action("WAIT"), {}, "1.0")


# ---------------------------------------------------------------------------
# No-key loud failure (default client factory).
# ---------------------------------------------------------------------------

def test_missing_api_key_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    v = TutorVerbalizer()  # default factory — should demand a key at call time
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        v.verbalize(_action("WAIT"), {}, "1.0")
