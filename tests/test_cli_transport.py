"""T4.3 — ClaudeCliTransport: mock-subprocess tests (CI-safe, no live calls).

Covers:
  * Happy path: canned ``claude`` JSON envelope -> utterance extracted.
  * Malformed outer JSON from the CLI -> RuntimeError raised.
  * Non-zero subprocess exit -> RuntimeError with stderr in message.
  * Subprocess timeout -> RuntimeError raised.
  * Timeout parameter is plumbed through to subprocess.run.
  * Prompt sent to the CLI contains the strict-JSON instruction.
  * Prompt sent to the CLI contains the system prompt content (versioned file).
  * transport_meta records transport/model.
  * cli_transport() factory returns a ClaudeCliTransport.
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from readcoach.llm_client import (
    CLI_MODEL_ID,
    ClaudeCliTransport,
    cli_transport,
)
from readcoach.tutor import TutorAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(move: str = "ENCOURAGE", **kw) -> TutorAction:
    base = dict(target_word=None, rationale="[R-TEST] test")
    base.update(kw)
    return TutorAction(move=move, **base)


def _make_envelope(result_text: str, returncode: int = 0) -> MagicMock:
    """Build a mock CompletedProcess with --output-format json envelope."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = json.dumps({
        "type": "result",
        "subtype": "success",
        "result": result_text,
    })
    proc.stderr = ""
    return proc


def _run_verbalize(transport, action=None, **kw):
    if action is None:
        action = _action("ENCOURAGE")
    return transport.verbalize(action, {"page": "1"}, "1.0", **kw)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_cli_happy_path_extracts_text():
    """Canned well-formed model output -> utterance returned."""
    expected = "Nice work on that page!"
    proc = _make_envelope(json.dumps({"text": expected}))

    with patch("subprocess.run", return_value=proc) as mock_run:
        transport = ClaudeCliTransport(timeout_s=60)
        result = _run_verbalize(transport)

    assert result == expected
    mock_run.assert_called_once()


def test_cli_command_uses_pinned_model():
    """The subprocess call must use CLI_MODEL_ID and --output-format json."""
    proc = _make_envelope(json.dumps({"text": "ok"}))

    with patch("subprocess.run", return_value=proc) as mock_run:
        transport = ClaudeCliTransport()
        _run_verbalize(transport)

    cmd = mock_run.call_args[0][0]  # first positional arg is the cmd list
    assert "claude" in cmd[0]
    assert "--model" in cmd
    model_idx = cmd.index("--model")
    assert cmd[model_idx + 1] == CLI_MODEL_ID
    assert "--output-format" in cmd
    fmt_idx = cmd.index("--output-format")
    assert cmd[fmt_idx + 1] == "json"


def test_cli_timeout_parameter_passed_to_subprocess():
    """The timeout_s parameter must be forwarded to subprocess.run."""
    proc = _make_envelope(json.dumps({"text": "ok"}))

    with patch("subprocess.run", return_value=proc) as mock_run:
        transport = ClaudeCliTransport(timeout_s=42.5)
        _run_verbalize(transport)

    kwargs = mock_run.call_args[1]  # keyword args
    assert kwargs.get("timeout") == 42.5


# ---------------------------------------------------------------------------
# Prompt content checks
# ---------------------------------------------------------------------------

def test_cli_prompt_contains_strict_json_instruction():
    """The -p prompt must include the strict-JSON output instruction."""
    proc = _make_envelope(json.dumps({"text": "ok"}))

    with patch("subprocess.run", return_value=proc) as mock_run:
        transport = ClaudeCliTransport()
        _run_verbalize(transport)

    cmd = mock_run.call_args[0][0]
    # -p is followed by the full prompt string
    p_idx = cmd.index("-p")
    full_prompt = cmd[p_idx + 1]
    assert '{"text":' in full_prompt or '"text"' in full_prompt
    assert "JSON" in full_prompt


def test_cli_prompt_contains_system_prompt_content():
    """The -p prompt must embed the versioned system prompt (ReadCoach appears in 1.0)."""
    proc = _make_envelope(json.dumps({"text": "ok"}))

    with patch("subprocess.run", return_value=proc) as mock_run:
        transport = ClaudeCliTransport()
        _run_verbalize(transport)

    cmd = mock_run.call_args[0][0]
    p_idx = cmd.index("-p")
    full_prompt = cmd[p_idx + 1]
    # The versioned prompt file (1.0.md) contains "ReadCoach"
    assert "ReadCoach" in full_prompt


def test_cli_prompt_contains_move():
    """The user portion of the prompt must include the action move."""
    proc = _make_envelope(json.dumps({"text": "ok"}))

    with patch("subprocess.run", return_value=proc) as mock_run:
        transport = ClaudeCliTransport()
        _run_verbalize(transport, action=_action("SCAFFOLDED_HINT", hint_level="bounce"))

    cmd = mock_run.call_args[0][0]
    p_idx = cmd.index("-p")
    full_prompt = cmd[p_idx + 1]
    assert "SCAFFOLDED_HINT" in full_prompt


def test_cli_ai_reminder_flag_reaches_prompt():
    """is_ai_reminder=True must surface in the prompt text."""
    proc = _make_envelope(json.dumps({"text": "I'm your computer buddy — great!"}))

    with patch("subprocess.run", return_value=proc) as mock_run:
        transport = ClaudeCliTransport()
        _run_verbalize(transport, is_ai_reminder=True)

    cmd = mock_run.call_args[0][0]
    p_idx = cmd.index("-p")
    full_prompt = cmd[p_idx + 1]
    assert "AI reminder" in full_prompt or "ai reminder" in full_prompt.lower()


# ---------------------------------------------------------------------------
# Fail-loud: malformed outer JSON
# ---------------------------------------------------------------------------

def test_cli_malformed_outer_json_raises():
    """If the CLI returns non-JSON stdout, RuntimeError is raised."""
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = "this is not json at all"
    proc.stderr = ""

    with patch("subprocess.run", return_value=proc):
        transport = ClaudeCliTransport()
        with pytest.raises(RuntimeError, match="parse outer JSON"):
            _run_verbalize(transport)


# ---------------------------------------------------------------------------
# Fail-loud: malformed inner JSON (model disobeyed strict-JSON instruction)
# ---------------------------------------------------------------------------

def test_cli_model_returns_prose_raises():
    """If the model returns prose instead of JSON, RuntimeError is raised."""
    proc = _make_envelope("Sure, let me help you with that word!")

    with patch("subprocess.run", return_value=proc):
        transport = ClaudeCliTransport()
        with pytest.raises(RuntimeError, match="did not return strict JSON"):
            _run_verbalize(transport)


def test_cli_missing_text_key_raises():
    """If the model JSON has no 'text' key, RuntimeError is raised."""
    proc = _make_envelope(json.dumps({"response": "oops wrong key"}))

    with patch("subprocess.run", return_value=proc):
        transport = ClaudeCliTransport()
        with pytest.raises(RuntimeError, match="missing 'text' key"):
            _run_verbalize(transport)


# ---------------------------------------------------------------------------
# Fail-loud: non-zero exit
# ---------------------------------------------------------------------------

def test_cli_nonzero_exit_raises_with_stderr():
    """Non-zero subprocess exit -> RuntimeError containing stderr."""
    proc = MagicMock()
    proc.returncode = 1
    proc.stdout = ""
    proc.stderr = "authentication error: token invalid"

    with patch("subprocess.run", return_value=proc):
        transport = ClaudeCliTransport()
        with pytest.raises(RuntimeError, match="authentication error"):
            _run_verbalize(transport)


def test_cli_nonzero_exit_message_includes_exit_code():
    """RuntimeError for non-zero exit must mention the exit code."""
    proc = MagicMock()
    proc.returncode = 2
    proc.stdout = ""
    proc.stderr = "some error"

    with patch("subprocess.run", return_value=proc):
        transport = ClaudeCliTransport()
        with pytest.raises(RuntimeError, match="2"):
            _run_verbalize(transport)


# ---------------------------------------------------------------------------
# Fail-loud: subprocess timeout
# ---------------------------------------------------------------------------

def test_cli_timeout_raises_runtime_error():
    """subprocess.TimeoutExpired must be re-raised as RuntimeError."""
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=120)):
        transport = ClaudeCliTransport(timeout_s=120)
        with pytest.raises(RuntimeError, match="timed out"):
            _run_verbalize(transport)


# ---------------------------------------------------------------------------
# transport_meta / factory
# ---------------------------------------------------------------------------

def test_cli_transport_meta():
    """ClaudeCliTransport.transport_meta must record transport and model."""
    meta = ClaudeCliTransport.transport_meta
    assert meta["transport"] == "claude-cli"
    assert meta["model"] == CLI_MODEL_ID


def test_cli_transport_factory():
    """cli_transport() factory must return a ClaudeCliTransport."""
    t = cli_transport()
    assert isinstance(t, ClaudeCliTransport)


def test_cli_transport_factory_custom_timeout():
    """cli_transport(timeout_s=...) must plumb through to the instance."""
    t = cli_transport(timeout_s=30)
    assert t._timeout_s == 30  # noqa: SLF001
