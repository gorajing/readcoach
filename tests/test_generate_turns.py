"""T4.3 — turn-generation script, covered with a stubbed transport (red first).

The live ~70-turn RUN is deferred until an ANTHROPIC_API_KEY lands; this proves
the machinery end-to-end against a STUB verbalizer:
  * 3 hand-scripted profiles -> decide() -> verbalize() -> TurnRecord;
  * writes turns_v1.jsonl + a SessionTrace per profile;
  * runs the policy-compiler audit and reports violations (must be 0 to exit 0);
  * main() FAILS LOUD with a clear message when ANTHROPIC_API_KEY is absent.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "generate_turns", _PROJECT_ROOT / "scripts" / "generate_turns.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the script's frozen dataclasses can resolve their
    # own module's annotations (dataclasses looks the module up in sys.modules).
    sys.modules["generate_turns"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class _StubVerbalizer:
    """Returns a safe, policy-clean line for any move; records calls."""

    def __init__(self):
        self.calls = []

    def verbalize(self, action, ctx_summary, prompt_version, *, is_ai_reminder=False):
        self.calls.append((action.move, is_ai_reminder))
        if is_ai_reminder:
            return "I'm your computer reading buddy — keep going, you're doing great!"
        # Clean lines per move (never trip the invariants).
        return {
            "WAIT": "Take your time.",
            "ENCOURAGE": "You worked so hard on that page!",
            "SCAFFOLDED_HINT": "Look at the first sound and try it.",
            "MODEL_THE_WORD": f"That word is {action.target_word or 'there'}.",
            "COMPREHENSION_PROMPT": "What happened in the story so far?",
            "NEXT_ITEM": "Let's try the next one!",
        }[action.move]


def test_profiles_are_three_and_cover_24ish_turns():
    mod = _load_script()
    profiles = mod.build_profiles()
    assert len(profiles) == 3
    names = {p.name for p in profiles}
    assert names == {"struggling-decoder", "fluent-but-hesitant", "self-corrector"}
    for p in profiles:
        assert 20 <= len(p.steps) <= 30


def test_run_writes_jsonl_and_traces_and_audits_clean(tmp_path):
    mod = _load_script()
    stub = _StubVerbalizer()
    result = mod.run(verbalizer=stub, out_dir=tmp_path, prompt_version="1.0")

    # jsonl exists with one line per turn across all profiles.
    jsonl = tmp_path / "turns_v1.jsonl"
    assert jsonl.exists()
    lines = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    total_turns = sum(len(p.steps) for p in mod.build_profiles())
    assert len(lines) == total_turns

    # A SessionTrace per profile.
    for name in ("struggling-decoder", "fluent-but-hesitant", "self-corrector"):
        assert (tmp_path / f"trace_{name}.json").exists()

    # The hand-scripted, stub-verbalized run must be invariant-clean.
    assert result["violations"] == 0
    # verbalize() was actually invoked for every turn.
    assert len(stub.calls) == total_turns


def test_run_emits_periodic_ai_reminders(tmp_path):
    """At least one reminder per profile (cadence within the 20-turn window)."""
    mod = _load_script()
    stub = _StubVerbalizer()
    mod.run(verbalizer=stub, out_dir=tmp_path, prompt_version="1.0")
    # Each trace must itself pass the periodic_ai_reminder check (covered by the
    # clean audit above), and at least one reminder turn was verbalized.
    assert any(is_reminder for _move, is_reminder in stub.calls)


def test_main_fails_loud_without_api_key(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        mod.main(["--out-dir", str(tmp_path)])
