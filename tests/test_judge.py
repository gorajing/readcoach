"""Tests for evals/judge.py — T5.1 cross-family judge.

CI-safe: all tests use mocked subprocess — no real codex calls.
No credentials, no network.

CONTRACT UNDER TEST
-------------------
JUDGED_DIMENSIONS       tuple of judged dim names
Verdict                 frozen dataclass: dimension, score, passing, issues, rationale, model_meta
JudgeError              raised when retries exhausted (error_class, attempts)
judge_turn(turn, dim, transport) -> Verdict
judge_trace(trace_or_turns, dims, transport) -> list[Verdict]
CodexCliTransport       subprocess wrapper (mocked in all tests below)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from evals.judge import (
    JUDGED_DIMENSIONS,
    CodexCliTransport,
    InconsistentVerdictError,
    JudgeError,
    TransportError,
    Verdict,
    _build_retry_inconsistency_prompt,
    _build_retry_parse_prompt,
    _check_consistency,
    judge_trace,
    judge_turn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_verdict_json(
    dimension: str = "guidance",
    score: int = 4,
    passing: bool = True,
    issues: list[str] | None = None,
    rationale: str = "Looks good.",
) -> str:
    """Build a valid verdict JSON string."""
    if issues is None:
        issues = []
    return json.dumps(
        {
            "dimension": dimension,
            "score": score,
            "passing": passing,
            "issues": issues,
            "rationale": rationale,
        }
    )


def _make_transport(responses: list[str]) -> CodexCliTransport:
    """Return a CodexCliTransport whose run() yields responses in order."""
    transport = MagicMock(spec=CodexCliTransport)
    transport.model_meta = {"model": "codex-default", "cli_version": "codex-cli 0.130.0"}
    transport.run.side_effect = responses
    return transport


def _sample_turn() -> dict:
    return {
        "move": "SCAFFOLDED_HINT",
        "utterance": "Nice try! Look at the first letter — can you sound it out?",
        "miscue": "substitution: 'dog' for 'dig'",
    }


# ---------------------------------------------------------------------------
# Verdict: frozen dataclass
# ---------------------------------------------------------------------------


class TestVerdictFrozen:
    def test_verdict_is_frozen(self):
        v = Verdict(
            dimension="guidance",
            score=4,
            passing=True,
            issues=[],
            rationale="Good",
            model_meta={},
        )
        with pytest.raises((AttributeError, TypeError)):
            v.score = 5  # type: ignore[misc]

    def test_verdict_fields(self):
        v = Verdict(
            dimension="guidance",
            score=3,
            passing=False,
            issues=["Too generic"],
            rationale="Flat",
            model_meta={"model": "test"},
        )
        assert v.dimension == "guidance"
        assert v.score == 3
        assert v.passing is False
        assert v.issues == ["Too generic"]


# ---------------------------------------------------------------------------
# JUDGED_DIMENSIONS constant
# ---------------------------------------------------------------------------


class TestJudgedDimensions:
    def test_contains_guidance(self):
        assert "guidance" in JUDGED_DIMENSIONS

    def test_contains_actionability(self):
        assert "actionability" in JUDGED_DIMENSIONS

    def test_contains_icap(self):
        assert "icap" in JUDGED_DIMENSIONS

    def test_exactly_three_dims(self):
        assert len(JUDGED_DIMENSIONS) == 3


# ---------------------------------------------------------------------------
# Deterministic-dim refusal
# ---------------------------------------------------------------------------


class TestDeterministicDimRefusal:
    def test_mistake_id_raises_value_error(self):
        t = _make_transport([])
        with pytest.raises(ValueError, match="deterministic"):
            judge_turn(_sample_turn(), "mistake_id", transport=t)

    def test_location_raises_value_error(self):
        t = _make_transport([])
        with pytest.raises(ValueError, match="deterministic"):
            judge_turn(_sample_turn(), "location", transport=t)

    def test_miscue_id_raises_value_error(self):
        t = _make_transport([])
        with pytest.raises(ValueError, match="deterministic"):
            judge_turn(_sample_turn(), "miscue_id", transport=t)

    def test_unknown_dim_raises_value_error(self):
        t = _make_transport([])
        with pytest.raises(ValueError):
            judge_turn(_sample_turn(), "totally_unknown_dim", transport=t)

    def test_no_transport_call_made_for_deterministic_dim(self):
        t = _make_transport([])
        with pytest.raises(ValueError):
            judge_turn(_sample_turn(), "mistake_id", transport=t)
        t.run.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path: strict JSON, single attempt
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_score_4_passing_true_no_issues(self):
        payload = _make_verdict_json("guidance", score=4, passing=True, issues=[])
        t = _make_transport([payload])
        v = judge_turn(_sample_turn(), "guidance", transport=t)
        assert isinstance(v, Verdict)
        assert v.score == 4
        assert v.passing is True
        assert v.issues == []

    def test_score_5_passing_true(self):
        payload = _make_verdict_json("actionability", score=5, passing=True, issues=[])
        t = _make_transport([payload])
        v = judge_turn(_sample_turn(), "actionability", transport=t)
        assert v.score == 5
        assert v.passing is True

    def test_score_1_passing_false_with_issues(self):
        payload = _make_verdict_json(
            "icap", score=1, passing=False, issues=["No engagement at all."]
        )
        t = _make_transport([payload])
        v = judge_turn(_sample_turn(), "icap", transport=t)
        assert v.score == 1
        assert v.passing is False
        assert len(v.issues) == 1

    def test_score_2_passing_false_with_issues(self):
        payload = _make_verdict_json(
            "guidance", score=2, passing=False, issues=["Too clinical."]
        )
        t = _make_transport([payload])
        v = judge_turn(_sample_turn(), "guidance", transport=t)
        assert v.score == 2
        assert v.passing is False

    def test_score_3_passing_false_with_issues(self):
        # score=3, passing=False is allowed if issues non-empty
        payload = _make_verdict_json(
            "guidance", score=3, passing=False, issues=["Flat but ok."]
        )
        t = _make_transport([payload])
        v = judge_turn(_sample_turn(), "guidance", transport=t)
        assert v.score == 3
        assert v.passing is False

    def test_score_3_passing_true_no_issues(self):
        # score=3, passing=True is allowed if no issues
        payload = _make_verdict_json("guidance", score=3, passing=True, issues=[])
        t = _make_transport([payload])
        v = judge_turn(_sample_turn(), "guidance", transport=t)
        assert v.score == 3
        assert v.passing is True

    def test_dimension_preserved_in_verdict(self):
        payload = _make_verdict_json("actionability", score=4, passing=True, issues=[])
        t = _make_transport([payload])
        v = judge_turn(_sample_turn(), "actionability", transport=t)
        assert v.dimension == "actionability"

    def test_model_meta_attached(self):
        payload = _make_verdict_json("guidance", score=4, passing=True)
        t = _make_transport([payload])
        v = judge_turn(_sample_turn(), "guidance", transport=t)
        assert "model" in v.model_meta
        assert "cli_version" in v.model_meta

    def test_markdown_fences_stripped(self):
        """If model wraps JSON in ``` fences, it should still parse."""
        inner = _make_verdict_json("guidance", score=4, passing=True)
        fenced = f"```json\n{inner}\n```"
        t = _make_transport([fenced])
        v = judge_turn(_sample_turn(), "guidance", transport=t)
        assert v.score == 4


# ---------------------------------------------------------------------------
# Parse retry: garbage then valid
# ---------------------------------------------------------------------------


class TestParseRetry:
    def test_garbage_then_valid_succeeds(self):
        """Mock returns garbage first, then valid JSON — should succeed on retry."""
        good = _make_verdict_json("guidance", score=4, passing=True)
        t = _make_transport(["NOT JSON AT ALL %%$#", good])
        v = judge_turn(_sample_turn(), "guidance", transport=t)
        assert v.score == 4
        assert t.run.call_count == 2

    def test_two_garbage_then_valid_succeeds(self):
        """Two garbage outputs, then valid — succeeds (total 3 attempts, max_retries=2)."""
        good = _make_verdict_json("guidance", score=5, passing=True)
        t = _make_transport(["bad1", "bad2", good])
        v = judge_turn(_sample_turn(), "guidance", transport=t)
        assert v.score == 5
        assert t.run.call_count == 3

    def test_retry_prompt_contains_error_feedback(self):
        """The retry prompt must contain the error feedback from the previous parse failure."""
        good = _make_verdict_json("guidance", score=4, passing=True)
        t = _make_transport(["NOT JSON", good])
        judge_turn(_sample_turn(), "guidance", transport=t)
        # Second call (index 1) should have a prompt that contains error info.
        second_prompt = t.run.call_args_list[1][0][0]
        assert "PREVIOUS ATTEMPT FAILED" in second_prompt or "failed" in second_prompt.lower()
        assert "NOT JSON" in second_prompt


# ---------------------------------------------------------------------------
# Parse exhaustion raises JudgeError
# ---------------------------------------------------------------------------


class TestParseExhaustion:
    def test_all_garbage_raises_judge_error(self):
        """If all 3 attempts (1 + 2 retries) return garbage, raises JudgeError."""
        t = _make_transport(["bad", "bad", "bad"])
        with pytest.raises(JudgeError) as exc_info:
            judge_turn(_sample_turn(), "guidance", transport=t)
        assert exc_info.value.error_class == "parse_error"
        assert exc_info.value.attempts == 3

    def test_judge_error_names_class_and_attempts(self):
        t = _make_transport(["x", "y", "z"])
        with pytest.raises(JudgeError) as exc_info:
            judge_turn(_sample_turn(), "guidance", transport=t)
        err = exc_info.value
        assert "parse_error" in str(err)
        assert "3" in str(err)

    def test_no_default_verdict_returned(self):
        """JudgeError must be raised, not a default verdict."""
        t = _make_transport(["", "", ""])
        with pytest.raises(JudgeError):
            judge_turn(_sample_turn(), "guidance", transport=t)


# ---------------------------------------------------------------------------
# Inconsistency detection — all 4 matrix cases
# ---------------------------------------------------------------------------


class TestConsistencyMatrix:
    def test_case1_score_gte4_passing_false_is_inconsistent(self):
        """score >= 4 with passing=false → InconsistentVerdictError."""
        with pytest.raises(InconsistentVerdictError):
            _check_consistency(score=4, passing=False, issues=[])

    def test_case1_score5_passing_false_is_inconsistent(self):
        """score=5 with passing=false → InconsistentVerdictError."""
        with pytest.raises(InconsistentVerdictError):
            _check_consistency(score=5, passing=False, issues=[])

    def test_case2_score_lte2_passing_true_is_inconsistent(self):
        """score <= 2 with passing=true → InconsistentVerdictError."""
        with pytest.raises(InconsistentVerdictError):
            _check_consistency(score=2, passing=True, issues=[])

    def test_case2_score1_passing_true_is_inconsistent(self):
        """score=1 with passing=true → InconsistentVerdictError."""
        with pytest.raises(InconsistentVerdictError):
            _check_consistency(score=1, passing=True, issues=[])

    def test_case3_passing_true_with_issues_is_inconsistent(self):
        """passing=true with non-empty issues → InconsistentVerdictError."""
        with pytest.raises(InconsistentVerdictError):
            _check_consistency(score=4, passing=True, issues=["Some issue"])

    def test_case4_passing_false_with_empty_issues_is_inconsistent(self):
        """passing=false with empty issues → InconsistentVerdictError."""
        with pytest.raises(InconsistentVerdictError):
            _check_consistency(score=3, passing=False, issues=[])

    def test_valid_score2_passing_false_issues_nonempty(self):
        """score=2, passing=False, issues non-empty → valid."""
        _check_consistency(score=2, passing=False, issues=["Clinical tone"])

    def test_valid_score4_passing_true_no_issues(self):
        """score=4, passing=True, issues empty → valid."""
        _check_consistency(score=4, passing=True, issues=[])

    def test_valid_score3_passing_true_no_issues(self):
        """score=3, passing=True, issues empty → valid (score=3 is neutral)."""
        _check_consistency(score=3, passing=True, issues=[])

    def test_valid_score3_passing_false_issues_nonempty(self):
        """score=3, passing=False, issues non-empty → valid."""
        _check_consistency(score=3, passing=False, issues=["Flat"])


# ---------------------------------------------------------------------------
# Inconsistency → retry with feedback, then exhaustion raises
# ---------------------------------------------------------------------------


class TestInconsistencyRetry:
    def test_inconsistent_then_valid_succeeds(self):
        """Inconsistent JSON on first attempt, valid on retry → succeeds."""
        bad = _make_verdict_json("guidance", score=4, passing=False, issues=[])  # inconsistent: score>=4 but passing=False
        good = _make_verdict_json("guidance", score=4, passing=True, issues=[])
        t = _make_transport([bad, good])
        v = judge_turn(_sample_turn(), "guidance", transport=t)
        assert v.score == 4
        assert v.passing is True
        assert t.run.call_count == 2

    def test_inconsistency_retry_prompt_contains_feedback(self):
        """Retry prompt for inconsistency must name the inconsistency."""
        bad = _make_verdict_json("guidance", score=5, passing=False)  # inconsistent
        good = _make_verdict_json("guidance", score=5, passing=True)
        t = _make_transport([bad, good])
        judge_turn(_sample_turn(), "guidance", transport=t)
        second_prompt = t.run.call_args_list[1][0][0]
        assert "inconsistent" in second_prompt.lower() or "PREVIOUS ATTEMPT FAILED" in second_prompt

    def test_all_inconsistent_raises_judge_error(self):
        """3 inconsistent verdicts → JudgeError(error_class='inconsistent_verdict')."""
        bad = _make_verdict_json("guidance", score=4, passing=False)  # inconsistent
        t = _make_transport([bad, bad, bad])
        with pytest.raises(JudgeError) as exc_info:
            judge_turn(_sample_turn(), "guidance", transport=t)
        assert exc_info.value.error_class == "inconsistent_verdict"
        assert exc_info.value.attempts == 3


# ---------------------------------------------------------------------------
# Transport error: nonzero exit / timeout
# ---------------------------------------------------------------------------


class TestTransportError:
    def test_nonzero_exit_raises_transport_error_then_judge_error(self):
        """Transport raising TransportError three times → JudgeError(transport_error)."""
        t = MagicMock(spec=CodexCliTransport)
        t.model_meta = {"model": "codex-default", "cli_version": "test"}
        t.run.side_effect = TransportError("nonzero exit code 1; stderr: 'err'")
        with pytest.raises(JudgeError) as exc_info:
            judge_turn(_sample_turn(), "guidance", transport=t)
        assert exc_info.value.error_class == "transport_error"

    def test_transport_error_includes_stderr_detail(self):
        """TransportError raised by CodexCliTransport subprocess includes stderr."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "fatal: something went wrong"
            mock_result.stdout = ""
            mock_run.return_value = mock_result
            # Also patch _get_codex_cli_version to avoid subprocess call
            with patch("evals.judge._get_codex_cli_version", return_value="test"):
                transport = CodexCliTransport()
                with pytest.raises(TransportError) as exc_info:
                    transport.run("hello")
                assert "stderr" in str(exc_info.value).lower() or "1" in str(exc_info.value)

    def test_timeout_raises_transport_error(self):
        """subprocess.TimeoutExpired → TransportError."""
        import subprocess as sp

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = sp.TimeoutExpired(cmd=["codex"], timeout=120)
            with patch("evals.judge._get_codex_cli_version", return_value="test"):
                transport = CodexCliTransport()
                with pytest.raises(TransportError):
                    transport.run("hello")


# ---------------------------------------------------------------------------
# Retry prompt contains error feedback
# ---------------------------------------------------------------------------


class TestRetryPromptContent:
    def test_parse_retry_prompt_contains_bad_output(self):
        """_build_retry_parse_prompt must embed the bad output in the retry prompt."""
        prompt = _build_retry_parse_prompt(
            original_prompt="original",
            bad_output="garbage xyz",
            error_msg="JSON decode failed: ...",
        )
        assert "garbage xyz" in prompt
        assert "JSON decode failed" in prompt

    def test_parse_retry_prompt_contains_original(self):
        """_build_retry_parse_prompt must include original prompt context."""
        prompt = _build_retry_parse_prompt(
            original_prompt="ORIGINAL CONTENT",
            bad_output="bad",
            error_msg="err",
        )
        assert "ORIGINAL CONTENT" in prompt

    def test_inconsistency_retry_prompt_contains_bad_json(self):
        """_build_retry_inconsistency_prompt must embed the bad JSON."""
        bad_json = '{"score": 4, "passing": false}'
        prompt = _build_retry_inconsistency_prompt(
            original_prompt="original",
            bad_json=bad_json,
            inconsistency_msg="score=4 requires passing=true",
        )
        assert bad_json in prompt
        assert "score=4" in prompt

    def test_inconsistency_retry_prompt_reminds_hard_rules(self):
        """Retry prompt for inconsistency should remind the model of the hard rules."""
        prompt = _build_retry_inconsistency_prompt(
            original_prompt="orig",
            bad_json="{}",
            inconsistency_msg="violated",
        )
        # Must mention the key rules.
        assert "passing" in prompt.lower()
        assert "score" in prompt.lower()


# ---------------------------------------------------------------------------
# judge_trace: fail-loud / no partial results
# ---------------------------------------------------------------------------


class TestJudgeTrace:
    def test_returns_all_verdicts(self):
        """judge_trace returns one verdict per (turn, dimension) pair."""
        good4 = _make_verdict_json("guidance", score=4, passing=True)
        good5 = _make_verdict_json("actionability", score=5, passing=True)
        # 2 turns × 2 dims = 4 calls
        t = _make_transport([good4, good5, good4, good5])
        turns = [_sample_turn(), _sample_turn()]
        verdicts = judge_trace(turns, ["guidance", "actionability"], transport=t)
        assert len(verdicts) == 4

    def test_trace_dict_with_turns_key(self):
        """judge_trace accepts a trace dict with a 'turns' key."""
        good = _make_verdict_json("guidance", score=4, passing=True)
        t = _make_transport([good])
        trace = {"trace_id": "t1", "turns": [_sample_turn()]}
        verdicts = judge_trace(trace, ["guidance"], transport=t)
        assert len(verdicts) == 1

    def test_fail_loud_no_partial_results(self):
        """If any turn exhausts retries, JudgeError propagates (no partial return)."""
        good = _make_verdict_json("guidance", score=4, passing=True)
        # First turn succeeds; second turn gets all garbage.
        t = _make_transport([good, "bad1", "bad2", "bad3"])
        turns = [_sample_turn(), _sample_turn()]
        with pytest.raises(JudgeError):
            judge_trace(turns, ["guidance"], transport=t)

    def test_invalid_dimension_raises_value_error_before_transport(self):
        """Deterministic dim in judge_trace raises ValueError before any transport call."""
        t = _make_transport([])
        with pytest.raises(ValueError, match="deterministic"):
            judge_trace([_sample_turn()], ["mistake_id"], transport=t)
        t.run.assert_not_called()

    def test_single_turn_single_dim_returns_one_verdict(self):
        good = _make_verdict_json("icap", score=3, passing=True)
        t = _make_transport([good])
        verdicts = judge_trace([_sample_turn()], ["icap"], transport=t)
        assert len(verdicts) == 1
        assert verdicts[0].dimension == "icap"
