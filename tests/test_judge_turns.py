"""Tests for scripts/judge_turns.py — T5.2 batch runner.

CI-safe: all tests use mocked transports — no real codex calls.

Contract under test
-------------------
- Label-set → turn-set join: a labeled turn_id missing from turns JSONL → abort
  naming it.
- Workfile resume: already-done (turn_id, dimension) pairs are skipped.
- Malformed workfile line: any malformed line aborts (no skipping).
- Final assembly only at 180/180 (or N/N for the pair-set used).
- Verdict line format parses through validate_judge's loader (integration test).
- _turn_id: correct profile-to-abbrev mapping.
- _turn_to_judge_dict: correct field mapping.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evals.judge import JUDGED_DIMENSIONS, CodexCliTransport, Verdict
from scripts.judge_turns import (
    _load_workfile,
    _turn_id,
    _turn_to_judge_dict,
    _verdict_line,
    _write_final,
    main,
)
from scripts.validate_judge import _load_verdicts


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_PROFILE_ABBREVS = {
    "fluent-but-hesitant": "fh",
    "self-corrector": "sc",
    "struggling-decoder": "sd",
}


def _make_turn(profile: str = "struggling-decoder", turn_index: int = 0) -> dict:
    return {
        "profile": profile,
        "turn_index": turn_index,
        "action_move": "SCAFFOLDED_HINT",
        "utterance": "Try tapping each sound.",
        "miscue_type": "substitution",
    }


def _make_verdict(
    dimension: str = "guidance",
    score: int = 4,
    passing: bool = True,
    issues: list[str] | None = None,
    rationale: str = "Good.",
) -> Verdict:
    return Verdict(
        dimension=dimension,
        score=score,
        passing=passing,
        issues=issues if issues is not None else [],
        rationale=rationale,
        model_meta={"model": "codex-default", "cli_version": "test"},
    )


def _make_transport_mock(verdicts: list[Verdict]) -> CodexCliTransport:
    """Return a mock transport whose judge_turn produces verdicts in order."""
    transport = MagicMock(spec=CodexCliTransport)
    transport.model_meta = {"model": "codex-default", "cli_version": "test"}
    # Return serialized JSON for each verdict in sequence.
    transport.run.side_effect = [
        json.dumps(
            {
                "dimension": v.dimension,
                "score": v.score,
                "passing": v.passing,
                "issues": list(v.issues),
                "rationale": v.rationale,
            }
        )
        for v in verdicts
    ]
    return transport


def _write_labels_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal human labels CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["turn_id", "dimension", "human_score", "human_passing", "rater_initials"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_turns_jsonl(path: Path, turns: list[dict]) -> None:
    """Write turns JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for t in turns:
            fh.write(json.dumps(t) + "\n")


def _write_workfile(path: Path, lines: list[dict]) -> None:
    """Write workfile lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")


def _make_verdict_dict(
    turn_id: str,
    dimension: str = "guidance",
    score: int = 4,
    passing: bool = True,
) -> dict:
    return {
        "turn_id": turn_id,
        "dimension": dimension,
        "score": score,
        "passing": passing,
        "issues": [],
        "rationale": "Test rationale.",
        "model_meta": {"model": "codex-default", "cli_version": "test"},
    }


# ---------------------------------------------------------------------------
# _turn_id: profile → abbreviation mapping
# ---------------------------------------------------------------------------


class TestTurnId:
    def test_struggling_decoder(self):
        t = _make_turn("struggling-decoder", 3)
        assert _turn_id(t) == "sd-t03"

    def test_fluent_but_hesitant(self):
        t = _make_turn("fluent-but-hesitant", 7)
        assert _turn_id(t) == "fh-t07"

    def test_self_corrector(self):
        t = _make_turn("self-corrector", 12)
        assert _turn_id(t) == "sc-t12"

    def test_zero_padded_index(self):
        t = _make_turn("struggling-decoder", 0)
        assert _turn_id(t) == "sd-t00"

    def test_two_digit_index(self):
        t = _make_turn("struggling-decoder", 23)
        assert _turn_id(t) == "sd-t23"

    def test_unknown_profile_raises_value_error(self):
        t = _make_turn("unknown-profile", 0)
        with pytest.raises(ValueError, match="Unknown profile"):
            _turn_id(t)


# ---------------------------------------------------------------------------
# _turn_to_judge_dict: field mapping
# ---------------------------------------------------------------------------


class TestTurnToJudgeDict:
    def test_utterance_mapped(self):
        t = _make_turn()
        d = _turn_to_judge_dict(t)
        assert d["utterance"] == t["utterance"]

    def test_action_move_mapped_to_move(self):
        t = _make_turn()
        t["action_move"] = "MODEL_THE_WORD"
        d = _turn_to_judge_dict(t)
        assert d["move"] == "MODEL_THE_WORD"

    def test_miscue_type_mapped_to_miscue(self):
        t = _make_turn()
        t["miscue_type"] = "omission"
        d = _turn_to_judge_dict(t)
        assert d["miscue"] == "omission"

    def test_none_miscue_becomes_empty_string(self):
        t = _make_turn()
        t["miscue_type"] = None
        d = _turn_to_judge_dict(t)
        assert d["miscue"] == ""


# ---------------------------------------------------------------------------
# _verdict_line: serialization
# ---------------------------------------------------------------------------


class TestVerdictLine:
    def test_contains_required_fields(self):
        v = _make_verdict()
        line = _verdict_line("sd-t00", "guidance", v)
        obj = json.loads(line)
        for field in ("turn_id", "dimension", "score", "passing", "issues", "rationale", "model_meta"):
            assert field in obj, f"missing field: {field}"

    def test_turn_id_preserved(self):
        v = _make_verdict()
        obj = json.loads(_verdict_line("fh-t01", "guidance", v))
        assert obj["turn_id"] == "fh-t01"

    def test_score_preserved(self):
        v = _make_verdict(score=3, passing=True)
        obj = json.loads(_verdict_line("sd-t00", "guidance", v))
        assert obj["score"] == 3

    def test_issues_is_list(self):
        v = _make_verdict(score=2, passing=False, issues=["Too cold."])
        obj = json.loads(_verdict_line("sd-t00", "guidance", v))
        assert isinstance(obj["issues"], list)

    def test_model_meta_preserved(self):
        v = _make_verdict()
        obj = json.loads(_verdict_line("sd-t00", "guidance", v))
        assert "model" in obj["model_meta"]
        assert "cli_version" in obj["model_meta"]


# ---------------------------------------------------------------------------
# Label-set → turn-set join: missing labeled turn → abort
# ---------------------------------------------------------------------------


class TestLabelTurnJoin:
    def test_missing_labeled_turn_aborts(self, tmp_path, capsys):
        """A labeled turn_id not present in the turns JSONL → sys.exit(1)."""
        labels_path = tmp_path / "labels.csv"
        turns_path = tmp_path / "turns.jsonl"
        out_path = tmp_path / "out.jsonl"
        workfile_path = tmp_path / ".work.jsonl"

        # Label references "sd-t99" which won't be in the JSONL.
        _write_labels_csv(labels_path, [
            {"turn_id": "sd-t99", "dimension": "guidance",
             "human_score": "4", "human_passing": "y", "rater_initials": "JC"},
        ])
        _write_turns_jsonl(turns_path, [_make_turn("struggling-decoder", 0)])

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--labels", str(labels_path),
                "--turns", str(turns_path),
                "--out", str(out_path),
                "--workfile", str(workfile_path),
            ])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "sd-t99" in captured.err

    def test_all_labeled_turns_present_proceeds(self, tmp_path):
        """All labeled turns in the JSONL → no abort."""
        labels_path = tmp_path / "labels.csv"
        turns_path = tmp_path / "turns.jsonl"
        out_path = tmp_path / "out.jsonl"
        workfile_path = tmp_path / ".work.jsonl"

        _write_labels_csv(labels_path, [
            {"turn_id": "sd-t00", "dimension": "guidance",
             "human_score": "4", "human_passing": "y", "rater_initials": "JC"},
        ])
        turn = _make_turn("struggling-decoder", 0)
        _write_turns_jsonl(turns_path, [turn])

        # Provide a pre-populated workfile with all pairs done to skip live calls.
        all_verdicts = [_make_verdict_dict("sd-t00", dim, 4, True) for dim in JUDGED_DIMENSIONS]
        _write_workfile(workfile_path, all_verdicts)

        # Should write final output without transport calls.
        main([
            "--labels", str(labels_path),
            "--turns", str(turns_path),
            "--out", str(out_path),
            "--workfile", str(workfile_path),
        ])
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Workfile resume: done pairs are skipped
# ---------------------------------------------------------------------------


class TestWorkfileResume:
    def test_done_pairs_skipped(self, tmp_path):
        """Already-done (turn_id, dimension) pairs are not re-judged."""
        labels_path = tmp_path / "labels.csv"
        turns_path = tmp_path / "turns.jsonl"
        out_path = tmp_path / "out.jsonl"
        workfile_path = tmp_path / ".work.jsonl"

        # One turn, all 3 dims labeled.
        _write_labels_csv(labels_path, [
            {"turn_id": "sd-t00", "dimension": dim,
             "human_score": "4", "human_passing": "y", "rater_initials": "JC"}
            for dim in JUDGED_DIMENSIONS
        ])
        _write_turns_jsonl(turns_path, [_make_turn("struggling-decoder", 0)])

        # Pre-populate workfile with all 3 dims done.
        done_verdicts = [_make_verdict_dict("sd-t00", dim, 4, True) for dim in JUDGED_DIMENSIONS]
        _write_workfile(workfile_path, done_verdicts)

        # main should detect all pairs done, write final, NOT call transport.
        with patch("scripts.judge_turns.CodexCliTransport") as mock_cls:
            main([
                "--labels", str(labels_path),
                "--turns", str(turns_path),
                "--out", str(out_path),
                "--workfile", str(workfile_path),
            ])
            # Transport should NOT be instantiated (nothing to judge).
            mock_cls.assert_not_called()

        assert out_path.exists()

    def test_partial_workfile_resumes_missing(self, tmp_path):
        """Only missing pairs are judged when workfile has partial results."""
        labels_path = tmp_path / "labels.csv"
        turns_path = tmp_path / "turns.jsonl"
        out_path = tmp_path / "out.jsonl"
        workfile_path = tmp_path / ".work.jsonl"

        dims = list(JUDGED_DIMENSIONS)  # all 3
        _write_labels_csv(labels_path, [
            {"turn_id": "sd-t00", "dimension": dim,
             "human_score": "4", "human_passing": "y", "rater_initials": "JC"}
            for dim in dims
        ])
        _write_turns_jsonl(turns_path, [_make_turn("struggling-decoder", 0)])

        # Pre-populate 2 of 3 dims.
        done_verdicts = [_make_verdict_dict("sd-t00", dims[0], 4, True),
                         _make_verdict_dict("sd-t00", dims[1], 4, True)]
        _write_workfile(workfile_path, done_verdicts)

        # Mock transport returns one verdict for the missing dim.
        missing_dim = dims[2]
        verdict_json = json.dumps({
            "dimension": missing_dim,
            "score": 3,
            "passing": True,
            "issues": [],
            "rationale": "Ok.",
        })
        with patch("scripts.judge_turns.CodexCliTransport") as mock_cls:
            mock_instance = MagicMock(spec=CodexCliTransport)
            mock_instance.model_meta = {"model": "codex-default", "cli_version": "test"}
            mock_instance.run.return_value = verdict_json
            mock_cls.return_value = mock_instance

            main([
                "--labels", str(labels_path),
                "--turns", str(turns_path),
                "--out", str(out_path),
                "--workfile", str(workfile_path),
            ])

        # Transport called exactly once (for the one missing dim).
        mock_instance.run.assert_called_once()
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Malformed workfile line aborts
# ---------------------------------------------------------------------------


class TestMalformedWorkfile:
    def test_invalid_json_line_aborts(self, tmp_path, capsys):
        """Workfile with a malformed JSON line → sys.exit(1)."""
        labels_path = tmp_path / "labels.csv"
        turns_path = tmp_path / "turns.jsonl"
        out_path = tmp_path / "out.jsonl"
        workfile_path = tmp_path / ".work.jsonl"

        _write_labels_csv(labels_path, [
            {"turn_id": "sd-t00", "dimension": "guidance",
             "human_score": "4", "human_passing": "y", "rater_initials": "JC"},
        ])
        _write_turns_jsonl(turns_path, [_make_turn("struggling-decoder", 0)])

        # Write a corrupt workfile.
        workfile_path.parent.mkdir(parents=True, exist_ok=True)
        workfile_path.write_text('{"turn_id": "sd-t00", "dimension": "guidance" INVALID\n')

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--labels", str(labels_path),
                "--turns", str(turns_path),
                "--out", str(out_path),
                "--workfile", str(workfile_path),
            ])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "MALFORMED" in captured.err or "malformed" in captured.err.lower()

    def test_missing_required_field_aborts(self, tmp_path, capsys):
        """Workfile line missing a required field → sys.exit(1)."""
        labels_path = tmp_path / "labels.csv"
        turns_path = tmp_path / "turns.jsonl"
        out_path = tmp_path / "out.jsonl"
        workfile_path = tmp_path / ".work.jsonl"

        _write_labels_csv(labels_path, [
            {"turn_id": "sd-t00", "dimension": "guidance",
             "human_score": "4", "human_passing": "y", "rater_initials": "JC"},
        ])
        _write_turns_jsonl(turns_path, [_make_turn("struggling-decoder", 0)])

        # A line with no 'score' field.
        bad_line = json.dumps({"turn_id": "sd-t00", "dimension": "guidance",
                               "passing": True, "issues": [], "rationale": "x",
                               "model_meta": {}})
        workfile_path.parent.mkdir(parents=True, exist_ok=True)
        workfile_path.write_text(bad_line + "\n")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--labels", str(labels_path),
                "--turns", str(turns_path),
                "--out", str(out_path),
                "--workfile", str(workfile_path),
            ])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "MALFORMED" in captured.err or "malformed" in captured.err.lower()


# ---------------------------------------------------------------------------
# Final assembly only at N/N (all pairs complete)
# ---------------------------------------------------------------------------


class TestFinalAssembly:
    def test_final_output_not_written_when_incomplete(self, tmp_path):
        """Final output is NOT written when not all pairs are done."""
        labels_path = tmp_path / "labels.csv"
        turns_path = tmp_path / "turns.jsonl"
        out_path = tmp_path / "out.jsonl"
        workfile_path = tmp_path / ".work.jsonl"

        dims = list(JUDGED_DIMENSIONS)
        _write_labels_csv(labels_path, [
            {"turn_id": "sd-t00", "dimension": dim,
             "human_score": "4", "human_passing": "y", "rater_initials": "JC"}
            for dim in dims
        ])
        _write_turns_jsonl(turns_path, [_make_turn("struggling-decoder", 0)])

        # Only 1 of 3 dims done — use --limit 0 to process nothing extra.
        done_verdicts = [_make_verdict_dict("sd-t00", dims[0], 4, True)]
        _write_workfile(workfile_path, done_verdicts)

        with patch("scripts.judge_turns.CodexCliTransport") as mock_cls:
            mock_instance = MagicMock(spec=CodexCliTransport)
            mock_instance.model_meta = {"model": "codex-default", "cli_version": "test"}
            mock_cls.return_value = mock_instance
            # Provide valid responses for the missing dims.
            mock_instance.run.return_value = json.dumps({
                "dimension": "guidance", "score": 3, "passing": True,
                "issues": [], "rationale": "ok",
            })
            # Use --limit 0 to not process any pairs.
            main([
                "--labels", str(labels_path),
                "--turns", str(turns_path),
                "--out", str(out_path),
                "--workfile", str(workfile_path),
                "--limit", "0",
            ])

        # Final output not written.
        assert not out_path.exists()

    def test_final_output_written_when_all_complete(self, tmp_path):
        """Final output IS written when all pairs are done."""
        labels_path = tmp_path / "labels.csv"
        turns_path = tmp_path / "turns.jsonl"
        out_path = tmp_path / "out.jsonl"
        workfile_path = tmp_path / ".work.jsonl"

        dims = list(JUDGED_DIMENSIONS)
        _write_labels_csv(labels_path, [
            {"turn_id": "sd-t00", "dimension": dim,
             "human_score": "4", "human_passing": "y", "rater_initials": "JC"}
            for dim in dims
        ])
        _write_turns_jsonl(turns_path, [_make_turn("struggling-decoder", 0)])

        # All 3 dims done in workfile.
        done_verdicts = [_make_verdict_dict("sd-t00", dim, 4, True) for dim in dims]
        _write_workfile(workfile_path, done_verdicts)

        main([
            "--labels", str(labels_path),
            "--turns", str(turns_path),
            "--out", str(out_path),
            "--workfile", str(workfile_path),
        ])

        assert out_path.exists()
        # Workfile removed.
        assert not workfile_path.exists()

    def test_final_output_deterministic_order(self, tmp_path):
        """Final output lines are sorted by (turn_id, dimension)."""
        labels_path = tmp_path / "labels.csv"
        turns_path = tmp_path / "turns.jsonl"
        out_path = tmp_path / "out.jsonl"
        workfile_path = tmp_path / ".work.jsonl"

        dims = list(JUDGED_DIMENSIONS)
        _write_labels_csv(labels_path, [
            {"turn_id": "sd-t00", "dimension": dim,
             "human_score": "4", "human_passing": "y", "rater_initials": "JC"}
            for dim in dims
        ])
        _write_turns_jsonl(turns_path, [_make_turn("struggling-decoder", 0)])

        # Write verdicts in reverse order to verify sorting.
        done_verdicts = [_make_verdict_dict("sd-t00", dim, 4, True) for dim in reversed(dims)]
        _write_workfile(workfile_path, done_verdicts)

        main([
            "--labels", str(labels_path),
            "--turns", str(turns_path),
            "--out", str(out_path),
            "--workfile", str(workfile_path),
        ])

        with out_path.open() as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        keys = [(obj["turn_id"], obj["dimension"]) for obj in lines]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Integration test: verdict line format parses through validate_judge loader
# ---------------------------------------------------------------------------


class TestVerdictFormatIntegration:
    def test_verdict_line_parses_through_validate_judge_loader(self, tmp_path):
        """_verdict_line output passes validate_judge's _load_verdicts without error."""
        verdicts_path = tmp_path / "verdicts.jsonl"

        # Write some verdict lines in the format judge_turns produces.
        lines = [
            _verdict_line("sd-t00", "guidance", _make_verdict("guidance", 4, True)),
            _verdict_line("sd-t00", "actionability", _make_verdict("actionability", 2, False, ["Too vague."])),
            _verdict_line("sd-t00", "icap", _make_verdict("icap", 3, True)),
        ]
        with verdicts_path.open("w") as fh:
            for line in lines:
                fh.write(line + "\n")

        # validate_judge._load_verdicts must load this without error.
        result = _load_verdicts(verdicts_path)
        assert ("sd-t00", "guidance") in result
        assert ("sd-t00", "actionability") in result
        assert ("sd-t00", "icap") in result
        # Check passing values.
        assert result[("sd-t00", "guidance")] is True
        assert result[("sd-t00", "actionability")] is False
        assert result[("sd-t00", "icap")] is True

    def test_full_workfile_parses_through_validate_judge_loader(self, tmp_path):
        """A workfile written by _write_final parses through _load_verdicts."""
        out_path = tmp_path / "verdicts.jsonl"

        done: dict[tuple[str, str], dict] = {}
        for dim in JUDGED_DIMENSIONS:
            tid = "fh-t01"
            v = _make_verdict(dim, 4, True)
            line = _verdict_line(tid, dim, v)
            done[(tid, dim)] = json.loads(line)

        _write_final(out_path, done)

        result = _load_verdicts(out_path)
        for dim in JUDGED_DIMENSIONS:
            assert ("fh-t01", dim) in result


# ---------------------------------------------------------------------------
# Smoke-test: --limit exits without final output, workfile kept
# ---------------------------------------------------------------------------


class TestSmokeLimit:
    def test_limit_zero_no_transport(self, tmp_path):
        """--limit 0: no pairs processed, no transport created, no final output."""
        labels_path = tmp_path / "labels.csv"
        turns_path = tmp_path / "turns.jsonl"
        out_path = tmp_path / "out.jsonl"
        workfile_path = tmp_path / ".work.jsonl"

        _write_labels_csv(labels_path, [
            {"turn_id": "sd-t00", "dimension": "guidance",
             "human_score": "4", "human_passing": "y", "rater_initials": "JC"},
        ])
        _write_turns_jsonl(turns_path, [_make_turn("struggling-decoder", 0)])

        with patch("scripts.judge_turns.CodexCliTransport") as mock_cls:
            main([
                "--labels", str(labels_path),
                "--turns", str(turns_path),
                "--out", str(out_path),
                "--workfile", str(workfile_path),
                "--limit", "0",
            ])
            mock_cls.assert_not_called()

        assert not out_path.exists()

    def test_limit_one_processes_one_pair(self, tmp_path, capsys):
        """--limit 1: exactly one pair judged, workfile updated, final output NOT written."""
        labels_path = tmp_path / "labels.csv"
        turns_path = tmp_path / "turns.jsonl"
        out_path = tmp_path / "out.jsonl"
        workfile_path = tmp_path / ".work.jsonl"

        dims = list(JUDGED_DIMENSIONS)
        _write_labels_csv(labels_path, [
            {"turn_id": "sd-t00", "dimension": dim,
             "human_score": "4", "human_passing": "y", "rater_initials": "JC"}
            for dim in dims
        ])
        _write_turns_jsonl(turns_path, [_make_turn("struggling-decoder", 0)])

        with patch("scripts.judge_turns.CodexCliTransport") as mock_cls:
            mock_instance = MagicMock(spec=CodexCliTransport)
            mock_instance.model_meta = {"model": "codex-default", "cli_version": "test"}
            mock_instance.run.return_value = json.dumps({
                "dimension": dims[0], "score": 4, "passing": True,
                "issues": [], "rationale": "Looks good.",
            })
            mock_cls.return_value = mock_instance

            main([
                "--labels", str(labels_path),
                "--turns", str(turns_path),
                "--out", str(out_path),
                "--workfile", str(workfile_path),
                "--limit", "1",
            ])

        # Transport called once.
        mock_instance.run.assert_called_once()
        # Final output NOT written.
        assert not out_path.exists()
        # Workfile written.
        assert workfile_path.exists()
        captured = capsys.readouterr()
        assert "smoke" in captured.err


# ---------------------------------------------------------------------------
# _load_workfile: direct unit tests
# ---------------------------------------------------------------------------


class TestLoadWorkfile:
    def test_empty_path_returns_empty(self, tmp_path):
        p = tmp_path / "nonexistent.jsonl"
        result = _load_workfile(p)
        assert result == {}

    def test_valid_lines_loaded(self, tmp_path):
        p = tmp_path / "work.jsonl"
        v = _make_verdict_dict("sd-t00", "guidance", 4, True)
        _write_workfile(p, [v])
        result = _load_workfile(p)
        assert ("sd-t00", "guidance") in result

    def test_malformed_json_aborts(self, tmp_path):
        p = tmp_path / "work.jsonl"
        p.write_text("NOTJSON\n")
        with pytest.raises(SystemExit) as exc_info:
            _load_workfile(p)
        assert exc_info.value.code == 1

    def test_missing_field_aborts(self, tmp_path):
        p = tmp_path / "work.jsonl"
        # Missing 'score'
        bad = {"turn_id": "sd-t00", "dimension": "guidance", "passing": True,
               "issues": [], "rationale": "x", "model_meta": {}}
        p.write_text(json.dumps(bad) + "\n")
        with pytest.raises(SystemExit) as exc_info:
            _load_workfile(p)
        assert exc_info.value.code == 1

    def test_duplicate_key_aborts(self, tmp_path):
        p = tmp_path / "work.jsonl"
        v = _make_verdict_dict("sd-t00", "guidance", 4, True)
        _write_workfile(p, [v, v])  # duplicate
        with pytest.raises(SystemExit) as exc_info:
            _load_workfile(p)
        assert exc_info.value.code == 1
