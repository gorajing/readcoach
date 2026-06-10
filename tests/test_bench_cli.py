"""Tests for the BYO-ASR scoring CLI (T2.5) — ``readcoach-bench score``.

CI-safe: no network, no model, no audio.  Uses the committed gold.jsonl (88
items, tracked in git; only the .wav clips are gitignored) plus synthetic,
hand-built hypotheses.

The load-bearing test here is ``test_import_audit`` — it runs a REAL scoring
call through the library path and asserts that no model/service module
(faster_whisper / weave / anthropic / google.genai / wandb) ends up in
``sys.modules``.  That is the "auditable in five minutes" claim made in
docs/BENCHMARK.md, expressed as an executable assertion.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from readcoach.asr import Word
from readcoach.bench_cli import (
    DuplicateUttError,
    GoldItem,
    HypothesisError,
    PartialCoverageError,
    ScoredSet,
    UnknownUttError,
    _parse_hypotheses_arg,
    build_json_output,
    check_coverage,
    load_baseline,
    load_gold,
    load_hypotheses,
    main,
    render_table,
    score_hypotheses,
)
from readcoach.miscue import Miscue, detect, match_counts

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_GOLD = _PROJECT_ROOT / "data" / "benchmark" / "gold.jsonl"
_BASELINE = _PROJECT_ROOT / "evals" / "results" / "v0.json"


# ---------------------------------------------------------------------------
# Helpers — build small synthetic gold + hypotheses files on disk
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _gold_row(utt_id: str, target_text: str, gold: list[dict]) -> dict:
    return {
        "utt_id": utt_id,
        "target_text": target_text,
        "gold": gold,
    }


def _hyp_row(utt_id: str, words: list) -> dict:
    """words: list of either bare strings or full word dicts."""
    word_dicts = [{"text": w} if isinstance(w, str) else w for w in words]
    return {"utt_id": utt_id, "words": word_dicts}


# ===========================================================================
# Schema parsing — optional fields and defaults
# ===========================================================================

class TestSchemaParsing:
    def test_text_only_words_default_timings_and_confidence(self, tmp_path):
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        _write_jsonl(gold_p, [_gold_row("u1", "the cat sat", [])])
        _write_jsonl(hyp_p, [_hyp_row("u1", ["the", "cat", "sat"])])

        gold = load_gold(gold_p)
        hyp = load_hypotheses(hyp_p, gold)

        words = hyp["u1"]
        assert [w.text for w in words] == ["the", "cat", "sat"]
        # Missing timings -> None.
        assert all(w.start is None and w.end is None for w in words)
        # Missing confidence -> 1.0.
        assert all(w.confidence == 1.0 for w in words)

    def test_explicit_timings_and_confidence_preserved(self, tmp_path):
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        _write_jsonl(gold_p, [_gold_row("u1", "the cat", [])])
        _write_jsonl(
            hyp_p,
            [
                _hyp_row(
                    "u1",
                    [
                        {"text": "the", "start": 0.0, "end": 0.4, "confidence": 0.9},
                        {"text": "cat", "start": 0.5, "end": 0.9},
                    ],
                )
            ],
        )
        gold = load_gold(gold_p)
        hyp = load_hypotheses(hyp_p, gold)
        the, cat = hyp["u1"]
        assert (the.start, the.end, the.confidence) == (0.0, 0.4, 0.9)
        # Second word: explicit timings, defaulted confidence.
        assert (cat.start, cat.end, cat.confidence) == (0.5, 0.9, 1.0)

    def test_missing_text_field_fails_loud(self, tmp_path):
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        _write_jsonl(gold_p, [_gold_row("u1", "the cat", [])])
        _write_jsonl(hyp_p, [{"utt_id": "u1", "words": [{"start": 0.0}]}])
        gold = load_gold(gold_p)
        with pytest.raises(HypothesisError, match="no required 'text' field"):
            load_hypotheses(hyp_p, gold)

    def test_missing_words_field_fails_loud(self, tmp_path):
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        _write_jsonl(gold_p, [_gold_row("u1", "the cat", [])])
        _write_jsonl(hyp_p, [{"utt_id": "u1"}])
        gold = load_gold(gold_p)
        with pytest.raises(HypothesisError, match="no 'words' field"):
            load_hypotheses(hyp_p, gold)

    def test_missing_utt_id_field_fails_loud(self, tmp_path):
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        _write_jsonl(gold_p, [_gold_row("u1", "the cat", [])])
        _write_jsonl(hyp_p, [{"words": [{"text": "the"}]}])
        gold = load_gold(gold_p)
        with pytest.raises(HypothesisError, match="no 'utt_id' field"):
            load_hypotheses(hyp_p, gold)


# ===========================================================================
# Fail-loud coverage cases
# ===========================================================================

class TestFailLoud:
    def test_duplicate_utt_id_in_hypotheses(self, tmp_path):
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        _write_jsonl(gold_p, [_gold_row("u1", "the cat", [])])
        _write_jsonl(
            hyp_p,
            [_hyp_row("u1", ["the", "cat"]), _hyp_row("u1", ["the", "dog"])],
        )
        gold = load_gold(gold_p)
        with pytest.raises(DuplicateUttError, match="duplicate utt_id 'u1'"):
            load_hypotheses(hyp_p, gold)

    def test_unknown_utt_id_in_hypotheses(self, tmp_path):
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        _write_jsonl(gold_p, [_gold_row("u1", "the cat", [])])
        _write_jsonl(hyp_p, [_hyp_row("u_other", ["the", "cat"])])
        gold = load_gold(gold_p)
        with pytest.raises(UnknownUttError, match="not in the gold set"):
            load_hypotheses(hyp_p, gold)

    def test_partial_coverage_without_flag_errors(self, tmp_path):
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        _write_jsonl(
            gold_p,
            [_gold_row("u1", "the cat", []), _gold_row("u2", "the dog", [])],
        )
        _write_jsonl(hyp_p, [_hyp_row("u1", ["the", "cat"])])
        gold = load_gold(gold_p)
        hyp = load_hypotheses(hyp_p, gold)
        with pytest.raises(PartialCoverageError, match="covers 1/2 gold items"):
            check_coverage(hyp, gold, hyp_p, allow_partial=False)

    def test_partial_coverage_with_flag_allowed(self, tmp_path):
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        _write_jsonl(
            gold_p,
            [_gold_row("u1", "the cat", []), _gold_row("u2", "the dog", [])],
        )
        _write_jsonl(hyp_p, [_hyp_row("u1", ["the", "cat"])])
        gold = load_gold(gold_p)
        hyp = load_hypotheses(hyp_p, gold)
        # Should NOT raise.
        check_coverage(hyp, gold, hyp_p, allow_partial=True)

    def test_full_coverage_never_partial(self, tmp_path):
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        _write_jsonl(gold_p, [_gold_row("u1", "the cat", [])])
        _write_jsonl(hyp_p, [_hyp_row("u1", ["the", "cat"])])
        gold = load_gold(gold_p)
        hyp = load_hypotheses(hyp_p, gold)
        check_coverage(hyp, gold, hyp_p, allow_partial=False)  # no raise

    def test_malformed_json_line_fails_loud(self, tmp_path):
        gold_p = tmp_path / "gold.jsonl"
        _write_jsonl(gold_p, [_gold_row("u1", "the cat", [])])
        hyp_p = tmp_path / "hyp.jsonl"
        hyp_p.write_text('{"utt_id": "u1", "words": [}\n', encoding="utf-8")
        gold = load_gold(gold_p)
        with pytest.raises(HypothesisError, match="invalid JSON"):
            load_hypotheses(hyp_p, gold)

    def test_malformed_json_error_contains_file_and_lineno(self, tmp_path):
        """HypothesisError for a malformed JSON line must name the file and line number.

        CI-safe: no network, no model, no audio.  The error path goes through
        _load_jsonl -> HypothesisError -> cmd_score's except HypothesisError handler,
        so the CLI surfaces it as a clean 'ERROR:' message + exit 1, not a traceback.
        """
        gold_p = tmp_path / "gold.jsonl"
        _write_jsonl(gold_p, [_gold_row("u1", "the cat", [])])
        hyp_p = tmp_path / "bad.jsonl"
        # Line 1 is valid; line 2 is malformed — the error must cite line 2.
        hyp_p.write_text(
            '{"utt_id": "u1", "words": [{"text": "the"}]}\n'
            'NOT VALID JSON\n',
            encoding="utf-8",
        )
        gold = load_gold(gold_p)

        # 1. Exception type must be HypothesisError (so the CLI handler catches it).
        with pytest.raises(HypothesisError) as exc_info:
            load_hypotheses(hyp_p, gold)

        msg = str(exc_info.value)
        # 2. Message must name the file.
        assert str(hyp_p) in msg, f"file path missing from error: {msg!r}"
        # 3. Message must name the line number.
        assert "line 2" in msg, f"line number missing from error: {msg!r}"

    def test_duplicate_gold_utt_id_fails_loud(self, tmp_path):
        gold_p = tmp_path / "gold.jsonl"
        _write_jsonl(
            gold_p,
            [_gold_row("u1", "the cat", []), _gold_row("u1", "the dog", [])],
        )
        with pytest.raises(DuplicateUttError, match="duplicate utt_id 'u1' in gold"):
            load_gold(gold_p)


# ===========================================================================
# --hypotheses NAME= parsing
# ===========================================================================

class TestHypothesesArgParsing:
    def test_named_prefix(self):
        name, path = _parse_hypotheses_arg("wav2vec2=examples/h.jsonl")
        assert name == "wav2vec2"
        assert path == Path("examples/h.jsonl")

    def test_bare_path_defaults_to_stem(self):
        name, path = _parse_hypotheses_arg("/tmp/whisper_tiny.jsonl")
        assert name == "whisper_tiny"
        assert path == Path("/tmp/whisper_tiny.jsonl")

    def test_empty_name_before_equals_errors(self):
        with pytest.raises(ValueError, match="empty NAME"):
            _parse_hypotheses_arg("=foo.jsonl")


# ===========================================================================
# Micro-aggregation equals match_counts ground truth
# ===========================================================================

class TestMicroAggregationGroundTruth:
    def test_aggregation_matches_match_counts_sum(self, tmp_path):
        """score_hypotheses must equal the hand-summed match_counts over items.

        Build two synthetic items with known miscues, compute the ground-truth
        micro-aggregate directly via detect()+match_counts(), and assert the CLI
        aggregation produces identical TP/FP/FN per class.
        """
        target1 = "the cat sat on the mat"
        target2 = "a big dog ran fast"
        # Item 1: substitution "cat"->"bat" at index 1.
        hyp1_words = ["the", "bat", "sat", "on", "the", "mat"]
        # Item 2: omission of "dog" (index 2).
        hyp2_words = ["a", "big", "ran", "fast"]

        gold1 = [Miscue("substitution", "cat", "bat", 1)]
        gold2 = [Miscue("omission", "dog", None, 2)]

        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        _write_jsonl(
            gold_p,
            [
                _gold_row(
                    "u1", target1,
                    [{"type": "substitution", "target_word": "cat",
                      "said_word": "bat", "index": 1}],
                ),
                _gold_row(
                    "u2", target2,
                    [{"type": "omission", "target_word": "dog",
                      "said_word": None, "index": 2}],
                ),
            ],
        )
        _write_jsonl(
            hyp_p,
            [_hyp_row("u1", hyp1_words), _hyp_row("u2", hyp2_words)],
        )

        # --- Ground truth: detect + match_counts summed by hand ---
        classes = ("substitution", "omission", "insertion",
                   "self_correction", "hesitation")
        gt = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
        gt_correct = 0
        for tgt, hyp_words, goldm in (
            (target1, [Word(w) for w in hyp1_words], gold1),
            (target2, [Word(w) for w in hyp2_words], gold2),
        ):
            pred = detect(hyp_words, tgt)
            counts = match_counts(pred, goldm, len(tgt.split()))
            for c in classes:
                gt[c]["tp"] += counts[c]["tp"]
                gt[c]["fp"] += counts[c]["fp"]
                gt[c]["fn"] += counts[c]["fn"]
            gt_correct += counts["_correct_words"]

        # --- CLI path ---
        gold = load_gold(gold_p)
        hyp = load_hypotheses(hyp_p, gold)
        result = score_hypotheses(hyp, gold)
        metrics = result["metrics"]

        for c in classes:
            assert metrics[c]["tp"] == gt[c]["tp"], c
            assert metrics[c]["fp"] == gt[c]["fp"], c
            assert metrics[c]["fn"] == gt[c]["fn"], c

        # Substitution should be a TP; omission should be a TP.
        assert metrics["substitution"]["tp"] == 1
        assert metrics["omission"]["tp"] == 1
        assert result["n_covered"] == 2

    def test_perfect_clean_read_has_no_false_positives(self, tmp_path):
        """A clean transcript of a clean passage must produce zero FPs."""
        target = "the cat sat on the mat"
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        _write_jsonl(gold_p, [_gold_row("u1", target, [])])
        _write_jsonl(hyp_p, [_hyp_row("u1", target.split())])
        gold = load_gold(gold_p)
        hyp = load_hypotheses(hyp_p, gold)
        metrics = score_hypotheses(hyp, gold)["metrics"]
        for c in ("substitution", "omission", "insertion",
                  "self_correction", "hesitation"):
            assert metrics[c]["fp"] == 0, c
        assert metrics["fp_per_100_correct_words"] == 0.0


# ===========================================================================
# Import audit — the auditable-in-five-minutes claim, as an assertion
# ===========================================================================

_BANNED_MODULES = (
    "faster_whisper",
    "weave",
    "anthropic",
    "google.genai",
    "wandb",
)


def test_import_audit_subprocess():
    """A real scoring run imports NONE of the model/service modules.

    Run in a FRESH interpreter (subprocess) so the assertion is not contaminated
    by other tests that may have imported these modules in-process.  This scores
    3 real gold items with synthetic hypotheses through the public library path
    (load_gold -> load_hypotheses -> score_hypotheses), then inspects
    sys.modules.
    """
    prog = r"""
import json, sys, tempfile
from pathlib import Path
from readcoach.bench_cli import load_gold, load_hypotheses, check_coverage, score_hypotheses

gold = load_gold(Path("data/benchmark/gold.jsonl"))
# Build a synthetic 3-item hypotheses file from the gold's miscued_text.
utts = list(gold)[:3]
rows = []
for u in utts:
    item = gold[u]
    rows.append({"utt_id": u, "words": [{"text": w} for w in item.target_text.split()]})
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
    hp = f.name
hyp = load_hypotheses(Path(hp), gold)
check_coverage(hyp, gold, Path(hp), allow_partial=True)
res = score_hypotheses(hyp, gold)
assert res["n_covered"] == 3, res

banned = ["faster_whisper", "weave", "anthropic", "google.genai", "wandb"]
present = [m for m in banned if m in sys.modules]
if present:
    print("AUDIT_FAIL:" + ",".join(present))
    sys.exit(2)
# readcoach + jiwer MUST be present (proves we ran the real path, not a stub).
assert "readcoach.miscue" in sys.modules
assert "jiwer" in sys.modules
print("AUDIT_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", prog],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import-audit subprocess failed:\nSTDOUT:{result.stdout}\nSTDERR:{result.stderr}"
    )
    assert "AUDIT_OK" in result.stdout, result.stdout


def test_import_audit_in_process_after_score():
    """In-process belt-and-suspenders: after a score call, the banned modules
    must not have been imported BY the scoring path.

    This is weaker than the subprocess test (another test could have imported a
    banned module first), so we only assert for modules not already present
    before the call — the subprocess test is the authoritative one.
    """
    before = {m for m in _BANNED_MODULES if m in sys.modules}

    gold = load_gold(_GOLD)
    utts = list(gold)[:3]
    words_by_utt = {
        u: [Word(w) for w in gold[u].target_text.split()] for u in utts
    }
    res = score_hypotheses(words_by_utt, gold)
    assert res["n_covered"] == 3

    after = {m for m in _BANNED_MODULES if m in sys.modules}
    newly_imported = after - before
    assert not newly_imported, (
        f"scoring path imported banned modules: {newly_imported}"
    )


# ===========================================================================
# Table rendering — with two consumer sets (masking curve)
# ===========================================================================

def _fake_metrics(sub_f1, om_f1, fp100):
    """Build a finalized metrics dict with the given values; others minimal."""
    def cell(f1):
        return {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0,
                "recall": 1.0, "f1": f1, "n_gold": 1, "n_pred": 1}
    none_cell = {"tp": 0, "fp": 0, "fn": 0, "precision": None,
                 "recall": None, "f1": None, "n_gold": 0, "n_pred": 0}
    return {
        "substitution": cell(sub_f1),
        "omission": cell(om_f1),
        "insertion": none_cell,
        "self_correction": none_cell,
        "hesitation": none_cell,
        "fp_per_100_correct_words": fp100,
    }


class TestRenderTable:
    def test_two_sets_is_masking_curve(self):
        baseline = load_baseline(_BASELINE)
        scored = [
            ScoredSet("setA", n_covered=88, n_gold=88,
                      metrics=_fake_metrics(0.5, 0.6, 3.0)),
            ScoredSet("setB", n_covered=88, n_gold=88,
                      metrics=_fake_metrics(0.7, 0.8, 1.0)),
        ]
        table = render_table(baseline, scored)
        # Structural assertions (no over-fitting to exact spacing).
        assert "ALIGNMENT" in table
        assert "TRANSCRIPT-STYLE-SENSITIVE" in table
        assert "self_correction + hesitation test the ASR's transcript STYLE" in table
        assert "masking-curve table" in table  # footer present with 2 sets
        assert "setA" in table and "setB" in table
        assert "fp_per_100_correct_words" in table
        # Baseline column present (faster-whisper).
        assert "faster-whisper-small" in table

    def test_single_set_no_masking_curve_footer(self):
        baseline = load_baseline(_BASELINE)
        scored = [
            ScoredSet("solo", n_covered=88, n_gold=88,
                      metrics=_fake_metrics(0.5, 0.6, 3.0)),
        ]
        table = render_table(baseline, scored)
        assert "masking-curve table" not in table
        assert "solo" in table

    def test_partial_coverage_flagged_in_table(self):
        baseline = load_baseline(_BASELINE)
        scored = [
            ScoredSet("partial", n_covered=10, n_gold=88,
                      metrics=_fake_metrics(0.5, 0.6, 3.0)),
        ]
        table = render_table(baseline, scored)
        assert "n_covered = 10/88" in table
        assert "PARTIAL" in table

    def test_none_metric_renders_dash(self):
        baseline = load_baseline(_BASELINE)
        scored = [
            ScoredSet("s", n_covered=88, n_gold=88,
                      metrics=_fake_metrics(0.5, 0.6, 3.0)),
        ]
        table = render_table(baseline, scored)
        # insertion is the None cell in _fake_metrics -> should show a dash.
        assert "—" in table


# ===========================================================================
# JSON output mirrors the table data
# ===========================================================================

def test_json_output_structure():
    baseline = load_baseline(_BASELINE)
    scored = [
        ScoredSet("setA", n_covered=88, n_gold=88,
                  metrics=_fake_metrics(0.5, 0.6, 3.0)),
        ScoredSet("setB", n_covered=44, n_gold=88,
                  metrics=_fake_metrics(0.7, 0.8, 1.0)),
    ]
    out = build_json_output(baseline, _BASELINE, scored, n_gold=88)
    assert out["schema"] == "readcoach-bench/score/v1"
    assert out["n_gold"] == 88
    assert out["class_groups"]["alignment"] == ["substitution", "omission", "insertion"]
    assert out["class_groups"]["transcript_style_sensitive"] == [
        "self_correction", "hesitation"
    ]
    assert out["is_masking_curve"] is True
    assert len(out["hypothesis_sets"]) == 2
    assert out["hypothesis_sets"][1]["n_covered"] == 44
    assert out["baseline"]["bias"] == "none"


# ===========================================================================
# End-to-end CLI via main() — exit codes and table on stdout
# ===========================================================================

class TestMainEndToEnd:
    def test_full_coverage_e2e(self, tmp_path, capsys):
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        target = "the cat sat on the mat"
        _write_jsonl(gold_p, [_gold_row("u1", target, [])])
        _write_jsonl(hyp_p, [_hyp_row("u1", target.split())])
        rc = main(
            [
                "score",
                "--hypotheses", f"mine={hyp_p}",
                "--gold", str(gold_p),
                "--baseline", str(_BASELINE),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "mine" in out
        assert "ALIGNMENT" in out

    def test_partial_without_flag_exit_1(self, tmp_path, capsys):
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        _write_jsonl(
            gold_p,
            [_gold_row("u1", "the cat", []), _gold_row("u2", "the dog", [])],
        )
        _write_jsonl(hyp_p, [_hyp_row("u1", ["the", "cat"])])
        rc = main(
            [
                "score",
                "--hypotheses", str(hyp_p),
                "--gold", str(gold_p),
                "--baseline", str(_BASELINE),
            ]
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "covers 1/2 gold items" in err

    def test_json_flag_writes_file(self, tmp_path):
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        out_p = tmp_path / "out.json"
        target = "the cat sat"
        _write_jsonl(gold_p, [_gold_row("u1", target, [])])
        _write_jsonl(hyp_p, [_hyp_row("u1", target.split())])
        rc = main(
            [
                "score",
                "--hypotheses", f"mine={hyp_p}",
                "--gold", str(gold_p),
                "--baseline", str(_BASELINE),
                "--json", str(out_p),
            ]
        )
        assert rc == 0
        data = json.loads(out_p.read_text(encoding="utf-8"))
        assert data["hypothesis_sets"][0]["label"] == "mine"

    def test_duplicate_label_exit_2(self, tmp_path, capsys):
        gold_p = tmp_path / "gold.jsonl"
        hyp_p = tmp_path / "hyp.jsonl"
        target = "the cat"
        _write_jsonl(gold_p, [_gold_row("u1", target, [])])
        _write_jsonl(hyp_p, [_hyp_row("u1", target.split())])
        rc = main(
            [
                "score",
                "--hypotheses", f"dup={hyp_p}",
                "--hypotheses", f"dup={hyp_p}",
                "--gold", str(gold_p),
                "--baseline", str(_BASELINE),
            ]
        )
        assert rc == 2
        assert "duplicate hypothesis-set label" in capsys.readouterr().err


# ===========================================================================
# load_gold / load_baseline against the committed real files
# ===========================================================================

def test_load_real_gold():
    gold = load_gold(_GOLD)
    assert len(gold) == 88
    assert all(isinstance(v, GoldItem) for v in gold.values())
    # p01-clean has no miscues.
    assert gold["p01-clean"].gold_miscues == []


def test_load_real_baseline():
    baseline = load_baseline(_BASELINE)
    # bias=none substitution F1 from the committed v0.json.
    assert baseline["substitution"]["f1"] == pytest.approx(0.12307692307692307)
    assert baseline["fp_per_100_correct_words"] == pytest.approx(7.1021957169964764)
