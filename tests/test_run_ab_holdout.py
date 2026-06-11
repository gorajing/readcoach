"""T5.4 — the HOLDOUT A/B runner (scripts/run_ab_holdout.py), mocked transports.

Pins the pre-registered plan (docs/ab_holdout_plan.md) end to end WITHOUT any live
CLI call:

  * Phase 0 — holdout hash-check aborts on a corpus/lock mismatch; eligibility
    loading aborts when judge_validation.json is absent and excludes ineligible
    dims when present.
  * Phase 2 — sampling is deterministic, 20/20/20 persona-stratified, and a pure
    function of the REPLAY ONLY (independent of any judged data).
  * Phases 3 & 4 — both checkpoint phases resume correctly after an interruption
    (a partial workfile is honored; only missing units are recomputed).
  * Phase 5 — the adjudication mapping yields all four outcomes (CONFIRMED,
    PARTIAL, MISSED, UNADJUDICABLE) on synthetic score sets, the honest-pairing
    note appears in the output, and the prediction-#5 doc append is gated behind
    the live flag (never fires in a mocked run).

Transports are stubbed (no claude/codex). The judge step is injected via a fake
judge_fn so no CodexCliTransport is constructed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (_PROJECT_ROOT, _PROJECT_ROOT / "src", _PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_ab_holdout", _PROJECT_ROOT / "scripts" / "run_ab_holdout.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_ab_holdout"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubVerbalizer:
    """Deterministic verbalizer — returns a line keyed on the move (no CLI)."""

    transport_meta = {"transport": "stub", "model": "stub"}

    def __init__(self) -> None:
        self.calls = 0

    def verbalize(self, action, ctx_summary, prompt_version, *, is_ai_reminder=False):  # noqa: ANN001
        self.calls += 1
        return f"[{action.move}] line for {ctx_summary['miscue']}"


@dataclass
class _StubVerdict:
    dimension: str
    score: int
    passing: bool
    issues: list
    rationale: str
    model_meta: dict


def _make_judge_fn(score_table):
    """Build a fake judge_fn that scores by (version_inferred_from_uid, dimension).

    score_table maps dimension -> {"v1": int, "v2": int}.  The version is read from
    the move-less turn dict's utterance prefix is unavailable, so we encode version
    via a closure over the sample (turn dict carries 'move'; we instead pass a
    uid->version map). Simpler: the fake reads a module-level marker injected on the
    turn dict by the caller. Here we accept a uid->version resolver.
    """

    def judge_fn(turn, dimension, transport=None):  # noqa: ANN001
        version = turn["__version__"]
        score = score_table[dimension][version]
        passing = score >= 4
        return _StubVerdict(
            dimension=dimension,
            score=score,
            passing=passing,
            issues=[] if passing else ["below threshold"],
            rationale="stub",
            model_meta={"model": "stub"},
        )

    return judge_fn


# ---------------------------------------------------------------------------
# Phase 0 — freeze hash check + eligibility loading
# ---------------------------------------------------------------------------


def _write_lock(tmp_path: Path, corpus: Path, *, sha: str | None = None) -> Path:
    import hashlib

    real_sha = hashlib.sha256(corpus.read_text(encoding="utf-8").encode()).hexdigest()
    lock = {"files": {corpus.name: {"sha256": sha or real_sha, "n_items": 1}}}
    lock_path = tmp_path / "holdout.lock"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return lock_path


def test_phase0_hash_check_passes_on_match(runner, tmp_path):
    corpus = tmp_path / "persona_sessions_holdout.jsonl"
    corpus.write_text('{"id": "x"}\n', encoding="utf-8")
    lock = _write_lock(tmp_path, corpus)
    # Must not raise.
    out = runner.verify_holdout_lock(lock_file=lock, holdout_file=corpus)
    assert corpus.name in out["files"]


def test_phase0_hash_check_aborts_on_mismatch(runner, tmp_path):
    corpus = tmp_path / "persona_sessions_holdout.jsonl"
    corpus.write_text('{"id": "x"}\n', encoding="utf-8")
    lock = _write_lock(tmp_path, corpus, sha="deadbeef" * 8)
    with pytest.raises(runner.HoldoutAbort, match="HASH MISMATCH"):
        runner.verify_holdout_lock(lock_file=lock, holdout_file=corpus)


def test_phase0_eligibility_absent_file_aborts(runner, tmp_path):
    missing = tmp_path / "judge_validation.json"
    with pytest.raises(runner.HoldoutAbort, match="validation has not run"):
        runner.load_eligible_dims(validation_file=missing)


def test_phase0_eligibility_excludes_ineligible_dims(runner, tmp_path):
    vf = tmp_path / "judge_validation.json"
    vf.write_text(
        json.dumps(
            {
                "dimensions": [
                    {"dimension": "guidance", "gate_eligible": True},
                    {"dimension": "actionability", "gate_eligible": False},
                    {"dimension": "icap", "gate_eligible": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    eligible, _ = runner.load_eligible_dims(validation_file=vf)
    assert eligible == {"guidance", "icap"}
    assert "actionability" not in eligible


# ---------------------------------------------------------------------------
# Phase 2 — sampling determinism, stratification, replay-only dependence
# ---------------------------------------------------------------------------


def _fake_decisions(runner, n_per_persona=40, version="v1"):
    """Build a synthetic decision population (no replay needed for sampling tests)."""
    out = []
    for persona in ("pa", "pb", "pc"):
        for i in range(n_per_persona):
            out.append(
                runner.DecisionTurn(
                    turn_uid=f"{version}__{persona}__s0__t{i:03d}",
                    version=version,
                    persona_id=persona,
                    session_id=f"{persona}__s0",
                    turn_index=i,
                    move="WAIT",
                    miscue_type="substitution",
                    at_page_end=False,
                    target_word="cat",
                    hint_level=None,
                    skill_id=None,
                    served_reason=None,
                )
            )
    return out


def test_phase2_sampling_is_stratified_20_per_persona(runner):
    sample = runner.sample_turns(_fake_decisions(runner), per_persona=20)
    from collections import Counter

    counts = Counter(d.persona_id for d in sample)
    assert counts == {"pa": 20, "pb": 20, "pc": 20}
    assert len(sample) == 60


def test_phase2_sampling_is_deterministic(runner):
    decisions = _fake_decisions(runner)
    a = [d.turn_uid for d in runner.sample_turns(decisions, per_persona=20)]
    b = [d.turn_uid for d in runner.sample_turns(decisions, per_persona=20)]
    assert a == b


def test_phase2_sampling_depends_on_replay_only_not_order(runner):
    """The sample is a function of the decision SET, not its input order.

    Shuffling the input population must not change the sampled uids (the per-stratum
    pool is sorted by turn_uid before the seeded draw)."""
    import random as _r

    decisions = _fake_decisions(runner)
    shuffled = decisions[:]
    _r.Random(99).shuffle(shuffled)
    a = sorted(d.turn_uid for d in runner.sample_turns(decisions, per_persona=20))
    b = sorted(d.turn_uid for d in runner.sample_turns(shuffled, per_persona=20))
    assert a == b


def test_phase2_sampling_aborts_when_stratum_too_small(runner):
    with pytest.raises(runner.HoldoutAbort, match="cannot be honored"):
        runner.sample_turns(_fake_decisions(runner, n_per_persona=5), per_persona=20)


# ---------------------------------------------------------------------------
# Phases 3 & 4 — checkpoint resume
# ---------------------------------------------------------------------------


def _small_sample(runner, n=3):
    return [
        runner.DecisionTurn(
            turn_uid=f"v1__s__t{i:03d}",
            version="v1",
            persona_id="pa",
            session_id="s",
            turn_index=i,
            move="WAIT",
            miscue_type="substitution",
            at_page_end=False,
            target_word="cat",
            hint_level=None,
            skill_id=None,
            served_reason=None,
        )
        for i in range(n)
    ]


def test_phase3_verbalize_resumes_from_partial_workfile(runner, tmp_path):
    sample = _small_sample(runner, n=3)
    wf = tmp_path / ".verbalize_work.jsonl"
    # Pre-seed one completed utterance.
    wf.write_text(
        json.dumps({"turn_uid": "v1__s__t000", "utterance": "PRESEEDED"}) + "\n",
        encoding="utf-8",
    )
    verbalizer = _StubVerbalizer()
    out = runner.verbalize_phase(sample, verbalizer, wf)
    # Only the 2 missing turns were verbalized; the pre-seeded one was reused.
    assert verbalizer.calls == 2
    assert out["v1__s__t000"] == "PRESEEDED"
    assert out["v1__s__t001"].startswith("[WAIT]")
    assert len(out) == 3


def test_phase3_verbalize_aborts_on_transport_failure(runner, tmp_path):
    class _Boom:
        def verbalize(self, *a, **k):  # noqa: ANN001
            raise RuntimeError("cli down")

    with pytest.raises(runner.HoldoutAbort, match="verbalization failed"):
        runner.verbalize_phase(
            _small_sample(runner, n=1), _Boom(), tmp_path / ".v.jsonl"
        )


def test_phase3_malformed_workfile_aborts(runner, tmp_path):
    wf = tmp_path / ".verbalize_work.jsonl"
    wf.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(runner.HoldoutAbort, match="MALFORMED"):
        runner.verbalize_phase(_small_sample(runner, n=1), _StubVerbalizer(), wf)


def test_phase4_judge_resumes_from_partial_workfile(runner, tmp_path):
    sample = _small_sample(runner, n=2)
    utterances = {d.turn_uid: "u" for d in sample}
    wf = tmp_path / ".judge_work.jsonl"
    # Pre-seed one (turn, dim) verdict.
    wf.write_text(
        json.dumps(
            {
                "turn_uid": "v1__s__t000",
                "dimension": "guidance",
                "score": 5,
                "passing": True,
                "persona_id": "pa",
                "version": "v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = {"n": 0}

    def judge_fn(turn, dimension, transport=None):  # noqa: ANN001
        calls["n"] += 1
        return _StubVerdict("guidance", 4, True, [], "x", {})

    verdicts = runner.judge_phase(
        sample, utterances, ["guidance"], transport=None, workfile=wf, judge_fn=judge_fn
    )
    # 2 turns × 1 dim = 2 pairs; 1 pre-seeded → only 1 new judge call.
    assert calls["n"] == 1
    assert len(verdicts) == 2
    assert verdicts[("v1__s__t000", "guidance")]["score"] == 5


# ---------------------------------------------------------------------------
# Phase 5 — adjudication mapping (all four outcomes) + pairing note
# ---------------------------------------------------------------------------


def _verdicts_from_means(g_v1, g_v2, a_v1, a_v2, *, n=60):
    """Build a verdicts dict where each (dim, version) has constant score.

    Constant per-version scores make the diff-of-means CI degenerate to a point
    (lo == hi == mean diff), so a strictly-positive diff has a CI strictly above
    zero — exactly the 'beats' condition.  This lets us drive each outcome.
    """
    verdicts = {}
    spec = {"guidance": (g_v1, g_v2), "actionability": (a_v1, a_v2)}
    for dim, (s1, s2) in spec.items():
        for ver, score in (("v1", s1), ("v2", s2)):
            for i in range(n):
                uid = f"{ver}__s__t{i:03d}"
                verdicts[(uid, dim)] = {
                    "turn_uid": uid,
                    "dimension": dim,
                    "score": score,
                    "passing": score >= 4,
                    "version": ver,
                    "persona_id": "pa",
                }
    return verdicts


def test_phase5_confirmed_when_both_dims_beat(runner):
    verdicts = _verdicts_from_means(3, 5, 3, 5)  # v2 strictly higher on both
    out = runner.analyze(verdicts, ["guidance", "actionability"], {"guidance", "actionability"})
    assert out["prediction_5"]["verdict"] == "CONFIRMED"
    assert out["per_dimension"]["guidance"]["v2_beats_v1"]
    assert out["per_dimension"]["actionability"]["v2_beats_v1"]


def test_phase5_partial_when_one_dim_beats(runner):
    verdicts = _verdicts_from_means(3, 5, 4, 4)  # guidance beats, actionability tie
    out = runner.analyze(verdicts, ["guidance", "actionability"], {"guidance", "actionability"})
    assert out["prediction_5"]["verdict"] == "PARTIAL"


def test_phase5_missed_when_neither_dim_beats(runner):
    verdicts = _verdicts_from_means(4, 4, 5, 3)  # tie + v2 worse
    out = runner.analyze(verdicts, ["guidance", "actionability"], {"guidance", "actionability"})
    assert out["prediction_5"]["verdict"] == "MISSED"


def test_phase5_unadjudicable_when_a_dim_is_ineligible(runner):
    verdicts = _verdicts_from_means(3, 5, 3, 5)
    # actionability NOT in the eligible set → that part of pred #5 is unadjudicable.
    out = runner.analyze(verdicts, ["guidance"], {"guidance"})
    assert out["prediction_5"]["verdict"] == "UNADJUDICABLE"
    assert "actionability" in out["prediction_5"]["finding"]


def test_phase5_honest_pairing_note_present(runner):
    verdicts = _verdicts_from_means(3, 5, 3, 5)
    out = runner.analyze(verdicts, ["guidance", "actionability"], {"guidance", "actionability"})
    assert out["pairing"] == "independent"
    assert "IMPOSSIBLE" in out["pairing_note"]
    assert "do NOT silently pretend" in out["pairing_note"].replace("\n", " ")


def test_phase5_icap_reported_not_adjudicated(runner):
    """icap, if judged, is reported per-dimension but never affects the verdict."""
    verdicts = _verdicts_from_means(3, 5, 3, 5)
    # Add icap with v2 WORSE — must not change the CONFIRMED verdict.
    for ver, score in (("v1", 5), ("v2", 2)):
        for i in range(60):
            uid = f"{ver}__s__t{i:03d}"
            verdicts[(uid, "icap")] = {
                "turn_uid": uid, "dimension": "icap", "score": score,
                "passing": score >= 4, "version": ver, "persona_id": "pa",
            }
    out = runner.analyze(
        verdicts, ["guidance", "actionability", "icap"],
        {"guidance", "actionability", "icap"},
    )
    assert out["prediction_5"]["verdict"] == "CONFIRMED"
    assert "icap" in out["per_dimension"]
    assert out["per_dimension"]["icap"]["note"].startswith("icap is reported")


# ---------------------------------------------------------------------------
# Doc append gated behind the live run
# ---------------------------------------------------------------------------


def test_doc_append_writes_verdict_block(runner, tmp_path):
    doc = tmp_path / "results_vs_predictions.md"
    doc.write_text("# Results\n\n## Prediction 5\n\n**Status: pending**\n", encoding="utf-8")
    verdicts = _verdicts_from_means(3, 5, 3, 5)
    analysis = runner.analyze(
        verdicts, ["guidance", "actionability"], {"guidance", "actionability"}
    )
    runner.append_prediction5_verdict(analysis, results_doc=doc)
    text = doc.read_text(encoding="utf-8")
    assert "Verdict: CONFIRMED" in text
    assert "holdout-prediction-5-verdict" in text
    # Idempotent: a second append does not duplicate the block.
    runner.append_prediction5_verdict(analysis, results_doc=doc)
    assert text.count("holdout-prediction-5-verdict") == 1
    assert (
        doc.read_text(encoding="utf-8").count("holdout-prediction-5-verdict") == 1
    )


def test_main_refuses_without_preregistered_flag(runner, capsys):
    rc = runner.main([])
    assert rc == 2
    assert "i-am-the-preregistered-run" in capsys.readouterr().err


def test_main_refuses_without_live_flag(runner, capsys):
    rc = runner.main(["--i-am-the-preregistered-run"])
    assert rc == 2
    assert "REFUSING to run without --live" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Full mocked run() over the REAL holdout corpus (no doc append; no live CLI)
# ---------------------------------------------------------------------------


def test_full_run_mocked_end_to_end(runner, tmp_path, monkeypatch):
    """run() over the real frozen holdout with stubbed transports + injected judge.

    Exercises Phase 0→5 wired together: real replay/aggregation/sampling, stubbed
    verbalization, injected judge_fn, analysis. Verifies the deterministic gate is
    evaluated, 60+60 turns are sampled, the manifest is written, and the prediction
    -#5 verdict is produced WITHOUT any doc append (live=False)."""
    # Eligibility: make both pred-5 dims eligible so the verdict is adjudicated.
    vf = tmp_path / "judge_validation.json"
    vf.write_text(
        json.dumps(
            {
                "gate_conditions": {"kappa_point_estimate_gte": 0.4, "n_gte": 30},
                "dimensions": [
                    {"dimension": "guidance", "gate_eligible": True},
                    {"dimension": "actionability", "gate_eligible": True},
                    {"dimension": "icap", "gate_eligible": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_VALIDATION_FILE", vf)

    # Judge: v2 strictly higher on both eligible dims → CONFIRMED.
    score_table = {"guidance": {"v1": 3, "v2": 5}, "actionability": {"v1": 3, "v2": 5}}

    # Stub verbalize_phase so each turn's "utterance" IS its uid (carries the
    # version). The injected judge_fn then reads the version straight off it — no
    # claude/codex CLI is ever touched.
    def verbalize_phase_stub(sample, verbalizer, workfile):  # noqa: ANN001
        return {d.turn_uid: d.turn_uid for d in sample}

    monkeypatch.setattr(runner, "verbalize_phase", verbalize_phase_stub)

    def judge_fn_from_uid(turn, dimension, transport=None):  # noqa: ANN001
        version = "v2" if turn["utterance"].startswith("v2__") else "v1"
        score = score_table[dimension][version]
        return _StubVerdict(dimension, score, score >= 4, [] if score >= 4 else ["x"], "s", {})

    payload = runner.run(
        verbalizer=_StubVerbalizer(),
        judge_transport=None,
        workdir=tmp_path / "work",
        live=False,
        judge_fn=judge_fn_from_uid,
        write_reports=False,
    )

    assert payload["split"] == "holdout"
    assert payload["deterministic"]["n_sessions"] == 49
    assert payload["sampling"]["n_per_version"] == 60
    assert payload["judge_eligibility"]["judged_dims"] == ["guidance", "actionability"]
    assert payload["judged"]["prediction_5"]["verdict"] == "CONFIRMED"
    assert payload["judged"]["pairing"] == "independent"
    # Manifest written to the workdir for auditability.
    assert (tmp_path / "work" / "sample_manifest.json").exists()
    # No doc append happened (live=False) — the real predictions doc is untouched.
    real_doc = _PROJECT_ROOT / "docs" / "results_vs_predictions.md"
    assert "holdout-prediction-5-verdict" not in real_doc.read_text(encoding="utf-8")
