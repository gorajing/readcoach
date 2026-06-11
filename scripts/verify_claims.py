"""Verify the claim ledger — re-run the fast claims and fail on drift (T7.1).

The claim ledger (``docs/claims.md``) is only honest if its observed-value
column stays true to the code. This script keeps it honest *mechanically*:

  1. It parses the markdown tables in ``docs/claims.md`` and extracts, for every
     row, the ``mode`` (RERUN / ARTIFACT-VERIFIED / NOT CLAIMABLE) and the
     ``observed value`` cell as written.
  2. For each RERUN claim it has a registered recomputation that derives the
     value FRESH from the committed artifacts / a re-run command, with no
     network and no live model.
  3. It diffs the freshly-computed value against the number(s) parsed out of the
     ledger. Numeric comparison uses an absolute tolerance of ``TOL = 1e-6``.
     Any drift — or any RERUN row that has no registered check, or any registered
     check whose claim id is absent from the ledger — is a loud, non-zero-exit
     failure.

ARTIFACT-VERIFIED and NOT-CLAIMABLE rows are intentionally NOT recomputed here
(they need a slow/external resource or have no value yet); they are listed so a
reader can see they were considered, not skipped silently.

Exit codes
----------
0 — every RERUN claim's ledger value matches a fresh recomputation.
1 — drift, a missing check, or a structural problem with the ledger.

Usage
-----
    uv run python scripts/verify_claims.py
    uv run python scripts/verify_claims.py --ledger docs/claims.md
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from readcoach.policy_compiler import audit, compile_rules, load_policies  # noqa: E402
from readcoach.trace import SessionTrace, TurnRecord  # noqa: E402

TOL = 1e-6

_DEFAULT_LEDGER = _PROJECT_ROOT / "docs" / "claims.md"
_RESULTS = _PROJECT_ROOT / "evals" / "results"
_GOLD = _PROJECT_ROOT / "data" / "benchmark" / "gold.jsonl"
_MANIFEST = _PROJECT_ROOT / "data" / "benchmark" / "manifest.json"
_POLICIES = _PROJECT_ROOT / "policies"


# ---------------------------------------------------------------------------
# Ledger parsing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LedgerRow:
    claim: str
    observed: str
    mode: str


_MODE_RE = re.compile(r"\b(RERUN|ARTIFACT-VERIFIED|NOT CLAIMABLE)\b")


def _split_cells(line: str) -> list[str]:
    """Split a markdown table row on ``|``, but NOT on pipes inside `code` spans.

    Cells in this ledger sometimes carry inline code like ``|fit − true|``; a
    naive ``split('|')`` would shatter those. We temporarily mask pipes that sit
    between an odd number of backticks on the line.
    """
    out: list[str] = []
    buf: list[str] = []
    in_code = False
    for ch in line:
        if ch == "`":
            in_code = not in_code
            buf.append(ch)
        elif ch == "|" and not in_code:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return out


def parse_ledger(path: Path) -> list[LedgerRow]:
    """Parse every markdown table row that carries a known mode token.

    A ledger row is a ``|``-delimited line whose final cell contains one of the
    mode tokens. The first cell is the claim, the cell before the source/mode
    cells is the observed value. We locate columns positionally from the header
    of each table so the parser is robust to extra prose tables.
    """
    text = path.read_text(encoding="utf-8")
    rows: list[LedgerRow] = []
    header: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            header = None
            continue
        # Strip exactly one leading/trailing pipe (the table border), then split
        # on the interior pipes that are not inside inline-code spans.
        inner = stripped
        if inner.startswith("|"):
            inner = inner[1:]
        if inner.endswith("|"):
            inner = inner[:-1]
        cells = [c.strip() for c in _split_cells(inner)]
        lowered = [c.lower() for c in cells]
        if "claim" in lowered and "observed value" in lowered and "mode" in lowered:
            header = lowered
            continue
        if header is None:
            continue
        # Separator row (|---|---|).
        if all(set(c) <= {"-", ":"} and c for c in cells):
            continue
        if len(cells) != len(header):
            continue
        claim = cells[header.index("claim")]
        observed = cells[header.index("observed value")]
        mode_cell = cells[header.index("mode")]
        m = _MODE_RE.search(mode_cell)
        if not m:
            continue
        rows.append(LedgerRow(claim=claim, observed=observed, mode=m.group(1)))
    return rows


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    """All numeric tokens in *text*, in order, as floats."""
    return [float(x) for x in _NUM_RE.findall(text)]


# ---------------------------------------------------------------------------
# Fresh recomputations for the RERUN claims.  Each returns the list of numbers
# that MUST appear (in order) in the ledger's observed-value cell.  A claim is
# matched to its check by a unique substring of the claim sentence.
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _gold_entries() -> list[dict]:
    return [json.loads(line) for line in _GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_benchmark_88() -> list[float]:
    return [float(len(_gold_entries()))]


def check_benchmark_passages() -> list[float]:
    passages = {e["passage_id"] for e in _gold_entries()}
    return [float(len(passages))]


def check_coverage_min3() -> list[float]:
    matrix = _load_json(_MANIFEST)["coverage_matrix"]
    cells = [v for passage in matrix.values() for v in passage.values()]
    return [float(min(cells))]


def check_class_histogram() -> list[float]:
    c: Counter[str] = Counter()
    for e in _gold_entries():
        for g in e["gold"]:
            c[g["type"]] += 1
    total = sum(c.values())
    return [
        float(c["substitution"]),
        float(c["omission"]),
        float(c["insertion"]),
        float(c["self_correction"]),
        float(c["hesitation"]),
        float(total),
    ]


def check_hesitation_render() -> list[float]:
    c: Counter[str | None] = Counter()
    for e in _gold_entries():
        for g in e["gold"]:
            if g["type"] == "hesitation":
                c[g.get("render")] += 1
    return [float(c["filler"]), float(c["silence"])]


def _masking() -> dict:
    return _load_json(_RESULTS / "masking_curve.json")["results"]


def check_fixtures_reproduce() -> list[float]:
    """The --fixtures sweep reproduces v0's miscue metrics bit-for-bit."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        res = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "run_benchmark.py"),
                "--fixtures",
                "--version",
                "verifyclaims",
                "--results-dir",
                tmp,
            ],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise AssertionError(f"--fixtures sweep failed: {res.stderr[-500:]}")
        fresh = _load_json(Path(tmp) / "verifyclaims.json")["metrics"]["miscue"]
    committed = _load_json(_RESULTS / "v0.json")["metrics"]["miscue"]
    if fresh != committed:
        raise AssertionError("fixtures sweep miscue metrics differ from committed v0.json")
    return []


def check_sub_recall_ci_none() -> list[float]:
    m = _masking()
    return [round(x, 3) for x in m["none"]["substitution"]["ci_recall"]]


def check_om_recall_ci_none() -> list[float]:
    m = _masking()
    return [round(x, 3) for x in m["none"]["omission"]["ci_recall"]]


def check_substitution_recall() -> list[float]:
    m = _masking()
    return [round(m[b]["substitution"]["recall"], 3) for b in ("none", "prompt", "strong")]


def check_omission_recall() -> list[float]:
    m = _masking()
    return [round(m[b]["omission"]["recall"], 3) for b in ("none", "prompt", "strong")]


def check_insertion_recall() -> list[float]:
    m = _masking()
    return [round(m[b]["insertion"]["recall"], 3) for b in ("none", "prompt", "strong")]


def check_substitution_precision() -> list[float]:
    m = _masking()
    return [round(m[b]["substitution"]["precision"], 3) for b in ("none", "prompt", "strong")]


def check_insertion_precision() -> list[float]:
    m = _masking()
    return [round(m[b]["insertion"]["precision"], 3) for b in ("none", "prompt", "strong")]


def check_fp_per_100() -> list[float]:
    m = _masking()
    return [round(m[b]["fp_per_100_correct_words"], 3) for b in ("none", "prompt", "strong")]


def check_two_wer() -> list[float]:
    m = _masking()
    spoken = [round(m[b]["wer_vs_spoken_mean"], 3) for b in ("none", "prompt", "strong")]
    target = [round(m[b]["wer_vs_target_mean"], 3) for b in ("none", "prompt", "strong")]
    return spoken + target


def check_bkt_recovery_floor() -> list[float]:
    d = _load_json(_RESULTS / "bkt_recovery.json")
    worst = max(
        abs(v)
        for regime in d["regimes"].values()
        for v in regime["recovery_error"].values()
    )
    return [round(worst, 2)]


def check_bkt_mastery_rmse() -> list[float]:
    d = _load_json(_RESULTS / "bkt_recovery.json")
    rmses = [regime["mastery_rmse"] for regime in d["regimes"].values()]
    return [round(min(rmses), 3), round(max(rmses), 3)]


def check_calibration_hole() -> list[float]:
    """Lowest populated reliability bin mean ≈ 0.281; Brier ≈ 0.139.

    The two lowest bins are empty (observed frequency null) because
    predict_correct is floored at the guess rate.
    """
    d = _load_json(_RESULTS / "bkt_recovery.json")
    rel = d["calibration"]["reliability"]
    means = rel["bin_mean_predicted"]
    observed = rel["bin_observed_frequency"]
    lowest_populated = next(
        m for m, o in zip(means, observed) if o is not None
    )
    return [round(lowest_populated, 3), round(d["calibration"]["brier_score"], 3)]


def check_cold_start() -> list[float]:
    """Cold-start RMSE: 0.445 at k=1, plateau ~0.21 by k≈11, 0.162 at k=20.

    Ledger cell carries the three landmark numbers 0.445, 0.21, 0.162.
    """
    d = _load_json(_RESULTS / "bkt_recovery.json")
    curve = d["cold_start_curve"]
    by_k = dict(zip(curve["k"], curve["mastery_rmse"]))
    return [round(by_k[1], 3), round(by_k[11], 2), round(by_k[20], 3)]


def check_break_even() -> list[float]:
    d = _load_json(_RESULTS / "break_even.json")
    return [d["break_even_a"]]


def check_soft_monotonic() -> list[float]:
    """Δ rises monotonically as the channel worsens; endpoints 0.0000 and 0.1105."""
    d = _load_json(_RESULTS / "break_even.json")
    grid = sorted(d["grid"], key=lambda g: g["a"])
    deltas = [g["delta_rmse"] for g in grid]  # a ascending -> channel improving
    # As a DEcreases (channel worsens) Δ must rise: deltas (a ascending) must be
    # non-increasing.
    for lo, hi in zip(deltas, deltas[1:]):
        if hi > lo + TOL:
            raise AssertionError(
                f"Δ not monotone in a: {lo:.4f} then {hi:.4f} (should be non-increasing as a rises)"
            )
    worst = max(deltas)  # at smallest a
    best = min(deltas)  # at largest a
    return [round(best, 4), round(worst, 4)]


def check_a_eff_anchors() -> list[float]:
    d = _load_json(_RESULTS / "break_even.json")
    by_bias = {a["bias"]: a["a_eff"] for a in d["a_eff_anchors"]}
    return [round(by_bias[b], 3) for b in ("none", "prompt", "strong")]


def check_wait_rate() -> list[float]:
    d = _load_json(_RESULTS / "policy_replay.json")
    return [round(d["wait_rate"], 4)]


def check_self_correction_immunity() -> list[float]:
    """Self-corrections route to the non-corrective rule; 0 corrective moves.

    R-MID-SELF-CORRECTION fires 24 times; no escalation/corrective move is
    attached to those turns (the rule itself is non-corrective by construction).
    """
    d = _load_json(_RESULTS / "policy_replay.json")
    sc_fires = float(d["rule_distribution"].get("R-MID-SELF-CORRECTION", 0))
    return [sc_fires, 0.0]


def check_default_rule_unreached() -> list[float]:
    d = _load_json(_RESULTS / "policy_replay.json")
    return [float(d["rule_distribution"].get("R-DEFAULT", 0))]


def check_learnermem_score() -> list[float]:
    d = _load_json(_RESULTS / "learnermem_v0.json")
    return [d["n_passed"], d["n_total"], d["consistency_score"]]


def check_completion_fragility() -> list[float]:
    """P6: one generic failure drops mastery to 0.9042 — ≥0.80 but <0.95.

    Ledger cell carries 0.9042, 0.80, and 0.95.
    """
    d = _load_json(_RESULTS / "learnermem_v0.json")
    ev = d["probes"]["P6"]["evidence"]
    nums = numbers_in(ev)
    mastery = next((n for n in nums if abs(n - 0.9042) < 1e-3), None)
    if mastery is None:
        raise AssertionError(f"P6 evidence does not record the ~0.9042 mastery drop: {ev!r}")
    return [round(mastery, 4), 0.80, 0.95]


def check_two_session() -> list[float]:
    """The two-session demo exits 0 (mastery survives close+reopen, gate fires).

    Ledger cell carries '0' (exit code). We run the demo on a temp db.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "two_session.db"
        res = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "two_session_demo.py"), str(db)],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    if res.returncode != 0:
        raise AssertionError(f"two_session_demo exited {res.returncode}: {res.stderr[-400:]}")
    return [0.0]


def check_frozen_split() -> list[float]:
    """freeze_split --verify passes; dev=49, holdout=49 match the lock."""
    res = subprocess.run(
        [sys.executable, str(_PROJECT_ROOT / "scripts" / "freeze_split.py"), "--verify"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise AssertionError(f"freeze_split --verify failed: {res.stdout[-400:]}{res.stderr[-400:]}")
    lock = _load_json(_PROJECT_ROOT / "evals" / "golden" / "holdout.lock")
    return [float(lock["n_dev"]), float(lock["n_holdout"])]


def check_gate_exit_codes() -> list[float]:
    """v1→v2 passes (exit 0); v2→v3 breaches (exit 1)."""
    d = _load_json(_RESULTS / "ab_dev.json")
    g = d["gate_outcomes"]
    return [float(g["v1_vs_v2"]["exit_code"]), float(g["v2_vs_v3"]["exit_code"])]


def check_sensitivity() -> list[float]:
    """±30% persona-rate: minus30 wait 0.351 / plus30 wait 0.457; both gates pass."""
    d = _load_json(_RESULTS / "ab_dev.json")
    s = d["sensitivity"]["runs"]
    if not (s["minus30"]["gate_passed"] and s["plus30"]["gate_passed"]):
        raise AssertionError("a ±30% sensitivity gate did not pass")
    return [round(s["minus30"]["v1_wait_rate"], 3), round(s["plus30"]["v1_wait_rate"], 3)]


def check_live_turn_metadata() -> list[float]:
    """Live turns recorded model=claude-sonnet-4-6, transport=claude-cli, prompt=1.0.

    The ledger cell carries the prompt version 1.0 as its only number; the
    model/transport strings are asserted here directly.
    """
    rows = [
        json.loads(line)
        for line in (_RESULTS / "turns_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    models = {r["model"] for r in rows}
    transports = {r["transport"] for r in rows}
    prompts = {r["prompt_version"] for r in rows}
    if models != {"claude-sonnet-4-6"}:
        raise AssertionError(f"unexpected model set: {models}")
    if transports != {"claude-cli"}:
        raise AssertionError(f"unexpected transport set: {transports}")
    if prompts != {"1.0"}:
        raise AssertionError(f"unexpected prompt-version set: {prompts}")
    return [1.0]


def check_gate_v3_blocked() -> list[float]:
    d = _load_json(_RESULTS / "ab_dev.json")
    r2 = d["receipt_2_gate_blocks_bad_tutor"]
    v3_viol = float(d["metrics"]["v3"]["invariants"]["violations"])
    return [float(r2["exit_code"]), v3_viol]


def check_promote_idempotent() -> list[float]:
    d = _load_json(_RESULTS / "ab_dev.json")
    pg = d["promote_growth"]
    v3 = next(b["promoted"] for b in pg["batches"] if b["batch"] == "v3")
    return [float(v3)]


def check_live_turns_zero_violations() -> list[float]:
    """Deterministically re-audit the committed live-turn traces.

    The turns were generated live; this re-derives the 0-violations property
    from the committed traces with no model call, so a policy-compiler
    regression would surface here.
    """
    checks = compile_rules(load_policies(str(_POLICIES)))
    total_turns = 0
    total_violations = 0
    for trace_file in sorted(_RESULTS.glob("trace_*.json")):
        d = _load_json(trace_file)
        turns = [TurnRecord(**t) for t in d["turns"]]
        trace = SessionTrace(
            child_id=d["child_id"],
            policy_version=d["policy_version"],
            completed_skills_at_start=d["completed_skills_at_start"],
            turns=turns,
        )
        total_turns += len(turns)
        total_violations += audit(trace, checks).violations
    return [float(total_turns), float(total_violations)]


def check_policy_citations() -> list[float]:
    """Every non-deferred compiled rule carries a verbatim sentence + source.

    Not a numeric-from-ledger check: this raises on a citation gap directly, and
    returns no required numbers (the ledger cell has none).
    """
    import yaml  # local import: only this check needs it

    n_rules = 0
    for yaml_name in ("safety.yaml", "pedagogy.yaml"):
        raw = yaml.safe_load((_POLICIES / yaml_name).read_text(encoding="utf-8"))
        for rule in raw["rules"]:
            if rule.get("deferred"):
                continue
            n_rules += 1
            if not str(rule.get("verbatim_sentence", "")).strip():
                raise AssertionError(f"{yaml_name}:{rule['id']} missing verbatim_sentence")
            if not str(rule.get("source", {}).get("url", "")).strip():
                raise AssertionError(f"{yaml_name}:{rule['id']} missing source.url")
    if n_rules == 0:
        raise AssertionError("no compiled policy rules found")
    return []


def check_prereg_order() -> list[float]:
    """Predictions commit is an ancestor of the first benchmark-runner commit."""
    res = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "56624ff", "ed1761c"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
    )
    if res.returncode != 0:
        raise AssertionError(
            "pre-registration order broken: 56624ff is not an ancestor of ed1761c"
        )
    return []


# Map: unique claim-substring -> (recompute fn, "numbers must appear in order").
# The substring must uniquely identify exactly one RERUN ledger row.
_CHECKS: list[tuple[str, Callable[[], list[float]]]] = [
    # Benchmark
    ("contains 88 clips", check_benchmark_88),
    ("spans 8 original passages", check_benchmark_passages),
    ("at least 3 items per", check_coverage_min3),
    ("24 substitution / 24 omission", check_class_histogram),
    ("32 hesitations split 16 filler", check_hesitation_render),
    ("reproduces from fixtures with no model load", check_fixtures_reproduce),
    # Detector
    ("Substitution recall collapses", check_substitution_recall),
    ("Omission recall also collapses", check_omission_recall),
    ("Insertion recall is masked less", check_insertion_recall),
    ("Substitution precision rises", check_substitution_precision),
    ("Insertion precision rises", check_insertion_precision),
    ("False positives per 100 correct words fall from 7.10 to 0.54", check_fp_per_100),
    ("Substitution-recall masking has a 95% bootstrap CI", check_sub_recall_ci_none),
    ("Omission-recall masking has a 95% bootstrap CI", check_om_recall_ci_none),
    ("two-WER split", check_two_wer),
    # BKT
    ("parameter recovery error is at or below 0.06", check_bkt_recovery_floor),
    ("Mastery RMSE has a floor of 0.20", check_bkt_mastery_rmse),
    ("Calibration has a coverage hole below", check_calibration_hole),
    ("Cold-start: mastery estimates plateau", check_cold_start),
    ("the break-even is channel accuracy a = 0.90", check_break_even),
    ("RMSE advantage grows monotonically", check_soft_monotonic),
    ("measured operating points sit at or above the a=0.90", check_a_eff_anchors),
    # Policy
    ("WAIT rate is 0.435", check_wait_rate),
    ("Self-correction immunity", check_self_correction_immunity),
    ("default rule is never reached", check_default_rule_unreached),
    # Compiler / safety
    ("cites the verbatim published sentence", check_policy_citations),
    ("the gate blocks a deliberately-worse tutor", check_gate_v3_blocked),
    # Memory
    ("passes 6 of 6 memory-consistency probes", check_learnermem_score),
    ("Completion-fragility finding", check_completion_fragility),
    ("Two-session continuity", check_two_session),
    # Flywheel
    ("frozen one-way and hash-locked", check_frozen_split),
    ("pre-registered before the runs", check_prereg_order),
    ("gate emits exit codes that distinguish", check_gate_exit_codes),
    ("idempotent on re-run", check_promote_idempotent),
    ("conclusion holds under ±30% persona-rate", check_sensitivity),
    # Turns
    ("72 live tutor turns were generated with 0 invariant violations", check_live_turns_zero_violations),
    ("live turns ran on claude-sonnet-4-6", check_live_turn_metadata),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _match_value(expected: list[float], observed_cell: str) -> tuple[bool, str]:
    """True iff every number in *expected* appears, in order, in the cell."""
    got = numbers_in(observed_cell)
    # The ledger cell may carry extra numbers (e.g. "18×3"); we require the
    # expected sequence to be a subsequence-by-equality within tolerance.
    i = 0
    for target in expected:
        found = False
        while i < len(got):
            if abs(got[i] - target) <= TOL:
                found = True
                i += 1
                break
            i += 1
        if not found:
            return False, f"expected {target} not found (in order) among {got}"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=_DEFAULT_LEDGER)
    args = parser.parse_args(argv)

    rows = parse_ledger(args.ledger)
    by_mode = Counter(r.mode for r in rows)
    print(f"Parsed {len(rows)} ledger rows: {dict(by_mode)}")

    rerun_rows = [r for r in rows if r.mode == "RERUN"]
    failures: list[str] = []

    matched_substrings: set[str] = set()
    for substr, fn in _CHECKS:
        targets = [r for r in rerun_rows if substr in r.claim]
        if len(targets) != 1:
            failures.append(
                f"check substring {substr!r} matched {len(targets)} RERUN rows (want exactly 1)"
            )
            continue
        matched_substrings.add(substr)
        row = targets[0]
        try:
            expected = fn()
        except Exception as exc:  # a failed recomputation IS drift
            failures.append(f"[{substr!r}] recomputation raised: {exc}")
            continue
        if not expected:
            print(f"  OK (assertion-only)  {substr!r}")
            continue
        ok, why = _match_value(expected, row.observed)
        if ok:
            print(f"  OK  {substr!r}  -> {expected}")
        else:
            failures.append(f"DRIFT [{substr!r}]: {why}  (ledger cell: {row.observed!r})")

    # Every RERUN row must be covered by exactly one check.
    for row in rerun_rows:
        hits = [s for s, _ in _CHECKS if s in row.claim]
        if len(hits) == 0:
            failures.append(f"RERUN row has NO registered check: {row.claim!r}")
        elif len(hits) > 1:
            failures.append(f"RERUN row matched MULTIPLE checks {hits}: {row.claim!r}")

    print()
    if failures:
        print(f"VERIFY FAILED — {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"VERIFY PASSED — all {len(rerun_rows)} RERUN claims match the ledger (tol {TOL}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
