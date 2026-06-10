"""Validate the LLM judge against human labels — T5.2.

Produces a per-dimension agreement table (TPR, TNR, Cohen's kappa with
bootstrap CIs) and writes evals/results/judge_validation.json.

## Expected input formats

### Human labels CSV (--labels)

Minimal format — one row per (turn, dimension) label:

    turn_id,dimension,human_score,human_passing,rater_initials
    t001,guidance,4,y,JC
    t001,actionability,3,n,JC
    ...

Columns:
  turn_id         — unique identifier matching the judged-verdicts JSONL
  dimension       — one of: guidance, actionability, icap
  human_score     — integer 1–5 (for reference; binary-only analysis uses human_passing)
  human_passing   — "y" / "yes" / "1" / "true"  → True
                    "n" / "no"  / "0" / "false" → False
                    (case-insensitive)
  rater_initials  — e.g. "JC" (kept for provenance; not used in stats)

### Judged verdicts JSONL (--verdicts)

One JSON object per line, each with at minimum:

    {"turn_id": "t001", "dimension": "guidance", "passing": true, ...}

This is the output format of evals/judge.py judge_trace written to JSONL.
Extra fields (score, rationale, model_meta) are ignored.

## Gate eligibility

A dimension is gate_eligible when BOTH conditions hold:
  1. kappa point estimate >= KAPPA_FLOOR (0.4, Landis & Koch 1977 moderate)
  2. n (matched labels) >= 30

Below-floor or below-n dimensions are marked gate_eligible: false and
reported PROMINENTLY in the output — this is a finding, not a failure.

## Silent-join policy

This script performs an inner join on (turn_id, dimension).  Any row
present in one file but not the other is an ERROR and is listed explicitly.
Use --allow-unmatched to override (prints counts prominently, exits 0).

## n < 10 refusal

If ANY dimension has fewer than 10 matched items, the script refuses to run.
Fewer than 10 items is too few to produce meaningful validation statistics.

## Usage

    uv run python scripts/validate_judge.py \\
        --labels  evals/human_labels.csv \\
        --verdicts evals/results/judged_turns.jsonl \\
        [--output evals/results/judge_validation.json] \\
        [--allow-unmatched]

Currently cannot run successfully — no human labels exist yet (labeling
session not yet conducted; n=60 target × 3 dimensions = 180 label rows).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Add project root to path so evals.stats is importable when run as a script.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from evals.stats import KAPPA_FLOOR, bootstrap_ci, cohens_kappa, tpr_tnr  # noqa: E402

# Minimum n per dimension to proceed.
_MIN_N: int = 10

# Minimum n per dimension for gate eligibility (documented gate condition).
_GATE_N: int = 30

# Bootstrap parameters.
_N_BOOT: int = 2000
_BOOT_SEED: int = 42
_CI: float = 0.95


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_human_passing(raw: str) -> bool:
    """Parse human_passing column from CSV."""
    v = raw.strip().lower()
    if v in ("y", "yes", "1", "true"):
        return True
    if v in ("n", "no", "0", "false"):
        return False
    raise ValueError(
        f"Unrecognised human_passing value {raw!r}; "
        f"expected y/yes/1/true or n/no/0/false (case-insensitive)"
    )


def _load_labels(path: Path) -> dict[tuple[str, str], bool]:
    """Load human labels CSV → {(turn_id, dimension): human_passing}.

    Raises
    ------
    SystemExit
        On missing file, bad format, or duplicate (turn_id, dimension) pairs.
    """
    if not path.exists():
        _die(f"Labels file not found: {path}")

    rows: dict[tuple[str, str], bool] = {}
    required_cols = {"turn_id", "dimension", "human_score", "human_passing", "rater_initials"}

    try:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                _die(f"Labels CSV is empty or has no header: {path}")
            missing = required_cols - set(reader.fieldnames)
            if missing:
                _die(
                    f"Labels CSV is missing required columns: {sorted(missing)}\n"
                    f"Expected columns: {sorted(required_cols)}\n"
                    f"Got: {sorted(reader.fieldnames)}"
                )
            for lineno, row in enumerate(reader, start=2):
                key = (row["turn_id"].strip(), row["dimension"].strip())
                if key in rows:
                    _die(
                        f"Duplicate (turn_id, dimension) in labels CSV at line {lineno}: "
                        f"turn_id={key[0]!r}, dimension={key[1]!r}"
                    )
                try:
                    rows[key] = _parse_human_passing(row["human_passing"])
                except ValueError as exc:
                    _die(f"Labels CSV line {lineno}: {exc}")
    except (csv.Error, UnicodeDecodeError) as exc:
        _die(f"Failed to parse labels CSV {path}: {exc}")

    if not rows:
        _die(f"Labels CSV contains no data rows: {path}")

    return rows


def _load_verdicts(path: Path) -> dict[tuple[str, str], bool]:
    """Load judged verdicts JSONL → {(turn_id, dimension): judge_passing}.

    Raises
    ------
    SystemExit
        On missing file, bad format, or duplicate (turn_id, dimension) pairs.
    """
    if not path.exists():
        _die(f"Verdicts file not found: {path}")

    rows: dict[tuple[str, str], bool] = {}

    try:
        with path.open() as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    _die(f"Verdicts JSONL line {lineno}: JSON decode error: {exc}")
                for field in ("turn_id", "dimension", "passing"):
                    if field not in obj:
                        _die(
                            f"Verdicts JSONL line {lineno}: missing required field {field!r}.\n"
                            f"Required fields: turn_id, dimension, passing"
                        )
                key = (str(obj["turn_id"]).strip(), str(obj["dimension"]).strip())
                if key in rows:
                    _die(
                        f"Duplicate (turn_id, dimension) in verdicts JSONL at line {lineno}: "
                        f"turn_id={key[0]!r}, dimension={key[1]!r}"
                    )
                rows[key] = bool(obj["passing"])
    except (OSError, UnicodeDecodeError) as exc:
        _die(f"Failed to read verdicts JSONL {path}: {exc}")

    if not rows:
        _die(f"Verdicts JSONL contains no data rows: {path}")

    return rows


# ---------------------------------------------------------------------------
# Join logic
# ---------------------------------------------------------------------------


def _inner_join(
    labels: dict[tuple[str, str], bool],
    verdicts: dict[tuple[str, str], bool],
    allow_unmatched: bool,
) -> dict[tuple[str, str], tuple[bool, bool]]:
    """Inner join on (turn_id, dimension) → {key: (judge_passing, human_passing)}.

    Reports or raises on unmatched rows per the --allow-unmatched flag.
    """
    label_keys = set(labels)
    verdict_keys = set(verdicts)

    only_in_labels = label_keys - verdict_keys
    only_in_verdicts = verdict_keys - label_keys

    if only_in_labels or only_in_verdicts:
        msg_parts = ["UNMATCHED ROWS DETECTED — would be silently dropped by inner join:"]
        if only_in_labels:
            msg_parts.append(
                f"\n  {len(only_in_labels)} row(s) in labels but NOT in verdicts:"
            )
            for k in sorted(only_in_labels)[:20]:
                msg_parts.append(f"    turn_id={k[0]!r}, dimension={k[1]!r}")
            if len(only_in_labels) > 20:
                msg_parts.append(f"    ... and {len(only_in_labels) - 20} more")
        if only_in_verdicts:
            msg_parts.append(
                f"\n  {len(only_in_verdicts)} row(s) in verdicts but NOT in labels:"
            )
            for k in sorted(only_in_verdicts)[:20]:
                msg_parts.append(f"    turn_id={k[0]!r}, dimension={k[1]!r}")
            if len(only_in_verdicts) > 20:
                msg_parts.append(f"    ... and {len(only_in_verdicts) - 20} more")

        if allow_unmatched:
            print("WARNING: " + "\n".join(msg_parts), file=sys.stderr)
            print(
                f"WARNING: --allow-unmatched set; proceeding with inner join "
                f"({len(label_keys & verdict_keys)} matched rows).",
                file=sys.stderr,
            )
        else:
            msg_parts.append(
                "\nFix: align turn_id/dimension values across both files, or "
                "pass --allow-unmatched to proceed anyway (not recommended)."
            )
            _die("\n".join(msg_parts))

    matched_keys = label_keys & verdict_keys
    return {k: (verdicts[k], labels[k]) for k in matched_keys}


# ---------------------------------------------------------------------------
# Per-dimension statistics
# ---------------------------------------------------------------------------


def _compute_dim_stats(
    joined: dict[tuple[str, str], tuple[bool, bool]],
    dimension: str,
) -> dict:
    """Compute TPR, TNR, kappa + CIs for one dimension.

    Returns a dict with keys: dimension, n, tpr, tnr, tpr_ci, tnr_ci,
    kappa, kappa_ci, gate_eligible.
    """
    rows = [(j, h) for (_, dim), (j, h) in joined.items() if dim == dimension]
    n = len(rows)

    if n < _MIN_N:
        _die(
            f"Dimension {dimension!r} has only {n} matched items — minimum is "
            f"{_MIN_N}.  Too few to produce meaningful validation statistics.\n"
            f"Collect at least {_MIN_N} human labels for this dimension before "
            f"running validate_judge."
        )

    judge_labels = [j for j, _ in rows]
    human_labels = [h for _, h in rows]

    # TPR / TNR.
    tpr, tnr = tpr_tnr(judge_labels, human_labels)

    # Bootstrap CIs for TPR.
    def _tpr_fn(j: list, h: list) -> float | None:
        v, _ = tpr_tnr(j, h)
        return v

    def _tnr_fn(j: list, h: list) -> float | None:
        _, v = tpr_tnr(j, h)
        return v

    tpr_ci_lo, tpr_ci_hi = bootstrap_ci(
        _tpr_fn, judge_labels, human_labels,
        n_boot=_N_BOOT, seed=_BOOT_SEED, ci=_CI,
    )
    tnr_ci_lo, tnr_ci_hi = bootstrap_ci(
        _tnr_fn, judge_labels, human_labels,
        n_boot=_N_BOOT, seed=_BOOT_SEED, ci=_CI,
    )

    # Kappa.
    try:
        kappa = cohens_kappa(judge_labels, human_labels)
    except ValueError as exc:
        _die(
            f"Dimension {dimension!r}: kappa is undefined for the full dataset.\n"
            f"Detail: {exc}"
        )

    kappa_ci_lo, kappa_ci_hi = bootstrap_ci(
        cohens_kappa, judge_labels, human_labels,
        n_boot=_N_BOOT, seed=_BOOT_SEED, ci=_CI,
    )

    # Gate eligibility: kappa point estimate >= floor AND n >= 30.
    # Both conditions are required and documented; failing either → excluded from gate.
    gate_eligible = (kappa >= KAPPA_FLOOR) and (n >= _GATE_N)

    return {
        "dimension": dimension,
        "n": n,
        "tpr": tpr,
        "tnr": tnr,
        "tpr_ci_95": [tpr_ci_lo, tpr_ci_hi],
        "tnr_ci_95": [tnr_ci_lo, tnr_ci_hi],
        "kappa": kappa,
        "kappa_ci_95": [kappa_ci_lo, kappa_ci_hi],
        "gate_eligible": gate_eligible,
        "gate_reason": (
            None if gate_eligible else (
                f"kappa={kappa:.3f} < floor={KAPPA_FLOOR}"
                if kappa < KAPPA_FLOOR
                else f"n={n} < required={_GATE_N}"
            )
        ),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_COL_W = 16


def _print_table(results: list[dict]) -> None:
    """Print the validation table to stdout."""
    header = (
        f"{'dimension':<14} {'n':>4} {'TPR':>7} {'TNR':>7} "
        f"{'kappa':>7} {'kappa_CI_95':>22} {'gate_eligible':>14}"
    )
    print()
    print("Judge validation — binary agreement (judge.passing vs human.passing)")
    print("kappa floor: {:.2f} (Landis & Koch 1977 moderate).  "
          "n>=30 also required for gate.".format(KAPPA_FLOOR))
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        tpr_s = f"{r['tpr']:.3f}" if r["tpr"] is not None else "  N/A"
        tnr_s = f"{r['tnr']:.3f}" if r["tnr"] is not None else "  N/A"
        kappa_s = f"{r['kappa']:.3f}"
        ci_s = f"[{r['kappa_ci_95'][0]:.3f}, {r['kappa_ci_95'][1]:.3f}]"
        gate_s = "YES" if r["gate_eligible"] else "NO *** BELOW FLOOR ***"
        print(
            f"{r['dimension']:<14} {r['n']:>4} {tpr_s:>7} {tnr_s:>7} "
            f"{kappa_s:>7} {ci_s:>22} {gate_s:>14}"
        )
    print("-" * len(header))
    ineligible = [r for r in results if not r["gate_eligible"]]
    if ineligible:
        print()
        print("BELOW-FLOOR DIMENSIONS (excluded from gating — reported as untrusted):")
        for r in ineligible:
            print(f"  {r['dimension']}: {r['gate_reason']}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help=(
            "Path to human labels CSV.  "
            "Required columns: turn_id, dimension, human_score, human_passing, rater_initials.  "
            "human_passing: y/yes/1/true or n/no/0/false (case-insensitive)."
        ),
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        required=True,
        help=(
            "Path to judged verdicts JSONL (output of judge_trace).  "
            "Required fields per line: turn_id, dimension, passing."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/results/judge_validation.json"),
        help="Output path for the validation JSON (default: evals/results/judge_validation.json).",
    )
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        default=False,
        help=(
            "If set, unmatched rows (present in one file but not the other) are "
            "reported prominently but do not cause an error.  By default, any "
            "unmatched rows are an error (no silent inner-join shrinkage)."
        ),
    )

    args = parser.parse_args(argv)

    # Load files.
    labels = _load_labels(args.labels)
    verdicts = _load_verdicts(args.verdicts)

    # Inner join — fails loud on unmatched rows unless --allow-unmatched.
    joined = _inner_join(labels, verdicts, args.allow_unmatched)

    if not joined:
        _die("No matched (turn_id, dimension) pairs after join.  Nothing to validate.")

    # Discover dimensions present in the joined data.
    dims_present = sorted({dim for _, dim in joined})

    # Compute per-dimension stats.  Refuses to continue if n < _MIN_N for any dim.
    results = []
    for dim in dims_present:
        results.append(_compute_dim_stats(joined, dim))

    # Print table.
    _print_table(results)

    # Write JSON.
    output = {
        "kappa_floor": KAPPA_FLOOR,
        "kappa_floor_reference": "Landis & Koch (1977) banding — 0.41–0.60 = moderate",
        "gate_conditions": {
            "kappa_point_estimate_gte": KAPPA_FLOOR,
            "n_gte": _GATE_N,
        },
        "dimensions": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(output, fh, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
