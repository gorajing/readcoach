"""Judge batch runner — T5.2.

Runs the codex judge (evals/judge.py) over the 60 human-labeled turns ×
3 dimensions = 180 verdicts.  Writes a checkpointed workfile so that the
batch can be interrupted and resumed without re-running completed pairs.
The final output JSONL is written only when all 180 verdicts are present.

## Usage

    uv run python scripts/judge_turns.py

    # Resume after interruption (workfile is auto-loaded):
    uv run python scripts/judge_turns.py

    # Smoke-test (2 pairs only, keeps workfile, exits 0):
    uv run python scripts/judge_turns.py --limit 2

## Inputs

    --labels   evals/golden/turn_labels_v1.csv
    --turns    evals/results/turns_v1.jsonl
    --out      evals/results/judge_verdicts_v1.jsonl
    --workfile evals/results/.judge_verdicts_work.jsonl

## Output (verdict line format)

Each line in the workfile and in the final output is a JSON object:

    {
      "turn_id":    "<str>",
      "dimension":  "<str>",
      "score":      <int 1-5>,
      "passing":    <bool>,
      "issues":     [<str>, ...],
      "rationale":  "<str>",
      "model_meta": {"model": "...", "cli_version": "..."}
    }

This format is consumed by scripts/validate_judge.py (_load_verdicts).
Required fields for validate_judge: turn_id, dimension, passing.

## Checkpointing protocol

- Every completed verdict is appended (atomic line-append) to the workfile.
- On start: workfile is loaded and every line validated; malformed → abort.
- Missing (turn_id, dimension) pairs are processed; done pairs are skipped.
- A JudgeError from judge_turn aborts the run; the workfile is preserved.
- Final output written only when all 180 pairs are complete; workfile removed.

## Profile → turn_id mapping

The turns JSONL does not contain a turn_id field.  The turn_id is
reconstructed as:  <profile-abbrev>-t<turn_index:02d>

    fluent-but-hesitant  → fh
    self-corrector       → sc
    struggling-decoder   → sd

This matches the turn_id format in the human labels CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Make the project root importable when run as a script.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from evals.judge import JUDGED_DIMENSIONS, CodexCliTransport, JudgeError, judge_turn  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROFILE_ABBREVS: dict[str, str] = {
    "fluent-but-hesitant": "fh",
    "self-corrector": "sc",
    "struggling-decoder": "sd",
}

# Total expected pairs: 60 turns × 3 dimensions.
_TOTAL = 180


# ---------------------------------------------------------------------------
# Turn-id computation
# ---------------------------------------------------------------------------


def _turn_id(turn: dict) -> str:
    """Compute the turn_id string from a turn dict.

    Raises KeyError if 'profile' or 'turn_index' is missing.
    Raises ValueError if the profile is not in the known abbreviation map.
    """
    profile = turn["profile"]
    abbrev = _PROFILE_ABBREVS.get(profile)
    if abbrev is None:
        raise ValueError(
            f"Unknown profile {profile!r}. "
            f"Known profiles: {sorted(_PROFILE_ABBREVS)}"
        )
    index: int = turn["turn_index"]
    return f"{abbrev}-t{index:02d}"


def _turn_to_judge_dict(turn: dict) -> dict:
    """Return a dict in the shape judge_turn expects.

    judge_turn reads: utterance/tutor_utterance, move/policy_move,
    miscue/miscue_context.  Map from the turns JSONL field names.
    """
    return {
        "utterance": turn.get("utterance", ""),
        "move": turn.get("action_move", turn.get("move", "UNKNOWN")),
        "miscue": turn.get("miscue_type") or "",
    }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_labeled_turn_ids(labels_path: Path) -> set[str]:
    """Return the set of turn_ids present in the human labels CSV.

    Aborts loudly if the file is missing or malformed.
    """
    if not labels_path.exists():
        _abort(f"Labels file not found: {labels_path}")

    turn_ids: set[str] = set()
    try:
        with labels_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                _abort(f"Labels CSV is empty or has no header: {labels_path}")
            if "turn_id" not in (reader.fieldnames or []):
                _abort(
                    f"Labels CSV missing 'turn_id' column. "
                    f"Got columns: {reader.fieldnames}"
                )
            for row in reader:
                turn_ids.add(row["turn_id"].strip())
    except (csv.Error, UnicodeDecodeError) as exc:
        _abort(f"Failed to parse labels CSV {labels_path}: {exc}")

    if not turn_ids:
        _abort(f"Labels CSV contains no data rows: {labels_path}")
    return turn_ids


def _load_turns(turns_path: Path) -> list[dict]:
    """Load the turns JSONL and return a list of turn dicts.

    Aborts loudly if the file is missing or malformed.
    """
    if not turns_path.exists():
        _abort(f"Turns file not found: {turns_path}")

    turns: list[dict] = []
    try:
        with turns_path.open() as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    _abort(f"Turns JSONL line {lineno}: JSON decode error: {exc}")
                turns.append(obj)
    except (OSError, UnicodeDecodeError) as exc:
        _abort(f"Failed to read turns JSONL {turns_path}: {exc}")

    if not turns:
        _abort(f"Turns JSONL contains no data rows: {turns_path}")
    return turns


def _load_workfile(workfile_path: Path) -> dict[tuple[str, str], dict]:
    """Load completed verdicts from the workfile.

    Returns a dict keyed by (turn_id, dimension) → verdict_line_dict.

    Aborts loudly on any malformed line (no skipping).
    """
    done: dict[tuple[str, str], dict] = {}
    if not workfile_path.exists():
        return done

    required_fields = {"turn_id", "dimension", "score", "passing", "issues", "rationale", "model_meta"}
    try:
        with workfile_path.open() as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    _abort(
                        f"MALFORMED workfile line {lineno} in {workfile_path}: "
                        f"JSON decode error: {exc}\n"
                        f"Line content: {line[:200]!r}\n"
                        f"Aborting — fix or delete the workfile to proceed."
                    )
                missing = required_fields - set(obj.keys())
                if missing:
                    _abort(
                        f"MALFORMED workfile line {lineno} in {workfile_path}: "
                        f"missing required fields {sorted(missing)!r}.\n"
                        f"Aborting — fix or delete the workfile to proceed."
                    )
                key = (str(obj["turn_id"]).strip(), str(obj["dimension"]).strip())
                if key in done:
                    _abort(
                        f"MALFORMED workfile: duplicate (turn_id, dimension) at line {lineno}: "
                        f"turn_id={key[0]!r}, dimension={key[1]!r}\n"
                        f"Aborting — fix or delete the workfile to proceed."
                    )
                done[key] = obj
    except (OSError, UnicodeDecodeError) as exc:
        _abort(f"Failed to read workfile {workfile_path}: {exc}")

    return done


# ---------------------------------------------------------------------------
# Verdict serialization
# ---------------------------------------------------------------------------


def _verdict_line(turn_id: str, dimension: str, verdict) -> str:
    """Serialize a Verdict + turn_id to a JSON line string (no trailing newline)."""
    obj = {
        "turn_id": turn_id,
        "dimension": verdict.dimension,
        "score": verdict.score,
        "passing": verdict.passing,
        "issues": list(verdict.issues),
        "rationale": verdict.rationale,
        "model_meta": dict(verdict.model_meta),
    }
    return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Final assembly
# ---------------------------------------------------------------------------


def _write_final(out_path: Path, done: dict[tuple[str, str], dict]) -> None:
    """Write the final all-or-nothing output JSONL in deterministic order.

    Sorted by (turn_id, dimension) for reproducibility.
    """
    sorted_keys = sorted(done.keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for key in sorted_keys:
            fh.write(json.dumps(done[key], ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _abort(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:  # noqa: C901  (complexity is inherent)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("evals/golden/turn_labels_v1.csv"),
        help="Human labels CSV (default: evals/golden/turn_labels_v1.csv)",
    )
    parser.add_argument(
        "--turns",
        type=Path,
        default=Path("evals/results/turns_v1.jsonl"),
        help="Turns JSONL (default: evals/results/turns_v1.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evals/results/judge_verdicts_v1.jsonl"),
        help="Final output JSONL (default: evals/results/judge_verdicts_v1.jsonl)",
    )
    parser.add_argument(
        "--workfile",
        type=Path,
        default=Path("evals/results/.judge_verdicts_work.jsonl"),
        help="Checkpoint workfile (default: evals/results/.judge_verdicts_work.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process at most N missing pairs then exit 0 (smoke-test mode). "
            "Workfile is kept; final output is NOT written."
        ),
    )

    args = parser.parse_args(argv)

    # --- Load inputs ---
    labeled_ids = _load_labeled_turn_ids(args.labels)
    all_turns = _load_turns(args.turns)

    # --- Build turn lookup: turn_id → turn dict ---
    # Compute turn_id for every turn; abort if a labeled id is not covered.
    turn_by_id: dict[str, dict] = {}
    for turn in all_turns:
        try:
            tid = _turn_id(turn)
        except (KeyError, ValueError) as exc:
            _abort(f"Could not compute turn_id for turn: {exc}\nTurn: {turn}")
        turn_by_id[tid] = turn

    missing_from_turns = labeled_ids - set(turn_by_id)
    if missing_from_turns:
        _abort(
            "The following labeled turn_id(s) are missing from turns JSONL:\n"
            + "\n".join(f"  {tid}" for tid in sorted(missing_from_turns))
        )

    # --- Build the ordered list of (turn_id, dimension) pairs to judge ---
    # Only labeled turns, in deterministic order (turn_id, dimension).
    all_pairs: list[tuple[str, str]] = [
        (tid, dim)
        for tid in sorted(labeled_ids)
        for dim in JUDGED_DIMENSIONS
    ]

    # --- Load workfile ---
    done = _load_workfile(args.workfile)
    n_done = len(done)
    total = len(all_pairs)

    if n_done > 0:
        print(f"resuming: {n_done}/{total} done", file=sys.stderr)
    else:
        print(f"starting: 0/{total} done", file=sys.stderr)

    # --- Process missing pairs ---
    missing_pairs = [(tid, dim) for (tid, dim) in all_pairs if (tid, dim) not in done]

    if not missing_pairs:
        print(f"all {total} verdicts already complete — writing final output", file=sys.stderr)
    else:
        # Determine the pairs we'll actually process in this run.
        pairs_to_run = missing_pairs
        if args.limit is not None:
            pairs_to_run = missing_pairs[: args.limit]

        # Only create the transport if there's work to do.
        transport = CodexCliTransport() if pairs_to_run else None
        processed = 0

        for tid, dim in pairs_to_run:

            turn = turn_by_id[tid]
            judge_dict = _turn_to_judge_dict(turn)

            # Judge the pair — abort on JudgeError (no default verdicts).
            try:
                verdict = judge_turn(judge_dict, dim, transport=transport)
            except JudgeError as exc:
                print(
                    f"\nERROR: judge_turn failed for turn_id={tid!r}, dimension={dim!r}: {exc}",
                    file=sys.stderr,
                )
                print(
                    "Aborting — workfile preserved; rerun to resume from this point.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Serialize and append to workfile atomically (line-append).
            line = _verdict_line(tid, dim, verdict)
            args.workfile.parent.mkdir(parents=True, exist_ok=True)
            with args.workfile.open("a", encoding="utf-8") as wf:
                wf.write(line + "\n")

            key = (tid, dim)
            done[key] = json.loads(line)
            processed += 1
            n_done += 1

            # Progress to stderr: i/total, turn_id, dim, score
            print(
                f"{n_done}/{total}  {tid}  {dim}  score={verdict.score}",
                file=sys.stderr,
            )

        # Smoke-test mode: exit without writing final output.
        if args.limit is not None:
            print(
                f"smoke: {processed}/{total} done, workfile kept",
                file=sys.stderr,
            )
            return

    # --- Final assembly (only when all pairs complete) ---
    if len(done) < total:
        remaining = total - len(done)
        print(
            f"Not complete — {remaining}/{total} pairs still missing. "
            f"Rerun to continue.",
            file=sys.stderr,
        )
        return

    # All 180 done — write final output and remove workfile.
    _write_final(args.out, done)

    # Per-dimension counts.
    dim_counts: dict[str, int] = {dim: 0 for dim in JUDGED_DIMENSIONS}
    for (_, dim) in done:
        if dim in dim_counts:
            dim_counts[dim] += 1

    print(f"\nAll {total} verdicts complete.", file=sys.stderr)
    for dim in JUDGED_DIMENSIONS:
        print(f"  {dim}: {dim_counts[dim]} verdicts", file=sys.stderr)
    print(f"Wrote final output: {args.out}", file=sys.stderr)

    # Remove workfile (final output is all-or-nothing).
    if args.workfile.exists():
        args.workfile.unlink()
        print(f"Removed workfile: {args.workfile}", file=sys.stderr)


if __name__ == "__main__":
    main()
