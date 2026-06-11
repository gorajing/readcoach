#!/usr/bin/env python3
"""T5.L — Human turn-labeling CLI for ReadCoach.

Labels tutor turns on 3 dimensions (guidance, actionability, icap), 1-5 each,
producing exactly the CSV that scripts/validate_judge.py expects.

## How to run the session

    uv run python scripts/label_turns.py --init --n 60 --seed 42   # one-time setup
    uv run python scripts/label_turns.py                             # label (resumable)
    uv run python scripts/label_turns.py --report                   # summary stats

## Modes

    --init --n N --seed S
        Stratified sample of N turns from evals/results/turns_v1.jsonl
        (balanced across 3 profiles; mix of miscue/clean/page-end contexts;
        deterministic for a given seed) → evals/results/label_queue.json.

    default (no flags)
        For each pending turn: display context, utterance, then collect
        a 1-5 score per dimension with inline anchor reminders.
        Ratings appended atomically to evals/results/turn_labels.csv.
        Resumable: skips (turn_id, dimension) pairs already in the CSV.

    --report
        Counts per dimension, score distribution, time-per-turn estimate.
        Refuses (exits 1) if the CSV has no data rows.

## CSV format (matches validate_judge.py's _load_labels exactly)

    turn_id,dimension,human_score,human_passing,rater_initials
    sd-t00,guidance,4,y,JC
    sd-t00,actionability,3,n,JC
    ...

## Passing rule (hard rule from labeling_rubric.md)

    score >= 4  → human_passing = "y"
    score <= 2  → human_passing = "n"
    score == 3  → prompt rater: "borderline — pass? [y/n]"
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pathlib
import random
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).parent.parent
TURNS_PATH = ROOT / "evals" / "results" / "turns_v1.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"
QUEUE_FILE = RESULTS_DIR / "label_queue.json"
LABELS_FILE = RESULTS_DIR / "turn_labels.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMENSIONS = ("guidance", "actionability", "icap")

PROFILES = ("struggling-decoder", "fluent-but-hesitant", "self-corrector")

# CSV field order — must match validate_judge.py _load_labels exactly.
CSV_FIELDNAMES = ["turn_id", "dimension", "human_score", "human_passing", "rater_initials"]

# One-liner anchor summaries embedded here (full anchors: docs/labeling_rubric.md).
ANCHOR_ONELINER: dict[str, dict[int, str]] = {
    "guidance": {
        5: "Warm, age-perfect, motivation-protective, exactly fits the move — a skilled teacher would say this.",
        4: "Good and appropriate; minor stiffness or slightly-too-adult word, but lands well.",
        3: "Serviceable but flat/generic ('Good job.') — not harmful, not memorable.",
        2: "Off-tone for a young child (clinical, condescending, or over-effusive), loosely fits the move.",
        1: "Cold, discouraging, confusing, or contradicts the move.",
    },
    "actionability": {
        5: "One clear, concrete, child-doable next action — child knows exactly what to try.",
        4: "Actionable, but step is slightly broad or assumes an uncued skill ('sound it out').",
        3: "Points in a direction but leaves the *how* unspecified ('try again').",
        2: "Mostly affect with a faint nudge — little to act on.",
        1: "No actionable content, or action is wrong for the move.",
    },
    "icap": {
        5: "Interactive/strongly Constructive — invites child to generate, explain, or reason.",
        4: "Constructive — prompts child to produce something new (predict, infer, build from parts).",
        3: "Active — asks child to *do* the focused thing without generating new reasoning.",
        2: "Passive-leaning — mostly tells; child receives rather than acts.",
        1: "Passive/disengaging — no cognitive invitation, or shuts engagement down.",
    },
}

FULL_ANCHOR_FILE = ROOT / "docs" / "labeling_rubric.md"

# ---------------------------------------------------------------------------
# Pure-logic layer (tested; no I/O)
# ---------------------------------------------------------------------------


def load_turns(path: pathlib.Path = TURNS_PATH) -> list[dict]:
    """Load turns_v1.jsonl → list of turn dicts."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_turn_id(turn: dict) -> str:
    """Produce a stable, human-readable turn_id from profile + turn_index."""
    profile_abbr = {
        "struggling-decoder": "sd",
        "fluent-but-hesitant": "fh",
        "self-corrector": "sc",
    }.get(turn["profile"], turn["profile"][:2].lower())
    return f"{profile_abbr}-t{turn['turn_index']:02d}"


def build_queue(turns: list[dict], n: int = 60, seed: int = 42) -> list[dict]:
    """Stratified sample of n turns.

    Strategy:
    - Equal share per profile: n // 3 turns each (remainder distributed to
      first profiles alphabetically so total is exactly n).
    - Within each profile, mix: include page-end and non-page-end turns;
      include miscue and clean turns; prefer variety of action_move.
    - Final list is shuffled deterministically for the given seed.

    Returns a list of turn dicts, each enriched with a "turn_id" field.
    """
    rng = random.Random(seed)

    by_profile: dict[str, list[dict]] = {p: [] for p in PROFILES}
    for t in turns:
        if t["profile"] in by_profile:
            by_profile[t["profile"]].append(t)

    base_per_profile = n // 3
    remainders = n - base_per_profile * 3  # 0, 1, or 2

    selected: list[dict] = []

    for i, profile in enumerate(sorted(PROFILES)):
        quota = base_per_profile + (1 if i < remainders else 0)
        pool = list(by_profile[profile])

        # Partition into page-end vs mid-page, then miscue vs clean
        page_end = [t for t in pool if t["at_page_end"]]
        miscue = [t for t in pool if t.get("miscue_type") is not None]
        clean = [t for t in pool if t.get("miscue_type") is None]

        chosen: list[dict] = []
        chosen_ids: set[int] = set()  # turn_index

        def _pick(candidates: list[dict]) -> dict | None:
            avail = [t for t in candidates if t["turn_index"] not in chosen_ids]
            if not avail:
                return None
            return rng.choice(avail)

        # Seed mandatory slots to guarantee coverage
        for mandatory_pool in (page_end, miscue, clean):
            if len(chosen) >= quota:
                break
            t = _pick(mandatory_pool)
            if t is not None:
                chosen.append(t)
                chosen_ids.add(t["turn_index"])

        # Fill remaining quota from full pool
        remaining = [t for t in pool if t["turn_index"] not in chosen_ids]
        rng.shuffle(remaining)
        for t in remaining:
            if len(chosen) >= quota:
                break
            chosen.append(t)
            chosen_ids.add(t["turn_index"])

        selected.extend(chosen[:quota])

    # Shuffle the combined list
    rng.shuffle(selected)

    # Enrich with turn_id
    result = []
    for t in selected:
        entry = dict(t)
        entry["turn_id"] = make_turn_id(t)
        result.append(entry)

    return result


def derive_passing(score: int, borderline_answer: str | None = None) -> str:
    """Derive human_passing from score (hard rule from rubric).

    score >= 4 → "y"
    score <= 2 → "n"
    score == 3 → borderline_answer must be "y" or "n"

    Returns "y" or "n".
    """
    if score >= 4:
        return "y"
    if score <= 2:
        return "n"
    # score == 3
    if borderline_answer is None:
        raise ValueError("score=3 requires borderline_answer ('y' or 'n')")
    v = borderline_answer.strip().lower()
    if v not in ("y", "n"):
        raise ValueError(
            f"borderline_answer must be 'y' or 'n', got {borderline_answer!r}"
        )
    return v


def load_labeled_pairs(csv_file: io.TextIOBase) -> set[tuple[str, str]]:
    """Read a labels CSV (file-like) → set of (turn_id, dimension) already labeled."""
    labeled: set[tuple[str, str]] = set()
    reader = csv.DictReader(csv_file)
    if reader.fieldnames is None:
        return labeled
    for row in reader:
        tid = row.get("turn_id", "").strip()
        dim = row.get("dimension", "").strip()
        if tid and dim:
            labeled.add((tid, dim))
    return labeled


def validate_label_row(row: dict) -> None:
    """Raise ValueError if row is malformed.

    Checks:
    - All required fields present and non-empty (except human_score and human_passing have explicit rules)
    - turn_id: non-empty string
    - dimension: one of DIMENSIONS
    - human_score: integer 1-5
    - human_passing: "y" or "n"
    - rater_initials: non-empty
    """
    for field in CSV_FIELDNAMES:
        if field not in row:
            raise ValueError(f"Missing required field: {field!r}")

    if not row["turn_id"].strip():
        raise ValueError("turn_id is empty")

    dim = row["dimension"].strip()
    if dim not in DIMENSIONS:
        raise ValueError(
            f"dimension {dim!r} is not one of {DIMENSIONS}"
        )

    try:
        score = int(row["human_score"])
    except (ValueError, TypeError):
        raise ValueError(
            f"human_score {row['human_score']!r} is not an integer"
        )
    if not (1 <= score <= 5):
        raise ValueError(
            f"human_score {score} is out of range [1, 5]"
        )

    passing = row["human_passing"].strip().lower()
    if passing not in ("y", "n"):
        raise ValueError(
            f"human_passing {row['human_passing']!r} must be 'y' or 'n'"
        )

    if not row["rater_initials"].strip():
        raise ValueError("rater_initials is empty")


def stratification_summary(queue: list[dict]) -> dict:
    """Return a summary dict describing the stratification of a queue."""
    profiles: dict[str, int] = {}
    page_end_count = 0
    miscue_counts: dict[str | None, int] = {}
    move_counts: dict[str, int] = {}
    for t in queue:
        p = t["profile"]
        profiles[p] = profiles.get(p, 0) + 1
        if t.get("at_page_end"):
            page_end_count += 1
        mt = t.get("miscue_type")
        miscue_counts[mt] = miscue_counts.get(mt, 0) + 1
        mv = t.get("action_move", "?")
        move_counts[mv] = move_counts.get(mv, 0) + 1
    return {
        "n": len(queue),
        "profiles": profiles,
        "page_end": page_end_count,
        "mid_page": len(queue) - page_end_count,
        "miscue_types": miscue_counts,
        "moves": move_counts,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _append_label_row(csv_path: pathlib.Path, row: dict) -> None:
    """Atomically append one validated label row to the CSV.

    Atomic: read → append to list → write to .tmp → os.replace.
    Creates the header if the file doesn't exist yet.
    """
    validate_label_row(row)

    existing_rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                validate_label_row(r)
                existing_rows.append(r)

    existing_rows.append(row)

    tmp_path = csv_path.with_suffix(".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        w.writerows(existing_rows)

    os.replace(tmp_path, csv_path)


def _load_queue(queue_path: pathlib.Path) -> list[dict]:
    if not queue_path.exists():
        _die(f"Queue file not found: {queue_path}\nRun --init first.")
    return json.loads(queue_path.read_text())


def _get_rater_initials() -> str:
    """Prompt once for rater initials; keep asking until non-empty."""
    while True:
        initials = input("Rater initials (e.g. JC): ").strip()
        if initials:
            return initials
        print("  Initials cannot be empty.")


def _show_anchors(dimension: str) -> None:
    """Print the full anchor table for a dimension from the rubric file."""
    print(f"\n  Full anchors for [{dimension}]:")
    print(f"  (source: {FULL_ANCHOR_FILE})")
    for score in (5, 4, 3, 2, 1):
        text = ANCHOR_ONELINER[dimension][score]
        print(f"    {score} — {text}")
    print()


def _collect_score(
    dimension: str,
    turn_num: int,
    total_turns: int,
    dim_num: int,
) -> tuple[int, str]:
    """Prompt for a single dimension score.

    Returns (score: int, human_passing: str).
    """
    anchor_lines = "  Anchors:\n" + "\n".join(
        f"    {s} — {ANCHOR_ONELINER[dimension][s]}" for s in (5, 4, 3, 2, 1)
    )

    print(f"\n  [turn {turn_num}/{total_turns}, dim {dim_num}/3 — {dimension.upper()}]")
    print(anchor_lines)

    borderline_answer: str | None = None
    while True:
        raw = input(
            f"  Score for [{dimension}] (1-5, s=show full anchors, q=quit): "
        ).strip().lower()

        if raw == "q":
            print("\nSession paused. Progress saved. Re-run to continue.")
            sys.exit(0)

        if raw == "s":
            _show_anchors(dimension)
            continue

        try:
            score = int(raw)
        except ValueError:
            print("  Enter a number 1-5, 's' to show anchors, or 'q' to quit.")
            continue

        if not (1 <= score <= 5):
            print("  Score must be between 1 and 5.")
            continue

        # Handle borderline
        if score == 3:
            while True:
                ba = input(
                    "  Score 3 is borderline — pass? [y/n]: "
                ).strip().lower()
                if ba in ("y", "n"):
                    borderline_answer = ba
                    break
                print("  Enter y or n.")

        human_passing = derive_passing(score, borderline_answer)
        return score, human_passing


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def cmd_init(turns_path: pathlib.Path, n: int, seed: int) -> None:
    """Sample and write the label queue JSON; print stratification summary."""
    turns = load_turns(turns_path)
    queue = build_queue(turns, n=n, seed=seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_FILE.write_text(json.dumps(queue, indent=2))

    summary = stratification_summary(queue)
    print(f"Queue written: {QUEUE_FILE}")
    print(f"Seed: {seed}  |  n={n}")
    print("\nStratification summary:")
    print(f"  Total turns: {summary['n']}")
    for p in sorted(PROFILES):
        print(f"  {p}: {summary['profiles'].get(p, 0)}")
    print(f"  at_page_end: {summary['page_end']}")
    print(f"  mid_page: {summary['mid_page']}")
    print("\nMiscue context:")
    for mt, cnt in sorted(summary["miscue_types"].items(), key=lambda kv: (kv[0] is None, kv[0])):
        label = mt if mt is not None else "clean/none"
        print(f"  {label}: {cnt}")
    print("\nAction moves:")
    for mv, cnt in sorted(summary["moves"].items()):
        print(f"  {mv}: {cnt}")


def cmd_label(
    queue_path: pathlib.Path,
    csv_path: pathlib.Path,
    rater_initials: str | None,
) -> None:
    """Interactive label session."""
    queue = _load_queue(queue_path)
    total_turns = len(queue)

    # Load already-labeled pairs
    labeled_pairs: set[tuple[str, str]] = set()
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            labeled_pairs = load_labeled_pairs(f)

    # A turn is complete only when all 3 dimensions are labeled
    def _turn_complete(turn_id: str) -> bool:
        return all((turn_id, dim) in labeled_pairs for dim in DIMENSIONS)

    pending = [t for t in queue if not _turn_complete(t["turn_id"])]

    if not pending:
        print("All turns labeled. Run --report to see results.")
        return

    n_complete = total_turns - len(pending)
    print(f"\n{n_complete}/{total_turns} turns fully labeled. "
          f"{len(pending)} remaining. (ctrl+C or 'q' to stop — progress saved)\n")

    if rater_initials is None:
        rater_initials = _get_rater_initials()

    for turn_num_0, turn in enumerate(pending):
        turn_id = turn["turn_id"]
        turn_seq = n_complete + turn_num_0 + 1  # 1-based position in overall progress

        # Which dims still need labels for this turn?
        pending_dims = [
            d for d in DIMENSIONS if (turn_id, d) not in labeled_pairs
        ]

        # Display context once per turn
        print(f"\n{'='*64}")
        print(f"Turn {turn_seq}/{total_turns}  |  id={turn_id}")
        print(f"{'='*64}")
        print(f"  Profile:      {turn['profile']}")
        print(f"  Move:         {turn['action_move']}", end="")
        if turn.get("hint_level"):
            print(f" (hint={turn['hint_level']})", end="")
        print()
        if turn.get("miscue_type"):
            print(f"  Miscue:       {turn['miscue_type']}")
        else:
            print("  Miscue:       none (clean)")
        print(f"  Page end:     {'yes' if turn.get('at_page_end') else 'no'}")
        print(f"  Skill ID:     {turn.get('skill_id') or 'n/a'}")
        print(f"\n  UTTERANCE:\n  >>> {turn['utterance']}")

        for dim_num_0, dim in enumerate(pending_dims):
            dim_num = list(DIMENSIONS).index(dim) + 1
            score, human_passing = _collect_score(
                dim,
                turn_num=turn_seq,
                total_turns=total_turns,
                dim_num=dim_num,
            )

            row = {
                "turn_id": turn_id,
                "dimension": dim,
                "human_score": str(score),
                "human_passing": human_passing,
                "rater_initials": rater_initials,
            }

            csv_path.parent.mkdir(parents=True, exist_ok=True)
            _append_label_row(csv_path, row)
            labeled_pairs.add((turn_id, dim))
            print(f"  Saved: {turn_id} / {dim} = {score} → {human_passing}")

    print("\nAll turns labeled! Run --report to see results.")


def cmd_report(csv_path: pathlib.Path) -> None:
    """Print counts per dimension, score distribution, refuse on no data."""
    if not csv_path.exists():
        print(
            f"ERROR: No labels file found at {csv_path}.\n"
            "Run the labeling session first (default mode).",
            file=sys.stderr,
        )
        sys.exit(1)

    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for lineno, row in enumerate(reader, start=2):
            try:
                validate_label_row(row)
            except ValueError as exc:
                print(
                    f"ERROR: Malformed row at line {lineno}: {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
            rows.append(row)

    if not rows:
        print(
            "ERROR: No label rows found — run the labeling session first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Per-dimension stats
    print(f"\nLabeling report — {csv_path}")
    print(f"Total label rows: {len(rows)}")

    for dim in DIMENSIONS:
        dim_rows = [r for r in rows if r["dimension"] == dim]
        if not dim_rows:
            print(f"\n  {dim.upper()}: 0 labels")
            continue
        scores = [int(r["human_score"]) for r in dim_rows]
        passing = sum(1 for r in dim_rows if r["human_passing"] == "y")
        dist = {s: scores.count(s) for s in range(1, 6)}
        print(f"\n  {dim.upper()}: {len(dim_rows)} labels")
        print(f"    passing: {passing}/{len(dim_rows)}")
        print(f"    score distribution: {dist}")
        print(f"    mean score: {sum(scores)/len(scores):.2f}")

    # Unique turns
    turn_ids = {r["turn_id"] for r in rows}
    print(f"\nUnique turns labeled: {len(turn_ids)}")

    # Time estimate (rough: assume session just started)
    queue_path = QUEUE_FILE
    if queue_path.exists():
        queue = json.loads(queue_path.read_text())
        total = len(queue)
        complete = sum(
            1 for t in queue
            if all((t["turn_id"], d) in {(r["turn_id"], r["dimension"]) for r in rows}
                   for d in DIMENSIONS)
        )
        remaining = total - complete
        # Rough estimate: 2 min per turn (3 dimensions × ~40s each)
        est_min = remaining * 2
        print(f"Turns complete: {complete}/{total} (est. {est_min} min remaining at 2 min/turn)")


# ---------------------------------------------------------------------------
# CLI
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
        "--init",
        action="store_true",
        help="Sample turns and write label_queue.json.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=60,
        help="Number of turns to sample (default: 60).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print labeling summary and stats.",
    )
    parser.add_argument(
        "--queue",
        type=pathlib.Path,
        default=QUEUE_FILE,
        help=f"Path to queue JSON (default: {QUEUE_FILE}).",
    )
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=LABELS_FILE,
        help=f"Path to labels CSV (default: {LABELS_FILE}).",
    )
    parser.add_argument(
        "--turns",
        type=pathlib.Path,
        default=TURNS_PATH,
        help=f"Path to turns_v1.jsonl (default: {TURNS_PATH}).",
    )
    parser.add_argument(
        "--initials",
        type=str,
        default=None,
        help="Rater initials (prompted if omitted).",
    )

    args = parser.parse_args(argv)

    if args.init:
        cmd_init(args.turns, args.n, args.seed)
    elif args.report:
        cmd_report(args.csv)
    else:
        cmd_label(args.queue, args.csv, args.initials)


if __name__ == "__main__":
    main()
