#!/usr/bin/env python3
"""T1.4 Blind verification tool — shuffled opaque ids, resumable CSV, enum-validated.

Blind scorer for the ReadCoach benchmark:
  --init   : sample 30 clips, write evals/results/blind_verify_queue.json
  (default): interactive rate-then-reveal loop, append to blind_verify_ratings.csv
  --report : read CSV → print + write blind_verify_report.json

The mismatch rate from --report is the published artifact from the human session.

Miscue class keystrokes (rate mode):
  s  substitution
  o  omission
  i  insertion
  c  self_correction
  h  hesitation

Usage:
  uv run python scripts/blind_verify.py --init [--seed N] [--gold PATH]
  uv run python scripts/blind_verify.py [--queue PATH] [--csv PATH] [--initials XY]
  uv run python scripts/blind_verify.py --report [--csv PATH]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import os
import pathlib
import random
import subprocess
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).parent.parent
GOLD_PATH = ROOT / "data" / "benchmark" / "gold.jsonl"
CLIPS_DIR = ROOT / "data" / "benchmark" / "clips"
RESULTS_DIR = ROOT / "evals" / "results"

QUEUE_FILE = RESULTS_DIR / "blind_verify_queue.json"
RATINGS_FILE = RESULTS_DIR / "blind_verify_ratings.csv"
REPORT_FILE = RESULTS_DIR / "blind_verify_report.json"

# ---------------------------------------------------------------------------
# Constants / enums
# ---------------------------------------------------------------------------

VALID_MATCH_VALUES = frozenset({"y", "n"})

CSV_FIELDNAMES = [
    "opaque_id",
    "utt_id",
    "heard",
    "gold_summary",
    "match",
    "reason",
    "timestamp",
    "rater_initials",
]

CLASS_KEY_MAP = {
    "s": "substitution",
    "o": "omission",
    "i": "insertion",
    "c": "self_correction",
    "h": "hesitation",
}

ALL_CLASSES = ("substitution", "omission", "insertion", "self_correction", "hesitation")

# ---------------------------------------------------------------------------
# Pure-logic layer (tested by tests/test_blind_verify.py, no I/O here)
# ---------------------------------------------------------------------------


def load_gold(path: pathlib.Path = GOLD_PATH) -> list[dict]:
    """Load gold.jsonl → list of gold entry dicts."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _gold_classes(entry: dict) -> Counter:
    """Return a Counter of miscue types in this gold entry."""
    return Counter(m["type"] for m in entry["gold"])


def build_queue(gold: list[dict], n: int = 30, seed: int = 42) -> list[dict]:
    """Stratified sample of n clips from gold: every class >=4, both hesitation
    renders, clean items, then topped up to n.  Returns a SHUFFLED list of
    {opaque_id, wav_path, utt_id} — utt_id is in the queue for the mapping file
    only; the rating loop must never display it.

    Strategy:
    1. Mandatory slots: 1 per class (5 clips) + 1 hesitation-filler + 1 hesitation-silence
       + 3 clean items → 10 mandatory seeds.
    2. Fill remaining slots (n - mandatory) from the unused pool, ensuring per-class
       counts all reach >= 4.
    3. Shuffle the final selection with the given seed.
    """
    rng = random.Random(seed)

    # Index gold by type characteristics
    clean = [g for g in gold if g["gold"] == []]
    by_class: dict[str, list[dict]] = {cls: [] for cls in ALL_CLASSES}
    hesF_entries: list[dict] = []
    hesS_entries: list[dict] = []

    for g in gold:
        types = set(m["type"] for m in g["gold"])
        renders = set(m.get("render") for m in g["gold"] if m["type"] == "hesitation")
        for cls in types:
            by_class[cls].append(g)
        if "hesitation" in types:
            if "filler" in renders:
                hesF_entries.append(g)
            if "silence" in renders:
                hesS_entries.append(g)

    selected: set[str] = set()  # utt_ids
    mandatory: list[dict] = []

    def _pick_one(pool: list[dict], exclude: set[str]) -> dict | None:
        candidates = [g for g in pool if g["utt_id"] not in exclude]
        if not candidates:
            return None
        return rng.choice(candidates)

    # Step 1: Seed mandatory slots
    # 3 clean items
    pool_clean = list(clean)
    rng.shuffle(pool_clean)
    for g in pool_clean[:3]:
        mandatory.append(g)
        selected.add(g["utt_id"])

    # 1 filler hesitation (may not be clean)
    g = _pick_one(hesF_entries, selected)
    if g:
        mandatory.append(g)
        selected.add(g["utt_id"])

    # 1 silence hesitation
    g = _pick_one(hesS_entries, selected)
    if g:
        mandatory.append(g)
        selected.add(g["utt_id"])

    # At least 1 per class (other classes — sub, om, ins, sc; hesitation already seeded)
    for cls in ("substitution", "omission", "insertion", "self_correction"):
        if Counter(m["type"] for entry in mandatory for m in entry["gold"])[cls] == 0:
            g = _pick_one(by_class[cls], selected)
            if g:
                mandatory.append(g)
                selected.add(g["utt_id"])

    # Step 2: Fill up to n ensuring >= 4 per class
    remaining = [g for g in gold if g["utt_id"] not in selected]
    rng.shuffle(remaining)

    def _current_counts() -> Counter:
        c: Counter = Counter()
        for entry in mandatory:
            c.update(m["type"] for m in entry["gold"])
        return c

    # First, ensure each class reaches >= 4
    for cls in ALL_CLASSES:
        while _current_counts()[cls] < 4 and remaining and len(mandatory) < n:
            # Find next entry in remaining that contributes to this class
            for j, g in enumerate(remaining):
                if any(m["type"] == cls for m in g["gold"]):
                    mandatory.append(g)
                    selected.add(g["utt_id"])
                    remaining.pop(j)
                    break
            else:
                # No more entries for this class — can't satisfy, break
                break

    # Fill to n from remaining
    for g in remaining:
        if len(mandatory) >= n:
            break
        mandatory.append(g)

    # Trim to exactly n if over (shouldn't happen normally)
    final = mandatory[:n]

    # Step 3: Shuffle
    rng.shuffle(final)

    # Assign opaque ids q01..q30
    queue = []
    for i, g in enumerate(final, 1):
        wav_path = str(CLIPS_DIR / f"{g['utt_id']}.wav")
        queue.append({
            "opaque_id": f"q{i:02d}",
            "wav_path": wav_path,
            "utt_id": g["utt_id"],
        })

    return queue


def load_rated_ids(csv_file: io.TextIOBase) -> set[str]:
    """Read a ratings CSV (file-like) and return the set of already-rated opaque_ids."""
    rated: set[str] = set()
    reader = csv.DictReader(csv_file)
    for row in reader:
        oid = row.get("opaque_id", "").strip()
        if oid:
            rated.add(oid)
    return rated


def pending_queue(queue: list[dict], rated_set: set[str]) -> list[dict]:
    """Return queue entries whose opaque_id is NOT in rated_set."""
    return [e for e in queue if e["opaque_id"] not in rated_set]


def auto_compare(heard: list[dict], gold: list[dict]) -> dict:
    """Compare heard miscues to gold by class multiset equality.

    heard: list of {type, word} — rater's enumerated deviations
    gold: list of gold miscue dicts (from gold.jsonl)

    Returns {match: bool, miss: list[str], extra: list[str]}
      miss  = classes present in gold but not (enough) in heard
      extra = classes present in heard but not (enough) in gold
    """
    heard_counts: Counter = Counter(m["type"] for m in heard)
    gold_counts: Counter = Counter(m["type"] for m in gold)

    miss = []
    extra = []

    all_types = set(heard_counts) | set(gold_counts)
    for cls in sorted(all_types):
        h = heard_counts.get(cls, 0)
        g = gold_counts.get(cls, 0)
        if g > h:
            miss.append(cls)
        elif h > g:
            extra.append(cls)

    return {
        "match": len(miss) == 0 and len(extra) == 0,
        "miss": miss,
        "extra": extra,
    }


def _gold_summary(gold: list[dict]) -> str:
    """Compact human-readable summary of gold miscues for the CSV."""
    if not gold:
        return "clean"
    parts = []
    for m in gold:
        cls = m["type"]
        tw = m.get("target_word") or ""
        sw = m.get("said_word") or ""
        render = m.get("render") or ""
        if cls == "substitution":
            parts.append(f"sub({tw}->{sw})")
        elif cls == "omission":
            parts.append(f"om({tw})")
        elif cls == "insertion":
            parts.append(f"ins({sw})")
        elif cls == "self_correction":
            parts.append(f"sc({tw}<-{sw})")
        elif cls == "hesitation":
            r = f"/{render}" if render else ""
            parts.append(f"hes{r}({tw or sw})")
    return "; ".join(parts)


def validate_ratings_csv(csv_file: io.TextIOBase) -> list[dict]:
    """Read and validate a ratings CSV.

    Raises ValueError with the 1-based CSV file row number (including header) on:
      - match not in {y, n}
      - match == 'n' with empty reason

    Returns the list of validated row dicts.
    """
    rows = []
    reader = csv.DictReader(csv_file)
    for data_row_idx, row in enumerate(reader, start=1):
        csv_row_num = data_row_idx + 1  # +1 for header
        match_val = row.get("match", "").strip()
        reason_val = row.get("reason", "").strip()

        if match_val not in VALID_MATCH_VALUES:
            raise ValueError(
                f"Invalid match value {match_val!r} at CSV row {csv_row_num} "
                f"(opaque_id={row.get('opaque_id')!r}). Must be 'y' or 'n'."
            )
        if match_val == "n" and not reason_val:
            raise ValueError(
                f"Mismatch (match='n') at CSV row {csv_row_num} "
                f"(opaque_id={row.get('opaque_id')!r}) has an empty reason. "
                "A non-empty reason is required for every mismatch."
            )
        rows.append(row)
    return rows


def compute_report(csv_file: io.TextIOBase) -> dict:
    """Compute the report dict from a validated ratings CSV file-like object.

    Raises if the CSV has no data rows (pre-session / absent file).
    Raises ValueError on invalid enum fields (delegates to validate_ratings_csv).
    """
    rows = validate_ratings_csv(csv_file)
    if not rows:
        raise RuntimeError(
            "Ratings CSV has no data rows. Run the rating session first."
        )

    n_rated = len(rows)
    n_match = sum(1 for r in rows if r["match"].strip() == "y")
    n_mismatch = n_rated - n_match
    mismatch_rate = n_mismatch / n_rated if n_rated else 0.0

    # Per-class breakdown from gold_summary field (best-effort parse)
    # gold_summary is a human-readable string; for structured per-class breakdown
    # we parse the class tokens from gold_summary.
    class_match: dict[str, dict[str, int]] = {
        cls: {"match": 0, "mismatch": 0} for cls in ALL_CLASSES
    }
    class_match["clean"] = {"match": 0, "mismatch": 0}

    for row in rows:
        gs = row.get("gold_summary", "").strip()
        is_match = row["match"].strip() == "y"
        # Determine which classes are represented
        if gs == "clean":
            key = "clean"
            class_match[key]["match" if is_match else "mismatch"] += 1
        else:
            # Extract class names present in gold_summary
            found_classes = set()
            for cls in ALL_CLASSES:
                abbr = cls[:3]  # sub, om_, ins, sel, hes
                if cls == "self_correction":
                    abbr = "sc"
                elif cls == "hesitation":
                    abbr = "hes"
                elif cls == "substitution":
                    abbr = "sub"
                elif cls == "omission":
                    abbr = "om"
                elif cls == "insertion":
                    abbr = "ins"
                if abbr in gs:
                    found_classes.add(cls)
            for cls in found_classes:
                class_match[cls]["match" if is_match else "mismatch"] += 1

    mismatches = [
        {
            "opaque_id": r["opaque_id"],
            "utt_id": r.get("utt_id", ""),
            "gold_summary": r.get("gold_summary", ""),
            "heard": r.get("heard", ""),
            "reason": r.get("reason", ""),
        }
        for r in rows
        if r["match"].strip() == "n"
    ]

    return {
        "n_rated": n_rated,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "mismatch_rate": mismatch_rate,
        "per_class": class_match,
        "mismatches": mismatches,
    }


def compute_report_from_path(csv_path: pathlib.Path) -> dict:
    """Load and compute report from a CSV file path. Raises if missing."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Ratings CSV not found: {csv_path}. "
            "Run the rating session first (default mode)."
        )
    return compute_report(csv_path.open(newline="", encoding="utf-8"))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _play_clip(wav_path: str) -> None:
    """Play a WAV file via afplay. Fails loud on nonzero exit."""
    result = subprocess.run(["afplay", wav_path], capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"afplay returned exit code {result.returncode} for {wav_path!r}: {err}"
        )


def _collect_heard() -> list[dict]:
    """Interactively collect the rater's perceived miscues.

    Returns list of {type, word} dicts.
    """
    heard: list[dict] = []
    print("\n  What did you hear? Enter miscues one at a time.")
    print("  Keys: s=substitution  o=omission  i=insertion  c=self_correction  h=hesitation")
    print("  Type 'done' when finished, 'none' if you heard nothing unusual.\n")
    while True:
        raw = input("  miscue (s/o/i/c/h or done/none): ").strip().lower()
        if raw in ("done", ""):
            break
        if raw == "none":
            heard = []
            break
        if raw not in CLASS_KEY_MAP:
            print(f"  Unknown key {raw!r}. Use: {', '.join(CLASS_KEY_MAP.keys())}")
            continue
        cls = CLASS_KEY_MAP[raw]
        word = input(f"  Approximate word for {cls}: ").strip()
        heard.append({"type": cls, "word": word})
        print(f"  Added: {cls} ({word!r})")
    return heard


def _get_rater_initials() -> str:
    """Prompt once for rater initials; keep asking until non-empty."""
    while True:
        initials = input("Rater initials (e.g. JC): ").strip()
        if initials:
            return initials
        print("  Initials cannot be empty.")


def _append_rating_row(
    csv_path: pathlib.Path,
    row: dict,
) -> None:
    """Atomically append one validated row to the ratings CSV.

    Atomic: write to a temp file, then os.replace into the real path.
    If the CSV doesn't exist yet, write the header first.
    """
    # Validate before writing
    match_val = row.get("match", "").strip()
    reason_val = row.get("reason", "").strip()
    if match_val not in VALID_MATCH_VALUES:
        raise ValueError(f"Invalid match value {match_val!r} — must be 'y' or 'n'.")
    if match_val == "n" and not reason_val:
        raise ValueError("Mismatch rows require a non-empty reason.")

    existing_rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    existing_rows.append(row)

    tmp_path = csv_path.with_suffix(".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        w.writerows(existing_rows)

    os.replace(tmp_path, csv_path)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def cmd_init(gold_path: pathlib.Path, seed: int, n: int = 30) -> None:
    """Sample and write the blind queue JSON."""
    gold = load_gold(gold_path)
    queue = build_queue(gold, n=n, seed=seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_FILE.write_text(json.dumps(queue, indent=2))

    # Print coverage summary
    gold_map = {g["utt_id"]: g for g in gold}
    counts: Counter = Counter()
    clean_count = 0
    render_counts: Counter = Counter()

    for entry in queue:
        uid = entry["utt_id"]
        g = gold_map[uid]
        if g["gold"] == []:
            clean_count += 1
        for m in g["gold"]:
            counts[m["type"]] += 1
            if m["type"] == "hesitation" and m.get("render"):
                render_counts[m["render"]] += 1

    print(f"Queue written: {QUEUE_FILE}")
    print(f"Seed: {seed}  |  n={n}")
    print("\nClass coverage in sample:")
    for cls in ALL_CLASSES:
        print(f"  {cls:<20} {counts[cls]:>3} instance(s)")
    print(f"  {'clean items':<20} {clean_count:>3}")
    print(f"\nHesitation renders:")
    print(f"  filler:  {render_counts['filler']}")
    print(f"  silence: {render_counts['silence']}")


def cmd_rate(
    queue_path: pathlib.Path,
    csv_path: pathlib.Path,
    rater_initials: str | None,
    gold_path: pathlib.Path,
) -> None:
    """Interactive rate-then-reveal loop."""
    if not queue_path.exists():
        sys.exit(f"Queue file not found: {queue_path}\nRun --init first.")

    queue: list[dict] = json.loads(queue_path.read_text())
    gold_map = {g["utt_id"]: g for g in load_gold(gold_path)}

    # Load already-rated ids
    rated_set: set[str] = set()
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            rated_set = load_rated_ids(f)

    todo = pending_queue(queue, rated_set)

    if not todo:
        print("All items rated. Run --report to see results.")
        return

    if rater_initials is None:
        rater_initials = _get_rater_initials()

    print(f"\n{len(rated_set)} rated, {len(todo)} remaining. (ctrl+C to stop — progress saved)\n")

    for entry in todo:
        oid = entry["opaque_id"]
        wav_path = entry["wav_path"]
        utt_id = entry["utt_id"]
        gold_entry = gold_map.get(utt_id)
        if gold_entry is None:
            print(f"WARNING: utt_id {utt_id!r} not found in gold. Skipping.")
            continue

        target_text = gold_entry["target_text"]
        gold_miscues = gold_entry["gold"]

        print(f"\n{'='*60}")
        print(f"Item {oid}")
        print(f"{'='*60}")
        print(f"\nPASSAGE TEXT (what they should have read):\n  {target_text}\n")

        # Play
        _play_clip(wav_path)

        # Replay option
        while True:
            replay = input("  [r] to replay, or press Enter to continue: ").strip().lower()
            if replay == "r":
                _play_clip(wav_path)
            else:
                break

        # Collect heard
        heard = _collect_heard()

        # Reveal gold
        gold_sum = _gold_summary(gold_miscues)
        print(f"\n  GOLD: {gold_sum}")

        # Auto-compare
        cmp = auto_compare(heard, gold_miscues)
        if cmp["match"]:
            print("  Auto-compare: MATCH")
        else:
            if cmp["miss"]:
                print(f"  Auto-compare: MISS in heard — {', '.join(cmp['miss'])}")
            if cmp["extra"]:
                print(f"  Auto-compare: EXTRA in heard — {', '.join(cmp['extra'])}")

        # Rater confirms
        while True:
            verdict = input("  Match? [y/n]: ").strip().lower()
            if verdict in ("y", "n"):
                break
            print("  Enter y or n.")

        reason = ""
        if verdict == "n":
            while True:
                reason = input("  Reason (required): ").strip()
                if reason:
                    break
                print("  Reason cannot be empty for a mismatch.")

        heard_str = json.dumps(heard)
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        row = {
            "opaque_id": oid,
            "utt_id": utt_id,
            "heard": heard_str,
            "gold_summary": gold_sum,
            "match": verdict,
            "reason": reason,
            "timestamp": ts,
            "rater_initials": rater_initials,
        }

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _append_rating_row(csv_path, row)
        print(f"  Saved. (ctrl+C to stop)")

    print("\nAll items rated! Run --report to see results.")


def cmd_report(csv_path: pathlib.Path, report_path: pathlib.Path) -> None:
    """Compute and print the report; write JSON artifact."""
    report = compute_report_from_path(csv_path)

    # Print
    print(f"\nBlind Verification Report")
    print(f"{'='*40}")
    print(f"n_rated:       {report['n_rated']}")
    print(f"n_match:       {report['n_match']}")
    print(f"n_mismatch:    {report['n_mismatch']}")
    print(f"mismatch_rate: {report['mismatch_rate']:.3f} ({report['n_mismatch']}/{report['n_rated']})")

    print(f"\nPer-class breakdown:")
    for cls, counts in report["per_class"].items():
        total = counts["match"] + counts["mismatch"]
        if total > 0:
            print(f"  {cls:<20} match={counts['match']} mismatch={counts['mismatch']}")

    if report["mismatches"]:
        print(f"\nMismatches ({len(report['mismatches'])}):")
        for mm in report["mismatches"]:
            print(f"  {mm['opaque_id']}  gold={mm['gold_summary']!r}  reason={mm['reason']!r}")

    # Write JSON
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blind verification tool for ReadCoach benchmark clips."
    )
    parser.add_argument("--init", action="store_true", help="Sample 30 clips and write queue.")
    parser.add_argument("--report", action="store_true", help="Compute and print report.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling (default: 42).")
    parser.add_argument("--gold", type=pathlib.Path, default=GOLD_PATH, help="Path to gold.jsonl.")
    parser.add_argument("--queue", type=pathlib.Path, default=QUEUE_FILE, help="Path to queue JSON.")
    parser.add_argument("--csv", type=pathlib.Path, default=RATINGS_FILE, help="Path to ratings CSV.")
    parser.add_argument("--report-out", type=pathlib.Path, default=REPORT_FILE, help="Path to report JSON.")
    parser.add_argument("--initials", type=str, default=None, help="Rater initials (prompted if omitted).")

    args = parser.parse_args()

    if args.init:
        cmd_init(args.gold, args.seed)
    elif args.report:
        cmd_report(args.csv, args.report_out)
    else:
        cmd_rate(args.queue, args.csv, args.initials, args.gold)


if __name__ == "__main__":
    main()
