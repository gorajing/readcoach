#!/usr/bin/env python3
"""T5.3 — FREEZE the persona dev/held-out split (one-way, hash-locked).

DOCTRINE
--------
The persona session corpus is split 50/50 into a DEV half and a HELD-OUT half.
The tutor A/B (later, key-blocked) may DEVELOP against the dev half but may ONLY
EVALUATE the held-out half — never tune to it.  The freeze is the audit anchor:
``evals/golden/holdout.lock`` records the generation seed, the split seed, the
per-file content hashes, and the per-persona counts, committed BEFORE any v1/v2
tutor run, so the frozen-ness is auditable from the commit timestamp.

THE FREEZE IS ONE-WAY.  There is NO regenerate-and-overwrite mode:

    uv run python scripts/freeze_split.py            # freeze (refuses if already frozen + differs)
    uv run python scripts/freeze_split.py --verify   # re-hash both files against the lock

If the frozen artifacts already exist and a fresh generation would differ from
them, the script REFUSES and exits non-zero (a re-freeze would destroy the audit
anchor).  Re-running when the freeze is already in place and IDENTICAL is a no-op
that confirms reproducibility.

Calibration is re-asserted before writing (no bypass): the generated corpus must
track each persona's published-rate spec within the documented tolerance, or the
freeze aborts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from readcoach.inject import load_passages  # noqa: E402
from readcoach.persona_gen import (  # noqa: E402
    assert_corpus_calibration,
    generate_all,
    load_personas,
    observed_rates,
    session_item_to_dict,
)

# ---------------------------------------------------------------------------
# Fixed seeds + paths.  Changing either seed is a corpus-identity change and is
# only legitimate with a fresh (un-frozen) split — never to chase a number.
# ---------------------------------------------------------------------------

GENERATION_SEED = 2026
SPLIT_SEED = 2026
CALIBRATION_TOLERANCE = 0.5

PERSONAS_DIR = PROJECT_ROOT / "data" / "personas"
PASSAGES_DIR = PROJECT_ROOT / "data" / "passages"
GOLDEN_DIR = PROJECT_ROOT / "evals" / "golden"
DEV_FILE = GOLDEN_DIR / "persona_sessions_dev.jsonl"
HOLDOUT_FILE = GOLDEN_DIR / "persona_sessions_holdout.jsonl"
LOCK_FILE = GOLDEN_DIR / "holdout.lock"


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_str(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Generation + split (pure; reused by build, verify, and the refusal check)
# ---------------------------------------------------------------------------

def _build_corpus() -> list:
    """Generate the full persona corpus and re-assert calibration (no bypass)."""
    personas = load_personas(PERSONAS_DIR)
    passages = load_passages(PASSAGES_DIR)
    items = generate_all(personas, passages, seed=GENERATION_SEED)
    # Fail loud if the corpus drifts from the published-rate spec.
    assert_corpus_calibration(personas, passages, items, tolerance=CALIBRATION_TOLERANCE)
    return items


def _split(items: list) -> tuple[list, list]:
    """Deterministic 50/50 dev/held-out split under the fixed split seed.

    Shuffle a COPY (sorted-by-id for a stable starting order) with the split RNG,
    then deal alternately into dev/held-out so each half is balanced in size even
    when the corpus is odd.  Every persona appears in both halves (the corpus has
    >= 28 sessions per persona; alternate dealing of a shuffled list cannot strand
    a persona on one side).
    """
    import random

    ordered = sorted(items, key=lambda it: it.id)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(ordered)
    dev = ordered[0::2]
    holdout = ordered[1::2]
    return dev, holdout


def _jsonl(items: list) -> str:
    """Serialize items to sorted-key JSONL (one line per item, trailing newline)."""
    lines = [
        json.dumps(session_item_to_dict(it), sort_keys=True, ensure_ascii=False)
        for it in items
    ]
    return "\n".join(lines) + "\n"


def _per_persona_counts(items: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for it in items:
        counts[it.persona_id] = counts.get(it.persona_id, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Freeze (one-way) + verify
# ---------------------------------------------------------------------------

def _frozen_content() -> tuple[str, str, dict]:
    """Compute (dev_jsonl, holdout_jsonl, lock_dict) from a fresh generation.

    Pure — touches no files.  The freeze and the refusal check both call this so
    a re-freeze is compared byte-for-byte against what is already on disk.
    """
    items = _build_corpus()
    dev, holdout = _split(items)
    dev_content = _jsonl(dev)
    holdout_content = _jsonl(holdout)

    personas = load_personas(PERSONAS_DIR)
    passages = load_passages(PASSAGES_DIR)
    obs = observed_rates(items)

    lock = {
        "generation_seed": GENERATION_SEED,
        "split_seed": SPLIT_SEED,
        "calibration_tolerance": CALIBRATION_TOLERANCE,
        "n_total": len(items),
        "n_dev": len(dev),
        "n_holdout": len(holdout),
        "per_persona_counts": {
            "total": _per_persona_counts(items),
            "dev": _per_persona_counts(dev),
            "holdout": _per_persona_counts(holdout),
        },
        "observed_rates_per_100": {
            pid: {cls: round(rate, 4) for cls, rate in by_cls.items()}
            for pid, by_cls in sorted(obs.items())
        },
        "persona_band_ceilings": {p.id: p.band_ceiling for p in personas},
        "n_passages": len(passages),
        "files": {
            DEV_FILE.name: {"sha256": _sha256_str(dev_content), "n_items": len(dev)},
            HOLDOUT_FILE.name: {
                "sha256": _sha256_str(holdout_content),
                "n_items": len(holdout),
            },
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "doctrine": (
            "ONE-WAY freeze. The held-out half may ONLY be evaluated by a tutor "
            "version, never developed against. Re-running freeze_split.py will "
            "REFUSE if a fresh generation differs from these artifacts."
        ),
    }
    return dev_content, holdout_content, lock


def _lock_json(lock: dict) -> str:
    return json.dumps(lock, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def freeze() -> int:
    """Write the dev/held-out JSONL + lock.  REFUSE if already frozen and differs."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    dev_content, holdout_content, lock = _frozen_content()

    already = DEV_FILE.exists() or HOLDOUT_FILE.exists() or LOCK_FILE.exists()
    if already:
        # The freeze is one-way: compare the fresh generation to what is on disk.
        # Identical -> no-op (reproducibility confirmed).  Different -> REFUSE.
        existing_dev = DEV_FILE.read_text(encoding="utf-8") if DEV_FILE.exists() else None
        existing_holdout = (
            HOLDOUT_FILE.read_text(encoding="utf-8") if HOLDOUT_FILE.exists() else None
        )
        # Compare data files by content; the lock carries a timestamp so compare it
        # by its file-hash fields, not the whole document.
        same = existing_dev == dev_content and existing_holdout == holdout_content
        if same and LOCK_FILE.exists():
            existing_lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            same = existing_lock.get("files") == lock["files"]
        if same:
            print(
                "FREEZE already in place and IDENTICAL — no-op (reproducibility "
                "confirmed).  Nothing rewritten."
            )
            return 0
        print(
            "REFUSING to re-freeze.\n"
            "  The persona dev/held-out split is already frozen and a fresh "
            "generation DIFFERS from the committed artifacts.\n"
            "  The freeze is ONE-WAY: re-writing it would destroy the audit anchor "
            "the tutor A/B depends on (the held-out half must never be re-derived "
            "after a tutor has seen it).\n"
            "  If a corpus change is genuinely intended, delete the frozen "
            "artifacts in a dedicated, reviewed commit that documents WHY, then "
            "re-freeze — never silently overwrite.",
            file=sys.stderr,
        )
        return 1

    DEV_FILE.write_text(dev_content, encoding="utf-8")
    HOLDOUT_FILE.write_text(holdout_content, encoding="utf-8")
    LOCK_FILE.write_text(_lock_json(lock), encoding="utf-8")

    print("FREEZE complete — one-way, hash-locked.")
    print(f"  {DEV_FILE.name:32s} {lock['n_dev']:3d} items  sha256={lock['files'][DEV_FILE.name]['sha256'][:16]}…")
    print(f"  {HOLDOUT_FILE.name:32s} {lock['n_holdout']:3d} items  sha256={lock['files'][HOLDOUT_FILE.name]['sha256'][:16]}…")
    print(f"  {LOCK_FILE.name}")
    print(f"  generation_seed={lock['generation_seed']}  split_seed={lock['split_seed']}")
    print(f"  per-persona (total): {lock['per_persona_counts']['total']}")
    print(f"  per-persona (dev):   {lock['per_persona_counts']['dev']}")
    print(f"  per-persona (holdout):{lock['per_persona_counts']['holdout']}")
    return 0


def verify() -> int:
    """Re-hash both frozen files against the lock; LOUD + non-zero on mismatch."""
    if not LOCK_FILE.exists():
        print(f"ERROR: lock file not found: {LOCK_FILE}", file=sys.stderr)
        return 1
    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    expected_files = lock["files"]

    failures: list[str] = []
    for name, meta in sorted(expected_files.items()):
        path = GOLDEN_DIR / name
        if not path.exists():
            failures.append(f"MISSING  {name}")
            continue
        actual = _sha256_file(path)
        if actual != meta["sha256"]:
            failures.append(
                f"MISMATCH {name}\n"
                f"  expected: {meta['sha256']}\n"
                f"  actual  : {actual}"
            )
        else:
            print(f"  OK  {name}  ({meta['n_items']} items)  sha256={actual[:16]}…")

    if failures:
        print("\n" + "=" * 60, file=sys.stderr)
        print("VERIFY FAILED — holdout lock violations:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print("VERIFY PASSED — both frozen files match holdout.lock.")
    print(
        f"  generation_seed={lock['generation_seed']}  "
        f"split_seed={lock['split_seed']}  "
        f"n_dev={lock['n_dev']}  n_holdout={lock['n_holdout']}"
    )
    print("=" * 60)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze (one-way) or verify the persona dev/held-out split."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-hash both frozen files against holdout.lock (CI mode).",
    )
    args = parser.parse_args(argv)
    return verify() if args.verify else freeze()


if __name__ == "__main__":
    sys.exit(main())
