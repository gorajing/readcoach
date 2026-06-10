#!/usr/bin/env python3
"""Build (or verify) the synthetic reading-benchmark v0.1.0.

USAGE
-----
Build mode (renders all clips, writes gold.jsonl + manifest + lock):

    uv run python scripts/build_benchmark.py

Verify mode (re-hashes everything on disk, exits non-zero on any mismatch):

    uv run python scripts/build_benchmark.py --verify

DESIGN NOTES
------------
* Benchmark seed is fixed at 42 — documented, never changed without a new
  version bump.
* Voice/rate assignment is derived deterministically from each utt_id via a
  seeded hash, so the mapping is stable even if the item list order changes.
* 88 items (8 passages × 11 items each): 1 clean + 6 single-miscue + 4 multi.
* Every clip is validated with ffprobe after conversion; any failure aborts
  with the utt_id named (fail loud).
* The content lock (evals/golden/benchmark.lock) records the sha256 of every
  artifact.  --verify checks the lock; a silent mutation of any artifact fails
  loudly naming the changed file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure the package src tree is on the path when run via uv run.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from readcoach.audio_utils import run_say, to_16k_mono, validate_clip
from readcoach.inject import load_passages, plan_benchmark

# ---------------------------------------------------------------------------
# Paths (all absolute, relative to project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PASSAGES_DIR = DATA_DIR / "passages"
BENCHMARK_DIR = DATA_DIR / "benchmark"
CLIPS_DIR = BENCHMARK_DIR / "clips"
GOLD_JSONL = BENCHMARK_DIR / "gold.jsonl"
MANIFEST_JSON = BENCHMARK_DIR / "manifest.json"
LOCK_FILE = PROJECT_ROOT / "evals" / "golden" / "benchmark.lock"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "benchmark"

# Benchmark identity — bump version when content changes.
BENCHMARK_VERSION = "0.1.0"
BENCHMARK_SEED = 42

# ---------------------------------------------------------------------------
# Voice/rate pool — deterministically varied per utt_id.
#
# Selected from macOS en_US voices that produce intelligible, natural-sounding
# speech for ASR evaluation purposes.  Novelty/robot voices excluded.
# Rates range 140–180 wpm to simulate different reader tempos.
# ---------------------------------------------------------------------------

_VOICE_RATE_POOL: tuple[tuple[str, int], ...] = (
    ("Samantha",              150),
    ("Fred",                  140),
    ("Kathy",                 155),
    ("Grandpa (English (US))", 145),
    ("Junior",                160),
    ("Eddy (English (US))",   170),
)


def _voice_rate_for(utt_id: str) -> tuple[str, int]:
    """Deterministically assign a (voice, rate) pair to ``utt_id``.

    Uses a SHA-256 hash of the utt_id string, mod the pool size.  This
    ensures the assignment is stable even if the item list is reordered.
    """
    digest = int(hashlib.sha256(utt_id.encode()).hexdigest(), 16)
    idx = digest % len(_VOICE_RATE_POOL)
    return _VOICE_RATE_POOL[idx]


# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Tool version helpers
# ---------------------------------------------------------------------------

def _ffmpeg_version() -> str:
    """Return the first line of ``ffmpeg -version``."""
    result = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True, text=True, timeout=10,
    )
    return result.stdout.splitlines()[0] if result.stdout else "unknown"


def _say_version() -> str:
    """Return macOS version as a proxy for ``say`` version."""
    result = subprocess.run(
        ["sw_vers", "-productVersion"],
        capture_output=True, text=True, timeout=10,
    )
    return f"macOS/{result.stdout.strip()}" if result.returncode == 0 else "unknown"


# ---------------------------------------------------------------------------
# Fixture selection: pick 3 representative clips to commit
# ---------------------------------------------------------------------------

def _pick_fixture_utt_ids(items: list) -> list[str]:
    """Return 3 utt_ids that represent clean / substitution / silence-hesitation.

    We search the full item list for the FIRST item from each category:
      1. clean (no gold)
      2. single substitution
      3. single silence-hesitation
    """
    fixtures: list[str] = []

    # 1. Clean
    for it in items:
        if not it.gold:
            fixtures.append(it.utt_id)
            break

    # 2. Single substitution
    for it in items:
        if len(it.gold) == 1 and it.gold[0].type == "substitution":
            fixtures.append(it.utt_id)
            break

    # 3. Single silence-hesitation
    for it, render in [(it, it.gold_render) for it in items]:
        if (
            len(it.gold) == 1
            and it.gold[0].type == "hesitation"
            and it.gold_render[0] == "silence"
        ):
            fixtures.append(it.utt_id)
            break

    if len(fixtures) != 3:
        raise RuntimeError(
            f"Could not find 3 representative fixture items (found {len(fixtures)}); "
            f"check that all three categories exist in the benchmark plan."
        )
    return fixtures


# ---------------------------------------------------------------------------
# Build mode
# ---------------------------------------------------------------------------

def _build() -> None:
    """Render all 88 clips, write gold.jsonl, manifest.json, and lock."""

    t_start = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Plan benchmark items (deterministic, seeded)
    # ------------------------------------------------------------------
    print(f"Loading passages from {PASSAGES_DIR} …")
    passages = load_passages(PASSAGES_DIR)
    print(f"  {len(passages)} passages loaded.")

    print(f"Planning benchmark (seed={BENCHMARK_SEED}) …")
    items = plan_benchmark(passages, seed=BENCHMARK_SEED)
    print(f"  {len(items)} items planned.")

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Render each item to a validated 16 kHz mono WAV
    # ------------------------------------------------------------------
    gold_rows: list[dict] = []
    n_rendered = 0
    total_duration_s = 0.0

    voices_used: set[str] = set()

    with tempfile.TemporaryDirectory(prefix="readcoach_tts_") as tmpdir:
        tmp = Path(tmpdir)

        for i, item in enumerate(items, 1):
            voice, rate = _voice_rate_for(item.utt_id)
            voices_used.add(voice)

            wav_path = CLIPS_DIR / f"{item.utt_id}.wav"
            aiff_path = tmp / f"{item.utt_id}.aiff"

            print(
                f"  [{i:3d}/{len(items)}] {item.utt_id}  voice={voice!r}  rate={rate}",
                end=" … ",
                flush=True,
            )

            # Render AIFF via macOS say.
            run_say(item.tts_text, voice=voice, rate=rate, out_aiff=aiff_path)

            # Convert to 16 kHz mono WAV.
            to_16k_mono(aiff_path, wav_path)

            # Validate via ffprobe (aborts on any violation).
            info = validate_clip(wav_path)
            duration_s = float(info["duration_s"])
            total_duration_s += duration_s

            wav_sha256 = _sha256_file(wav_path)

            print(f"ok ({duration_s:.2f}s)")

            # Build the gold row — self-describing JSONL.
            gold_entry: dict = {
                "utt_id": item.utt_id,
                "passage_id": item.passage_id,
                "band": next(p.band for p in passages if p.id == item.passage_id),
                "target_text": item.target_text,
                "miscued_text": item.miscued_text,
                "gold": [
                    {
                        "type": g.type,
                        "target_word": g.target_word,
                        "said_word": g.said_word,
                        "index": g.index,
                        "render": item.gold_render[j],
                    }
                    for j, g in enumerate(item.gold)
                ],
                "voice": voice,
                "rate_wpm": rate,
                "wav_sha256": wav_sha256,
                "duration_s": round(duration_s, 6),
            }
            gold_rows.append(gold_entry)
            n_rendered += 1

    # ------------------------------------------------------------------
    # 3. Write gold.jsonl (sorted keys, one line per item, utf-8)
    # ------------------------------------------------------------------
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    gold_lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in gold_rows]
    gold_content = "\n".join(gold_lines) + "\n"
    GOLD_JSONL.write_text(gold_content, encoding="utf-8")
    print(f"\nWrote {GOLD_JSONL} ({len(gold_lines)} lines).")

    gold_sha256 = _sha256_str(gold_content)

    # ------------------------------------------------------------------
    # 4. Build coverage matrix (class × passage)
    # ------------------------------------------------------------------
    from readcoach.inject import _ALL_CLASSES, validate_coverage
    coverage_matrix = validate_coverage(items)  # raises if incomplete

    # Convert to a JSON-serialisable flat form: {passage_id: {class: count}}.
    coverage_json = {
        pid: dict(sorted(by_class.items()))
        for pid, by_class in sorted(coverage_matrix.items())
    }

    # ------------------------------------------------------------------
    # 5. Write manifest.json
    # ------------------------------------------------------------------
    total_bytes = sum((CLIPS_DIR / f"{row['utt_id']}.wav").stat().st_size for row in gold_rows)

    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "seed": BENCHMARK_SEED,
        "n_items": n_rendered,
        "coverage_matrix": coverage_json,
        "generation_tools": {
            "say": _say_version(),
            "ffmpeg": _ffmpeg_version(),
        },
        "license": "passages original to this project, MIT",
        "gold_jsonl_sha256": gold_sha256,
        "total_clip_bytes": total_bytes,
        "total_duration_s": round(total_duration_s, 3),
        "voices_used": sorted(voices_used),
    }
    MANIFEST_JSON.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {MANIFEST_JSON}.")

    # ------------------------------------------------------------------
    # 6. Copy 3 representative clips to tests/fixtures/benchmark/
    # ------------------------------------------------------------------
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixture_ids = _pick_fixture_utt_ids(items)
    print(f"\nCopying fixture clips: {fixture_ids}")
    for uid in fixture_ids:
        src = CLIPS_DIR / f"{uid}.wav"
        dst = FIXTURES_DIR / f"{uid}.wav"
        shutil.copy2(src, dst)
        print(f"  {dst}")

    # ------------------------------------------------------------------
    # 7. Build and write the content lock
    # ------------------------------------------------------------------
    lock: dict[str, str] = {}

    # All 88 clip wavs.
    for row in gold_rows:
        wav_path = CLIPS_DIR / f"{row['utt_id']}.wav"
        lock[str(wav_path.relative_to(PROJECT_ROOT))] = row["wav_sha256"]

    # gold.jsonl and manifest.json.
    lock[str(GOLD_JSONL.relative_to(PROJECT_ROOT))] = gold_sha256
    lock[str(MANIFEST_JSON.relative_to(PROJECT_ROOT))] = _sha256_file(MANIFEST_JSON)

    # Fixture clips (referenced by their tests/ path as well).
    for uid in fixture_ids:
        dst = FIXTURES_DIR / f"{uid}.wav"
        lock[str(dst.relative_to(PROJECT_ROOT))] = _sha256_file(dst)

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_content = json.dumps(
        {"benchmark_version": BENCHMARK_VERSION, "artifacts": dict(sorted(lock.items()))},
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    ) + "\n"
    LOCK_FILE.write_text(lock_content, encoding="utf-8")
    print(f"Wrote content lock → {LOCK_FILE} ({len(lock)} entries).")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    elapsed = time.monotonic() - t_start
    print("\n" + "=" * 60)
    print(f"BUILD COMPLETE — benchmark v{BENCHMARK_VERSION}")
    print(f"  items rendered : {n_rendered}")
    print(f"  total duration : {total_duration_s:.1f}s  ({total_duration_s/60:.1f}m)")
    print(f"  total bytes    : {total_bytes:,}  ({total_bytes/1024/1024:.1f} MiB)")
    print(f"  voices used    : {', '.join(sorted(voices_used))}")
    print(f"  elapsed        : {elapsed:.1f}s")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Verify mode
# ---------------------------------------------------------------------------

def _verify() -> None:
    """Re-hash every artifact in the lock; exit non-zero on ANY mismatch."""

    if not LOCK_FILE.exists():
        print(f"ERROR: lock file not found: {LOCK_FILE}", file=sys.stderr)
        sys.exit(1)

    lock_data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    artifacts: dict[str, str] = lock_data["artifacts"]
    # "tarball" is a top-level key added by make_benchmark_tarball.py; ignored here.

    print(f"Verifying {len(artifacts)} artifacts against {LOCK_FILE} …")
    failures: list[str] = []

    for rel_path, expected_sha256 in sorted(artifacts.items()):
        abs_path = PROJECT_ROOT / rel_path
        if not abs_path.exists():
            failures.append(f"MISSING  {rel_path}")
            continue

        actual = _sha256_file(abs_path)
        if actual != expected_sha256:
            failures.append(
                f"MISMATCH {rel_path}\n"
                f"  expected: {expected_sha256}\n"
                f"  actual  : {actual}"
            )
        else:
            print(f"  OK  {rel_path}")

    if failures:
        print("\n" + "=" * 60, file=sys.stderr)
        print("VERIFY FAILED — content lock violations:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"VERIFY PASSED — all {len(artifacts)} artifacts match the lock.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build or verify the ReadCoach synthetic benchmark v0.1.0."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-hash all artifacts against the content lock (CI mode).",
    )
    args = parser.parse_args()

    if args.verify:
        _verify()
    else:
        _build()
