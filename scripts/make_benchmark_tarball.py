#!/usr/bin/env python3
"""Pack the rendered benchmark into a versioned release tarball.

USAGE
-----
    uv run python scripts/make_benchmark_tarball.py

Produces:
    dist/readcoach-benchmark-0.1.0.tar.gz

The tarball root is ``readcoach-benchmark-0.1.0/`` and contains:
    gold.jsonl
    manifest.json
    clips/*.wav  (88 clips)

Before packing, the content lock is verified (same logic as
``build_benchmark.py --verify``).  Any mismatch aborts with exit code 1.

The tarball's own sha256 and byte size are printed on stdout.  The
caller is responsible for recording that sha256 into the lock file
(the ``tarball`` top-level key) before committing.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BENCHMARK_DIR = DATA_DIR / "benchmark"
CLIPS_DIR = BENCHMARK_DIR / "clips"
GOLD_JSONL = BENCHMARK_DIR / "gold.jsonl"
MANIFEST_JSON = BENCHMARK_DIR / "manifest.json"
LOCK_FILE = PROJECT_ROOT / "evals" / "golden" / "benchmark.lock"
DIST_DIR = PROJECT_ROOT / "dist"

BENCHMARK_VERSION = "0.1.0"
TARBALL_NAME = f"readcoach-benchmark-{BENCHMARK_VERSION}.tar.gz"
TARBALL_ROOT = f"readcoach-benchmark-{BENCHMARK_VERSION}"


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Lock verify (mirrors build_benchmark._verify, but operates on artifact entries only)
# ---------------------------------------------------------------------------

def _verify_lock() -> None:
    """Re-hash every artifact in the lock; abort on any mismatch."""

    if not LOCK_FILE.exists():
        print(f"ERROR: lock file not found: {LOCK_FILE}", file=sys.stderr)
        sys.exit(1)

    lock_data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    artifacts: dict[str, str] = lock_data["artifacts"]

    print(f"Pre-pack verify: checking {len(artifacts)} artifact(s) in lock …")
    failures: list[str] = []

    for rel_path, expected in sorted(artifacts.items()):
        abs_path = PROJECT_ROOT / rel_path
        if not abs_path.exists():
            failures.append(f"MISSING  {rel_path}")
            continue
        actual = _sha256_file(abs_path)
        if actual != expected:
            failures.append(
                f"MISMATCH {rel_path}\n"
                f"  expected: {expected}\n"
                f"  actual  : {actual}"
            )

    if failures:
        print("\nERROR: lock verify failed — aborting tarball build.", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)

    print(f"  All {len(artifacts)} artifacts OK.\n")


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------

def _pack() -> None:
    t_start = time.monotonic()

    # 1. Verify lock first.
    _verify_lock()

    # 2. Collect files to pack.
    files_to_pack: list[tuple[Path, str]] = []

    # gold.jsonl
    files_to_pack.append((GOLD_JSONL, f"{TARBALL_ROOT}/gold.jsonl"))

    # manifest.json
    files_to_pack.append((MANIFEST_JSON, f"{TARBALL_ROOT}/manifest.json"))

    # clips/*.wav (88)
    clip_wavs = sorted(CLIPS_DIR.glob("*.wav"))
    if len(clip_wavs) != 88:
        print(
            f"ERROR: expected 88 clips in {CLIPS_DIR}, found {len(clip_wavs)}",
            file=sys.stderr,
        )
        sys.exit(1)
    for wav in clip_wavs:
        files_to_pack.append((wav, f"{TARBALL_ROOT}/clips/{wav.name}"))

    print(
        f"Packing {len(files_to_pack)} files into {TARBALL_NAME} …"
        f"  (1 gold.jsonl + 1 manifest.json + {len(clip_wavs)} clips)"
    )

    # 3. Write tarball.
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    tarball_path = DIST_DIR / TARBALL_NAME

    with tarfile.open(tarball_path, "w:gz") as tar:
        for src_path, arc_name in files_to_pack:
            tar.add(src_path, arcname=arc_name)

    # 4. Compute sha256 + size.
    tarball_sha256 = _sha256_file(tarball_path)
    tarball_bytes = tarball_path.stat().st_size
    elapsed = time.monotonic() - t_start

    print()
    print("=" * 60)
    print(f"TARBALL BUILT — {tarball_path}")
    print(f"  name   : {TARBALL_NAME}")
    print(f"  sha256 : {tarball_sha256}")
    print(f"  bytes  : {tarball_bytes:,}  ({tarball_bytes / 1024 / 1024:.1f} MiB)")
    print(f"  elapsed: {elapsed:.1f}s")
    print("=" * 60)
    print()
    print("Record the tarball sha256 in evals/golden/benchmark.lock")
    print('under the top-level key "tarball" before committing.')


if __name__ == "__main__":
    _pack()
