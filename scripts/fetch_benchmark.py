#!/usr/bin/env python3
"""Download and verify the ReadCoach synthetic miscue benchmark v0.1.0.

STDLIB ONLY — no third-party packages.  Run with any Python >= 3.8:

    python3 scripts/fetch_benchmark.py [--force] [--dest PATH]

Flow
----
1.  Download the release tarball from GitHub (progress to stderr).
2.  Verify the tarball sha256 against ``evals/golden/benchmark.lock``
    (the ``tarball.sha256`` entry committed with the lock).
3.  Extract into ``data/benchmark/`` (gold.jsonl, manifest.json, clips/).
4.  Verify every extracted file's sha256 against the lock ``artifacts``
    entries.
5.  Print summary; exit 0.  ANY mismatch or missing file → loud error,
    non-zero exit, no partial-success message.

Without ``--force``: if ``data/benchmark/`` already exists and every
artifact verifies, report success and exit 0 without downloading.

Safe extraction: paths escaping the target directory are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Release constants
# ---------------------------------------------------------------------------

RELEASE_TAG = "benchmark-v0.1.0"
ASSET_NAME = "readcoach-benchmark-0.1.0.tar.gz"
DOWNLOAD_URL = (
    f"https://github.com/gorajing/readcoach/releases/download/{RELEASE_TAG}/{ASSET_NAME}"
)

# ---------------------------------------------------------------------------
# Paths (resolved relative to this script's project root)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
LOCK_FILE = PROJECT_ROOT / "evals" / "golden" / "benchmark.lock"
DEFAULT_DEST = PROJECT_ROOT / "data" / "benchmark"


# ---------------------------------------------------------------------------
# SHA-256 helpers (importable for unit tests, no side effects)
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of *data*."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Safe extraction helper (importable for unit tests)
# ---------------------------------------------------------------------------

def safe_extract(
    tar: tarfile.TarFile,
    dest: Path,
    members: list[tarfile.TarInfo] | None = None,
) -> list[Path]:
    """Extract *members* from *tar* into *dest*, rejecting any member whose
    resolved path escapes *dest*.

    If *members* is ``None``, all members of *tar* are extracted (original
    behaviour).  Pass an explicit list to extract only a curated subset (e.g.
    after remapping names to strip a root prefix).

    Directory entries are skipped — ``tarfile.extract`` creates parent
    directories as needed, so explicit dir entries are never required.

    Returns the list of extracted absolute file paths.

    Raises ``ValueError`` for any member that would escape the destination.
    This guards against tarbombs and directory-traversal payloads.
    """
    dest_resolved = dest.resolve()
    extracted: list[Path] = []

    iter_members = members if members is not None else tar.getmembers()

    for member in iter_members:
        # Skip directory entries — parent dirs are created automatically.
        if member.isdir():
            continue

        # Compute what the resolved path would be.
        # Strip leading '/' or '..' components before joining.
        member_path = (dest / member.name).resolve()

        # Reject paths that escape dest.
        try:
            member_path.relative_to(dest_resolved)
        except ValueError:
            raise ValueError(
                f"Tarball member {member.name!r} would escape extraction "
                f"directory {dest_resolved} — aborting."
            )

        # Use filter="data" when available (Python >= 3.12); fall back for older.
        try:
            tar.extract(member, path=dest, set_attrs=False, filter="data")
        except TypeError:
            tar.extract(member, path=dest, set_attrs=False)
        if member.isfile():
            extracted.append(dest / member.name)

    return extracted


# ---------------------------------------------------------------------------
# Lock loading
# ---------------------------------------------------------------------------

def _load_lock() -> dict:
    if not LOCK_FILE.exists():
        print(
            f"ERROR: lock file not found: {LOCK_FILE}\n"
            "This file is committed in the repository and is required to verify downloads.",
            file=sys.stderr,
        )
        sys.exit(1)
    return json.loads(LOCK_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Classify existing artifacts (pure function — importable for unit tests)
# ---------------------------------------------------------------------------

def classify_existing(
    artifacts: dict[str, str],
    base_dir: Path,
) -> dict[str, list[str]]:
    """Classify every lock artifact under ``data/benchmark/`` into one of three buckets.

    Parameters
    ----------
    artifacts:
        The ``artifacts`` dict from the lock — keys are paths relative to the
        project root (e.g. ``"data/benchmark/clips/p01-clean.wav"``), values are
        expected SHA-256 hex strings.
    base_dir:
        The local extraction directory (``data/benchmark/``).  The function
        remaps each lock key to a path inside *base_dir* for the check.

    Returns
    -------
    A dict with three sorted lists of lock-relative path strings:

    ``"missing"``
        Artifact whose file does not exist on disk at all.  *Nothing to clobber*
        — downloading is safe without ``--force``.
    ``"mismatched"``
        Artifact that exists on disk but whose sha256 does not match the lock.
        *Locally modified / corrupted* — requires ``--force`` to overwrite.
    ``"ok"``
        Artifact that exists and whose sha256 matches the lock exactly.
    """
    benchmark_prefix = "data/benchmark/"
    missing: list[str] = []
    mismatched: list[str] = []
    ok: list[str] = []

    for rel_path, expected in sorted(artifacts.items()):
        if not rel_path.startswith(benchmark_prefix):
            continue  # non-benchmark artifact (e.g. test fixtures) — skip
        sub = rel_path[len(benchmark_prefix):]
        abs_path = base_dir / sub
        if not abs_path.exists():
            missing.append(rel_path)
        elif sha256_file(abs_path) != expected:
            mismatched.append(rel_path)
        else:
            ok.append(rel_path)

    return {"missing": missing, "mismatched": mismatched, "ok": ok}


# ---------------------------------------------------------------------------
# Verify all extracted artifacts against the lock
# ---------------------------------------------------------------------------

def _verify_artifacts(dest: Path, artifacts: dict[str, str]) -> bool:
    """Return True if every lock artifact under data/benchmark/ exists and matches.

    Does not abort — returns False so the caller can decide.
    """
    classification = classify_existing(artifacts, dest)
    failures: list[str] = []

    for rel_path in classification["missing"]:
        failures.append(f"MISSING   {rel_path}")
    for rel_path in classification["mismatched"]:
        # Reconstruct verbose mismatch message.
        benchmark_prefix = "data/benchmark/"
        sub = rel_path[len(benchmark_prefix):]
        abs_path = dest / sub
        actual = sha256_file(abs_path)
        expected = artifacts[rel_path]
        failures.append(
            f"MISMATCH  {rel_path}\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual}"
        )

    if failures:
        print("\nERROR: artifact verification FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Download with progress
# ---------------------------------------------------------------------------

def _download(url: str, dest_file: Path) -> None:
    print(f"Downloading {url}", file=sys.stderr)

    def _reporthook(block_count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            done = min(block_count * block_size, total_size)
            pct = done * 100 // total_size
            mb_done = done / 1_048_576
            mb_total = total_size / 1_048_576
            print(
                f"\r  {pct:3d}%  {mb_done:.1f}/{mb_total:.1f} MiB",
                end="",
                file=sys.stderr,
                flush=True,
            )
        else:
            done = block_count * block_size
            print(
                f"\r  {done / 1_048_576:.1f} MiB downloaded",
                end="",
                file=sys.stderr,
                flush=True,
            )

    urllib.request.urlretrieve(url, dest_file, reporthook=_reporthook)
    print(file=sys.stderr)  # newline after progress


# ---------------------------------------------------------------------------
# Main fetch logic
# ---------------------------------------------------------------------------

def fetch(dest: Path = DEFAULT_DEST, force: bool = False) -> None:
    lock_data = _load_lock()
    tarball_meta: dict = lock_data.get("tarball", {})
    expected_tarball_sha256: str = tarball_meta.get("sha256", "")
    expected_tarball_bytes: int = tarball_meta.get("bytes", 0)
    artifacts: dict[str, str] = lock_data.get("artifacts", {})

    if not expected_tarball_sha256:
        print(
            "ERROR: benchmark.lock has no 'tarball.sha256' entry — "
            "update the lock (run make_benchmark_tarball.py and record the sha256).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Classify what is already on disk (when dest exists and --force is off).
    # Three outcomes:
    #
    #   all ok          → already complete; skip download, exit 0.
    #   missing only    → normal incomplete install; proceed to download.
    #                     (gold.jsonl + manifest.json are committed, so
    #                     data/benchmark/ always exists on a fresh clone even
    #                     though the 88 clips are absent — nothing to clobber.)
    #   any mismatched  → locally modified / corrupted; refuse without --force.
    # ------------------------------------------------------------------
    if dest.exists() and not force:
        classification = classify_existing(artifacts, dest)
        n_ok = len(classification["ok"])
        n_missing = len(classification["missing"])
        n_mismatched = len(classification["mismatched"])
        n_checked = sum(1 for k in artifacts if k.startswith("data/benchmark/"))

        if n_mismatched == 0 and n_missing == 0:
            # All present and verified — nothing to do.
            print(
                f"data/benchmark/ already complete: {n_ok}/{n_checked} artifact(s) verified. "
                "Use --force to re-download."
            )
            return

        if n_mismatched > 0:
            # Some files exist but have the wrong hash — protect them.
            msg_parts = [f"{n_mismatched} artifact(s) present but hash-MISMATCHED"]
            if n_missing > 0:
                msg_parts.append(f"{n_missing} artifact(s) missing")
            print(
                "ERROR: " + "; ".join(msg_parts) + ".\n"
                "Re-run with --force to overwrite.",
                file=sys.stderr,
            )
            sys.exit(1)

        # n_mismatched == 0 and n_missing > 0 — fall through to download.
        print(
            f"data/benchmark/ exists but {n_missing}/{n_checked} artifact(s) are missing "
            "(normal on a fresh clone). Downloading …",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Download to a temp file, verify tarball sha256.
    # ------------------------------------------------------------------
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="readcoach_fetch_") as tmpdir:
        tmp_tarball = Path(tmpdir) / ASSET_NAME

        _download(DOWNLOAD_URL, tmp_tarball)

        actual_bytes = tmp_tarball.stat().st_size
        print(
            f"  Downloaded {actual_bytes:,} bytes "
            f"(expected {expected_tarball_bytes:,}).",
            file=sys.stderr,
        )

        print("Verifying tarball sha256 …", file=sys.stderr)
        actual_sha256 = sha256_file(tmp_tarball)
        if actual_sha256 != expected_tarball_sha256:
            print(
                f"\nERROR: tarball sha256 MISMATCH — aborting, nothing extracted.\n"
                f"  expected : {expected_tarball_sha256}\n"
                f"  actual   : {actual_sha256}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"  Tarball sha256 OK ({actual_sha256[:16]}…)", file=sys.stderr)

        # ------------------------------------------------------------------
        # Safe extraction.
        # ------------------------------------------------------------------
        print(f"Extracting into {dest} …", file=sys.stderr)
        with tarfile.open(tmp_tarball, "r:gz") as tar:
            # Strip the leading "readcoach-benchmark-0.1.0/" prefix by
            # extracting each member individually with a remapped name.
            prefix = "readcoach-benchmark-0.1.0/"
            members_to_extract: list[tarfile.TarInfo] = []
            for member in tar.getmembers():
                if member.name == prefix.rstrip("/"):
                    continue  # skip the root dir entry
                if not member.name.startswith(prefix):
                    raise ValueError(
                        f"Unexpected tarball member {member.name!r} — "
                        "does not share the expected root prefix."
                    )
                # Remap: strip the leading prefix.
                remapped = member.name[len(prefix):]
                if not remapped:
                    continue
                member.name = remapped
                members_to_extract.append(member)

            safe_extract(tar, dest, members_to_extract)

    # ------------------------------------------------------------------
    # Verify every extracted artifact against the lock.
    # ------------------------------------------------------------------
    print("Verifying extracted artifacts …", file=sys.stderr)
    ok = _verify_artifacts(dest, artifacts)
    if not ok:
        print(
            "\nERROR: Post-extraction verification FAILED — "
            "data/benchmark/ is in an inconsistent state.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_checked = sum(1 for k in artifacts if k.startswith("data/benchmark/"))
    print(
        f"\nSUCCESS: {n_checked} artifact(s) downloaded, extracted, and verified."
    )
    print(f"  Benchmark is ready at {dest}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify the ReadCoach synthetic miscue benchmark v0.1.0. "
            "stdlib only — works without installing any project dependencies."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-download even if data/benchmark/ already exists and verifies. "
            "Without this flag, an already-verified benchmark exits 0 immediately."
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        metavar="PATH",
        help=f"Directory to extract the benchmark into (default: {DEFAULT_DEST}).",
    )
    args = parser.parse_args()
    fetch(dest=args.dest, force=args.force)


if __name__ == "__main__":
    main()
