#!/usr/bin/env python3
"""Regenerate evals/golden/asr_cache_manifest.json.

MAINTAINER-ONLY — requires data/benchmark/clips/*.wav locally (excluded from
git via .gitignore).  Build or fetch the clips first with build_benchmark.py /
fetch_benchmark.py.

The manifest maps every (utt_id × bias × backend) triple to the content-addressed
cache key (hex filename stem) produced by transcribe().  run_benchmark.py reads
this manifest in --fixtures mode so CI can look up ASR results without audio.

USAGE
-----
Regenerate (writes evals/golden/asr_cache_manifest.json):

    uv run python scripts/build_cache_manifest.py

Verify (check that the committed manifest is byte-equivalent to what would be
regenerated — exits non-zero on any mismatch; no files written):

    uv run python scripts/build_cache_manifest.py --check

Key function
------------
The cache key is reused directly from readcoach.asr._cache_key — not
duplicated here — so this script is always in sync with transcribe().
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is importable when invoked via `uv run python scripts/...`
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from readcoach.asr import _cache_key, _CACHE_DIR  # noqa: E402 — after sys.path fixup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_GOLD_JSONL = _PROJECT_ROOT / "data" / "benchmark" / "gold.jsonl"
_CLIPS_DIR = _PROJECT_ROOT / "data" / "benchmark" / "clips"
_MANIFEST_OUT = _PROJECT_ROOT / "evals" / "golden" / "asr_cache_manifest.json"

_ALL_BIASES = ("none", "prompt", "strong")
_DEFAULT_BACKEND = "faster-whisper-small"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gold_rows() -> list[dict]:
    rows: list[dict] = []
    with _GOLD_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_manifest(backend: str) -> dict[str, str]:
    """Return the full manifest dict, aborting loudly on any missing cache file."""
    rows = _load_gold_rows()
    manifest: dict[str, str] = {}
    missing: list[str] = []

    for row in rows:
        utt_id: str = row["utt_id"]
        target_text: str = row["target_text"]
        clip_path = str(_CLIPS_DIR / f"{utt_id}.wav")

        if not Path(clip_path).exists():
            print(
                f"ABORT: clip missing for utt_id={utt_id!r}: {clip_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        for bias in _ALL_BIASES:
            # bias="none" must NOT include target_text in the key — mirrors
            # the exact logic in run_benchmark.py and transcribe().
            asr_target = None if bias == "none" else target_text

            key_hex = _cache_key(
                clip_path,
                target_text=asr_target,
                bias=bias,
                backend=backend,
            )

            cache_file = _CACHE_DIR / f"{key_hex}.json"
            if not cache_file.exists():
                missing.append(f"{utt_id}|{bias}|{backend}")
                continue

            manifest_key = f"{utt_id}|{bias}|{backend}"
            manifest[manifest_key] = key_hex

    if missing:
        print(
            f"\nABORT: {len(missing)} cache file(s) missing — "
            "run the full benchmark without --fixtures to populate them:",
            file=sys.stderr,
        )
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    return dict(sorted(manifest.items()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate evals/golden/asr_cache_manifest.json (maintainer-only; "
            "requires data/benchmark/clips/*.wav)."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help=(
            "Verify mode: recompute the manifest and check it matches the "
            "committed file byte-for-byte.  Exits non-zero on any mismatch.  "
            "No files are written."
        ),
    )
    parser.add_argument(
        "--backend",
        default=_DEFAULT_BACKEND,
        help=f"ASR backend identifier (default: {_DEFAULT_BACKEND}).",
    )
    args = parser.parse_args(argv)

    manifest = _build_manifest(args.backend)
    n = len(manifest)

    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)

    if args.check:
        if not _MANIFEST_OUT.exists():
            print(
                f"ERROR: committed manifest not found at {_MANIFEST_OUT}",
                file=sys.stderr,
            )
            sys.exit(1)
        committed = _MANIFEST_OUT.read_text(encoding="utf-8")
        if manifest_json == committed:
            print(
                f"OK — committed manifest is EXACTLY reproduced ({n}/{n} entries match)."
            )
        else:
            # Find first differing key for a useful error message.
            committed_dict: dict = json.loads(committed)
            new_keys = set(manifest)
            old_keys = set(committed_dict)
            added = sorted(new_keys - old_keys)
            removed = sorted(old_keys - new_keys)
            changed = sorted(
                k for k in new_keys & old_keys if manifest[k] != committed_dict[k]
            )
            print(
                "MISMATCH — committed manifest does not match recomputed manifest.",
                file=sys.stderr,
            )
            if added:
                print(f"  added ({len(added)}): {added[:5]}{'...' if len(added) > 5 else ''}",
                      file=sys.stderr)
            if removed:
                print(f"  removed ({len(removed)}): {removed[:5]}{'...' if len(removed) > 5 else ''}",
                      file=sys.stderr)
            if changed:
                print(f"  changed ({len(changed)}): {changed[:5]}{'...' if len(changed) > 5 else ''}",
                      file=sys.stderr)
            sys.exit(1)
        return

    # Write mode.
    _MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST_OUT.write_text(manifest_json, encoding="utf-8")
    print(f"Wrote {_MANIFEST_OUT} ({n} entries).")  # no trailing newline — matches committed format


if __name__ == "__main__":
    main()
