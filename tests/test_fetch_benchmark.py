"""Unit tests for the sha256-verify and safe-extract helpers in fetch_benchmark.

CI-safe: no network access, no audio tools.  The helpers are imported from
scripts/fetch_benchmark.py and exercised entirely with in-memory / tmp data.
"""
from __future__ import annotations

import hashlib
import io
import os
import tarfile
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import helpers from the script (not an installed package).
# ---------------------------------------------------------------------------

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from fetch_benchmark import sha256_file, sha256_bytes, safe_extract  # noqa: E402


# ---------------------------------------------------------------------------
# sha256_file tests
# ---------------------------------------------------------------------------

class TestSha256File:
    def test_correct_hash_of_known_content(self, tmp_path: Path) -> None:
        content = b"ReadCoach benchmark test content"
        expected = hashlib.sha256(content).hexdigest()
        f = tmp_path / "test.bin"
        f.write_bytes(content)
        assert sha256_file(f) == expected

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert sha256_file(f) == hashlib.sha256(b"").hexdigest()

    def test_tampered_byte_produces_different_hash(self, tmp_path: Path) -> None:
        content = b"original content for tamper test"
        f = tmp_path / "original.bin"
        f.write_bytes(content)
        original_hash = sha256_file(f)

        # Tamper: flip one byte.
        tampered = bytearray(content)
        tampered[0] ^= 0xFF
        f.write_bytes(bytes(tampered))
        tampered_hash = sha256_file(f)

        assert tampered_hash != original_hash, (
            "Tampered file must produce a different sha256"
        )


class TestSha256Bytes:
    def test_correct_hash_of_known_bytes(self) -> None:
        data = b"hello world"
        assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()

    def test_empty_bytes(self) -> None:
        assert sha256_bytes(b"") == hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# safe_extract tests
# ---------------------------------------------------------------------------

def _make_tarball(members: list[tuple[str, bytes]]) -> bytes:
    """Build an in-memory tar.gz with the given (name, content) members."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    buf.seek(0)
    return buf.read()


class TestSafeExtract:
    def test_normal_extraction_succeeds(self, tmp_path: Path) -> None:
        data = b"clip content"
        tarball_bytes = _make_tarball([("clips/p01-clean.wav", data)])
        tarball_path = tmp_path / "test.tar.gz"
        tarball_path.write_bytes(tarball_bytes)

        dest = tmp_path / "out"
        dest.mkdir()
        with tarfile.open(tarball_path, "r:gz") as tar:
            extracted = safe_extract(tar, dest)

        assert len(extracted) == 1
        assert (dest / "clips" / "p01-clean.wav").read_bytes() == data

    def test_dot_dot_path_is_rejected(self, tmp_path: Path) -> None:
        """A tarball member with a path-traversal component must be rejected."""
        # Craft a member whose resolved path escapes the dest directory.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="../escape.txt")
            content = b"malicious"
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        buf.seek(0)

        tarball_path = tmp_path / "escape.tar.gz"
        tarball_path.write_bytes(buf.getvalue())

        dest = tmp_path / "safe_dest"
        dest.mkdir()

        with tarfile.open(tarball_path, "r:gz") as tar:
            with pytest.raises(ValueError, match="escape"):
                safe_extract(tar, dest)

        # Confirm the file was NOT written outside dest.
        assert not (tmp_path / "escape.txt").exists()

    def test_absolute_path_member_is_rejected(self, tmp_path: Path) -> None:
        """Tarball members with absolute paths must be rejected."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="/etc/passwd")
            content = b"overwrite"
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        buf.seek(0)

        tarball_path = tmp_path / "abs.tar.gz"
        tarball_path.write_bytes(buf.getvalue())

        dest = tmp_path / "safe_dest2"
        dest.mkdir()

        with tarfile.open(tarball_path, "r:gz") as tar:
            with pytest.raises(ValueError, match="escape"):
                safe_extract(tar, dest)

        assert not Path("/etc/passwd_readcoach_test").exists()

    def test_multiple_files_extracted_correctly(self, tmp_path: Path) -> None:
        members = [
            ("gold.jsonl", b'{"utt_id": "p01-clean"}\n'),
            ("manifest.json", b'{"n_items": 88}\n'),
            ("clips/p01-clean.wav", b"RIFF...."),
        ]
        tarball_bytes = _make_tarball(members)
        tarball_path = tmp_path / "multi.tar.gz"
        tarball_path.write_bytes(tarball_bytes)

        dest = tmp_path / "multi_out"
        dest.mkdir()

        with tarfile.open(tarball_path, "r:gz") as tar:
            extracted = safe_extract(tar, dest)

        assert len(extracted) == 3
        assert (dest / "gold.jsonl").read_bytes() == members[0][1]
        assert (dest / "manifest.json").read_bytes() == members[1][1]
        assert (dest / "clips" / "p01-clean.wav").read_bytes() == members[2][1]


# ---------------------------------------------------------------------------
# Lock structure test (CI-safe: just reads the committed lock file)
# ---------------------------------------------------------------------------

class TestLockHasTarballEntry:
    """The committed lock must have a 'tarball' top-level key with sha256 + bytes."""

    def test_tarball_key_present_in_lock(self) -> None:
        import json
        lock_path = Path(__file__).parent.parent / "evals" / "golden" / "benchmark.lock"
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        assert "tarball" in data, "benchmark.lock is missing 'tarball' key"
        t = data["tarball"]
        assert "sha256" in t, "benchmark.lock tarball entry missing 'sha256'"
        assert "bytes" in t, "benchmark.lock tarball entry missing 'bytes'"
        assert "name" in t, "benchmark.lock tarball entry missing 'name'"
        assert len(t["sha256"]) == 64, "tarball sha256 should be 64 hex chars"
        assert t["bytes"] > 0, "tarball bytes should be positive"
