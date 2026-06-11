"""Unit tests for the sha256-verify and safe-extract helpers in fetch_benchmark.

CI-safe: no network access, no audio tools.  The helpers are imported from
scripts/fetch_benchmark.py and exercised entirely with in-memory / tmp data.
"""
from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import helpers from the script (not an installed package).
# ---------------------------------------------------------------------------

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from fetch_benchmark import sha256_file, sha256_bytes, safe_extract, classify_existing  # noqa: E402


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


def _make_tarball_with_dirs(
    dir_names: list[str],
    file_members: list[tuple[str, bytes]],
) -> bytes:
    """Build an in-memory tar.gz with explicit directory entries followed by files."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in dir_names:
            info = tarfile.TarInfo(name=name)
            info.type = tarfile.DIRTYPE
            info.size = 0
            tar.addfile(info)
        for name, content in file_members:
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

    def test_explicit_members_list_extracts_only_those(self, tmp_path: Path) -> None:
        """safe_extract with an explicit members list extracts only those members."""
        data_a = b"file A content"
        data_b = b"file B content"
        tarball_bytes = _make_tarball([("a.txt", data_a), ("b.txt", data_b)])
        tarball_path = tmp_path / "partial.tar.gz"
        tarball_path.write_bytes(tarball_bytes)

        dest = tmp_path / "partial_out"
        dest.mkdir()

        with tarfile.open(tarball_path, "r:gz") as tar:
            # Extract only the first member.
            only_a = [m for m in tar.getmembers() if m.name == "a.txt"]
            extracted = safe_extract(tar, dest, only_a)

        assert len(extracted) == 1
        assert (dest / "a.txt").read_bytes() == data_a
        assert not (dest / "b.txt").exists()


class TestSafeExtractRootDirEntry:
    """safe_extract correctly handles tarballs with an explicit root-dir entry.

    A real-world tarball may contain an explicit directory entry for the root
    prefix (e.g. "readcoach-benchmark-0.1.0/") in addition to file entries.
    After the fetch() loop remaps names and skips the root-dir entry from
    members_to_extract, safe_extract must not create a spurious nested
    directory.
    """

    def test_root_dir_entry_plus_nested_file(self, tmp_path: Path) -> None:
        """Tarball with explicit root-dir entry + file → file lands at right
        place, NO spurious nested directory inside dest."""
        prefix = "readcoach-benchmark-0.1.0/"
        file_content = b'{"n_items": 88}'

        # Build tarball: explicit root-dir entry + one file under that prefix.
        tarball_bytes = _make_tarball_with_dirs(
            dir_names=[prefix.rstrip("/")],
            file_members=[(prefix + "manifest.json", file_content)],
        )
        tarball_path = tmp_path / "rootdir.tar.gz"
        tarball_path.write_bytes(tarball_bytes)

        dest = tmp_path / "bench"
        dest.mkdir()

        # Simulate what fetch() does: remap names, skip root-dir entry.
        with tarfile.open(tarball_path, "r:gz") as tar:
            members_to_extract: list[tarfile.TarInfo] = []
            for member in tar.getmembers():
                if member.name == prefix.rstrip("/"):
                    continue  # skip explicit root-dir entry
                if member.name.startswith(prefix):
                    remapped = member.name[len(prefix):]
                    if remapped:
                        member.name = remapped
                        members_to_extract.append(member)

            extracted = safe_extract(tar, dest, members_to_extract)

        # The file must be extracted at the remapped location.
        assert (dest / "manifest.json").read_bytes() == file_content

        # No spurious nested directory (e.g. dest/readcoach-benchmark-0.1.0/).
        spurious_dir = dest / "readcoach-benchmark-0.1.0"
        assert not spurious_dir.exists(), (
            f"Spurious nested directory created: {spurious_dir}"
        )

        # Exactly one file extracted.
        assert len(extracted) == 1

    def test_root_dir_entry_is_skipped_harmlessly(self, tmp_path: Path) -> None:
        """Passing a curated members list that includes a dir entry → dir is
        skipped, only files are counted in the returned list."""
        data = b"clip bytes"
        tarball_bytes = _make_tarball_with_dirs(
            dir_names=["clips"],
            file_members=[("clips/p01-clean.wav", data)],
        )
        tarball_path = tmp_path / "withdir.tar.gz"
        tarball_path.write_bytes(tarball_bytes)

        dest = tmp_path / "withdir_out"
        dest.mkdir()

        with tarfile.open(tarball_path, "r:gz") as tar:
            # Pass ALL members (including the "clips" dir entry) explicitly.
            extracted = safe_extract(tar, dest, tar.getmembers())

        assert len(extracted) == 1
        assert (dest / "clips" / "p01-clean.wav").read_bytes() == data


# ---------------------------------------------------------------------------
# classify_existing tests (offline, pure logic — no network)
# ---------------------------------------------------------------------------

def _make_lock_artifacts(tmp_path: Path, items: list[tuple[str, bytes]]) -> dict[str, str]:
    """Write files into tmp_path and return a lock artifacts dict."""
    import hashlib
    artifacts: dict[str, str] = {}
    for rel, content in items:
        abs_path = tmp_path / rel[len("data/benchmark/"):]
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(content)
        artifacts[rel] = hashlib.sha256(content).hexdigest()
    return artifacts


class TestClassifyExisting:
    """classify_existing(lock_artifacts, base_dir) returns {missing, mismatched, ok}."""

    def test_all_present_and_correct_is_ok(self, tmp_path: Path) -> None:
        items = [
            ("data/benchmark/gold.jsonl", b'{"utt_id":"p01-clean"}\n'),
            ("data/benchmark/clips/p01-clean.wav", b"RIFF-data"),
        ]
        artifacts = _make_lock_artifacts(tmp_path, items)
        result = classify_existing(artifacts, tmp_path)
        assert result["ok"] == sorted([k for k, _ in items])
        assert result["missing"] == []
        assert result["mismatched"] == []

    def test_missing_only_is_missing_not_mismatched(self, tmp_path: Path) -> None:
        """Files absent from disk → missing[], NOT mismatched[]. Download proceeds."""
        # Write gold.jsonl but NOT the clip.
        gold_content = b'{"utt_id":"p01-clean"}\n'
        import hashlib
        gold_path = tmp_path / "gold.jsonl"
        gold_path.write_bytes(gold_content)
        artifacts = {
            "data/benchmark/gold.jsonl": hashlib.sha256(gold_content).hexdigest(),
            "data/benchmark/clips/p01-clean.wav": "a" * 64,  # clip absent
        }
        result = classify_existing(artifacts, tmp_path)
        assert "data/benchmark/clips/p01-clean.wav" in result["missing"]
        assert "data/benchmark/gold.jsonl" in result["ok"]
        assert result["mismatched"] == []

    def test_hash_mismatch_is_mismatched_not_missing(self, tmp_path: Path) -> None:
        """File present but wrong hash → mismatched[], NOT missing[]."""
        clip_path = tmp_path / "clips" / "p01-clean.wav"
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        clip_path.write_bytes(b"original bytes")
        artifacts = {
            "data/benchmark/clips/p01-clean.wav": "b" * 64,  # wrong hash
        }
        result = classify_existing(artifacts, tmp_path)
        assert "data/benchmark/clips/p01-clean.wav" in result["mismatched"]
        assert result["missing"] == []
        assert result["ok"] == []

    def test_mixed_missing_and_mismatched(self, tmp_path: Path) -> None:
        """Some missing + some mismatched → both lists populated correctly."""
        import hashlib
        good_content = b"good file"
        bad_content = b"tampered"

        # Write the good file and the bad file (tampered hash).
        (tmp_path / "gold.jsonl").write_bytes(good_content)
        (tmp_path / "clips").mkdir()
        (tmp_path / "clips" / "p01-clean.wav").write_bytes(bad_content)
        # The clip for p02 is simply absent.

        artifacts = {
            "data/benchmark/gold.jsonl": hashlib.sha256(good_content).hexdigest(),
            "data/benchmark/clips/p01-clean.wav": "c" * 64,  # wrong hash
            "data/benchmark/clips/p02-clean.wav": "d" * 64,  # absent
        }
        result = classify_existing(artifacts, tmp_path)
        assert result["ok"] == ["data/benchmark/gold.jsonl"]
        assert result["mismatched"] == ["data/benchmark/clips/p01-clean.wav"]
        assert result["missing"] == ["data/benchmark/clips/p02-clean.wav"]

    def test_non_benchmark_prefix_artifacts_ignored(self, tmp_path: Path) -> None:
        """Artifacts outside data/benchmark/ prefix are not checked."""
        artifacts = {
            "tests/fixtures/something.wav": "e" * 64,
            "data/benchmark/gold.jsonl": "f" * 64,  # absent → missing
        }
        result = classify_existing(artifacts, tmp_path)
        # only benchmark artifacts are checked; tests/fixtures key ignored
        assert result["missing"] == ["data/benchmark/gold.jsonl"]
        assert result["ok"] == []
        assert result["mismatched"] == []

    def test_empty_artifacts_returns_all_empty(self, tmp_path: Path) -> None:
        result = classify_existing({}, tmp_path)
        assert result == {"missing": [], "mismatched": [], "ok": []}


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
