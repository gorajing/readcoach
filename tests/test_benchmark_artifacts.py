"""CI-safe tests for the committed benchmark artifacts (T1.3b).

No audio tools, no TTS, no network.  Everything is validated against
the committed gold.jsonl, manifest.json, benchmark.lock, and the 3
fixture clips.

CONTRACT
--------
* gold.jsonl: parseable; exactly 88 lines; unique utt_ids; every line
  schema-complete; gold indices valid against target word counts.
* manifest.json: n_items == 88; coverage matrix has all 40 cells >= 3;
  gold_jsonl_sha256 matches the committed file.
* benchmark.lock: every fixture wav is listed; lock's gold/manifest
  hashes match disk.
* Fixture clips: sha256 of each fixture clip matches its lock entry.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path constants (relative to project root, all kept absolute in code)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
GOLD_JSONL = BENCHMARK_DIR / "gold.jsonl"
MANIFEST_JSON = BENCHMARK_DIR / "manifest.json"
LOCK_FILE = PROJECT_ROOT / "evals" / "golden" / "benchmark.lock"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "benchmark"

_EXPECTED_CLASSES = frozenset(
    {"substitution", "omission", "insertion", "self_correction", "hesitation"}
)
_EXPECTED_PASSAGES = {f"p0{i}" for i in range(1, 9)}
_EXPECTED_N_ITEMS = 88

# Required keys in every gold.jsonl row.
_ROW_REQUIRED_KEYS = {
    "utt_id",
    "passage_id",
    "band",
    "target_text",
    "miscued_text",
    "gold",
    "voice",
    "rate_wpm",
    "wav_sha256",
    "duration_s",
}
_GOLD_ENTRY_REQUIRED_KEYS = {"type", "target_word", "said_word", "index", "render"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_gold_rows() -> list[dict]:
    """Parse gold.jsonl into a list of dicts.  Fails loud on any parse error."""
    rows: list[dict] = []
    for lineno, line in enumerate(GOLD_JSONL.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"gold.jsonl line {lineno} is not valid JSON: {exc}") from exc
    return rows


def _load_lock() -> dict[str, str]:
    """Return the artifacts dict from benchmark.lock."""
    data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    return data["artifacts"]


# ---------------------------------------------------------------------------
# gold.jsonl tests
# ---------------------------------------------------------------------------

class TestGoldJsonl:
    """Tests that require only the committed JSONL file — no audio tools."""

    def test_file_exists(self) -> None:
        assert GOLD_JSONL.exists(), f"gold.jsonl missing: {GOLD_JSONL}"

    def test_parses_to_exactly_88_lines(self) -> None:
        rows = _load_gold_rows()
        assert len(rows) == _EXPECTED_N_ITEMS, (
            f"expected {_EXPECTED_N_ITEMS} lines, got {len(rows)}"
        )

    def test_utt_ids_unique(self) -> None:
        rows = _load_gold_rows()
        ids = [r["utt_id"] for r in rows]
        assert len(ids) == len(set(ids)), "duplicate utt_ids in gold.jsonl"

    def test_every_row_schema_complete(self) -> None:
        rows = _load_gold_rows()
        for row in rows:
            missing = _ROW_REQUIRED_KEYS - set(row.keys())
            assert not missing, (
                f"row {row.get('utt_id', '?')} missing keys: {missing}"
            )
            assert isinstance(row["gold"], list), (
                f"row {row['utt_id']}: 'gold' must be a list"
            )
            for entry in row["gold"]:
                emissing = _GOLD_ENTRY_REQUIRED_KEYS - set(entry.keys())
                assert not emissing, (
                    f"row {row['utt_id']} gold entry missing keys: {emissing}"
                )

    def test_gold_entry_types_valid(self) -> None:
        rows = _load_gold_rows()
        for row in rows:
            for entry in row["gold"]:
                assert entry["type"] in _EXPECTED_CLASSES, (
                    f"row {row['utt_id']}: unknown gold type {entry['type']!r}"
                )

    def test_gold_indices_within_target_word_count(self) -> None:
        rows = _load_gold_rows()
        for row in rows:
            n_words = len(row["target_text"].split())
            for entry in row["gold"]:
                idx = entry["index"]
                assert 0 <= idx < n_words, (
                    f"row {row['utt_id']}: gold index {idx} out of range [0, {n_words})"
                )

    def test_passage_ids_cover_all_8_passages(self) -> None:
        rows = _load_gold_rows()
        found = {r["passage_id"] for r in rows}
        assert found == _EXPECTED_PASSAGES, (
            f"passage_ids mismatch: got {found}, expected {_EXPECTED_PASSAGES}"
        )

    def test_bands_one_through_four(self) -> None:
        rows = _load_gold_rows()
        for row in rows:
            assert row["band"] in (1, 2, 3, 4), (
                f"row {row['utt_id']}: band {row['band']!r} not in 1..4"
            )

    def test_duration_s_positive(self) -> None:
        rows = _load_gold_rows()
        for row in rows:
            assert row["duration_s"] > 0, (
                f"row {row['utt_id']}: duration_s must be positive"
            )


# ---------------------------------------------------------------------------
# manifest.json tests
# ---------------------------------------------------------------------------

class TestManifest:
    """Tests against the committed manifest.json."""

    def test_file_exists(self) -> None:
        assert MANIFEST_JSON.exists(), f"manifest.json missing: {MANIFEST_JSON}"

    def test_n_items_is_88(self) -> None:
        m = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        assert m["n_items"] == _EXPECTED_N_ITEMS, (
            f"manifest n_items={m['n_items']}, expected {_EXPECTED_N_ITEMS}"
        )

    def test_coverage_matrix_all_40_cells_ge_3(self) -> None:
        """8 passages × 5 classes = 40 cells; each must have >= 3 items."""
        m = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        matrix = m["coverage_matrix"]
        assert set(matrix.keys()) == _EXPECTED_PASSAGES, (
            f"coverage_matrix passages mismatch: {set(matrix.keys())}"
        )
        for pid, by_class in matrix.items():
            for cls in _EXPECTED_CLASSES:
                count = by_class.get(cls, 0)
                assert count >= 3, (
                    f"coverage_matrix[{pid}][{cls}] = {count} (need >= 3)"
                )

    def test_gold_jsonl_sha256_matches_committed_file(self) -> None:
        m = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        expected = m["gold_jsonl_sha256"]
        actual = _sha256_str(GOLD_JSONL.read_text(encoding="utf-8"))
        assert actual == expected, (
            f"manifest gold_jsonl_sha256 mismatch:\n"
            f"  manifest: {expected}\n"
            f"  disk    : {actual}"
        )

    def test_seed_is_42(self) -> None:
        m = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        assert m["seed"] == 42, f"manifest seed={m['seed']}, expected 42"

    def test_benchmark_version_present(self) -> None:
        m = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        assert "benchmark_version" in m
        assert m["benchmark_version"]


# ---------------------------------------------------------------------------
# Content lock tests
# ---------------------------------------------------------------------------

class TestBenchmarkLock:
    """Tests that the committed lock file is self-consistent."""

    def test_lock_file_exists(self) -> None:
        assert LOCK_FILE.exists(), f"benchmark.lock missing: {LOCK_FILE}"

    def test_lock_contains_gold_and_manifest(self) -> None:
        artifacts = _load_lock()
        gold_rel = str(GOLD_JSONL.relative_to(PROJECT_ROOT))
        manifest_rel = str(MANIFEST_JSON.relative_to(PROJECT_ROOT))
        assert gold_rel in artifacts, f"lock missing gold.jsonl entry ({gold_rel})"
        assert manifest_rel in artifacts, f"lock missing manifest.json entry ({manifest_rel})"

    def test_lock_gold_jsonl_sha256_matches_disk(self) -> None:
        artifacts = _load_lock()
        key = str(GOLD_JSONL.relative_to(PROJECT_ROOT))
        expected = artifacts[key]
        actual = _sha256_str(GOLD_JSONL.read_text(encoding="utf-8"))
        assert actual == expected, (
            f"lock gold.jsonl sha256 mismatch:\n"
            f"  lock: {expected}\n  disk: {actual}"
        )

    def test_lock_manifest_sha256_matches_disk(self) -> None:
        artifacts = _load_lock()
        key = str(MANIFEST_JSON.relative_to(PROJECT_ROOT))
        expected = artifacts[key]
        actual = _sha256_file(MANIFEST_JSON)
        assert actual == expected, (
            f"lock manifest.json sha256 mismatch:\n"
            f"  lock: {expected}\n  disk: {actual}"
        )

    def test_all_fixture_clips_present_in_lock(self) -> None:
        artifacts = _load_lock()
        for wav in sorted(FIXTURES_DIR.glob("*.wav")):
            rel = str(wav.relative_to(PROJECT_ROOT))
            assert rel in artifacts, (
                f"fixture {rel} not found in benchmark.lock"
            )


# ---------------------------------------------------------------------------
# Fixture clip tests
# ---------------------------------------------------------------------------

class TestFixtureClips:
    """Tests for the 3 sample clips committed to tests/fixtures/benchmark/."""

    def test_fixture_dir_exists(self) -> None:
        assert FIXTURES_DIR.exists(), f"fixtures dir missing: {FIXTURES_DIR}"

    def test_exactly_3_fixture_clips(self) -> None:
        clips = sorted(FIXTURES_DIR.glob("*.wav"))
        assert len(clips) == 3, (
            f"expected 3 fixture clips, found {len(clips)}: {clips}"
        )

    def test_fixture_clips_are_clean_sub_and_silence_hesitation(self) -> None:
        """The 3 representative clips must be clean, substitution, and silence-hes."""
        rows_by_id = {r["utt_id"]: r for r in _load_gold_rows()}
        clips = sorted(FIXTURES_DIR.glob("*.wav"))
        utt_ids = [c.stem for c in clips]

        # At least one clean (no gold entries)
        clean_ids = [uid for uid in utt_ids if uid in rows_by_id and rows_by_id[uid]["gold"] == []]
        assert clean_ids, f"no clean fixture found in {utt_ids}"

        # At least one substitution
        sub_ids = [
            uid for uid in utt_ids
            if uid in rows_by_id
            and any(g["type"] == "substitution" for g in rows_by_id[uid]["gold"])
        ]
        assert sub_ids, f"no substitution fixture found in {utt_ids}"

        # At least one silence hesitation
        silence_ids = [
            uid for uid in utt_ids
            if uid in rows_by_id
            and any(
                g["type"] == "hesitation" and g["render"] == "silence"
                for g in rows_by_id[uid]["gold"]
            )
        ]
        assert silence_ids, f"no silence-hesitation fixture found in {utt_ids}"

    def test_fixture_clips_sha256_match_lock(self) -> None:
        """Each committed fixture clip's sha256 must match its lock entry."""
        artifacts = _load_lock()
        for wav in sorted(FIXTURES_DIR.glob("*.wav")):
            rel = str(wav.relative_to(PROJECT_ROOT))
            assert rel in artifacts, f"fixture {rel} not in lock"
            expected = artifacts[rel]
            actual = _sha256_file(wav)
            assert actual == expected, (
                f"fixture {rel} sha256 mismatch:\n"
                f"  lock: {expected}\n  disk: {actual}"
            )

    def test_fixture_clips_sha256_match_gold_jsonl(self) -> None:
        """The sha256 of each fixture clip must also match the gold.jsonl entry."""
        rows_by_id = {r["utt_id"]: r for r in _load_gold_rows()}
        for wav in sorted(FIXTURES_DIR.glob("*.wav")):
            utt_id = wav.stem
            assert utt_id in rows_by_id, (
                f"fixture {utt_id}.wav has no entry in gold.jsonl"
            )
            expected = rows_by_id[utt_id]["wav_sha256"]
            actual = _sha256_file(wav)
            assert actual == expected, (
                f"fixture {utt_id}.wav sha256 mismatch vs gold.jsonl:\n"
                f"  gold.jsonl: {expected}\n  disk      : {actual}"
            )
