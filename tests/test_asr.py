"""Tests for ASR backend (T1.1).

CI-safe tests run by default (no model, no network).
Network tests require the real faster-whisper model and are marked accordingly.
"""
from __future__ import annotations

import contextlib
import json
import sys
import hashlib
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIXTURE_WAV = Path(__file__).parent / "fixtures" / "asr" / "cat_passage.wav"
CACHE_DIR = Path(__file__).parent.parent / "evals" / "golden" / "asr_cache"


# ---------------------------------------------------------------------------
# Helper: evict a cache entry for the duration of a test
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _evicted_cache_entry(cache_file: Path, tmp_path: Path):
    """Context manager that removes *cache_file* before the body and restores it after.

    Backup/restore semantics (leave-no-trace):
    - If a committed cache entry exists it is moved aside, restored in finally.
    - If no entry existed before but the body created one, it is unlinked in finally.
    - Either way the cache is left byte-identical to its pre-test state.

    This ensures each call to transcribe() inside the body exercises the real
    model path rather than silently returning a committed-cache hit.
    """
    backup = tmp_path / (cache_file.name + ".bak")
    had_cache = cache_file.exists()

    if had_cache:
        cache_file.rename(backup)

    try:
        yield
    finally:
        if had_cache:
            # Restore the committed entry (overwrite whatever transcribe wrote).
            backup.rename(cache_file)
        else:
            # No prior entry — unlink any entry the body may have created.
            if cache_file.exists():
                cache_file.unlink()


# ===========================================================================
# CI-safe tests (no model import, no audio hardware, no network)
# ===========================================================================

class TestCacheKeyDeterminism:
    """Cache key must depend on content, not file path."""

    def test_same_bytes_same_params_same_key(self, tmp_path):
        from readcoach.asr import _cache_key

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"FAKE_AUDIO_DATA_12345")

        k1 = _cache_key(str(audio), target_text="the cat", bias="none", backend="faster-whisper-small")
        k2 = _cache_key(str(audio), target_text="the cat", bias="none", backend="faster-whisper-small")
        assert k1 == k2

    def test_different_bias_different_key(self, tmp_path):
        from readcoach.asr import _cache_key

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"FAKE_AUDIO_DATA_12345")

        k_none = _cache_key(str(audio), target_text="the cat", bias="none", backend="faster-whisper-small")
        k_prompt = _cache_key(str(audio), target_text="the cat", bias="prompt", backend="faster-whisper-small")
        assert k_none != k_prompt

    def test_different_audio_bytes_different_key(self, tmp_path):
        from readcoach.asr import _cache_key

        a1 = tmp_path / "a1.wav"
        a2 = tmp_path / "a2.wav"
        a1.write_bytes(b"AUDIO_ONE")
        a2.write_bytes(b"AUDIO_TWO")

        k1 = _cache_key(str(a1), target_text=None, bias="none", backend="faster-whisper-small")
        k2 = _cache_key(str(a2), target_text=None, bias="none", backend="faster-whisper-small")
        assert k1 != k2

    def test_same_bytes_different_path_same_key(self, tmp_path):
        """Key is content-addressed, not path-addressed."""
        from readcoach.asr import _cache_key

        a1 = tmp_path / "dir1" / "a.wav"
        a2 = tmp_path / "dir2" / "a.wav"
        a1.parent.mkdir()
        a2.parent.mkdir()
        a1.write_bytes(b"IDENTICAL_BYTES")
        a2.write_bytes(b"IDENTICAL_BYTES")

        k1 = _cache_key(str(a1), target_text=None, bias="none", backend="faster-whisper-small")
        k2 = _cache_key(str(a2), target_text=None, bias="none", backend="faster-whisper-small")
        assert k1 == k2


class TestSerializationRoundTrip:
    """AsrResult must survive JSON serialization with equality."""

    def test_roundtrip(self):
        from readcoach.asr import AsrResult, Word, _serialize_result, _deserialize_result

        original = AsrResult(
            text="the cat sat",
            words=[
                Word(text="the", start=0.0, end=0.3, confidence=0.99),
                Word(text="cat", start=0.4, end=0.7, confidence=0.95),
                Word(text="sat", start=0.8, end=1.1, confidence=0.88),
            ],
            rtf=0.12,
            backend="faster-whisper-small",
        )
        data = _serialize_result(original)
        restored = _deserialize_result(data)

        assert restored.text == original.text
        assert restored.backend == original.backend
        assert restored.rtf == original.rtf
        assert len(restored.words) == len(original.words)
        for orig_w, rest_w in zip(original.words, restored.words):
            assert orig_w.text == rest_w.text
            assert orig_w.start == rest_w.start
            assert orig_w.end == rest_w.end
            assert orig_w.confidence == rest_w.confidence

    def test_missing_key_raises(self):
        from readcoach.asr import _deserialize_result

        with pytest.raises(KeyError):
            _deserialize_result({"text": "hi"})  # missing 'words'


class TestCacheMissBehavior:
    """cache_only=True must raise CacheMiss without importing faster_whisper."""

    def test_cache_miss_raises(self, tmp_path):
        from readcoach.asr import transcribe, CacheMiss

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"FAKE_WAV")

        with pytest.raises(CacheMiss):
            transcribe(str(audio), cache_only=True)

    def test_cache_miss_no_faster_whisper_import(self, tmp_path):
        """Proves CI is model-free: faster_whisper must NOT be imported."""
        # Remove faster_whisper from sys.modules if somehow already loaded
        # so we get a clean read on whether cache_only path imports it.
        fw_key = "faster_whisper"
        was_present = fw_key in sys.modules
        saved = sys.modules.pop(fw_key, None)

        try:
            from readcoach.asr import transcribe, CacheMiss

            audio = tmp_path / "b.wav"
            audio.write_bytes(b"MORE_FAKE_WAV")

            with pytest.raises(CacheMiss):
                transcribe(str(audio), cache_only=True)

            assert fw_key not in sys.modules, (
                "faster_whisper was imported during a cache_only=True miss — "
                "the lazy-import contract is broken"
            )
        finally:
            if was_present:
                sys.modules[fw_key] = saved


class TestBiasValidation:
    """bias != 'none' with target_text=None must raise ValueError."""

    def test_prompt_without_target_text_raises(self, tmp_path):
        from readcoach.asr import transcribe

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"FAKE")

        with pytest.raises(ValueError, match="target_text"):
            transcribe(str(audio), target_text=None, bias="prompt", cache_only=True)

    def test_strong_without_target_text_raises(self, tmp_path):
        from readcoach.asr import transcribe

        audio = tmp_path / "a.wav"
        audio.write_bytes(b"FAKE")

        with pytest.raises(ValueError, match="target_text"):
            transcribe(str(audio), target_text=None, bias="strong", cache_only=True)


# ===========================================================================
# Network tests (require model download ~500MB on first run)
# ===========================================================================

@pytest.mark.network
class TestRealTranscription:
    """Tests that hit the real faster-whisper model."""

    def test_transcribe_fixture_bias_none(self, tmp_path):
        """Real transcription of cat_passage.wav at bias=none.

        Temporarily moves the cache entry aside so we exercise the real
        transcription path (and can assert rtf > 0), then puts it back.
        """
        assert FIXTURE_WAV.exists(), f"Fixture missing: {FIXTURE_WAV}"

        import readcoach.asr as asr_mod

        key = asr_mod._cache_key(
            str(FIXTURE_WAV), target_text=None, bias="none", backend="faster-whisper-small"
        )
        cache_file = CACHE_DIR / f"{key}.json"

        with _evicted_cache_entry(cache_file, tmp_path):
            result = asr_mod.transcribe(str(FIXTURE_WAV), bias="none")

            # words populated
            assert len(result.words) > 0, "Expected non-empty words list"

            # every word has required fields
            for w in result.words:
                assert isinstance(w.start, float), f"word.start not float: {w}"
                assert isinstance(w.end, float), f"word.end not float: {w}"
                assert isinstance(w.confidence, float), f"word.confidence not float: {w}"
                assert w.start <= w.end, f"start > end for word {w}"

            # transcript sanity
            assert "cat" in result.text.lower(), (
                f"Expected 'cat' in transcript, got: {result.text!r}"
            )

            # rtf populated and positive (real transcription path)
            assert result.rtf is not None and result.rtf > 0, (
                f"Expected rtf > 0, got: {result.rtf}"
            )
            print(f"\n  bias=none  rtf={result.rtf:.4f}")

            # cache file was created after real transcription
            assert cache_file.exists(), f"Cache file not created: {cache_file}"

            # second call returns cached result (text equality)
            result2 = asr_mod.transcribe(str(FIXTURE_WAV), bias="none")
            assert result2.text == result.text
            assert len(result2.words) == len(result.words)

    def test_transcribe_fixture_bias_strong(self, tmp_path):
        """bias=strong exercises the hotwords path + RuntimeError guard."""
        assert FIXTURE_WAV.exists(), f"Fixture missing: {FIXTURE_WAV}"

        import readcoach.asr as asr_mod

        key = asr_mod._cache_key(
            str(FIXTURE_WAV),
            target_text="the cat sat on the mat",
            bias="strong",
            backend="faster-whisper-small",
        )
        cache_file = CACHE_DIR / f"{key}.json"

        with _evicted_cache_entry(cache_file, tmp_path):
            result = asr_mod.transcribe(
                str(FIXTURE_WAV),
                target_text="the cat sat on the mat",
                bias="strong",
            )
            assert len(result.words) > 0, "Expected non-empty words list"
            assert result.rtf is not None and result.rtf > 0, (
                f"Expected rtf > 0 (real transcription), got: {result.rtf}"
            )
            print(f"\n  bias=strong rtf={result.rtf:.4f}")

    def test_transcribe_fixture_bias_prompt(self, tmp_path):
        """bias=prompt uses initial_prompt."""
        assert FIXTURE_WAV.exists(), f"Fixture missing: {FIXTURE_WAV}"

        import readcoach.asr as asr_mod

        key = asr_mod._cache_key(
            str(FIXTURE_WAV),
            target_text="the cat sat on the mat",
            bias="prompt",
            backend="faster-whisper-small",
        )
        cache_file = CACHE_DIR / f"{key}.json"

        with _evicted_cache_entry(cache_file, tmp_path):
            result = asr_mod.transcribe(
                str(FIXTURE_WAV),
                target_text="the cat sat on the mat",
                bias="prompt",
            )
            assert len(result.words) > 0
            assert result.rtf is not None and result.rtf > 0, (
                f"Expected rtf > 0 (real transcription), got: {result.rtf}"
            )
            print(f"\n  bias=prompt rtf={result.rtf:.4f}")


# ===========================================================================
# Cache-hermetic test: runs against committed cache entries
# This test must PASS in CI using cache entries committed in this ticket.
# It does NOT skip — it asserts real data.
# ===========================================================================

class TestCommittedCacheHermetic:
    """Proves that CI can run transcription via committed cache entries."""

    def test_cache_only_returns_committed_result(self):
        """
        Uses cache_only=True against the cache entry committed in this ticket.
        If the cache entry is missing (first clone, before network run), this
        test fails loudly — that is intentional. The cache entry for
        cat_passage.wav at bias='none' must be committed.
        """
        assert FIXTURE_WAV.exists(), (
            f"Fixture wav missing: {FIXTURE_WAV}\n"
            "Run: say -v Samantha 'the cat sat on the mat' -o /tmp/x.aiff && "
            "ffmpeg -i /tmp/x.aiff -ar 16000 -ac 1 <dest>"
        )

        import readcoach.asr as asr_mod

        key = asr_mod._cache_key(
            str(FIXTURE_WAV),
            target_text=None,
            bias="none",
            backend="faster-whisper-small",
        )
        cache_file = CACHE_DIR / f"{key}.json"
        assert cache_file.exists(), (
            f"Committed cache entry missing: {cache_file}\n"
            "Run the network tests locally first, then commit the resulting "
            "asr_cache/*.json files."
        )

        result = asr_mod.transcribe(str(FIXTURE_WAV), bias="none", cache_only=True)
        assert len(result.words) > 0
        assert "cat" in result.text.lower()

        # verify words match the committed entry exactly
        raw = json.loads(cache_file.read_text())
        committed_words = [w["text"] for w in raw["words"]]
        result_words = [w.text for w in result.words]
        assert result_words == committed_words, (
            "Deserialized words differ from committed cache — "
            "serialization round-trip is broken"
        )
