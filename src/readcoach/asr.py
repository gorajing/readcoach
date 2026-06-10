"""ASR layer — swappable on purpose. The interesting measurement lives above it.

Backends sit behind one interface so accuracy vs. latency can be compared:
  - faster-whisper (default, "faster-whisper-small")
  - whisper.cpp / Moonshine v2  (on-device stretch)

===========================================================================
Bias-strength knob semantics
===========================================================================
The ``bias`` parameter controls how strongly the model is nudged toward
``target_text`` (which must be provided whenever bias != "none"):

  "none"   — no target-text influence at all; purely acoustic decoding.
  "prompt" — passes ``target_text`` as ``initial_prompt``; steers the
             language-model prior but does NOT force word choices.
  "strong" — ``initial_prompt=target_text`` AND ``hotwords=target_text``;
             maximum pull toward the expected passage.  This is where the
             bias-vs-accuracy tradeoff (arXiv:2505.23627, 2506.11079) is
             sharpest.

``bias != "none"`` with ``target_text=None`` raises ``ValueError``.

===========================================================================
Cache
===========================================================================
Results are stored under ``evals/golden/asr_cache/<hexdigest>.json``.
The key is sha256 over:
  * the audio file BYTES  (content-addressed, not path-addressed)
  * a canonical JSON of   {target_text, bias, backend}

This means the same audio processed with the same params anywhere on any
machine hits the same cache entry.  The cache is committed to git so CI
runs without touching a model.

``cache_only=True`` raises ``CacheMiss`` instead of loading any model —
this is what makes CI provably model-free (checked by test).

``rtf`` (real-time factor = processing_time / audio_duration) is populated
on real transcription; it is ``None`` on cache hits (the latency proxy is
only meaningful for the transcription pass, not the cache lookup).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

BiasStrength = Literal["none", "prompt", "strong"]

_CACHE_DIR = Path(__file__).parents[2] / "evals" / "golden" / "asr_cache"


@dataclass
class Word:
    text: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None


@dataclass
class AsrResult:
    text: str
    words: list[Word] = field(default_factory=list)
    # rtf = processing_time / audio_duration; None on cache hits (latency proxy
    # is only meaningful during the transcription pass, not the cache lookup).
    rtf: float | None = None
    backend: str = "faster-whisper-small"


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class CacheMiss(Exception):
    """Raised when cache_only=True and no cache entry exists for the key."""


# ---------------------------------------------------------------------------
# Module-level model cache (memoized per backend string)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Cache key computation (content-addressed)
# ---------------------------------------------------------------------------

def _cache_key(
    audio_path: str,
    target_text: str | None,
    bias: str,
    backend: str,
) -> str:
    """Return a hex digest that uniquely identifies audio content + params.

    Keyed on file *bytes*, not path — same audio anywhere hits the same entry.
    """
    audio_bytes = Path(audio_path).read_bytes()
    params_json = json.dumps(
        {"target_text": target_text, "bias": bias, "backend": backend},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(audio_bytes + params_json).hexdigest()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_result(result: AsrResult) -> dict:
    """Convert AsrResult to a JSON-serializable dict."""
    return {
        "text": result.text,
        "words": [
            {
                "text": w.text,
                "start": w.start,
                "end": w.end,
                "confidence": w.confidence,
            }
            for w in result.words
        ],
        "rtf": result.rtf,
        "backend": result.backend,
    }


def _deserialize_result(data: dict) -> AsrResult:
    """Reconstruct AsrResult from a dict.  Raises KeyError on missing fields."""
    # Explicit key access (not .get) — fail-loud on corrupt/incomplete entries.
    words = [
        Word(
            text=w["text"],
            start=w["start"],
            end=w["end"],
            confidence=w["confidence"],
        )
        for w in data["words"]  # KeyError if "words" missing
    ]
    return AsrResult(
        text=data["text"],
        words=words,
        rtf=data["rtf"],
        backend=data["backend"],
    )


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.json"


def _read_cache(key: str) -> AsrResult | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _deserialize_result(raw)


def _write_cache(
    key: str,
    result: AsrResult,
    params: dict,
) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entry = _serialize_result(result)
    # Store rtf=None for cache entries (rtf is latency of the transcription
    # pass; it should not be set on cache hits when this entry is replayed).
    entry["rtf"] = None
    entry["params"] = params  # echo block for human debuggability
    path = _cache_path(key)
    path.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")


def build_params_index(cache_dir: Path | None = None) -> dict[str, "AsrResult"]:
    """Build a params-keyed index of all committed cache entries.

    Scans every ``*.json`` in ``cache_dir`` (default: ``_CACHE_DIR``) and builds
    a dict keyed by the canonical params JSON string.  Cache entries without a
    ``params`` field are silently skipped.

    This index lets ``--fixtures`` mode look up ASR results by
    ``(target_text, bias, backend)`` without needing the audio files — which are
    excluded from git (``*.wav`` is in ``.gitignore``).

    Returns
    -------
    dict[params_json_str, AsrResult]
        Key is the canonical JSON string (sorted keys, no spaces) of the params
        dict: ``{"target_text": ..., "bias": ..., "backend": ...}``.
    """
    d = cache_dir if cache_dir is not None else _CACHE_DIR
    index: dict[str, AsrResult] = {}
    for p in d.glob("*.json"):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # corrupt entry — skip
        if "params" not in raw:
            continue
        params = raw["params"]
        key = json.dumps(
            {
                "target_text": params.get("target_text"),
                "bias": params.get("bias"),
                "backend": params.get("backend"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            index[key] = _deserialize_result(raw)
        except (KeyError, TypeError):
            continue  # incomplete entry — skip
    return index


def transcribe_from_index(
    params_index: dict[str, "AsrResult"],
    target_text: str | None,
    bias: str,
    backend: str,
) -> "AsrResult":
    """Look up an ASR result from a pre-built params index.

    Raises ``CacheMiss`` if the params combination is not in the index.
    Use this in ``--fixtures`` mode to avoid needing audio files on disk.

    Parameters
    ----------
    params_index:
        Built by :func:`build_params_index`.
    target_text, bias, backend:
        Same semantics as :func:`transcribe`.
    """
    key = json.dumps(
        {"target_text": target_text, "bias": bias, "backend": backend},
        sort_keys=True,
        separators=(",", ":"),
    )
    result = params_index.get(key)
    if result is None:
        raise CacheMiss(
            f"No cache entry for params {key!r}\n"
            "Run without --fixtures to transcribe and populate the cache."
        )
    return result


# ---------------------------------------------------------------------------
# Backend loading (lazy import — only on real-transcription path)
# ---------------------------------------------------------------------------

def _load_model(backend: str):
    """Return (possibly cached) WhisperModel for ``backend``.

    ``backend`` format: "faster-whisper-<size>"  →  WhisperModel(size, ...)
    Lazy import: faster_whisper is NOT imported until this function runs.
    """
    if backend in _MODEL_CACHE:
        return _MODEL_CACHE[backend]

    from faster_whisper import WhisperModel  # noqa: PLC0415 — intentional lazy import

    prefix = "faster-whisper-"
    if not backend.startswith(prefix):
        raise ValueError(
            f"Unknown backend {backend!r}.  Expected 'faster-whisper-<size>'."
        )
    size = backend[len(prefix):]
    model = WhisperModel(size, device="cpu", compute_type="int8")
    _MODEL_CACHE[backend] = model
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(
    audio_path: str,
    target_text: str | None = None,
    bias: BiasStrength = "none",
    backend: str = "faster-whisper-small",
    cache_only: bool = False,
) -> AsrResult:
    """Transcribe *audio_path* and return an :class:`AsrResult` with word timings.

    Parameters
    ----------
    audio_path:
        Path to the audio file (wav / mp3 / etc.).
    target_text:
        The expected passage text.  Required when ``bias != "none"``.
    bias:
        Bias-strength knob — see module docstring for semantics.
    backend:
        Backend identifier.  ``"faster-whisper-<size>"`` maps to
        ``WhisperModel(size, device="cpu", compute_type="int8")``.
    cache_only:
        If ``True``, return the cached result or raise :class:`CacheMiss`.
        The faster_whisper library is **never imported** on this path —
        that is the contract that keeps CI provably model-free.
    """
    # Bias validation — before any I/O.
    if bias != "none" and target_text is None:
        raise ValueError(
            "target_text must be provided when bias != 'none' "
            f"(got bias={bias!r}, target_text=None)"
        )

    key = _cache_key(audio_path, target_text=target_text, bias=bias, backend=backend)
    params = {"target_text": target_text, "bias": bias, "backend": backend}

    # --- Cache lookup (no model import happens here) ---
    cached = _read_cache(key)
    if cached is not None:
        return cached

    # --- Cache miss ---
    if cache_only:
        raise CacheMiss(
            f"No cache entry for key={key!r}  params={params!r}\n"
            "Run without cache_only=True to transcribe and populate the cache."
        )

    # --- Real transcription (lazy model import on this path only) ---
    model = _load_model(backend)

    transcribe_kwargs: dict = {
        "word_timestamps": True,
    }
    if bias == "prompt":
        transcribe_kwargs["initial_prompt"] = target_text
    elif bias == "strong":
        transcribe_kwargs["initial_prompt"] = target_text
        # Guard: verify the installed faster-whisper version supports hotwords.
        # RuntimeError (not assert) so the check is never stripped under python -O.
        if "hotwords" not in inspect.signature(model.transcribe).parameters:
            raise RuntimeError(
                "faster-whisper model.transcribe() does not have a 'hotwords' parameter; "
                "upgrade to faster-whisper>=1.1 or check the installed version."
            )
        transcribe_kwargs["hotwords"] = target_text

    t0 = time.monotonic()
    segments_gen, info = model.transcribe(audio_path, **transcribe_kwargs)
    segments = list(segments_gen)  # consume generator; triggers actual computation
    elapsed = time.monotonic() - t0

    # Build word list from segment word timestamps.
    words: list[Word] = []
    full_text_parts: list[str] = []
    for seg in segments:
        full_text_parts.append(seg.text)
        if seg.words:
            for w in seg.words:
                words.append(
                    Word(
                        text=w.word,
                        start=float(w.start),
                        end=float(w.end),
                        confidence=float(w.probability),
                    )
                )

    full_text = "".join(full_text_parts).strip()
    audio_duration = info.duration if info.duration and info.duration > 0 else elapsed
    rtf = elapsed / audio_duration

    result = AsrResult(
        text=full_text,
        words=words,
        rtf=rtf,
        backend=backend,
    )

    _write_cache(key, result, params)
    return result
