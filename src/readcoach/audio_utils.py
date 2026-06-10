"""Audio utilities for benchmark TTS rendering (T1.3b).

Fail-loud subprocess wrappers for macOS ``say``, ffmpeg, and ffprobe.
Pattern lifted from music-analyzer/utils/ffmpeg.py: every non-zero
exit aborts with the command named; every structural invariant is
checked before returning.

Speech targets: 16 kHz mono PCM WAV (not 44.1k stereo — these clips
are ASR input, not music).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Low-level subprocess helper
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Run ``cmd``, raising loudly on any non-zero exit.

    Never swallows stderr — it is included in the exception message so the
    caller immediately knows which item and which tool failed.
    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    return result


# ---------------------------------------------------------------------------
# macOS say → AIFF
# ---------------------------------------------------------------------------

def run_say(text: str, voice: str, rate: int, out_aiff: Path) -> Path:
    """Render ``text`` to ``out_aiff`` via macOS ``say``.

    ``text`` may contain ``[[slnc N]]`` silence markers; ``say`` interprets
    them natively on macOS.  ``out_aiff`` is created (or overwritten).

    Args:
        text:     TTS text (may include [[slnc N]] markers).
        voice:    macOS voice name, e.g. ``"Samantha"`` or ``"Fred"``.
        rate:     Words-per-minute rate (e.g. 140–180).
        out_aiff: Destination path (must end in .aiff; created if absent).

    Returns:
        ``out_aiff`` (the AIFF file just written).

    Raises:
        RuntimeError: if ``say`` exits non-zero.
        FileNotFoundError: if no file appeared at ``out_aiff`` after success.
    """
    out_aiff = Path(out_aiff)
    out_aiff.parent.mkdir(parents=True, exist_ok=True)

    _run(["say", "-v", voice, "-r", str(rate), "-o", str(out_aiff), text])

    if not out_aiff.exists():
        raise FileNotFoundError(
            f"say produced no output at {out_aiff} "
            f"(voice={voice!r}, rate={rate})"
        )
    return out_aiff


# ---------------------------------------------------------------------------
# ffmpeg: AIFF → 16 kHz mono PCM WAV
# ---------------------------------------------------------------------------

def to_16k_mono(src: Path, dst: Path) -> Path:
    """Convert ``src`` (any format ffmpeg reads) to 16 kHz mono PCM WAV at ``dst``.

    Uses ``-acodec pcm_s16le`` for interoperability with every ASR tool.
    Overwrites ``dst`` if it already exists.

    Args:
        src: Source audio (typically an AIFF from ``run_say``).
        dst: Destination WAV path (parent created if absent).

    Returns:
        ``dst`` (the WAV file just written).

    Raises:
        RuntimeError: if ffmpeg exits non-zero.
        FileNotFoundError: if no file appeared at ``dst`` after success.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    _run([
        "ffmpeg",
        "-y",                    # Overwrite output without asking.
        "-i", str(src),          # Input file first (keyframe-accurate for AIFF anyway).
        "-ar", "16000",          # 16 kHz — ASR target, NOT 44.1k music default.
        "-ac", "1",              # Mono — speech, not stereo.
        "-acodec", "pcm_s16le",  # Uncompressed PCM, universally compatible.
        str(dst),
    ])

    if not dst.exists():
        raise FileNotFoundError(
            f"ffmpeg produced no output at {dst} (src={src})"
        )
    return dst


# ---------------------------------------------------------------------------
# ffprobe: structural metadata
# ---------------------------------------------------------------------------

def probe(path: Path) -> dict[str, object]:
    """Return audio stream metadata for ``path`` via ffprobe.

    Returns a dict with at minimum:
      ``sample_rate`` (int), ``channels`` (int), ``duration_s`` (float).

    Raises:
        RuntimeError: if ffprobe exits non-zero.
        ValueError: if the file has no audio stream or the JSON is malformed.
    """
    path = Path(path)
    result = _run([
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(path),
    ])

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe returned non-JSON output for {path}: {exc}") from exc

    streams = data.get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise ValueError(f"ffprobe found no audio stream in {path}")

    s = audio_streams[0]
    try:
        return {
            "sample_rate": int(s["sample_rate"]),
            "channels": int(s["channels"]),
            # WAV PCM streams may not have "duration"; fall back to tags or 0.
            "duration_s": float(
                s.get("duration")
                or s.get("tags", {}).get("DURATION", 0)
                or 0
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"ffprobe stream metadata incomplete for {path}: {exc}\n"
            f"  stream dict: {s}"
        ) from exc


# ---------------------------------------------------------------------------
# Validation (raises on every violation)
# ---------------------------------------------------------------------------

# Minimum reasonable clip duration (seconds).  A text fragment at the lowest
# planned rate (140 wpm) over the shortest passage excerpt should still
# produce > 0.3 s of audio; 0.1 s is a conservative floor.
_MIN_DURATION_S = 0.1

# Maximum silence fraction by a very cheap RMS check.  We don't import numpy
# here — if we did this check would be more principled — but a zero-byte WAV
# body (or a near-silent file) is almost certainly a renderer bug worth catching.
# We keep it cheap and optional: any structurally valid clip passes the silence
# gate; only clips whose ffprobe duration rounds to 0 fail.
_SILENCE_FLOOR_S = 0.01


def validate_clip(path: Path) -> dict[str, object]:
    """Probe ``path`` and raise on any structural violation.

    Checks:
      1. Sample rate is exactly 16 000 Hz.
      2. Channels is exactly 1 (mono).
      3. Duration is > ``_MIN_DURATION_S``.

    Returns the probe dict on success (so callers can record duration).

    Raises:
        RuntimeError: on any violation, naming the file and the failing check.
    """
    path = Path(path)
    info = probe(path)

    errors: list[str] = []

    if info["sample_rate"] != 16000:
        errors.append(
            f"sample_rate={info['sample_rate']} (expected 16000)"
        )
    if info["channels"] != 1:
        errors.append(
            f"channels={info['channels']} (expected 1/mono)"
        )
    if info["duration_s"] <= _MIN_DURATION_S:
        errors.append(
            f"duration_s={info['duration_s']:.4f} (expected > {_MIN_DURATION_S})"
        )

    if errors:
        raise RuntimeError(
            f"validate_clip FAILED for {path}:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    return info
