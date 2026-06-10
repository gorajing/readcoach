#!/usr/bin/env python3
"""Worked BYO-ASR adapter: facebook/wav2vec2-base-960h -> hypotheses.jsonl.

This is the "second public ASR" receipt for the ReadCoach BYO-ASR contract
(docs/BENCHMARK.md). It proves the hypotheses schema is NOT shaped around
faster-whisper: a completely different model (a CTC wav2vec2, not Whisper) feeds
the same scorer with no changes to the contract.

What it does
------------
1. Loads facebook/wav2vec2-base-960h (CTC, ~360 MB; CPU is fine).
2. Transcribes each of the 88 fetched clips in data/benchmark/clips/.
3. Extracts PER-WORD TIMINGS from CTC output (output_word_offsets=True), so the
   silence-hesitation rule (gap > 1.0 s) has a chance to fire — see §3.2 of
   docs/BENCHMARK.md.
4. Emits hypotheses_wav2vec2.jsonl in the schema from §3 of docs/BENCHMARK.md,
   DELIBERATELY OMITTING the `confidence` field — demonstrating that field's
   optionality (it defaults to 1.0 and does not affect today's scoring).

This file imports transformers + torch + soundfile, which are NOT project
dependencies. Run it standalone (deps injected on the command line):

    uv run --with transformers --with torch --with soundfile \
        python examples/wav2vec2_adapter/run_wav2vec2.py

Then score it (this command imports ONLY jiwer + stdlib — the auditable path):

    uv run readcoach-bench score \
        --hypotheses wav2vec2=examples/wav2vec2_adapter/hypotheses_wav2vec2.jsonl

The committed hypotheses_wav2vec2.jsonl is generated OUTPUT (the worked example's
receipt). Regenerate it with this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root: examples/wav2vec2_adapter/run_wav2vec2.py -> parents[2].
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CLIPS_DIR = _PROJECT_ROOT / "data" / "benchmark" / "clips"
_GOLD = _PROJECT_ROOT / "data" / "benchmark" / "gold.jsonl"
_DEFAULT_OUT = Path(__file__).resolve().parent / "hypotheses_wav2vec2.jsonl"

_MODEL_ID = "facebook/wav2vec2-base-960h"
_TARGET_SR = 16_000  # wav2vec2-base-960h is trained at 16 kHz.


def _load_utt_ids() -> list[str]:
    """Return the gold utt_ids in file order (== the clip stems to transcribe)."""
    utt_ids: list[str] = []
    with _GOLD.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            utt_ids.append(json.loads(line)["utt_id"])
    return utt_ids


def _read_audio_16k(path: Path):
    """Read a WAV as mono float32 at 16 kHz.

    The benchmark clips are already 16 kHz mono (verified); we assert that rather
    than silently resampling, so a surprising input fails loud.
    """
    import soundfile as sf  # noqa: PLC0415

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        # Downmix to mono if a clip is unexpectedly multi-channel.
        audio = audio.mean(axis=1)
    if sr != _TARGET_SR:
        raise ValueError(
            f"{path}: expected {_TARGET_SR} Hz, got {sr} Hz. "
            "The ReadCoach clips are 16 kHz; resample upstream if this changes."
        )
    return audio


def _transcribe_one(model, processor, audio):
    """Run CTC inference and return a list of {text, start, end} word dicts.

    Word offsets come from output_word_offsets=True: the processor maps CTC frame
    indices back to time via the model's total stride. Confidence is intentionally
    NOT computed or emitted (proving the field is optional).
    """
    import torch  # noqa: PLC0415

    input_values = processor(
        audio, sampling_rate=_TARGET_SR, return_tensors="pt"
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    # Seconds per output frame = (input samples / output frames) / sample rate.
    n_frames = logits.shape[1]
    time_per_frame = (input_values.shape[1] / n_frames) / _TARGET_SR

    decoded = processor.batch_decode(
        predicted_ids, output_word_offsets=True
    )
    word_offsets = decoded["word_offsets"][0]

    words: list[dict] = []
    for wo in word_offsets:
        # NOTE: no "confidence" key — deliberately omitted (schema §3.3).
        words.append(
            {
                "text": wo["word"],
                "start": round(wo["start_offset"] * time_per_frame, 3),
                "end": round(wo["end_offset"] * time_per_frame, 3),
            }
        )
    return words


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output hypotheses JSONL (default: {_DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Transcribe only the first N clips (smoke test).",
    )
    args = parser.parse_args(argv)

    if not _CLIPS_DIR.exists():
        print(
            f"ERROR: clips not found at {_CLIPS_DIR}.\n"
            "Run `python3 scripts/fetch_benchmark.py` first.",
            file=sys.stderr,
        )
        return 2

    # Lazy heavy imports — only when we actually transcribe.
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # noqa: PLC0415

    print(f"Loading {_MODEL_ID} (CPU) …", file=sys.stderr)
    processor = Wav2Vec2Processor.from_pretrained(_MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(_MODEL_ID)
    model.eval()

    utt_ids = _load_utt_ids()
    if args.limit is not None:
        utt_ids = utt_ids[: args.limit]

    rows: list[dict] = []
    for i, utt_id in enumerate(utt_ids, start=1):
        clip = _CLIPS_DIR / f"{utt_id}.wav"
        if not clip.exists():
            raise FileNotFoundError(f"Missing clip for {utt_id!r}: {clip}")
        audio = _read_audio_16k(clip)
        words = _transcribe_one(model, processor, audio)
        rows.append({"utt_id": utt_id, "words": words})
        print(f"  [{i:3d}/{len(utt_ids)}] {utt_id}  ({len(words)} words)", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} hypotheses to {args.out}", file=sys.stderr)
    print(
        "Score it with:\n"
        "  uv run readcoach-bench score "
        f"--hypotheses wav2vec2={args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
