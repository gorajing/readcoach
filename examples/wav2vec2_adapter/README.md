# wav2vec2 adapter — a second public ASR for the ReadCoach BYO-ASR contract

This is the **worked second-ASR example** for the bring-your-own-ASR scoring path
(see `docs/BENCHMARK.md`). It exists to prove the hypotheses contract is **not
shaped around faster-whisper**: a structurally different model — a CTC
`facebook/wav2vec2-base-960h`, not Whisper — feeds the same scorer with no
changes to the schema or the metrics.

## What it does

`run_wav2vec2.py`:

1. Loads `facebook/wav2vec2-base-960h` (CTC, ~360 MB download; CPU is fine).
2. Transcribes the 88 fetched clips in `data/benchmark/clips/`.
3. Extracts **per-word timings** from the CTC output
   (`output_word_offsets=True`), so the silence-hesitation rule (gap > 1.0 s)
   has a chance to fire.
4. Emits `hypotheses_wav2vec2.jsonl` in the schema from §3 of `docs/BENCHMARK.md`,
   **deliberately omitting the `confidence` field** — demonstrating that field is
   optional (it defaults to `1.0` and does not affect today's scoring).

## Dependencies — NOT project dependencies

This adapter needs `transformers`, `torch`, and `soundfile`. They are **not**
added to the ReadCoach project dependencies (the whole point of the
auditable scoring path is that scoring needs none of this). Inject them on the
command line with `uv`:

```bash
uv run --with transformers --with torch --with soundfile \
    python examples/wav2vec2_adapter/run_wav2vec2.py
```

(Smoke test first if you like: add `--limit 2`.)

Prerequisite: fetch the clips once (stdlib only, no credentials):

```bash
python3 scripts/fetch_benchmark.py
```

## Then score it (the auditable path — jiwer + stdlib only)

```bash
uv run readcoach-bench score \
    --hypotheses wav2vec2=examples/wav2vec2_adapter/hypotheses_wav2vec2.jsonl
```

This scoring command imports **only jiwer + the standard library** — no model, no
service client. (Verify with the one-liner in §0 of `docs/BENCHMARK.md`.)

## The committed `hypotheses_wav2vec2.jsonl`

`hypotheses_wav2vec2.jsonl` is committed **generated output** — the worked
example's receipt (~100 KB of text, 88 lines). It is checked in so a reviewer can
see the exact hypotheses that produced the wav2vec2 column without downloading a
360 MB model. **Regenerate it with `run_wav2vec2.py`** (the numbers will match
within run-to-run determinism of CPU inference).

Each line omits `confidence` on purpose:

```json
{"utt_id": "p01-clean", "words": [{"text": "THE", "start": 0.02, "end": 0.12}, ...]}
```

## Reading the result

wav2vec2-base-960h is an off-the-shelf CTC model with **no target-text biasing**
and an **uppercase, punctuation-free, disfluency-normalized** transcript style.
Expect it to:

- do reasonably on the alignment classes (substitution / omission / insertion),
  which only need the word sequence; but
- score low on the **transcript-style-sensitive** classes (self_correction,
  hesitation), because the model normalizes away fillers and sound-out attempts —
  exactly the confound `docs/BENCHMARK.md` §4.6 calls out. The timings it emits do
  give the silence-hesitation rule a chance, unlike a no-timings adapter.

That contrast — a different ASR, scored unchanged, with the transcript-style
caveat doing real work — is the point of this example.
