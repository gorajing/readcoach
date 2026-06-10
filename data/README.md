# Data

Raw and generated audio is git-ignored; committed fixtures live under `tests/fixtures/`.

## The synthetic miscue benchmark (primary)

No public passage-aligned child miscue corpus exists — every relevant dataset is
licensed or private. So the primary benchmark here is **synthetic by construction**:

1. Hand-written decodable passages (`data/passages/*.yaml`).
2. Scripted miscue injection (substitutions, omissions, insertions, self-corrections,
   hesitations) with an explicit seed — gold labels are known because we wrote them.
3. TTS rendering → 16 kHz mono clips, validated per clip.
4. A blind hand-verification pass on a subset; the mismatch rate is published.

Stated caveat, always: TTS is not child speech. Consented real-child clips serve as
the validity anchor, and hesitation/self-correction classes additionally test whether
an ASR *preserves disfluencies at all* — many production models normalize them away.

The rendered benchmark (clips + gold labels + manifest) ships as a versioned release
with a checksum-verified fetch script, so consumers never need the TTS toolchain:

```bash
python scripts/fetch_benchmark.py   # downloads + verifies against the committed lockfile
```

## Bring your own ASR

The scorer accepts hypothesis files from any ASR — run your model over the clips,
emit `hypotheses.jsonl`, and score it without installing this project's ASR stack.
See `docs/BENCHMARK.md` (schema, defaults, and what missing timings/confidences do
to each metric).

## speechocean762 (optional, different task)

`mispeech/speechocean762` supports word-level **mispronunciation detection** (it has
pronunciation-accuracy scores, not miscue-class labels). If reported, it is reported
as that task, never as miscue F1.
