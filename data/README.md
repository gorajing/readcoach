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

The rendered benchmark (clips + gold labels + manifest) ships as a versioned GitHub
release tarball with a checksum-verified fetch script.  Consumers — including CI and
anyone who doesn't have macOS — use the fetch script; they never need to rebuild.

```bash
# From the project root (stdlib only — no pip install required):
python3 scripts/fetch_benchmark.py

# Re-download even if data/benchmark/ already exists and verifies:
python3 scripts/fetch_benchmark.py --force
```

The script:
1. Downloads `readcoach-benchmark-0.1.0.tar.gz` from the GitHub release.
2. Verifies the tarball sha256 against `evals/golden/benchmark.lock` (`tarball.sha256`).
3. Safely extracts `gold.jsonl`, `manifest.json`, and `clips/*.wav` into
   `data/benchmark/` (rejects any tarball path that escapes the target directory).
4. Verifies every extracted file's sha256 against the lock `artifacts` entries.
5. Exits 0 only if all checks pass; any mismatch → non-zero exit with a loud error.

**Rebuilding from source (maintainers only — requires macOS + `say` + ffmpeg):**

```bash
uv run python scripts/build_benchmark.py        # renders all 88 clips
uv run python scripts/build_benchmark.py --verify  # re-checks the lock
uv run python scripts/make_benchmark_tarball.py    # packs dist/readcoach-benchmark-0.1.0.tar.gz
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
