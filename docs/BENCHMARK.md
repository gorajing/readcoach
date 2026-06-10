# ReadCoach miscue benchmark — BYO-ASR scoring

This is the audit document for the **bring-your-own-ASR** scoring path. It lets
you run *your* speech recognizer over the ReadCoach clips, emit a small JSONL,
and score it against the committed gold using the **exact same** miscue detector
and metrics that produced the published faster-whisper baseline.

The goal is that a skeptical engineer can confirm, in about five minutes, that:

1. the scoring code imports **only jiwer + the Python standard library** — no
   model, no LLM client, no telemetry; and
2. every published number reproduces from a committed command.

---

## 0. Auditable in five minutes — the import claim, stated precisely

A scoring run imports the `readcoach` package (for the miscue detector and the
`Word` / `AsrResult` dataclasses), which transitively imports **jiwer** and the
**standard library** — and nothing else of consequence. In particular, a real
scoring run **never imports** any of:

```
faster_whisper   weave   anthropic   google.genai   wandb
```

To be precise about what *is* and *is not* imported:

- **Imported:** `readcoach.bench_cli`, `readcoach.miscue`, `readcoach.asr`
  (only the `Word` / `AsrResult` dataclasses and cache helpers — the
  faster-whisper import inside `readcoach.asr` is **lazy**, on the real
  transcription path only, which scoring never takes), and `jiwer`.
- **Not imported:** the model and service dependencies listed above. The
  `readcoach` package is imported; its heavyweight model deps are not.

Verify it yourself in one command (run from the repo root, after the fetch in
§1):

```bash
python -c '
import sys
from pathlib import Path
from readcoach.bench_cli import load_gold, load_hypotheses, check_coverage, score_hypotheses
import json, tempfile
gold = load_gold(Path("data/benchmark/gold.jsonl"))
utts = list(gold)[:3]
rows = [{"utt_id": u, "words": [{"text": w} for w in gold[u].target_text.split()]} for u in utts]
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    for r in rows: f.write(json.dumps(r)+"\n")
    p = f.name
hyp = load_hypotheses(Path(p), gold); check_coverage(hyp, gold, Path(p), allow_partial=True)
score_hypotheses(hyp, gold)
banned = ["faster_whisper","weave","anthropic","google.genai","wandb"]
present = [m for m in banned if m in sys.modules]
print("BANNED present:", present)
assert not present, present
print("OK: scoring imported none of the model/service modules")
'
```

This is the same assertion the test suite makes
(`tests/test_bench_cli.py::test_import_audit_subprocess`, run in a fresh
interpreter so it cannot be contaminated by other tests).

---

## 1. Fetch the benchmark (no credentials)

```bash
python3 scripts/fetch_benchmark.py        # stdlib only; downloads + verifies
```

This downloads the release tarball, verifies its sha256 against
`evals/golden/benchmark.lock`, extracts into `data/benchmark/`, and verifies
every extracted file's sha256. Any mismatch is a loud, non-zero-exit failure —
there is no partial-success path.

After fetching you have:

```
data/benchmark/
  gold.jsonl        # 88 items (this file is also committed in git)
  manifest.json     # benchmark_version, coverage_matrix
  clips/            # 88 WAVs (gitignored; fetched, not committed)
```

`gold.jsonl` is small and committed to the repo, so the scorer and its tests run
even before you fetch the audio. You only need the WAVs to run *your own ASR*
over them (§5).

---

## 2. Gold schema (`gold.jsonl`)

One JSON object per line. 88 items across passages `p01`…`pNN`, three difficulty
bands.

| field         | type                | meaning                                                      |
|---------------|---------------------|--------------------------------------------------------------|
| `utt_id`      | str                 | unique id, e.g. `p01-sub9`; also the WAV stem                |
| `passage_id`  | str                 | source passage, e.g. `p01`                                   |
| `band`        | int                 | difficulty band                                              |
| `target_text` | str                 | the passage the reader was asked to read (the alignment reference) |
| `miscued_text`| str                 | the text actually realized in the synthesized clip          |
| `gold`        | list[object]        | the gold miscues (see below); empty list for a clean read   |
| `voice`       | str                 | TTS voice                                                    |
| `rate_wpm`    | int                 | speaking rate                                                |
| `wav_sha256`  | str                 | sha256 of the clip (also in the lock)                        |
| `duration_s`  | float               | clip duration                                                |

Each entry in `gold` is one labeled miscue:

| field         | type                     | meaning                                              |
|---------------|--------------------------|------------------------------------------------------|
| `type`        | one of the 5 classes     | `substitution` / `omission` / `insertion` / `self_correction` / `hesitation` |
| `target_word` | str \| null              | the expected word (null for insertions / repeat-hesitations) |
| `said_word`   | str \| null              | what was actually read (null for omissions / silence-hesitations) |
| `index`       | int                      | target-text word index (0-based)                     |
| `render`      | null \| "filler" \| "silence" | hesitation subtype annotation (see §4.5)        |

The benchmark contains, by class: **24 substitution · 24 omission · 24 insertion
· 24 self_correction · 32 hesitation** gold miscues. The 32 hesitations split
into **16 `filler`** and **16 `silence`** (this split matters for the no-timings
recall floor — §3.2).

---

## 3. Hypotheses schema (`hypotheses.jsonl`)

This is the file **you** produce by running your ASR. One JSON object per line:

```json
{"utt_id": "p01-clean",
 "words": [{"text": "The", "start": 0.10, "end": 0.32, "confidence": 0.98},
           {"text": "cat", "start": 0.33, "end": 0.61}]}
```

| field          | required | default | effect on scoring                              |
|----------------|----------|---------|------------------------------------------------|
| `utt_id`       | yes      | —       | must be a gold `utt_id`; unknown ids are an error |
| `words`        | yes      | —       | ordered list of word objects                   |
| `words[].text` | yes      | —       | the recognized word; missing `text` is an error |
| `words[].start`| no       | `null`  | word onset (s); see §3.2                        |
| `words[].end`  | no       | `null`  | word offset (s); see §3.2                       |
| `words[].confidence` | no | `1.0`   | see §3.3                                        |

### 3.1 Coverage

By default a hypotheses file **must cover every gold item**. A file covering a
strict subset is an **error** — silent partial scoring is forbidden. Pass
`--allow-partial` to score the covered subset; `n_covered` is then printed
prominently in the table and the JSON output. Other fail-loud conditions:
duplicate `utt_id` within the file, a `utt_id` not present in gold, a missing
`text` / `words` / `utt_id` field, or a malformed JSON line.

### 3.2 Missing timings → silence-hesitation recall floor ≈ 0

Word timings (`start` / `end`) are **optional**. If you omit them, alignment-based
classes (substitution, omission, insertion, self_correction) and **filler**
hesitations are unaffected — they need no timing.

But the **silence-hesitation** rule (a pause longer than `GAP_THRESHOLD_S = 1.0 s`
between two consecutive words) **requires both `prev.end` and `cur.start`**. With
no timings, that rule never fires, so the **16 `silence`-render gold hesitations
become undetectable**: recall on that subset has a hard floor of ≈ 0. The
detector skips untimed gaps silently (an accepted, documented limitation in
`readcoach.miscue._timing_hesitations`), so the only honest mitigation is to
**emit timings**.

Concretely: of the 32 gold hesitations, **16 are `filler`** (detectable from
`text` alone, if your ASR preserves the filler token) and **16 are `silence`**
(detectable only with timings). The worked wav2vec2 example (§6) *does* emit
timings, so it has a shot at the silence subset.

### 3.3 Missing confidence → no effect on scoring today

`confidence` defaults to `1.0` when omitted. It is carried through to the `Word`
objects for a future learner model (T3) that consumes per-observation confidence,
but **today's scoring does not read it** — the rule-based detector treats every
word identically regardless of confidence. Omitting it changes nothing in the
numbers. (The wav2vec2 example deliberately omits it, to prove the field is
optional.)

---

## 4. Published class definitions

Alignment is done by `jiwer` over the **normalized** token streams (strip edge
punctuation, casefold; fillers stripped — §4.4) of `target_text` and your
hypothesis. Matching during scoring uses a **±1 index tolerance**, greedy
one-to-one within each class. Metrics are **micro-aggregated** (TP/FP/FN summed
across all items before dividing).

### 4.1 substitution / omission / insertion (ALIGNMENT classes)

Derived directly from the edit-distance alignment:

- **substitution** — a target word was read as a different word.
- **omission** — a target word was not read.
- **insertion** — an extra word with no target slot (and not a repeat / not a
  self-correction attempt — see below).

### 4.2 self_correction — the similarity gate (a stated limitation)

A **wrong attempt immediately followed by the correct word** counts as
`self_correction` **only if** the wrong attempt is an orthographic near-miss of
the target: `difflib.SequenceMatcher(None, wrong, target).ratio() >= 0.5`.

- `bat` → `cat` (ratio 0.67) and `ran` → `run` (0.67) **are** self-corrections.
- A **dissimilar** wrong-then-correct pair — `big cat` where the target is `cat`
  (ratio 0.0) — is classified as an **insertion**, not a self-correction.

This is a deliberate proxy: we have no phonetic model, so orthographic similarity
stands in for "sound-out attempt." It is a **published limitation, not a hidden
one** — a real but dissimilar self-correction will be scored as an insertion.
Self-corrections detected via this heuristic carry a confidence discount inside
the detector (`0.7`), reflecting the structural ambiguity.

### 4.3 hesitation — repeats, fillers, and silences

Three things produce a `hesitation`:

- **Repeat** — an inserted token identical to the immediately preceding target
  word (`the the …`). Repeats are **always** hesitations, never self-correction
  attempts. The hesitation is anchored to that preceding target index.
- **Filler** — a disfluency token (`um uh er hmm mm …`) — see §4.4.
- **Silence** — a pause `> 1.0 s` between consecutive timed words — see §3.2.

### 4.4 Filler stripping (pre-alignment)

Fillers (`um`, `uh`, `er`, `hmm`, `mm`, and a few variants) are **removed from the
hypothesis before alignment**, so they never surface as spurious insertions.
Each removed filler instead emits a `hesitation` bound to the next surviving
target index. Note: most production ASR **normalizes fillers out** of its
transcript, so a filler lexicon alone has roughly zero recall unless your ASR is
configured to preserve disfluencies — which is exactly why the two
transcript-style classes are reported separately (§4.6).

### 4.5 The `render` annotation

Gold hesitations carry `render` ∈ {`filler`, `silence`} purely to document which
detection path each one exercises. Both are honest targets; scoring treats them
identically (a hesitation TP is a hesitation TP). The split only matters for
understanding the no-timings recall floor (§3.2).

### 4.6 Why self_correction + hesitation are reported separately

`self_correction` and `hesitation` are **transcript-style-sensitive**: they
measure whether your ASR *preserves* disfluencies (repeated words, filler tokens,
sound-out attempts) at least as much as they measure the detector. An ASR that
cleans up its transcript will score low on these classes for reasons that have
nothing to do with miscue detection. The alignment classes (sub/om/ins) do not
have this confound. The scoring table therefore puts these two classes in a
separate **TRANSCRIPT-STYLE-SENSITIVE** group with a one-line caveat.

---

## 5. Quickstart (≈ 5 minutes, zero credentials)

```bash
# 1. Fetch clips + gold (stdlib only).
python3 scripts/fetch_benchmark.py

# 2. Run YOUR ASR over data/benchmark/clips/*.wav and write hypotheses.jsonl
#    (one line per utt_id, schema in §3). The wav2vec2 example in
#    examples/wav2vec2_adapter/ is a complete worked adapter.

# 3. Score it beside the committed baseline.
uv run readcoach-bench score --hypotheses mine=hypotheses.jsonl
#   (or: python -m readcoach.bench_cli score --hypotheses mine=hypotheses.jsonl
#    after `pip install -e .`)
```

The table prints per-class P/R/F1 and `fp_per_100_correct_words` for your set
beside the committed faster-whisper-small (bias=none) baseline.

### Scoring several sets → your own masking curve

```bash
uv run readcoach-bench score \
  --hypotheses base=hyp_greedy.jsonl \
  --hypotheses biased=hyp_with_target_prompt.jsonl \
  --json curve.json
```

With **two or more** `--hypotheses` sets, the consumer columns **are** your own
masking-curve table — one column per ASR / decoding configuration. Compare
`fp_per_100` and per-class recall across the columns to see the
bias-vs-false-positive tradeoff on *your* stack. `--json` writes the same data
machine-readably.

CLI flags:

```
readcoach-bench score
  --hypotheses NAME=path.jsonl   # repeatable; NAME= optional (defaults to file stem)
  --gold        PATH             # default: data/benchmark/gold.jsonl
  --baseline    PATH             # default: evals/results/v0.json
  --allow-partial                # permit a subset; prints n_covered prominently
  --json        PATH             # also write machine-readable output
```

---

## 6. Worked second ASR — wav2vec2

`examples/wav2vec2_adapter/` is a complete, self-contained adapter for a second
**public** ASR (`facebook/wav2vec2-base-960h`). It proves the contract is **not**
shaped around faster-whisper. It transcribes the 88 fetched clips with word
timings (`output_word_offsets=True`) and emits a hypotheses file that **omits
`confidence`** — demonstrating that field's optionality. See its README; the
generated `hypotheses_wav2vec2.jsonl` is committed as the worked example's
receipt (regenerate it with the script).

---

## 7. Baseline provenance

The baseline column reads `evals/results/v0.json → metrics.miscue`, which is the
**bias=none** (raw-acoustic, no target-text leakage into the ASR),
micro-aggregated result over all 88 items.

| property         | value                                                            |
|------------------|------------------------------------------------------------------|
| results file     | `evals/results/v0.json`                                          |
| model / backend  | `faster-whisper-small`                                           |
| bias setting     | `none` (purely acoustic decoding)                                |
| aggregation      | micro (TP/FP/FN summed across items, then divided)               |
| commit           | `65bec863bf6bc3d4a4a107242a5bfebdc2f9c5f6`                       |
| gold (golden) sha256 | `886366474ffbf0d709a773f05a929d9b3e8de3887d96f62dea6e1380ee2e7688` |
| date             | 2026-06-10                                                       |

Reproduce the baseline from committed fixtures (hermetic, no model load):

```bash
uv run python scripts/run_benchmark.py --fixtures --version v0
```

`v0.json` also carries all three bias settings under `metrics.by_bias`
(`none` / `prompt` / `strong`) for reference; the BYO-ASR table uses `none`.
