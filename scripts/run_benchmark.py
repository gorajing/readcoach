"""Benchmark runner — micro-aggregated per-class × bias scoring.

Reads data/benchmark/gold.jsonl; for each item × each requested bias setting
(none, prompt, strong):
  1. Transcribes the clip with the appropriate bias (ASR result cached
     content-addressed under evals/golden/asr_cache/).
  2. Runs detect() against target_text to produce predicted Miscues.
  3. Computes match_counts() (TP/FP/FN per class) vs the gold Miscues.

Micro-aggregation
-----------------
Corpus-level P/R/F1 is computed by summing TP/FP/FN across ALL items in a
(bias, class) cell before dividing — this is MICRO-aggregation.  It gives equal
weight to each individual miscue instance rather than equal weight to each
utterance (macro).  Micro is the standard choice for imbalanced detection tasks:
a passage with 3 gold substitutions contributes 3 denominator slots, not 1.
Macro would up-weight short passages with few events; micro weights by event
frequency, which better reflects operational recall on a real reading session.

fp_per_100_correct_words is also micro-aggregated: total_fp / total_correct_words
* 100 across all items in the cell.

Output JSON (--out, default evals/results/miscue-v0.json)
---------------------------------------------------------
{
  "metadata": {
    "backend": str,
    "benchmark_version": str,
    "gold_sha256": str,          # from benchmark.lock
    "git_commit": str,           # HEAD at run time
    "date": str,                 # ISO date
    "n_items": int,              # items actually processed
    "biases_run": [str, ...],
    "aggregation": "micro"
  },
  "results": {
    "<bias>": {
      "<class>": {
        "tp": int, "fp": int, "fn": int,
        "precision": float | null,
        "recall": float | null,
        "f1": float | null,
        "n_gold": int,
        "n_pred": int
      },
      ...
      "fp_per_100_correct_words": float
    },
    ...
  }
}

Flags
-----
--bias          subset of {none, prompt, strong}; may repeat; default all three
--limit N       process only the first N items (for smoke tests)
--out PATH      output JSON path (default: evals/results/miscue-v0.json)
--backend       ASR backend string (default: faster-whisper-small)
--fixtures      CI-hermetic mode: cache_only=True, WEAVE_DISABLED=1, passes
                report through evaluate() into evals/results/<version>.json;
                requires --version; aborts loudly on any CacheMiss.
--version VER   version string for the EvalReport (required with --fixtures).
--results-dir   results directory override for --fixtures mode
                (default: evals/results).

--fixtures mode output structure
----------------------------------
The metrics dict passed to evaluate() is shaped for gate.py:
  {
    "miscue": {
      "substitution": {precision, recall, f1},
      "omission": ..., "insertion": ..., "self_correction": ...,
      "hesitation": ...,
      "fp_per_100_correct_words": float
    },
    "by_bias": {
      "none":   <same per-class structure>,
      "prompt": ...,
      "strong": ...,
    },
    "invariants": {
      "violations": 0   # reserved; no tutor invariant checks exist yet
    },
    "latency": {
      "decision_ms_p50": null,
      "decision_ms_p95": null,
      "rtf_offline_proxy": null
    }
  }

The "miscue" block at top level uses the bias=none results (the raw-acoustic
measurements that gate rules operate on).  All three biases are nested under
"by_bias" for reference.

Fail-loud contract
------------------
Any missing clip, transcription error, or detect() error aborts the run with
a non-zero exit code.  There is NO per-item try/except.  This makes failures
visible immediately rather than silently skewing aggregate numbers.
In --fixtures mode, a CacheMiss aborts with exit code 1 — it means the
committed asr_cache is incomplete, which must be fixed, not papered over.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `evals` is importable when
# this script is invoked directly (e.g. `uv run python scripts/run_benchmark.py`).
# pytest adds the project root via pyproject.toml pythonpath=["."], but direct
# invocation does not.
# ---------------------------------------------------------------------------
_SCRIPT_PROJECT_ROOT = Path(__file__).parent.parent
if str(_SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project root — all paths are absolute
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_GOLD_JSONL = _PROJECT_ROOT / "data" / "benchmark" / "gold.jsonl"
_LOCK_FILE = _PROJECT_ROOT / "evals" / "golden" / "benchmark.lock"
_DEFAULT_OUT = _PROJECT_ROOT / "evals" / "results" / "miscue-v0.json"
_DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "evals" / "results"
_ALL_BIASES = ("none", "prompt", "strong")


# ---------------------------------------------------------------------------
# Gold parsing helpers
# ---------------------------------------------------------------------------

def _load_gold_rows(limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with _GOLD_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]
    return rows


def _gold_sha256() -> str:
    """Return the gold.jsonl sha256 from the committed benchmark.lock."""
    lock = json.loads(_LOCK_FILE.read_text(encoding="utf-8"))
    key = "data/benchmark/gold.jsonl"
    return lock["artifacts"][key]


def _row_to_gold_miscues(row: dict):  # -> list[Miscue]
    """Parse the gold list from a gold.jsonl row into Miscue objects.

    Gold entries carry a ``render`` field (None | "filler" | "silence") that
    distinguishes hesitation subtypes.  Both subtypes are included in gold — the
    filler path and the silence/timing path are both honest targets.
    """
    from readcoach.miscue import Miscue  # noqa: PLC0415

    miscues = []
    for entry in row["gold"]:
        miscues.append(
            Miscue(
                type=entry["type"],
                target_word=entry["target_word"],
                said_word=entry["said_word"],
                index=entry["index"],
            )
        )
    return miscues


# ---------------------------------------------------------------------------
# Git HEAD helper
# ---------------------------------------------------------------------------

def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_CLASSES = ("substitution", "omission", "insertion", "self_correction", "hesitation")


def _empty_accum() -> dict:
    """Return zero-initialized per-class accumulators for one (bias,) cell."""
    out: dict = {}
    for cls in _CLASSES:
        out[cls] = {"tp": 0, "fp": 0, "fn": 0}
    out["_correct_words"] = 0
    out["_total_fp"] = 0
    return out


def _add_counts(accum: dict, counts: dict) -> None:
    """Add match_counts() output into the running accumulator (in-place)."""
    for cls in _CLASSES:
        accum[cls]["tp"] += counts[cls]["tp"]
        accum[cls]["fp"] += counts[cls]["fp"]
        accum[cls]["fn"] += counts[cls]["fn"]
        accum["_total_fp"] += counts[cls]["fp"]
    accum["_correct_words"] += counts["_correct_words"]


def _finalize(accum: dict) -> dict:
    """Compute P/R/F1 from accumulated micro-sums; emit fp_per_100_correct_words."""
    result: dict = {}
    for cls in _CLASSES:
        tp = accum[cls]["tp"]
        fp = accum[cls]["fp"]
        fn = accum[cls]["fn"]
        n_pred = tp + fp
        n_gold = tp + fn

        if n_gold == 0 and n_pred == 0:
            precision = recall = f1 = None
        else:
            precision = tp / n_pred if n_pred else 0.0
            recall = tp / n_gold if n_gold else 0.0
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

        result[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_gold": n_gold,
            "n_pred": n_pred,
        }

    correct_words = accum["_correct_words"]
    total_fp = accum["_total_fp"]
    if correct_words > 0:
        result["fp_per_100_correct_words"] = total_fp / correct_words * 100
    else:
        result["fp_per_100_correct_words"] = float(total_fp * 100)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run miscue benchmark: per-class × bias micro-aggregated scoring."
    )
    parser.add_argument(
        "--bias",
        choices=list(_ALL_BIASES),
        action="append",
        dest="biases",
        default=None,
        help="Bias setting(s) to run.  May be repeated.  Default: all three.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N items (for smoke tests).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output JSON path (default: {_DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--backend",
        default="faster-whisper-small",
        help="ASR backend identifier (default: faster-whisper-small).",
    )
    parser.add_argument(
        "--fixtures",
        action="store_true",
        default=False,
        help=(
            "CI-hermetic mode: cache_only=True, WEAVE_DISABLED=1. "
            "Runs the full 88×3 sweep purely from committed asr_cache; aborts "
            "loudly on any CacheMiss.  Passes report through evaluate() so it "
            "lands as an immutable evals/results/<version>.json.  Requires --version."
        ),
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version string for the EvalReport (required with --fixtures).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_DEFAULT_RESULTS_DIR,
        help=f"Results directory for --fixtures mode (default: {_DEFAULT_RESULTS_DIR}).",
    )
    args = parser.parse_args(argv)

    # --fixtures validation
    if args.fixtures:
        if not args.version:
            print(
                "ERROR: --fixtures requires --version (e.g. --version v0)",
                file=sys.stderr,
            )
            sys.exit(1)
        # Set WEAVE_DISABLED before any tracing import has an effect.
        os.environ["WEAVE_DISABLED"] = "1"

    biases: tuple[str, ...] = tuple(args.biases) if args.biases else _ALL_BIASES

    # --- Load gold ---
    rows = _load_gold_rows(args.limit)
    if not rows:
        print("ERROR: no items to process (gold.jsonl empty or --limit 0)", file=sys.stderr)
        sys.exit(1)

    gold_sha = _gold_sha256()
    manifest_path = _PROJECT_ROOT / "data" / "benchmark" / "manifest.json"
    benchmark_version = json.loads(manifest_path.read_text(encoding="utf-8"))["benchmark_version"]

    # --- Imports (lazy — avoid model load just for --help) ---
    from readcoach.asr import CacheMiss, _CACHE_DIR, _deserialize_result, transcribe  # noqa: PLC0415,F401
    from readcoach.miscue import detect, match_counts  # noqa: PLC0415

    clips_dir = _PROJECT_ROOT / "data" / "benchmark" / "clips"

    # --fixtures: load the committed asr_cache_manifest.json so we can look up
    # ASR results by (utt_id, bias, backend) without reading audio files.
    # *.wav files are excluded from git via .gitignore; the manifest provides the
    # cache-key (hex filename) for each (utt_id × bias × backend) combination.
    asr_cache_manifest: dict | None = None
    if args.fixtures:
        manifest_path = _PROJECT_ROOT / "evals" / "golden" / "asr_cache_manifest.json"
        if not manifest_path.exists():
            print(
                f"ERROR: --fixtures requires evals/golden/asr_cache_manifest.json "
                f"(not found at {manifest_path}).  "
                "Regenerate it with scripts/build_cache_manifest.py.",
                file=sys.stderr,
            )
            sys.exit(1)
        asr_cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            f"  --fixtures: loaded manifest with {len(asr_cache_manifest)} entries",
            file=sys.stderr,
        )

    # accum[bias] = running per-class TP/FP/FN
    accum: dict[str, dict] = {b: _empty_accum() for b in biases}

    for row in rows:
        utt_id = row["utt_id"]
        target_text = row["target_text"]
        clip_path = str(clips_dir / f"{utt_id}.wav")

        # In non-fixtures mode, audio files must exist.  In --fixtures mode we
        # use the params index and never touch the audio files.
        if not args.fixtures and not Path(clip_path).exists():
            raise FileNotFoundError(
                f"Clip missing for utt_id={utt_id!r}: expected {clip_path}"
            )

        gold_miscues = _row_to_gold_miscues(row)
        n_target_words = len(target_text.split())

        for bias in biases:
            print(f"  {utt_id}  bias={bias}", file=sys.stderr)

            # bias="none" must NOT pass target_text to the model — the cache key
            # bakes the params, so this is also how the cache is keyed correctly.
            asr_target = None if bias == "none" else target_text

            if args.fixtures:
                # --fixtures: look up the cache key from the committed manifest,
                # then read the cache file directly.  Never loads a model or audio.
                manifest_key = f"{utt_id}|{bias}|{args.backend}"
                cache_hex = asr_cache_manifest.get(manifest_key)  # type: ignore[union-attr]
                if cache_hex is None:
                    print(
                        f"\nFATAL: CacheMiss in --fixtures mode — manifest has no entry "
                        f"for key {manifest_key!r}.  The committed manifest is incomplete.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                cache_file = _CACHE_DIR / f"{cache_hex}.json"
                if not cache_file.exists():
                    print(
                        f"\nFATAL: CacheMiss in --fixtures mode — cache file missing: "
                        f"{cache_file}\n  key={manifest_key!r}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                raw = json.loads(cache_file.read_text(encoding="utf-8"))
                asr_result = _deserialize_result(raw)
            else:
                try:
                    asr_result = transcribe(
                        clip_path,
                        target_text=asr_target,
                        bias=bias,
                        backend=args.backend,
                    )
                except CacheMiss as exc:
                    # This branch only runs in non-fixtures mode; CacheMiss here means
                    # the cache_only flag was somehow set externally — abort loudly.
                    print(
                        f"\nFATAL: unexpected CacheMiss (non-fixtures mode) — "
                        f"utt_id={utt_id!r}  bias={bias!r}\n  {exc}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            # detect() always gets target_text (the alignment reference is
            # independent of the ASR bias setting).
            predicted = detect(asr_result, target_text)

            counts = match_counts(predicted, gold_miscues, n_target_words)
            _add_counts(accum[bias], counts)

    # --- Finalize per-bias results ---
    finalized: dict[str, dict] = {bias: _finalize(accum[bias]) for bias in biases}

    if args.fixtures:
        # --fixtures mode: pass through evaluate() so the report is immutable
        # and versioned.  The metrics dict is structured for gate.py:
        #   - "miscue" (top level) = bias=none metrics (gate rules operate here)
        #   - "by_bias" = all three biases for reference
        #   - "invariants.violations" = 0 (reserved; no tutor checks exist yet)
        #   - "latency" = injected by evaluate() with None values
        if "none" not in finalized:
            print(
                "ERROR: --fixtures requires bias=none to be in the run "
                "(gate rules operate on bias=none metrics).  "
                "Do not use --bias with --fixtures.",
                file=sys.stderr,
            )
            sys.exit(1)

        none_result = finalized["none"]
        metrics: dict = {
            "miscue": {
                "substitution": {
                    "precision": none_result["substitution"]["precision"],
                    "recall": none_result["substitution"]["recall"],
                    "f1": none_result["substitution"]["f1"],
                },
                "omission": {
                    "precision": none_result["omission"]["precision"],
                    "recall": none_result["omission"]["recall"],
                    "f1": none_result["omission"]["f1"],
                },
                "insertion": {
                    "precision": none_result["insertion"]["precision"],
                    "recall": none_result["insertion"]["recall"],
                    "f1": none_result["insertion"]["f1"],
                },
                "self_correction": {
                    "precision": none_result["self_correction"]["precision"],
                    "recall": none_result["self_correction"]["recall"],
                    "f1": none_result["self_correction"]["f1"],
                },
                "hesitation": {
                    "precision": none_result["hesitation"]["precision"],
                    "recall": none_result["hesitation"]["recall"],
                    "f1": none_result["hesitation"]["f1"],
                },
                "fp_per_100_correct_words": none_result["fp_per_100_correct_words"],
            },
            "by_bias": finalized,
            "invariants": {
                # violations=0: no tutor invariant checks exist yet; metric path
                # is reserved and gated from day one so CI enforces it immediately
                # once real checks are added.
                "violations": 0,
            },
            # latency block: evaluate() will inject it (see harness._inject_latency_block).
            # Values remain None until the replay machinery exists (T3+).
        }

        from evals.harness import evaluate  # noqa: PLC0415

        report = evaluate(
            version=args.version,
            golden_path=str(_LOCK_FILE),
            metrics=metrics,
            results_dir=str(args.results_dir),
        )
        out_path = args.results_dir / f"{args.version}.json"
        print(
            f"\n--fixtures run complete.  Report: {out_path}  "
            f"(weave: {report.metadata.get('weave', 'unknown')})",
            file=sys.stderr,
        )
        return

    # --- Legacy / direct mode: write flat JSON to --out ---
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "backend": args.backend,
            "benchmark_version": benchmark_version,
            "gold_sha256": gold_sha,
            "git_commit": _git_head(),
            "date": datetime.date.today().isoformat(),
            "n_items": len(rows),
            "biases_run": list(biases),
            "aggregation": "micro",
        },
        "results": finalized,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
