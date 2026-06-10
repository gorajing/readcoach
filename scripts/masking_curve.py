"""Masking curve — per-class recall/precision vs bias with bootstrap CIs.

For each item × bias setting: transcribes the clip (cache-fast), computes
per-item match_counts (TP/FP/FN per class), and measures two WERs:

  wer_vs_spoken  : hypothesis vs miscued_text  (ASR fidelity — what was said)
  wer_vs_target  : hypothesis vs target_text   (prompt-echo effect)

Bootstrap
---------
1 000 resamples of the 88 items WITH replacement, seeded (default 1337).
Per resample: recompute micro per-class P/R + fp_per_100 + both mean WERs
from the pre-computed per-item statistics.  No re-transcription happens
during bootstrap — only numpy index arithmetic over cached per-item arrays.

Normalization note
------------------
WER is computed on tokens normalized with the SAME function the miscue
detector uses (_normalize: strip + casefold).  This makes the WER metric
consistent with the alignment that produces the miscue predictions — a token
the detector considers "equal" also counts as equal for WER purposes.

Agreement check
---------------
The per-bias micro P/R/F1 numbers recomputed here are asserted against
evals/results/miscue-v0.json within AGREE_TOL (1e-9).  Any drift between
the two scripts raises immediately.

Output
------
  evals/results/masking_curve.json   — per-bias × class stats + CIs + metadata
  evals/results/masking_curve.png    — 2-panel figure

Usage
-----
  uv run python scripts/masking_curve.py [--seed N] [--n-boot N] [--limit N]
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_GOLD_JSONL = _PROJECT_ROOT / "data" / "benchmark" / "gold.jsonl"
_BASELINE_JSON = _PROJECT_ROOT / "evals" / "results" / "miscue-v0.json"
_OUT_JSON = _PROJECT_ROOT / "evals" / "results" / "masking_curve.json"
_OUT_PNG = _PROJECT_ROOT / "evals" / "results" / "masking_curve.png"

_ALL_BIASES = ("none", "prompt", "strong")
_CLASSES = ("substitution", "omission", "insertion", "self_correction", "hesitation")

# Agreement tolerance: recomputed micro P/R must match miscue-v0.json within this.
AGREE_TOL = 1e-9


# ---------------------------------------------------------------------------
# Git HEAD
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
# Normalization (mirrors readcoach.miscue._normalize exactly)
# ---------------------------------------------------------------------------

_EDGE_PUNCT = ".,!?;:\"'`()[]{}…—–-"


def _normalize(token: str) -> str:
    """Strip whitespace + leading/trailing punctuation, then casefold.

    Identical to readcoach.miscue._normalize — used here for WER computation
    so that WER and the miscue detector share the same tokenization.
    """
    return token.strip().strip(_EDGE_PUNCT).casefold()


def _normalize_sentence(text: str) -> str:
    """Return a space-joined string of normalized non-empty tokens."""
    return " ".join(t for t in (_normalize(tok) for tok in text.split()) if t)


# ---------------------------------------------------------------------------
# WER (on pre-normalized strings, no jiwer transforms applied)
# ---------------------------------------------------------------------------

def _wer_normalized(reference: str, hypothesis: str) -> float:
    """Compute WER on already-normalized space-separated token strings.

    Applies the same edit-distance logic as jiwer but operates on
    pre-normalized strings (no additional transforms).  A blank reference
    with a blank hypothesis = 0.0; a blank reference with a non-blank
    hypothesis = 1.0 per-word.
    """
    # jiwer.wer applies its own default transforms; we pass identity-transformed
    # strings by using ReduceToListOfListOfWords on pre-split token lists.
    ref_tokens = reference.split() if reference.strip() else []
    hyp_tokens = hypothesis.split() if hypothesis.strip() else []

    if not ref_tokens:
        # Degenerate: nothing to transcribe — treat as 0 errors / 0 words.
        return 0.0

    # Levenshtein WER manually to avoid jiwer transform overhead on already-clean tokens.
    # Faster and cleaner than fighting jiwer's pipeline for pre-normalized strings.
    r, h = ref_tokens, hyp_tokens
    nr, nh = len(r), len(h)

    # DP table: standard edit-distance on word sequences.
    # dp[j] = edit distance between r[:i] and h[:j].
    # After the loop, dp[nh] = edit distance between the full r and h.
    dp = list(range(nh + 1))
    for i in range(1, nr + 1):
        new_dp = [i] + [0] * nh
        for j in range(1, nh + 1):
            if r[i - 1] == h[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp

    # dp[nh] is the edit distance; WER = edit_distance / len(reference).
    return dp[nh] / nr


# ---------------------------------------------------------------------------
# Gold parsing
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


def _row_to_gold_miscues(row: dict):  # -> list[Miscue]
    from readcoach.miscue import Miscue  # noqa: PLC0415
    return [
        Miscue(
            type=e["type"],
            target_word=e["target_word"],
            said_word=e["said_word"],
            index=e["index"],
        )
        for e in row["gold"]
    ]


# ---------------------------------------------------------------------------
# Per-item statistics (the resampling unit)
# ---------------------------------------------------------------------------

def _compute_per_item_stats(
    rows: list[dict],
    biases: tuple[str, ...],
    backend: str,
) -> dict[str, list[dict]]:
    """Return per_item_stats[bias] = list of per-item dicts.

    Each per-item dict holds:
      "counts"       : match_counts() output (TP/FP/FN per class + _correct_words)
      "wer_vs_spoken": WER(hyp, miscued_text)   [normalized]
      "wer_vs_target": WER(hyp, target_text)    [normalized]
    """
    from readcoach.asr import transcribe  # noqa: PLC0415
    from readcoach.miscue import detect, match_counts  # noqa: PLC0415

    clips_dir = _PROJECT_ROOT / "data" / "benchmark" / "clips"

    per_item: dict[str, list[dict]] = {b: [] for b in biases}
    n = len(rows)

    for idx, row in enumerate(rows):
        utt_id = row["utt_id"]
        target_text = row["target_text"]
        miscued_text = row["miscued_text"]
        clip_path = str(clips_dir / f"{utt_id}.wav")

        if not Path(clip_path).exists():
            raise FileNotFoundError(f"Clip missing for utt_id={utt_id!r}: {clip_path}")

        gold_miscues = _row_to_gold_miscues(row)
        n_target_words = len(target_text.split())

        for bias in biases:
            print(f"  [{idx+1}/{n}] {utt_id}  bias={bias}", file=sys.stderr)

            asr_target = None if bias == "none" else target_text
            asr_result = transcribe(
                clip_path,
                target_text=asr_target,
                bias=bias,
                backend=backend,
            )

            predicted = detect(asr_result, target_text)
            counts = match_counts(predicted, gold_miscues, n_target_words)

            # WER on normalized tokens (same normalization as the detector).
            hyp_norm = _normalize_sentence(asr_result.text)
            spoken_norm = _normalize_sentence(miscued_text)
            target_norm_str = _normalize_sentence(target_text)

            wer_vs_spoken = _wer_normalized(spoken_norm, hyp_norm)
            wer_vs_target = _wer_normalized(target_norm_str, hyp_norm)

            per_item[bias].append({
                "counts": counts,
                "wer_vs_spoken": wer_vs_spoken,
                "wer_vs_target": wer_vs_target,
            })

    return per_item


# ---------------------------------------------------------------------------
# Micro-aggregation over a list of per-item dicts
# ---------------------------------------------------------------------------

def _micro_aggregate(items: list[dict]) -> dict:
    """Compute micro-aggregated P/R/F1, fp_per_100, and mean WERs over items.

    Returns a flat dict with keys:
      {cls}_tp, {cls}_fp, {cls}_fn, {cls}_precision, {cls}_recall, {cls}_f1
      fp_per_100_correct_words
      wer_vs_spoken_mean, wer_vs_target_mean
    """
    total_tp: dict[str, int] = {c: 0 for c in _CLASSES}
    total_fp: dict[str, int] = {c: 0 for c in _CLASSES}
    total_fn: dict[str, int] = {c: 0 for c in _CLASSES}
    total_correct_words = 0
    total_all_fp = 0
    wer_spoken_sum = 0.0
    wer_target_sum = 0.0

    for item in items:
        counts = item["counts"]
        for cls in _CLASSES:
            total_tp[cls] += counts[cls]["tp"]
            total_fp[cls] += counts[cls]["fp"]
            total_fn[cls] += counts[cls]["fn"]
            total_all_fp += counts[cls]["fp"]
        total_correct_words += counts["_correct_words"]
        wer_spoken_sum += item["wer_vs_spoken"]
        wer_target_sum += item["wer_vs_target"]

    n = len(items)
    result: dict = {}
    for cls in _CLASSES:
        tp = total_tp[cls]
        fp = total_fp[cls]
        fn = total_fn[cls]
        n_pred = tp + fp
        n_gold = tp + fn

        if n_gold == 0 and n_pred == 0:
            precision = recall = f1 = None
        else:
            precision = tp / n_pred if n_pred else 0.0
            recall = tp / n_gold if n_gold else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and (precision + recall) > 0
                else 0.0
            )
        result[f"{cls}_tp"] = tp
        result[f"{cls}_fp"] = fp
        result[f"{cls}_fn"] = fn
        result[f"{cls}_precision"] = precision
        result[f"{cls}_recall"] = recall
        result[f"{cls}_f1"] = f1

    if total_correct_words > 0:
        result["fp_per_100_correct_words"] = total_all_fp / total_correct_words * 100
    else:
        result["fp_per_100_correct_words"] = float(total_all_fp * 100)

    result["wer_vs_spoken_mean"] = wer_spoken_sum / n if n else 0.0
    result["wer_vs_target_mean"] = wer_target_sum / n if n else 0.0

    return result


# ---------------------------------------------------------------------------
# Agreement check vs miscue-v0.json
# ---------------------------------------------------------------------------

def _check_agreement(per_item: dict[str, list[dict]], biases: tuple[str, ...]) -> None:
    """Assert that recomputed micro P/R matches miscue-v0.json within AGREE_TOL.

    Raises ValueError on any drift — guards against logic divergence between
    masking_curve.py and run_benchmark.py.
    """
    baseline = json.loads(_BASELINE_JSON.read_text(encoding="utf-8"))
    b_results = baseline["results"]

    for bias in biases:
        if bias not in b_results:
            # Baseline wasn't run for this bias (--limit smoke test path); skip.
            continue
        agg = _micro_aggregate(per_item[bias])
        for cls in _CLASSES:
            for metric in ("precision", "recall"):
                key = f"{cls}_{metric}"
                got = agg[key]
                want = b_results[bias][cls][metric]
                # Both None → OK
                if want is None and got is None:
                    continue
                if want is None or got is None:
                    raise ValueError(
                        f"Agreement mismatch bias={bias!r} class={cls!r} "
                        f"{metric}: recomputed={got!r}, baseline={want!r}"
                    )
                if abs(got - want) > AGREE_TOL:
                    raise ValueError(
                        f"Agreement mismatch bias={bias!r} class={cls!r} "
                        f"{metric}: |{got:.10f} - {want:.10f}| = "
                        f"{abs(got - want):.2e} > {AGREE_TOL:.0e}"
                    )

        fp_got = agg["fp_per_100_correct_words"]
        fp_want = b_results[bias]["fp_per_100_correct_words"]
        if abs(fp_got - fp_want) > AGREE_TOL:
            raise ValueError(
                f"Agreement mismatch bias={bias!r} fp_per_100: "
                f"|{fp_got:.10f} - {fp_want:.10f}| = {abs(fp_got - fp_want):.2e} > {AGREE_TOL:.0e}"
            )


# ---------------------------------------------------------------------------
# Bootstrap helpers (pure functions — independently testable)
# ---------------------------------------------------------------------------

def bootstrap_micro_stats(
    per_item_counts: list[dict],
    per_item_wers_spoken: list[float],
    per_item_wers_target: list[float],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return 1-D arrays of per-resample micro statistics.

    Parameters
    ----------
    per_item_counts:
        List of match_counts() outputs (one per item).
    per_item_wers_spoken, per_item_wers_target:
        Per-item WER values (same length as per_item_counts).
    n_boot:
        Number of bootstrap resamples.
    rng:
        Seeded numpy Generator.

    Returns
    -------
    Dict mapping stat name → 1-D float array of length n_boot.
    Keys:
        {cls}_precision, {cls}_recall  (for each class in _CLASSES)
        fp_per_100
        wer_vs_spoken_mean, wer_vs_target_mean
    """
    n = len(per_item_counts)

    # Convert per-item counts to flat arrays for fast bootstrap indexing.
    # Shape: (n,) per metric
    arrays: dict[str, np.ndarray] = {}
    for cls in _CLASSES:
        arrays[f"{cls}_tp"] = np.array([c[cls]["tp"] for c in per_item_counts], dtype=float)
        arrays[f"{cls}_fp"] = np.array([c[cls]["fp"] for c in per_item_counts], dtype=float)
        arrays[f"{cls}_fn"] = np.array([c[cls]["fn"] for c in per_item_counts], dtype=float)
    arrays["correct_words"] = np.array([c["_correct_words"] for c in per_item_counts], dtype=float)
    arr_wer_spoken = np.array(per_item_wers_spoken, dtype=float)
    arr_wer_target = np.array(per_item_wers_target, dtype=float)

    # Accumulators: shape (n_boot,)
    out: dict[str, list[float]] = {k: [] for k in
        [f"{cls}_precision" for cls in _CLASSES] +
        [f"{cls}_recall" for cls in _CLASSES] +
        ["fp_per_100", "wer_vs_spoken_mean", "wer_vs_target_mean"]
    }

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)

        total_correct = arrays["correct_words"][idx].sum()
        total_all_fp = 0.0

        for cls in _CLASSES:
            tp = arrays[f"{cls}_tp"][idx].sum()
            fp = arrays[f"{cls}_fp"][idx].sum()
            fn = arrays[f"{cls}_fn"][idx].sum()
            total_all_fp += fp
            n_pred = tp + fp
            n_gold = tp + fn

            out[f"{cls}_precision"].append(tp / n_pred if n_pred > 0 else 0.0)
            out[f"{cls}_recall"].append(tp / n_gold if n_gold > 0 else 0.0)

        fp100 = total_all_fp / total_correct * 100 if total_correct > 0 else float(total_all_fp * 100)
        out["fp_per_100"].append(fp100)
        out["wer_vs_spoken_mean"].append(arr_wer_spoken[idx].mean())
        out["wer_vs_target_mean"].append(arr_wer_target[idx].mean())

    return {k: np.array(v) for k, v in out.items()}


def percentile_ci(
    boot_samples: np.ndarray,
    lo: float = 2.5,
    hi: float = 97.5,
) -> tuple[float, float]:
    """Return (lo_pct, hi_pct) percentile CI from a bootstrap sample array."""
    return float(np.percentile(boot_samples, lo)), float(np.percentile(boot_samples, hi))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_BIAS_LABELS = {"none": "None", "prompt": "Prompt", "strong": "Strong"}
_CLASS_COLORS = {
    "substitution": "#e41a1c",
    "omission": "#377eb8",
    "insertion": "#4daf4a",
    "self_correction": "#984ea3",
    "hesitation": "#ff7f00",
}


def _make_figure(
    point_estimates: dict[str, dict],  # bias -> {cls_precision, cls_recall, ...}
    boot_stats: dict[str, dict[str, np.ndarray]],  # bias -> boot arrays
    n_items: int,
    out_path: Path,
) -> None:
    """Generate the 2-panel masking curve figure."""
    biases = list(_ALL_BIASES)
    x = np.arange(len(biases))
    bias_tick_labels = [_BIAS_LABELS[b] for b in biases]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Masking Curve — n={n_items} TTS clips  (synthetic benchmark; TTS ≠ child speech)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    # --- Panel A: per-class recall vs bias ---
    offsets = np.linspace(-0.15, 0.15, len(_CLASSES))
    for i, cls in enumerate(_CLASSES):
        recalls = [point_estimates[b][f"{cls}_recall"] for b in biases]
        recalls_plot = [r if r is not None else 0.0 for r in recalls]
        ci_lo = [percentile_ci(boot_stats[b][f"{cls}_recall"])[0] for b in biases]
        ci_hi = [percentile_ci(boot_stats[b][f"{cls}_recall"])[1] for b in biases]
        yerr_lo = [max(0.0, recalls_plot[j] - ci_lo[j]) for j in range(len(biases))]
        yerr_hi = [max(0.0, ci_hi[j] - recalls_plot[j]) for j in range(len(biases))]
        ax_a.errorbar(
            x + offsets[i],
            recalls_plot,
            yerr=[yerr_lo, yerr_hi],
            fmt="o-",
            color=_CLASS_COLORS[cls],
            label=cls.replace("_", " ").title(),
            capsize=3,
            linewidth=1.5,
            markersize=5,
        )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(bias_tick_labels)
    ax_a.set_xlabel("Bias setting")
    ax_a.set_ylabel("Recall (micro, ±95% CI)")
    ax_a.set_title("Panel A — Per-class Recall vs Bias")
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax_a.grid(axis="y", alpha=0.3)

    # --- Panel B: substitution precision, fp_per_100, and both WERs vs bias ---
    ax_b2 = ax_b.twinx()

    # Left axis: substitution precision + both WERs
    sub_prec = [point_estimates[b]["substitution_precision"] for b in biases]
    sub_prec_plot = [v if v is not None else 0.0 for v in sub_prec]
    ci_lo_sp = [percentile_ci(boot_stats[b]["substitution_precision"])[0] for b in biases]
    ci_hi_sp = [percentile_ci(boot_stats[b]["substitution_precision"])[1] for b in biases]
    err_sp = [
        [max(0.0, sub_prec_plot[j] - ci_lo_sp[j]) for j in range(len(biases))],
        [max(0.0, ci_hi_sp[j] - sub_prec_plot[j]) for j in range(len(biases))],
    ]
    ax_b.errorbar(x, sub_prec_plot, yerr=err_sp, fmt="s--", color="#e41a1c",
                  label="Sub precision", capsize=3, linewidth=1.5, markersize=5)

    wer_spoken = [point_estimates[b]["wer_vs_spoken_mean"] for b in biases]
    ci_lo_ws = [percentile_ci(boot_stats[b]["wer_vs_spoken_mean"])[0] for b in biases]
    ci_hi_ws = [percentile_ci(boot_stats[b]["wer_vs_spoken_mean"])[1] for b in biases]
    err_ws = [
        [max(0.0, wer_spoken[j] - ci_lo_ws[j]) for j in range(len(biases))],
        [max(0.0, ci_hi_ws[j] - wer_spoken[j]) for j in range(len(biases))],
    ]
    ax_b.errorbar(x, wer_spoken, yerr=err_ws, fmt="^-", color="#984ea3",
                  label="WER vs spoken", capsize=3, linewidth=1.5, markersize=5)

    wer_target = [point_estimates[b]["wer_vs_target_mean"] for b in biases]
    ci_lo_wt = [percentile_ci(boot_stats[b]["wer_vs_target_mean"])[0] for b in biases]
    ci_hi_wt = [percentile_ci(boot_stats[b]["wer_vs_target_mean"])[1] for b in biases]
    err_wt = [
        [max(0.0, wer_target[j] - ci_lo_wt[j]) for j in range(len(biases))],
        [max(0.0, ci_hi_wt[j] - wer_target[j]) for j in range(len(biases))],
    ]
    ax_b.errorbar(x, wer_target, yerr=err_wt, fmt="D-", color="#4daf4a",
                  label="WER vs target", capsize=3, linewidth=1.5, markersize=5)

    # Right axis: fp_per_100
    fp100 = [point_estimates[b]["fp_per_100_correct_words"] for b in biases]
    ci_lo_fp = [percentile_ci(boot_stats[b]["fp_per_100"])[0] for b in biases]
    ci_hi_fp = [percentile_ci(boot_stats[b]["fp_per_100"])[1] for b in biases]
    err_fp = [
        [max(0.0, fp100[j] - ci_lo_fp[j]) for j in range(len(biases))],
        [max(0.0, ci_hi_fp[j] - fp100[j]) for j in range(len(biases))],
    ]
    ax_b2.errorbar(x + 0.05, fp100, yerr=err_fp, fmt="x:", color="#ff7f00",
                   label="FP per 100 correct\n(right axis)", capsize=3, linewidth=1.5,
                   markersize=7)
    ax_b2.set_ylabel("FP per 100 correct words", color="#ff7f00")
    ax_b2.tick_params(axis="y", labelcolor="#ff7f00")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(bias_tick_labels)
    ax_b.set_xlabel("Bias setting")
    ax_b.set_ylabel("Precision / WER (±95% CI)")
    ax_b.set_title("Panel B — Substitution Precision, WERs & FP Rate vs Bias")
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.grid(axis="y", alpha=0.3)

    # Combine legends from both axes on ax_b
    lines_b, labels_b = ax_b.get_legend_handles_labels()
    lines_b2, labels_b2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(lines_b + lines_b2, labels_b + labels_b2, loc="upper right",
                fontsize=8, framealpha=0.8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Masking curve: per-class recall/precision vs bias with bootstrap CIs."
    )
    parser.add_argument("--seed", type=int, default=1337, help="Bootstrap RNG seed (default 1337)")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples (default 1000)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Process only first N items (smoke test)")
    parser.add_argument("--backend", default="faster-whisper-small",
                        help="ASR backend (default: faster-whisper-small)")
    args = parser.parse_args(argv)

    print(f"=== Masking Curve  seed={args.seed}  n_boot={args.n_boot} ===", file=sys.stderr)

    # --- Load gold rows ---
    rows = _load_gold_rows(args.limit)
    if not rows:
        print("ERROR: no rows to process", file=sys.stderr)
        sys.exit(1)
    n_items = len(rows)
    print(f"Loaded {n_items} items from {_GOLD_JSONL}", file=sys.stderr)

    # --- Per-item statistics (cache-fast) ---
    per_item = _compute_per_item_stats(rows, _ALL_BIASES, args.backend)

    # --- Agreement check vs miscue-v0.json ---
    if args.limit is None:
        print("\nChecking agreement with miscue-v0.json ...", file=sys.stderr)
        _check_agreement(per_item, _ALL_BIASES)
        print("  PASSED — all P/R values agree within 1e-9", file=sys.stderr)
        agreement_passed = True
    else:
        print("  Skipping agreement check (--limit active)", file=sys.stderr)
        agreement_passed = None  # N/A

    # --- Point estimates ---
    point_estimates: dict[str, dict] = {}
    for bias in _ALL_BIASES:
        point_estimates[bias] = _micro_aggregate(per_item[bias])

    # --- Bootstrap ---
    print(f"\nBootstrapping ({args.n_boot} resamples, seed={args.seed}) ...", file=sys.stderr)
    rng = np.random.default_rng(args.seed)
    boot_stats: dict[str, dict[str, np.ndarray]] = {}
    for bias in _ALL_BIASES:
        items = per_item[bias]
        counts_list = [item["counts"] for item in items]
        wers_spoken = [item["wer_vs_spoken"] for item in items]
        wers_target = [item["wer_vs_target"] for item in items]
        boot_stats[bias] = bootstrap_micro_stats(
            counts_list, wers_spoken, wers_target, args.n_boot, rng
        )
    print("  Bootstrap done.", file=sys.stderr)

    # --- Build output JSON ---
    results_out: dict = {}
    for bias in _ALL_BIASES:
        pe = point_estimates[bias]
        bs = boot_stats[bias]
        bias_out: dict = {}

        for cls in _CLASSES:
            prec = pe[f"{cls}_precision"]
            rec = pe[f"{cls}_recall"]
            ci_prec = percentile_ci(bs[f"{cls}_precision"])
            ci_rec = percentile_ci(bs[f"{cls}_recall"])
            bias_out[cls] = {
                "precision": prec,
                "recall": rec,
                "f1": pe[f"{cls}_f1"],
                "tp": pe[f"{cls}_tp"],
                "fp": pe[f"{cls}_fp"],
                "fn": pe[f"{cls}_fn"],
                "ci_precision": list(ci_prec),
                "ci_recall": list(ci_rec),
            }

        fp100 = pe["fp_per_100_correct_words"]
        ci_fp100 = percentile_ci(bs["fp_per_100"])
        bias_out["fp_per_100_correct_words"] = fp100
        bias_out["ci_fp_per_100"] = list(ci_fp100)

        wer_sp = pe["wer_vs_spoken_mean"]
        ci_wer_sp = percentile_ci(bs["wer_vs_spoken_mean"])
        bias_out["wer_vs_spoken_mean"] = wer_sp
        bias_out["ci_wer_vs_spoken"] = list(ci_wer_sp)

        wer_tgt = pe["wer_vs_target_mean"]
        ci_wer_tgt = percentile_ci(bs["wer_vs_target_mean"])
        bias_out["wer_vs_target_mean"] = wer_tgt
        bias_out["ci_wer_vs_target"] = list(ci_wer_tgt)

        results_out[bias] = bias_out

    output = {
        "metadata": {
            "backend": args.backend,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "n_items": n_items,
            "git_commit": _git_head(),
            "date": datetime.date.today().isoformat(),
            "biases_run": list(_ALL_BIASES),
            "aggregation": "micro",
            "normalization": "same as miscue detector (_normalize: strip+casefold, edge-punct removed)",
            "agreement_check_vs_miscue_v0": "passed" if agreement_passed else ("n/a (limit active)" if agreement_passed is None else "failed"),
        },
        "results": results_out,
    }

    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    _OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults JSON written to {_OUT_JSON}", file=sys.stderr)

    # --- Figure ---
    _make_figure(point_estimates, boot_stats, n_items, _OUT_PNG)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("MASKING CURVE SUMMARY")
    print("=" * 70)
    print(f"{'Bias':<8}  {'Class':<16}  {'Recall':>7}  {'CI 95%':>18}  {'Precision':>10}  {'CI 95%':>18}")
    print("-" * 85)
    for bias in _ALL_BIASES:
        pe = point_estimates[bias]
        bs = boot_stats[bias]
        for cls in _CLASSES:
            rec = pe[f"{cls}_recall"]
            prec = pe[f"{cls}_precision"]
            ci_r = percentile_ci(bs[f"{cls}_recall"])
            ci_p = percentile_ci(bs[f"{cls}_precision"])
            rec_s = f"{rec:.3f}" if rec is not None else "None"
            prec_s = f"{prec:.3f}" if prec is not None else "None"
            ci_r_s = f"[{ci_r[0]:.3f}, {ci_r[1]:.3f}]"
            ci_p_s = f"[{ci_p[0]:.3f}, {ci_p[1]:.3f}]"
            print(f"{bias:<8}  {cls:<16}  {rec_s:>7}  {ci_r_s:>18}  {prec_s:>10}  {ci_p_s:>18}")
        fp100 = pe["fp_per_100_correct_words"]
        ci_fp = percentile_ci(bs["fp_per_100"])
        wer_sp = pe["wer_vs_spoken_mean"]
        wer_tgt = pe["wer_vs_target_mean"]
        ci_ws = percentile_ci(bs["wer_vs_spoken_mean"])
        ci_wt = percentile_ci(bs["wer_vs_target_mean"])
        print(f"{bias:<8}  {'fp_per_100':>16}  {fp100:>7.3f}  [{ci_fp[0]:.3f}, {ci_fp[1]:.3f}]")
        print(f"{bias:<8}  {'wer_vs_spoken':>16}  {wer_sp:>7.3f}  [{ci_ws[0]:.3f}, {ci_ws[1]:.3f}]")
        print(f"{bias:<8}  {'wer_vs_target':>16}  {wer_tgt:>7.3f}  [{ci_wt[0]:.3f}, {ci_wt[1]:.3f}]")
        print()
    print("=" * 70)


if __name__ == "__main__":
    main()
