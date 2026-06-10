"""``readcoach-bench score`` — the BYO-ASR scoring path (T2.5).

This is the artifact external consumers actually run: bring your own ASR, emit a
``hypotheses.jsonl``, and score it against the committed ReadCoach gold with the
SAME miscue detector and metrics that produced the published faster-whisper
baseline.

Auditable in five minutes
-------------------------
A scoring run imports **only jiwer + the Python standard library** (transitively
through :mod:`readcoach.miscue` / :mod:`readcoach.asr`, which pull in jiwer and
the ``Word`` / ``AsrResult`` dataclasses).  It NEVER imports a model or a service
client.  Specifically, after a real scoring call, none of ``faster_whisper``,
``weave``, ``anthropic``, ``google.genai``, or ``wandb`` appear in
``sys.modules``.  The readcoach package itself is imported; its heavyweight model
dependencies are not.  ``tests/test_bench_cli.py::test_import_audit`` enforces
this, and ``docs/BENCHMARK.md`` documents the one-line command to verify it.

The command runs from a fresh clone + ``python3 scripts/fetch_benchmark.py`` with
zero credentials.

Output
------
A plain-text, column-aligned table.  Rows are grouped:

  ALIGNMENT                  substitution / omission / insertion   (P / R / F1)
  TRANSCRIPT-STYLE-SENSITIVE self_correction / hesitation          (P / R / F1)
  fp_per_100_correct_words

Columns are the committed faster-whisper-small baseline (bias=none) plus one per
consumer hypothesis set.  With two or more consumer sets the table IS the
consumer's own masking-curve table (one column per ASR / decoding configuration)
— stated in a footer.

The transcript-style group carries a one-line caveat: self_correction and
hesitation test the ASR's transcript STYLE (does it preserve disfluencies?) as
much as the detector, so they are reported separately from the alignment classes.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Class grouping (matches the published taxonomy / docs/BENCHMARK.md)
# ---------------------------------------------------------------------------

# Alignment classes: derived purely from the edit-distance alignment of the ASR
# hypothesis against the target text.
_ALIGNMENT_CLASSES: tuple[str, ...] = ("substitution", "omission", "insertion")

# Transcript-style-sensitive classes: these depend on whether the ASR PRESERVES
# disfluencies in its transcript (most production ASR normalizes them away), so
# they measure transcript style as much as detector quality — reported separately.
_TRANSCRIPT_STYLE_CLASSES: tuple[str, ...] = ("self_correction", "hesitation")

# Full class order (alignment first, then transcript-style) — used for ordered
# iteration over the per-class metric rows.
_ALL_CLASSES: tuple[str, ...] = _ALIGNMENT_CLASSES + _TRANSCRIPT_STYLE_CLASSES

# The committed faster-whisper baseline column reports bias=none (the raw-acoustic
# measurement — no target-text leakage into the ASR), which is the published
# headline in evals/results/v0.json -> metrics.miscue.
_BASELINE_LABEL = "faster-whisper-small\n(bias=none)"

_TRANSCRIPT_STYLE_CAVEAT = (
    "self_correction + hesitation test the ASR's transcript STYLE "
    "(disfluency preservation) as much as the detector — see docs/BENCHMARK.md."
)


# ---------------------------------------------------------------------------
# Errors — every failure mode is a distinct, named exception (fail-loud)
# ---------------------------------------------------------------------------

class HypothesisError(ValueError):
    """Base class for hypotheses.jsonl parsing / coverage errors."""


class DuplicateUttError(HypothesisError):
    """A utt_id appears more than once in a single hypotheses file."""


class UnknownUttError(HypothesisError):
    """A hypotheses file references a utt_id that is not in the gold set."""


class PartialCoverageError(HypothesisError):
    """A hypotheses file covers a strict subset of gold and --allow-partial is off."""


# ---------------------------------------------------------------------------
# Parsing — gold and hypotheses (stdlib json only; fail loud on every defect)
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of dicts.

    Fail-loud: a malformed line raises ``ValueError`` naming the file and line
    number.  Blank lines are skipped (trailing newline tolerance only).
    """
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise HypothesisError(
                    f"{path}: line {lineno}: invalid JSON — {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"{path}: line {lineno}: expected a JSON object, got "
                    f"{type(obj).__name__}"
                )
            rows.append(obj)
    return rows


@dataclass(frozen=True)
class GoldItem:
    utt_id: str
    target_text: str
    n_target_words: int
    gold_miscues: list  # list[readcoach.miscue.Miscue]


def load_gold(path: Path) -> dict[str, GoldItem]:
    """Load gold.jsonl into ``{utt_id: GoldItem}``.

    Each gold entry's ``{type, target_word, said_word, index}`` becomes a
    ``Miscue``.  The ``render`` field (None | "filler" | "silence") is a gold
    annotation only — it distinguishes hesitation subtypes for documentation and
    does not affect scoring (both subtypes are real targets).
    """
    from readcoach.miscue import Miscue  # noqa: PLC0415

    rows = _load_jsonl(path)
    if not rows:
        raise ValueError(f"{path}: gold set is empty")

    out: dict[str, GoldItem] = {}
    for row in rows:
        utt_id = row["utt_id"]
        if utt_id in out:
            raise DuplicateUttError(f"{path}: duplicate utt_id {utt_id!r} in gold")
        target_text = row["target_text"]
        miscues = [
            Miscue(
                type=g["type"],
                target_word=g["target_word"],
                said_word=g["said_word"],
                index=g["index"],
            )
            for g in row["gold"]
        ]
        out[utt_id] = GoldItem(
            utt_id=utt_id,
            target_text=target_text,
            n_target_words=len(target_text.split()),
            gold_miscues=miscues,
        )
    return out


def load_hypotheses(path: Path, gold: dict[str, GoldItem]) -> dict[str, list]:
    """Load a hypotheses.jsonl into ``{utt_id: list[Word]}``.

    Schema per line: ``{"utt_id": str, "words": [{"text": str, "start"?: float,
    "end"?: float, "confidence"?: float}, ...]}``.

      * ``text`` is required (KeyError -> ValueError if missing).
      * ``start`` / ``end`` default to ``None`` (timing-based silence-hesitation
        detection is then inactive for that word).
      * ``confidence`` defaults to ``1.0`` (downstream-only; does not affect
        today's scoring).

    Fail-loud coverage contract:
      * duplicate utt_id within the file        -> DuplicateUttError
      * utt_id not present in gold              -> UnknownUttError

    Returns the per-utt Word lists; coverage-vs-gold (partial / missing) is
    validated by the caller via :func:`check_coverage` so the ``--allow-partial``
    policy lives in one place.
    """
    from readcoach.asr import Word  # noqa: PLC0415

    rows = _load_jsonl(path)
    out: dict[str, list] = {}
    for i, row in enumerate(rows):
        if "utt_id" not in row:
            raise HypothesisError(f"{path}: row {i} has no 'utt_id' field")
        utt_id = row["utt_id"]
        if utt_id in out:
            raise DuplicateUttError(
                f"{path}: duplicate utt_id {utt_id!r} in hypotheses file"
            )
        if utt_id not in gold:
            raise UnknownUttError(
                f"{path}: utt_id {utt_id!r} is not in the gold set "
                f"({len(gold)} gold items) — cannot score it"
            )
        if "words" not in row:
            raise HypothesisError(
                f"{path}: utt_id {utt_id!r} has no 'words' field"
            )
        words: list = []
        for j, w in enumerate(row["words"]):
            if "text" not in w:
                raise HypothesisError(
                    f"{path}: utt_id {utt_id!r} word {j} has no required 'text' field"
                )
            words.append(
                Word(
                    text=w["text"],
                    start=w.get("start"),
                    end=w.get("end"),
                    confidence=w.get("confidence", 1.0),
                )
            )
        out[utt_id] = words
    return out


def check_coverage(
    hyp: dict[str, list],
    gold: dict[str, GoldItem],
    path: Path,
    allow_partial: bool,
) -> None:
    """Enforce the coverage contract; ``allow_partial`` controls subset behaviour.

    Full coverage is required by default.  A hypotheses file covering a strict
    subset of gold raises :class:`PartialCoverageError` unless ``allow_partial``
    is set, in which case the caller prints ``n_covered`` prominently.  Silent
    partial scoring is forbidden either way.
    """
    missing = [u for u in gold if u not in hyp]
    if missing and not allow_partial:
        raise PartialCoverageError(
            f"{path}: covers {len(hyp)}/{len(gold)} gold items — "
            f"{len(missing)} missing (e.g. {', '.join(sorted(missing)[:3])}). "
            "Pass --allow-partial to score the covered subset (n_covered is "
            "printed prominently); silent partial scoring is forbidden."
        )


# ---------------------------------------------------------------------------
# Scoring — micro-aggregate one hypothesis set against gold
# ---------------------------------------------------------------------------

def _empty_accum() -> dict:
    out: dict = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in _ALL_CLASSES}
    out["_correct_words"] = 0
    out["_total_fp"] = 0
    return out


def _finalize(accum: dict) -> dict:
    """Compute per-class P/R/F1 + fp_per_100 from accumulated micro-sums.

    Mirrors :func:`readcoach.miscue.score` exactly: a class absent from both
    gold and predicted reports ``None`` (never a fabricated 1.0).
    """
    result: dict = {}
    for cls in _ALL_CLASSES:
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
            if (precision + recall) > 0:
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


def score_hypotheses(
    hyp: dict[str, list],
    gold: dict[str, GoldItem],
) -> dict:
    """Run ``detect`` + ``match_counts`` per covered utt and micro-aggregate.

    Only utts present in ``hyp`` are scored (coverage is validated upstream).
    Returns ``{"n_covered": int, "metrics": <finalized dict>}``.

    This is the function exercised by the import-audit test: the entire call
    graph (detect, match_counts) stays inside jiwer + stdlib + the readcoach
    dataclasses — no model or service client is imported.
    """
    from readcoach.miscue import detect, match_counts  # noqa: PLC0415

    accum = _empty_accum()
    # Deterministic order so any per-item iteration is reproducible.
    for utt_id in sorted(hyp):
        words = hyp[utt_id]
        item = gold[utt_id]
        predicted = detect(words, item.target_text)
        counts = match_counts(predicted, item.gold_miscues, item.n_target_words)
        for cls in _ALL_CLASSES:
            accum[cls]["tp"] += counts[cls]["tp"]
            accum[cls]["fp"] += counts[cls]["fp"]
            accum[cls]["fn"] += counts[cls]["fn"]
            accum["_total_fp"] += counts[cls]["fp"]
        accum["_correct_words"] += counts["_correct_words"]

    return {"n_covered": len(hyp), "metrics": _finalize(accum)}


# ---------------------------------------------------------------------------
# Baseline column (committed faster-whisper-small, bias=none)
# ---------------------------------------------------------------------------

def load_baseline(path: Path) -> dict:
    """Return the bias=none per-class metrics from a committed results JSON.

    The published headline (``metrics.miscue`` in evals/results/v0.json) is the
    bias=none, micro-aggregated faster-whisper-small result.  We read that block
    so the consumer's column sits directly beside the exact published numbers.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        miscue = data["metrics"]["miscue"]
    except KeyError as exc:
        raise ValueError(
            f"{path}: expected metrics.miscue block (bias=none baseline); "
            f"missing key {exc}"
        ) from exc
    return miscue


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _fmt(value) -> str:
    """Format a metric for the table: 3 decimals, or '  —  ' for None."""
    if value is None:
        return "  —  "
    return f"{value:.3f}"


@dataclass
class ScoredSet:
    label: str
    n_covered: int
    n_gold: int
    metrics: dict


def render_table(
    baseline: dict,
    scored: list[ScoredSet],
) -> str:
    """Render the aligned plain-text scoring table.

    Columns: the committed faster-whisper baseline (bias=none) then one per
    consumer set.  Rows: ALIGNMENT group (sub/om/ins) then TRANSCRIPT-STYLE group
    (self_correction/hesitation), each as three P/R/F1 rows, then fp_per_100.
    With >=2 consumer sets, a footer states the table IS their masking curve.
    """
    # Column headers (baseline first, then each consumer set).  Multi-line headers
    # are supported (the baseline header has a "(bias=none)" subline).
    headers = [_BASELINE_LABEL] + [s.label for s in scored]
    header_lines = [h.split("\n") for h in headers]
    header_height = max(len(hl) for hl in header_lines)
    # Pad each header to header_height lines (top blank padding, bottom-aligned).
    header_lines = [[""] * (header_height - len(hl)) + hl for hl in header_lines]

    # Build each group's body rows as (label, [value-per-column]); the first
    # column is the baseline, then one per consumer set.
    def metric_rows(classes: tuple[str, ...]) -> list[tuple[str, list[str]]]:
        out: list[tuple[str, list[str]]] = []
        for cls in classes:
            base_cell = baseline.get(cls)
            for field, field_label in (("precision", "P"), ("recall", "R"), ("f1", "F1")):
                label = f"  {cls:<16s} {field_label:<2s}"
                cols = [_fmt(None if base_cell is None else base_cell.get(field))]
                for s in scored:
                    cell = s.metrics.get(cls)
                    cols.append(_fmt(None if cell is None else cell.get(field)))
                out.append((label, cols))
        return out

    align_rows = metric_rows(_ALIGNMENT_CLASSES)
    style_rows = metric_rows(_TRANSCRIPT_STYLE_CLASSES)

    # fp_per_100 row
    fp_cols = [_fmt(baseline.get("fp_per_100_correct_words"))]
    for s in scored:
        fp_cols.append(_fmt(s.metrics.get("fp_per_100_correct_words")))
    fp_row = ("  fp_per_100_correct_words", fp_cols)

    # Column widths: each data column is as wide as the widest of its header lines,
    # its body values, and a sensible minimum.
    n_cols = len(headers)
    all_value_rows = align_rows + style_rows + [fp_row]
    label_width = max(len(lbl) for lbl, _ in all_value_rows)
    label_width = max(label_width, len("  fp_per_100_correct_words"))

    col_widths: list[int] = []
    for c in range(n_cols):
        # Widest header line for this column ...
        w = max((len(line) for line in header_lines[c]), default=0)
        # ... and widest body value in this column.
        for _lbl, cols in all_value_rows:
            w = max(w, len(cols[c]))
        col_widths.append(max(w, 6))

    def fmt_data_line(label: str, cols: list[str]) -> str:
        cells = [label.ljust(label_width)]
        for c in range(n_cols):
            cells.append(cols[c].rjust(col_widths[c]))
        return "   ".join(cells)

    lines: list[str] = []
    # Header block.
    for hrow in range(header_height):
        cells = [" " * label_width]
        for c in range(n_cols):
            cells.append(header_lines[c][hrow].rjust(col_widths[c]))
        lines.append("   ".join(cells))

    total_width = len(lines[0])
    lines.append("-" * total_width)

    lines.append("ALIGNMENT (edit-distance of hypothesis vs target)".ljust(label_width))
    for lbl, cols in align_rows:
        lines.append(fmt_data_line(lbl, cols))

    lines.append("")
    lines.append(
        "TRANSCRIPT-STYLE-SENSITIVE (reported separately)".ljust(label_width)
    )
    for lbl, cols in style_rows:
        lines.append(fmt_data_line(lbl, cols))
    # The disfluency-preservation caveat, one line, under the group.
    lines.append(f"  ⚠ {_TRANSCRIPT_STYLE_CAVEAT}")

    lines.append("")
    lines.append(fmt_data_line(*fp_row))

    lines.append("-" * total_width)

    # Coverage line per consumer set (n_covered is always shown — prominent when
    # partial).
    for s in scored:
        flag = "" if s.n_covered == s.n_gold else "   ⚠ PARTIAL"
        lines.append(
            f"  {s.label.splitlines()[0]}: n_covered = {s.n_covered}/{s.n_gold}{flag}"
        )

    # Masking-curve footer when >= 2 consumer sets.
    if len(scored) >= 2:
        lines.append("")
        lines.append(
            "  With >=2 hypothesis sets, the consumer columns above ARE your "
            "masking-curve table"
        )
        lines.append(
            "  (one column per ASR / decoding configuration — compare "
            "fp_per_100 and per-class recall across them)."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON output (machine-readable mirror of the table)
# ---------------------------------------------------------------------------

def build_json_output(
    baseline: dict,
    baseline_path: Path,
    scored: list[ScoredSet],
    n_gold: int,
) -> dict:
    return {
        "schema": "readcoach-bench/score/v1",
        "n_gold": n_gold,
        "class_groups": {
            "alignment": list(_ALIGNMENT_CLASSES),
            "transcript_style_sensitive": list(_TRANSCRIPT_STYLE_CLASSES),
        },
        "transcript_style_caveat": _TRANSCRIPT_STYLE_CAVEAT,
        "baseline": {
            "label": "faster-whisper-small",
            "bias": "none",
            "source": str(baseline_path),
            "metrics": baseline,
        },
        "hypothesis_sets": [
            {
                "label": s.label,
                "n_covered": s.n_covered,
                "metrics": s.metrics,
            }
            for s in scored
        ],
        "is_masking_curve": len(scored) >= 2,
    }


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def _parse_hypotheses_arg(spec: str) -> tuple[str, Path]:
    """Parse a ``--hypotheses`` value: ``NAME=path`` or bare ``path``.

    The ``NAME=`` prefix is optional; without it the label defaults to the file's
    stem.  A path containing ``=`` (rare) can be disambiguated with the prefix.
    """
    if "=" in spec:
        name, _, raw_path = spec.partition("=")
        name = name.strip()
        path = Path(raw_path.strip())
        if not name:
            raise ValueError(
                f"--hypotheses {spec!r}: empty NAME before '=' "
                "(use NAME=path or a bare path)"
            )
        return name, path
    path = Path(spec)
    return path.stem, path


def _resolve_default(path_str: str) -> Path:
    """Resolve a default path relative to the project root (parents[2] of this
    file: src/readcoach/bench_cli.py -> project root)."""
    return Path(__file__).resolve().parents[2] / path_str


def cmd_score(args: argparse.Namespace) -> int:
    gold_path = Path(args.gold) if args.gold else _resolve_default("data/benchmark/gold.jsonl")
    baseline_path = (
        Path(args.baseline) if args.baseline else _resolve_default("evals/results/v0.json")
    )

    if not gold_path.exists():
        print(
            f"ERROR: gold file not found: {gold_path}\n"
            "Run `python3 scripts/fetch_benchmark.py` first (no credentials needed).",
            file=sys.stderr,
        )
        return 2
    if not baseline_path.exists():
        print(f"ERROR: baseline file not found: {baseline_path}", file=sys.stderr)
        return 2

    gold = load_gold(gold_path)
    baseline = load_baseline(baseline_path)
    n_gold = len(gold)

    # Parse each --hypotheses spec; labels must be unique across sets.
    specs = [_parse_hypotheses_arg(s) for s in args.hypotheses]
    seen_labels: set[str] = set()
    scored: list[ScoredSet] = []
    for label, hpath in specs:
        if label in seen_labels:
            print(
                f"ERROR: duplicate hypothesis-set label {label!r} "
                "(use NAME=path to disambiguate)",
                file=sys.stderr,
            )
            return 2
        seen_labels.add(label)
        if not hpath.exists():
            print(f"ERROR: hypotheses file not found: {hpath}", file=sys.stderr)
            return 2
        # HypothesisError (duplicate / unknown / partial / schema) is an expected,
        # named failure: surface it as a clean message + exit 1, not a traceback.
        try:
            hyp = load_hypotheses(hpath, gold)
            check_coverage(hyp, gold, hpath, allow_partial=args.allow_partial)
        except HypothesisError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        result = score_hypotheses(hyp, gold)
        scored.append(
            ScoredSet(
                label=label,
                n_covered=result["n_covered"],
                n_gold=n_gold,
                metrics=result["metrics"],
            )
        )

    table = render_table(baseline, scored)
    print(table)

    if args.json:
        out = build_json_output(baseline, baseline_path, scored, n_gold)
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nJSON written to {out_path}", file=sys.stderr)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="readcoach-bench",
        description=(
            "Bring-your-own-ASR scoring against the ReadCoach miscue benchmark. "
            "Imports only jiwer + stdlib — no models, no service clients."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser(
        "score",
        help="Score one or more hypotheses.jsonl files against the committed gold.",
        description=(
            "Score your ASR's hypotheses against the ReadCoach gold and print a "
            "per-class P/R/F1 + fp_per_100 table beside the committed "
            "faster-whisper-small baseline.  Two or more --hypotheses sets give "
            "you your own masking-curve table."
        ),
    )
    score.add_argument(
        "--hypotheses",
        action="append",
        required=True,
        metavar="NAME=path.jsonl",
        help=(
            "A hypotheses.jsonl to score.  Repeat for multiple sets.  The 'NAME=' "
            "prefix is optional (defaults to the filename stem) and labels the "
            "column."
        ),
    )
    score.add_argument(
        "--gold",
        default=None,
        metavar="PATH",
        help="Gold JSONL (default: data/benchmark/gold.jsonl).",
    )
    score.add_argument(
        "--baseline",
        default=None,
        metavar="PATH",
        help="Committed baseline results JSON (default: evals/results/v0.json).",
    )
    score.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Permit a hypotheses file that covers a strict subset of gold.  "
            "n_covered is printed prominently.  Without this flag, partial "
            "coverage is an ERROR (silent partial scoring is forbidden)."
        ),
    )
    score.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="Also write the same data machine-readably to PATH.",
    )
    score.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
