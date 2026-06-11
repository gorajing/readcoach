"""T6.2 — Pure, tested dashboard state computation.

build_dashboard_state(results_dir, db_path|None) -> dict

All numbers come from committed artifacts.  Missing file → section
{"available": False, "reason": "..."}.  No fake numbers, no Streamlit import.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict | None:
    """Return parsed JSON or None on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _section_detector(results_dir: Path) -> dict:
    """miscue-v0.json → per-class P/R/F1 × bias + fp_per_100 headline."""
    path = results_dir / "miscue-v0.json"
    data = _load_json(path)
    if data is None:
        return _unavailable(f"miscue-v0.json not found or unparseable at {path}")

    results = data.get("results", {})
    biases = list(results.keys())

    per_class_per_bias: dict[str, dict] = {}
    fp_per_100_by_bias: dict[str, float] = {}
    for bias, bdata in results.items():
        fp_per_100_by_bias[bias] = bdata.get("fp_per_100_correct_words", float("nan"))
        for cls, cdata in bdata.items():
            if cls == "fp_per_100_correct_words":
                continue
            if cls not in per_class_per_bias:
                per_class_per_bias[cls] = {}
            per_class_per_bias[cls][bias] = {
                "precision": cdata.get("precision"),
                "recall": cdata.get("recall"),
                "f1": cdata.get("f1"),
            }

    # Headline: substitution recall @ none-bias, fp_per_100 @ strong bias
    sub_recall_none = (
        per_class_per_bias.get("substitution", {})
        .get("none", {})
        .get("recall")
    )
    fp_per_100_strong = fp_per_100_by_bias.get("strong")

    return {
        "available": True,
        "biases": biases,
        "per_class_per_bias": per_class_per_bias,
        "fp_per_100_by_bias": fp_per_100_by_bias,
        # headlines
        "sub_recall_none": sub_recall_none,
        "fp_per_100_strong": fp_per_100_strong,
        "metadata": data.get("metadata", {}),
        "repro": "uv run python -m readcoach.bench_cli --no-audio --results-dir evals/results",
    }


def _section_masking(results_dir: Path) -> dict:
    """masking_curve.json → curve points + CIs per bias."""
    path = results_dir / "masking_curve.json"
    data = _load_json(path)
    if data is None:
        return _unavailable(f"masking_curve.json not found or unparseable at {path}")

    results = data.get("results", {})
    curve_points: dict[str, dict] = {}
    for bias, bdata in results.items():
        curve_points[bias] = {
            "fp_per_100_correct_words": bdata.get("fp_per_100_correct_words"),
            "ci_fp_per_100": bdata.get("ci_fp_per_100"),
            "wer_vs_spoken_mean": bdata.get("wer_vs_spoken_mean"),
            "ci_wer_vs_spoken": bdata.get("ci_wer_vs_spoken"),
        }

    png_path = results_dir / "masking_curve.png"
    return {
        "available": True,
        "curve_points": curve_points,
        "png_path": str(png_path) if png_path.exists() else None,
        "metadata": data.get("metadata", {}),
        "repro": "uv run python scripts/masking_curve.py",
    }


def _section_learner(results_dir: Path) -> dict:
    """break_even.json + bkt_recovery.json → BKT headlines."""
    be_path = results_dir / "break_even.json"
    bkt_path = results_dir / "bkt_recovery.json"

    be_data = _load_json(be_path)
    bkt_data = _load_json(bkt_path)

    if be_data is None and bkt_data is None:
        return _unavailable(
            f"both break_even.json and bkt_recovery.json missing at {results_dir}"
        )

    section: dict[str, Any] = {"available": True}

    # break_even
    if be_data is not None:
        section["break_even_a"] = be_data.get("break_even_a")
        section["a_eff_anchors"] = be_data.get("a_eff_anchors", [])
        section["break_even_grid"] = be_data.get("grid", [])
        section["break_even_metadata"] = be_data.get("metadata", {})
        section["break_even_png"] = str(results_dir / "break_even.png") if (results_dir / "break_even.png").exists() else None
        section["repro_break_even"] = "uv run python scripts/break_even.py"
    else:
        section["break_even_available"] = False
        section["break_even_reason"] = f"break_even.json not found at {be_path}"

    # bkt_recovery
    if bkt_data is not None:
        section["bkt_weak_spot_note"] = bkt_data.get("weak_spot_note", "")
        section["bkt_brier_score"] = bkt_data.get("calibration", {}).get("brier_score")
        section["bkt_regimes"] = bkt_data.get("regimes", {})
        section["bkt_mean_recovery_error"] = bkt_data.get("mean_recovery_error", {})
        section["bkt_cold_start_curve"] = bkt_data.get("cold_start_curve", {})
        section["bkt_recovery_png"] = str(results_dir / "bkt_recovery.png") if (results_dir / "bkt_recovery.png").exists() else None
        section["repro_bkt_recovery"] = "uv run python scripts/bkt_recovery.py"
    else:
        section["bkt_recovery_available"] = False
        section["bkt_recovery_reason"] = f"bkt_recovery.json not found at {bkt_path}"

    return section


def _section_tutor(results_dir: Path) -> dict:
    """policy_replay.json + ab_dev.json + naive audit files → tutor headlines."""
    pr_path = results_dir / "policy_replay.json"
    ab_path = results_dir / "ab_dev.json"
    live_path = results_dir / "naive_live_audit.json"
    stub_path = results_dir / "naive_stub_audit.json"

    pr_data = _load_json(pr_path)
    ab_data = _load_json(ab_path)
    live_data = _load_json(live_path)
    stub_data = _load_json(stub_path)

    if pr_data is None and ab_data is None:
        return _unavailable(
            f"Both policy_replay.json and ab_dev.json missing at {results_dir}"
        )

    section: dict[str, Any] = {"available": True}

    # policy_replay
    if pr_data is not None:
        section["wait_rate"] = pr_data.get("wait_rate")
        section["move_distribution"] = pr_data.get("move_distribution", {})
        section["rule_distribution"] = pr_data.get("rule_distribution", {})
        section["policy_metadata"] = pr_data.get("metadata", {})
        section["repro_policy_replay"] = "uv run python scripts/policy_replay.py"
    else:
        section["policy_replay_available"] = False
        section["policy_replay_reason"] = f"policy_replay.json not found at {pr_path}"

    # ab_dev
    if ab_data is not None:
        section["ab_metrics"] = ab_data.get("metrics", {})
        section["ab_comparison_table"] = ab_data.get("comparison_table", [])
        section["ab_gate_outcomes"] = ab_data.get("gate_outcomes", {})
        section["ab_receipt_2"] = ab_data.get("receipt_2_gate_blocks_bad_tutor", {})
        section["promote_growth"] = ab_data.get("promote_growth", {})
        section["ab_metadata"] = {
            "ticket": ab_data.get("ticket"),
            "split": ab_data.get("split"),
            "n_sessions": ab_data.get("n_sessions"),
            "date": ab_data.get("date"),
        }
        section["repro_ab"] = "uv run python scripts/run_ab.py"
        promote_png = results_dir / "promote_growth.png"
        section["promote_growth_png"] = str(promote_png) if promote_png.exists() else None
    else:
        section["ab_available"] = False
        section["ab_reason"] = f"ab_dev.json not found at {ab_path}"

    # naive audits
    if live_data is not None:
        live_total = sum(
            p.get("violations", 0) for p in live_data.get("profiles", {}).values()
        )
        section["naive_live_violations"] = live_total
        section["naive_live_profiles"] = live_data.get("profiles", {})
        section["naive_live_model"] = live_data.get("model")
        section["repro_naive_live"] = "uv run python scripts/naive_replay.py --transport live"
    else:
        section["naive_live_violations"] = None

    if stub_data is not None:
        stub_total = sum(
            p.get("violations", 0) for p in stub_data.get("profiles", {}).values()
        )
        section["naive_stub_violations"] = stub_total
        section["naive_stub_profiles"] = stub_data.get("profiles", {})
        section["repro_naive_stub"] = "uv run python scripts/naive_replay.py --transport stub"
    else:
        section["naive_stub_violations"] = None

    return section


def _section_memory(results_dir: Path) -> dict:
    """learnermem_v0.json → consistency score + per-probe results."""
    path = results_dir / "learnermem_v0.json"
    data = _load_json(path)
    if data is None:
        return _unavailable(f"learnermem_v0.json not found or unparseable at {path}")

    return {
        "available": True,
        "consistency_score": data.get("consistency_score"),
        "n_passed": data.get("n_passed"),
        "n_total": data.get("n_total"),
        "probes": data.get("probes", {}),
        "findings": data.get("findings", []),
        "metadata": data.get("metadata", {}),
        "repro": "uv run python scripts/learnermem_probes.py",
    }


def _section_flywheel(results_dir: Path) -> dict:
    """v0.json baseline + promote_growth from ab_dev.json."""
    v0_path = results_dir / "v0.json"
    ab_path = results_dir / "ab_dev.json"

    v0_data = _load_json(v0_path)
    ab_data = _load_json(ab_path)

    if v0_data is None and ab_data is None:
        return _unavailable(
            f"Both v0.json and ab_dev.json missing at {results_dir}"
        )

    section: dict[str, Any] = {"available": True}

    if v0_data is not None:
        section["baseline_version"] = v0_data.get("version")
        section["baseline_commit"] = v0_data.get("metadata", {}).get("commit")
        section["baseline_date"] = v0_data.get("metadata", {}).get("date")
        section["baseline_metrics"] = v0_data.get("metrics", {})
        section["baseline_ci_status"] = (
            "committed artifact: evals/results/v0.json — "
            "run `uv run pytest tests/test_benchmark_artifacts.py -q` to verify"
        )
        section["repro_baseline"] = "uv run python scripts/run_benchmark.py"
    else:
        section["baseline_available"] = False

    if ab_data is not None:
        pg = ab_data.get("promote_growth", {})
        batches = pg.get("batches", [])
        promote_counts = {b["batch"]: b["promoted"] for b in batches}
        cumulative = [b["cumulative_golden_size"] for b in batches]
        section["promote_growth_batches"] = batches
        section["promote_counts"] = promote_counts
        section["promote_cumulative_total"] = cumulative[-1] if cumulative else 0
        section["promote_idempotent"] = pg.get("idempotent_on_rerun")
        promote_png = results_dir / "promote_growth.png"
        section["promote_growth_png"] = str(promote_png) if promote_png.exists() else None
        section["repro_promote"] = "uv run python scripts/run_ab.py"
    else:
        section["promote_growth_available"] = False

    return section


def _section_live_db(db_path: Path) -> dict:
    """Optional: mastery heatmap data + due reviews from the learner store."""
    try:
        # Import here to avoid hard dep at module level; still no streamlit.
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from readcoach.learner_store import SqliteLearnerStore
        from readcoach.planner import load_curriculum

        store = SqliteLearnerStore(db_path)
        curriculum = load_curriculum()
        skill_ids = list(curriculum.keys())

        # Mastery for all learners × skills
        all_learner_ids = store.all_learner_ids() if hasattr(store, "all_learner_ids") else []

        heatmap_rows = []
        due_reviews = []
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        for lid in all_learner_ids:
            row = {"learner_id": lid}
            for sid in skill_ids:
                m = store.get_mastery(lid, sid)
                row[sid] = round(m, 4) if m is not None else None
            heatmap_rows.append(row)

            # Due reviews
            for sid in skill_ids:
                card = store.get_card(lid, sid) if hasattr(store, "get_card") else None
                if card and card.due <= now:
                    due_reviews.append({"learner_id": lid, "skill_id": sid, "due": str(card.due)})

        store.close()

        return {
            "available": True,
            "heatmap_rows": heatmap_rows,
            "due_reviews": due_reviews,
            "n_learners": len(all_learner_ids),
            "n_skills": len(skill_ids),
            "skill_ids": skill_ids,
        }
    except Exception as exc:
        return _unavailable(f"live db error: {exc}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_dashboard_state(
    results_dir: str | Path,
    db_path: str | Path | None = None,
) -> dict:
    """Build the complete dashboard state dict from committed artifacts.

    Parameters
    ----------
    results_dir:
        Directory containing the committed eval result JSON files
        (e.g. ``evals/results/``).
    db_path:
        Optional path to a SQLite learner store.  If given and readable,
        adds a ``live_db`` section with mastery heatmap and due reviews.

    Returns
    -------
    dict with keys: detector, masking, learner, tutor, memory, flywheel,
    and optionally live_db.  Each section has at minimum ``available: bool``.
    Missing files produce ``{"available": False, "reason": "..."}``.
    """
    results_dir = Path(results_dir)

    state = {
        "results_dir": str(results_dir),
        "detector": _section_detector(results_dir),
        "masking": _section_masking(results_dir),
        "learner": _section_learner(results_dir),
        "tutor": _section_tutor(results_dir),
        "memory": _section_memory(results_dir),
        "flywheel": _section_flywheel(results_dir),
    }

    if db_path is not None:
        state["live_db"] = _section_live_db(Path(db_path))

    return state
