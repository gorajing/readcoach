"""T6.2 — ReadCoach Dashboard (Streamlit).

One screenshot-quality view over the committed eval artifacts.
Pure render layer: all computation is in dashboard_state.build_dashboard_state().

Usage:
    uv run streamlit run scripts/dashboard.py --server.headless true

Optional: pass --db-path <path> to include the live learner store section.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src importable when running via `uv run streamlit run scripts/dashboard.py`
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import streamlit as st  # noqa: E402

from readcoach.dashboard_state import build_dashboard_state  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ReadCoach — Eval Dashboard",
    page_icon="📖",
    layout="wide",
)

RESULTS_DIR = _PROJECT_ROOT / "evals" / "results"

# Optional db path from CLI args (streamlit passes unknown args after --)
_db_path: Path | None = None
_args = sys.argv[1:]
if "--db-path" in _args:
    idx = _args.index("--db-path")
    if idx + 1 < len(_args):
        _db_path = Path(_args[idx + 1])

# ---------------------------------------------------------------------------
# Load state
# ---------------------------------------------------------------------------

state = build_dashboard_state(RESULTS_DIR, db_path=_db_path)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("ReadCoach — Eval Dashboard")
st.caption(f"Artifacts from `{RESULTS_DIR}`")

# ---------------------------------------------------------------------------
# Big-number tiles
# ---------------------------------------------------------------------------

det = state["detector"]
tut = state["tutor"]
mem = state["memory"]
lrn = state["learner"]

t1, t2, t3, t4, t5, t6 = st.columns(6)

with t1:
    if det.get("available"):
        val = det.get("sub_recall_none")
        st.metric(
            "Sub recall @ none-bias",
            f"{val:.3f}" if val is not None else "—",
            help="substitution recall with no bias prompt (miscue-v0.json)",
        )
    else:
        st.metric("Sub recall @ none-bias", "n/a")

with t2:
    if det.get("available"):
        val = det.get("fp_per_100_strong")
        st.metric(
            "FP/100 @ strong bias",
            f"{val:.3f}" if val is not None else "—",
            help="false positives per 100 correct words at strong bias (miscue-v0.json)",
        )
    else:
        st.metric("FP/100 @ strong bias", "n/a")

with t3:
    if lrn.get("available"):
        val = lrn.get("break_even_a")
        st.metric(
            "Break-even a",
            f"{val:.2f}" if val is not None else "—",
            help="soft-evidence vs naive BKT break-even accuracy (break_even.json)",
        )
    else:
        st.metric("Break-even a", "n/a")

with t4:
    if tut.get("available"):
        val = tut.get("wait_rate")
        st.metric(
            "WAIT-rate (policy replay)",
            f"{val:.3f}" if val is not None else "—",
            help="fraction of decisions that are WAIT (policy_replay.json)",
        )
    else:
        st.metric("WAIT-rate", "n/a")

with t5:
    if mem.get("available"):
        val = mem.get("consistency_score")
        st.metric(
            "LearnerMem score",
            f"{val:.1f}" if val is not None else "—",
            help="consistency score over 6 probes (learnermem_v0.json)",
        )
    else:
        st.metric("LearnerMem score", "n/a")

with t6:
    if tut.get("available"):
        live_v = tut.get("naive_live_violations")
        stub_v = tut.get("naive_stub_violations")
        if live_v is not None and stub_v is not None:
            st.metric(
                "Naive violations live / stub",
                f"{live_v} / {stub_v}",
                help="invariant violations: live LLM vs stub (naive_live_audit.json / naive_stub_audit.json)",
            )
        else:
            st.metric("Naive violations", "n/a")
    else:
        st.metric("Naive violations", "n/a")

st.divider()

# ---------------------------------------------------------------------------
# Section 1: Detector
# ---------------------------------------------------------------------------

st.header("1. Miscue Detector")

if not det.get("available"):
    st.warning(f"Not available: {det.get('reason')}")
else:
    import pandas as pd

    biases = det.get("biases", [])
    per_class = det.get("per_class_per_bias", {})

    # Build table: rows = classes, cols = bias × metric
    rows = []
    for cls in sorted(per_class.keys()):
        row = {"class": cls}
        for bias in biases:
            bd = per_class[cls].get(bias, {})
            row[f"{bias} P"] = f"{bd.get('precision', 0):.3f}" if bd.get("precision") is not None else "—"
            row[f"{bias} R"] = f"{bd.get('recall', 0):.3f}" if bd.get("recall") is not None else "—"
            row[f"{bias} F1"] = f"{bd.get('f1', 0):.3f}" if bd.get("f1") is not None else "—"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("class")
    st.dataframe(df, use_container_width=True)

    st.write("**FP per 100 correct words by bias**")
    fp_df = pd.DataFrame(
        [{"bias": b, "fp_per_100": v} for b, v in det.get("fp_per_100_by_bias", {}).items()]
    )
    st.dataframe(fp_df, use_container_width=True, hide_index=True)

    masking = state.get("masking", {})
    if masking.get("available") and masking.get("png_path"):
        png = Path(masking["png_path"])
        if png.exists():
            st.image(str(png), caption="Masking curve (bias sweep)", use_container_width=True)

    st.caption(f"Repro: `{det.get('repro')}`")

st.divider()

# ---------------------------------------------------------------------------
# Section 2: Masking Curve
# ---------------------------------------------------------------------------

st.header("2. Masking Curve")

masking = state.get("masking", {})
if not masking.get("available"):
    st.warning(f"Not available: {masking.get('reason')}")
else:
    import pandas as pd

    curve = masking.get("curve_points", {})
    rows = []
    for bias, bd in curve.items():
        fp = bd.get("fp_per_100_correct_words")
        ci = bd.get("ci_fp_per_100") or []
        wer = bd.get("wer_vs_spoken_mean")
        ci_wer = bd.get("ci_wer_vs_spoken") or []
        rows.append({
            "bias": bias,
            "fp_per_100": round(fp, 4) if fp is not None else None,
            "ci_fp [lo, hi]": f"[{ci[0]:.4f}, {ci[1]:.4f}]" if len(ci) == 2 else "—",
            "wer_vs_spoken": round(wer, 4) if wer is not None else None,
            "ci_wer [lo, hi]": f"[{ci_wer[0]:.4f}, {ci_wer[1]:.4f}]" if len(ci_wer) == 2 else "—",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"Repro: `{masking.get('repro')}`")

st.divider()

# ---------------------------------------------------------------------------
# Section 3: Learner / BKT
# ---------------------------------------------------------------------------

st.header("3. Learner Model (BKT)")

lrn = state.get("learner", {})
if not lrn.get("available"):
    st.warning(f"Not available: {lrn.get('reason')}")
else:
    be_a = lrn.get("break_even_a")
    if be_a is not None:
        st.subheader("Break-even: soft evidence vs. naive BKT")
        st.write(
            f"Break-even accuracy: **a = {be_a}** "
            f"(soft ≈ naive RMSE at this detector accuracy level)"
        )

        # a_eff anchors table
        anchors = lrn.get("a_eff_anchors", [])
        if anchors:
            import pandas as pd
            df = pd.DataFrame(anchors)
            st.dataframe(df, use_container_width=True, hide_index=True)

        if lrn.get("break_even_png"):
            png = Path(lrn["break_even_png"])
            if png.exists():
                st.image(str(png), caption="Break-even curve", use_container_width=True)

        st.caption(f"Repro: `{lrn.get('repro_break_even')}`")

    brier = lrn.get("bkt_brier_score")
    if brier is not None:
        st.subheader("BKT Parameter Recovery")
        st.write(f"Brier score: **{brier:.4f}**")

        note = lrn.get("bkt_weak_spot_note", "")
        if note:
            with st.expander("Weak-spot analysis (full note)"):
                st.write(note)

        regimes = lrn.get("bkt_regimes", {})
        if regimes:
            import pandas as pd
            rows = []
            for regime, rd in regimes.items():
                true_p = rd.get("true", {})
                fit_p = rd.get("fit", {})
                err = rd.get("recovery_error", {})
                rows.append({
                    "regime": regime,
                    "mastery_rmse": round(rd.get("mastery_rmse", 0), 4),
                    "err_s": round(err.get("s", 0), 4),
                    "err_g": round(err.get("g", 0), 4),
                    "err_t": round(err.get("t", 0), 4),
                    "err_L0": round(err.get("L0", 0), 4),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        if lrn.get("bkt_recovery_png"):
            png = Path(lrn["bkt_recovery_png"])
            if png.exists():
                st.image(str(png), caption="BKT recovery curves", use_container_width=True)

        st.caption(f"Repro: `{lrn.get('repro_bkt_recovery')}`")

st.divider()

# ---------------------------------------------------------------------------
# Section 4: Tutor Policy
# ---------------------------------------------------------------------------

st.header("4. Tutor Policy")

tut = state.get("tutor", {})
if not tut.get("available"):
    st.warning(f"Not available: {tut.get('reason')}")
else:
    # Policy replay
    wait_rate = tut.get("wait_rate")
    if wait_rate is not None:
        st.subheader("Policy Replay")
        st.write(f"WAIT-rate: **{wait_rate:.4f}** (target band 0.35–0.50)")

        move_dist = tut.get("move_distribution", {})
        if move_dist:
            import pandas as pd
            df = pd.DataFrame(
                [{"move": k, "count": v} for k, v in sorted(move_dist.items(), key=lambda x: -x[1])]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.caption(f"Repro: `{tut.get('repro_policy_replay')}`")

    # A/B table
    ab_table = tut.get("ab_comparison_table", [])
    if ab_table:
        st.subheader("A/B Comparison (dev split)")
        import pandas as pd
        df = pd.DataFrame(ab_table)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Gate outcomes
        gates = tut.get("ab_gate_outcomes", {})
        if gates:
            for comparison, outcome in gates.items():
                passed = outcome.get("passed")
                icon = "✅" if passed else "❌"
                st.write(f"{icon} **{comparison}**: gate {'PASSED' if passed else 'BLOCKED'}")
                if not passed:
                    for breach in outcome.get("breaches", []):
                        st.code(breach)

        # Receipt #2
        r2 = tut.get("ab_receipt_2", {})
        if r2:
            with st.expander("Receipt #2: gate blocks bad tutor (v3)"):
                st.write(r2.get("evidence", ""))
                st.write(f"Gate passed: **{r2.get('gate_passed')}**")

        st.caption(f"Repro: `{tut.get('repro_ab')}`")

    # Promote-growth chart
    pg_png = tut.get("promote_growth_png")
    if pg_png and Path(pg_png).exists():
        st.image(pg_png, caption="Promote-growth (golden failures injected per batch)", use_container_width=True)

    # Naive violations
    st.subheader("Naive Tutor Invariant Violations")
    live_v = tut.get("naive_live_violations")
    stub_v = tut.get("naive_stub_violations")
    live_model = tut.get("naive_live_model", "?")
    c1, c2 = st.columns(2)
    with c1:
        if live_v is not None:
            st.metric(f"Live LLM ({live_model})", f"{live_v} violations")
            st.caption(f"Repro: `{tut.get('repro_naive_live')}`")
        else:
            st.metric("Live LLM", "not available")
    with c2:
        if stub_v is not None:
            st.metric("Stub transport", f"{stub_v} violations")
            st.caption(f"Repro: `{tut.get('repro_naive_stub')}`")
        else:
            st.metric("Stub transport", "not available")

st.divider()

# ---------------------------------------------------------------------------
# Section 5: LearnerMem
# ---------------------------------------------------------------------------

st.header("5. LearnerMem")

mem = state.get("memory", {})
if not mem.get("available"):
    st.warning(f"Not available: {mem.get('reason')}")
else:
    score = mem.get("consistency_score")
    n_passed = mem.get("n_passed")
    n_total = mem.get("n_total")
    st.write(
        f"Consistency score: **{score:.1f}** "
        f"({n_passed}/{n_total} probes passed)"
    )

    probes = mem.get("probes", {})
    if probes:
        import pandas as pd
        rows = []
        for pid, pd_data in sorted(probes.items()):
            rows.append({
                "probe": pid,
                "passed": "✅" if pd_data.get("passed") else "❌",
                "description": pd_data.get("description", ""),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("Probe evidence"):
            for pid, pd_data in sorted(probes.items()):
                st.write(f"**{pid}**: {pd_data.get('evidence', '')}")

    findings = mem.get("findings", [])
    if findings:
        st.subheader("Findings")
        for f in findings:
            st.info(f)

    st.caption(f"Repro: `{mem.get('repro')}`")

st.divider()

# ---------------------------------------------------------------------------
# Section 6: Flywheel
# ---------------------------------------------------------------------------

st.header("6. Flywheel")

fly = state.get("flywheel", {})
if not fly.get("available"):
    st.warning(f"Not available: {fly.get('reason')}")
else:
    # Baseline gate status
    baseline_commit = fly.get("baseline_commit")
    baseline_date = fly.get("baseline_date")
    ci_status = fly.get("baseline_ci_status", "")
    st.write(
        f"Baseline: **v0** — commit `{baseline_commit[:12] if baseline_commit else '?'}` "
        f"({baseline_date})"
    )
    st.caption(ci_status)

    # Promote-growth counts
    batches = fly.get("promote_growth_batches", [])
    if batches:
        import pandas as pd
        df = pd.DataFrame(batches)
        st.dataframe(df, use_container_width=True, hide_index=True)

    cumulative = fly.get("promote_cumulative_total", 0)
    st.write(f"Total promoted to golden failures: **{cumulative}**")

    pg_png = fly.get("promote_growth_png")
    if pg_png and Path(pg_png).exists():
        st.image(pg_png, caption="Promote-growth chart", use_container_width=True)

    st.caption(f"Repro: `{fly.get('repro_baseline')}`  |  promote: `{fly.get('repro_promote')}`")

st.divider()

# ---------------------------------------------------------------------------
# Section 7: Live DB (optional)
# ---------------------------------------------------------------------------

if "live_db" in state:
    st.header("7. Live Learner Store")
    ldb = state["live_db"]
    if not ldb.get("available"):
        st.warning(f"Not available: {ldb.get('reason')}")
    else:
        n_learners = ldb.get("n_learners", 0)
        n_skills = ldb.get("n_skills", 0)
        n_due = len(ldb.get("due_reviews", []))
        st.write(f"Learners: **{n_learners}** | Skills: **{n_skills}** | Due reviews: **{n_due}**")

        heatmap = ldb.get("heatmap_rows", [])
        if heatmap:
            import pandas as pd
            df = pd.DataFrame(heatmap).set_index("learner_id")
            st.dataframe(df, use_container_width=True)

        due = ldb.get("due_reviews", [])
        if due:
            import pandas as pd
            st.write("**Due reviews**")
            df = pd.DataFrame(due)
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

# ---------------------------------------------------------------------------
# Footer: repro commands
# ---------------------------------------------------------------------------

st.header("Repro Commands")
st.caption("Every number above is reproducible with the command shown in each section. Summary:")

repro_lines = []
if state["detector"].get("available"):
    repro_lines.append(("Detector (miscue-v0.json)", state["detector"].get("repro", "")))
if state["masking"].get("available"):
    repro_lines.append(("Masking curve", state["masking"].get("repro", "")))
if state["learner"].get("available"):
    repro_lines.append(("Break-even", state["learner"].get("repro_break_even", "")))
    repro_lines.append(("BKT recovery", state["learner"].get("repro_bkt_recovery", "")))
if state["tutor"].get("available"):
    repro_lines.append(("Policy replay", state["tutor"].get("repro_policy_replay", "")))
    repro_lines.append(("A/B dev", state["tutor"].get("repro_ab", "")))
    repro_lines.append(("Naive live audit", state["tutor"].get("repro_naive_live", "")))
    repro_lines.append(("Naive stub audit", state["tutor"].get("repro_naive_stub", "")))
if state["memory"].get("available"):
    repro_lines.append(("LearnerMem probes", state["memory"].get("repro", "")))
if state["flywheel"].get("available"):
    repro_lines.append(("Baseline benchmark", state["flywheel"].get("repro_baseline", "")))
    repro_lines.append(("Promote-growth", state["flywheel"].get("repro_promote", "")))

for label, cmd in repro_lines:
    if cmd:
        st.code(f"# {label}\n{cmd}", language="bash")
