"""T5.L — ReadCoach Human Labeling App (Streamlit front-end).

A friendlier alternative to scripts/label_turns.py.  Both front-ends share
the same label_queue.json and turn_labels.csv, so sessions are interchangeable.

Usage:
    uv run streamlit run scripts/label_app.py --server.headless true
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Python-path: make scripts/ importable as "label_turns" (same as test suite)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import importlib.util  # noqa: E402
import streamlit as st  # noqa: E402


def _load_label_turns():
    """Load scripts/label_turns.py without triggering __main__ / argparse."""
    spec = importlib.util.spec_from_file_location(
        "label_turns", _PROJECT_ROOT / "scripts" / "label_turns.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lt = _load_label_turns()

# Convenience aliases
QUEUE_FILE = lt.QUEUE_FILE
LABELS_FILE = lt.LABELS_FILE
TURNS_PATH = lt.TURNS_PATH
DIMENSIONS = lt.DIMENSIONS
ANCHOR_ONELINER = lt.ANCHOR_ONELINER
FULL_ANCHOR_FILE = lt.FULL_ANCHOR_FILE

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ReadCoach — Labeling Session",
    page_icon="📝",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "rater_initials" not in st.session_state:
    st.session_state.rater_initials = ""

if "submitted_this_session" not in st.session_state:
    st.session_state.submitted_this_session = 0

# ---------------------------------------------------------------------------
# Helper: load labeled pairs from disk
# ---------------------------------------------------------------------------


def _labeled_pairs() -> set[tuple[str, str]]:
    if not LABELS_FILE.exists():
        return set()
    with LABELS_FILE.open(newline="", encoding="utf-8") as f:
        return lt.load_labeled_pairs(f)


def _turn_complete(turn_id: str, labeled: set[tuple[str, str]]) -> bool:
    return all((turn_id, dim) in labeled for dim in DIMENSIONS)


# ---------------------------------------------------------------------------
# Helper: full rubric text for a dimension (from labeling_rubric.md)
# ---------------------------------------------------------------------------

_RUBRIC_CACHE: dict[str, str] = {}


def _dim_rubric_section(dimension: str) -> str:
    """Extract the section for `dimension` from labeling_rubric.md."""
    if dimension in _RUBRIC_CACHE:
        return _RUBRIC_CACHE[dimension]
    if not FULL_ANCHOR_FILE.exists():
        return "(rubric file not found)"
    full_text = FULL_ANCHOR_FILE.read_text(encoding="utf-8")
    # Each dimension has a header like "## Dimension N — Guidance quality"
    dim_header_map = {
        "guidance": "Guidance quality",
        "actionability": "Actionability",
        "icap": "ICAP engagement level",
    }
    header_frag = dim_header_map.get(dimension, dimension)
    lines = full_text.splitlines()
    collecting = False
    section_lines: list[str] = []
    for line in lines:
        if not collecting:
            if header_frag in line and line.startswith("## Dimension"):
                collecting = True
                section_lines.append(line)
        else:
            # Stop at next level-2 section or ruler
            if (line.startswith("## ") and header_frag not in line) or line.startswith("---"):
                break
            section_lines.append(line)
    result = "\n".join(section_lines).strip() or "(section not found)"
    _RUBRIC_CACHE[dimension] = result
    return result


# ---------------------------------------------------------------------------
# Helper: render inline report (mirrors cmd_report output)
# ---------------------------------------------------------------------------


def _render_report() -> None:
    import csv

    if not LABELS_FILE.exists():
        st.warning("No labels file found.")
        return

    rows: list[dict] = []
    with LABELS_FILE.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        st.warning("No label rows yet.")
        return

    st.write(f"**Total label rows:** {len(rows)}")
    for dim in DIMENSIONS:
        dim_rows = [r for r in rows if r["dimension"] == dim]
        if not dim_rows:
            st.write(f"**{dim.upper()}:** 0 labels")
            continue
        scores = [int(r["human_score"]) for r in dim_rows]
        passing = sum(1 for r in dim_rows if r["human_passing"] == "y")
        dist = {s: scores.count(s) for s in range(1, 6)}
        mean_score = sum(scores) / len(scores)
        st.write(
            f"**{dim.upper()}:** {len(dim_rows)} labels | "
            f"passing {passing}/{len(dim_rows)} | "
            f"mean {mean_score:.2f} | "
            f"dist {dist}"
        )

    turn_ids = {r["turn_id"] for r in rows}
    st.write(f"**Unique turns labeled:** {len(turn_ids)}")


# ---------------------------------------------------------------------------
# Helper: human-readable context fields
# ---------------------------------------------------------------------------

_PROFILE_DISPLAY = {
    "struggling-decoder": "Struggling Decoder",
    "fluent-but-hesitant": "Fluent but Hesitant",
    "self-corrector": "Self-Corrector",
}

_MISCUE_DISPLAY = {
    "substitution": "Substitution",
    "omission": "Omission",
    "insertion": "Insertion",
    "hesitation": "Hesitation",
    "self_correction": "Self-Correction",
}


def _fmt_profile(p: str) -> str:
    return _PROFILE_DISPLAY.get(p, p)


def _fmt_miscue(m: str | None) -> str:
    if m is None:
        return "None (clean read)"
    return _MISCUE_DISPLAY.get(m, m)


# ---------------------------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------------------------

st.title("ReadCoach — Human Labeling Session")

# ---------------------------------------------------------------------------
# Queue bootstrap: if no queue file offer creation
# ---------------------------------------------------------------------------

if not QUEUE_FILE.exists():
    st.info(
        "No label queue found. Click the button below to create a 60-turn queue "
        "(stratified sample, seed=42, deterministic)."
    )
    if st.button("Create my 60-turn queue"):
        with st.spinner("Sampling turns…"):
            lt.cmd_init(TURNS_PATH, n=60, seed=42)
        st.success(f"Queue written to `{QUEUE_FILE}`.")
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Load queue and compute progress
# ---------------------------------------------------------------------------

queue: list[dict] = lt._load_queue(QUEUE_FILE)
total_turns = len(queue)

labeled = _labeled_pairs()

complete_turns = [t for t in queue if _turn_complete(t["turn_id"], labeled)]
pending_turns = [t for t in queue if not _turn_complete(t["turn_id"], labeled)]
n_complete = len(complete_turns)

# ---------------------------------------------------------------------------
# DONE state
# ---------------------------------------------------------------------------

if not pending_turns:
    st.balloons()
    st.success(
        f"All {total_turns} turns labeled — session complete! "
        "Tell the agent: 'the labels are done.'"
    )
    st.divider()
    st.subheader("Session report")
    _render_report()
    st.stop()

# ---------------------------------------------------------------------------
# Progress header
# ---------------------------------------------------------------------------

current_turn = pending_turns[0]
turn_seq = n_complete + 1  # 1-based position in overall progress

st.write(f"**Turn {turn_seq} of {total_turns}** — 3 dimensions each")
st.progress(n_complete / total_turns)
st.caption(
    f"{n_complete} complete, {len(pending_turns)} remaining. "
    "Quit and re-open to resume — progress is auto-saved after each submit."
)

# ---------------------------------------------------------------------------
# Rater initials (sticky)
# ---------------------------------------------------------------------------

col_init, _ = st.columns([1, 3])
with col_init:
    new_initials = st.text_input(
        "Your initials (required before first submit)",
        value=st.session_state.rater_initials,
        placeholder="e.g. JC",
        key="initials_input",
    )
    st.session_state.rater_initials = new_initials.strip()

st.divider()

# ---------------------------------------------------------------------------
# Context card
# ---------------------------------------------------------------------------

turn_id = current_turn["turn_id"]

with st.container(border=True):
    st.markdown("### Context")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Profile", _fmt_profile(current_turn.get("profile", "?")))
    with c2:
        st.metric("Page position", "End of page" if current_turn.get("at_page_end") else "Mid-page")
    with c3:
        st.metric("Miscue type", _fmt_miscue(current_turn.get("miscue_type")))
    with c4:
        struggle_count = current_turn.get("struggle_count", current_turn.get("hint_level", "n/a"))
        st.metric("Hint / struggle", str(struggle_count) if struggle_count is not None else "n/a")

    st.caption(
        f"**Move:** `{current_turn.get('action_move', '?')}`"
        + (f"  ·  hint_level={current_turn['hint_level']}" if current_turn.get("hint_level") else "")
        + (f"  ·  skill={current_turn['skill_id']}" if current_turn.get("skill_id") else "")
        + f"  ·  turn_id=`{turn_id}`"
    )

# ---------------------------------------------------------------------------
# Tutor utterance (big + quoted)
# ---------------------------------------------------------------------------

st.markdown("")
utterance = current_turn.get("utterance", "")
st.markdown(
    f"""<div style="font-size:1.45rem; font-weight:600; padding:0.75rem 1.2rem;
    border-left: 4px solid #4A90D9; background:#f0f6ff; border-radius:6px;
    margin-bottom:1rem; line-height:1.5;">
    &ldquo;{utterance}&rdquo;
    </div>""",
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Score widgets — one per dimension
# ---------------------------------------------------------------------------

# Figure out which dims already done for this turn (partial resume)
pending_dims = [d for d in DIMENSIONS if (turn_id, d) not in labeled]

scores: dict[str, int | None] = {}
passing_calls: dict[str, str | None] = {}

for dim in pending_dims:
    st.markdown(f"#### {dim.upper()}")

    # Compact anchor oneliners under the radio
    anchor_opts = [f"{s} — {ANCHOR_ONELINER[dim][s]}" for s in (5, 4, 3, 2, 1)]

    chosen_idx = st.radio(
        f"Score for {dim} (1–5)",
        options=list(range(5)),
        format_func=lambda i, d=dim: f"{5-i} — {ANCHOR_ONELINER[d][5-i]}",
        horizontal=True,
        index=None,
        key=f"radio_{turn_id}_{dim}",
        label_visibility="collapsed",
    )

    # Show score value (the radio encodes 0=5, 1=4, 2=3, 3=2, 4=1)
    score_val: int | None = (5 - chosen_idx) if chosen_idx is not None else None
    scores[dim] = score_val

    # Borderline passing call when score == 3
    if score_val == 3:
        ba = st.radio(
            "Score 3 is borderline — does this turn pass?",
            options=["y", "n"],
            format_func=lambda v: "Yes — pass" if v == "y" else "No — fail",
            horizontal=True,
            index=None,
            key=f"pass_{turn_id}_{dim}",
        )
        passing_calls[dim] = ba
    else:
        passing_calls[dim] = None

    # Full rubric expander
    with st.expander(f"Full anchors for {dim}…"):
        st.markdown(_dim_rubric_section(dim))

    st.markdown("")

# ---------------------------------------------------------------------------
# Submit button
# ---------------------------------------------------------------------------

st.divider()

if st.button("Submit all scores", type="primary", use_container_width=True):
    # Validate
    initials = st.session_state.rater_initials
    if not initials:
        st.error("Please enter your rater initials before submitting.")
        st.stop()

    errors: list[str] = []
    for dim in pending_dims:
        s = scores.get(dim)
        if s is None:
            errors.append(f"Missing score for **{dim}**.")
        elif s == 3 and passing_calls.get(dim) is None:
            errors.append(f"Score 3 for **{dim}** requires a pass/fail call.")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # Write rows via shared writer
    wrote_count = 0
    try:
        for dim in pending_dims:
            s = scores[dim]
            human_passing = lt.derive_passing(s, passing_calls.get(dim))
            row = {
                "turn_id": turn_id,
                "dimension": dim,
                "human_score": str(s),
                "human_passing": human_passing,
                "rater_initials": initials,
            }
            LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
            lt._append_label_row(LABELS_FILE, row)
            wrote_count += 1
    except Exception as exc:
        st.error(f"Error writing label: {exc}")
        st.stop()

    st.session_state.submitted_this_session += 1
    st.success(f"Saved {wrote_count} rows for `{turn_id}`. Loading next turn…")
    st.rerun()
