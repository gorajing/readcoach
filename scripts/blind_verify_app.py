"""T1.4 Blind Verification Web App (Streamlit front-end).

Two-step blind flow per clip:
  Step 1 — LISTEN (blind): hear audio + see passage text → build deviation list → lock
  Step 2 — REVEAL: see gold + auto_compare verdict → confirm match → submit

Both front-ends share blind_verify_queue.json + blind_verify_ratings.csv.
Sessions are fully interchangeable with the CLI.

Usage:
    uv run streamlit run scripts/blind_verify_app.py \\
        --server.headless true --server.port 8502
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import streamlit as st  # noqa: E402


# ---------------------------------------------------------------------------
# Load blind_verify.py via importlib (same pattern as label_app.py)
# ---------------------------------------------------------------------------


def _load_blind_verify():
    """Load scripts/blind_verify.py without triggering __main__ / argparse."""
    spec = importlib.util.spec_from_file_location(
        "blind_verify", _PROJECT_ROOT / "scripts" / "blind_verify.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bv = _load_blind_verify()

# Convenience aliases from the shared module
QUEUE_FILE = bv.QUEUE_FILE
RATINGS_FILE = bv.RATINGS_FILE
GOLD_PATH = bv.GOLD_PATH
ALL_CLASSES = bv.ALL_CLASSES

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ReadCoach — Blind Verification",
    page_icon="🎧",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "rater_initials" not in st.session_state:
    st.session_state.rater_initials = ""

# Two-step state machine per clip:
#   "listen"  → step 1 (blind: hear audio, build deviation list)
#   "reveal"  → step 2 (reveal gold, confirm match)
if "step" not in st.session_state:
    st.session_state.step = "listen"

# Deviation list accumulated in step 1
if "heard" not in st.session_state:
    st.session_state.heard: list[dict] = []

# Locked copy of heard (set when "Lock in what I heard" is pressed)
if "heard_locked" not in st.session_state:
    st.session_state.heard_locked: list[dict] | None = None

# ---------------------------------------------------------------------------
# Queue bootstrap: if no queue file offer creation
# ---------------------------------------------------------------------------

st.title("ReadCoach — Blind Verification")

if not QUEUE_FILE.exists():
    st.info(
        "No queue found. Click the button below to create a 30-clip queue "
        "(stratified sample, seed=42, deterministic)."
    )
    if st.button("Create my 30-clip queue (seed 42)"):
        if not GOLD_PATH.exists():
            st.error(
                f"Gold file not found: `{GOLD_PATH}`. "
                "Run `python3 scripts/fetch_benchmark.py` to download it."
            )
            st.stop()
        with st.spinner("Sampling clips…"):
            bv.cmd_init(GOLD_PATH, seed=42, n=30)
        st.success(f"Queue written to `{QUEUE_FILE}`.")
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Load queue and gold map
# ---------------------------------------------------------------------------

queue: list[dict] = json.loads(QUEUE_FILE.read_text())
gold_map = {g["utt_id"]: g for g in bv.load_gold(GOLD_PATH)}

# Load already-rated ids (validate existing CSV if it exists)
if RATINGS_FILE.exists():
    try:
        with RATINGS_FILE.open(newline="", encoding="utf-8") as f:
            bv.validate_ratings_csv(f)
    except ValueError as exc:
        st.error(
            f"Malformed ratings CSV detected:\n\n`{exc}`\n\n"
            f"Fix `{RATINGS_FILE}` before continuing."
        )
        st.stop()
    with RATINGS_FILE.open(newline="", encoding="utf-8") as f:
        rated_set = bv.load_rated_ids(f)
else:
    rated_set: set[str] = set()

pending = bv.pending_queue(queue, rated_set)
total = len(queue)
n_done = total - len(pending)

# ---------------------------------------------------------------------------
# DONE state
# ---------------------------------------------------------------------------

if not pending:
    st.success(
        f"All {total} clips verified — session complete!  "
        "Tell the agent: **'blind verification is done'**"
    )
    st.divider()
    st.subheader("Verification report")
    try:
        with RATINGS_FILE.open(newline="", encoding="utf-8") as f:
            report = bv.compute_report(f)
    except Exception as exc:
        st.error(f"Could not compute report: {exc}")
        st.stop()

    # Summary table
    import pandas as pd  # noqa: PLC0415

    summary_rows = [
        {"Metric": "Clips rated", "Value": str(report["n_rated"])},
        {"Metric": "Matches", "Value": str(report["n_match"])},
        {"Metric": "Mismatches", "Value": str(report["n_mismatch"])},
        {
            "Metric": "Mismatch rate",
            "Value": f"{report['mismatch_rate']:.3f} ({report['n_mismatch']}/{report['n_rated']})",
        },
    ]
    st.table(pd.DataFrame(summary_rows))

    # Per-class breakdown
    pc = report["per_class"]
    pc_rows = []
    for cls in list(ALL_CLASSES) + ["clean"]:
        counts = pc.get(cls, {"match": 0, "mismatch": 0})
        total_cls = counts["match"] + counts["mismatch"]
        if total_cls > 0:
            pc_rows.append(
                {
                    "Class": cls,
                    "Match": counts["match"],
                    "Mismatch": counts["mismatch"],
                    "Total": total_cls,
                }
            )
    if pc_rows:
        st.subheader("Per-class breakdown")
        st.table(pd.DataFrame(pc_rows))

    # Mismatch reasons
    if report["mismatches"]:
        st.subheader(f"Mismatch reasons ({len(report['mismatches'])})")
        for mm in report["mismatches"]:
            with st.container(border=True):
                st.markdown(f"**{mm['opaque_id']}** — gold: `{mm['gold_summary']}`")
                st.markdown(f"Reason: {mm['reason']}")
    st.stop()

# ---------------------------------------------------------------------------
# Current clip
# ---------------------------------------------------------------------------

entry = pending[0]
opaque_id = entry["opaque_id"]
wav_path = pathlib.Path(entry["wav_path"])
utt_id = entry["utt_id"]

gold_entry = gold_map.get(utt_id)
if gold_entry is None:
    st.error(
        f"Queue/gold desync: `{utt_id}` not found in gold.  "
        "Re-initialize the queue with the button on the home page."
    )
    st.stop()

if not wav_path.exists():
    st.error(
        f"WAV file not found: `{wav_path}`  \n"
        "Run `python3 scripts/fetch_benchmark.py` to download the benchmark clips."
    )
    st.stop()

target_text = gold_entry["target_text"]
gold_miscues = gold_entry["gold"]

# ---------------------------------------------------------------------------
# Progress header
# ---------------------------------------------------------------------------

clip_seq = n_done + 1  # 1-based
st.write(f"**Clip {clip_seq} of {total}**")
st.progress(n_done / total)
st.caption(
    f"{n_done} verified, {len(pending)} remaining. "
    "Close and re-open to resume — progress is auto-saved after each submit."
)

# ---------------------------------------------------------------------------
# Rater initials (sticky, required before first submit)
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

# ============================================================================
# STEP 1 — LISTEN (blind)
# ============================================================================

if st.session_state.step == "listen":
    st.subheader("Step 1 — Listen (blind)")

    # Target text in a quote box — ONLY identifying info shown to rater
    st.markdown(
        f"""<div style="font-size:1.15rem; font-weight:500; padding:0.75rem 1.2rem;
        border-left: 4px solid #4A90D9; background:#f0f6ff; border-radius:6px;
        margin-bottom:1rem; line-height:1.6;">
        <strong>What the reader was supposed to read:</strong><br>
        &ldquo;{target_text}&rdquo;
        </div>""",
        unsafe_allow_html=True,
    )

    # Audio player — pass BYTES not path to avoid leaking utt_id/filename in DOM
    audio_bytes = wav_path.read_bytes()
    st.audio(audio_bytes, format="audio/wav")
    st.caption("You can replay the clip as many times as you like.")

    st.markdown("---")
    st.markdown("**What deviations did you hear?**")

    # Show current deviation list
    if st.session_state.heard:
        st.markdown("**Deviations logged so far:**")
        for idx, dev in enumerate(st.session_state.heard):
            col_desc, col_rm = st.columns([5, 1])
            with col_desc:
                st.markdown(f"- `{dev['type']}` — word: *{dev['word'] or '(none)'}*")
            with col_rm:
                if st.button("Remove", key=f"rm_{idx}"):
                    st.session_state.heard.pop(idx)
                    st.rerun()
    else:
        st.markdown("*No deviations logged yet.*")

    st.markdown("")

    # Deviation builder
    with st.container(border=True):
        st.markdown("**Add a deviation:**")
        col_cls, col_word, col_add = st.columns([2, 3, 1])
        with col_cls:
            dev_class = st.selectbox(
                "Type",
                options=list(ALL_CLASSES),
                format_func=lambda c: c.replace("_", " ").title(),
                key="dev_class",
                label_visibility="collapsed",
            )
        with col_word:
            dev_word = st.text_input(
                "Approximate word",
                key="dev_word",
                placeholder="approximate word (optional)",
                label_visibility="collapsed",
            )
        with col_add:
            if st.button("Add", use_container_width=True):
                st.session_state.heard.append({"type": dev_class, "word": dev_word.strip()})
                st.rerun()

    st.markdown("")

    # Lock button
    col_lock, col_info = st.columns([2, 3])
    with col_lock:
        lock_label = (
            "Lock in what I heard — clean read (no deviations)"
            if not st.session_state.heard
            else "Lock in what I heard"
        )
        if st.button(lock_label, type="primary", use_container_width=True):
            # Freeze the heard list and advance to reveal step
            st.session_state.heard_locked = list(st.session_state.heard)
            st.session_state.step = "reveal"
            st.rerun()
    with col_info:
        if not st.session_state.heard:
            st.caption("Zero deviations = clean read. This is valid — just click Lock.")

# ============================================================================
# STEP 2 — REVEAL
# ============================================================================

elif st.session_state.step == "reveal":
    # Safety guard: should never happen if state machine is correct
    if st.session_state.heard_locked is None:
        st.error("Internal error: heard list was not locked. Reloading step 1.")
        st.session_state.step = "listen"
        st.rerun()

    heard_locked: list[dict] = st.session_state.heard_locked
    st.subheader("Step 2 — Reveal & confirm")

    # Show what the rater logged
    with st.container(border=True):
        st.markdown("**What you heard (locked):**")
        if heard_locked:
            for dev in heard_locked:
                st.markdown(f"- `{dev['type']}` — word: *{dev['word'] or '(none)'}*")
        else:
            st.markdown("*Clean read — no deviations.*")

    st.markdown("")

    # Compute gold summary and auto_compare
    gold_sum = bv._gold_summary(gold_miscues)
    cmp = bv.auto_compare(heard_locked, gold_miscues)

    # Gold reveal
    with st.container(border=True):
        st.markdown("**Gold label (ground truth):**")
        st.markdown(f"`{gold_sum}`")

        # Auto-compare verdict
        if cmp["match"]:
            st.success("Auto-compare: **MATCH** — your heard list matches the gold class counts.")
        else:
            parts = []
            if cmp["miss"]:
                parts.append(f"**MISSED** (in gold, not in heard): {', '.join(cmp['miss'])}")
            if cmp["extra"]:
                parts.append(f"**EXTRA** (in heard, not in gold): {', '.join(cmp['extra'])}")
            st.warning("Auto-compare: " + "  |  ".join(parts))

    st.markdown("")

    # Rater confirms match
    match_val = st.radio(
        "Does the gold match what you heard?",
        options=["y", "n"],
        format_func=lambda v: "Yes — it matches" if v == "y" else "No — mismatch",
        horizontal=True,
        index=None,
        key="match_radio",
    )

    reason_val = ""
    if match_val == "n":
        reason_val = st.text_input(
            "Reason (required for mismatch)",
            placeholder="e.g. heard hesitation but gold has none",
            key="reason_input",
        )

    st.markdown("")

    if st.button("Submit & next clip", type="primary", use_container_width=True):
        # Validate
        initials = st.session_state.rater_initials
        if not initials:
            st.error("Please enter your rater initials before submitting.")
            st.stop()

        if match_val is None:
            st.error("Please select y or n before submitting.")
            st.stop()

        if match_val == "n" and not reason_val.strip():
            st.error("A non-empty reason is required when the match is 'n'.")
            st.stop()

        heard_str = json.dumps(heard_locked)
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        row = {
            "opaque_id": opaque_id,
            "utt_id": utt_id,
            "heard": heard_str,
            "gold_summary": gold_sum,
            "match": match_val,
            "reason": reason_val.strip(),
            "timestamp": ts,
            "rater_initials": initials,
        }

        try:
            RATINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            bv._append_rating_row(RATINGS_FILE, row)
        except Exception as exc:
            st.error(f"Error writing rating: {exc}")
            st.stop()

        # Reset state for next clip
        st.session_state.step = "listen"
        st.session_state.heard = []
        st.session_state.heard_locked = None
        st.rerun()
