"""T4.2 — Two-session quest-planner memory demo.

Demonstrates the planner's quest state carrying over across a SQLite close
and reopen.  All claims are printed and observable; the script exits 0.

Observable claims demonstrated:
  (a) Mastery carried over from session 1 survives store.close() + reopen.
  (b) FSRS due-reviews are present in session 2 (failed obs → card due soon).
  (c) The planner's first pick in session 2 is blocked by the CLASS gate on
      the cvc_short_a → cvc_short_i_u edge: mastery[cvc_short_a] is above the
      0.80 mastery_min (mastery gate PASSES) but the last-5 window contains a
      substitution-tagged incorrect obs (class gate FIRES → BLOCKED).  Both
      gate evaluations are printed explicitly so the observable IS the class
      gate, not a mastery drop wearing its clothes.
  (d) A completed skill (cvc_short_i_u, mastery >= 0.95 from session 1) is
      never re-served with reason='new' in session 2.

Session 1 staging:
  1. Drive cvc_short_a to >= 0.98 (4 correct obs; one failure from 0.98
     drops to ~0.90, still above 0.80 → mastery gate PASSES).
  2. Drive cvc_short_i_u to >= 0.95 (MASTERY_COMPLETED) while cvc_short_a
     is clean — cvc_short_i_u is now a completed skill in the served log.
  3. Inject ONE substitution-tagged failure on cvc_short_a.
     cvc_short_a mastery drops to ~0.90 (>= 0.80 mastery gate passes;
     class gate fires → cvc_short_i_u locked in session 2).

Usage:
    uv run python scripts/two_session_demo.py [db_path]

If db_path is omitted a temporary file is used and cleaned up on exit.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure the src package is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from readcoach.learner_store import SqliteLearnerStore
from readcoach.planner import (
    MASTERY_COMPLETED,
    MASTERY_THRESHOLD,
    load_curriculum,
    next_item,
    unlocked,
)

CURRICULUM_PATH = (
    Path(__file__).parent.parent / "data" / "curriculum" / "scope_sequence.yaml"
)

CHILD_ID = "demo_child"
BASE_TS = datetime(2026, 6, 10, 9, 0, 0, tzinfo=timezone.utc)


def _obs(store, skill, correct, *, miscue_class=None, n=1, offset_s=0,
         session_id="session_1"):
    """Record n observations on skill."""
    for i in range(n):
        store.record_observation(
            child_id=CHILD_ID,
            skill=skill,
            correct=correct,
            confidence=1.0,
            session_id=session_id,
            ts=BASE_TS + timedelta(seconds=offset_s + i),
            miscue_class=miscue_class,
        )


def _mastery_to(store, skill, target, session_id="session_1"):
    """Drive mastery on skill to >= target via correct observations."""
    for i in range(60):
        state = store.get_state(CHILD_ID)
        if state.mastery.get(skill, 0.0) >= target:
            return
        store.record_observation(
            child_id=CHILD_ID,
            skill=skill,
            correct=True,
            confidence=1.0,
            session_id=session_id,
            ts=BASE_TS + timedelta(seconds=i),
        )
    raise RuntimeError(f"Could not reach mastery={target} on {skill!r} in 60 steps")


def _print_edge_eval(store, prereq_skill, dep_skill, mastery, k=5):
    """Print both gate evaluations for a prerequisite edge.

    Prints one line per gate (mastery gate, class gate) and whether the
    overall edge is satisfied.  This makes the class gate observable as a
    distinct, independently evaluated mechanism.
    """
    curriculum = load_curriculum(CURRICULUM_PATH)
    dep_node = curriculum.nodes[dep_skill]
    for edge in dep_node.prerequisites:
        if edge.skill != prereq_skill:
            continue
        m = mastery.get(edge.skill, 0.0)
        mastery_pass = m >= edge.mastery_min
        mastery_label = "PASS" if mastery_pass else f"FAIL (needs >= {edge.mastery_min})"

        recent = store.get_last_k_observations(CHILD_ID, edge.skill, k=k)
        class_blockers = [
            o for o in recent
            if not o["correct"] and o["miscue_class"] in edge.classes
        ]
        class_pass = len(class_blockers) == 0
        class_label = (
            "PASS (no blocking miscues in last 5)"
            if class_pass
            else f"{len(class_blockers)} unresolved in last {k} -> BLOCKED"
        )

        edge_verdict = "UNLOCKED" if (mastery_pass and class_pass) else "LOCKED"
        print(
            f"    edge {edge.skill}->{dep_skill}: "
            f"mastery {m:.2f} >= {edge.mastery_min} {mastery_label}; "
            f"class gate {list(edge.classes)}: {class_label} "
            f"=> {edge_verdict}"
        )


def run_demo(db_path: str) -> None:
    curriculum = load_curriculum(CURRICULUM_PATH)

    # ── SESSION 1 ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("SESSION 1")
    print("=" * 60)

    s1 = SqliteLearnerStore(db_path)

    # Step 1: Drive cvc_short_a to >= 0.98.
    # One failure from 0.98 drops mastery to ~0.90 (still >= 0.80 mastery_min).
    # This ensures in session 2 the mastery gate PASSES and only the class gate
    # can block cvc_short_i_u.
    _mastery_to(s1, "cvc_short_a", 0.98, session_id="session_1")
    m_a_pre = s1.get_state(CHILD_ID).mastery.get("cvc_short_a", 0.0)
    print(f"[S1] cvc_short_a mastery (pre-failure):  {m_a_pre:.4f}")

    # Step 2: Drive cvc_short_i_u to MASTERY_COMPLETED while cvc_short_a is
    # clean (no substitution errors in last-5 window), so cvc_short_i_u is
    # unlocked and can be completed.  Record it in the served log.
    _mastery_to(s1, "cvc_short_i_u", MASTERY_COMPLETED, session_id="session_1")
    m_iu_pre = s1.get_state(CHILD_ID).mastery.get("cvc_short_i_u", 0.0)
    print(
        f"[S1] cvc_short_i_u mastery (completed):  {m_iu_pre:.4f} "
        f"(>= {MASTERY_COMPLETED} = MASTERY_COMPLETED)"
    )
    s1.record_served(CHILD_ID, "cvc_short_i_u", BASE_TS + timedelta(minutes=5), "new")

    # Step 3: Inject ONE substitution-tagged failure on cvc_short_a at the END
    # of session 1 — this stays in the k=5 window when session 2 opens.
    s1.record_observation(
        child_id=CHILD_ID,
        skill="cvc_short_a",
        correct=False,
        confidence=1.0,
        session_id="session_1",
        ts=BASE_TS + timedelta(minutes=10),
        miscue_class="substitution",
    )

    state_s1 = s1.get_state(CHILD_ID)
    m_a_post = state_s1.mastery.get("cvc_short_a", 0.0)
    print(
        f"[S1] cvc_short_a mastery (post-failure): {m_a_post:.4f} "
        f"(>= {MASTERY_THRESHOLD} mastery_min? {m_a_post >= MASTERY_THRESHOLD})"
    )
    assert m_a_post >= MASTERY_THRESHOLD, (
        f"Demo staging error: one failure dropped mastery to {m_a_post:.4f} < "
        f"{MASTERY_THRESHOLD}.  Start from a higher mastery (use >= 0.98)."
    )

    # CLOSE session 1 — simulate end of tutoring session
    mastery_before_close = state_s1.mastery.copy()
    print("\n[S1] Closing store.")
    s1.close()

    # ── SESSION 2 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SESSION 2 — reopened from same SQLite file")
    print("=" * 60)

    s2 = SqliteLearnerStore(db_path)
    state_s2 = s2.get_state(CHILD_ID)

    # Claim (a): Mastery carried over
    m_a_s1 = mastery_before_close.get("cvc_short_a", 0.0)
    m_a_s2 = state_s2.mastery.get("cvc_short_a", 0.0)
    print("\n(a) Mastery carried over:")
    print(
        f"    cvc_short_a  S1={m_a_s1:.4f}  S2={m_a_s2:.4f}  "
        f"match={abs(m_a_s1 - m_a_s2) < 1e-9}"
    )
    assert abs(m_a_s1 - m_a_s2) < 1e-9, "FAIL: mastery did not carry over"

    # Claim (b): Due reviews present (failed obs → card due soon)
    due_s2 = s2.due_reviews(CHILD_ID, datetime(2099, 1, 1, tzinfo=timezone.utc))
    print(f"\n(b) Due reviews in session 2 (far-future cutoff): {sorted(due_s2)}")
    assert len(due_s2) > 0, "FAIL: no due reviews in session 2"

    # Claim (c): CLASS gate blocks cvc_short_i_u independently of mastery gate.
    # Both gate evaluations are printed explicitly for each edge.
    print("\n(c) Mastery gate vs class gate — per-edge evaluation:")
    print(
        f"    cvc_short_a mastery in session 2: "
        f"{m_a_s2:.4f} (mastery_min=0.80 -> mastery gate "
        f"{'PASSES' if m_a_s2 >= MASTERY_THRESHOLD else 'FAILS'})"
    )
    recent_obs = s2.get_last_k_observations(CHILD_ID, "cvc_short_a", k=5)
    recent_failures = [o for o in recent_obs if not o["correct"]]
    print(
        f"    Last-5 obs on cvc_short_a: {len(recent_obs)} total, "
        f"{len(recent_failures)} incorrect"
    )
    for o in recent_failures:
        print(f"      incorrect: miscue_class={o['miscue_class']!r}")

    _print_edge_eval(s2, "cvc_short_a", "cvc_short_i_u", state_s2.mastery)

    unlocked_s2 = unlocked(curriculum, CHILD_ID, s2)
    is_gated = "cvc_short_i_u" not in unlocked_s2
    print(f"    cvc_short_i_u in unlocked set: {not is_gated}")
    assert is_gated, (
        "FAIL: cvc_short_i_u should be LOCKED by the class gate; "
        "mastery gate passes but substitution error is in last-5 window"
    )

    served_log_s2 = s2.get_served_log(CHILD_ID)
    first_pick = next_item(
        curriculum, CHILD_ID, s2, served_log_s2, now=BASE_TS + timedelta(days=1)
    )
    print(f"\n[S2] Planner first pick: {first_pick}")
    if first_pick is not None:
        assert first_pick[0] != "cvc_short_i_u", (
            "FAIL: planner served gated skill cvc_short_i_u; "
            "class gate should block it even though mastery gate passes"
        )
        print(
            f"    PASS: class gate blocked cvc_short_i_u "
            f"(mastery {m_a_s2:.4f} >= 0.80 but substitution in last-5 window); "
            f"planner serves {first_pick[0]!r} instead"
        )

    # Claim (d): cvc_short_i_u was completed (mastery >= 0.95) in session 1
    # and is never re-served with reason='new' in session 2.
    print("\n(d) Never-re-serve completed nodes:")
    completed = [
        s for s, m in state_s2.mastery.items()
        if m >= MASTERY_COMPLETED
    ]
    print(f"    Completed skills: {sorted(completed)}")
    assert "cvc_short_i_u" in completed, (
        f"FAIL: cvc_short_i_u should be completed in session 2 "
        f"(mastery={state_s2.mastery.get('cvc_short_i_u', 0.0):.4f})"
    )
    # Run next_item several more times and verify no completed skill is
    # returned with reason='new'
    new_violations = []
    for step in range(5):
        sl = s2.get_served_log(CHILD_ID)
        pick = next_item(
            curriculum, CHILD_ID, s2, sl,
            now=BASE_TS + timedelta(days=1, seconds=step)
        )
        if pick is not None:
            sk, reason = pick
            if sk in completed:
                if reason != "review":
                    new_violations.append((sk, reason))
                print(f"    step {step + 1}: {sk!r} [{reason}] — completed skill re-serve OK (review)")
            else:
                print(f"    step {step + 1}: {sk!r} [{reason}]")
    assert not new_violations, (
        f"FAIL: completed skills served as 'new': {new_violations}"
    )
    served_for_completed = [
        e for e in s2.get_served_log(CHILD_ID)
        if e["skill"] == "cvc_short_i_u"
    ]
    print(
        f"    cvc_short_i_u served_log entries (from S1): "
        f"{[e['reason'] for e in served_for_completed]}"
    )
    print("    PASS: no completed skill served as 'new'")

    s2.close()
    print("\n" + "=" * 60)
    print("Demo complete — all claims verified, exit 0")
    print("=" * 60)


def main() -> None:
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        cleanup = False
    else:
        fd, db_path = tempfile.mkstemp(suffix=".db", prefix="readcoach_demo_")
        os.close(fd)
        os.unlink(db_path)  # let SqliteLearnerStore create it
        cleanup = True

    try:
        run_demo(db_path)
    finally:
        if cleanup and os.path.exists(db_path):
            os.unlink(db_path)
            # Also remove WAL/SHM files if present
            for ext in ("-wal", "-shm"):
                p = db_path + ext
                if os.path.exists(p):
                    os.unlink(p)


if __name__ == "__main__":
    main()
