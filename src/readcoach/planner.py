"""T4.2 — Quest planner: mastery-gated DAG traversal with typed prerequisite edges.

PREREQUISITE EDGE SEMANTICS
───────────────────────────
An edge (A → B, classes=[c1, c2], mastery_min=0.80) means B is unlocked only
when ALL of:
  (a) mastery[A] >= mastery_min  (BKT posterior from the store)
  (b) the last-k (k=5) observations for skill A contain NO incorrect
      observation whose miscue_class is in the edge's classes list.

An observation with miscue_class=None is "generic" and never triggers any
class-keyed gate — absence of a tag means the error type was not identified
and must not be conflated with a specific phonics breakdown.

MASTERY CONSTANTS
─────────────────
MASTERY_THRESHOLD  = 0.80  — minimum mastery to satisfy a prerequisite.
MASTERY_COMPLETED  = 0.95  — skill is "done"; never re-served as "new".

SERVED LOG
──────────
next_item records each served item (skill, ts, reason) via store.record_served.
The served log is used to detect completed-but-not-due skills (never-re-serve
guarantee).  reason ∈ {"new", "review"}.

NEVER-RE-SERVE GUARANTEE
────────────────────────
A completed skill (mastery >= MASTERY_COMPLETED) is returned by next_item ONLY
when it appears on the FSRS due list (store.due_reviews), and then only with
reason="review".  Without review intent it is skipped entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants (documented in module docstring)
# ---------------------------------------------------------------------------

MASTERY_THRESHOLD: float = 0.80
MASTERY_COMPLETED: float = 0.95
_LAST_K: int = 5  # observation window for miscue-class gate

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CurriculumError(ValueError):
    """Base class for curriculum structure errors."""


class CyclicCurriculumError(CurriculumError):
    """Raised when the curriculum graph contains a cycle."""


class UnknownPrerequisiteError(CurriculumError):
    """Raised when a prerequisite references a skill id that does not exist."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrerequisiteEdge:
    """One prerequisite edge in the curriculum DAG.

    Attributes:
        skill:       id of the required upstream skill.
        mastery_min: minimum BKT posterior (default MASTERY_THRESHOLD).
        classes:     miscue classes that block advancement when present in
                     the last-k incorrect observations on the prerequisite.
    """

    skill: str
    mastery_min: float
    classes: tuple[str, ...]  # frozenset-like but ordered for determinism


@dataclass
class CurriculumNode:
    """One node in the curriculum DAG."""

    id: str
    band: int
    label: str
    description: str
    prerequisites: list[PrerequisiteEdge] = field(default_factory=list)


@dataclass
class Curriculum:
    """Validated curriculum DAG.

    nodes: dict[skill_id -> CurriculumNode]
    topological_order: list[skill_id] in dependency order (roots first).
    """

    nodes: dict[str, CurriculumNode]
    topological_order: list[str]


# ---------------------------------------------------------------------------
# Load and validate curriculum
# ---------------------------------------------------------------------------


def _parse_node(raw: dict[str, Any]) -> CurriculumNode:
    prereqs = []
    for p in raw.get("prerequisites", []):
        prereqs.append(
            PrerequisiteEdge(
                skill=p["skill"],
                mastery_min=float(p.get("mastery_min", MASTERY_THRESHOLD)),
                classes=tuple(p.get("classes", [])),
            )
        )
    return CurriculumNode(
        id=raw["id"],
        band=int(raw["band"]),
        label=raw["label"],
        description=raw["description"],
        prerequisites=prereqs,
    )


def _toposort(nodes: dict[str, CurriculumNode]) -> list[str]:
    """Kahn's algorithm.  Raises CyclicCurriculumError if a cycle exists."""
    # Build adjacency and in-degree
    in_degree: dict[str, int] = {nid: 0 for nid in nodes}
    adjacency: dict[str, list[str]] = {nid: [] for nid in nodes}

    for nid, node in nodes.items():
        for edge in node.prerequisites:
            adjacency[edge.skill].append(nid)
            in_degree[nid] += 1

    queue = sorted([nid for nid, deg in in_degree.items() if deg == 0])
    result = []
    while queue:
        current = queue.pop(0)
        result.append(current)
        for successor in sorted(adjacency[current]):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    if len(result) != len(nodes):
        cyclic = sorted(set(nodes) - set(result))
        raise CyclicCurriculumError(
            f"Curriculum contains a cycle involving: {cyclic}"
        )
    return result


def load_curriculum(path: str | Path) -> Curriculum:
    """Parse and validate a curriculum YAML file.

    Raises:
        CyclicCurriculumError:       if the prerequisite graph has a cycle.
        UnknownPrerequisiteError:    if a prerequisite references a missing id.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    nodes: dict[str, CurriculumNode] = {}
    for raw_node in raw["nodes"]:
        node = _parse_node(raw_node)
        nodes[node.id] = node

    # Validate: all prerequisite skill ids must exist
    for nid, node in nodes.items():
        for edge in node.prerequisites:
            if edge.skill not in nodes:
                raise UnknownPrerequisiteError(
                    f"Node {nid!r} references unknown prerequisite {edge.skill!r}."
                )

    # Topological sort (detects cycles)
    order = _toposort(nodes)

    return Curriculum(nodes=nodes, topological_order=order)


# ---------------------------------------------------------------------------
# Unlock logic
# ---------------------------------------------------------------------------


def _edge_is_satisfied(
    edge: PrerequisiteEdge,
    child_id: str,
    store: Any,
    mastery: dict[str, float],
) -> bool:
    """Return True iff a single prerequisite edge is satisfied.

    Conditions:
      (a) mastery[edge.skill] >= edge.mastery_min
      (b) last-k observations for child/edge.skill contain no incorrect
          observation whose miscue_class is in edge.classes.
    """
    # (a) mastery gate
    current_mastery = mastery.get(edge.skill, 0.0)
    if current_mastery < edge.mastery_min:
        return False

    # (b) miscue-class gate — only applies when edge has class restrictions
    if edge.classes:
        recent = store.get_last_k_observations(child_id, edge.skill, k=_LAST_K)
        for obs in recent:
            if not obs["correct"] and obs["miscue_class"] in edge.classes:
                return False

    return True


def unlocked(curriculum: Curriculum, child_id: str, store: Any) -> list[str]:
    """Return list of skill ids whose prerequisites are all satisfied.

    Skills with no prerequisites are always unlocked.
    Skills whose every prerequisite edge passes _edge_is_satisfied are unlocked.

    The mastery dict is fetched once from the store for efficiency.
    """
    state = store.get_state(child_id)
    mastery = state.mastery

    result = []
    for skill_id, node in curriculum.nodes.items():
        if not node.prerequisites:
            result.append(skill_id)
            continue
        if all(
            _edge_is_satisfied(edge, child_id, store, mastery)
            for edge in node.prerequisites
        ):
            result.append(skill_id)
    return result


# ---------------------------------------------------------------------------
# next_item
# ---------------------------------------------------------------------------


def next_item(
    curriculum: Curriculum,
    child_id: str,
    store: Any,
    served_log: list[dict],
    now: datetime | None = None,
) -> tuple[str, str] | None:
    """Select the next skill to serve.

    Algorithm:
      1. Fetch unlocked skills.
      2. Fetch FSRS due list (these may override completion status).
      3. For each skill in unlocked:
         - If mastery >= MASTERY_COMPLETED AND skill is NOT due for review:
           skip (never-re-serve guarantee).
         - If mastery >= MASTERY_COMPLETED AND skill IS due for review:
           candidate with reason="review".
         - If mastery < MASTERY_COMPLETED:
           candidate with reason="new".
      4. Among "new" candidates: pick argmax mastery-gap
         (= lowest mastery = furthest from MASTERY_COMPLETED).
      5. If no "new" candidates: return the first "review" candidate (lowest
         mastery), or None if nothing is available.

    Returns:
        (skill_id, reason) where reason ∈ {"new", "review"}, or None.

    Side-effect:
        Calls store.record_served(child_id, skill_id, now, reason) on success.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Build a set of already-served skills (to track never-re-serve);
    # this is used only for the record — actual decisions use mastery+due.
    # The served_log passed in may come from store.get_served_log() or be
    # empty for the first session.

    state = store.get_state(child_id)
    mastery = state.mastery
    due_set = set(store.due_reviews(child_id, now))
    unlocked_ids = set(unlocked(curriculum, child_id, store))

    new_candidates: list[tuple[float, str]] = []   # (mastery, skill_id)
    review_candidates: list[tuple[float, str]] = []  # (mastery, skill_id)

    for skill_id in unlocked_ids:
        m = mastery.get(skill_id, 0.0)
        if m >= MASTERY_COMPLETED:
            if skill_id in due_set:
                review_candidates.append((m, skill_id))
            # else: skip — never-re-serve guarantee
        else:
            new_candidates.append((m, skill_id))

    # Among new candidates: argmax mastery-gap = argmin mastery
    if new_candidates:
        # Sort by mastery ascending (most urgent first), then by id for
        # determinism across backends.
        new_candidates.sort(key=lambda t: (t[0], t[1]))
        chosen_mastery, chosen_skill = new_candidates[0]
        reason = "new"
    elif review_candidates:
        review_candidates.sort(key=lambda t: (t[0], t[1]))
        chosen_mastery, chosen_skill = review_candidates[0]
        reason = "review"
    else:
        return None

    store.record_served(child_id, chosen_skill, now, reason)
    return (chosen_skill, reason)
