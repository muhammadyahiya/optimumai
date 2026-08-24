"""The prerequisite graph, as data rather than prose.

Prerequisites used to be free text -- ``["vector algebra", "algebra"]`` -- which
reads well and enforces nothing. Here they are edges between real concept ids,
so the graph can be *checked* (no dangling edges, no cycles), *ordered*
(topological sort), and *enforced* (a concept whose prerequisites are unmet is
locked).

Human-readable background stays on the concept as ``assumed_knowledge``; it was
never the same thing as a prerequisite edge and conflating the two is what made
the graph unenforceable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol


class _HasPrereqs(Protocol):
    concept_id: str
    prerequisites: list[str]


def validate_dag(concepts: Mapping[str, _HasPrereqs]) -> list[str]:
    """Return a list of problems with the graph; empty means it is sound.

    Checks, in order: an edge pointing at a concept that does not exist, a
    concept listing itself, and any cycle. Returning problems rather than
    raising lets a caller report all of them at once.
    """
    problems: list[str] = []

    for cid, concept in concepts.items():
        if cid != concept.concept_id:
            problems.append(f"{cid!r} is registered under a different concept_id")
        for prereq in concept.prerequisites:
            if prereq == cid:
                problems.append(f"{cid!r} lists itself as a prerequisite")
            elif prereq not in concepts:
                problems.append(f"{cid!r} requires unknown concept {prereq!r}")

    problems.extend(_find_cycles(concepts))
    return problems


def _find_cycles(concepts: Mapping[str, _HasPrereqs]) -> list[str]:
    """Depth-first cycle detection, reporting the offending path."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(concepts, WHITE)
    found: list[str] = []

    def visit(node: str, path: list[str]) -> None:
        colour[node] = GREY
        for prereq in concepts[node].prerequisites:
            if prereq not in concepts:
                continue  # already reported as dangling
            if colour[prereq] == GREY:
                cycle = path[path.index(prereq):] if prereq in path else [prereq]
                found.append("cycle: " + " -> ".join([*cycle, prereq]))
            elif colour[prereq] == WHITE:
                visit(prereq, [*path, prereq])
        colour[node] = BLACK

    for node in sorted(concepts):
        if colour[node] == WHITE:
            visit(node, [node])
    return found


def learning_order(concepts: Mapping[str, _HasPrereqs]) -> list[str]:
    """Topologically sort the concepts, prerequisites first.

    Ties break alphabetically so the order is stable across runs -- a wobbling
    sidebar is a bug, not a cosmetic issue.

    Raises:
        ValueError: if the graph is unsound, since no valid order exists.
    """
    problems = validate_dag(concepts)
    if problems:
        raise ValueError("prerequisite graph is invalid: " + "; ".join(problems))

    remaining = {cid: set(c.prerequisites) for cid, c in concepts.items()}
    order: list[str] = []
    while remaining:
        ready = sorted(cid for cid, deps in remaining.items() if not deps)
        if not ready:  # pragma: no cover - validate_dag already rejects cycles
            raise ValueError("prerequisite graph is invalid: cycle detected")
        for cid in ready:
            order.append(cid)
            del remaining[cid]
        for deps in remaining.values():
            deps.difference_update(ready)
    return order


def unmet_prerequisites(
    concept: _HasPrereqs, completed: Iterable[str]
) -> list[str]:
    """Prerequisites of ``concept`` that are not in ``completed``."""
    done = set(completed)
    return [p for p in concept.prerequisites if p not in done]


def is_locked(concept: _HasPrereqs, completed: Iterable[str]) -> bool:
    """True when at least one prerequisite is still outstanding."""
    return bool(unmet_prerequisites(concept, completed))
