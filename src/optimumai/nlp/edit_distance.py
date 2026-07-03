"""Levenshtein edit distance — the DP table behind spell-check and fuzzy match.

**Intuition.** How many single-character edits (insert, delete, substitute) turn
one string into another? "kitten" → "sitting" takes 3: swap k→s, swap e→i,
insert g. Trying every possible sequence of edits is exponential, but the
problem has *optimal substructure*: the cheapest way to align the first ``i``
characters of ``a`` with the first ``j`` characters of ``b`` only depends on
three smaller subproblems, so a dynamic-programming table fills it in
``O(m·n)``.

**Math.** Let ``a`` have length ``m`` and ``b`` have length ``n``. Build a
``(m+1)×(n+1)`` table ``D`` where ``D[i][j]`` is the edit distance between
``a[:i]`` and ``b[:j]``. The base cases are the cost of turning an empty
string into a prefix (or vice versa) by pure insertion/deletion:

    D[i][0] = i                       (delete all i characters of a)
    D[0][j] = j                       (insert all j characters of b)

The recurrence compares the next pair of characters. If they match, no edit is
needed and we inherit the diagonal; otherwise we pay 1 for the best of
substitute / delete / insert:

    D[i][j] = D[i-1][j-1]                                  if a[i-1] == b[j-1]
    D[i][j] = 1 + min(D[i-1][j-1], D[i-1][j], D[i][j-1])   otherwise
              (substitute)   (delete from a)  (insert into b)

``D[m][n]`` is the answer. Walking backward from the bottom-right corner,
always stepping toward whichever neighbor produced the current cell, recovers
the actual edit *operations* (a backtrace) — not just their count.

**Why modern LLMs still rest on this.** Edit distance is the yardstick behind
spell-checkers, OCR/ASR post-correction, and fuzzy deduplication of training
data. Its recurrence is also the direct ancestor of *sequence alignment* used
in bioinformatics, and the same "fill a table by combining smaller
subproblems" pattern reappears as CTC alignment for speech-to-text and as the
dynamic-programming decoders used before beam search became standard.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.trace import Trace


def _dp_table(a: str, b: str) -> np.ndarray:
    """Fill the ``(len(a)+1) x (len(b)+1)`` Levenshtein DP table."""
    m, n = len(a), len(b)
    d = np.zeros((m + 1, n + 1), dtype=int)
    d[:, 0] = np.arange(m + 1)
    d[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j - 1], d[i - 1, j], d[i, j - 1])
    return d


def _backtrace(a: str, b: str, d: np.ndarray) -> list[tuple[str, str, str]]:
    """Walk from ``D[m][n]`` back to ``D[0][0]``, recovering the edit ops.

    Returns a list of ``(op, from_char, to_char)`` triples in left-to-right
    (forward) order, where ``op`` is one of ``"match"``, ``"substitute"``,
    ``"delete"``, or ``"insert"``.
    """
    i, j = len(a), len(b)
    ops: list[tuple[str, str, str]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and d[i, j] == d[i - 1, j - 1]:
            ops.append(("match", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + 1:
            ops.append(("substitute", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("delete", a[i - 1], "-"))
            i -= 1
        else:
            ops.append(("insert", "-", b[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def edit_distance_trace(a: str, b: str) -> Trace:
    """Build the full DP-table + backtrace trace for the Levenshtein distance."""
    if not isinstance(a, str) or not isinstance(b, str):
        raise TypeError(f"a and b must be strings, got {type(a).__name__} and {type(b).__name__}")

    t = Trace(
        op="edit_distance",
        formula=(
            "D[i][j] = D[i-1][j-1] if a[i]=b[j] else "
            "1 + min(D[i-1][j-1], D[i-1][j], D[i][j-1])"
        ),
        complexity="O(m·n) time and space for strings of length m, n",
        why_ai=[
            "Spell-checkers and search-query correction rank candidates by edit distance",
            "OCR and ASR pipelines use it to score a hypothesis against ground truth (WER/CER)",
            "The same 'combine smaller subproblems in a table' pattern underlies CTC alignment "
            "and DP-based sequence decoders",
        ],
        meta={"a": a, "b": b, "len_a": len(a), "len_b": len(b)},
    )

    d = _dp_table(a, b)
    t.add(
        "Base cases: empty-string prefixes",
        f"D[i][0] = i for i in 0..{len(a)}; D[0][j] = j for j in 0..{len(b)}",
        d[:, 0].copy(),
        detail="Turning an empty string into a prefix of length k costs k insertions "
        "(and symmetrically for deletions).",
    )
    t.add(
        f"Fill the {d.shape[0]}x{d.shape[1]} DP table",
        f"rows = '' + {list(a)}, cols = '' + {list(b)}\n{arr(d)}",
        d,
        detail="Each cell either inherits the diagonal for free (chars match) or pays 1 "
        "for the cheapest of substitute/delete/insert.",
    )

    distance = int(d[len(a), len(b)])
    t.add(
        "Read off the distance",
        f"D[{len(a)}][{len(b)}] = {num(distance)}",
        distance,
        detail="The bottom-right corner is the edit distance between the full strings.",
    )

    ops = _backtrace(a, b, d)
    op_str = " → ".join(
        f"{op}({frm}->{to})" if op != "match" else frm for op, frm, to in ops
    )
    t.add(
        "Backtrace the edit script",
        op_str,
        ops,
        detail="Walk from the bottom-right cell to the top-left, always stepping toward "
        "whichever neighbor produced the current value.",
    )

    t.result = distance
    t.meta["ops"] = ops
    return t


def edit_distance(a: str, b: str) -> int:
    """Return the Levenshtein distance between ``a`` and ``b``."""
    return edit_distance_trace(a, b).result


def edit_script(a: str, b: str) -> list[tuple[str, str, str]]:
    """Return the backtraced edit script turning ``a`` into ``b``.

    Each element is ``(op, from_char, to_char)`` with ``op`` in
    ``{"match", "substitute", "delete", "insert"}``; ``"-"`` marks the absent
    side of an insert/delete.
    """
    return edit_distance_trace(a, b).meta["ops"]


def demo(seed: int = 0) -> Trace:
    """The textbook 'kitten' → 'sitting' example (distance 3)."""
    return edit_distance_trace("kitten", "sitting")
