"""Loop memory — the compaction ratio behind rolling cross-iteration memory.

A loop must recall prior iterations (what the checker rejected, the last approved
artifact) without dragging the whole history into every prompt. The compacting
strategy keeps the newest ``k`` iterations verbatim and collapses the rest to a
one-line digest. This lesson computes the compressed size and the compression
ratio, and checks it against the context window.
"""

from __future__ import annotations

from collections.abc import Iterable

from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def loop_memory_trace(
    iteration_tokens: Iterable[float],
    keep_verbatim: int = 3,
    digest_tokens: float = 40.0,
    context_window: float = 8000.0,
) -> Trace:
    """Trace compaction of a run history to a bounded rolling memory."""
    tokens = [float(x) for x in iteration_tokens]
    if not tokens:
        raise ValueError("iteration_tokens must be non-empty")
    if keep_verbatim < 0:
        raise ValueError(f"keep_verbatim must be ≥ 0, got {keep_verbatim}")
    if digest_tokens < 0 or context_window <= 0:
        raise ValueError("digest_tokens must be ≥ 0 and context_window > 0")

    n = len(tokens)
    full = sum(tokens)
    t = Trace(
        op="loop_memory",
        formula="compacted = Σ newest_k tokens + (n − k) · digest;  ratio = compacted / full",
        complexity="O(iterations)",
        why_ai=[
            "Enough context to continue, small enough to fit the context window",
            "Rolling summary keeps recent detail and collapses old iterations",
            "Bounds prompt growth so a long-running loop stays affordable",
        ],
        meta={"iterations": n, "keep_verbatim": keep_verbatim, "context_window": context_window},
    )

    t.add("Full history size", f"Σ tokens over {n} iterations = {num(full)}", full)

    k = min(keep_verbatim, n)
    verbatim = sum(tokens[-k:]) if k else 0.0
    t.add(
        f"Keep newest {k} verbatim",
        f"Σ last {k} = {num(verbatim)} tokens",
        verbatim,
        detail="Recent rejections/feedback matter most to the next attempt.",
    )

    digested = (n - k) * digest_tokens
    t.add(
        f"Digest the older {n - k}",
        f"({n} − {k}) · {num(digest_tokens)} = {num(digested)} tokens",
        digested,
    )

    compacted = verbatim + digested
    ratio = compacted / full if full > 0 else 0.0
    t.add(
        "Compaction ratio",
        f"{num(compacted)} / {num(full)} = {num(ratio)}",
        ratio,
        detail=f"Fits context window ({num(context_window)}): {compacted <= context_window}.",
    )
    t.result = ratio
    return t


def loop_memory(
    iteration_tokens: Iterable[float],
    keep_verbatim: int = 3,
    digest_tokens: float = 40.0,
    context_window: float = 8000.0,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> float:
    """Return the compaction ratio (compacted / full). Set ``explain=True`` to print."""
    t = loop_memory_trace(
        iteration_tokens, keep_verbatim=keep_verbatim,
        digest_tokens=digest_tokens, context_window=context_window,
    )
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """Ten iterations of ~1.2k tokens each, compacted to the newest 3 + digests."""
    tokens = [1200, 1300, 1100, 1250, 1400, 1150, 1350, 1200, 1300, 1250]
    return loop_memory_trace(tokens, keep_verbatim=3, digest_tokens=40.0, context_window=8000.0)
