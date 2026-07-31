"""Loop state — why the ledger must live outside the model.

A loop's memory can't live only in a context window: the process restarts, the
context compacts, and you still need to know the running approval rate and spend.
The fix is an append-only run log (JSONL) on disk. This lesson replays such a log
and computes the running approval rate and mean tokens per run — statistics you
simply cannot recover without persisted state.
"""

from __future__ import annotations

from collections.abc import Iterable

from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def loop_state_trace(run_log: Iterable[dict]) -> Trace:
    """Trace running approval-rate + mean-token statistics over a persisted run log."""
    entries = list(run_log)
    if not entries:
        raise ValueError("run_log must be non-empty")

    t = Trace(
        op="loop_state",
        formula="approval_rate = approvals / runs;  mean_tokens = Σ tokens / runs",
        complexity="O(runs)",
        why_ai=[
            "State survives process restarts and context compaction",
            "The append-only run log is the loop's audit trail",
            "Running metrics (approval rate, spend) require history, not just the last turn",
        ],
        meta={"runs": len(entries)},
    )

    approvals = 0
    total_tokens = 0.0
    for i, e in enumerate(entries, start=1):
        verdict = str(e.get("verdict", "")).upper()
        tokens = float(e.get("tokens", 0))
        approvals += 1 if verdict == "APPROVE" else 0
        total_tokens += tokens
        t.add(
            f"Replay run {i}",
            f"{verdict or '—'} · {num(tokens)} tok → "
            f"running approval {approvals}/{i} = {num(approvals / i)}",
            approvals / i,
        )

    n = len(entries)
    approval_rate = approvals / n
    mean_tokens = total_tokens / n
    t.add(
        "Final statistics",
        f"approval_rate = {approvals}/{n} = {num(approval_rate)}; mean_tokens = {num(mean_tokens)}",
        approval_rate,
        detail="Recoverable only because every iteration was persisted to the log.",
    )
    t.result = approval_rate
    return t


def loop_state(
    run_log: Iterable[dict],
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> float:
    """Return the overall approval rate from a run log. Set ``explain=True`` to print."""
    t = loop_state_trace(run_log)
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """Five persisted runs — three approvals, two escalations — replayed from the log."""
    log = [
        {"verdict": "APPROVE", "tokens": 11000},
        {"verdict": "REJECT", "tokens": 15000},
        {"verdict": "APPROVE", "tokens": 9000},
        {"verdict": "APPROVE", "tokens": 12000},
        {"verdict": "REJECT", "tokens": 16000},
    ]
    return loop_state_trace(log)
