"""Expected Calibration Error (ECE) — does confidence mean what it says?

A model is *calibrated* when its stated confidence matches its empirical
accuracy: among all the times it says "I'm 90% sure," it should be right
about 90% of the time. A model can be highly accurate yet badly calibrated
(always shouting 99% confidence while being wrong 20% of the time), which
matters enormously wherever a downstream system *acts* on the confidence
number — deciding when to abstain, when to ask a human, when to trust a
retrieved fact, or when to stop generating.

The reliability-diagram recipe
--------------------------------
1. Bucket every prediction into ``M`` equal-width confidence bins, e.g.
   ``[0.0, 0.2), [0.2, 0.4), ..., [0.8, 1.0]``.
2. Within each bin, compute the *mean confidence* (average of the stated
   probabilities) and the *accuracy* (fraction actually correct).
3. A perfectly calibrated model has confidence == accuracy in every bin —
   plotting one against the other traces the diagonal, hence "reliability
   diagram."
4. **ECE** collapses the whole diagram into one number: the accuracy-weighted
   average gap between confidence and accuracy across bins,

       ECE = Σ_b (n_b / N) · |confidence_b − accuracy_b|

   where ``n_b`` is how many predictions landed in bin ``b`` and ``N`` is the
   total. Weighting by bin size means a bin with one lonely prediction can't
   dominate the score the way an unweighted average of gaps would.

What ECE does NOT capture
----------------------------
* It only checks calibration *on average within each bin* — a bin can have
  confidence matching accuracy in aggregate while individual predictions are
  wildly over- or under-confident in opposite directions that cancel out.
* It is sensitive to bin count/width: too few bins hide miscalibration inside
  wide buckets; too many bins make each bin's accuracy a noisy estimate from
  few samples. (MCE — Maximum Calibration Error, the worst single-bin gap —
  is a common complement that reports the worst case instead of the average.)
* Calibration says nothing about *sharpness* (how far confidences spread from
  the base rate) or accuracy itself: a model that always predicts the base
  rate with 100% "confidence" in that base rate can be perfectly calibrated
  and useless.
* For LLMs specifically, "confidence" is itself ambiguous — it could mean the
  softmax probability of the sampled token, a verbalized "I'm 80% sure," or
  something derived from sampling consistency — and these frequently
  disagree with each other.
"""

from __future__ import annotations

from collections.abc import Sequence

from optimumai.core._fmt import num
from optimumai.core.trace import Trace


def _validate(confidences: Sequence[float], correct: Sequence[bool], n_bins: int) -> None:
    if len(confidences) != len(correct):
        raise ValueError(
            f"confidences and correct must have the same length, got "
            f"{len(confidences)} and {len(correct)}"
        )
    if not confidences:
        raise ValueError("confidences must contain at least one prediction")
    for c in confidences:
        if not (0.0 <= c <= 1.0):
            raise ValueError(f"each confidence must be in [0, 1], got {c}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")


def _bin_index(confidence: float, n_bins: int) -> int:
    """Map a confidence in [0, 1] to a bin in [0, n_bins - 1] (last bin is inclusive of 1.0)."""
    return min(int(confidence * n_bins), n_bins - 1)


def ece_trace(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 5
) -> Trace:
    """Build the full ECE trace: bin the predictions, then show each bin's confidence-vs-accuracy
    gap.

    Args:
        confidences: The model's stated confidence for each prediction, in ``[0, 1]``.
        correct: Whether each prediction was actually correct.
        n_bins: Number of equal-width reliability bins (default 5, i.e. 20%-wide bins).
    """
    _validate(confidences, correct, n_bins)
    n = len(confidences)
    edges = [i / n_bins for i in range(n_bins + 1)]

    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct, strict=True):
        bins[_bin_index(c, n_bins)].append((c, ok))

    t = Trace(
        op="ece",
        formula="ECE = Σ_b (n_b/N) · |mean_confidence_b − accuracy_b|",
        complexity=f"O(N + M) for N={n} predictions, M={n_bins} bins",
        why_ai=[
            "Confidence calibration governs when a system should abstain, escalate to "
            "a human, or trust a retrieved/generated claim",
            "A model can be accurate but badly calibrated (overconfident) or the reverse",
            "Temperature scaling — dividing logits by a learned T before softmax — is "
            "the standard post-hoc fix once ECE reveals miscalibration",
        ],
        meta={"n_predictions": n, "n_bins": n_bins, "bin_edges": edges},
    )
    t.add(
        "Bin predictions by confidence",
        f"{n_bins} equal-width bins over [0, 1]: edges = {[num(e) for e in edges]}",
        detail="Each prediction falls into exactly one bin based on its stated confidence.",
    )

    weighted_gap_sum = 0.0
    bin_rows = []
    for b, items in enumerate(bins):
        lo, hi = edges[b], edges[b + 1]
        if not items:
            t.add(
                f"Bin {b}  [{num(lo)}, {num(hi)})",
                "empty — no predictions in this confidence range",
                detail="Empty bins contribute 0 to ECE (weight n_b/N = 0).",
            )
            continue
        confs = [c for c, _ in items]
        oks = [1.0 if ok else 0.0 for _, ok in items]
        mean_conf = sum(confs) / len(confs)
        accuracy = sum(oks) / len(oks)
        gap = abs(mean_conf - accuracy)
        weight = len(items) / n
        weighted_gap_sum += weight * gap
        bin_rows.append(
            {"bin": b, "range": (lo, hi), "n": len(items), "mean_confidence": mean_conf,
             "accuracy": accuracy, "gap": gap, "weight": weight}
        )
        t.add(
            f"Bin {b}  [{num(lo)}, {num(hi)})",
            f"n={len(items)}, mean confidence={num(mean_conf)}, accuracy={num(accuracy)}, "
            f"gap=|{num(mean_conf)} − {num(accuracy)}|={num(gap)}, weight={num(weight)}",
            gap,
            detail="A well-calibrated bin has mean confidence close to accuracy — gap ≈ 0.",
        )

    t.add(
        "Expected Calibration Error",
        f"ECE = Σ (weight × gap) = {num(weighted_gap_sum)}",
        weighted_gap_sum,
        detail="0 = perfectly calibrated. There is no universal 'good' threshold, but ECE "
        "above ~0.1 usually signals a confidence number not worth trusting as-is.",
    )
    t.meta["bins"] = bin_rows
    t.result = weighted_gap_sum
    return t


def ece(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 5) -> float:
    """Return the Expected Calibration Error in ``[0, 1]`` (lower is better calibrated)."""
    return ece_trace(confidences, correct, n_bins=n_bins).result


def demo(seed: int = 0) -> Trace:
    """ECE on ten toy predictions: an overconfident-in-the-middle, mostly-sane classifier."""
    confidences = [0.9, 0.85, 0.95, 0.9, 0.65, 0.7, 0.6, 0.3, 0.4, 0.35]
    correct = [True, True, False, True, True, False, True, False, False, True]
    return ece_trace(confidences, correct, n_bins=5)
