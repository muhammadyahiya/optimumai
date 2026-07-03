"""LoRA — Low-Rank Adaptation, the workhorse of parameter-efficient fine-tuning.

Fine-tuning a large model by updating *every* weight is expensive: you copy the
whole checkpoint and pay full gradient/optimizer memory. LoRA freezes the
pretrained weight ``W₀`` and instead learns a tiny *low-rank* update
``ΔW = B·A`` where ``B`` is ``d_out × r``, ``A`` is ``r × d_in``, and the rank
``r`` is far smaller than ``min(d_in, d_out)``.

The trick that makes this safe: ``B`` starts at **zero** and ``A`` starts as
random Gaussian noise, so ``ΔW = B·A = 0`` at initialization. Training therefore
begins from *exactly* the pretrained model and only nudges it as ``B`` moves off
zero.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def lora_trace(d_in: int = 8, d_out: int = 8, rank: int = 2, seed: int = 0) -> Trace:
    """Build the full trace of a LoRA adapter over a frozen weight ``W₀``.

    Constructs a seeded frozen ``W₀`` plus the LoRA factors ``A`` (Gaussian) and
    ``B`` (zeros), shows that ``ΔW = B·A = 0`` at init, counts the trainable
    parameters saved, and demonstrates the forward pass ``y = (W₀ + B·A)·x``.
    """
    if rank >= min(d_in, d_out):
        raise ValueError(
            f"LoRA needs rank ≪ min(d_in, d_out); got rank={rank}, "
            f"min={min(d_in, d_out)}"
        )

    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((d_out, d_in))  # frozen pretrained weight
    A = rng.standard_normal((rank, d_in))  # trainable, random Gaussian
    B = np.zeros((d_out, rank))  # trainable, initialized to ZERO
    x = rng.standard_normal(d_in)  # a seeded input vector

    full_params = d_in * d_out
    lora_params = rank * (d_in + d_out)
    reduction = full_params / lora_params

    t = Trace(
        op="lora",
        formula="W = W₀ + BA,  B∈ℝ^{d_out×r}, A∈ℝ^{r×d_in},  r ≪ d",
        complexity="O(r·(d_in + d_out)) trainable params vs O(d_in·d_out) full",
        why_ai=[
            "Only A and B are trained; W₀ stays frozen → the base model is "
            "never overwritten",
            "Adapters are tiny checkpoints: keep many per base model and swap "
            "them at inference time",
            "The low-rank assumption is that the fine-tuning update lives in a "
            "small subspace of weight space",
        ],
        meta={
            "d_in": d_in,
            "d_out": d_out,
            "rank": rank,
            "full_params": full_params,
            "lora_params": lora_params,
            "reduction_factor": reduction,
        },
    )

    t.add(
        "Freeze the pretrained weight W₀",
        f"W₀ has shape ({d_out} × {d_in})  →\n{arr(W0)}",
        W0,
        detail="W₀ is never updated during fine-tuning — its gradients are off.",
    )
    t.add(
        "Initialize A (random Gaussian)",
        f"A ~ N(0, 1), shape ({rank} × {d_in})  →\n{arr(A)}",
        A,
        detail="A is trainable and starts as noise so the update has a direction to grow into.",
    )
    t.add(
        "Initialize B (zeros)",
        f"B = 0, shape ({d_out} × {rank})  →\n{arr(B)}",
        B,
        detail="B starts at zero — this is what forces ΔW = 0 at the very first step.",
    )

    dW = B @ A
    t.add(
        "Form the low-rank update ΔW = B·A",
        f"ΔW = B·A, shape ({d_out} × {d_in})  →\n{arr(dW)}",
        dW,
        detail="Every entry is 0 at init, so W = W₀ + ΔW = W₀ exactly.",
    )

    t.add(
        "Count full fine-tuning parameters",
        f"d_in · d_out = {d_in} · {d_out} = {full_params}",
        float(full_params),
        detail="A full fine-tune updates the entire weight matrix.",
    )
    t.add(
        "Count LoRA parameters",
        f"r · (d_in + d_out) = {rank} · ({d_in} + {d_out}) = {lora_params}",
        float(lora_params),
        detail="Only the two thin factors A and B carry gradients.",
    )
    t.add(
        "Reduction factor",
        f"{full_params} / {lora_params} = {num(reduction)}×  fewer trainable params",
        float(reduction),
        detail=(
            "On GPT-3 175B, LoRA reported ~10,000× fewer trainable parameters "
            "and ~3× less GPU memory than full fine-tuning."
        ),
    )

    y0 = (W0 + dW) @ x
    base = W0 @ x
    t.add(
        "Forward pass at init: y = (W₀ + B·A)·x",
        f"y  →  {arr(y0)}",
        y0,
        detail=f"y equals W₀·x  →  {arr(base)}  because ΔW = 0 (max |y − W₀·x| = "
        f"{num(float(np.max(np.abs(y0 - base))))}).",
    )

    # Simulate a tiny "trained" B so that ΔW ≠ 0 and the output shifts.
    B_trained = 0.1 * rng.standard_normal((d_out, rank))
    dW_trained = B_trained @ A
    y_trained = (W0 + dW_trained) @ x
    t.add(
        "After a little training: nudge B off zero",
        f"y_trained  →  {arr(y_trained)}",
        y_trained,
        detail=(
            "With B ≠ 0 we get ΔW ≠ 0, so the output shifts from the base model "
            f"(max |Δy| = {num(float(np.max(np.abs(y_trained - base))))}) — the "
            "adapter is now doing work."
        ),
    )

    t.result = float(reduction)
    return t


def lora(
    d_in: int = 8,
    d_out: int = 8,
    rank: int = 2,
    seed: int = 0,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> float:
    """Return the LoRA parameter reduction factor. ``explain=True`` prints the trace."""
    t = lora_trace(d_in=d_in, d_out=d_out, rank=rank, seed=seed)
    return t.render(level) if explain else t.result


def demo() -> Trace:
    """Return a ready-to-render LoRA trace with default settings."""
    return lora_trace()
