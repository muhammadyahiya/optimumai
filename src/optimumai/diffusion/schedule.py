"""DDPM forward diffusion — how image generators turn data into noise, and back.

A diffusion model is trained on a simple game: take clean data ``x₀``, corrupt
it into pure noise over ``T`` steps, and learn to undo one step at a time. The
forward (noising) process is fixed, not learned. Its magic is a closed form —
you can jump straight to any timestep ``t`` without simulating the ones before:

    xₜ = √(ᾱₜ)·x₀ + √(1 − ᾱₜ)·ε ,   ε ~ 𝒩(0, I)

where ``β`` is a schedule of tiny variances, ``α = 1 − β``, and ``ᾱₜ`` is their
cumulative product. This module builds a linear ``β`` schedule and shows the
noising at a few timesteps, then describes the reverse direction the model
actually learns.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def forward_diffusion_trace(
    x0: np.ndarray,
    timesteps: int = 10,
    beta_start: float = 1e-4,
    beta_end: float = 0.2,
    seed: int = 0,
) -> Trace:
    """Build the full trace of DDPM forward diffusion applied to ``x0``."""
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError(f"forward_diffusion_trace expects a 1-D signal, got shape {x0.shape}")
    if timesteps < 1:
        raise ValueError(f"timesteps must be >= 1, got {timesteps}")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError(
            f"need 0 < beta_start < beta_end < 1, got {beta_start} and {beta_end}"
        )

    betas = np.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    rng = np.random.default_rng(seed)

    t = Trace(
        op="forward_diffusion",
        formula="xₜ = √(ᾱₜ)·x₀ + √(1 − ᾱₜ)·ε ,   ᾱₜ = Πₛ₌₁ᵗ (1 − βₛ)",
        complexity="O(T + n) — build the schedule once, then one blend per timestep",
        why_ai=[
            "Diffusion models (Stable Diffusion, DALL·E, Imagen) learn to REVERSE noising",
            "The closed form lets training sample any timestep t directly, no simulation",
            "The network's job is to predict the noise ε that was added at step t",
        ],
        meta={
            "timesteps": timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "seed": seed,
            "x0_shape": x0.shape,
        },
    )
    t.add(
        "Clean signal x₀",
        f"{arr(x0)}",
        x0,
        detail="The data we will gradually destroy — an image, audio clip, or here a 1-D signal.",
    )
    t.add(
        "Linear β schedule",
        f"β = linspace({num(beta_start)}, {num(beta_end)}, {timesteps})\n{arr(betas)}",
        betas,
        detail="Each βₜ is the tiny variance of Gaussian noise injected at step t.",
    )
    t.add(
        "Cumulative ᾱₜ = Π(1 − βₛ)",
        f"α = 1 − β,   ᾱ = cumprod(α)\n{arr(alpha_bars)}",
        alpha_bars,
        detail="ᾱₜ decays from ~1 (clean) toward 0 (pure noise) as t grows.",
    )

    # Show the closed-form noising at a handful of timesteps spanning 1..T.
    sample_ts = sorted(set(np.linspace(1, timesteps, min(4, timesteps), dtype=int)))
    x_t = x0
    for step in sample_ts:
        i = step - 1  # schedule is 0-indexed
        ab = float(alpha_bars[i])
        eps = rng.normal(size=x0.shape)
        x_t = np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps
        t.add(
            f"Noise to t = {step}",
            f"√{num(ab)}·x₀ + √{num(1.0 - ab)}·ε\n{arr(x_t)}",
            x_t,
            detail=f"ᾱ_{step} = {num(ab)}; as t grows the signal fades and noise dominates.",
        )

    t.add(
        "Reverse direction (what the model learns)",
        "predict ε̂ = εθ(xₜ, t), then subtract it to step xₜ → xₜ₋₁ toward x₀",
        None,
        detail=(
            "Sampling runs this learned denoiser from t=T (pure noise) down to t=0, "
            "recovering a clean sample one step at a time — that is 'generation'."
        ),
    )
    t.result = x_t
    return t


def forward_diffusion(
    x0: np.ndarray,
    timesteps: int = 10,
    beta_start: float = 1e-4,
    beta_end: float = 0.2,
    seed: int = 0,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> np.ndarray:
    """Noise ``x0`` to timestep T. Set ``explain=True`` to print the trace."""
    t = forward_diffusion_trace(
        x0, timesteps=timesteps, beta_start=beta_start, beta_end=beta_end, seed=seed
    )
    return t.render(level) if explain else t.result


def demo() -> Trace:
    """A reproducible forward-diffusion example over a small 1-D ramp."""
    return forward_diffusion_trace(np.linspace(-1, 1, 6))
