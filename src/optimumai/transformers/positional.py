"""Sinusoidal positional encoding — how a transformer learns *where* a token is.

Self-attention is permutation-invariant: shuffle the input tokens and the raw
attention math produces the same (shuffled) output. Word order is meaning, so we
must inject position explicitly. "Attention Is All You Need" adds a fixed
sinusoidal signal to each token embedding:

    PE[pos, 2i]   = sin( pos / 10000^(2i/d_model) )
    PE[pos, 2i+1] = cos( pos / 10000^(2i/d_model) )

Each dimension is a sinusoid whose wavelength grows geometrically from 2π up to
10000·2π. Because ``sin``/``cos`` of a shifted angle are linear combinations of
the unshifted ones, the model can attend by *relative* offset (pos + k is a fixed
linear function of pos). Note that GPT-style models such as nanoGPT drop these
fixed sinusoids in favour of a *learned* positional embedding table.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def positional_encoding_trace(seq_len: int, d_model: int) -> Trace:
    """Build the full trace of the sinusoidal positional encoding matrix."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")

    t = Trace(
        op="positional_encoding",
        formula="PE[pos,2i]=sin(pos/10000^(2i/d)), PE[pos,2i+1]=cos(pos/10000^(2i/d))",
        complexity="O(seq_len · d_model)",
        why_ai=[
            "Attention is permutation-invariant, so position must be injected explicitly",
            "Sinusoids of geometrically-spaced wavelengths let the model attend by relative offset",
            "GPT/nanoGPT instead learn a positional embedding table rather than fix these waves",
        ],
        meta={"seq_len": seq_len, "d_model": d_model},
    )

    pos = np.arange(seq_len)[:, None].astype(float)
    i = np.arange(d_model)[None, :].astype(float)
    div_term = np.power(10000.0, (2.0 * (i // 2)) / d_model)
    angles = pos / div_term
    t.add(
        "Angle table: pos / 10000^(2i/d_model)",
        f"row = position, column = dimension\n{arr(angles)}",
        angles,
        detail="Even/odd columns share a frequency; wavelengths grow from 2π up to 10000·2π.",
    )

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    # Show a couple of concrete entries so the formula is not just abstract.
    for pos_idx, dim_idx in ((min(1, seq_len - 1), 0), (min(1, seq_len - 1), min(1, d_model - 1))):
        fn = "sin" if dim_idx % 2 == 0 else "cos"
        angle = float(angles[pos_idx, dim_idx])
        t.add(
            f"PE[{pos_idx},{dim_idx}] = {fn}({num(angle)})",
            f"{fn}(pos / 10000^(2·{dim_idx // 2}/{d_model})) = {num(float(pe[pos_idx, dim_idx]))}",
            float(pe[pos_idx, dim_idx]),
            detail=f"Column {dim_idx} is a {fn}e wave (even dims use sin, odd dims use cos).",
        )

    t.add(
        "Positional encoding matrix",
        f"add this to the token embeddings\n{arr(pe)}",
        pe,
        detail="Shape (seq_len × d_model): one fixed vector per position, no parameters to learn.",
    )
    t.result = pe
    return t


def positional_encoding(
    seq_len: int,
    d_model: int,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> np.ndarray:
    """Return the sinusoidal PE matrix. Set ``explain=True`` to print the trace."""
    t = positional_encoding_trace(seq_len, d_model)
    return t.render(level) if explain else t.result
