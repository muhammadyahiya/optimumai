"""Scaled dot-product attention, zoomed all the way in.

Where :mod:`optimumai.flows.transformer_flow` shows attention as one stage in
a bigger pipeline, ``attention_flow`` is the magnifying glass: it renders
each arithmetic step of ``Attention(Q, K, V) = softmax(Q·Kᵀ/√dₖ)·V`` as its
own stage, using the exact formula and scaling from
:class:`optimumai.transformers.attention.Attention`.

    Q, K, V  ->  Q·Kᵀ  ->  ÷√dₖ  ->  softmax (row-wise)  ->  ·V

Every matrix is a real, seeded numpy array; every cell is hoverable.
"""

from __future__ import annotations

import numpy as np

from optimumai.flows._shared import (
    HOVER_TOOLTIP_JS,
    arrow,
    flow_controls_html,
    matrix_grid,
    page,
    runtime_script,
    stage_box,
    stage_group_close,
    stage_group_open,
    svg_open,
    write,
)

_CSS_EXTRA = """
 #flow-svg{min-width:1150px}
 .flow-stage{transition:opacity .25s}
"""


def attention_flow(
    tokens: tuple[str, ...] = ("the", "cat", "sat", "down"),
    d_k: int = 4,
    seed: int = 0,
    out: str | None = None,
) -> str:
    """Build the zoomed-in scaled dot-product attention flow as self-contained HTML.

    Args:
        tokens: Token labels for the toy sequence (only used as row/column labels;
            Q/K/V are freshly sampled, matching
            :meth:`optimumai.transformers.attention.Attention.demo`'s style).
        d_k: Query/key/value dimension.
        seed: Seed for the Q/K/V sample matrices.
        out: Path to write the HTML to (defaults to ``"attention_flow.html"``).

    Returns:
        The path the HTML was written to.
    """
    tokens = tuple(tokens) or ("t0",)
    n = len(tokens)

    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, d_k)).round(2)
    k = rng.normal(size=(n, d_k)).round(2)
    v = rng.normal(size=(n, d_k)).round(2)

    raw_scores = q @ k.T
    scale = float(np.sqrt(d_k))
    scaled = raw_scores / scale
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    weights = exps / exps.sum(axis=-1, keepdims=True)
    output = weights @ v

    stage_defs = [
        ("qkv", "1. Q, K, V"),
        ("rawscores", "2. Q · Kᵀ"),
        ("scaled", "3. ÷ √dₖ"),
        ("softmax", "4. Softmax (row-wise)"),
        ("output", "5. Weighted sum · V"),
    ]
    n_stages = len(stage_defs)
    box_w, box_h, gap = 220, 42, 70
    svg_w = 40 + n_stages * (box_w + gap)
    svg_h = 380
    row_y = 60

    svg_parts = [svg_open(svg_w, svg_h)]
    for idx, (sid, title) in enumerate(stage_defs):
        gx = 20 + idx * (box_w + gap)
        svg_parts.append(stage_group_open(sid, gx, row_y))
        svg_parts.append(stage_box(box_w, box_h, title))
        svg_parts.append('<g transform="translate(0,52)">')

        if sid == "qkv":
            for label, mat, y0, tip_word in (
                ("Q", q, 0, "query"),
                ("K", k, (n * 18) + 20, "key"),
                ("V", v, 2 * ((n * 18) + 20), "value"),
            ):
                svg_parts.append(
                    f'<text class="flow-stage-label" x="0" y="{y0 + 8}" '
                    f'font-size="10">{label}</text>'
                )
                svg_parts.append(
                    matrix_grid(
                        mat.tolist(),
                        0,
                        y0 + 12,
                        cell=16,
                        row_labels=list(tokens),
                        id_prefix=f"{label.lower()}mat",
                        tooltip_fn=(
                            lambda i, j, v_, lbl=label, w=tip_word: (
                                f"{lbl}[{tokens[i]}][{j}] = {v_:.4f} (the {w} vector)"
                            )
                        ),
                    )
                )
        elif sid == "rawscores":
            svg_parts.append(
                matrix_grid(
                    raw_scores.round(3).tolist(),
                    0,
                    0,
                    cell=26,
                    row_labels=list(tokens),
                    col_labels=list(tokens),
                    id_prefix="raw",
                    tooltip_fn=lambda i, j, v_: (
                        f"Q[{tokens[i]}] · K[{tokens[j]}] = {v_:.4f} (raw dot-product score)"
                    ),
                )
            )
        elif sid == "scaled":
            svg_parts.append(
                matrix_grid(
                    scaled.round(3).tolist(),
                    0,
                    0,
                    cell=26,
                    row_labels=list(tokens),
                    col_labels=list(tokens),
                    id_prefix="scal",
                    tooltip_fn=lambda i, j, v_: (
                        f"score[{tokens[i]},{tokens[j]}] / √{d_k} = {v_:.4f} "
                        f"(keeps softmax out of its saturated region)"
                    ),
                )
            )
        elif sid == "softmax":
            svg_parts.append(
                matrix_grid(
                    weights.round(4).tolist(),
                    0,
                    0,
                    cell=26,
                    row_labels=list(tokens),
                    col_labels=list(tokens),
                    lo=0.0,
                    hi=1.0,
                    id_prefix="w",
                    tooltip_fn=lambda i, j, v_: (
                        f"{tokens[i]!r} attends to {tokens[j]!r} = {v_:.3f} "
                        f"({v_ * 100:.1f}% — row {i} sums to 1)"
                    ),
                )
            )
        elif sid == "output":
            svg_parts.append(
                matrix_grid(
                    output.round(3).tolist(),
                    0,
                    0,
                    cell=26,
                    row_labels=list(tokens),
                    id_prefix="out",
                    tooltip_fn=lambda i, j, v_: (
                        f"output[{tokens[i]}][{j}] = {v_:.4f} "
                        f"(convex combination of V rows, weighted by row {i} above)"
                    ),
                )
            )

        svg_parts.append("</g>")
        svg_parts.append(stage_group_close())
        if idx < n_stages - 1:
            ax = gx + box_w
            ay = row_y + box_h / 2
            svg_parts.append(arrow(ax, ay, ax + gap, ay, idx))

    svg_parts.append("</svg>")
    svg = "\n".join(svg_parts)

    row_sums = weights.sum(axis=-1).round(4).tolist()
    captions = [
        (
            "<b>Start with Q, K, V.</b> Each token already has three learned "
            "projections: a Query (what it's looking for), a Key (what it advertises "
            "about itself), and a Value (what it actually contributes if attended to)."
        ),
        (
            "<b>Score every pair: Q · Kᵀ.</b> Entry [i,j] is the dot product of query "
            "token i with key token j — a bigger dot product means the vectors point "
            "the same way, i.e. token j looks relevant to token i."
        ),
        (
            f"<b>Scale by 1/√dₖ = 1/√{d_k}.</b> Dot products grow with dimension, which "
            "would push softmax's inputs to extremes and starve the gradient. Dividing "
            "by √dₖ keeps the scores in a well-behaved range."
        ),
        (
            "<b>Softmax, one row at a time.</b> Each row becomes a probability "
            f"distribution over which tokens to attend to. Row sums: {row_sums} — "
            "always 1.0, by construction."
        ),
        (
            "<b>Weighted sum: weights · V.</b> The final output for each token is a "
            "blend of every value vector, mixed exactly according to that token's row "
            "of attention weights — this is the payload attention actually delivers."
        ),
    ]
    stages = [
        {"id": sid, "title": title.split(". ", 1)[1], "caption": cap}
        for (sid, title), cap in zip(stage_defs, captions, strict=True)
    ]

    body = f"""
<h1>Scaled dot-product attention — zoomed in</h1>
<p class="sub">Attention(Q,K,V) = softmax(Q·Kᵀ/√dₖ)·V, worked out arithmetic
step by arithmetic step for {n} tokens {list(tokens)} with dₖ = {d_k}. Step
through the pipeline, or hover any cell to see the exact number and what it
means.</p>
{flow_controls_html()}
<div id="flow-wrap">
{svg}
</div>
<p class="legend">Q, K, V here are freshly sampled (not learned from real
text) so every arithmetic step is easy to verify by hand for a tiny example —
the formula is identical to what a trained transformer runs at every layer.</p>
"""
    script = runtime_script(HOVER_TOOLTIP_JS, stages)
    html = page(
        title="OptimumAI — scaled dot-product attention flow",
        heading_sr=(
            "Interactive scaled dot-product attention: step through Q, K, V, "
            "the raw score matrix, scaling, row-wise softmax, and the final "
            "weighted sum, hovering any cell for its exact value."
        ),
        body=body,
        script=script,
        css_extra=_CSS_EXTRA,
    )
    return write(html, out, "attention_flow.html")
