"""The transformer forward pass, drawn as a circuit — the flagship flow.

``transformer_flow`` takes a short sentence and renders every stage of a toy
(but numerically real) transformer forward pass as an interactive, self-
contained HTML page:

    tokens -> embeddings -> + positional -> Q,K,V -> scores = QKᵀ/√d ->
    softmax -> attention matrix -> weighted sum (·V) -> FFN -> logits ->
    softmax over vocab

Every number on the page — the embedding table, Q/K/V, the attention matrix,
the FFN activations, the final next-token distribution — is computed once in
Python with a seeded RNG (so the page is fully deterministic), then embedded
as JSON and drawn as inline SVG. The browser owns only the *interaction*: a
Step control walks through the pipeline one stage at a time, dimming stages
not yet reached, and every matrix/vector cell responds to hover with the
exact value and a plain-language explanation of what it represents.

This intentionally mirrors ``optimumai.transformers.attention`` and
``optimumai.transformers.text_pipeline`` (same formulas, same scaling), but
recomputes a single, simplified head directly so the diagram can show every
intermediate (Q, K, V separately, not just the fused block output).
"""

from __future__ import annotations

import re

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
    vector_chip,
    write,
)
from optimumai.transformers.positional import positional_encoding

_WORD_RE = re.compile(r"[a-z0-9]+")

_CSS_EXTRA = """
 #flow-svg{min-width:1500px}
 .flow-stage{transition:opacity .25s}
"""


def _tokenize(text: str) -> list[str]:
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        raise ValueError("text must contain at least one word character")
    return tokens


def transformer_flow(
    text: str = "the cat sat on the mat",
    d_model: int = 8,
    seed: int = 0,
    out: str | None = None,
) -> str:
    """Build the full transformer-forward-pass flow for ``text`` as self-contained HTML.

    Args:
        text: A short sentence; whitespace/word tokenized and lowercased.
        d_model: Embedding / model dimension (also the Q/K/V dimension here).
        seed: Seed for the (untrained) embedding, Q/K/V, and FFN weight matrices.
        out: Path to write the HTML to (defaults to ``"transformer_flow.html"``).

    Returns:
        The path the HTML was written to.
    """
    tokens = _tokenize(text)
    n = len(tokens)
    vocab = sorted(set(tokens))
    vocab_size = len(vocab)
    tok_to_id = {t: i for i, t in enumerate(vocab)}
    ids = [tok_to_id[t] for t in tokens]

    rng = np.random.default_rng(seed)
    scale_e = 1.0 / np.sqrt(d_model)
    embed_table = rng.normal(size=(vocab_size, d_model)) * scale_e
    embed = embed_table[ids]

    pe = positional_encoding(n, d_model)
    x = embed + pe

    w_q = rng.normal(size=(d_model, d_model)) * 0.5
    w_k = rng.normal(size=(d_model, d_model)) * 0.5
    w_v = rng.normal(size=(d_model, d_model)) * 0.5
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    d_k = d_model
    scores = q @ k.T
    scaled = scores / np.sqrt(d_k)
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    attn = exps / exps.sum(axis=-1, keepdims=True)
    attn_out = attn @ v

    d_ff = 4 * d_model
    w1 = rng.normal(size=(d_model, d_ff)) * (1.0 / np.sqrt(d_model))
    w2 = rng.normal(size=(d_ff, d_model)) * (1.0 / np.sqrt(d_ff))
    hidden = np.maximum(attn_out @ w1, 0.0)  # ReLU FFN for a readable diagram
    ffn_out = hidden @ w2

    head = rng.normal(size=(d_model, vocab_size)) * (1.0 / np.sqrt(d_model))
    logits = ffn_out[-1] @ head
    l_shift = logits - logits.max()
    l_exp = np.exp(l_shift)
    probs = l_exp / l_exp.sum()
    top_idx = np.argsort(probs)[::-1][: min(3, vocab_size)]

    # ---------------------------------------------------------------- SVG
    box_w, box_h, gap = 150, 42, 60
    stage_defs = [
        ("tokens", "1. Tokens"),
        ("embed", "2. Embeddings"),
        ("posenc", "3. + Positional"),
        ("qkv", "4. Q, K, V"),
        ("scores", "5. Scores QKᵀ/√d"),
        ("softmax", "6. Softmax"),
        ("weighted", "7. Weighted sum ·V"),
        ("ffn", "8. Feed-forward"),
        ("logits", "9. Logits"),
        ("output", "10. Softmax → next token"),
    ]
    n_stages = len(stage_defs)
    content_h = 260
    svg_w = 40 + n_stages * (box_w + gap)
    svg_h = content_h + 80

    svg_parts = [svg_open(svg_w, svg_h)]
    row_y = 60
    for idx, (sid, title) in enumerate(stage_defs):
        gx = 20 + idx * (box_w + gap)
        svg_parts.append(stage_group_open(sid, gx, row_y))
        svg_parts.append(stage_box(box_w, box_h, title))
        svg_parts.append('<g transform="translate(0,52)">')

        if sid == "tokens":
            chips = "".join(
                f'<g class="cell" data-tip="{_esc(t)} → id {i}">'
                f'<rect x="{j * 34}" y="0" width="30" height="26" rx="4" '
                f'fill="#eef2ff" stroke="#c7d2fe"/>'
                f'<text class="cell-text" x="{j * 34 + 15}" y="13" font-size="9">{t}</text>'
                f"</g>"
                for j, (t, i) in enumerate(zip(tokens, ids, strict=True))
            )
            svg_parts.append(chips)
        elif sid == "embed":
            svg_parts.append(
                matrix_grid(
                    embed.round(3).tolist(),
                    0,
                    0,
                    cell=22,
                    row_labels=tokens,
                    id_prefix="embed",
                    tooltip_fn=lambda i, j, v: (
                        f"embed[{tokens[i]}][{j}] = {v:.4f} "
                        f"(row {i}'s lookup vector, dim {j})"
                    ),
                )
            )
        elif sid == "posenc":
            svg_parts.append(
                matrix_grid(
                    x.round(3).tolist(),
                    0,
                    0,
                    cell=22,
                    row_labels=tokens,
                    id_prefix="posenc",
                    tooltip_fn=lambda i, j, v: (
                        f"x[{tokens[i]}][{j}] = embed + PE = {v:.4f} "
                        f"(position {i} baked in)"
                    ),
                )
            )
        elif sid == "qkv":
            svg_parts.append('<text class="flow-stage-label" x="0" y="10" font-size="10">Q</text>')
            svg_parts.append(
                matrix_grid(
                    q.round(3).tolist(),
                    0,
                    14,
                    cell=16,
                    row_labels=tokens,
                    id_prefix="qmat",
                    tooltip_fn=lambda i, j, v: f"Q[{tokens[i]}][{j}] = {v:.4f} (what I seek)",
                )
            )
            k_y = 14 + n * 16 + 18
            svg_parts.append(
                f'<text class="flow-stage-label" x="0" y="{k_y - 4}" font-size="10">K</text>'
            )
            svg_parts.append(
                matrix_grid(
                    k.round(3).tolist(),
                    0,
                    k_y,
                    cell=16,
                    row_labels=tokens,
                    id_prefix="kmat",
                    tooltip_fn=lambda i, j, v: f"K[{tokens[i]}][{j}] = {v:.4f} (what I offer)",
                )
            )
        elif sid == "scores":
            svg_parts.append(
                matrix_grid(
                    scaled.round(3).tolist(),
                    0,
                    0,
                    cell=22,
                    row_labels=tokens,
                    col_labels=tokens,
                    id_prefix="scores",
                    tooltip_fn=lambda i, j, v: (
                        f"{tokens[i]} · {tokens[j]} / √d = {v:.4f} "
                        f"(raw relevance before softmax)"
                    ),
                )
            )
        elif sid == "softmax":
            svg_parts.append(
                matrix_grid(
                    attn.round(4).tolist(),
                    0,
                    0,
                    cell=22,
                    row_labels=tokens,
                    col_labels=tokens,
                    lo=0.0,
                    hi=1.0,
                    id_prefix="attn",
                    decimals=2,
                    tooltip_fn=lambda i, j, v: (
                        f"token {tokens[i]!r} attends to token {tokens[j]!r} = {v:.3f} "
                        f"({v * 100:.1f}% of {tokens[i]!r}'s attention)"
                    ),
                )
            )
        elif sid == "weighted":
            svg_parts.append(
                matrix_grid(
                    attn_out.round(3).tolist(),
                    0,
                    0,
                    cell=22,
                    row_labels=tokens,
                    id_prefix="attnout",
                    tooltip_fn=lambda i, j, v: (
                        f"context[{tokens[i]}][{j}] = {v:.4f} "
                        f"(blend of V rows, weighted by row {i} of the attention matrix)"
                    ),
                )
            )
        elif sid == "ffn":
            svg_parts.append(
                matrix_grid(
                    ffn_out.round(3).tolist(),
                    0,
                    0,
                    cell=22,
                    row_labels=tokens,
                    id_prefix="ffnout",
                    tooltip_fn=lambda i, j, v: (
                        f"ffn_out[{tokens[i]}][{j}] = {v:.4f} "
                        f"(ReLU(x·W1)·W2 — a per-token nonlinear transform)"
                    ),
                )
            )
        elif sid == "logits":
            svg_parts.append(
                vector_chip(
                    logits.round(3).tolist(),
                    0,
                    0,
                    cell=26,
                    vertical=True,
                    id_prefix="logit",
                    tooltip_fn=lambda i, v: f"logit[{vocab[i]!r}] = {v:.4f}",
                )
            )
        elif sid == "output":
            svg_parts.append(
                vector_chip(
                    probs.round(4).tolist(),
                    0,
                    0,
                    cell=26,
                    vertical=True,
                    lo=0.0,
                    hi=1.0,
                    id_prefix="prob",
                    tooltip_fn=lambda i, v: (
                        f"P(next={vocab[i]!r}) = {v:.4f}"
                        + (" <- top pick" if i == top_idx[0] else "")
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

    top_str = ", ".join(f"{vocab[i]!r} ({probs[i] * 100:.1f}%)" for i in top_idx)
    captions = [
        (
            f"<b>Tokenize.</b> {text!r} splits into {n} word tokens: {tokens}. "
            f"Vocabulary has {vocab_size} unique words, each getting a stable integer id."
        ),
        (
            "<b>Embed.</b> Every token id looks up one row of a "
            f"{vocab_size}×{d_model} table — a learned (here, seeded/random) vector "
            "that stands in for the word's meaning before any context is added."
        ),
        (
            "<b>Add positional encoding.</b> Self-attention has no notion of order — "
            "shuffle the tokens and the math doesn't change — so a sinusoidal signal "
            "unique to each position is added on top of the embedding."
        ),
        (
            "<b>Project to Q, K, V.</b> Three learned matrices turn each token's vector "
            "into a Query ('what am I looking for'), a Key ('what do I offer'), and a "
            "Value ('what do I pass on if picked')."
        ),
        (
            "<b>Score every pair.</b> scores = Q·Kᵀ / √d. Each cell [i,j] is how well "
            "query token i matches key token j, scaled down so softmax doesn't saturate."
        ),
        (
            "<b>Softmax, row by row.</b> Each row is turned into a probability "
            "distribution over which tokens to attend to — this IS the attention matrix; "
            "rows sum to 1."
        ),
        (
            "<b>Weighted sum.</b> attention · V blends every value vector by how much "
            "attention its token received — each output row is a convex combination of "
            "the other tokens' values."
        ),
        (
            "<b>Feed-forward network.</b> Each token's blended vector is pushed through "
            "an independent 2-layer MLP (widen, ReLU, narrow) — this is where most of a "
            "transformer's parameters (and 'knowledge') actually live."
        ),
        (
            "<b>Project to logits.</b> The final token's vector is matmul'd against an "
            "output head to produce one raw score per vocabulary word."
        ),
        (
            f"<b>Softmax over the vocabulary.</b> The logits become a probability "
            f"distribution over what comes next. Top picks here: {top_str}."
        ),
    ]
    stages = [
        {"id": sid, "title": title.split(". ", 1)[1], "caption": cap}
        for (sid, title), cap in zip(stage_defs, captions, strict=True)
    ]

    body = f"""
<h1>Transformer forward pass — {text!r}</h1>
<p class="sub">Every stage of a toy (but real) transformer forward pass, computed
once in Python with a seeded RNG and rendered live below. Step through the
pipeline, or hover any cell in a matrix to see its exact value.</p>
{flow_controls_html()}
<div id="flow-wrap">
{svg}
</div>
<p class="legend">Attention(Q,K,V) = softmax(Q·Kᵀ/√d)·V — the core operation of
every transformer. Q/K/V weights and the FFN are randomly initialized (untrained),
so treat the numbers as illustrative of the *mechanism*, not a trained model's
actual beliefs.</p>
"""
    script = runtime_script(HOVER_TOOLTIP_JS, stages)
    html = page(
        title="OptimumAI — transformer forward-pass flow",
        heading_sr=(
            "Interactive transformer forward pass: step through tokenize, embed, "
            "positional encoding, Q/K/V projection, attention scores, softmax, "
            "weighted sum, feed-forward, logits, and final softmax, hovering any "
            "matrix cell for its exact value."
        ),
        body=body,
        script=script,
        css_extra=_CSS_EXTRA,
    )
    return write(html, out, "transformer_flow.html")


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
