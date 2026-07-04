"""Skip-gram word2vec, drawn as a flow — the origin story of embedding tables.

``word2vec_flow`` renders one skip-gram forward pass end to end:

    center word -> one-hot -> embedding lookup (W_in) -> hidden vector ->
    scores over vocab (W_out) -> softmax -> predicted context

plus a nearest-neighbor panel computed after a handful of real SGD steps, so
the "words that share contexts end up nearby" claim is backed by an actual
(tiny) trained model. Everything is pulled straight from
:mod:`optimumai.nlp.word2vec` — the same ``SkipGramModel``, the same
full-softmax gradient.
"""

from __future__ import annotations

import numpy as np

from optimumai.flows._shared import (
    HOVER_TOOLTIP_JS,
    arrow,
    flow_controls_html,
    page,
    runtime_script,
    stage_box,
    stage_group_close,
    stage_group_open,
    svg_open,
    vector_chip,
    write,
)
from optimumai.nlp.word2vec import SkipGramModel, build_vocab, skipgram_pairs

_CSS_EXTRA = """
 #flow-svg{min-width:1350px}
 .flow-stage{transition:opacity .25s}
 .nn-chip{display:inline-block;background:#eef2ff;border:1px solid #c7d2fe;
        border-radius:6px;padding:4px 10px;margin:3px;font-family:monospace}
"""


def word2vec_flow(
    corpus: tuple[str, ...] = (
        "the cat sat on the mat",
        "the dog sat on the mat",
        "the cat chased the mouse",
        "the dog chased the mouse",
    ),
    center: str = "cat",
    window: int = 1,
    dim: int = 4,
    steps: int = 30,
    lr: float = 0.2,
    seed: int = 0,
    out: str | None = None,
) -> str:
    """Build the skip-gram word2vec flow for ``corpus`` as self-contained HTML.

    Args:
        corpus: Sentences to train on (mirrors :func:`optimumai.nlp.word2vec.demo`).
        center: Which vocabulary word to spotlight as the "center word" example.
        window: Skip-gram context window radius.
        dim: Embedding dimensionality.
        steps: Number of full-softmax SGD steps to train.
        lr: SGD learning rate.
        seed: Seed for model init and the training sampler.
        out: Path to write the HTML to (defaults to ``"word2vec_flow.html"``).

    Returns:
        The path the HTML was written to.
    """
    corpus = tuple(corpus)
    vocab = build_vocab(list(corpus))
    if center not in vocab:
        center = vocab[0]
    pairs = skipgram_pairs(list(corpus), window=window)
    vocab_size = len(vocab)
    idx_of = {w: i for i, w in enumerate(vocab)}
    c_idx = idx_of[center]

    model = SkipGramModel(vocab, dim=dim, seed=seed)
    one_hot = [1.0 if i == c_idx else 0.0 for i in range(vocab_size)]
    v_c_before = model.w_in[c_idx].copy()
    hidden_before = v_c_before.copy()
    scores_before = model.w_out @ v_c_before
    probs_before = model.predict(center)

    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(steps):
        c, o = pairs[rng.integers(len(pairs))]
        losses.append(model.step(c, o, lr=lr))

    probs_after = model.predict(center)
    top_idx_after = np.argsort(probs_after)[::-1][: min(3, vocab_size)]
    neighbors = model.most_similar(center, k=min(4, vocab_size - 1))

    stage_defs = [
        ("center", "1. Center word"),
        ("onehot", "2. One-hot"),
        ("lookup", "3. Embedding lookup"),
        ("hidden", "4. Hidden vector"),
        ("scores", "5. Scores over vocab"),
        ("softmax", "6. Softmax"),
        ("neighbors", "7. Nearest neighbors"),
    ]
    n_stages = len(stage_defs)
    box_w, box_h, gap = 200, 42, 55
    svg_w = 40 + n_stages * (box_w + gap)
    svg_h = 90 + max(vocab_size, 6) * 24 + 60
    row_y = 60

    svg_parts = [svg_open(svg_w, svg_h)]
    for idx, (sid, title) in enumerate(stage_defs):
        gx = 20 + idx * (box_w + gap)
        svg_parts.append(stage_group_open(sid, gx, row_y))
        svg_parts.append(stage_box(box_w, box_h, title))
        svg_parts.append('<g transform="translate(0,52)">')

        if sid == "center":
            svg_parts.append(
                f'<g class="cell" data-tip="center word = {center!r}, id={c_idx}">'
                f'<rect x="0" y="0" width="120" height="34" rx="6" '
                f'fill="#2563eb"/>'
                f'<text class="cell-text" x="60" y="17" fill="#fff" '
                f'font-size="13" font-weight="700">{center}</text>'
                f"</g>"
            )
        elif sid == "onehot":
            svg_parts.append(
                vector_chip(
                    one_hot,
                    0,
                    0,
                    cell=22,
                    vertical=True,
                    lo=0.0,
                    hi=1.0,
                    decimals=0,
                    id_prefix="oh",
                    tooltip_fn=lambda i, v: (
                        f"one_hot[{vocab[i]!r}] = {int(v)}"
                        + (" <- the center word" if i == c_idx else "")
                    ),
                )
            )
        elif sid == "lookup":
            svg_parts.append(
                f'<text class="flow-stage-label" x="0" y="10" font-size="10">'
                f"W_in[{center!r}]</text>"
            )
            svg_parts.append(
                vector_chip(
                    v_c_before.round(4).tolist(),
                    0,
                    16,
                    cell=22,
                    vertical=True,
                    id_prefix="win",
                    tooltip_fn=lambda i, v: (
                        f"W_in[{center!r}][{i}] = {v:.4f} "
                        f"(row picked out by the one-hot dot product)"
                    ),
                )
            )
        elif sid == "hidden":
            svg_parts.append(
                vector_chip(
                    hidden_before.round(4).tolist(),
                    0,
                    0,
                    cell=22,
                    vertical=True,
                    id_prefix="hid",
                    tooltip_fn=lambda i, v: (
                        f"h[{i}] = {v:.4f} "
                        f"(the hidden vector — identical to W_in[center], no nonlinearity)"
                    ),
                )
            )
        elif sid == "scores":
            svg_parts.append(
                vector_chip(
                    scores_before.round(3).tolist(),
                    0,
                    0,
                    cell=20,
                    vertical=True,
                    id_prefix="sc",
                    tooltip_fn=lambda i, v: f"score[{vocab[i]!r}] = W_out[{i}] . h = {v:.4f}",
                )
            )
        elif sid == "softmax":
            top_before = int(np.argmax(probs_before))
            svg_parts.append(
                vector_chip(
                    probs_before.round(4).tolist(),
                    0,
                    0,
                    cell=20,
                    vertical=True,
                    lo=0.0,
                    hi=1.0,
                    id_prefix="pb",
                    tooltip_fn=lambda i, v, best=top_before: (
                        f"P({vocab[i]!r} | {center!r}) before training = {v:.4f}"
                        + (" <- argmax" if i == best else "")
                    ),
                )
            )
        elif sid == "neighbors":
            chip_y = 10
            for w, score in neighbors:
                svg_parts.append(
                    f'<g class="cell" data-tip="cosine({center!r}, {w!r}) = {score:.4f}">'
                    f'<rect x="0" y="{chip_y}" width="150" height="26" rx="5" '
                    f'fill="#eef2ff" stroke="#c7d2fe"/>'
                    f'<text class="cell-text" x="75" y="{chip_y + 13}" '
                    f'font-size="10">{w} ({score:.3f})</text>'
                    f"</g>"
                )
                chip_y += 30

        svg_parts.append("</g>")
        svg_parts.append(stage_group_close())
        if idx < n_stages - 1:
            ax = gx + box_w
            ay = row_y + box_h / 2
            svg_parts.append(arrow(ax, ay, ax + gap, ay, idx))

    svg_parts.append("</svg>")
    svg = "\n".join(svg_parts)

    after_str = ", ".join(f"{vocab[i]!r}={probs_after[i]:.3f}" for i in top_idx_after)
    neighbor_str = ", ".join(f"{w} ({s:.3f})" for w, s in neighbors)

    captions = [
        (
            f"<b>Pick a center word.</b> {center!r} is one of {vocab_size} vocabulary "
            f"words drawn from {len(pairs)} (center, context) training pairs built by "
            f"sliding a window={window} over the corpus."
        ),
        (
            "<b>One-hot encode.</b> The center word becomes a sparse vector: 1 at its "
            "own index, 0 everywhere else. This is just an index-selector — it carries "
            "no information about meaning yet."
        ),
        (
            "<b>Embedding lookup.</b> one_hot @ W_in picks out exactly one row of the "
            "learned W_in matrix — that row IS the center-word embedding being trained."
        ),
        (
            "<b>Hidden vector.</b> Skip-gram has no hidden nonlinearity: the hidden "
            "vector is literally the embedding row from the previous stage, passed "
            "straight through."
        ),
        (
            "<b>Score every vocabulary word.</b> scores = W_out @ h — a second learned "
            "matrix (the 'context' embeddings) scores how well each vocabulary word "
            "fits as a neighbor of the center word."
        ),
        (
            f"<b>Softmax → P(context | center).</b> Before any training this is close "
            f"to uniform. After {steps} SGD steps (lr={lr}), predicting the first "
            f"training pair's context sharpens toward: {after_str}."
        ),
        (
            f"<b>Nearest neighbors after training.</b> Cosine similarity in W_in space "
            f"for {center!r}: {neighbor_str}. Even a few steps nudge words that share "
            "contexts closer together — the entire mechanism behind 'king - man + woman "
            "≈ queen', just scaled up."
        ),
    ]
    stages = [
        {"id": sid, "title": title.split(". ", 1)[1], "caption": cap}
        for (sid, title), cap in zip(stage_defs, captions, strict=True)
    ]

    body = f"""
<h1>Skip-gram word2vec flow</h1>
<p class="sub">One forward pass for center word {center!r}, plus what
{steps} real SGD steps do to its neighborhood. Step through the pipeline, or
hover any cell/chip for the exact value.</p>
{flow_controls_html()}
<div id="flow-wrap">
{svg}
</div>
<p class="legend">P(context|center) = softmax(W_out @ W_in[center]);
grad(scores) = predicted_probs - one_hot(actual_context). Corpus:
{" | ".join(corpus)}</p>
"""
    script = runtime_script(HOVER_TOOLTIP_JS, stages)
    html = page(
        title="OptimumAI — word2vec skip-gram flow",
        heading_sr=(
            "Interactive skip-gram word2vec: step through center word, one-hot "
            "encoding, embedding lookup, hidden vector, vocabulary scores, softmax, "
            "and trained nearest neighbors, hovering any cell for its exact value."
        ),
        body=body,
        script=script,
        css_extra=_CSS_EXTRA,
    )
    return write(html, out, "word2vec_flow.html")
