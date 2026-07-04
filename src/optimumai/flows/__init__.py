"""Interactive circuit-flow visualizations — distill.pub-style, self-contained HTML.

Each builder in this subpackage renders the step-by-step "circuit" a piece of
data takes through a small AI model as a single offline ``.html`` file: real
numbers (computed once in Python with a seeded RNG), drawn as inline SVG
boxes/arrows/heatmap chips, with a Step control that walks through the
pipeline stage by stage and hover-to-inspect tooltips on every matrix cell.

No CDN, no server, no build step — open the file in any browser, online or
off.

    * :func:`transformer_flow` — the full toy-transformer forward pass:
      tokens -> embeddings -> +positional -> Q,K,V -> scores -> softmax ->
      attention matrix -> weighted sum -> feed-forward -> logits -> softmax.
    * :func:`attention_flow` — scaled dot-product attention zoomed all the
      way in: Q,K,V -> Q·Kᵀ -> ÷√d -> softmax -> ·V.
    * :func:`tfidf_flow` — corpus -> term counts -> TF -> document frequency
      -> IDF -> the final TF-IDF matrix.
    * :func:`word2vec_flow` — skip-gram: center word -> one-hot -> embedding
      lookup -> hidden vector -> vocab scores -> softmax -> nearest neighbors.

Use :func:`flow` to build any of them by name (handy for a CLI).
"""

from __future__ import annotations

from optimumai.flows.attention_flow import attention_flow
from optimumai.flows.tfidf_flow import tfidf_flow
from optimumai.flows.transformer_flow import transformer_flow
from optimumai.flows.word2vec_flow import word2vec_flow

_FLOWS = {
    "transformer": transformer_flow,
    "attention": attention_flow,
    "tfidf": tfidf_flow,
    "word2vec": word2vec_flow,
}


def flow(name: str, out: str | None = None) -> str:
    """Build the named flow ("transformer", "attention", "tfidf", or "word2vec")."""
    try:
        builder = _FLOWS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(_FLOWS))
        raise ValueError(f"unknown flow {name!r}; choose from: {valid}") from exc
    return builder(out=out)


__all__ = [
    "attention_flow",
    "flow",
    "tfidf_flow",
    "transformer_flow",
    "word2vec_flow",
]
