"""Your text, all the way through a (toy) transformer — the v0.5 headline feature.

Type a sentence and watch it become a next-token prediction, stage by stage:

    text → tokens → embed → +PE → (transformer block)×N → logits → softmax

Every stage is a matrix operation you have already met elsewhere in OptimumAI:
tokenising builds an integer index, embedding is a table lookup, positional
encoding is an add, each transformer block is attention + feed-forward, and the
final head is one matmul followed by a softmax. This *is* how a real LLM turns
your prompt into a probability distribution over what comes next — only the
vocabulary size, the model dimension, the number of layers, and the training
separate this from GPT.
"""

from __future__ import annotations

import re

import numpy as np

from optimumai.core._fmt import arr, shape_of
from optimumai.core.base_op import BaseOp
from optimumai.core.trace import Trace
from optimumai.transformers.block import TransformerBlock
from optimumai.transformers.positional import positional_encoding

# Lowercase, then keep runs of word characters as tokens (splitting off punctuation).
_WORD_RE = re.compile(r"[a-z0-9]+")


class TextPipeline(BaseOp):
    """Push a user's own text through a full, tiny, untrained transformer.

    Args:
        text: The prompt to run through the pipeline (must be non-empty).
        layers: How many :class:`TransformerBlock` layers to stack.
        d_model: Model / residual-stream dimension (must divide by ``n_heads``).
        n_heads: Number of attention heads per block.
        seed: Seed for the (fixed, untrained) embedding and output-head weights.
        tokenizer: ``"whitespace"`` (word-level) or ``"bpe"`` (via ``tiktoken``).
    """

    name = "text_pipeline"

    def __init__(
        self,
        text: str,
        layers: int = 2,
        d_model: int = 16,
        n_heads: int = 2,
        seed: int = 0,
        tokenizer: str = "whitespace",
    ):
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string with at least one token")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.text = text
        self.layers = layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.seed = seed
        self.tokenizer = tokenizer

    def _tokenize(self, t: Trace) -> tuple[list[str], list[int], list[str]]:
        """Turn raw text into tokens + integer ids, recording the steps on ``t``."""
        if self.tokenizer == "whitespace":
            tokens = _WORD_RE.findall(self.text.lower())
            if not tokens:
                raise ValueError(
                    "text produced no tokens after lowercasing and splitting"
                )
            vocab = sorted(set(tokens))
            token_to_id = {tok: i for i, tok in enumerate(vocab)}
            ids = [token_to_id[tok] for tok in tokens]
            t.add(
                "Tokenize (whitespace, lowercased)",
                f"tokens = {tokens}",
                None,
                detail="Lowercase, then split on whitespace/punctuation into word tokens.",
            )
            t.add(
                f"Build vocab ({len(vocab)} unique) and map token → id",
                f"vocab = {vocab}  |  ids = {ids}",
                np.asarray(ids),
                detail="Each unique token gets a stable integer index (sorted for determinism).",
            )
            return tokens, ids, vocab

        if self.tokenizer == "bpe":
            try:
                import tiktoken
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    'bpe tokenizer needs tiktoken; install with '
                    'pip install "optimumai[tokenize]"'
                ) from exc
            enc = tiktoken.get_encoding("gpt2")
            ids = enc.encode(self.text)
            if not ids:
                raise ValueError("text produced no BPE tokens")
            tokens = [enc.decode([i]) for i in ids]
            # Treat the raw byte-pair ids as the vocab for this toy run.
            vocab = [enc.decode([i]) for i in sorted(set(ids))]
            remap = {tid: pos for pos, tid in enumerate(sorted(set(ids)))}
            ids = [remap[i] for i in ids]
            t.add(
                "Tokenize (BPE via tiktoken gpt2)",
                f"tokens = {tokens}",
                None,
                detail="Byte-pair encoding splits text into sub-word pieces GPT actually uses.",
            )
            t.add(
                f"Compact ids into a {len(vocab)}-token local vocab",
                f"ids = {ids}",
                np.asarray(ids),
                detail="Original gpt2 ids are remapped to a dense range for this toy head.",
            )
            return tokens, ids, vocab

        raise ValueError(
            f"unknown tokenizer {self.tokenizer!r}; choose 'whitespace' or 'bpe'"
        )

    def trace(self) -> Trace:
        t = Trace(
            op="text_pipeline",
            formula="text → tokens → embed → +PE → (transformer block)×N → logits → softmax",
            complexity="O(N·(n²·d + n·d·d_ff)) for N layers, n tokens, dimension d",
            why_ai=[
                "This IS how an LLM turns your prompt into a next-token distribution",
                "Every stage is a matrix op you have already seen elsewhere in OptimumAI",
                "Only vocabulary, dimension, depth, and training separate this from GPT",
            ],
            meta={
                "tokenizer": self.tokenizer,
                "layers": self.layers,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
            },
        )

        # --- Stage 1: TOKENIZE ---
        tokens, ids, vocab = self._tokenize(t)
        seq_len = len(ids)
        vocab_size = len(vocab)

        # --- Stage 2: EMBED ---
        rng = np.random.default_rng(self.seed)
        embed_table = rng.normal(size=(vocab_size, self.d_model)) * (
            1.0 / np.sqrt(self.d_model)
        )
        x = embed_table[np.asarray(ids)]
        t.add(
            "Embed: look up each token id → its row",
            f"embedding table {embed_table.shape} → sequence {shape_of(x)}\n{arr(x)}",
            x,
            detail=(
                "A seeded (untrained) vocab_size × d_model table; row i is token i's vector."
            ),
        )

        # --- Stage 3: POSITIONAL ---
        pe = positional_encoding(seq_len, self.d_model)
        x = x + pe
        t.add(
            "Add positional encoding (+PE)",
            f"embeddings + sinusoidal PE {shape_of(pe)}\n{arr(x)}",
            x,
            detail="Attention is order-blind, so we add a per-position signal to each token.",
        )

        # --- Stage 4: TRANSFORMER LAYERS ---
        block = TransformerBlock(self.d_model, self.n_heads)
        for layer_idx in range(self.layers):
            x = block.run(x)
            t.add(
                f"Transformer block {layer_idx + 1}/{self.layers} (causal)",
                f"pre-norm attention + feed-forward → {shape_of(x)}\n{arr(x)}",
                x,
                detail=(
                    "x = x + MHA(LN(x), causal); x = x + FFN(LN(x)); "
                    "output feeds the next layer."
                ),
            )

        # --- Stage 5: LOGITS → next-token distribution ---
        final = x[-1]
        head = rng.normal(size=(self.d_model, vocab_size)) * (
            1.0 / np.sqrt(self.d_model)
        )
        logits = final @ head
        t.add(
            "Logits: project final token → vocab scores",
            f"x[-1] @ head {head.shape} → {vocab_size} logits\n{arr(logits)}",
            logits,
            detail="Only the last position matters for predicting the very next token.",
        )

        shifted = logits - np.max(logits)
        exps = np.exp(shifted)
        dist = exps / np.sum(exps)
        top_k = min(3, vocab_size)
        top_idx = np.argsort(dist)[::-1][:top_k]
        top_str = ", ".join(
            f"{vocab[i]!r} p={float(dist[i]):.4f}" for i in top_idx
        )
        t.add(
            f"Softmax → next-token distribution (top {top_k})",
            f"softmax(logits) sums to {float(np.sum(dist)):.4f}\ntop: {top_str}",
            dist,
            detail="A real LLM would sample or argmax this to pick the next token, then repeat.",
        )

        t.result = dist
        t.meta.update(
            {
                "tokens": tokens,
                "vocab_size": vocab_size,
                "seq_len": seq_len,
                "layers": self.layers,
                "d_model": self.d_model,
            }
        )
        return t

    def forward(self, explain: bool = False, level: str = "engineer"):
        """Run the pipeline. Set ``explain=True`` to print the full stage-by-stage trace."""
        t = self.trace()
        return t.render(level) if explain else t.result

    @classmethod
    def demo(cls) -> Trace:
        """A tiny, reproducible run on a fixed prompt for docs and the CLI."""
        return cls("why is the sky blue", layers=2).trace()
