"""JEPA — Yann LeCun's Joint-Embedding Predictive Architecture.

The dominant self-supervised recipe (BERT, GPT, diffusion) is *generative*: mask
or corrupt an input and train the model to reconstruct the missing pixels or
tokens. LeCun's critique is that most of that signal is noise — reconstructing
the exact texture of grass or the grain of a photo burns capacity on detail that
is fundamentally unpredictable.

JEPA sidesteps this. Given two related views ``x`` (context) and ``y`` (target),
it encodes *both* into an abstract representation space with a shared encoder
``f`` and predicts the *target's embedding* ``f(y)`` from the *context's
embedding* ``f(x)`` via a predictor ``g``. It is trained like an Energy-Based
Model: the energy ``E(x, y) = ‖g(f(x)) − f(y)‖²`` is low when the prediction
lands on the true target representation and high otherwise. Nothing is ever
reconstructed in pixel space — the model only has to be right about *meaning*.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.base_op import BaseOp
from optimumai.core.trace import Trace


class JEPA(BaseOp):
    """A minimal Joint-Embedding Predictive Architecture.

    Uses a fixed, seeded linear encoder ``f`` (shared by context and target) and
    a linear predictor ``g`` living entirely in embedding space. Everything is
    deterministic given ``seed`` so traces are reproducible.

    Args:
        input_dim: Dimension of a raw input view (e.g. flattened patch).
        embed_dim: Dimension of the abstract representation space.
        seed: RNG seed for the encoder/predictor weights.
    """

    name = "jepa"

    def __init__(self, input_dim: int, embed_dim: int, seed: int = 0):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.seed = seed
        rng = np.random.default_rng(seed)
        # f: shared encoder mapping raw views -> representation space.
        self.f = rng.normal(size=(embed_dim, input_dim)) / np.sqrt(input_dim)
        # g: predictor that maps a context embedding -> predicted target embedding.
        self.g = rng.normal(size=(embed_dim, embed_dim)) / np.sqrt(embed_dim)

    def trace(self, context, target) -> Trace:
        x = np.asarray(context, dtype=float)
        y = np.asarray(target, dtype=float)
        for label, vec in ("context", x), ("target", y):
            if vec.shape != (self.input_dim,):
                raise ValueError(
                    f"{label} must be a length-{self.input_dim} vector, got shape {vec.shape}"
                )

        t = Trace(
            op="jepa",
            formula="E(x,y) = ‖ g(f(x)) − f(y) ‖²",
            complexity="O(d·D) for input dim d and embed dim D",
            why_ai=[
                "LeCun's thesis: predict abstract representations, not raw pixels/tokens",
                "Energy-based & self-supervised — low energy when the prediction matches meaning",
                "I-JEPA (images) and V-JEPA (video) learn without pixel reconstruction",
                "Yields emergent intuitive physics and reusable world models",
            ],
            meta={"embed_dim": self.embed_dim, "input_dim": self.input_dim},
        )

        s_x = self.f @ x
        t.add(
            "Encode the context view: sₓ = f(x)",
            f"f · context  →  {arr(s_x)}",
            s_x,
            detail="The shared encoder f lifts the raw context into representation space.",
        )

        s_y = self.f @ y
        t.add(
            "Encode the target view: s_y = f(y)",
            f"f · target  →  {arr(s_y)}",
            s_y,
            detail=(
                "This embedding — not the raw target — is what we predict. The abstract "
                "'target representation' is the prediction target, so unpredictable pixel "
                "detail never enters the loss."
            ),
        )

        pred = self.g @ s_x
        t.add(
            "Predict the target embedding: pred = g(sₓ)",
            f"g · sₓ  →  {arr(pred)}",
            pred,
            detail=(
                "g predicts where the target lands in representation space. Collapse "
                "(f mapping everything to one point, driving E→0 trivially) is the key "
                "training challenge — real JEPA uses stop-gradients / EMA target encoders "
                "and variance regularizers to prevent it."
            ),
        )

        diff = pred - s_y
        energy = float(diff @ diff)
        t.add(
            "Energy = squared latent distance: E = ‖pred − s_y‖²",
            f"‖ {arr(pred)} − {arr(s_y)} ‖²  =  {num(energy)}",
            energy,
            detail=(
                "This is the whole training signal: low energy = the context correctly "
                "predicts the target's meaning. A generative model would instead try to "
                "reconstruct the raw target in input space, spending capacity on "
                "unpredictable pixel-level detail that JEPA deliberately ignores."
            ),
        )

        t.result = energy
        return t

    def forward(self, context, target, explain: bool = False, level="engineer"):
        """Compute the energy. Set ``explain=True`` to print the four-stage trace."""
        t = self.trace(context, target)
        return t.render(level) if explain else t.result

    @classmethod
    def demo(cls, seed: int = 0) -> Trace:
        """A tiny, reproducible example: two noisy views of one latent vector.

        ``target = context + small noise`` mimics two augmentations of the same
        image, so a well-behaved encoder keeps the energy small.
        """
        rng = np.random.default_rng(seed)
        base = rng.normal(size=6).round(2)
        context = base
        target = (base + 0.05 * rng.normal(size=6)).round(2)
        return cls(input_dim=6, embed_dim=4, seed=seed).trace(context, target)
