"""Superposition — Anthropic's "Toy Models of Superposition".

Why does a single neuron in a language model light up for both "the Golden Gate
Bridge" and "guilt" and "italics"? Because there are far more concepts worth
representing than there are neurons to represent them. When a network has *fewer
neurons than features*, it is forced to pack multiple features into shared
directions of activation space — this is **superposition**, and it is why
individual neurons are *polysemantic* (they respond to many unrelated things).

This toy model makes the phenomenon concrete. Features are embedded into a small
activation space by a matrix ``W`` whose columns are feature directions. Because
there are more features than dimensions, those directions *cannot* be mutually
orthogonal — they interfere. Yet if the features are **sparse** (few active at
once), a simple linear read-out recovers them anyway. That recoverability is the
intuition behind sparse autoencoders / dictionary learning, the tools
mechanistic interpretability uses to pull monosemantic features back out.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def superposition_trace(
    n_features: int, n_neurons: int, sparsity: float = 0.7, seed: int = 0
) -> Trace:
    """Build the full trace of a toy superposition encode/decode cycle.

    Args:
        n_features: Number of distinct features (must exceed ``n_neurons``).
        n_neurons: Number of neurons / activation dimensions.
        sparsity: Fraction of features that are *inactive* on a given input.
        seed: RNG seed for the weight matrix and the sparse feature vector.
    """
    if n_features <= n_neurons:
        raise ValueError(
            f"superposition requires n_features > n_neurons, got "
            f"{n_features} ≤ {n_neurons} (no packing needed if they fit)"
        )
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")

    rng = np.random.default_rng(seed)

    t = Trace(
        op="superposition",
        formula="x = W·h  (encode),  ĥ = Wᵀ·x  (decode);  n_features > n_neurons ⇒ superposition",
        complexity="O(n_features · n_neurons)",
        why_ai=[
            "Neurons are polysemantic — one neuron responds to many unrelated concepts",
            "Features live as linear directions in activation space, not as single neurons",
            "Sparse autoencoders (SAEs) / dictionary learning recover monosemantic features",
            "Foundational to mechanistic interpretability and model safety / steering",
        ],
        meta={"n_features": n_features, "n_neurons": n_neurons},
    )

    # 1. Embed n_features directions into n_neurons-dim space, columns L2-normalized.
    W = rng.normal(size=(n_neurons, n_features))
    W = W / np.linalg.norm(W, axis=0, keepdims=True)
    t.add(
        "Set up feature directions: W  (n_neurons × n_features)",
        f"columns are unit feature directions\n{arr(W)}",
        W,
        detail=(
            f"{n_features} features must fit in {n_neurons} dimensions. With more "
            f"features than dimensions they CANNOT all be orthogonal — the network is "
            f"forced to overlap them."
        ),
    )

    # 2. Interference: the Gram matrix Wᵀ·W. Off-diagonals measure feature overlap.
    gram = W.T @ W
    off_diag = gram - np.diag(np.diag(gram))
    max_offdiag = float(np.max(np.abs(off_diag)))
    t.add(
        "Show the interference: Gram matrix Wᵀ·W",
        f"off-diagonals ≠ 0 ⇒ features share directions\n{arr(gram)}",
        gram,
        detail=(
            f"Diagonal ≈ 1 (unit columns); the largest off-diagonal magnitude is "
            f"{num(max_offdiag)}. Nonzero off-diagonals ARE superposition — two features "
            f"partly point along the same neuron direction."
        ),
    )

    # 3. A sparse feature vector: only ~(1 - sparsity) of features are active.
    active = rng.random(n_features) > sparsity
    if not active.any():  # guarantee at least one active feature for a meaningful demo
        active[int(rng.integers(n_features))] = True
    h = np.where(active, rng.random(n_features), 0.0)
    x = W @ h
    t.add(
        "Encode a SPARSE feature vector: x = W·h",
        f"h = {arr(h)}\n→ activations x = {arr(x)}",
        x,
        detail=(
            f"{int(active.sum())}/{n_features} features are active. Each of the "
            f"{n_neurons} neurons in x is a mixture of several features — polysemantic "
            f"by construction, exactly as observed in real models."
        ),
    )

    # 4. Recover features with a simple linear read-out: ĥ = Wᵀ·x.
    h_hat = W.T @ x
    recon_error = float(np.linalg.norm(h_hat - h))
    t.add(
        "Decode / recover: ĥ = Wᵀ·x",
        f"ĥ = {arr(h_hat)}\nreconstruction error ‖ĥ − h‖ = {num(recon_error)}",
        h_hat,
        detail=(
            "Because the true features are sparse, this crude linear read-out recovers "
            "them approximately despite the superposition. That is the core intuition "
            "behind sparse autoencoders / dictionary learning extracting monosemantic "
            "features from a polysemantic model."
        ),
    )

    t.meta["max_offdiag"] = max_offdiag
    t.meta["recon_error"] = recon_error
    t.result = h_hat
    return t


def superposition(
    n_features: int,
    n_neurons: int,
    sparsity: float = 0.7,
    seed: int = 0,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> np.ndarray:
    """Run the toy superposition model, returning the recovered vector ``ĥ``.

    Set ``explain=True`` to print the encode/interference/decode trace.
    """
    t = superposition_trace(n_features, n_neurons, sparsity=sparsity, seed=seed)
    return t.render(level) if explain else t.result
