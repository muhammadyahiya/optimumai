"""What JAX does under the hood — one idea: transforming pure functions.

PyTorch gives you an autograd engine. JAX gives you something more abstract and,
once it clicks, simpler: a set of *composable transformations* that take a
**pure function** and return a new pure function.

    grad(f)   → a function computing f's derivative (reverse-mode autodiff)
    vmap(f)   → a function that runs f over a batch axis, no Python loop
    jit(f)    → a function compiled to fused machine code by XLA
    pmap(f)   → a function sharded across devices

Because they are all just ``function → function``, they compose freely:
``grad(jit(vmap(f)))`` is a perfectly ordinary thing to write. The one
non-negotiable is *purity*: f must have no side effects and depend only on its
inputs. Purity is what lets JAX **trace** f (run it once with abstract values to
record the operations) and hand that trace to XLA to compile.

This module teaches ``grad`` / ``vmap`` / pytrees with plain numpy — no compiler,
no ``jax`` dependency — so you can see the ideas rather than the machinery.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.trace import Trace

_JAX_WHY = [
    "JAX = composable transformations (grad, jit, vmap, pmap) of PURE functions.",
    "Purity + tracing is what lets XLA fuse and compile the function; "
    "you can freely stack them, e.g. grad(jit(vmap(f))).",
    "Real models are pytrees of params; every transformation operates over pytrees.",
]


def grad_trace(f: Callable[[float], float], x: float, h: float = 1e-5) -> Trace:
    """Show JAX's ``grad`` idea via a central-difference derivative of ``f`` at ``x``.

    We approximate f'(x) numerically here so the concept is visible without a
    tracer. JAX computes the *exact* derivative by tracing the pure function f
    and running reverse-mode autodiff over that trace — same answer, no round-off.
    """
    x = float(x)
    fx = float(f(x))
    f_plus = float(f(x + h))
    f_minus = float(f(x - h))
    derivative = (f_plus - f_minus) / (2.0 * h)

    t = Trace(
        op="jax.grad",
        formula="grad(f)(x) ≈ (f(x+h) − f(x−h)) / 2h   (central difference)",
        complexity="O(cost of f); exact autodiff is ~2–3× a forward pass",
        why_ai=_JAX_WHY,
        meta={"framework": "jax", "transform": "grad", "h": h},
    )
    t.add(
        "Evaluate the pure function",
        f"f(x) = {num(fx)}  at  x = {num(x)}",
        fx,
        detail=(
            "grad requires f to be pure: same inputs → same outputs, no side "
            "effects. That is exactly what lets JAX trace it once and reuse it."
        ),
    )
    t.add(
        "Probe both sides",
        f"f(x+h) = {num(f_plus)},  f(x−h) = {num(f_minus)}   with h = {num(h)}",
        [f_plus, f_minus],
        detail="Central difference cancels the first-order error, so it is O(h²) accurate.",
    )
    t.add(
        "Form the slope",
        f"(f(x+h) − f(x−h)) / 2h = {num(derivative)}",
        derivative,
        detail=(
            "JAX's grad returns this *exactly* via reverse-mode autodiff on the "
            "traced function — no finite h, no truncation error. grad(f) is itself "
            "a pure function, so grad(grad(f)) gives the second derivative."
        ),
    )
    t.result = derivative
    return t


def vmap_trace(f: Callable[[Any], Any], batch: np.ndarray) -> Trace:
    """Demonstrate ``vmap``: apply pure ``f`` across a batch axis with no Python loop.

    We compute the loop version and the stacked version and show they agree.
    JAX's vmap does the same thing by *pushing a batch dimension through the
    traced function*, letting XLA vectorize it — no interpreter loop at runtime.
    """
    batch = np.asarray(batch, dtype=float)

    t = Trace(
        op="jax.vmap",
        formula="vmap(f)(xs) = stack([f(x) for x in xs])   — but with an added batch axis",
        complexity="O(batch) elementwise, but vectorized (no Python-level loop)",
        why_ai=_JAX_WHY,
        meta={"framework": "jax", "transform": "vmap", "batch_shape": tuple(batch.shape)},
    )
    t.add(
        "The loop version (what you'd write by hand)",
        f"[f(x) for x in {arr(batch)}]",
        None,
        detail=(
            "A Python loop calls f once per element. It works but is slow and "
            "cannot be fused: the interpreter is in the hot path."
        ),
    )

    looped = np.stack([np.asarray(f(x), dtype=float) for x in batch])

    t.add(
        "The vmap version (auto-vectorized)",
        f"vmap(f)({arr(batch)}) = {arr(looped)}",
        looped,
        detail=(
            "vmap adds a leading batch axis and pushes it through f's traced "
            "operations, so f is expressed once over the whole batch. No Python "
            "loop runs at execution time — XLA handles the vectorization."
        ),
    )
    t.add(
        "Equivalence check",
        f"loop result == vmap result  →  {arr(looped)}",
        looped,
        detail="Same numbers, different execution: the loop is unrolled into vectorized ops.",
    )
    t.result = looped
    return t


def _flatten(tree: Any) -> tuple[list[float], Any]:
    """Recursively flatten a pytree into (leaves, structure).

    A pytree is any nesting of dict/list/tuple with scalar leaves. The returned
    ``structure`` is a lightweight spec that :func:`_unflatten` uses to rebuild
    the exact same shape from a flat list of leaves.
    """
    if isinstance(tree, dict):
        keys = list(tree.keys())
        leaves: list[float] = []
        child_structs = []
        for k in keys:
            sub_leaves, sub_struct = _flatten(tree[k])
            leaves.extend(sub_leaves)
            child_structs.append(sub_struct)
        return leaves, ("dict", keys, child_structs)
    if isinstance(tree, (list, tuple)):
        kind = "list" if isinstance(tree, list) else "tuple"
        leaves = []
        child_structs = []
        for item in tree:
            sub_leaves, sub_struct = _flatten(item)
            leaves.extend(sub_leaves)
            child_structs.append(sub_struct)
        return leaves, (kind, child_structs)
    # scalar leaf
    return [float(tree)], ("leaf",)


def _unflatten(structure: Any, leaves: list[float]) -> Any:
    """Rebuild a pytree from its ``structure`` and a flat list of ``leaves``.

    Consumes ``leaves`` left-to-right (matching :func:`_flatten`'s order).
    """
    kind = structure[0]
    if kind == "leaf":
        return leaves.pop(0)
    if kind == "dict":
        _, keys, child_structs = structure
        return {k: _unflatten(s, leaves) for k, s in zip(keys, child_structs, strict=True)}
    if kind == "list":
        _, child_structs = structure
        return [_unflatten(s, leaves) for s in child_structs]
    if kind == "tuple":
        _, child_structs = structure
        return tuple(_unflatten(s, leaves) for s in child_structs)
    raise ValueError(f"unknown pytree node kind: {kind!r}")


def pytree_trace(tree: Any) -> Trace:
    """Flatten a nested dict/list/tuple ``tree`` to leaves + structure, then rebuild.

    Every JAX transformation operates over pytrees: model parameters are nested
    dicts, optimizer state is a pytree, ``grad`` returns a pytree matching the
    inputs. Flatten → transform the flat leaves → unflatten is the pattern.
    """
    t = Trace(
        op="jax.pytree",
        formula="tree ⇄ (leaves, structure)   via tree_flatten / tree_unflatten",
        complexity="O(nodes) to walk the tree",
        why_ai=_JAX_WHY,
        meta={"framework": "jax", "concept": "pytree"},
    )
    t.add(
        "The pytree",
        f"{tree}",
        None,
        detail=(
            "A pytree is any nesting of containers (dict/list/tuple) with numeric "
            "leaves — the natural shape of a model's parameters."
        ),
    )

    leaves, structure = _flatten(tree)
    t.add(
        "Flatten → leaves",
        f"leaves = {arr(np.asarray(leaves))}   ({len(leaves)} leaves)",
        list(leaves),
        detail=(
            "tree_flatten pulls every leaf into a flat list and records the "
            "structure separately. Transformations act on this flat list."
        ),
    )
    t.add(
        "Flatten → structure",
        f"structure = {structure}",
        None,
        detail="The structure (a 'treedef' in JAX) is the recipe for rebuilding the tree.",
    )

    rebuilt = _unflatten(structure, list(leaves))
    t.add(
        "Unflatten → round-trip",
        f"tree_unflatten(structure, leaves) == original  →  {rebuilt == tree}",
        None,
        detail=(
            "jit/XLA never sees your dict — it sees the flat leaves. JAX flattens "
            "before compiling and unflattens the result, which is how "
            "grad(loss)(params) can hand you gradients shaped exactly like params."
        ),
    )

    t.result = list(leaves)
    return t


def demo() -> Trace:
    """Return the reference JAX-foundations trace (vmap of z² over a small batch)."""
    return vmap_trace(lambda z: z**2, np.array([1.0, 2.0, 3.0, 4.0]))
