"""A scalar-valued autograd engine — OptimumAI's take on Karpathy's micrograd.

A :class:`Value` wraps a single number and remembers how it was produced. Every
arithmetic operation builds a node in a computation DAG and stores a local
``_backward`` closure. Calling :meth:`Value.backward` walks the graph in reverse
topological order, seeds the output gradient to 1, and applies the chain rule at
each node — exactly the reverse-mode autodiff that powers every deep-learning
framework.

The twist: :meth:`Value.backward_trace` records *where every gradient came from*,
so you can watch the chain rule flow backwards through the graph.

    >>> from optimumai.autograd import Value
    >>> a = Value(2.0, label="a"); b = Value(-3.0, label="b")
    >>> L = (a * b + a).tanh(); L.label = "L"
    >>> L.backward()
    >>> a.grad, b.grad          # dL/da, dL/db
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace

# Human-readable local derivative rules, keyed by the op that produced a node.
_LOCAL_RULE = {
    "+": "∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1  →  gradient copies straight through",
    "*": "∂(a·b)/∂a = b, ∂(a·b)/∂b = a  →  each factor scales by the other",
    "**": "∂(xⁿ)/∂x = n·xⁿ⁻¹",
    "tanh": "∂tanh(x)/∂x = 1 − tanh²(x)",
    "relu": "∂relu(x)/∂x = 1 if x>0 else 0  →  the gate is open or shut",
    "exp": "∂eˣ/∂x = eˣ (= the output itself)",
    "log": "∂ln(x)/∂x = 1/x",
}


class Value:
    """A differentiable scalar node in an autograd graph."""

    def __init__(
        self, data: float, _children: Iterable[Value] = (), _op: str = "", label: str = ""
    ):
        self.data = float(data)
        self.grad = 0.0
        self.label = label
        self._op = _op
        self._prev = tuple(_children)  # ordered so non-commutative ops trace correctly
        self._backward = lambda: None

    # ------------------------------------------------------------------ arithmetic
    def _wrap(self, other: Value | float) -> Value:
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other: Value | float) -> Value:
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> Value:
        if not isinstance(exponent, (int, float)):
            raise TypeError("only int/float powers are supported")
        out = Value(self.data**exponent, (self,), "**")

        def _backward() -> None:
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        out._op = f"**{exponent}"
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward() -> None:
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ---- dunder conveniences built from the primitives above -----------------
    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * self._wrap(other) ** -1

    def __rtruediv__(self, other):
        return self._wrap(other) * self**-1

    def __repr__(self) -> str:
        name = f"{self.label}=" if self.label else ""
        return f"Value({name}data={num(self.data)}, grad={num(self.grad)})"

    # ------------------------------------------------------------- graph utilities
    def _topo(self) -> list[Value]:
        """Return nodes in topological order (dependencies before dependents)."""
        order: list[Value] = []
        seen: set[int] = set()

        def build(node: Value) -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            for child in node._prev:
                build(child)
            order.append(node)

        build(self)
        return order

    def zero_grad(self) -> None:
        """Reset the gradient of every node reachable from this one."""
        for node in self._topo():
            node.grad = 0.0

    # ------------------------------------------------------------------ backward
    def backward(self) -> None:
        """Run reverse-mode autodiff: seed grad=1 and apply the chain rule."""
        topo = self._topo()
        for node in topo:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def backward_trace(self) -> Trace:
        """Like :meth:`backward`, but record how each gradient is produced.

        For every node processed (output → inputs) we snapshot its children's
        gradients before and after applying the local ``_backward``, so the delta
        we log is the *exact* contribution the chain rule pushed to each child.
        """
        topo = self._topo()
        for node in topo:
            node.grad = 0.0
        self.grad = 1.0

        t = Trace(
            op="backprop",
            formula="∂L/∂x = Σ (∂L/∂out)·(∂out/∂x)   — the chain rule, applied in reverse",
            complexity="O(V+E) over the computation graph",
            why_ai=[
                "How every neural network learns: gradients of the loss w.r.t. each weight",
                "Reverse-mode autodiff — the engine inside PyTorch, JAX, and TensorFlow",
                "Karpathy's micrograd distilled: a DAG of scalars + one chain-rule pass",
            ],
            meta={"nodes": len(topo)},
        )
        t.add(
            "Seed the output gradient",
            f"∂L/∂L = 1   (L = {num(self.data)})",
            1.0,
            detail="Backprop starts by asking how the loss changes with respect to itself.",
        )

        for node in reversed(topo):
            if not node._prev:
                continue
            before = [child.grad for child in node._prev]
            node._backward()
            for child, prev_grad in zip(node._prev, before, strict=True):
                delta = child.grad - prev_grad
                child_name = child.label or f"·{child._op or 'leaf'}"
                node_name = node.label or node._op or "node"
                rule = _LOCAL_RULE.get(node._op.rstrip("0123456789.-"), "")
                if node._op.startswith("**"):
                    rule = _LOCAL_RULE["**"]
                t.add(
                    f"{node_name} → {child_name}",
                    f"{child_name}.grad += {num(delta)}   "
                    f"(upstream ∂L/∂{node_name} = {num(node.grad)}, via '{node._op}')",
                    child.grad,
                    detail=rule,
                )

        leaves = [n for n in topo if not n._prev]
        summary = ", ".join(f"{(n.label or '?')}={num(n.grad)}" for n in leaves)
        t.add("Leaf gradients", summary or "(none)", None,
              detail="These are the numbers an optimizer uses to update parameters.")
        t.result = self.data
        return t

    def backprop(
        self, explain: bool = False, level: str | ExplainLevel = ExplainLevel.INTERMEDIATE
    ) -> Value:
        """Convenience: run backward, optionally rendering the chain-rule trace."""
        if explain:
            self.backward_trace().render(level)
        else:
            self.backward()
        return self
