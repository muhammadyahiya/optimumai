"""What PyTorch does under the hood — reverse-mode autodiff over a dynamic graph.

PyTorch feels magical: you write ordinary Python, call ``loss.backward()``, and
every parameter suddenly has a ``.grad``. There is no magic. PyTorch is a
scalar (well, tensor) autograd engine — the *same idea* as Karpathy's micrograd,
which is exactly OptimumAI's :class:`~optimumai.autograd.Value`.

The mapping is one-to-one:

    OptimumAI Value          PyTorch
    ---------------          -------
    Value(x)                 torch.tensor(x, requires_grad=True)
    ops build the DAG        the DYNAMIC computational graph, rebuilt each forward
    loss.backward()          torch.autograd.backward(loss)
    node.grad                tensor.grad
    a leaf Value             nn.Parameter (requires_grad=True by default)

Understand :class:`Value` and you understand ``torch.autograd``. This module
walks a tiny graph and narrates the PyTorch equivalent at every step.
"""

from __future__ import annotations

from optimumai.autograd import Value
from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def pytorch_autograd_trace() -> Trace:
    """Build a tiny neuron with :class:`Value` and narrate the PyTorch equivalents.

    We compute ``y = tanh(w*x + b)`` and ``loss = (y - target)**2``, then call
    ``loss.backward()``. Every step maps back to a line of PyTorch so you can see
    that ``torch.autograd`` is just reverse-mode autodiff over a dynamic graph.
    """
    target = 0.0

    # Leaves == nn.Parameter. In OptimumAI a Value with no children is a leaf;
    # in PyTorch a tensor with requires_grad=True is a leaf that accumulates grad.
    w = Value(0.5, label="w")
    x = Value(1.5, label="x")
    b = Value(-0.2, label="b")

    t = Trace(
        op="pytorch_autograd",
        formula="loss = (tanh(w·x + b) − target)²   →   loss.backward()",
        complexity="O(V+E) over the dynamic graph (one reverse traversal)",
        why_ai=[
            "PyTorch's 'magic' is just reverse-mode autodiff over a dynamic graph.",
            "Understanding OptimumAI's Value means you understand torch.autograd.",
            "The graph is rebuilt every forward pass, which is why plain Python "
            "control flow (if/for/while) 'just works' inside a PyTorch model.",
        ],
        meta={"framework": "pytorch", "target": target},
    )

    t.add(
        "Create leaf parameters",
        f"w={num(w.data)}, x={num(x.data)}, b={num(b.data)}",
        [w.data, x.data, b.data],
        detail=(
            "Each leaf Value ↔ torch.tensor(..., requires_grad=True). A leaf with "
            "no parents is what PyTorch wraps as nn.Parameter — requires_grad is "
            "True by default there, so gradients accumulate onto it."
        ),
    )

    # Forward pass: running the ops IS the graph construction. PyTorch does the
    # same thing eagerly — every op records itself so backward can replay it.
    z = w * x + b
    z.label = "z"
    y = z.tanh()
    y.label = "y"
    loss = (y - target) ** 2
    loss.label = "loss"

    t.add(
        "Forward pass builds the graph",
        f"z = w·x + b = {num(z.data)};  y = tanh(z) = {num(y.data)}",
        [z.data, y.data],
        detail=(
            "Running each op appends a node to the DAG — this IS PyTorch's dynamic "
            "computational graph, constructed on the fly during the forward pass. "
            "No static graph is compiled ahead of time (contrast: TF1, JAX/XLA)."
        ),
    )

    t.add(
        "Compute the loss",
        f"loss = (y − {num(target)})² = {num(loss.data)}",
        loss.data,
        detail="A scalar loss is the single output we differentiate — the root of backprop.",
    )

    # backward(): reverse traversal, seed grad=1 at the output, chain rule inward.
    loss.backward()

    t.add(
        "Call loss.backward()",
        "seed ∂loss/∂loss = 1, then walk the graph in reverse",
        1.0,
        detail=(
            "loss.backward() ↔ torch.autograd.backward(loss). It topologically "
            "sorts the graph, seeds the output gradient to 1.0, and applies each "
            "node's local derivative in reverse order, accumulating into .grad."
        ),
    )

    t.add(
        "Read the leaf gradients",
        f"w.grad={num(w.grad)}, x.grad={num(x.grad)}, b.grad={num(b.grad)}",
        [w.grad, x.grad, b.grad],
        detail=(
            "our .grad ↔ tensor.grad. An optimizer (SGD/Adam) then does "
            "w -= lr * w.grad. This is the entire learning loop, one step of it."
        ),
    )

    t.result = loss.data
    return t


def pytorch_autograd(
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> float:
    """Return the loss of the demo neuron. Set ``explain=True`` to print the trace."""
    t = pytorch_autograd_trace()
    return t.render(level) if explain else t.result


def demo() -> Trace:
    """Return the reference PyTorch-foundations trace."""
    return pytorch_autograd_trace()
