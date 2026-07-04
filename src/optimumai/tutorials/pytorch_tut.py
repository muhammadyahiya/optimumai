"""Learn PyTorch by building each idea on OptimumAI's own numpy/autograd engine.

Every concept below runs twice: once as a tiny numpy/``optimumai`` demo that
executes on the base install and *proves* the idea with real printed numbers,
and once as the equivalent real ``torch`` code, shown so you can read the
actual PyTorch API even if ``torch`` is not installed. When ``torch`` *is*
installed, those cells run too — so the tutorial doubles as a torch smoke test.

    from optimumai.tutorials import get_tutorial
    get_tutorial("pytorch").run()
"""

from __future__ import annotations

from optimumai.tutorials.core import Tutorial


def build() -> Tutorial:
    t = Tutorial(
        name="pytorch",
        title="PyTorch, built from first principles",
        summary="Every PyTorch idea, first on OptimumAI's numpy/autograd engine, then real torch.",
    )

    # ------------------------------------------------------------------ tensors
    t.md(
        "## 1. Tensors\n\n"
        "A tensor is just a numpy array with a fancier name: a block of numbers "
        "plus a shape and a dtype. `torch.tensor` and `np.array` are, for our "
        "purposes, the same object — same `.shape`, same broadcasting rules, "
        "same elementwise ops. Learn the numpy version and you already know the "
        "torch version."
    )
    t.code(
        """
import numpy as np

x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("shape:", x.shape, "dtype:", x.dtype)
print("x + 10 (broadcast):\\n", x + 10)
print("row-wise sum:", x.sum(axis=1))
print("matmul x @ x.T:\\n", x @ x.T)
""",
        note="numpy: shape, dtype, ops, broadcasting",
    )
    t.code(
        """
import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("shape:", x.shape, "dtype:", x.dtype)
print("x + 10 (broadcast):\\n", x + 10)
print("row-wise sum:", x.sum(dim=1))
print("matmul x @ x.T:\\n", x @ x.T)
""",
        note="the real PyTorch — identical shape/dtype/broadcast/matmul story",
        requires=("torch",),
    )

    # ------------------------------------------------------------------ autograd
    t.md(
        "## 2. Autograd\n\n"
        "PyTorch's 'magic' — call `.backward()` and every parameter grows a "
        "`.grad` — is just reverse-mode autodiff over a graph built during the "
        "forward pass. OptimumAI's `Value` (see `optimumai.autograd.value`, and "
        "the guided walkthrough in `optimumai.foundations.pytorch_foundations`) "
        "is a scalar version of exactly this: every op records its inputs and a "
        "local `_backward` closure, and `.backward()` walks the graph in reverse "
        "topological order applying the chain rule."
    )
    t.code(
        """
from optimumai.autograd import Value

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
L = a * b + a  # L = a*b + a
L.backward()
print(f"L.data={L.data}, dL/da={a.grad}, dL/db={b.grad}")
""",
        note="optimumai: build a scalar graph and call .backward()",
    )
    t.code(
        """
import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-3.0, requires_grad=True)
L = a * b + a
L.backward()
print(f"L.item()={L.item()}, dL/da={a.grad.item()}, dL/db={b.grad.item()}")
""",
        note="the real PyTorch — requires_grad=True leaves + L.backward()",
        requires=("torch",),
    )

    # -------------------------------------------------------------- linear layer
    t.md(
        "## 3. A linear layer\n\n"
        "`nn.Linear(d_in, d_out)` is nothing but `y = x @ W.T + b` with `W` and "
        "`b` tracked for gradients. Below we build the exact same affine map "
        "with a plain numpy weight matrix and bias vector."
    )
    t.code(
        """
import numpy as np

rng = np.random.default_rng(0)
d_in, d_out = 4, 3
W = rng.standard_normal((d_in, d_out))
b = np.zeros(d_out)
x = rng.standard_normal(d_in)

y = x @ W + b
print("y = x @ W + b ->", np.round(y, 4))
""",
        note="numpy: a linear layer is y = x @ W + b",
    )
    t.code(
        """
import torch
import torch.nn as nn

torch.manual_seed(0)
layer = nn.Linear(in_features=4, out_features=3)
x = torch.randn(4)
y = layer(x)
print("y = layer(x) ->", y)
""",
        note="the real PyTorch — nn.Linear is this affine map plus autograd",
        requires=("torch",),
    )

    # --------------------------------------------------------- loss + optimizer
    t.md(
        "## 4. Loss + optimizer step\n\n"
        "Training is: compute a scalar loss, call `.backward()` to fill in "
        "gradients, then let an optimizer nudge the parameters downhill. "
        "`optimumai.optimization` implements `SGD` and `Adam` over `Value` "
        "parameters exactly the way `torch.optim.SGD`/`Adam` do over tensors."
    )
    t.code(
        """
from optimumai.autograd import Value
from optimumai.optimization import SGD

# Minimize f(w) = (w - 3)**2 with one SGD step.
w = Value(0.0, label="w")
opt = SGD([w], lr=0.1)

loss = (w - 3.0) ** 2
loss.backward()
before = w.data
opt.step()
print(f"loss={loss.data}, w before={before}, w after one SGD step={w.data}")
""",
        note="optimumai: a toy loss, .backward(), then optimizer.step()",
    )
    t.code(
        """
import torch

w = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1)

loss = (w - 3.0) ** 2
opt.zero_grad()
loss.backward()
before = w.item()
opt.step()
print(f"loss={loss.item()}, w before={before}, w after one SGD step={w.item()}")
""",
        note="the real PyTorch — torch.optim.SGD does the identical update",
        requires=("torch",),
    )

    # ------------------------------------------------------------- training loop
    t.md(
        "## 5. The full training loop\n\n"
        "Every deep learning training loop is the same four lines: predict, "
        "compute loss, backward, step. Below we train an `optimumai.neural_networks` "
        "`MLP` (built entirely from `Value`s) on a tiny regression toy and watch "
        "the loss fall — then the identical loop, written in canonical torch."
    )
    t.code(
        """
from optimumai.autograd import Value
from optimumai.neural_networks import MLP
from optimumai.optimization import SGD

# Toy dataset: y = 2*x0 - x1 (2 features -> 1 output).
xs = [[2.0, 1.0], [-1.0, 0.5], [0.5, -2.0], [1.5, 1.5]]
ys = [3.0, -2.5, 2.5, 1.5]

model = MLP(n_in=2, n_outs=[4, 1], seed=0)
opt = SGD(model.parameters(), lr=0.02)

losses = []
for epoch in range(50):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys, strict=True)) * (1.0 / len(xs))
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()
    opt.step()
    losses.append(loss.data)

print(f"loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f}")
print("loss decreased:", losses[-1] < losses[0])
""",
        note="optimumai: predict -> loss -> backward -> step, MLP on a toy",
    )
    t.code(
        """
import torch
import torch.nn as nn

torch.manual_seed(0)
xs = torch.tensor([[2.0, 1.0], [-1.0, 0.5], [0.5, -2.0], [1.5, 1.5]])
ys = torch.tensor([3.0, -2.5, 2.5, 1.5]).unsqueeze(1)

model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.02)
loss_fn = nn.MSELoss()

for epoch in range(50):
    pred = model(xs)
    loss = loss_fn(pred, ys)
    opt.zero_grad()
    loss.backward()
    opt.step()
""",
        note="the real PyTorch — the canonical pred/loss/backward/step loop",
        requires=("torch",),
    )

    # --------------------------------------------------------------- device/GPU
    t.md(
        "## 6. Device / GPU\n\n"
        "OptimumAI's `Value` and numpy arrays only ever run on the CPU — there is "
        "no device concept. Real PyTorch tensors and modules carry a `.device` "
        "and can be moved to an accelerator with `.to(device)` / `.cuda()`; every "
        "tensor in an operation must live on the same device."
    )
    t.code(
        """
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(3, 3).to(device)
model = torch.nn.Linear(3, 3).to(device)
y = model(x)
print("device:", y.device)
""",
        note="the real PyTorch — .to(device)/.cuda() move tensors + modules together",
        requires=("torch",),
    )

    # --------------------------------------------------------------- save/load
    t.md(
        "## 7. Save / load\n\n"
        "A trained model is just its parameter values. PyTorch serializes a "
        "`state_dict` (a plain dict of tensor -> value) with `torch.save`/`torch.load` "
        "so you can checkpoint training and restore it later — the same idea as "
        "pickling the `.data` of every `Value` in `model.parameters()`."
    )
    t.code(
        """
import torch
import torch.nn as nn

model = nn.Linear(3, 3)
torch.save(model.state_dict(), "linear.pt")

restored = nn.Linear(3, 3)
restored.load_state_dict(torch.load("linear.pt"))
print("weights match:", torch.equal(model.weight, restored.weight))
""",
        note="the real PyTorch — torch.save/load a state_dict checkpoint",
        requires=("torch",),
    )

    t.md(
        "## Where this lives in OptimumAI\n\n"
        "`optimumai.foundations.pytorch_foundations.pytorch_autograd_trace()` walks "
        "this exact `Value` <-> `torch.autograd` mapping step by step with "
        "`explain=True`-style narration — run it for a guided replay of section 2."
    )

    return t
