# Learn PyTorch

PyTorch is not installed in this environment, so every PyTorch snippet on this
page is **reference code** — accurate, idiomatic, and written the way you'd
actually write it, but not executed here. Every idea is paired with a runnable
OptimumAI equivalent (pure NumPy / OptimumAI's own `Value` autograd engine) so
you can see the *same computation actually run*, with real printed numbers,
even without a GPU or `torch` installed.

!!! tip "Run this interactively"
    ```bash
    optimumai tutorial pytorch
    ```

    or from Python:

    ```python
    from optimumai.tutorials import get_tutorial
    get_tutorial("pytorch").run()
    ```

    Every concept below runs twice inside that tutorial: once as an
    OptimumAI/NumPy demo (always executes), and once as the real `torch` code
    (executes too, *if* `torch` happens to be installed on your machine —
    otherwise it's shown, unexecuted, so the lesson stays fully readable).

!!! warning "Labeling convention on this page"
    Blocks marked **PyTorch (reference, not run here)** are real, correct torch
    API usage — copy them into an environment with `torch` installed. Blocks
    marked **OptimumAI (runnable)** were executed in this docs environment
    (NumPy 2.4, no torch) while writing this page.

---

## 1. Tensors — the same idea as a NumPy array, plus a device and a graph

**Intuition.** A `torch.Tensor` is, for almost every practical purpose, a
NumPy array: a shape, a dtype, and a block of numbers, with the same
broadcasting rules and the same `@` for matrix multiply. Learn NumPy (see
[Learn NumPy](learn-numpy.md)) and you already know 90% of the tensor API.
Two things a tensor adds on top: it can track gradients through operations
(autograd, next section), and it can live on a GPU (`device`, later section).

```python
# OptimumAI (runnable) — the numpy version
import numpy as np

x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("shape:", x.shape, "dtype:", x.dtype)
print("x + 10 (broadcast):\n", x + 10)
print("row-wise sum:", x.sum(axis=1))
print("matmul x @ x.T:\n", x @ x.T)
```

```python
# PyTorch (reference, not run here) — line-for-line the same operations
import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("shape:", x.shape, "dtype:", x.dtype)
print("x + 10 (broadcast):\n", x + 10)
print("row-wise sum:", x.sum(dim=1))     # note: dim=, not axis=
print("matmul x @ x.T:\n", x @ x.T)
```

The API is nearly a 1:1 rename: NumPy's `axis=` becomes torch's `dim=`,
`np.array` becomes `torch.tensor`, and `.reshape`/indexing/broadcasting all
behave identically. `torch.from_numpy(arr)` and `tensor.numpy()` convert
between the two without copying data (when on CPU), which is why PyTorch code
so often interleaves NumPy preprocessing with tensor computation.

---

## 2. Autograd — `requires_grad`, `.backward()`, `.grad`

**Intuition.** PyTorch's headline feature — write ordinary math, call
`.backward()`, and every parameter grows a `.grad` — is not magic. It's
**reverse-mode automatic differentiation** over a graph built while your code
runs: every operation on a tensor with `requires_grad=True` records itself
(its inputs and how to locally differentiate) into a dynamic computation
graph. Calling `.backward()` walks that graph backward from the output,
seeding the output's gradient to `1.0`, and applies the chain rule at each
recorded node, accumulating into each leaf's `.grad`.

```python
# PyTorch (reference, not run here)
import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-3.0, requires_grad=True)
L = a * b + a
L.backward()
print(f"L.item()={L.item()}, dL/da={a.grad.item()}, dL/db={b.grad.item()}")
# L.item()=-4.0, dL/da=-2.0, dL/db=2.0
```

### OptimumAI's `Value` is the same idea, and you can read the source

`optimumai.autograd.value.Value` (see
[`src/optimumai/autograd/value.py`](https://github.com/muhammadyahiya/optimumai/blob/main/src/optimumai/autograd/value.py))
is a **scalar** implementation of exactly this mechanism — a teaching-sized
version of what `torch.autograd` does at tensor scale. Reading its ~250 lines
is one of the fastest ways to actually understand what "autograd" means,
because there's no C++/CUDA dispatch machinery in the way — just Python.

```python
# OptimumAI (runnable)
from optimumai.autograd import Value

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
L = a * b + a
L.backward()
print(f"L.data={L.data}, dL/da={a.grad}, dL/db={b.grad}")
# L.data=-4.0, dL/da=-2.0, dL/db=2.0
```

Notice the numbers match the torch example exactly — same computation, same
gradients, same algorithm. The mapping is one-to-one:

| OptimumAI `Value`             | PyTorch                                  |
| ------------------------------ | ----------------------------------------- |
| `Value(x)`                     | `torch.tensor(x, requires_grad=True)`     |
| ops build a DAG of `Value`s     | the *dynamic* computational graph, rebuilt every forward pass |
| `loss.backward()`              | `torch.autograd.backward(loss)` / `loss.backward()` |
| `node.grad`                    | `tensor.grad`                             |
| a leaf `Value` (no parents)    | `nn.Parameter` (`requires_grad=True` by default) |

Under the hood, `Value.backward()` does exactly what the paragraph above
described: it computes a topological order of the graph, zeroes every
gradient, seeds `self.grad = 1.0`, then walks the nodes in reverse, calling
each node's stored `_backward` closure — the literal chain rule, one line per
operation (`+` copies gradient through both branches; `*` scales each
branch's gradient by the *other* branch's value; `tanh` scales by
`1 - tanh(x)**2`; and so on). There's a narrated, step-by-step version of this
exact walk in `optimumai.foundations.pytorch_foundations.pytorch_autograd_trace()`,
which explicitly annotates each step with its PyTorch equivalent — worth
running once with `explain=True`-style rendering if you want the guided replay:

```python
# OptimumAI (runnable)
from optimumai.foundations.pytorch_foundations import pytorch_autograd_trace

trace = pytorch_autograd_trace()
trace.render("engineer")   # prints the step-by-step narration
```

!!! note "Why the graph is 'dynamic'"
    Because the graph is rebuilt every forward pass (running the ops *is*
    building the graph — there's no separate "compile" step), ordinary Python
    control flow — `if`, `for`, recursion — works inside a PyTorch model
    exactly as you'd expect. This is called **define-by-run**, in contrast to
    older "define-then-run" frameworks (TensorFlow 1.x) that built a static
    graph up front.

### `no_grad()` and detaching

Not every tensor operation needs to build graph history — evaluation,
inference, and manual weight updates all want the *compute* without the
*bookkeeping* (which costs memory and time).

```python
# PyTorch (reference, not run here)
with torch.no_grad():
    y = model(x)          # no graph is built; nothing here gets a .grad

z = y.detach()             # a tensor sharing y's data, but severed from the graph
```

---

## 3. `nn.Module` and `nn.Linear`

**Intuition.** `nn.Linear(d_in, d_out)` *is* the affine map `y = x @ W.T + b`
— nothing more — except `W` and `b` are `nn.Parameter` tensors
(`requires_grad=True` leaves that autograd tracks and an optimizer updates).
`nn.Module` is the base class that (a) tracks all the parameters and
sub-modules you assign as attributes, so `.parameters()` finds them
automatically, and (b) defines `forward()` as the thing that runs when you
call the module like a function.

```python
# PyTorch (reference, not run here)
import torch
import torch.nn as nn

torch.manual_seed(0)
layer = nn.Linear(in_features=4, out_features=3)
x = torch.randn(4)
y = layer(x)          # calls layer.forward(x) under the hood
print(y)
```

```python
# OptimumAI (runnable) — the identical affine map, written explicitly
import numpy as np

rng = np.random.default_rng(0)
d_in, d_out = 4, 3
W = rng.standard_normal((d_in, d_out))
b = np.zeros(d_out)
x = rng.standard_normal(d_in)

y = x @ W + b
print("y = x @ W + b ->", np.round(y, 4))
```

A real model subclasses `nn.Module`, declares its layers in `__init__`, and
defines the forward computation:

```python
# PyTorch (reference, not run here)
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

model = MLP(2, 4, 1)
print(sum(p.numel() for p in model.parameters()))   # total scalar parameters
```

Or the equivalent shortcut for a simple feedforward stack, `nn.Sequential`:

```python
# PyTorch (reference, not run here)
model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
```

OptimumAI's `optimumai.neural_networks.MLP` is the same idea (a stack of
linear layers + nonlinearity) built directly on `Value`, with each weight a
leaf `Value` instead of a tensor:

```python
# OptimumAI (runnable)
from optimumai.neural_networks import MLP

model = MLP(n_in=2, n_outs=[4, 1], seed=1)     # 2 -> 4 (tanh) -> 1
print("total scalar parameters:", len(model.parameters()))   # 17
```

---

## 4. Loss functions

**Intuition.** A loss function collapses "how wrong was the prediction" into
one scalar number — the thing `.backward()` differentiates. The loss you pick
encodes what kind of wrongness you care about.

```python
# PyTorch (reference, not run here)
import torch.nn as nn

mse = nn.MSELoss()                 # regression: mean squared error
ce = nn.CrossEntropyLoss()         # multi-class classification (takes raw logits)
bce = nn.BCEWithLogitsLoss()       # binary classification (takes raw logits)
```

- `nn.MSELoss` — regression, penalizes squared distance from the target.
  Sensitive to outliers (squaring amplifies large errors).
- `nn.CrossEntropyLoss` — multi-class classification. Takes **raw logits**
  (not probabilities) and internally applies `log_softmax` + negative
  log-likelihood — don't apply your own softmax before passing predictions in.
- `nn.BCEWithLogitsLoss` — binary classification, same "give it raw logits"
  contract, internally applies a numerically stable sigmoid + binary
  cross-entropy in one fused op.

The OptimumAI equivalent is just Python arithmetic over `Value`s — mean
squared error is `sum((pred - target) ** 2) / n`, written out explicitly
rather than hidden behind a class, which is exactly what you see in the
training loop below.

---

## 5. Optimizers — `torch.optim`, SGD and Adam

**Intuition.** Once you have a gradient, an optimizer decides *how* to use it
to update a parameter. The simplest rule, **SGD**, just steps downhill:
`param -= lr * param.grad`. **Adam** adds momentum (an exponential moving
average of past gradients) and a per-parameter adaptive learning rate (an
exponential moving average of past *squared* gradients), which in practice
converges faster and is more forgiving of the learning-rate choice — it's the
default starting point for most deep learning training.

```python
# PyTorch (reference, not run here)
import torch

w = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1)

loss = (w - 3.0) ** 2
opt.zero_grad()      # gradients accumulate by default — clear them first
loss.backward()
opt.step()            # apply the update rule using w.grad
print(w.item())       # moved toward 3.0
```

```python
# OptimumAI (runnable) — the identical update, over a Value
from optimumai.autograd import Value
from optimumai.optimization import SGD

w = Value(0.0, label="w")
opt = SGD([w], lr=0.1)

loss = (w - 3.0) ** 2
loss.backward()
before = w.data
opt.step()
print(f"loss={loss.data}, w before={before}, w after one SGD step={w.data}")
# loss=9.0, w before=0.0, w after one SGD step=0.6
```

`optimumai.optimization` also implements `Adam` over `Value` parameters with
the same moment-tracking logic as `torch.optim.Adam` — read
`src/optimumai/optimization/optimizers.py` alongside `torch.optim.Adam`'s docs
if you want to see the bias-corrected first/second moment update side by side
with a from-scratch implementation.

!!! warning "`opt.zero_grad()` — the step everyone forgets once"
    PyTorch **accumulates** gradients into `.grad` by default (`+=`, not `=`)
    — this is what makes gradient accumulation over multiple mini-batches
    possible. But it means if you forget `opt.zero_grad()` before
    `loss.backward()`, gradients from the previous step silently add onto the
    new ones. Call it every iteration unless you specifically want
    accumulation.

---

## 6. The canonical training loop

**Intuition.** Every supervised training loop in deep learning is the same
four steps, repeated: **predict → compute loss → backward → step.** Once you
can say those four words, you can read almost any training script, regardless
of the model.

```python
# PyTorch (reference, not run here)
import torch
import torch.nn as nn

torch.manual_seed(0)
xs = torch.tensor([[2.0, 1.0], [-1.0, 0.5], [0.5, -2.0], [1.5, 1.5]])
ys = torch.tensor([3.0, -2.5, 2.5, 1.5]).unsqueeze(1)   # shape (4, 1)

model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.02)
loss_fn = nn.MSELoss()

for epoch in range(50):
    pred = model(xs)             # 1. predict
    loss = loss_fn(pred, ys)     # 2. compute loss
    opt.zero_grad()
    loss.backward()              # 3. backward
    opt.step()                   # 4. step
```

```python
# OptimumAI (runnable) — the identical loop, over Value/MLP, and it actually runs
from optimumai.neural_networks import MLP
from optimumai.optimization import SGD

xs = [[2.0, 1.0], [-1.0, 0.5], [0.5, -2.0], [1.5, 1.5]]
ys = [3.0, -2.5, 2.5, 1.5]

model = MLP(n_in=2, n_outs=[4, 1], seed=0)
opt = SGD(model.parameters(), lr=0.02)

losses = []
for epoch in range(50):
    preds = [model(x) for x in xs]                                          # 1. predict
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys, strict=True)) / len(xs)  # 2. loss
    for p in model.parameters():
        p.grad = 0.0                                                        # (zero_grad)
    loss.backward()                                                          # 3. backward
    opt.step()                                                               # 4. step
    losses.append(loss.data)

print(f"loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f}")
print("loss decreased:", losses[-1] < losses[0])
```

Run this and you'll see the loss actually fall — the exact same algorithm
`torch.optim.SGD` runs, just over scalar `Value`s instead of tensors.

---

## 7. `device` and CUDA

**Intuition.** A tensor and a model both live *somewhere* — CPU RAM or a
GPU's VRAM — and every tensor participating in one operation must live on the
same device. `.to(device)` moves a tensor or a module's parameters onto a
target device; the standard idiom checks availability first so the same code
runs on a laptop or a GPU box.

```python
# PyTorch (reference, not run here)
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(3, 3).to(device)
model = torch.nn.Linear(3, 3).to(device)
y = model(x)
print("device:", y.device)
```

OptimumAI's `Value` and NumPy arrays only ever run on CPU — there's no device
concept, which is exactly why they're a safe teaching substrate for anyone
without a GPU: the *math* is identical, only the hardware backend differs.
`optimumai.foundations` has a separate module on the CUDA execution/memory
model itself (`optimumai learn cuda_matmul`) if you want to go one level
deeper into *why* GPUs are fast, independent of PyTorch's API for using one.

---

## 8. `DataLoader`

**Intuition.** Real datasets don't fit in memory as one tensor, and you
usually want shuffled mini-batches, not the whole dataset at once. `Dataset` +
`DataLoader` is PyTorch's abstraction for "give me batches, optionally
shuffled, optionally in parallel worker processes."

```python
# PyTorch (reference, not run here)
import torch
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = torch.tensor(xs, dtype=torch.float32)
        self.ys = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

dataset = ToyDataset(
    xs=[[2.0, 1.0], [-1.0, 0.5], [0.5, -2.0], [1.5, 1.5]],
    ys=[3.0, -2.5, 2.5, 1.5],
)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for epoch in range(3):
    for xb, yb in loader:      # iterate mini-batches
        ...                     # predict / loss / backward / step, per batch
```

A `Dataset` only needs `__len__` and `__getitem__`; `DataLoader` handles
batching, shuffling (a new random order each epoch when `shuffle=True`), and
optional multiprocessing (`num_workers=`) for loading/preprocessing in
parallel with GPU compute. In the toy training loop above, `xs`/`ys` stood in
for what a `DataLoader` would normally hand you batch by batch — at real
dataset sizes, you always want the `Dataset`/`DataLoader` machinery instead of
holding everything in one Python list.

---

## 9. Save / load — `state_dict`

**Intuition.** A trained model *is* its parameter values — nothing else needs
to survive a checkpoint. PyTorch's `state_dict()` is a plain
`OrderedDict` mapping each parameter's name to its tensor value; saving and
loading that dict is how you checkpoint and resume training, or ship a
trained model.

```python
# PyTorch (reference, not run here)
import torch
import torch.nn as nn

model = nn.Linear(3, 3)
torch.save(model.state_dict(), "linear.pt")

restored = nn.Linear(3, 3)          # must have the same architecture
restored.load_state_dict(torch.load("linear.pt"))
print(torch.equal(model.weight, restored.weight))   # True
```

`state_dict()` holds only *values*, not architecture — `load_state_dict`
requires you to have already constructed a module with matching parameter
shapes/names. This is also exactly the file format LoRA checkpoints exploit:
because only the small adapter matrices are trained (see
[Learn fine-tuning: LoRA](learn-finetuning.md#lora-w--w0--ba)), a LoRA
`state_dict` is tiny compared to the full base model's.

The conceptual equivalent for a `Value`-based model is just pickling (or
JSON-dumping) `[p.data for p in model.parameters()]` — there's no framework
magic in either case, just "write the numbers to disk, read them back into a
model with the same shape."

---

## 10. Eval vs. train mode, and `no_grad`

**Intuition.** Some layers behave differently at training time vs. inference
time — most commonly **dropout** (randomly zeroes activations during
training as a regularizer, does nothing at inference) and **batch
normalization** (uses the current batch's statistics during training, but
frozen running statistics at inference). `model.train()` and `model.eval()`
tell those layers which behavior to use; they don't affect autograd tracking.

```python
# PyTorch (reference, not run here)
model.train()          # dropout active, batchnorm uses batch stats
# ... training loop ...

model.eval()            # dropout off, batchnorm uses running stats
with torch.no_grad():   # also skip building the autograd graph — inference is cheaper
    predictions = model(x_test)
```

Two independent switches, easy to conflate:

- `model.train()` / `model.eval()` — changes layer *behavior* (dropout,
  batchnorm), has nothing to do with gradients.
- `torch.no_grad()` — changes whether autograd *tracks* operations at all
  (saves memory and time), has nothing to do with layer behavior.

You almost always want both together for evaluation: `model.eval()` plus a
`with torch.no_grad():` block. Forgetting `.eval()` before validation/test is
one of the most common silent bugs in PyTorch code — the model still runs and
gives an answer, it's just a *wrong-mode* answer (e.g. still applying
dropout), which quietly makes validation metrics noisy or pessimistic.

---

## Where this lives in OptimumAI

Every mapping on this page — `Value` ↔ `torch.autograd`, `SGD`/`Adam` over
`Value` ↔ `torch.optim`, `MLP` ↔ `nn.Sequential` — is deliberate: OptimumAI's
`autograd`, `optimization`, and `neural_networks` packages implement the same
algorithms PyTorch does, at a scale small enough to read start to finish in
one sitting. If you want the fully guided, step-by-step version of the
`Value`/`torch.autograd` mapping specifically, run:

```python
from optimumai.foundations.pytorch_foundations import pytorch_autograd_trace
pytorch_autograd_trace().render("engineer")
```

or from the CLI:

```bash
optimumai learn pytorch
```

!!! note "Where to go next"
    [Learn fine-tuning](learn-finetuning.md) picks up exactly where this page
    ends: taking a pretrained model (built from these same `nn.Module`
    building blocks) and adapting it with LoRA, QLoRA, DPO, or full RLHF.
