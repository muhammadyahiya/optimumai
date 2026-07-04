# Learn NumPy

NumPy is the substrate everything else in this ecosystem sits on: matplotlib
plots NumPy arrays, PyTorch tensors copy NumPy's API almost verbatim, and every
fine-tuning library eventually bottoms out in array math. If you can think in
NumPy — shapes, broadcasting, vectorized ops — you can read the internals of
almost any ML codebase.

!!! tip "Run this interactively"
    Every idea on this page also exists as a runnable, narrated tutorial cell:

    ```bash
    optimumai tutorial numpy
    ```

    or from Python:

    ```python
    from optimumai.tutorials import get_tutorial
    get_tutorial("numpy").run()
    ```

    That tutorial executes real NumPy code, cell by cell, with the output printed
    right below it — a good companion to read alongside this page.

All snippets on this page were executed against NumPy 2.4 while writing these
docs (`python -c "import numpy; print(numpy.__version__)"`).

```python
import numpy as np
```

---

## 1. Arrays and dtypes

**Intuition.** A Python list is a bag of pointers to arbitrary objects — each
element could be a different type, living anywhere in memory. A NumPy array
(`ndarray`) is the opposite: one contiguous block of memory holding elements of
a single, fixed type. That uniformity is *why* NumPy is fast — the CPU can
stream through a flat buffer of, say, 8-byte floats instead of chasing pointers.

```python
import numpy as np

a = np.array([1, 2, 3])
print(a.dtype)   # int64 (platform-dependent; int64 on most 64-bit systems)
print(a.shape)   # (3,)
print(a.ndim)    # 1

b = np.array([1, 2, 3], dtype=np.float32)
print(b.dtype)   # float32
```

Every array has three things worth knowing on sight:

- `dtype` — the element type (`int64`, `float32`, `bool`, ...). Mixed-type
  input gets upcast to the most general common type.
- `shape` — a tuple of sizes, one per axis.
- `ndim` — how many axes (`len(shape)`).

Common constructors:

```python
np.zeros((2, 3))        # 2x3 of 0.0
np.ones((2, 3))         # 2x3 of 1.0
np.arange(0, 10, 2)     # [0, 2, 4, 6, 8] — like range(), but an array
np.linspace(0, 1, 5)    # [0. 0.25 0.5 0.75 1.] — n evenly spaced points
```

!!! note "Why dtype matters"
    `dtype` is not just bookkeeping. `np.array([1, 2, 3]) / 2` gives you floats
    even though the input was integers (true division, NumPy 2.x default), but
    two integer arrays combined with `//` stay integers. Silent dtype changes
    are one of the most common sources of "why is my answer wrong" bugs — see
    [gotchas](#9-common-gotchas) below.

---

## 2. Shape and reshape

**Intuition.** The data underneath an array is just a flat run of numbers.
`shape` is a lens you look at that flat run through — `reshape` doesn't move
any data, it just changes the lens (as long as the total element count
matches).

```python
g = np.arange(12)          # [0, 1, 2, ..., 11], shape (12,)
g2 = g.reshape(3, 4)       # same 12 numbers, viewed as 3 rows x 4 cols
print(g2)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

g3 = g2.reshape(-1)        # flatten back to 1-D; -1 means "infer this size"
print(g3.shape)            # (12,)

g4 = g2.T                  # transpose: swap axes, shape becomes (4, 3)
print(g4.shape)
```

`-1` in any position of `reshape` tells NumPy "figure out this dimension from
the total size" — you can only use it once per call. `reshape` returns a
**view** whenever the data layout allows it (see [gotchas](#9-common-gotchas)),
so mutating the reshaped array can mutate the original.

---

## 3. Indexing and slicing

**Intuition.** NumPy indexing extends Python's `list[start:stop:step]` to
multiple axes at once, comma-separated: `array[rows, cols]`.

```python
m = np.arange(20).reshape(4, 5)

m[1, 2]        # single element, row 1 col 2 -> 7
m[1]           # entire row 1 -> [5 6 7 8 9]
m[:, 1]        # entire column 1 -> [ 1  6 11 16]
m[1:3, 2:4]    # a 2x2 sub-block: rows 1-2, cols 2-3
```

```python
>>> m[1:3, 2:4]
array([[ 7,  8],
       [12, 13]])
```

!!! warning "Basic slices are views, not copies"
    `row = m[0]; row[0] = 999` changes `m` too. This is the single most common
    NumPy surprise for people coming from lists. See
    [Views vs. copies](#9-common-gotchas).

---

## 4. Boolean and fancy indexing

**Intuition.** Instead of indexing by *position*, you can index by *condition*
(boolean masking) or by an explicit *list of positions* (fancy indexing). Both
always return copies, never views.

```python
arr = np.array([1, -2, 3, -4, 5])

mask = arr > 0
print(arr[mask])          # [1 3 5] — keep only positive entries

arr2 = arr.copy()
arr2[arr2 < 0] = 0        # clip negatives to 0, in place, no explicit loop
print(arr2)               # [1 0 3 0 5]

idx = np.array([0, 2, 4])
print(arr[idx])           # [1 3 5] — fancy indexing: pick specific positions
```

Boolean masking is how you write conditionals in array code: "if x < 0, set it
to 0" becomes `x[x < 0] = 0`, no `for`/`if` needed. This pattern — condition,
then index-assign — replaces the vast majority of loops you'd otherwise write
over array data.

---

## 5. Broadcasting

**Intuition.** Broadcasting is NumPy's rule for combining arrays of
*different* shapes without you writing a loop to line them up. Picture
stretching the smaller array's size-1 axes until both shapes match, without
actually copying any data — NumPy just reuses the same values along the
stretched axis.

**The rule**, applied from the trailing (rightmost) axis inward:

1. Align shapes on the right.
2. Two axes are compatible if they're equal, or one of them is 1.
3. Any missing leading axis is treated as size 1.
4. If any pair of axes disagrees (both >1 and unequal), broadcasting fails.

```python
x = np.array([[1], [2], [3]])   # shape (3, 1) — a column
y = np.array([10, 20, 30])      # shape (3,)   — a row

print(x + y)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
print(x.shape, y.shape, (x + y).shape)   # (3, 1) (3,) (3, 3)
```

Walk the rule: `y`'s shape `(3,)` is treated as `(1, 3)`. Compare trailing
axes: `1` vs `3` → compatible (stretch the `1`). Next axis: `3` vs the implicit
`1` → compatible (stretch the other `1`). Result: `(3, 3)`, and every
`out[i, j] = x[i, 0] + y[0, j]` — an outer sum, with zero explicit looping.

A concrete, realistic use: scaling each column of a matrix by a per-column
price vector.

```python
price = np.array([10.0, 20.0, 30.0])         # shape (3,)
qty   = np.array([[1, 2, 3], [4, 5, 6]])      # shape (2, 3)
print(price * qty)
# [[ 10.  40.  90.]
#  [ 40. 100. 180.]]
```

And the failure mode, so you recognize the error message:

```python
>>> np.array([1, 2, 3]) + np.array([1, 2])
ValueError: operands could not be broadcast together with shapes (3,) (2,)
```

`3` and `2` are both greater than 1 and unequal — no stretch rescues that.

!!! note "Mental model"
    Broadcasting never *duplicates memory* for the stretched axis; it's
    implemented as a zero-cost "reuse this value again" in the underlying C
    loop. This is also why broadcasting a huge array against a size-1 axis is
    cheap — you're not allocating a full copy first.

---

## 6. Vectorization

**Intuition.** A Python `for` loop over an array processes one Python object
at a time, and every `+`, every attribute lookup, every bounds-check pays
Python's interpreter overhead. A vectorized NumPy call instead dispatches
*once* into a tight, compiled C loop that walks the whole buffer. Same math,
wildly different constant factor.

```python
import time

n = 1_000_000
data = list(range(n))
arr = np.arange(n)

t0 = time.perf_counter()
out = [x * 2 + 1 for x in data]
print("python loop:", time.perf_counter() - t0)

t0 = time.perf_counter()
out_np = arr * 2 + 1
print("numpy vectorized:", time.perf_counter() - t0)
```

On the machine used to write this page, the loop took about 32ms and the
vectorized version about 2ms — roughly a 15-20x difference for this workload;
the gap widens further for larger arrays or more arithmetic per element.
Vectorized code is also usually *shorter and more readable* once the habit
sticks, which matters as much as the speed.

**How to think in arrays instead of loops:**

- Ask "what operation applies to *every* element?" and write that operation
  once (`arr * 2`), not "what do I do to *one* element?" wrapped in a loop.
- Reach for boolean masks (`arr[arr < 0] = 0`) instead of `if` inside a loop.
- Reach for `axis=` aggregations (next section) instead of manually summing
  rows/columns.
- If you truly need a custom per-element function NumPy has no primitive for,
  `np.vectorize` exists but is a thin, still-Python-speed wrapper — it's a
  convenience for broadcasting semantics, **not** a performance tool. Prefer
  expressing the function with existing ufuncs (`np.where`, `np.exp`, `np.abs`,
  ...) whenever possible.

```python
def slow_square(vals):
    out = []
    for val in vals:
        out.append(val ** 2)
    return out

data = list(range(5))
data_np = np.array(data)
print(slow_square(data) == list(data_np ** 2))   # True — same answer, different cost
```

---

## 7. Aggregations and `axis`

**Intuition.** `axis` tells an aggregation *which dimension to collapse*.
`axis=0` collapses rows (walks down each column), `axis=1` collapses columns
(walks across each row). The output shape has that axis removed.

```python
mat = np.arange(1, 13).reshape(3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

mat.sum()            # 78 — everything
mat.sum(axis=0)       # [15 18 21 24] — one value per column (summed down rows)
mat.sum(axis=1)       # [10 26 42]    — one value per row (summed across columns)
mat.mean(axis=1)      # [ 2.5  6.5 10.5]
mat.max(axis=0)       # [ 9 10 11 12]
mat.argmax(axis=1)    # [3 3 3] — index of the max in each row
```

!!! tip "Remembering axis direction"
    `axis=0` moves *down* (through rows) — the axis you name is the one that
    *disappears* after the reduction. If you sum `axis=0` on a `(3, 4)` matrix
    you get a `(4,)` result, one number per column. This trips almost everyone
    up at least once; write a small example and check `.shape` when unsure.

`keepdims=True` keeps the collapsed axis as size 1 instead of dropping it —
handy when you immediately want to broadcast the result back against the
original array (e.g. subtracting the row mean from every row).

---

## 8. Linear algebra

**Intuition.** `@` is matrix multiplication (or matrix-vector, or dot product
for two 1-D arrays) — read `A @ B` as "apply transformation `A`, then use the
result as input coordinates." `np.linalg` covers the operations built on top
of that: inverses, determinants, eigenvalues, solving linear systems.

```python
A = np.array([[1., 2.], [3., 4.]])
B = np.array([[5., 6.], [7., 8.]])

A @ B                     # matrix product
# [[19. 22.]
#  [43. 50.]]

np.linalg.inv(A)          # A^-1
np.linalg.det(A)          # determinant, -2.0
vals, vecs = np.linalg.eig(A)   # eigenvalues, eigenvectors
np.linalg.solve(A, np.array([1., 2.]))   # x such that A @ x == [1, 2]
np.linalg.norm(np.array([3., 4.]))        # 5.0 — Euclidean length
```

!!! warning "`*` is not `@`"
    `A * B` on two 2-D arrays is **elementwise** multiplication (Hadamard
    product), not matrix multiplication. This is arguably NumPy's sharpest
    edge for people coming from MATLAB or math notation — always reach for
    `@` (or `np.matmul`/`np.dot`) when you mean "matrix multiply."

    This is exactly the same operator OptimumAI's own
    `Matrix([[1, 2], [3, 4]]).matmul(...)` traces step by step — see the
    [Features guide](features.md) if you want the fully narrated version of
    `@`, dot products, and cosine similarity.

`np.linalg.solve(A, b)` is preferred over `np.linalg.inv(A) @ b` — it's both
faster and numerically more stable, since it never explicitly forms the
inverse.

---

## 9. Random and seeding

**Intuition.** "Random" in code almost always means "deterministic given a
seed" — you want reproducible randomness so a run today matches a run
tomorrow. NumPy's modern API is the `Generator` object from
`np.random.default_rng(seed)`, not the older global `np.random.seed(...)`
functions.

```python
rng = np.random.default_rng(42)
r1 = rng.normal(size=3)

rng2 = np.random.default_rng(42)
r2 = rng2.normal(size=3)

print(np.array_equal(r1, r2))   # True — same seed, same sequence
```

Prefer `default_rng` in new code: each `Generator` instance owns its own
state, so two parts of a program (or two workers) using their own `rng`
objects never secretly interfere with each other's random stream — unlike the
old global `np.random.seed()`, which is shared, mutable, global state.

---

## 10. Performance tips

- **Vectorize first.** If you're writing `for i in range(len(arr))`, stop and
  ask what NumPy primitive expresses that loop as a whole-array operation.
- **Preallocate, don't grow.** `np.empty(n)` + fill by index beats repeatedly
  concatenating (`np.concatenate`/`np.append` in a loop reallocates every
  time).
- **Avoid unnecessary copies.** Chained fancy indexing / boolean masks each
  allocate a new array — combine conditions (`arr[(a > 0) & (b < 5)]`) instead
  of filtering twice.
- **Use the right dtype.** `float32` halves the memory (and often the time)
  of `float64` if you don't need double precision — this matters a lot once
  arrays get into the millions of elements, and is exactly the same trade-off
  quantization makes for model weights (see the
  [fine-tuning guide](learn-finetuning.md#qlora-4-bit-base--lora-adapters)).
- **Let broadcasting do the work** instead of manually tiling arrays with
  `np.tile`/`np.repeat` — tiling allocates real memory, broadcasting doesn't.

---

## 11. Common gotchas

### Views vs. copies

Basic slicing (`arr[a:b]`, `arr[:, i]`) returns a **view** — it shares memory
with the original array. Fancy indexing (`arr[[0, 2, 4]]`) and boolean masking
(`arr[arr > 0]`) always return a **copy**.

```python
base = np.arange(10)
view = base[2:5]
view[0] = -1
print(base)   # [ 0  1 -1  3  4  5  6  7  8  9] — base changed too!

copy = base[2:5].copy()
copy[0] = -99
print(base)   # unchanged — .copy() explicitly breaks the sharing
```

When in doubt about whether you have a view or a copy, check `arr.base is
not None` (views have a non-`None` `.base` pointing at the owning array), or
just call `.copy()` defensively when you're about to mutate something you
don't want to leak into.

### Integer division and dtype surprises

```python
print(7 // 2, 7 / 2)              # 3 3.5 — Python floor vs true division

ai, bi = np.array([7]), np.array([2])
print(ai // bi, ai / bi)          # [3] [3.5] — NumPy mirrors Python's // vs /
```

`//` on integer arrays stays integer (floor division); `/` always promotes to
float, even for integer inputs, matching Python 3 semantics. The surprise
usually happens the other direction: doing integer-array arithmetic and
expecting a float result from `/`-like intent but having accidentally used
`//`.

### Floating-point precision

```python
print(0.1 + 0.2)                       # 0.30000000000000004
print(np.isclose(0.1 + 0.2, 0.3))      # True
```

This isn't a NumPy bug — it's IEEE-754 binary floating point, the same
representation Python floats use. Never compare floats with `==`; use
`np.isclose` (elementwise tolerance) or `np.allclose` (whole-array tolerance)
instead.

!!! note "Where to go next"
    Once shapes, broadcasting, and vectorization feel natural, the same
    mental model carries directly into [matplotlib](learn-matplotlib.md)
    (every plot call takes a NumPy array) and [PyTorch](learn-pytorch.md)
    (a tensor's `.shape`, broadcasting rules, and `@` all work exactly like
    what you just learned here).
