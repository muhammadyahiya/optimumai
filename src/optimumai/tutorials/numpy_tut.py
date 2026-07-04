"""The NumPy tutorial: arrays, broadcasting, vectorization, linear algebra.

Everything here runs on the base OptimumAI install (no extras) — NumPy is a
core dependency of the whole SDK, not an optional one.
"""

from __future__ import annotations

from optimumai.tutorials.core import Tutorial


def build() -> Tutorial:
    t = Tutorial(
        name="numpy",
        title="NumPy: the array language everything else is built on",
        summary="Arrays, broadcasting, vectorization, and a little linear algebra.",
    )

    # ------------------------------------------------------------------
    # 1. Why arrays at all
    # ------------------------------------------------------------------
    t.md(
        "## Why not just use Python lists?\n\n"
        "A Python list is a bag of *pointers* to separate objects scattered around "
        "memory. A NumPy `ndarray` is one contiguous block of memory holding values "
        "of a single dtype, plus a small header describing its shape. That single "
        "fact is why NumPy is fast: no per-element Python overhead, and math runs in "
        "tight, vectorized C loops instead of a Python `for` loop.\n\n"
        "Everything below — machine learning, plotting, signal processing — is "
        "arrays talking to arrays. Get comfortable with `ndarray` and the rest of "
        "the stack (PyTorch tensors included) will feel familiar."
    )
    t.code(
        """import numpy as np

x = np.array([1, 2, 3, 4, 5])
print(type(x), x.dtype, x.shape)
print(x)""",
        note="a 1-D array: fixed dtype, fixed shape",
        requires=(),
    )

    # ------------------------------------------------------------------
    # 2. dtypes
    # ------------------------------------------------------------------
    t.md(
        "## dtypes: the same value, different boxes\n\n"
        "Every array has one dtype for *all* its elements — that uniformity is what "
        "lets NumPy pack them tightly and skip per-element type checks. Mixing "
        "types upcasts everything to the most general one (int -> float here)."
    )
    t.code(
        """ints = np.array([1, 2, 3], dtype=np.int64)
floats = np.array([1, 2, 3], dtype=np.float32)
mixed = np.array([1, 2.5, 3])  # upcasts to float64
print(ints.dtype, floats.dtype, mixed.dtype)
print(ints.nbytes, "bytes for 3 int64s vs", floats.nbytes, "bytes for 3 float32s")""",
        note="dtype controls both precision and memory footprint",
        requires=(),
    )

    # ------------------------------------------------------------------
    # 3. shape / reshape / ravel
    # ------------------------------------------------------------------
    t.md(
        "## Shape is metadata, not a copy\n\n"
        "`reshape` reinterprets the *same* underlying buffer with a new shape — it's "
        "cheap (no data is copied, when possible) precisely because the memory is "
        "one contiguous block. `ravel` does the inverse: flatten back to 1-D."
    )
    t.code(
        """x = np.arange(6)
print("flat:", x)

grid = x.reshape(2, 3)
print("reshaped to (2, 3):")
print(grid)

back = grid.ravel()
print("raveled back to flat:", back)
print("reshape shares memory with the original:", np.shares_memory(x, grid))""",
        note="np.arange + reshape is the bread-and-butter array constructor",
        requires=(),
    )

    # ------------------------------------------------------------------
    # 3b. stacking & concatenation
    # ------------------------------------------------------------------
    t.md(
        "## Combining arrays: stack and concatenate\n\n"
        "`np.concatenate` joins arrays along an *existing* axis (the arrays must "
        "already agree on every other dimension). `np.stack` is subtly "
        "different: it joins arrays along a *brand-new* axis, so stacking two "
        "`(3,)` vectors gives a `(2, 3)` array rather than a longer `(6,)` one. "
        "`np.vstack`/`np.hstack` are convenience wrappers for the common "
        "row-wise/column-wise cases."
    )
    t.code(
        """a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

joined = np.concatenate([a, b])
print("concatenate along existing axis:", joined, joined.shape)

stacked = np.stack([a, b])
print("stack creates a new axis:")
print(stacked, stacked.shape)

print("vstack (rows):")
print(np.vstack([a, b]))
print("hstack (columns):", np.hstack([a, b]))""",
        note="concatenate joins along an existing axis; stack creates a new one",
        requires=(),
    )

    # ------------------------------------------------------------------
    # 4. indexing & slicing
    # ------------------------------------------------------------------
    t.md(
        "## Indexing and slicing\n\n"
        "Multi-dimensional indexing uses one tuple of indices, `arr[row, col]`, "
        "rather than Python's chained `arr[row][col]`. Slices are *views*: they "
        "point into the same memory rather than copying it."
    )
    t.code(
        """grid = np.arange(12).reshape(3, 4)
print("grid:")
print(grid)
print("element at row 1, col 2:", grid[1, 2])
print("row 1:", grid[1, :])
print("column 2:", grid[:, 2])
print("bottom-right 2x2 block:")
print(grid[1:, 2:])

view = grid[1:, 2:]
view[0, 0] = -1
print("mutating the slice also changed the original (it's a view):")
print(grid)""",
        note="slices are views, not copies — mutate with care",
        requires=(),
    )

    # ------------------------------------------------------------------
    # 5. boolean masking & fancy indexing
    # ------------------------------------------------------------------
    t.md(
        "## Boolean masks and fancy indexing\n\n"
        "Comparing an array to a value gives you a same-shaped array of `True`/"
        "`False`. Indexing with that mask keeps only the `True` positions — this "
        "is how you filter data without writing a loop. 'Fancy indexing' (indexing "
        "with an array of integers) picks out specific positions, in any order, "
        "and always returns a *copy*."
    )
    t.code(
        """x = np.array([3, -1, 4, -1, 5, -9, 2, 6])
mask = x > 0
print("mask:", mask)
print("positive values:", x[mask])

x[x < 0] = 0  # mask-assign: zero out every negative in place
print("negatives clamped to 0:", x)

order = np.array([0, 2, 4])
print("fancy indexing picks arbitrary positions:", x[order])""",
        note="boolean masks filter; integer arrays pick specific positions",
        requires=(),
    )

    # ------------------------------------------------------------------
    # 5b. np.where — vectorized if/else
    # ------------------------------------------------------------------
    t.md(
        "## `np.where`: a vectorized if/else\n\n"
        "`np.where(condition, if_true, if_false)` picks elementwise between two "
        "arrays (or an array and a scalar) based on a boolean mask — the "
        "vectorized equivalent of `x if cond else y` applied one element at a "
        "time. It's also useful with a single argument: `np.where(condition)` "
        "returns the *indices* where the condition holds."
    )
    t.code(
        """x = np.array([3, -1, 4, -1, 5, -9, 2, 6])
sign_labels = np.where(x >= 0, "non-negative", "negative")
print("elementwise if/else:", sign_labels)

clipped = np.where(x < 0, 0, x)  # same effect as the mask-assign above
print("clipped negatives to 0:", clipped)

indices = np.where(x > 2)
print("indices where x > 2:", indices[0])""",
        note="np.where(cond, a, b) is elementwise if/else; np.where(cond) finds indices",
        requires=(),
    )

    # ------------------------------------------------------------------
    # 6. broadcasting
    # ------------------------------------------------------------------
    t.md(
        "## Broadcasting: NumPy's rule for combining different shapes\n\n"
        "Broadcasting lets you combine arrays of *different* shapes without "
        "writing explicit loops or manually replicating data. Compare shapes from "
        "the **trailing** (rightmost) dimension leftward. Two dimensions are "
        "compatible when they're equal, or one of them is 1 (it gets stretched to "
        "match, conceptually — no memory is actually copied).\n\n"
        "Worked example: a `(3, 4)` matrix plus a `(4,)` vector. Trailing "
        "dimensions `4` and `4` match, and the matrix's leading `3` has nothing to "
        "compare against, so the vector is broadcast across every row."
    )
    t.code(
        """matrix = np.arange(12).reshape(3, 4).astype(float)
row_vec = np.array([100.0, 200.0, 300.0, 400.0])  # shape (4,)
print("matrix shape:", matrix.shape, "row_vec shape:", row_vec.shape)

result = matrix + row_vec  # (3, 4) + (4,) -> row_vec applied to every row
print("matrix + row_vec broadcasts across rows:")
print(result)

col_vec = np.array([[1.0], [10.0], [100.0]])  # shape (3, 1)
print("\\nmatrix + col_vec (shape (3, 1)) broadcasts across columns:")
print(matrix + col_vec)""",
        note="trailing dimensions must match or be 1 — that's the whole rule",
        requires=(),
    )
    t.md(
        "A broadcast that fails is just as instructive: shapes `(3, 4)` and `(3,)` "
        "don't line up on the trailing axis (`4` vs `3`, neither is `1`), so NumPy "
        "raises instead of silently doing the wrong thing."
    )
    t.code(
        """matrix = np.ones((3, 4))
bad = np.array([1.0, 2.0, 3.0])  # shape (3,) — wrong trailing axis for (3, 4)
try:
    matrix + bad
except ValueError as exc:
    print("broadcasting error (as expected):", exc)""",
        note="broadcasting fails loudly instead of guessing what you meant",
        requires=(),
    )

    # ------------------------------------------------------------------
    # 7. vectorization vs Python loops
    # ------------------------------------------------------------------
    t.md(
        "## Vectorization vs. Python loops\n\n"
        "The single biggest NumPy habit to build: reach for a whole-array "
        "operation instead of a Python `for` loop. Below, a hand-written loop and "
        "a vectorized expression compute the *exact same* result — squaring "
        "every element and summing — but the vectorized version pushes the work "
        "into compiled C loops instead of the Python interpreter, so it's both "
        "shorter to write and dramatically faster at scale."
    )
    t.code(
        """import time

n = 200_000
x = np.arange(n, dtype=np.float64)

start = time.perf_counter()
loop_total = 0.0
for v in x:
    loop_total += v * v
loop_time = time.perf_counter() - start

start = time.perf_counter()
vec_total = np.sum(x * x)
vec_time = time.perf_counter() - start

print(f"loop result:       {loop_total:.1f}  ({loop_time * 1000:.2f} ms)")
print(f"vectorized result: {vec_total:.1f}  ({vec_time * 1000:.2f} ms)")
print("same answer:", np.isclose(loop_total, vec_total))
print(f"vectorized was ~{loop_time / vec_time:.0f}x faster")""",
        note="same math, same answer, radically different speed",
        requires=(),
    )

    # ------------------------------------------------------------------
    # 8. aggregations & axis
    # ------------------------------------------------------------------
    t.md(
        "## Aggregations and the `axis` argument\n\n"
        "`sum`, `mean`, `max`, `min`, `std`, ... all accept an `axis`. Without it, "
        "they collapse the *entire* array to a scalar. With `axis=0` they collapse "
        "*down* the rows (one result per column); with `axis=1` they collapse "
        "*across* the columns (one result per row). A useful mnemonic: `axis=k` is "
        "the axis that *disappears* from the output shape."
    )
    t.code(
        """grid = np.arange(1, 13).reshape(3, 4)
print("grid:")
print(grid)
print("overall sum:", grid.sum())
print("column sums (axis=0):", grid.sum(axis=0))
print("row sums (axis=1):", grid.sum(axis=1))
print("row means (axis=1):", grid.mean(axis=1))
print("column-wise max (axis=0):", grid.max(axis=0))""",
        note="axis=k is the dimension that collapses away",
        requires=(),
    )

    # ------------------------------------------------------------------
    # 9. dot / @ / matmul
    # ------------------------------------------------------------------
    t.md(
        "## Matrix multiplication: `np.dot`, `@`, and `matmul`\n\n"
        "`*` between two arrays is *elementwise* multiplication. Real matrix "
        "multiplication — the row-times-column sum from linear algebra — is "
        "`np.dot`, `np.matmul`, or the `@` operator (they agree for 2-D arrays; "
        "`@` is just nicer to read)."
    )
    t.code(
        """a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0], [7.0, 8.0]])

print("elementwise a * b:")
print(a * b)
print("\\nmatrix product a @ b:")
print(a @ b)
print("np.dot and np.matmul agree with @:", np.array_equal(a @ b, np.dot(a, b)))
print(np.array_equal(a @ b, np.matmul(a, b)))""",
        note="`*` is elementwise; `@`/dot/matmul is real matrix multiplication",
        requires=(),
    )

    # ------------------------------------------------------------------
    # 10. a bit of linear algebra
    # ------------------------------------------------------------------
    t.md(
        "## A little linear algebra: inverse, eigenvalues, SVD\n\n"
        "`np.linalg` covers the classic decompositions. `inv` solves `A @ A_inv "
        "== I`; `eig` finds the vectors that `A` only stretches (never rotates); "
        "`svd` factors *any* matrix into rotate -> scale -> rotate, the workhorse "
        "behind PCA and low-rank approximation."
    )
    t.code(
        """a = np.array([[4.0, 2.0], [1.0, 3.0]])

a_inv = np.linalg.inv(a)
identity_check = a @ a_inv
print("A @ inv(A) is the identity matrix:")
print(np.round(identity_check, decimals=10))

eigenvalues, eigenvectors = np.linalg.eig(a)
print("\\neigenvalues:", eigenvalues)

u, s, vt = np.linalg.svd(a)
reconstructed = u @ np.diag(s) @ vt
print("\\nsingular values:", s)
print("U @ diag(S) @ Vt reconstructs A:", np.allclose(reconstructed, a))""",
        note="inv, eig, svd — the three decompositions you'll reach for most",
        requires=(),
    )

    # ------------------------------------------------------------------
    # 11. random + reproducibility
    # ------------------------------------------------------------------
    t.md(
        "## Random numbers, reproducibly\n\n"
        "Modern NumPy generates randomness through a `Generator` object created "
        "with `np.random.default_rng(seed)`, rather than the old global "
        "`np.random.seed(...)` state. Passing the same seed always reproduces the "
        "same sequence — essential for debugging and for reproducible experiments."
    )
    t.code(
        """rng_a = np.random.default_rng(seed=42)
rng_b = np.random.default_rng(seed=42)

sample_a = rng_a.normal(loc=0.0, scale=1.0, size=5)
sample_b = rng_b.normal(loc=0.0, scale=1.0, size=5)
print("sample_a:", np.round(sample_a, 4))
print("sample_b:", np.round(sample_b, 4))
print("same seed -> identical sequence:", np.array_equal(sample_a, sample_b))

rng_c = np.random.default_rng(seed=7)
sample_c = rng_c.normal(size=5)
print("different seed -> different sequence:", not np.array_equal(sample_a, sample_c))""",
        note="default_rng(seed) is the modern, reproducible way to get randomness",
        requires=(),
    )

    # ------------------------------------------------------------------
    # 12. capstone: vectorized distance matrix + softmax
    # ------------------------------------------------------------------
    t.md(
        "## Capstone: a vectorized distance matrix\n\n"
        "Given `n` points, the pairwise distance matrix is `n x n`, where entry "
        "`(i, j)` is the distance between point `i` and point `j`. The naive "
        "approach is a double Python loop. The vectorized approach uses "
        "broadcasting: reshape the points to `(n, 1, d)` and `(1, n, d)` so "
        "subtracting them broadcasts into every pairwise difference at once, "
        "`(n, n, d)`, then reduce over the last axis."
    )
    t.code(
        """rng = np.random.default_rng(seed=0)
points = rng.normal(size=(5, 2))  # 5 points in 2-D

diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (5, 1, 2) - (1, 5, 2)
dist_matrix = np.sqrt(np.sum(diffs**2, axis=-1))  # (5, 5)
print("pairwise distance matrix (vectorized):")
print(np.round(dist_matrix, 3))
print("diagonal is all zero (distance to self):", np.allclose(np.diag(dist_matrix), 0.0))
print("matrix is symmetric:", np.allclose(dist_matrix, dist_matrix.T))""",
        note="broadcasting turns an O(n^2) Python double-loop into one expression",
        requires=(),
    )
    t.md(
        "## Capstone, part two: softmax by hand\n\n"
        "Softmax turns a vector of raw scores into a probability distribution: "
        "`exp(x_i) / sum(exp(x_j))`. Subtracting the max first (`x - x.max()`) "
        "doesn't change the result mathematically but keeps the exponentials from "
        "overflowing — a numerically-stable trick you'll see in every ML codebase."
    )
    t.code(
        """def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)  # numerical stability, doesn't change the result
    exps = np.exp(shifted)
    return exps / np.sum(exps)


scores = np.array([2.0, 1.0, 0.1])
probs = softmax(scores)
print("scores:", scores)
print("softmax probabilities:", np.round(probs, 4))
print("probabilities sum to 1:", np.isclose(probs.sum(), 1.0))

# a huge score would overflow exp() without the max-subtraction trick
huge_scores = np.array([1000.0, 1001.0, 1002.0])
print("large-magnitude softmax still works:", np.round(softmax(huge_scores), 4))""",
        note="softmax: the numerically-stable formula every ML library uses",
        requires=(),
    )

    t.md(
        "## Where to go next\n\n"
        "You now have the core vocabulary — arrays, broadcasting, vectorization, "
        "aggregation, and a little linear algebra — that every other tool in this "
        "SDK (and the wider PyData/ML ecosystem) is built on top of. Next up: "
        "`optimumai.tutorials.get_tutorial(\"matplotlib\")` to turn these arrays "
        "into pictures."
    )
    return t
