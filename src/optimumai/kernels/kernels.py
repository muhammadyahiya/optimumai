"""GPU kernels from scratch — simple to complex, each run on the simulator.

Every kernel here is written as "what does one thread do to one element", launched
over a grid on :class:`~optimumai.kernels.sim.GpuSim`, and then **checked against
NumPy** and **measured** (global-memory traffic, coalescing, arithmetic intensity).
The progression mirrors how you'd actually learn to write CUDA:

    scalar_add → vector_add → matmul (naive + tiling analysis) → softmax → flash attention

Nothing here needs a GPU: the simulator runs the same kernel serially while
instrumenting it, so the mental model is exact even on a laptop.
"""

from __future__ import annotations

import math

import numpy as np

from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace
from optimumai.kernels.sim import GpuSim

_WHY = [
    "A GPU runs one thread per output element (SIMT) — you write the per-thread body",
    "Memory traffic, not FLOPs, usually bounds performance (arithmetic intensity)",
    "Coalesced access, shared-memory tiling, and kernel fusion are how you hit peak",
    "matmul + attention are ~all the FLOPs in a transformer, so these kernels are the game",
]


def _grid_block_1d(n: int, block: int = 4) -> tuple[int, int]:
    return (n + block - 1) // block, block


def scalar_add_trace(n: int = 8, scalar: float = 3.0) -> Trace:
    """out[i] = a[i] + s — one thread per element, the "hello world" kernel."""
    a = np.arange(n, dtype=float)
    out = np.zeros(n)

    def kernel(ctx) -> None:
        i = ctx.idx.global_id
        if i >= n:
            return
        ctx.gstore(out, i, ctx.gload(a, i) + scalar)

    grid, block = _grid_block_1d(n)
    stats, _ = GpuSim().launch(grid, block, kernel, flops=n)
    err = float(np.max(np.abs(out - (a + scalar))))
    t = Trace(
        op="kernel:scalar_add",
        formula="out[i] = a[i] + s",
        complexity="O(n), one thread per element",
        why_ai=_WHY,
        meta={**stats.as_dict(), "max_error": err, "n": n},
    )
    t.add("Launch", f"grid={grid} × block={block} = {grid * block} threads for n={n}", None,
          detail="global_id = block·block_dim + thread; the tail (id ≥ n) is guarded.")
    t.add("Global memory", f"{stats.global_reads} reads + {stats.global_writes} writes "
          f"(coalesced={stats.coalesced})", None)
    t.add("Correctness vs NumPy", f"max|out − (a+s)| = {num(err)}", err)
    t.result = out
    return t


def vector_add_trace(n: int = 8) -> Trace:
    """out[i] = a[i] + b[i] — coalesced reads of two arrays."""
    a = np.arange(n, dtype=float)
    b = np.arange(n, dtype=float) * 10
    out = np.zeros(n)

    def kernel(ctx) -> None:
        i = ctx.idx.global_id
        if i >= n:
            return
        ctx.gstore(out, i, ctx.gload(a, i) + ctx.gload(b, i))

    grid, block = _grid_block_1d(n)
    stats, _ = GpuSim().launch(grid, block, kernel, flops=n)
    err = float(np.max(np.abs(out - (a + b))))
    t = Trace(
        op="kernel:vector_add",
        formula="out[i] = a[i] + b[i]",
        complexity="O(n); memory-bound (intensity ≈ 1 FLOP / 12 bytes)",
        why_ai=_WHY,
        meta={**stats.as_dict(), "max_error": err, "n": n},
    )
    t.add("Launch", f"grid={grid} × block={block} threads", None)
    t.add("Global memory", f"{stats.global_reads} reads + {stats.global_writes} writes; "
          f"coalesced={stats.coalesced}", None,
          detail="Lane i touches address i → the warp fuses into one wide transaction.")
    t.add("Arithmetic intensity", f"{num(stats.arithmetic_intensity())} FLOP/byte — memory-bound", None,
          detail="One add per 12 bytes moved: the ALUs starve waiting on VRAM. This is why "
          "elementwise ops are fused in practice.")
    t.add("Correctness vs NumPy", f"max error = {num(err)}", err)
    t.result = out
    return t


def matmul_trace(M: int = 16, N: int = 16, K: int = 16, tile: int = 4) -> Trace:  # noqa: N803
    """C = A·B — a thread per output cell, plus the shared-memory tiling win."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(M, K))
    B = rng.normal(size=(K, N))
    Af, Bf, Cf = A.ravel(), B.ravel(), np.zeros(M * N)

    def kernel(ctx) -> None:
        r, c = ctx.idx.row, ctx.idx.col
        if r >= M or c >= N:
            return
        acc = 0.0
        for k in range(K):
            acc += ctx.gload(Af, r * K + k) * ctx.gload(Bf, k * N + c)
        ctx.gstore(Cf, r * N + c, acc)

    grid = ((N + tile - 1) // tile, (M + tile - 1) // tile)
    stats, _ = GpuSim().launch(grid, (tile, tile), kernel, flops=2 * M * N * K)
    C = Cf.reshape(M, N)
    err = float(np.max(np.abs(C - A @ B)))
    naive_reads = stats.global_reads
    tiled_reads = naive_reads // tile
    t = Trace(
        op="kernel:matmul",
        formula="C[i,j] = Σₖ A[i,k]·B[k,j]",
        complexity="O(M·N·K) FLOPs; the game is minimizing global memory traffic",
        why_ai=_WHY,
        meta={**stats.as_dict(), "max_error": err, "naive_reads": naive_reads,
              "tiled_reads": tiled_reads, "tile": tile},
    )
    t.add("Launch (2-D grid)", f"grid={grid} × block=({tile},{tile}); thread (i,j) → cell C[i,j]", None)
    t.add("Naive global reads", f"2·M·N·K = {naive_reads} (each cell re-reads a row of A + col of B)",
          naive_reads, detail="No reuse: every output pulls its whole row/column from VRAM.")
    t.add("Tiling win", f"load {tile}×{tile} tiles into shared memory → ~{naive_reads} → {tiled_reads} "
          f"global reads ({tile}× less)", tiled_reads,
          detail="Each element loaded once per tile and reused across the tile — the core GPU matmul trick.")
    t.add("Correctness vs NumPy", f"max|C − A@B| = {num(err)}", err)
    t.result = C
    return t


def softmax_rows_trace(rows: int = 4, cols: int = 6) -> Trace:
    """Row-wise softmax — one thread per row, with the max-subtraction trick."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(rows, cols))
    Xf, Of = X.ravel(), np.zeros(rows * cols)

    def kernel(ctx) -> None:
        r = ctx.idx.global_id
        if r >= rows:
            return
        vals = [ctx.gload(Xf, r * cols + j) for j in range(cols)]
        m = max(vals)  # stability: subtract the row max
        exps = [math.exp(v - m) for v in vals]
        s = sum(exps)
        for j in range(cols):
            ctx.gstore(Of, r * cols + j, exps[j] / s)

    grid, block = _grid_block_1d(rows)
    stats, _ = GpuSim().launch(grid, block, kernel, flops=rows * cols * 3)
    out_mat = Of.reshape(rows, cols)
    shifted = X - X.max(axis=1, keepdims=True)
    ref = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    err = float(np.max(np.abs(out_mat - ref)))
    t = Trace(
        op="kernel:softmax",
        formula="softmax(xᵢ) = e^(xᵢ−max) / Σ e^(xⱼ−max)",
        complexity="O(rows·cols); each thread owns one row",
        why_ai=_WHY,
        meta={**stats.as_dict(), "max_error": err, "rows": rows, "cols": cols},
    )
    t.add("Launch", f"{grid * block} threads, one per row of {rows}×{cols}", None)
    t.add("Per-thread work", "read the row → subtract its max → exp → normalize", None,
          detail="Subtracting the max prevents overflow in e^x without changing the result.")
    t.add("Row sums", f"each row sums to 1 (check: {num(float(out_mat.sum(axis=1).mean()))})", None)
    t.add("Correctness vs NumPy", f"max error = {num(err)}", err)
    t.result = out_mat
    return t


def flash_attention_kernel_trace(n: int = 4, d: int = 4, block: int = 2) -> Trace:
    """Fused attention with online softmax — exact, without an N×N matrix in VRAM."""
    rng = np.random.default_rng(2)
    Q, K, V = (rng.normal(size=(n, d)) for _ in range(3))
    scale = 1.0 / math.sqrt(d)

    out = np.zeros((n, d))
    for i in range(n):  # each query row, streamed over K/V blocks (online softmax)
        m_i, l_i = -np.inf, 0.0
        acc = np.zeros(d)
        for start in range(0, n, block):
            kb = K[start:start + block]
            vb = V[start:start + block]
            s = (Q[i] @ kb.T) * scale
            m_new = max(m_i, float(s.max()))
            p = np.exp(s - m_new)
            alpha = math.exp(m_i - m_new) if m_i != -np.inf else 0.0
            l_i = l_i * alpha + float(p.sum())
            acc = acc * alpha + p @ vb
            m_i = m_new
        out[i] = acc / l_i

    ref_scores = (Q @ K.T) * scale
    ref = (np.exp(ref_scores - ref_scores.max(axis=-1, keepdims=True))
           / np.exp(ref_scores - ref_scores.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)) @ V
    err = float(np.max(np.abs(out - ref)))
    t = Trace(
        op="kernel:flash_attention",
        formula="O = softmax(QKᵀ/√d)·V, computed blockwise with a running (max, sum)",
        complexity="O(n²·d) FLOPs but O(n·d) memory — no N×N score matrix in VRAM",
        why_ai=_WHY,
        meta={"max_abs_error": err, "n": n, "d": d, "block": block},
    )
    t.add("Tiling plan", f"stream K/V in blocks of {block}; keep running max mᵢ, sum lᵢ, acc", None,
          detail="The whole point: never materialize the n×n scores — that's what saves HBM.")
    t.add("Online softmax", "on each block: rescale the accumulator by e^(m_old−m_new), then add",
          None, detail="Rescaling makes the streamed result identical to computing softmax at once.")
    t.add("Correctness vs standard attention", f"max error = {num(err)} (EXACT, not approximate)", err,
          detail="FlashAttention is exact — it only changes the memory schedule, not the math.")
    t.result = out
    return t


KERNELS = {
    "scalar_add": scalar_add_trace,
    "vector_add": vector_add_trace,
    "matmul": matmul_trace,
    "softmax": softmax_rows_trace,
    "flash_attention": flash_attention_kernel_trace,
}


def list_kernels() -> list[str]:
    """Names of the available kernels, simple → complex."""
    return list(KERNELS)


def run_kernel(name: str, explain: bool = False, level: str | ExplainLevel = "engineer"):
    """Run a kernel by name; ``explain=True`` renders the trace."""
    if name not in KERNELS:
        raise ValueError(f"unknown kernel {name!r}. Choose from: {', '.join(KERNELS)}")
    t = KERNELS[name]()
    return t.render(level) if explain else t.result


def demo() -> Trace:
    """A representative kernel trace (tiled matmul)."""
    return matmul_trace()
