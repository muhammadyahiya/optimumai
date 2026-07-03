"""The canonical GPU optimization, worked out on paper: naive vs tiled matmul.

Matrix multiply is ~90% of the FLOPs in a transformer, so the way you multiply
matrices on a GPU basically *is* AI performance. A naive kernel — one thread per
output element, each re-reading a full row and column from slow global memory — is
correct but memory-bound: it moves far more bytes than it does math. The fix is
**tiling**: load small ``tile × tile`` blocks once into fast shared memory and let
every thread in the block reuse them, cutting global-memory traffic by roughly a
factor of ``tile``. Layer on **memory coalescing** — 32 threads in a warp reading
contiguous addresses collapse into one transaction — and you approach the
hand-tuned throughput of cuBLAS.

This is exactly the intuition behind FlashAttention: keep the working tiles in
on-chip SRAM and never round-trip the big score matrix through VRAM. There is no
GPU here, so the memory-traffic figures are computed with plain arithmetic and the
matrix product is verified for real with NumPy. Run the trace with ``explain=True``.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace

FP32_BYTES = 4  # single-precision float: 4 bytes each
WARP_SIZE = 32  # threads per warp (the coalescing width)


def _blocked_matmul(A: np.ndarray, B: np.ndarray, tile: int) -> np.ndarray:  # noqa: N803
    """Multiply ``A @ B`` in ``tile × tile`` blocks — mirrors what a tiled kernel does.

    Numerically identical to ``A @ B`` (up to floating-point summation order); the
    point is to show the *access pattern* a shared-memory kernel uses, not to be
    faster in NumPy.
    """
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise ValueError(f"inner dims must match: A is {A.shape}, B is {B.shape}")
    C = np.zeros((m, n), dtype=A.dtype)
    for i0 in range(0, m, tile):
        for j0 in range(0, n, tile):
            for p0 in range(0, k, tile):
                a_blk = A[i0 : i0 + tile, p0 : p0 + tile]
                b_blk = B[p0 : p0 + tile, j0 : j0 + tile]
                C[i0 : i0 + tile, j0 : j0 + tile] += a_blk @ b_blk
    return C


def tiled_matmul_trace(M: int = 64, N: int = 64, K: int = 64, tile: int = 16) -> Trace:  # noqa: N803
    """Trace the naive → tiled → coalesced optimization of ``C = A · B``.

    Args:
        M, N, K: Dimensions of ``C = A·B`` where ``A`` is ``M×K`` and ``B`` is ``K×N``.
        tile: Side length of the square tile loaded into shared memory. The
            global-memory reads drop by roughly this factor.

    The result is the real product matrix ``C`` (computed with a seeded NumPy
    ``A @ B``). ``meta`` carries ``naive_reads``, ``tiled_reads`` and
    ``reduction_factor`` (which equals ``tile``).
    """
    for name, dim in ("M", M), ("N", N), ("K", K), ("tile", tile):
        if dim < 1:
            raise ValueError(f"{name} must be >= 1, got {dim}")
    if tile > min(M, N, K):
        raise ValueError(f"tile ({tile}) must be <= min(M, N, K) = {min(M, N, K)}")

    flops = 2 * M * N * K  # one multiply + one add per inner-product term
    naive_reads = M * N * 2 * K  # each of M·N outputs reads K from A and K from B
    tiled_reads = naive_reads // tile  # each tile is loaded once, reused `tile` times
    reduction_factor = tile

    t = Trace(
        op="tiled_matmul",
        formula="C[i,j] = Σₖ A[i,k]·B[k,j]   (M×K · K×N → M×N)",
        complexity="O(M·N·K) FLOPs, but memory traffic is what you optimize",
        why_ai=[
            "Matmul is ~90% of the FLOPs in a transformer (QKV projections, attention "
            "scores, the MLP) — so matmul throughput basically sets training speed",
            "Tiling + coalescing is how cuBLAS and hand-written kernels reach peak "
            "hardware throughput; the naive kernel leaves most of the GPU idle",
            "It is the core intuition behind FlashAttention: keep tiles in on-chip SRAM "
            "and reuse them, never streaming the big score matrix through slow VRAM",
        ],
        meta={
            "M": M,
            "N": N,
            "K": K,
            "tile": tile,
            "flops": flops,
            "naive_reads": naive_reads,
            "tiled_reads": tiled_reads,
            "reduction_factor": reduction_factor,
        },
    )

    # --- Step 1: naive kernel is memory-bound -------------------------------
    naive_intensity = flops / (naive_reads * FP32_BYTES)
    t.add(
        "Naive kernel: one thread per output, all reads from global memory",
        f"global reads = M·N·2·K = {M}·{N}·2·{K} = {naive_reads} fp32 values; "
        f"arithmetic intensity ≈ {num(naive_intensity)} FLOP/byte",
        naive_reads,
        detail=(
            "Each of the M·N output threads streams a full row of A and a full column "
            "of B straight from global memory (VRAM). At ~0.25 FLOP/byte in fp32 the "
            "ALUs starve waiting on memory — the kernel is MEMORY-BOUND. The data is "
            "there; it is just being re-read from the slowest tier over and over."
        ),
    )

    # --- Step 2: tiling reuses data from shared memory ----------------------
    tiled_intensity = flops / (tiled_reads * FP32_BYTES)
    t.add(
        "Tiling: load tile×tile blocks into shared memory once, then reuse",
        f"reads drop ~{reduction_factor}× → M·N·2·K / tile = {naive_reads} / {tile} = "
        f"{tiled_reads}; intensity rises to ≈ {num(tiled_intensity)} FLOP/byte",
        tiled_reads,
        detail=(
            f"A {tile}×{tile} tile brought into shared memory is reused by every thread "
            f"in the block, so each element is fetched from global memory once instead "
            f"of {tile} times. Fewer bytes moved for the same FLOPs ⇒ higher arithmetic "
            "intensity ⇒ the kernel shifts from memory-bound toward compute-bound."
        ),
    )

    # --- Step 3: memory coalescing ------------------------------------------
    strided_cost = WARP_SIZE  # 32 separate transactions, one per thread
    coalesced_cost = 1  # one wide transaction serves the whole warp
    t.add(
        "Coalescing: a warp reading contiguous addresses → one transaction",
        f"strided access = {strided_cost} separate transactions per warp; "
        f"coalesced access = {coalesced_cost} transaction ({strided_cost}× fewer)",
        (strided_cost, coalesced_cost),
        detail=(
            f"When the {WARP_SIZE} threads of a warp read CONTIGUOUS addresses, the "
            "memory system fuses them into a single wide transaction. Strided or "
            "scattered access forces up to 32 separate transactions, wasting bandwidth. "
            "Laying tiles out so consecutive threads touch consecutive addresses is what "
            "makes tiling actually pay off."
        ),
    )

    # --- Step 4: verify correctness for real with NumPy ---------------------
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K))
    B = rng.standard_normal((K, N))
    C = A @ B
    C_tiled = _blocked_matmul(A, B, tile)
    max_abs_diff = float(np.max(np.abs(C - C_tiled)))
    assert max_abs_diff < 1e-9, f"tiled result diverged from A@B: max abs diff {max_abs_diff}"
    t.add(
        "Verify: blocked (tiled) product == direct A·B",
        f"max |A@B − tiled(A,B)| = {num(max_abs_diff)} (≈ 0); C shape = {C.shape}\n"
        f"C[:2,:2] = {arr(C[:2, :2])}",
        C,
        detail=(
            "Tiling changes only WHICH memory a value is read from, never the arithmetic, "
            "so the blocked multiply reproduces the direct product to floating-point "
            "precision. Correctness is preserved; only the memory traffic changed."
        ),
    )

    t.result = C
    return t


def tiled_matmul(  # noqa: N803
    M: int = 64,
    N: int = 64,
    K: int = 64,
    tile: int = 16,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> np.ndarray:
    """Return ``C = A·B`` (seeded). Set ``explain=True`` to print the tiling trace."""
    t = tiled_matmul_trace(M, N, K, tile)
    return t.render(level) if explain else t.result


def demo() -> Trace:
    """The headline demo for this module: the tiled-matmul optimization."""
    return tiled_matmul_trace()
