"""The GPU, made concrete: how threads are organized and how memory is tiered.

A GPU is not a fast CPU — it is a throughput machine that runs *thousands* of
threads in lockstep (SIMT: Same Instruction, Multiple Threads). Two mental models
unlock almost all GPU performance intuition. The **thread hierarchy** (grid →
block → warp → thread) is how work is mapped onto the hardware: you write a kernel
that says "what does one thread do to one element", and the hardware launches a
whole grid of them. The **memory hierarchy** (registers → shared memory → global
VRAM) is what actually decides whether that kernel is fast, because most real
kernels are limited by memory bandwidth, not arithmetic. Run either trace with
``explain=True`` to watch the reasoning.

There is no GPU here — every number below is the *concept* worked out with plain
arithmetic, which is exactly the point: you can reason about GPU performance on
paper before you ever touch one.
"""

from __future__ import annotations

from optimumai.core._fmt import num
from optimumai.core.trace import Trace

# A warp is the fundamental SIMT execution unit on NVIDIA GPUs: 32 threads that
# issue the *same* instruction on the same clock. This constant is hardware, not
# a tunable — it is why block sizes are almost always multiples of 32.
WARP_SIZE = 32


def thread_hierarchy_trace(
    grid_dim: int = 3, block_dim: int = 4, thread_id: int | None = None
) -> Trace:
    """Trace the grid → block → warp → thread hierarchy of a GPU launch.

    Args:
        grid_dim: Number of blocks in the (1-D) grid.
        block_dim: Number of threads per block.
        thread_id: A global thread index to decompose back into
            ``(block_idx, thread_idx)``. Defaults to the middle thread. If given,
            the trace's ``result`` is that ``(block_idx, thread_idx)`` tuple;
            otherwise the result is the total thread count.
    """
    if grid_dim < 1 or block_dim < 1:
        raise ValueError(f"grid_dim and block_dim must be >= 1, got {grid_dim}, {block_dim}")

    total = grid_dim * block_dim
    if thread_id is None:
        sample_id = total // 2  # a representative thread in the middle of the grid
        return_decomposition = False
    else:
        if not 0 <= thread_id < total:
            raise ValueError(f"thread_id must be in [0, {total}), got {thread_id}")
        sample_id = thread_id
        return_decomposition = True

    t = Trace(
        op="thread_hierarchy",
        formula="global_id = block_idx × block_dim + thread_idx",
        complexity="1 thread per data element — the grid scales with the problem, not the code",
        why_ai=[
            "GPUs run thousands of threads SIMT: the same instruction over many data "
            "elements at once, which is why they crush the dense matmuls in AI",
            "The index math maps each data element to exactly one thread — this is why "
            "GPU kernels are written 'per element' instead of as explicit loops",
            "A warp of 32 threads executes together, so branches that split a warp "
            "(divergence) and block sizes that are not multiples of 32 waste hardware",
        ],
        meta={"grid_dim": grid_dim, "block_dim": block_dim, "warp_size": WARP_SIZE},
    )

    t.add(
        "Grid → blocks → threads",
        f"grid_dim = {grid_dim} blocks, each block = {block_dim} threads",
        (grid_dim, block_dim),
        detail=(
            "A kernel launch creates a GRID of BLOCKS; each block holds a fixed number "
            "of THREADS. Blocks are scheduled independently across the GPU's cores."
        ),
    )

    t.add(
        "Total threads = grid_dim × block_dim",
        f"{grid_dim} × {block_dim} = {total}",
        total,
        detail="Every thread runs the same kernel code; only its index differs.",
    )

    warps_per_block = -(-block_dim // WARP_SIZE)  # ceil division
    t.add(
        "Warp = 32 threads (the SIMT unit)",
        f"a block of {block_dim} threads → ceil({block_dim} / {WARP_SIZE}) = "
        f"{warps_per_block} warp(s)",
        warps_per_block,
        detail=(
            f"The hardware executes threads {WARP_SIZE} at a time in a warp — one "
            "instruction, 32 lanes. A block of 4 still occupies a full warp, so 28 "
            "lanes idle: sizing blocks to multiples of 32 avoids that waste."
        ),
    )

    block_idx = sample_id // block_dim
    thread_idx = sample_id % block_dim
    t.add(
        "Map data element → thread (forward)",
        f"a thread in block {block_idx}, position {thread_idx}: "
        f"global_id = {block_idx} × {block_dim} + {thread_idx} = {sample_id}",
        sample_id,
        detail="This is the line at the top of nearly every CUDA kernel: it turns the "
        "thread's block/thread coordinates into the array index it should work on.",
    )

    t.add(
        "Decompose global_id → (block, thread) (reverse)",
        f"block_idx = {sample_id} // {block_dim} = {block_idx}; "
        f"thread_idx = {sample_id} % {block_dim} = {thread_idx}",
        (block_idx, thread_idx),
        detail=(
            f"Global thread {sample_id} is thread {thread_idx} of block {block_idx}. "
            "Divide by block_dim for the block, take the remainder for the position."
        ),
    )

    t.result = (block_idx, thread_idx) if return_decomposition else total
    return t


def memory_hierarchy_trace() -> Trace:
    """Trace the GPU memory hierarchy from fastest/smallest to slowest/largest.

    Walks registers → shared memory (L1) → global memory (VRAM) with approximate
    latencies and bandwidths, then explains arithmetic intensity (FLOP per byte)
    and why most kernels are memory-bound rather than compute-bound. The result is
    a small dict summarizing each level.
    """
    t = Trace(
        op="memory_hierarchy",
        formula="arithmetic_intensity = FLOPs / bytes_moved  (higher ⇒ compute-bound)",
        complexity="latency grows ~100× per level down; capacity grows to match",
        why_ai=[
            "The memory hierarchy, not the FLOP count, governs real GPU performance — "
            "a modern GPU can do far more math per second than it can feed from VRAM",
            "Fast kernels keep hot data in registers and shared memory and reuse it, "
            "instead of re-reading it from slow global memory every time",
            "'Memory-bound vs compute-bound' is the first question to ask about any "
            "kernel — it tells you whether to optimize data movement or arithmetic",
        ],
        meta={},
    )

    # Approximate, order-of-magnitude figures for a modern datacenter GPU. The
    # exact numbers vary by architecture; the ratios are the teaching point.
    levels = {
        "registers": {"size": "~256 KB/SM", "latency_cycles": 1, "bandwidth": "~tens of TB/s"},
        "shared_L1": {"size": "~100-200 KB/SM", "latency_cycles": 30, "bandwidth": "~tens of TB/s"},
        "global_VRAM": {"size": "~40-80 GB", "latency_cycles": 400, "bandwidth": "~1-3 TB/s"},
    }

    t.add(
        "Level 1 — Registers (fastest, smallest)",
        f"size {levels['registers']['size']}, latency ~{levels['registers']['latency_cycles']} "
        f"cycle, bandwidth {levels['registers']['bandwidth']}",
        levels["registers"],
        detail=(
            "Private per-thread scratch, right next to the ALUs. Effectively free to "
            "access — a kernel's live variables live here. Spilling to memory when you "
            "run out of registers is a classic performance cliff."
        ),
    )

    t.add(
        "Level 2 — Shared memory / L1 (per block)",
        f"size {levels['shared_L1']['size']}, latency "
        f"~{levels['shared_L1']['latency_cycles']} cycles, "
        f"bandwidth {levels['shared_L1']['bandwidth']}",
        levels["shared_L1"],
        detail=(
            "On-chip SRAM shared by every thread in a block — a programmer-managed "
            "cache. ~30× slower than a register but ~10× faster than VRAM. This is the "
            "scratchpad that tiling loads data into so threads can reuse it."
        ),
    )

    t.add(
        "Level 3 — Global memory / VRAM (slowest, largest)",
        f"size {levels['global_VRAM']['size']}, latency "
        f"~{levels['global_VRAM']['latency_cycles']} cycles, "
        f"bandwidth {levels['global_VRAM']['bandwidth']}",
        levels["global_VRAM"],
        detail=(
            "Off-chip HBM: gigabytes, but ~400 cycles away. All model weights and "
            "activations start here. Minimizing round-trips to VRAM is the single "
            "biggest lever in GPU optimization."
        ),
    )

    # Arithmetic intensity worked out for fp32: reading one 4-byte value to do one
    # FLOP is 0.25 FLOP/byte. Compare against the machine's FLOP:byte ratio.
    example_intensity = 1.0 / 4.0
    machine_ratio = 100.0  # ~ (peak FLOP/s) / (peak bytes/s) for a modern GPU
    t.add(
        "Arithmetic intensity: FLOP per byte moved",
        f"1 FLOP per fp32 read = 1 / 4 bytes = {num(example_intensity)} FLOP/byte; "
        f"the machine needs ~{num(machine_ratio)} FLOP/byte to be compute-bound",
        example_intensity,
        detail=(
            "The roofline idea: if a kernel does fewer FLOPs per byte than the machine's "
            f"FLOP:byte ratio (~{num(machine_ratio)}:1), it is MEMORY-BOUND — the ALUs sit "
            "idle waiting for data, so bandwidth, not FLOPs, sets the speed. Raising "
            "arithmetic intensity (reuse data once it is loaded) is how you escape that."
        ),
    )

    summary = {"num_levels": 3, "levels": levels, "example_flop_per_byte": example_intensity}
    t.result = summary
    return t


def demo() -> Trace:
    """The headline demo for this module: the GPU thread hierarchy."""
    return thread_hierarchy_trace()
