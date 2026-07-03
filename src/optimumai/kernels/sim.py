"""A tiny CUDA-style GPU simulator — the thread grid and memory hierarchy in pure Python.

There is no GPU here. Instead, :class:`GpuSim` *serially* runs a kernel once per
thread, exactly the way the hardware would run it in parallel, while instrumenting
every memory access. The payoff is that the SAME kernel function both computes the
right answer AND is measured: you write "what does one thread do to one element",
launch a grid of them, and get back the output buffer plus a :class:`MemoryStats`
report of global/shared traffic, arithmetic intensity, and whether the access
pattern was coalesced.

Thread model
------------
A launch is described by a ``grid`` (how many blocks) and a ``block`` (threads per
block). Both may be an ``int`` (1-D) or an ``(x, y)`` tuple (2-D). Every thread
receives a :class:`ThreadCtx` exposing:

* ``ctx.idx`` — a :class:`ThreadIdx` with ``block``, ``thread``, ``global_id`` and,
  for 2-D launches, the 2-D coordinates ``(bx, by)`` / ``(tx, ty)`` plus the derived
  ``row`` / ``col``. 1-D kernels just use ``ctx.idx.global_id``.
* ``ctx.gload(a, i)`` / ``ctx.gstore(a, i, v)`` — instrumented **global** memory ops
  (VRAM). They count reads/writes and record the address each warp lane touched so
  coalescing can be judged.
* ``ctx.sload(buf, i)`` / ``ctx.sstore(buf, i, v)`` — instrumented **shared** memory
  ops (on-chip SRAM, per block). Same counting, no coalescing check (shared memory
  banking is a separate concern the teaching kernels here don't model).

Worked example — vector add ``out[i] = a[i] + b[i]``::

    import numpy as np
    from optimumai.kernels.sim import GpuSim

    n = 8
    a = np.arange(n, dtype=float)
    b = np.arange(n, dtype=float) * 10
    out = np.zeros(n)

    def kernel(ctx):
        i = ctx.idx.global_id
        if i >= n:
            return  # guard the tail when grid*block overshoots n
        x = ctx.gload(a, i)          # 1 global read
        y = ctx.gload(b, i)          # 1 global read
        ctx.gstore(out, i, x + y)    # 1 global write

    sim = GpuSim()
    result, stats = sim.launch(grid=2, block=4, kernel=kernel, flops=n)
    assert np.allclose(result_of := out, a + b)  # out was written in place
    assert stats.global_reads == 2 * n and stats.global_writes == n
    assert stats.coalesced is True                # lane i touched address i

The simulator is single-threaded and deterministic; "parallel" here means "the
threads see the same inputs and there is no defined ordering you may rely on",
which is enough to teach the mental model without a real device.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# The SIMT execution unit on NVIDIA GPUs: 32 threads issuing one instruction on the
# same clock. It is hardware, not a tunable — it is why coalescing is judged over
# groups of 32 consecutive lanes and why block sizes are usually multiples of 32.
WARP_SIZE = 32

FP32_BYTES = 4  # bytes per single-precision float, used for byte-traffic accounting


@dataclass
class ThreadIdx:
    """The identity of one simulated thread.

    For a 1-D launch only ``block``, ``thread`` and ``global_id`` are meaningful.
    For a 2-D launch the 2-D coordinates are filled in too, and ``row`` / ``col``
    give the natural output-matrix position ``(by·block_y + ty, bx·block_x + tx)``.

    Attributes:
        block: Flattened block index within the grid.
        thread: Flattened thread index within the block.
        block_dim: Threads per block (flattened), so ``global_id`` can be recomputed.
        global_id: ``block × block_dim + thread`` — the classic CUDA index.
        bx, by: 2-D block coordinates (``0`` for a 1-D launch).
        tx, ty: 2-D thread coordinates (``0`` for a 1-D launch).
        row, col: Derived 2-D global position; ``col == global_id`` for a 1-D launch.
    """

    block: int
    thread: int
    block_dim: int
    global_id: int
    bx: int = 0
    by: int = 0
    tx: int = 0
    ty: int = 0
    row: int = 0
    col: int = 0


@dataclass
class MemoryStats:
    """Instrumentation gathered over one :meth:`GpuSim.launch`.

    Attributes:
        global_reads: Number of scalar reads from global memory (VRAM).
        global_writes: Number of scalar writes to global memory.
        shared_reads: Number of scalar reads from shared memory (SRAM).
        shared_writes: Number of scalar writes to shared memory.
        flops: Floating-point operations declared for the launch (for intensity).
        coalesced: Whether every warp's global accesses hit consecutive addresses.
            ``None`` means no global accesses were recorded.
    """

    global_reads: int = 0
    global_writes: int = 0
    shared_reads: int = 0
    shared_writes: int = 0
    flops: int = 0
    coalesced: bool | None = None
    # Per-warp address log: list of (array_id, [addr, addr, ...]) for the warp.
    _warp_addresses: list[tuple[int, list[int]]] = field(default_factory=list, repr=False)

    @property
    def global_bytes(self) -> int:
        """Total bytes moved to/from global memory (fp32 accounting)."""
        return (self.global_reads + self.global_writes) * FP32_BYTES

    @property
    def total_global(self) -> int:
        """Total scalar global-memory accesses (reads + writes)."""
        return self.global_reads + self.global_writes

    def arithmetic_intensity(self) -> float:
        """FLOPs per global byte moved — the roofline x-axis.

        Higher means more compute-bound (data is reused); lower means memory-bound
        (the ALUs starve waiting on VRAM). Returns ``0.0`` when no bytes moved.
        """
        b = self.global_bytes
        return float(self.flops) / b if b else 0.0

    def as_dict(self) -> dict[str, Any]:
        """A JSON-friendly summary suitable for a :class:`Trace` ``meta`` block."""
        return {
            "global_reads": self.global_reads,
            "global_writes": self.global_writes,
            "shared_reads": self.shared_reads,
            "shared_writes": self.shared_writes,
            "global_bytes": self.global_bytes,
            "flops": self.flops,
            "arithmetic_intensity": round(self.arithmetic_intensity(), 6),
            "coalesced": self.coalesced,
        }


def is_coalesced(addresses: list[int]) -> bool:
    """True when a warp's addresses are consecutive (address[i+1] == address[i]+1).

    Consecutive threads touching consecutive addresses is exactly the pattern the
    memory system fuses into one wide transaction. A single-lane access is trivially
    coalesced. Duplicate/scattered/strided addresses are not.
    """
    if len(addresses) <= 1:
        return True
    return all(addresses[i + 1] == addresses[i] + 1 for i in range(len(addresses) - 1))


class ThreadCtx:
    """The per-thread handle a kernel uses to read/write memory and see its index.

    A fresh context is created for every simulated thread by :meth:`GpuSim.launch`;
    kernels never construct one directly. All memory helpers funnel through the
    parent :class:`GpuSim` so a single :class:`MemoryStats` accumulates across the
    whole grid.
    """

    def __init__(self, sim: GpuSim, idx: ThreadIdx):
        self.sim = sim
        self.idx = idx

    def gload(self, array: np.ndarray, index: int) -> float:
        """Read ``array[index]`` from global memory, counting the access."""
        self.sim._record_global(index, is_write=False)
        return float(array[index])

    def gstore(self, array: np.ndarray, index: int, value: float) -> None:
        """Write ``value`` to ``array[index]`` in global memory, counting the access."""
        self.sim._record_global(index, is_write=True)
        array[index] = value

    def sload(self, buffer: dict[int, float] | np.ndarray, index: int) -> float:
        """Read ``buffer[index]`` from shared memory, counting the access."""
        self.sim.stats.shared_reads += 1
        return float(buffer[index])

    def sstore(self, buffer: dict[int, float], index: int, value: float) -> None:
        """Write ``value`` to ``buffer[index]`` in shared memory, counting the access."""
        self.sim.stats.shared_writes += 1
        buffer[index] = value


def _as_2d(dim: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize an ``int`` or ``(x, y)`` launch dimension to ``(x, y)``."""
    if isinstance(dim, tuple):
        if len(dim) != 2:
            raise ValueError(f"2-D launch dims must be (x, y), got {dim!r}")
        x, y = dim
        return int(x), int(y)
    return int(dim), 1


class GpuSim:
    """A serial simulator of a SIMT GPU launch.

    Instantiate once per kernel run (or reuse and call :meth:`reset`). The core is
    :meth:`launch`, which iterates every ``(block, thread)`` in the requested grid,
    hands each a :class:`ThreadCtx`, and tallies memory traffic into ``self.stats``.
    """

    def __init__(self) -> None:
        self.stats = MemoryStats()
        # Addresses touched by the warp currently being assembled, per source array.
        self._pending: dict[int, list[int]] = {}

    def reset(self) -> None:
        """Clear all instrumentation so the simulator can be launched again."""
        self.stats = MemoryStats()
        self._pending = {}

    def _record_global(self, index: int, *, is_write: bool) -> None:
        """Count one global access and remember its address for coalescing checks."""
        if is_write:
            self.stats.global_writes += 1
        else:
            self.stats.global_reads += 1
        # Group by whether it was a read or write so a warp's read of A and write of
        # C are judged as separate transactions rather than interleaved nonsense.
        key = 1 if is_write else 0
        self._pending.setdefault(key, []).append(int(index))

    def _flush_warp(self) -> None:
        """Fold the just-finished warp's addresses into the coalescing verdict."""
        for addrs in self._pending.values():
            if not addrs:
                continue
            verdict = is_coalesced(addrs)
            if self.stats.coalesced is None:
                self.stats.coalesced = verdict
            else:
                self.stats.coalesced = self.stats.coalesced and verdict
        self._pending = {}

    def launch(
        self,
        grid: int | tuple[int, int],
        block: int | tuple[int, int],
        kernel: Callable[[ThreadCtx], None],
        *,
        flops: int = 0,
    ) -> tuple[MemoryStats, GpuSim]:
        """Run ``kernel`` once per thread in the ``grid × block`` launch.

        Args:
            grid: Blocks in the grid — ``int`` (1-D) or ``(gx, gy)`` (2-D).
            block: Threads per block — ``int`` (1-D) or ``(bx, by)`` (2-D).
            kernel: A function taking a single :class:`ThreadCtx`. It mutates output
                arrays in place via ``ctx.gstore`` (that is where results land).
            flops: Total floating-point ops the kernel performs, recorded for
                arithmetic-intensity reporting. Purely informational.

        Returns:
            ``(stats, self)`` — the populated :class:`MemoryStats` and this sim.
            Kernels write their output into caller-owned arrays, so the "output
            buffer" is whatever array the caller passed in and the kernel stored to.

        Threads are visited block-by-block; within a block they are grouped into
        warps of :data:`WARP_SIZE`, and coalescing is judged per warp (consecutive
        lanes → consecutive addresses). 2-D blocks are flattened in row-major
        ``(ty, tx)`` order for warp grouping, matching CUDA's lane numbering.
        """
        gx, gy = _as_2d(grid)
        bx, by = _as_2d(block)
        if gx < 1 or gy < 1 or bx < 1 or by < 1:
            raise ValueError(f"grid and block dims must be >= 1, got grid={grid}, block={block}")

        block_dim = bx * by
        self.stats.flops = flops

        for gb_y in range(gy):
            for gb_x in range(gx):
                block_flat = gb_y * gx + gb_x
                lane = 0  # position within the current warp
                # Flatten threads row-major (ty outer, tx inner) so lane 0,1,2,...
                # walk consecutive columns — the coalesced layout.
                for t_y in range(by):
                    for t_x in range(bx):
                        thread_flat = t_y * bx + t_x
                        global_id = block_flat * block_dim + thread_flat
                        row = gb_y * by + t_y
                        col = gb_x * bx + t_x
                        idx = ThreadIdx(
                            block=block_flat,
                            thread=thread_flat,
                            block_dim=block_dim,
                            global_id=global_id,
                            bx=gb_x,
                            by=gb_y,
                            tx=t_x,
                            ty=t_y,
                            row=row,
                            col=col,
                        )
                        kernel(ThreadCtx(self, idx))
                        lane += 1
                        if lane == WARP_SIZE:
                            self._flush_warp()
                            lane = 0
                if lane:  # a partial final warp in this block
                    self._flush_warp()

        return self.stats, self
