# GPU kernels from scratch

Write per-thread kernels, run them on a pure-Python CUDA-style **simulator**
(no GPU needed), check them against NumPy, and get graded feedback — with
the thread/memory model and IO cost reported.

```bash
optimumai kernel                       # list all kernels
optimumai kernel matmul                # tiled matmul + the shared-memory tiling win
optimumai kernel flash_attention       # fused online-softmax attention — provably exact
optimumai kernel --backends            # what real backends are available here
```

The progression: `scalar_add → vector_add → tiled matmul → softmax → flash attention`

---

## The simulator model

The pure-Python simulator (`optimumai.kernels.sim`) models:

- **Thread grid** — a 2-D grid of blocks, each block a 1-D array of threads
- **Global memory** — read with `ctx.gload(buf, idx)`, write with `ctx.gstore(buf, idx, val)`
- **Shared memory** — `ctx.salloc(size)`, `ctx.sload(smem, idx)`, `ctx.sstore(smem, idx, val)`
- **Synchronization** — `ctx.syncthreads()` (barrier within a block)
- **Thread index** — `ctx.idx.global_id`, `ctx.idx.block_id`, `ctx.idx.thread_id`

All memory accesses are tracked, so the simulator can report IO efficiency.

---

## The built-in kernels

### scalar_add

Simplest possible: one global load per thread, one add, one store.

```bash
optimumai kernel scalar_add
```

### vector_add

Parallel element-wise addition — the "Hello World" of GPU programming.

```bash
optimumai kernel vector_add
```

### tiled matmul

Each block computes a tile of the output by loading tiles of A and B into
shared memory, then reading from shared memory instead of global memory.

```bash
optimumai kernel matmul
```

### softmax

Two-pass online algorithm: find the max (one pass), then compute exp(xᵢ−max)
and normalize (second pass).

```bash
optimumai kernel softmax
```

### flash_attention

Fused online-softmax attention: tiles Q, K, V through fast on-chip shared
memory, rescaling the online softmax as each K/V tile is processed. Provably
exact — verified error ~1e-16 vs. naive attention.

```bash
optimumai kernel flash_attention
```

---

## Write your own kernel — and get it graded

```python
from optimumai.kernels import KernelWorkbench

wb = KernelWorkbench()

# See the challenge prompt
print(wb.get_challenge("vector_add").prompt)

# Write the kernel
def my_vector_add(ctx, inp, out):
    i = ctx.idx.global_id
    if i < out.size:
        out[i] = ctx.gload(inp["a"], i) + ctx.gload(inp["b"], i)

# Submit for grading
result = wb.submit("vector_add", my_vector_add)
print(result.feedback)     # "✓ Correct — your kernel matches the reference."
print(result.io_report)    # memory access counts and efficiency
```

Available challenges: `scalar_add`, `vector_add`, `matmul`, `softmax`,
`flash_attention`.

---

## Real backends

The simulator runs everywhere — no GPU, no special installation.

If you have an NVIDIA GPU, install a real backend and it will be auto-detected:

| Backend | Install | Notes |
|---|---|---|
| **Numba** | `pip install "optimumai[gpu]"` | JIT-compiled CUDA kernels |
| **CuPy** | `pip install cupy-cudaXX` | Drop-in NumPy for GPU |
| **Triton** | `pip install triton` | OpenAI Triton, Python-first GPU kernels |

```bash
optimumai kernel --backends    # see what's available on this machine
```

The built-in kernel implementations for real backends are in
`optimumai.kernels.backends` for reference — copy and modify.
