# GPU kernels from scratch

Write per-thread kernels, run them on a pure-Python CUDA-style **simulator** (no
GPU needed), and check them against NumPy — with the thread/memory model and IO
cost reported.

```bash
optimumai kernel                 # list kernels
optimumai kernel matmul          # tiled matmul + the shared-memory tiling win
optimumai kernel flash_attention # fused online-softmax attention — provably exact
optimumai kernel --backends      # what real backends are available here
```

The progression: `scalar_add → vector_add → tiled matmul → softmax → flash attention`.

## Write your own — and get it graded

```python
from optimumai.kernels import KernelWorkbench
wb = KernelWorkbench()
print(wb.get_challenge("vector_add").prompt)

def my_kernel(ctx, inp, out):
    i = ctx.idx.global_id
    if i < out.size:
        out[i] = ctx.gload(inp["a"], i) + ctx.gload(inp["b"], i)

print(wb.submit("vector_add", my_kernel).feedback)   # ✓ correct
```

## Real backends

The default simulator runs everywhere. If you have an NVIDIA GPU, install
`optimumai[gpu]` (Numba) — or CuPy / Triton — and the backends are auto-detected.
The real-CUDA kernels are shown in `optimumai.kernels.backends` for reference.
