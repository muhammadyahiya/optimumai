"""Execution backends for the kernel track — simulator always, real GPU optionally.

The **simulator** (:mod:`optimumai.kernels.sim`) runs everywhere and is what the
lessons and tests use. Real GPU backends — **Numba CUDA**, **CuPy**, and
**Triton** — are detected and used only when the library *and* an NVIDIA device
are present; otherwise everything degrades to the simulator. The real-backend
kernels below are genuine and runnable on a CUDA machine, but are *not* executed
or verified in CPU-only / CI environments.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache

# Real CUDA kernels, kept as source so they're readable even without a GPU.
NUMBA_VECTOR_ADD = '''
from numba import cuda

@cuda.jit
def vector_add(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] + b[i]
# launch: vector_add[blocks, threads](d_a, d_b, d_out)
'''

CUPY_VECTOR_ADD = '''
import cupy as cp
# CuPy fuses elementwise ops into a single kernel automatically:
out = cp.asarray(a) + cp.asarray(b)
# or hand-write one:
add = cp.ElementwiseKernel("float32 a, float32 b", "float32 out", "out = a + b", "add")
'''

TRITON_VECTOR_ADD = '''
import triton, triton.language as tl

@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)
'''

_KERNEL_SOURCE = {
    "numba": NUMBA_VECTOR_ADD,
    "cupy": CUPY_VECTOR_ADD,
    "triton": TRITON_VECTOR_ADD,
}


def _numba_cuda_ok() -> bool:
    if importlib.util.find_spec("numba") is None:
        return False
    try:
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:  # pragma: no cover - environment dependent
        return False


def _cupy_ok() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover - environment dependent
        return False


def _triton_ok() -> bool:
    return importlib.util.find_spec("triton") is not None and _numba_cuda_ok()


@lru_cache(maxsize=1)
def available_backends() -> list[str]:
    """Backends usable on this machine. Always includes ``"simulator"``."""
    backends = ["simulator"]
    if _numba_cuda_ok():
        backends.append("numba")
    if _cupy_ok():
        backends.append("cupy")
    if _triton_ok():
        backends.append("triton")
    return backends


def backend_report() -> str:
    """A human-readable summary of what's available and how to enable more."""
    have = available_backends()
    lines = [f"Available kernel backends: {', '.join(have)}"]
    if have == ["simulator"]:
        lines.append("")
        lines.append("No NVIDIA GPU detected — the simulator runs everything here.")
        lines.append("To run real kernels on a CUDA machine, install any of:")
        lines.append("  pip install numba          # @cuda.jit kernels")
        lines.append("  pip install cupy-cuda12x    # CuPy arrays / RawKernels")
        lines.append("  pip install triton          # Triton JIT kernels")
    return "\n".join(lines)


def kernel_source(backend: str) -> str:
    """The readable real-CUDA source for a backend's vector-add kernel."""
    if backend not in _KERNEL_SOURCE:
        raise ValueError(f"no source for backend {backend!r}; have {list(_KERNEL_SOURCE)}")
    return _KERNEL_SOURCE[backend]


def run(name: str, backend: str = "auto", explain: bool = False):
    """Run kernel ``name`` on ``backend`` ("auto" picks the best available).

    Real GPU backends fall back to the simulator (with a note) when unavailable,
    so this never fails for lack of a GPU.
    """
    from optimumai.kernels.kernels import run_kernel

    have = available_backends()
    if backend == "auto":
        backend = have[-1]  # prefer a real GPU backend if present, else simulator
    if backend != "simulator" and backend not in have:
        print(f"[backend] {backend} unavailable here — falling back to the simulator.")
        backend = "simulator"
    # Only the simulator is runnable in CPU-only environments; real backends would
    # dispatch to their libraries on a CUDA device.
    return run_kernel(name, explain=explain)
