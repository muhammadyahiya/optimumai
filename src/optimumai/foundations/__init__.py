"""Foundations of the AI stack — the math, frameworks, and hardware underneath.

Tensors & integration, the PyTorch and JAX programming models (autograd,
grad/jit/vmap), and the systems layer: the CUDA execution & memory model, tiled
matmul kernels, the KV cache, and a VRAM budget calculator.
"""

from optimumai.foundations.cuda_kernel import tiled_matmul, tiled_matmul_trace
from optimumai.foundations.gpu_foundations import (
    memory_hierarchy_trace,
    thread_hierarchy_trace,
)
from optimumai.foundations.jax_foundations import grad_trace, pytree_trace, vmap_trace
from optimumai.foundations.kv_cache import kv_cache, kv_cache_size, kv_cache_trace
from optimumai.foundations.math_foundations import (
    integrate,
    integrate_trace,
    tensor_intro_trace,
)
from optimumai.foundations.pytorch_foundations import (
    pytorch_autograd,
    pytorch_autograd_trace,
)
from optimumai.foundations.vram import vram, vram_estimate, vram_trace

__all__ = [
    "grad_trace",
    "integrate",
    "integrate_trace",
    "kv_cache",
    "kv_cache_size",
    "kv_cache_trace",
    "memory_hierarchy_trace",
    "pytorch_autograd",
    "pytorch_autograd_trace",
    "pytree_trace",
    "tensor_intro_trace",
    "thread_hierarchy_trace",
    "tiled_matmul",
    "tiled_matmul_trace",
    "vmap_trace",
    "vram",
    "vram_estimate",
    "vram_trace",
]
