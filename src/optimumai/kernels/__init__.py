"""GPU kernels from scratch — a pure-Python simulator, real kernels, and challenges."""

from optimumai.kernels.backends import available_backends, backend_report
from optimumai.kernels.exercises import KernelWorkbench, list_challenges
from optimumai.kernels.kernels import KERNELS, list_kernels, run_kernel
from optimumai.kernels.sim import GpuSim, MemoryStats, ThreadCtx

__all__ = [
    "KERNELS",
    "GpuSim",
    "KernelWorkbench",
    "MemoryStats",
    "ThreadCtx",
    "available_backends",
    "backend_report",
    "list_challenges",
    "list_kernels",
    "run_kernel",
]
