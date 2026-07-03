import numpy as np

from optimumai.kernels.backends import available_backends, backend_report
from optimumai.kernels.exercises import KernelWorkbench, list_challenges
from optimumai.kernels.kernels import (
    KERNELS,
    flash_attention_kernel_trace,
    list_kernels,
    matmul_trace,
    run_kernel,
    vector_add_trace,
)
from optimumai.kernels.sim import GpuSim


# --- simulator ----------------------------------------------------------------
def test_sim_runs_and_counts():
    n = 8
    a = np.arange(n, dtype=float)
    out = np.zeros(n)

    def kernel(ctx):
        i = ctx.idx.global_id
        if i < n:
            ctx.gstore(out, i, ctx.gload(a, i) * 2)

    stats, _ = GpuSim().launch(2, 4, kernel, flops=n)
    assert np.allclose(out, a * 2)
    assert stats.global_reads == n and stats.global_writes == n
    assert stats.coalesced is True


# --- kernels correctness ------------------------------------------------------
def test_all_kernels_run_and_are_correct():
    assert set(list_kernels()) == set(KERNELS)
    for name in list_kernels():
        run_kernel(name)  # must not raise


def test_vector_add_matches_numpy():
    t = vector_add_trace(8)
    assert t.meta["max_error"] < 1e-12


def test_matmul_matches_numpy_and_tiling_helps():
    t = matmul_trace(M=8, N=8, K=8, tile=4)
    assert t.meta["max_error"] < 1e-9
    assert t.meta["tiled_reads"] < t.meta["naive_reads"]


def test_flash_attention_kernel_is_exact():
    assert flash_attention_kernel_trace().meta["max_abs_error"] < 1e-9


# --- backends -----------------------------------------------------------------
def test_backends_always_include_simulator():
    assert "simulator" in available_backends()
    assert isinstance(backend_report(), str)


# --- fill-in-the-kernel workbench ---------------------------------------------
def test_workbench_grades_correct_and_wrong():
    wb = KernelWorkbench()
    assert set(list_challenges()) >= {"vector_add", "saxpy", "relu", "matmul_cell"}

    def good(ctx, inp, out):
        i = ctx.idx.global_id
        if i < out.size:
            out[i] = ctx.gload(inp["a"], i) + ctx.gload(inp["b"], i)

    def wrong(ctx, inp, out):
        i = ctx.idx.global_id
        if i < out.size:
            out[i] = ctx.gload(inp["a"], i)

    assert wb.submit("vector_add", good).correct is True
    assert wb.submit("vector_add", wrong).correct is False


def test_workbench_reveal_solves_every_challenge():
    wb = KernelWorkbench()
    for cid in list_challenges():
        namespace: dict = {}
        exec(wb.reveal(cid), namespace)  # noqa: S102 - trusted reference solution
        assert wb.submit(cid, namespace["k"]).correct is True, cid
