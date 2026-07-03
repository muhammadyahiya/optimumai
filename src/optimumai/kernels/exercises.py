"""Fill-in-the-kernel challenges — get your hands dirty writing GPU kernels.

You write the per-thread body; the :class:`KernelWorkbench` launches it on the
simulator over the right grid, checks the output against a NumPy reference, and
gives feedback. Your kernel is a function ``k(ctx, inp, out)`` where ``ctx`` is a
:class:`~optimumai.kernels.sim.ThreadCtx`, ``inp`` is a dict of input arrays, and
``out`` is the output array you must fill with ``ctx.gstore``.

    from optimumai.kernels.exercises import KernelWorkbench
    wb = KernelWorkbench()
    def my_add(ctx, inp, out):
        i = ctx.idx.global_id
        if i < out.size:
            out[i] = ctx.gload(inp["a"], i) + ctx.gload(inp["b"], i)
    print(wb.submit("vector_add", my_add).correct)   # True
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from optimumai.kernels.sim import GpuSim


@dataclass
class KernelChallenge:
    """One fill-in-the-kernel task."""

    id: str
    prompt: str
    signature_hint: str
    make_inputs: Callable[[], dict[str, Any]]   # -> {"inp": dict, "out": ndarray, "grid":.., "block":..}
    reference: Callable[[dict], np.ndarray]      # (inp) -> expected output
    solution: str
    explanation: str


@dataclass
class SubmitResult:
    """The graded outcome of a submitted kernel."""

    correct: bool
    max_error: float
    feedback: str


def _vec_inputs() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    n = 8
    return {"inp": {"a": rng.normal(size=n), "b": rng.normal(size=n)},
            "out": np.zeros(n), "grid": 2, "block": 4}


def _saxpy_inputs() -> dict[str, Any]:
    rng = np.random.default_rng(1)
    n = 8
    return {"inp": {"x": rng.normal(size=n), "y": rng.normal(size=n), "alpha": 2.0},
            "out": np.zeros(n), "grid": 2, "block": 4}


def _relu_inputs() -> dict[str, Any]:
    rng = np.random.default_rng(2)
    n = 8
    return {"inp": {"x": rng.normal(size=n)}, "out": np.zeros(n), "grid": 2, "block": 4}


def _matmul_inputs() -> dict[str, Any]:
    rng = np.random.default_rng(3)
    M = N = K = 4
    A = rng.normal(size=(M, K))
    B = rng.normal(size=(K, N))
    return {"inp": {"A": A.ravel(), "B": B.ravel(), "M": M, "N": N, "K": K},
            "out": np.zeros(M * N), "grid": (1, 1), "block": (N, M)}


CHALLENGES: dict[str, KernelChallenge] = {
    "vector_add": KernelChallenge(
        id="vector_add",
        prompt="Each thread adds one element: out[i] = a[i] + b[i]. Guard the tail (i ≥ out.size).",
        signature_hint="def k(ctx, inp, out):  # use ctx.idx.global_id, ctx.gload(inp['a'], i), ctx.gstore",
        make_inputs=_vec_inputs,
        reference=lambda inp: inp["a"] + inp["b"],
        solution=("def k(ctx, inp, out):\n"
                  "    i = ctx.idx.global_id\n"
                  "    if i < out.size:\n"
                  "        out[i] = ctx.gload(inp['a'], i) + ctx.gload(inp['b'], i)"),
        explanation="One thread per element; the global id maps thread → element.",
    ),
    "saxpy": KernelChallenge(
        id="saxpy",
        prompt="SAXPY: out[i] = alpha*x[i] + y[i] with alpha = inp['alpha'] (a Python float).",
        signature_hint="def k(ctx, inp, out):  # inp['alpha'] is a scalar, not an array",
        make_inputs=_saxpy_inputs,
        reference=lambda inp: inp["alpha"] * inp["x"] + inp["y"],
        solution=("def k(ctx, inp, out):\n"
                  "    i = ctx.idx.global_id\n"
                  "    if i < out.size:\n"
                  "        out[i] = inp['alpha'] * ctx.gload(inp['x'], i) + ctx.gload(inp['y'], i)"),
        explanation="SAXPY (single-precision a·x + y) is the classic BLAS level-1 kernel.",
    ),
    "relu": KernelChallenge(
        id="relu",
        prompt="ReLU: out[i] = max(0, x[i]).",
        signature_hint="def k(ctx, inp, out):  # max(0.0, value)",
        make_inputs=_relu_inputs,
        reference=lambda inp: np.maximum(0.0, inp["x"]),
        solution=("def k(ctx, inp, out):\n"
                  "    i = ctx.idx.global_id\n"
                  "    if i < out.size:\n"
                  "        out[i] = max(0.0, ctx.gload(inp['x'], i))"),
        explanation="Elementwise activation — the simplest nonlinearity, one thread per element.",
    ),
    "matmul_cell": KernelChallenge(
        id="matmul_cell",
        prompt="Matmul: thread (row, col) computes one output cell C[row,col] = Σₖ A[row,k]·B[k,col]. "
               "Use ctx.idx.row / ctx.idx.col; A,B are row-major flattened; dims in inp['M','N','K'].",
        signature_hint="def k(ctx, inp, out):  # r=ctx.idx.row, c=ctx.idx.col; index A[r*K+k], B[k*N+c]",
        make_inputs=_matmul_inputs,
        reference=lambda inp: (inp["A"].reshape(inp["M"], inp["K"])
                               @ inp["B"].reshape(inp["K"], inp["N"])).ravel(),
        solution=("def k(ctx, inp, out):\n"
                  "    r, c = ctx.idx.row, ctx.idx.col\n"
                  "    M, N, K = inp['M'], inp['N'], inp['K']\n"
                  "    if r < M and c < N:\n"
                  "        acc = 0.0\n"
                  "        for k in range(K):\n"
                  "            acc += ctx.gload(inp['A'], r*K+k) * ctx.gload(inp['B'], k*N+c)\n"
                  "        out[r*N+c] = acc"),
        explanation="Each thread owns one output cell and reads a full row of A and column of B.",
    ),
}


class KernelWorkbench:
    """Run and grade user-written kernels against a NumPy reference."""

    def list_challenges(self) -> list[str]:
        return list(CHALLENGES)

    def get_challenge(self, challenge_id: str) -> KernelChallenge:
        if challenge_id not in CHALLENGES:
            raise KeyError(f"unknown challenge {challenge_id!r}; have {list(CHALLENGES)}")
        return CHALLENGES[challenge_id]

    def reveal(self, challenge_id: str) -> str:
        """Return a correct reference solution for the challenge."""
        return self.get_challenge(challenge_id).solution

    def submit(self, challenge_id: str, user_kernel: Callable) -> SubmitResult:
        """Run ``user_kernel`` on the simulator and grade it against the reference."""
        challenge = self.get_challenge(challenge_id)
        setup = challenge.make_inputs()
        inp, out = setup["inp"], setup["out"]
        expected = challenge.reference(inp)

        def wrapped(ctx) -> None:
            user_kernel(ctx, inp, out)

        try:
            GpuSim().launch(setup["grid"], setup["block"], wrapped)
        except Exception as exc:  # a bug in the user's kernel
            return SubmitResult(False, float("inf"), f"Your kernel raised {type(exc).__name__}: {exc}")

        err = float(np.max(np.abs(out - expected)))
        if err <= 1e-9:
            return SubmitResult(True, err, "✓ Correct — your kernel matches the reference.")
        if np.allclose(out, 0.0):
            return SubmitResult(False, err, "Output is all zeros — are you writing with ctx.gstore(out, i, ...)?")
        return SubmitResult(False, err, f"Off by up to {err:.4g} — check your indexing (thread id / row,col).")


def list_challenges() -> list[str]:
    return list(CHALLENGES)


def get_challenge(challenge_id: str) -> KernelChallenge:
    return KernelWorkbench().get_challenge(challenge_id)
