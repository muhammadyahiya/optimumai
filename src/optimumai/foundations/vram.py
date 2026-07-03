"""Where the memory actually goes: a VRAM budget for LLMs.

"Will this model fit?" decomposes into a few concrete terms. **Weights** are the
parameters themselves (``params × bytes``). Training then adds **gradients** (one
per weight) and **optimizer states** (Adam keeps two moment buffers in fp32 — 8
bytes per parameter — which is why training a 7B model needs roughly 4× the
weight memory). **Activations** are the intermediate tensors kept for the
backward pass, estimated here and clearly labelled as an estimate. Inference
drops grads and optimizer states, leaving weights plus the **KV cache**, which is
what grows with context length. Run :func:`vram_trace` with ``explain=True``.
"""

from __future__ import annotations

from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace
from optimumai.foundations.kv_cache import kv_cache_size

_GB = 1024**3
_BILLION = 1_000_000_000

# Optimizer state bytes per parameter:
#   Adam: two fp32 buffers (m, v) = 2 × 4 = 8 bytes/param.
#   SGD with momentum: one fp32 buffer = 4 bytes/param.
_OPTIMIZER_BYTES_PER_PARAM = {"adam": 8, "sgd": 4}


def _human_gb(bytes_val: float) -> float:
    """Convert a byte count to gibibytes."""
    return bytes_val / _GB


def vram_estimate(
    params_billions: float,
    precision_bytes: int = 2,
    training: bool = True,
    optimizer: str = "adam",
    batch: int = 1,
    seq_len: int = 2048,
    hidden: int = 4096,
    n_layers: int = 32,
    activation_factor: float | None = None,
) -> float:
    """Return the estimated total VRAM in GB (no trace)."""
    params = params_billions * _BILLION

    weights = params * precision_bytes
    grads = params * precision_bytes if training else 0.0

    opt_key = optimizer.lower()
    if opt_key not in _OPTIMIZER_BYTES_PER_PARAM:
        raise ValueError(f"optimizer must be 'adam' or 'sgd', got {optimizer!r}")
    opt_states = params * _OPTIMIZER_BYTES_PER_PARAM[opt_key] if training else 0.0

    if training:
        # Rough, documented approximation: activations kept for backprop scale
        # with batch × seq_len × hidden × n_layers × precision, times a small
        # constant (~a few tensors saved per layer).
        factor = 4.0 if activation_factor is None else activation_factor
        activations = factor * batch * seq_len * hidden * n_layers * precision_bytes
    else:
        activations = 0.0

    if training:
        kv = 0.0
    else:
        # Inference: KV cache dominates alongside weights. Estimate head_dim/heads
        # from the hidden size (a common 128-dim-per-head convention).
        n_heads = max(1, hidden // 128)
        head_dim = hidden // n_heads
        kv = float(
            kv_cache_size(n_layers, n_heads, head_dim, seq_len, batch, precision_bytes)
        )

    total = weights + grads + opt_states + activations + kv
    return _human_gb(total)


def vram_trace(
    params_billions: float,
    precision_bytes: int = 2,
    training: bool = True,
    optimizer: str = "adam",
    batch: int = 1,
    seq_len: int = 2048,
    hidden: int = 4096,
    n_layers: int = 32,
    activation_factor: float | None = None,
) -> Trace:
    """Build the full trace of an LLM's VRAM budget, component by component."""
    opt_key = optimizer.lower()
    if opt_key not in _OPTIMIZER_BYTES_PER_PARAM:
        raise ValueError(f"optimizer must be 'adam' or 'sgd', got {optimizer!r}")

    params = params_billions * _BILLION

    t = Trace(
        op="vram",
        formula="VRAM = weights + gradients + optimizer_states + activations + kv_cache",
        complexity="weights/grads/optimizer O(params); activations & KV O(batch·seq)",
        why_ai=[
            "A 7B model is ~14 GB in fp16 for weights alone (7e9 × 2 bytes)",
            "Training needs ~4× more than inference: gradients (1×) plus Adam's two "
            "fp32 moment buffers (another 2× the fp16 weights)",
            "Inference memory is dominated by weights + the KV cache, which grows "
            "with context length",
            "This budget is exactly why fine-tuning reaches for LoRA (train tiny "
            "adapters) and quantization (fewer bytes per weight)",
        ],
        meta={
            "params_billions": params_billions,
            "precision_bytes": precision_bytes,
            "training": training,
            "optimizer": opt_key,
            "batch": batch,
            "seq_len": seq_len,
        },
    )

    weights = params * precision_bytes
    t.add(
        "Weights",
        f"{params_billions}e9 params × {precision_bytes} B  =  {_human_gb(weights):.2f} GB",
        weights,
        detail="Always resident. fp16/bf16 = 2 B/param; int8 = 1 B; int4 = 0.5 B.",
    )

    if training:
        grads = params * precision_bytes
        t.add(
            "Gradients (training)",
            f"one per weight: {params_billions}e9 × {precision_bytes} B  "
            f"=  {_human_gb(grads):.2f} GB",
            grads,
            detail="Backprop stores a gradient for every trainable parameter.",
        )

        opt_bytes = _OPTIMIZER_BYTES_PER_PARAM[opt_key]
        opt_states = params * opt_bytes
        opt_note = (
            "Adam keeps two fp32 buffers (m, v) = 8 B/param"
            if opt_key == "adam"
            else "SGD-momentum keeps one fp32 buffer = 4 B/param"
        )
        t.add(
            f"Optimizer states ({opt_key})",
            f"{params_billions}e9 × {opt_bytes} B  =  {_human_gb(opt_states):.2f} GB",
            opt_states,
            detail=opt_note,
        )

        factor = 4.0 if activation_factor is None else activation_factor
        activations = factor * batch * seq_len * hidden * n_layers * precision_bytes
        t.add(
            "Activations (training, estimate)",
            f"≈ {factor:g} × batch {batch} × seq {seq_len} × hidden {hidden} × "
            f"layers {n_layers} × {precision_bytes} B  =  {_human_gb(activations):.2f} GB",
            activations,
            detail="ESTIMATE only — real usage depends on checkpointing, fusion, and "
            "attention implementation. Gradient checkpointing trades compute to cut this.",
        )
        kv = 0.0
    else:
        grads = opt_states = activations = 0.0
        n_heads = max(1, hidden // 128)
        head_dim = hidden // n_heads
        kv = float(
            kv_cache_size(n_layers, n_heads, head_dim, seq_len, batch, precision_bytes)
        )
        t.add(
            "KV cache (inference, estimate)",
            f"2 × {n_layers} layers × {n_heads} heads × {head_dim} head_dim × "
            f"seq {seq_len} × batch {batch} × {precision_bytes} B  "
            f"=  {_human_gb(kv):.2f} GB",
            kv,
            detail="ESTIMATE — head count inferred from hidden size (128-dim heads). "
            "Grows linearly with context length; see foundations.kv_cache.",
        )

    total_bytes = weights + grads + opt_states + activations + kv
    total_gb = _human_gb(total_bytes)
    mode = "training" if training else "inference"
    t.add(
        f"Total VRAM ({mode})",
        f"sum of the above  =  {total_gb:.2f} GB",
        total_gb,
        detail="Add ~10-20% headroom for fragmentation, CUDA context, and workspace.",
    )

    t.result = total_gb
    return t


def vram(
    params_billions: float,
    precision_bytes: int = 2,
    training: bool = True,
    optimizer: str = "adam",
    batch: int = 1,
    seq_len: int = 2048,
    hidden: int = 4096,
    n_layers: int = 32,
    activation_factor: float | None = None,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> float:
    """Return the estimated VRAM in GB. Set ``explain=True`` to print the trace."""
    t = vram_trace(
        params_billions,
        precision_bytes=precision_bytes,
        training=training,
        optimizer=optimizer,
        batch=batch,
        seq_len=seq_len,
        hidden=hidden,
        n_layers=n_layers,
        activation_factor=activation_factor,
    )
    return t.render(level) if explain else t.result


def demo() -> Trace:
    """A 7B model in fp16 training with Adam — the canonical "why is it 4×?" case."""
    return vram_trace(
        params_billions=7.0,
        precision_bytes=2,
        training=True,
        optimizer="adam",
        batch=1,
        seq_len=2048,
        hidden=4096,
        n_layers=32,
    )
