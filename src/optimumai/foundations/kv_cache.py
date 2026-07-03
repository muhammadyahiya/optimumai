"""Why context length eats VRAM: the transformer KV cache.

During generation a transformer caches the Key and Value tensors for every token
it has already seen, so each new token attends to the past without recomputing
it. That cache grows *linearly* with sequence length and batch size, and at long
context it — not the weights — is what fills the GPU. The standard fixes all
shrink the same term: Multi-Query Attention (one KV head), Grouped-Query
Attention (a handful of KV heads), and 8-bit KV quantization (fewer bytes per
element). Run :func:`kv_cache_trace` with ``explain=True`` to see the arithmetic.
"""

from __future__ import annotations

from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace

_KB = 1024
_MB = 1024**2
_GB = 1024**3


def _human_bytes(n: float) -> str:
    """Format a byte count as MB or GB (whichever reads best)."""
    if n >= _GB:
        return f"{n / _GB:.2f} GB"
    if n >= _MB:
        return f"{n / _MB:.2f} MB"
    if n >= _KB:
        return f"{n / _KB:.2f} KB"
    return f"{n:.0f} B"


def kv_cache_size(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int = 1,
    bytes_per_elem: int = 2,
    kv_heads: int | None = None,
) -> int:
    """Return the KV cache size in bytes (no trace).

    ``kv_heads`` defaults to ``n_heads`` (Multi-Head Attention); pass a smaller
    number for Grouped-Query Attention or ``1`` for Multi-Query Attention.
    """
    kvh = n_heads if kv_heads is None else kv_heads
    # 2 for K and V; each is (layers × kv_heads × head_dim) per token.
    return int(2 * n_layers * kvh * head_dim * seq_len * batch * bytes_per_elem)


def kv_cache_trace(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int = 1,
    bytes_per_elem: int = 2,
    kv_heads: int | None = None,
) -> Trace:
    """Build the full trace of a transformer KV cache's memory footprint."""
    if kv_heads is None:
        kv_heads = n_heads

    t = Trace(
        op="kv_cache",
        formula=(
            "KV bytes = 2 × n_layers × kv_heads × head_dim × seq_len × batch × bytes_per_elem"
        ),
        complexity="linear in sequence length × batch",
        why_ai=[
            "At long context the KV cache — not the model weights — is what fills VRAM",
            "It grows linearly with sequence length and batch, so doubling context "
            "doubles the cache",
            "MQA (1 KV head) and GQA (a few KV heads) cut the cache by the head "
            "reduction factor",
            "8-bit KV quantization halves bytes_per_elem (2 → 1) for another 2× saving",
        ],
        meta={
            "n_layers": n_layers,
            "n_heads": n_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "seq_len": seq_len,
            "batch": batch,
            "bytes_per_elem": bytes_per_elem,
        },
    )

    # Per token, per layer: K and V each hold kv_heads × head_dim elements.
    per_token = 2 * n_layers * kv_heads * head_dim * bytes_per_elem
    t.add(
        "Bytes per token",
        f"2 (K,V) × {n_layers} layers × {kv_heads} kv_heads × {head_dim} head_dim "
        f"× {bytes_per_elem} B  =  {_human_bytes(per_token)}",
        per_token,
        detail="Each generated token appends this many bytes to the cache, forever.",
    )

    per_seq = per_token * seq_len
    t.add(
        f"Total for {seq_len} tokens",
        f"{_human_bytes(per_token)} × {seq_len}  =  {_human_bytes(per_seq)}",
        per_seq,
        detail="Linear in sequence length: a 2× longer context needs 2× the cache.",
    )

    total = per_seq * batch
    t.add(
        f"Scale by batch = {batch}",
        f"{_human_bytes(per_seq)} × {batch}  =  {_human_bytes(total)}",
        total,
        detail="Serving many requests at once multiplies the cache by the batch size.",
    )

    # Compare attention variants at this same config.
    mha = kv_cache_size(n_layers, n_heads, head_dim, seq_len, batch, bytes_per_elem, n_heads)
    gqa_heads = max(1, n_heads // 8)
    gqa = kv_cache_size(n_layers, n_heads, head_dim, seq_len, batch, bytes_per_elem, gqa_heads)
    mqa = kv_cache_size(n_layers, n_heads, head_dim, seq_len, batch, bytes_per_elem, 1)
    t.add(
        "Compare MHA vs GQA vs MQA",
        (
            f"MHA ({n_heads} kv_heads) = {_human_bytes(mha)}\n"
            f"GQA ({gqa_heads} kv_heads) = {_human_bytes(gqa)}  "
            f"→  {mha / gqa:.0f}× smaller\n"
            f"MQA (1 kv_head)  = {_human_bytes(mqa)}  →  {mha / mqa:.0f}× smaller"
        ),
        {"mha": mha, "gqa": gqa, "mqa": mqa},
        detail="Fewer KV heads share their K/V across query heads — the memory drops "
        "by exactly the head reduction factor, at a small quality cost.",
    )

    t.result = total
    return t


def kv_cache(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int = 1,
    bytes_per_elem: int = 2,
    kv_heads: int | None = None,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> int:
    """Return the KV cache size in bytes. Set ``explain=True`` to print the trace."""
    t = kv_cache_trace(
        n_layers, n_heads, head_dim, seq_len, batch, bytes_per_elem, kv_heads
    )
    return t.render(level) if explain else t.result


def demo() -> Trace:
    """A Llama-2-7B-ish config: 32 layers, 32 heads, head_dim 128, 4096 ctx, fp16."""
    return kv_cache_trace(
        n_layers=32,
        n_heads=32,
        head_dim=128,
        seq_len=4096,
        batch=1,
        bytes_per_elem=2,
    )
