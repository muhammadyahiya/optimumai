"""Hand-written compute-the-value exercises. Every answer is verified."""

from __future__ import annotations

from optimumai.exercises.engine import Exercise

EXERCISES: list[Exercise] = [
    Exercise("dot_1", "dot", "Compute the dot product of [1, 2, 3] and [4, 5, 6].",
             answer=32.0, hint="Multiply componentwise, then sum: 1·4 + 2·5 + 3·6.",
             explanation="1·4 + 2·5 + 3·6 = 4 + 10 + 18 = 32."),
    Exercise("dot_2", "dot", "Compute the dot product of [1, 0, -1] and [2, 2, 2].",
             answer=0.0, hint="2 + 0 − 2.", explanation="1·2 + 0·2 + (−1)·2 = 0."),
    Exercise("cosine_1", "cosine", "Cosine similarity of [1,2,3] and [2,4,6] (they're parallel).",
             answer=1.0, hint="Parallel vectors point the same way.",
             explanation="The second vector is 2× the first, so the angle is 0 → cos = 1."),
    Exercise("matmul_1", "matmul",
             "For C = [[1,2],[3,4]] @ [[5,6],[7,8]], what is C[0,0]?",
             answer=19.0, hint="Row 0 of A dotted with column 0 of B.",
             explanation="1·5 + 2·7 = 5 + 14 = 19."),
    Exercise("softmax_1", "softmax", "What do the outputs of softmax always sum to?",
             answer=1.0, hint="It's a probability distribution.",
             explanation="Softmax normalizes by the sum of exponentials, so outputs sum to 1."),
    Exercise("derivative_1", "derivative", "What is d/dx of x³ evaluated at x = 2?",
             answer=12.0, hint="d/dx x³ = 3x².",
             explanation="3·2² = 3·4 = 12."),
    Exercise("chain_1", "chain_rule", "What is d/dx of (3x + 1)² evaluated at x = 2?",
             answer=42.0, hint="Chain rule: 2·(3x+1)·3.",
             explanation="2·(3·2+1)·3 = 2·7·3 = 42."),
    Exercise("backprop_1", "backprop", "For L = a·b with a=2, b=−3, what is dL/da?",
             answer=-3.0, hint="dL/da = b.", explanation="The gradient of a·b w.r.t. a is b = −3."),
    Exercise("backprop_2", "backprop", "For L = a·b with a=2, b=−3, what is dL/db?",
             answer=2.0, hint="dL/db = a.", explanation="The gradient of a·b w.r.t. b is a = 2."),
    Exercise("kv_cache_1", "kv_cache",
             "KV cache bytes for layers=2, heads=4, head_dim=8, seq=10, batch=1, 2 bytes/elem. "
             "(Formula: 2·layers·heads·head_dim·seq·batch·bytes.)",
             answer=5120.0, hint="2·2·4·8·10·1·2.",
             explanation="2·2·4·8·10·1·2 = 5120 bytes."),
    Exercise("vram_1", "vram", "A 7-billion-parameter model in fp16 (2 bytes/param): how many "
             "billion bytes do the weights take?", answer=14.0, tolerance=0.1,
             hint="params × bytes-per-param.", explanation="7e9 × 2 = 14e9 bytes ≈ 14 GB."),
    Exercise("attention_1", "attention", "After softmax, what does each row of the attention "
             "weight matrix sum to?", answer=1.0, hint="Softmax over each query's keys.",
             explanation="Each query distributes 100% of its attention across the keys → 1."),
]
