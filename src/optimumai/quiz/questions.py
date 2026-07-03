"""The hand-written question bank powering active recall in OptimumAI.

Every entry in :data:`QUESTIONS` maps a curriculum lesson ``id`` (the exact ids
from :mod:`optimumai.curriculum.course`) to a small list of multiple-choice
:class:`~optimumai.quiz.engine.Question` objects. Each question mixes conceptual
recall with a little arithmetic you can do in your head, has four plausible
choices, records the index of the correct one, and carries a one-sentence
explanation. Numeric answers are double-checked against the concept modules they
come from (e.g. ``dot([1,2,3],[4,5,6]) = 32``).

The engine treats this bank as pure data: it never mutates it and only ever
reads ``QUESTIONS``.
"""

from __future__ import annotations

from optimumai.quiz.engine import Question

QUESTIONS: dict[str, list[Question]] = {
    # --- Linear Algebra ------------------------------------------------------
    "dot": [
        Question(
            prompt="What is the dot product dot([1, 2, 3], [4, 5, 6])?",
            choices=["21", "32", "15", "45"],
            answer=1,
            explanation="1·4 + 2·5 + 3·6 = 4 + 10 + 18 = 32.",
            lesson_id="dot",
        ),
        Question(
            prompt="The dot product of two vectors produces a…",
            choices=[
                "vector of the same length",
                "single scalar",
                "matrix",
                "vector of length 2",
            ],
            answer=1,
            explanation="a · b sums the element-wise products into one scalar.",
            lesson_id="dot",
        ),
        Question(
            prompt="Two nonzero vectors are orthogonal exactly when their dot product is…",
            choices=["1", "0", "-1", "their length"],
            answer=1,
            explanation="A zero dot product means a 90° angle: the vectors are orthogonal.",
            lesson_id="dot",
        ),
    ],
    "cosine": [
        Question(
            prompt="What is the cosine similarity of [1, 2, 3] and [2, 4, 6]?",
            choices=["0.0", "0.5", "1.0", "3.0"],
            answer=2,
            explanation="The second vector is 2× the first, so they point the same way: cos = 1.",
            lesson_id="cosine",
        ),
        Question(
            prompt="Cosine similarity differs from a raw dot product because it…",
            choices=[
                "ignores the sign of the vectors",
                "normalizes out the vectors' magnitudes",
                "requires integer inputs",
                "is always negative",
            ],
            answer=1,
            explanation="Dividing by both norms leaves only the angle, independent of length.",
            lesson_id="cosine",
        ),
        Question(
            prompt="What is the range of cosine similarity for real vectors?",
            choices=["[0, 1]", "[-1, 1]", "[0, ∞)", "(-∞, ∞)"],
            answer=1,
            explanation="cos(θ) ranges from -1 (opposite) through 0 (orthogonal) to 1 (aligned).",
            lesson_id="cosine",
        ),
    ],
    "matmul": [
        Question(
            prompt="For [[1, 2], [3, 4]] · [[5, 6], [7, 8]], what is the top-left entry?",
            choices=["19", "17", "23", "12"],
            answer=0,
            explanation="Row 0 · column 0 = 1·5 + 2·7 = 5 + 14 = 19.",
            lesson_id="matmul",
        ),
        Question(
            prompt="To multiply an (m × k) matrix by a (k × n) matrix, what must match?",
            choices=[
                "m must equal n",
                "the inner dimensions k must match",
                "all four dimensions must be equal",
                "nothing — any shapes work",
            ],
            answer=1,
            explanation="The left matrix's columns must equal the right matrix's rows.",
            lesson_id="matmul",
        ),
        Question(
            prompt="Each entry of a matrix product is computed as…",
            choices=[
                "an element-wise product",
                "a dot product of a row and a column",
                "the sum of two entries",
                "the larger of two entries",
            ],
            answer=1,
            explanation="Entry (i, j) is the dot product of row i of A and column j of B.",
            lesson_id="matmul",
        ),
    ],
    # --- Probability & Transformers -----------------------------------------
    "softmax": [
        Question(
            prompt="The outputs of a softmax over a vector always sum to…",
            choices=["0", "1", "the input's max", "the vector length"],
            answer=1,
            explanation="Softmax normalizes exponentials into a distribution that sums to 1.",
            lesson_id="softmax",
        ),
        Question(
            prompt="Why does a stable softmax subtract the max of the inputs first?",
            choices=[
                "to change the result",
                "to avoid overflow in e^x while leaving the output unchanged",
                "to make outputs sum to the max",
                "to speed up multiplication",
            ],
            answer=1,
            explanation="Shifting by a constant cancels in the ratio and prevents overflow.",
            lesson_id="softmax",
        ),
        Question(
            prompt="As the temperature T → 0, a softmax distribution becomes…",
            choices=[
                "uniform (flat)",
                "sharper, approaching a one-hot argmax",
                "all zeros",
                "unchanged",
            ],
            answer=1,
            explanation="Low temperature sharpens the distribution toward the largest logit.",
            lesson_id="softmax",
        ),
    ],
    "attention": [
        Question(
            prompt="Scaled dot-product attention computes…",
            choices=[
                "softmax(Q + K)·V",
                "softmax(QKᵀ / √dₖ)·V",
                "Q·K·V",
                "softmax(V)·Q",
            ],
            answer=1,
            explanation="Scores QKᵀ are scaled by √dₖ, softmaxed into weights, then applied to V.",
            lesson_id="attention",
        ),
        Question(
            prompt="Why are the QKᵀ scores divided by √dₖ before the softmax?",
            choices=[
                "to make them sum to 1",
                "to keep the dot products from growing large and saturating the softmax",
                "to convert them to integers",
                "to add positional information",
            ],
            answer=1,
            explanation="Scaling counteracts the variance growth of dot products in high dims.",
            lesson_id="attention",
        ),
        Question(
            prompt="In attention, which matrix is weighted and summed to form the output?",
            choices=["the Query Q", "the Key K", "the Value V", "the score matrix"],
            answer=2,
            explanation="Attention weights (from softmax of QKᵀ) are applied to the Value vectors.",
            lesson_id="attention",
        ),
    ],
    "multihead": [
        Question(
            prompt="Multi-head attention runs several attention heads in parallel so it can…",
            choices=[
                "reduce the number of parameters to zero",
                "attend to different relationships in different subspaces at once",
                "avoid using softmax",
                "make attention sequential",
            ],
            answer=1,
            explanation="Each head learns its own projections, capturing distinct patterns.",
            lesson_id="multihead",
        ),
        Question(
            prompt="A causal (autoregressive) mask in multi-head attention prevents a token from…",
            choices=[
                "attending to itself",
                "attending to future tokens",
                "attending to past tokens",
                "using more than one head",
            ],
            answer=1,
            explanation="The mask zeros out future positions so predictions use only the past.",
            lesson_id="multihead",
        ),
        Question(
            prompt="With d_model split across h heads, each head typically has dimension…",
            choices=["d_model", "d_model · h", "d_model / h", "h"],
            answer=2,
            explanation="Heads partition the model dimension, so head_dim = d_model / h.",
            lesson_id="multihead",
        ),
    ],
    "positional": [
        Question(
            prompt="Why do transformers need positional encodings at all?",
            choices=[
                "attention is permutation-invariant and has no built-in notion of order",
                "softmax cannot handle large inputs",
                "to reduce the parameter count",
                "to normalize the embeddings",
            ],
            answer=0,
            explanation="Self-attention treats inputs as a set, so order must be injected.",
            lesson_id="positional",
        ),
        Question(
            prompt="Sinusoidal positional encodings use sine and cosine functions of…",
            choices=[
                "the token's embedding value",
                "the position, at a range of frequencies",
                "the attention scores",
                "the batch size",
            ],
            answer=1,
            explanation="Each dimension is a sinusoid of the position at a different frequency.",
            lesson_id="positional",
        ),
    ],
    "embeddings": [
        Question(
            prompt="An embedding layer turns a discrete token id into…",
            choices=[
                "a one-hot vector",
                "a learnable dense vector",
                "a scalar probability",
                "a softmax distribution",
            ],
            answer=1,
            explanation="Embeddings map each token id to a trainable dense vector.",
            lesson_id="embeddings",
        ),
        Question(
            prompt="A token embedding lookup is mathematically equivalent to…",
            choices=[
                "a softmax over the vocabulary",
                "selecting a row of the embedding matrix (a one-hot × matrix product)",
                "computing a dot product with every other token",
                "adding positional encodings",
            ],
            answer=1,
            explanation="Indexing row t equals multiplying a one-hot by the embedding matrix.",
            lesson_id="embeddings",
        ),
        Question(
            prompt="Similar meanings tend to produce embeddings that are…",
            choices=[
                "orthogonal",
                "close together (high cosine similarity)",
                "identical",
                "always unit length",
            ],
            answer=1,
            explanation="Training pushes semantically related tokens to nearby vectors.",
            lesson_id="embeddings",
        ),
    ],
    # --- Calculus & Autograd -------------------------------------------------
    "derivative": [
        Question(
            prompt="What is the derivative of f(x) = x³ evaluated at x = 2?",
            choices=["6", "8", "12", "3"],
            answer=2,
            explanation="f'(x) = 3x², so f'(2) = 3·4 = 12.",
            lesson_id="derivative",
        ),
        Question(
            prompt="A derivative measures…",
            choices=[
                "the area under a curve",
                "the instantaneous rate of change (slope) of a function",
                "the maximum of a function",
                "the average value of a function",
            ],
            answer=1,
            explanation="The derivative is the slope of the tangent line at a point.",
            lesson_id="derivative",
        ),
        Question(
            prompt="If a function's derivative at a point is 0, that point is a…",
            choices=[
                "guaranteed global minimum",
                "stationary point (possible min, max, or saddle)",
                "discontinuity",
                "point of maximum slope",
            ],
            answer=1,
            explanation="A zero slope marks a stationary point, but not necessarily a minimum.",
            lesson_id="derivative",
        ),
    ],
    "chain_rule": [
        Question(
            prompt="The chain rule states that for y = f(g(x)), dy/dx equals…",
            choices=[
                "f'(x) · g'(x)",
                "f'(g(x)) · g'(x)",
                "f'(g(x)) + g'(x)",
                "f(g'(x))",
            ],
            answer=1,
            explanation="Differentiate the outer at the inner value, times the inner derivative.",
            lesson_id="chain_rule",
        ),
        Question(
            prompt="Which core algorithm is essentially a systematic use of the chain rule?",
            choices=[
                "gradient descent",
                "backpropagation",
                "softmax",
                "matrix multiplication",
            ],
            answer=1,
            explanation="Backprop composes local derivatives via the chain rule through the graph.",
            lesson_id="chain_rule",
        ),
        Question(
            prompt="For y = (3x)² , using the chain rule dy/dx at x = 1 is…",
            choices=["6", "9", "18", "3"],
            answer=2,
            explanation="dy/dx = 2·(3x)·3 = 18x, so at x = 1 it is 18.",
            lesson_id="chain_rule",
        ),
    ],
    "backprop": [
        Question(
            prompt="For L = a · b at a = 2, b = -3, what is dL/da?",
            choices=["2", "-3", "-6", "3"],
            answer=1,
            explanation="∂(a·b)/∂a = b = -3.",
            lesson_id="backprop",
        ),
        Question(
            prompt="For L = a · b at a = 2, b = -3, what is dL/db?",
            choices=["2", "-3", "-6", "6"],
            answer=0,
            explanation="∂(a·b)/∂b = a = 2.",
            lesson_id="backprop",
        ),
        Question(
            prompt="At an addition node c = a + b during backprop, the upstream gradient is…",
            choices=[
                "split in half between a and b",
                "copied unchanged to both a and b",
                "multiplied by a and b respectively",
                "set to zero",
            ],
            answer=1,
            explanation="∂(a+b)/∂a = ∂(a+b)/∂b = 1, so the gradient passes straight through.",
            lesson_id="backprop",
        ),
    ],
    # --- Optimization --------------------------------------------------------
    "descent": [
        Question(
            prompt="Gradient descent updates a parameter by stepping in the direction of the…",
            choices=[
                "positive gradient",
                "negative gradient",
                "second derivative",
                "parameter's current value",
            ],
            answer=1,
            explanation="To reduce loss you move opposite the gradient (steepest ascent).",
            lesson_id="descent",
        ),
        Question(
            prompt="If the learning rate is far too large, gradient descent tends to…",
            choices=[
                "converge faster with no downside",
                "overshoot and diverge or oscillate",
                "stop updating entirely",
                "guarantee the global minimum",
            ],
            answer=1,
            explanation="Steps that are too big jump past the minimum and can blow up.",
            lesson_id="descent",
        ),
        Question(
            prompt="Adam improves on plain SGD mainly by…",
            choices=[
                "removing the learning rate",
                "using per-parameter adaptive step sizes from moment estimates",
                "computing exact second derivatives",
                "ignoring the gradient",
            ],
            answer=1,
            explanation="Adam adapts each parameter's step from running estimates of the moments.",
            lesson_id="descent",
        ),
    ],
    # --- Systems & Hardware --------------------------------------------------
    "kv_cache": [
        Question(
            prompt="The transformer KV cache grows linearly with…",
            choices=[
                "the number of model parameters",
                "the sequence length (tokens generated)",
                "the learning rate",
                "the vocabulary size",
            ],
            answer=1,
            explanation="Each new token appends its K and V, so the cache scales with length.",
            lesson_id="kv_cache",
        ),
        Question(
            prompt="What does caching Keys and Values during generation avoid?",
            choices=[
                "storing the model weights",
                "recomputing K and V for all past tokens at every step",
                "using softmax",
                "the need for positional encodings",
            ],
            answer=1,
            explanation="Cached K/V let each new token attend to the past without recomputation.",
            lesson_id="kv_cache",
        ),
        Question(
            prompt="Versus Multi-Head Attention, Multi-Query Attention (1 KV head) makes it…",
            choices=[
                "larger",
                "smaller by the head-reduction factor",
                "exactly the same size",
                "grow quadratically",
            ],
            answer=1,
            explanation="Sharing one K/V across all query heads shrinks the cache proportionally.",
            lesson_id="kv_cache",
        ),
    ],
    "vram": [
        Question(
            prompt="Which of these is NOT one of the main consumers of training VRAM?",
            choices=[
                "model weights",
                "gradients and optimizer states",
                "activations and the KV cache",
                "the training script's source code",
            ],
            answer=3,
            explanation="VRAM is dominated by weights, gradients, optimizer state, activations.",
            lesson_id="vram",
        ),
        Question(
            prompt="In fp16 (2 bytes/param), roughly how much VRAM do a 7B model's weights take?",
            choices=["about 3.5 GB", "about 7 GB", "about 14 GB", "about 28 GB"],
            answer=2,
            explanation="7e9 params × 2 bytes ≈ 14 GB for the weights alone.",
            lesson_id="vram",
        ),
        Question(
            prompt="The Adam optimizer adds VRAM cost mainly because it stores…",
            choices=[
                "a copy of the input data",
                "two extra moment tensors per parameter",
                "the entire training history",
                "nothing extra",
            ],
            answer=1,
            explanation="Adam keeps first- and second-moment estimates: extra per-parameter state.",
            lesson_id="vram",
        ),
    ],
    # --- World Models & Interpretability ------------------------------------
    "jepa": [
        Question(
            prompt="A JEPA (Joint-Embedding Predictive Architecture) makes its predictions in…",
            choices=[
                "raw pixel space",
                "an abstract latent representation space",
                "the token vocabulary",
                "the gradient space",
            ],
            answer=1,
            explanation="JEPA predicts latent representations rather than reconstructing pixels.",
            lesson_id="jepa",
        ),
        Question(
            prompt="Why does LeCun argue for predicting in latent space instead of pixels?",
            choices=[
                "pixels are faster to store",
                "latent prediction ignores unpredictable low-level detail and captures structure",
                "it removes the need for training data",
                "it guarantees zero loss",
            ],
            answer=1,
            explanation="Predicting abstractions avoids wasting capacity on irreducible noise.",
            lesson_id="jepa",
        ),
    ],
    "superposition": [
        Question(
            prompt="Superposition refers to a neural network storing…",
            choices=[
                "exactly one feature per neuron",
                "more features than it has neurons, by overlapping them in directions",
                "only its weights, never features",
                "features only in the final layer",
            ],
            answer=1,
            explanation="With more features than dimensions, features share directions.",
            lesson_id="superposition",
        ),
        Question(
            prompt="A single neuron activating for several unrelated concepts is described as…",
            choices=["monosemantic", "polysemantic", "orthogonal", "quantized"],
            answer=1,
            explanation="Polysemantic neurons respond to many features, a sign of superposition.",
            lesson_id="superposition",
        ),
    ],
    # --- Frontier (v0.8) -----------------------------------------------------
    "lora": [
        Question(
            prompt="LoRA fine-tunes a model by…",
            choices=[
                "updating every weight in the model",
                "freezing W₀ and learning a low-rank update ΔW = B·A",
                "retraining from scratch",
                "deleting layers",
            ],
            answer=1,
            explanation="LoRA keeps the pretrained weight frozen and trains a small low-rank ΔW.",
            lesson_id="lora",
        ),
        Question(
            prompt="In LoRA, B is initialized to zeros so that at the start of training…",
            choices=[
                "the model outputs zero",
                "ΔW = B·A = 0, so the model exactly matches the pretrained one",
                "gradients cannot flow",
                "A is also zero",
            ],
            answer=1,
            explanation="Zero-init B makes ΔW zero at first, so training starts at the base model.",
            lesson_id="lora",
        ),
        Question(
            prompt="What mainly makes LoRA parameter-efficient?",
            choices=[
                "the rank r is far smaller than the layer's dimensions",
                "it uses int4 weights",
                "it removes the optimizer",
                "it skips backpropagation",
            ],
            answer=0,
            explanation="A small rank r means B·A has far fewer trainable parameters than full W.",
            lesson_id="lora",
        ),
    ],
    "quantization": [
        Question(
            prompt="How many bytes does a single fp32 weight occupy?",
            choices=["1", "2", "4", "8"],
            answer=2,
            explanation="An IEEE-754 single-precision float is 32 bits = 4 bytes.",
            lesson_id="quantization",
        ),
        Question(
            prompt="Quantizing weights from fp32 to int8 cuts their memory by roughly…",
            choices=["2×", "4×", "8×", "no change"],
            answer=1,
            explanation="4 bytes → 1 byte per weight is a 4× reduction.",
            lesson_id="quantization",
        ),
        Question(
            prompt="In x̂ = (q − zero_point)·scale, what does 'scale' represent?",
            choices=[
                "the number of integer levels",
                "the size of one integer step in float space",
                "the mean of the weights",
                "the quantization error",
            ],
            answer=1,
            explanation="scale is the float width of one integer step; zero_point places 0.",
            lesson_id="quantization",
        ),
    ],
    "flash_attention": [
        Question(
            prompt="Relative to standard attention, FlashAttention's output is…",
            choices=[
                "a lossy approximation",
                "exact to floating-point precision",
                "always more accurate mathematically",
                "computed without softmax",
            ],
            answer=1,
            explanation="The online-softmax rescaling is algebraically exact, so results match.",
            lesson_id="flash_attention",
        ),
        Question(
            prompt="The key memory win of FlashAttention is that it never…",
            choices=[
                "computes a softmax",
                "materializes the full N×N score matrix in HBM",
                "reads the Query matrix",
                "uses the GPU",
            ],
            answer=1,
            explanation="Tiling + online softmax avoid writing the N×N scores to slow HBM.",
            lesson_id="flash_attention",
        ),
        Question(
            prompt="Standard attention on long sequences is primarily bottlenecked by…",
            choices=[
                "raw compute (FLOPs)",
                "memory traffic — it is memory-bound",
                "the number of parameters",
                "the learning rate",
            ],
            answer=1,
            explanation="Reading/writing the N×N matrix from HBM dominates; it is memory-bound.",
            lesson_id="flash_attention",
        ),
    ],
    "dpo": [
        Question(
            prompt="The main advantage of DPO over classic RLHF/PPO is that it…",
            choices=[
                "needs no preference data",
                "removes the separate reward model and RL loop",
                "does not require a reference model",
                "trains without any loss function",
            ],
            answer=1,
            explanation="DPO collapses reward modeling and RL into one classification-style loss.",
            lesson_id="dpo",
        ),
        Question(
            prompt="DPO trains directly on…",
            choices=[
                "scalar reward scores",
                "human preference pairs (chosen ≻ rejected)",
                "raw pixels",
                "unlabeled text only",
            ],
            answer=1,
            explanation="DPO optimizes over the same (chosen, rejected) pairs RLHF would use.",
            lesson_id="dpo",
        ),
        Question(
            prompt="In DPO, the implicit reward of a response is defined relative to the…",
            choices=[
                "reference (SFT) model's log-probabilities",
                "batch size",
                "vocabulary size",
                "learning rate",
            ],
            answer=0,
            explanation="The implicit reward r = β·(log π − log π_ref) is measured vs the ref.",
            lesson_id="dpo",
        ),
    ],
}
