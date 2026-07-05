# Courses

76 lessons across 20 tracks — every one a runnable, explained trace.  
Run any lesson with `optimumai learn <id>`, track progress, quiz yourself, and
review on a spaced-repetition schedule.

```bash
pip install optimumai
optimumai start      # 30-second guided tour — start here
optimumai course     # full syllabus with ✓/○ completion marks
optimumai progress   # progress bar + percentage + what's next
```

---

## How lessons work

Every lesson runs real code, prints a step-by-step `Trace`, and marks itself
complete:

```bash
optimumai learn attention           # run the lesson (auto-marks complete)
optimumai learn attention --level researcher   # with full depth
optimumai learn attention --no-track           # run without marking complete
```

Four detail levels — same math, progressively more revealed:

| `--level` | What you see |
|---|---|
| `beginner` | steps + plain-English "why AI uses this" |
| `intermediate` | per-step detail notes **(default)** |
| `engineer` | intermediate values + algorithmic complexity |
| `researcher` | everything: formulas, proofs, references |

---

## Track 1 · Linear Algebra

The atoms of all of deep learning. Every matrix multiply in a neural net is a
grid of dot products; every similarity search is a cosine distance.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 1.1 | `dot` | **Dot product** | Similarity and the atom of every matrix multiply. |
| 1.2 | `cosine` | **Cosine similarity** | Angle between vectors — the RAG / semantic-search score. |
| 1.3 | `matmul` | **Matrix multiplication** | Every dense layer, computed cell by cell. |

```bash
optimumai learn dot
optimumai learn cosine
optimumai learn matmul
```

```python
from optimumai import Vector, Matrix

# Expected output ↓
Vector([1, 2, 3]).dot(Vector([4, 5, 6]), explain=True)
# ╭──────────────── OptimumAI ─────────────────╮
# │ DOT  a · b = Σᵢ aᵢ·bᵢ                      │
# ╰─────────────────────────────────────────────╯
# 1  1 × 4 = 4
# 2  2 × 5 = 10
# 3  3 × 6 = 18
# 4  Sum: 4 + 10 + 18 = 32
# Result: 32

Vector([1, 2, 3]).cosine_similarity(Vector([2, 4, 6]), explain=True)
# Result: 1.0  (parallel vectors → perfect similarity)

Matrix([[1, 2], [3, 4]]).matmul(Matrix([[5, 6], [7, 8]]), explain=True)
# Result: [[19, 22], [43, 50]]
```

---

## Track 2 · Calculus & Autograd

The single idea behind all of deep learning: the chain rule. This track
builds from "what is a derivative" to a full reverse-mode autodiff engine.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 2.1 | `derivative` | **Derivatives** | The slope that tells a network which way to move. |
| 2.2 | `gradient` | **Gradients** | The vector of partial derivatives — one per parameter. |
| 2.3 | `chain_rule` | **The chain rule** | The single idea behind all of backpropagation. |
| 2.4 | `backprop` | **Backpropagation** | Reverse-mode autodiff on a scalar graph (micrograd). |

```bash
optimumai learn derivative
optimumai learn gradient
optimumai learn chain_rule
optimumai learn backprop
```

```python
from optimumai import derivative, gradient, Value

derivative(lambda x: x**2, 3.0, explain=True)
# Result: ≈ 6.0  (d/dx x² at x=3 = 2x = 6)

gradient(lambda p: p[0]**2 + p[1]**2, [3.0, 4.0], explain=True)
# Result: [6.0, 8.0]

# Backprop through a scalar graph
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
L = (a * b).tanh()
L.backprop(explain=True)
# a.grad = -0.0099  b.grad = 0.0066
```

---

## Track 3 · Optimization & Neural Nets

Gradient descent + the full training loop. An MLP is just the autograd engine
composed a few thousand times.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 3.1 | `descent` | **Gradient descent** | Adam walking a loss bowl to its minimum. |
| 3.2 | `train` | **Training an MLP** | The full loop: predict → loss → backprop → step. |

```bash
optimumai learn descent
optimumai learn train
optimumai train --steps 150 --lr 0.05
```

```python
from optimumai import MLP
from optimumai.neural_networks import train_demo

mlp = MLP(3, [4, 4, 1], activation="tanh", seed=0)
mlp([2.0, 3.0, -1.0])   # forward pass → scalar output

train_demo(steps=150).render("intermediate")
# Step   0  loss=3.471
# Step  10  loss=1.823
# Step  50  loss=0.412
# Step 100  loss=0.089
# Step 150  loss=0.021  ✓
```

---

## Track 4 · Probability & Transformers

From softmax to the full transformer block — the core of every modern LLM.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 4.1 | `softmax` | **Softmax** | Logits into a probability distribution. |
| 4.2 | `attention` | **Scaled dot-product attention** | softmax(QKᵀ/√dₖ)·V — the transformer core. |
| 4.3 | `multihead` | **Multi-head attention** | Parallel heads + a causal mask (nanoGPT-style). |
| 4.4 | `positional` | **Positional encoding** | Injecting word order into permutation-invariant attention. |
| 4.5 | `transformer` | **The transformer block** | LayerNorm → attention → FFN, with residuals. |

```bash
optimumai learn softmax
optimumai learn attention
optimumai learn multihead
optimumai learn positional
optimumai learn transformer
```

```python
from optimumai import softmax, Attention, MultiHeadAttention, TransformerBlock

softmax([2.0, 1.0, 0.1], temperature=0.5, explain=True)
# Result: [0.844, 0.114, 0.042]  (sharp, T=0.5)

softmax([2.0, 1.0, 0.1], temperature=2.0, explain=True)
# Result: [0.460, 0.335, 0.205]  (flatter, T=2.0)

Attention.demo().render("engineer")
# Q shape: (4, 4)  K shape: (4, 4)  V shape: (4, 4)
# Scores = QKᵀ/√4 ...  softmax → weights → weighted sum of V
# Result shape: (4, 4)

MultiHeadAttention.demo().render("engineer")
TransformerBlock.demo().render("researcher")
```

---

## Track 5 · Applied AI

Embeddings, RAG, and diffusion — the three pillars of modern deployed AI.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 5.1 | `embeddings` | **Embeddings** | Turning discrete tokens into learnable dense vectors. |
| 5.2 | `rag` | **Retrieval-augmented generation** | Query → embed → cosine search → top-k → prompt. |
| 5.3 | `diffusion` | **Diffusion** | Forward noising schedule and the reverse denoising idea. |

```bash
optimumai learn embeddings
optimumai learn rag
optimumai learn diffusion
```

```python
from optimumai import embedding_lookup, nearest_neighbors, RAGPipeline, forward_diffusion
import numpy as np

embedding_lookup(["cat", "dog", "car"], dim=4, explain=True)
# cat → [0.12, -0.34, 0.89, 0.45]  (randomly initialised, shape (3, 4))

nearest_neighbors("cat", vocab=["dog", "car", "kitten"], dim=4, k=2, explain=True)
# 1st: kitten  cos=0.91  2nd: dog  cos=0.73

RAGPipeline().forward("how do neural networks learn?", k=2, explain=True)
# Query embedded → cosine search over 3 docs → top-2 stuffed into prompt

forward_diffusion(np.array([1., 2., 3., 4., 5., 6.]), timesteps=10, explain=True)
# t=0  x=[1, 2, 3, 4, 5, 6]
# t=5  x=[0.62, 1.24, ...]  (noise growing)
# t=10 x≈ε  (mostly noise)
```

---

## Track 6 · World Models & Interpretability

What happens inside the model? JEPA predicts in latent space; superposition
explains why neurons are polysemantic.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 6.1 | `jepa` | **JEPA world models** | LeCun: predict in latent space, not pixels (energy-based). |
| 6.2 | `superposition` | **Superposition** | Anthropic: why single neurons are polysemantic. |

```bash
optimumai learn jepa
optimumai learn superposition
optimumai jepa --demo --level engineer
optimumai superposition --features 5 --neurons 2
```

```python
from optimumai import JEPA, superposition

JEPA.demo().render("engineer")
# E(x, y) = ‖g(f(x)) − f(y)‖²
# Predicted embedding: [0.12, -0.45, 0.67]
# Target embedding:    [0.11, -0.44, 0.66]
# Energy: 0.0003  ← low = good prediction

superposition(n_features=5, n_neurons=2, explain=True)
# WᵀW off-diagonal = interference between features sharing a neuron
# Max interference: 0.34  (feature 2 ↔ feature 4)
```

---

## Track 7 · Math Foundations

The mathematical building blocks: tensors and numerical integration.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 7.1 | `tensors` | **Tensors** | Scalars → vectors → matrices → n-D arrays, shapes, broadcasting. |
| 7.2 | `integration` | **Numerical integration** | Trapezoid & Monte Carlo — every expectation is an integral. |

```bash
optimumai learn tensors
optimumai learn integration
```

```python
from optimumai import integrate

integrate(lambda x: x**2, 0, 1, method="trapezoid", explain=True)
# Sliced into 1000 trapezoids, Δx=0.001
# Result: 0.3333335  (exact: 1/3)

integrate(lambda x: x**2, 0, 1, method="monte_carlo", explain=True)
# Sampled 10,000 points uniformly on [0, 1]
# Result: 0.3331  (stochastic, seeded for reproducibility)
```

---

## Track 8 · Framework Internals

What torch.autograd and JAX's `grad`/`jit`/`vmap` are actually doing.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 8.1 | `pytorch` | **PyTorch autograd** | What torch.autograd does under the hood — it's the Value engine. |
| 8.2 | `jax` | **JAX transforms** | grad / jit / vmap / pytrees — transforming pure functions. |

```bash
optimumai learn pytorch
optimumai learn jax
optimumai tutorial pytorch   # full interactive tutorial
optimumai tutorial numpy     # NumPy → PyTorch bridge
```

---

## Track 9 · Systems & Hardware

Why LLMs need hundreds of GBs of VRAM and how the GPU actually works.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 9.1 | `gpu_threads` | **GPU thread hierarchy** | Grid → block → warp → thread; the SIMT execution model. |
| 9.2 | `memory_hierarchy` | **GPU memory hierarchy** | Registers → shared → global; why kernels are memory-bound. |
| 9.3 | `cuda_matmul` | **Tiled matmul kernel** | Naive vs tiled + coalescing — the canonical GPU optimization. |
| 9.4 | `kv_cache` | **The KV cache** | Why context length eats VRAM; MHA vs GQA vs MQA. |
| 9.5 | `vram` | **VRAM budget** | Weights + gradients + optimizer states + activations + KV cache. |

```bash
optimumai learn gpu_threads
optimumai learn memory_hierarchy
optimumai learn cuda_matmul
optimumai learn kv_cache
optimumai learn vram

# Calculators
optimumai kvcache --seq-len 8192
optimumai kvcache --heads 32 --kv-heads 4    # GQA: llama-3 style
optimumai vram --params 70                   # 70B model training
optimumai vram --params 7 --inference        # 7B inference
```

```python
from optimumai import kv_cache_size, vram_estimate

kv_cache_size(n_layers=32, n_heads=32, head_dim=128, seq_len=4096)
# KV cache: 2 × 32 × 32 × 128 × 4096 × 2 bytes = 2.1 GB

vram_estimate(params_billions=7, training=True)
# Weights:           14 GB  (fp16)
# Gradients:         14 GB
# Adam states:       28 GB
# Activations:       ~8 GB  (estimated)
# Total:            ~64 GB
```

---

## Track 10 · Interactive Playground

Feed your own input and watch it flow through the model.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 10.1 | `trace_text` | **Text → transformer** | Watch your own words flow to a next-token distribution. |
| 10.2 | `compare` | **Compare activations** | ReLU vs GELU side by side — how activation shapes gradients. |
| 10.3 | `sweep` | **Temperature sweep** | How softmax sharpens (T→0) or flattens (T→∞). |

```bash
optimumai learn trace_text
optimumai learn compare
optimumai learn sweep

optimumai trace-text "attention is all you need" --layers 2 --level engineer
optimumai compare relu gelu --input "[-2,-1,0,1,2]"
optimumai sweep softmax --values "[0.1,0.5,1.0,2.0,5.0]"
```

---

## Track 11 · Frontier

How today's large models are actually built and run cheaply.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 11.1 | `flash_attention` | **FlashAttention** | IO-aware tiling + online softmax — exact attention, linear memory. |
| 11.2 | `quantization` | **Quantization** | int8/int4 with scale + zero-point — how models shrink to fit. |
| 11.3 | `lora` | **LoRA** | Low-rank adapters (W₀ + BA) — fine-tune ~10,000× fewer parameters. |
| 11.4 | `dpo` | **DPO / RLHF** | Align to human preferences without a reward model or RL. |

```bash
optimumai learn flash_attention
optimumai learn quantization
optimumai learn lora
optimumai learn dpo

optimumai quantize "[0.1,-2.3,4.5,3.14]" --bits 4
optimumai kernel flash_attention
```

```python
from optimumai.frontier import flash_attention, quantize
import numpy as np

Q, K, V = (np.random.default_rng(0).normal(size=(8, 4)) for _ in range(3))
result = flash_attention(Q, K, V, block_size=2, explain=True)
# Tiling Q into 4 blocks of size 2 ...
# Online softmax rescaled across 4 K/V tiles
# Max error vs naive attention: 1.8e-16  ✓ exact

quantize(np.array([0.1, -2.3, 4.5, 3.14]), bits=8, explain=True)
# scale=0.0275, zero_point=0
# [4, -84, 164, 114]  → dequantize → [0.110, -2.310, 4.510, 3.135]
```

---

## Track 12 · GPU Kernels

Write per-thread kernels, run on the simulator, grade yourself.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 12.1 | `kernel_vector_add` | **Kernel: vector add** | One thread per element — the GPU "hello world", run on a simulator. |
| 12.2 | `kernel_matmul` | **Kernel: tiled matmul** | A thread per output cell + the shared-memory tiling win. |
| 12.3 | `kernel_softmax` | **Kernel: softmax** | One thread per row, with the numerically-stable max trick. |
| 12.4 | `kernel_flash` | **Kernel: flash attention** | Fused online-softmax attention — exact, no N×N matrix in VRAM. |

```bash
optimumai learn kernel_vector_add
optimumai learn kernel_matmul
optimumai learn kernel_softmax
optimumai learn kernel_flash

optimumai kernel vector_add     # step through the simulator
optimumai kernel matmul
optimumai kernel flash_attention
```

```python
from optimumai.kernels import KernelWorkbench

wb = KernelWorkbench()
print(wb.get_challenge("vector_add").prompt)
# Write a CUDA-style kernel that adds two vectors element-wise.
# Each thread handles one output element.

def my_kernel(ctx, inp, out):
    i = ctx.idx.global_id
    if i < out.size:
        out[i] = ctx.gload(inp["a"], i) + ctx.gload(inp["b"], i)

result = wb.submit("vector_add", my_kernel)
print(result.feedback)
# ✓ Correct — output matches reference for 10/10 random seeds.
# IO efficiency: 2.0 global loads + 1.0 global stores per element (optimal).
```

---

## Track 13 · Classical ML

The models every ML practitioner needs to understand — they're still baselines,
building blocks, and diagnostics in 2025.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 13.1 | `linear_regression` | **Linear regression** | OLS via the normal equation θ=(XᵀX)⁻¹Xᵀy — the calculus behind the fit. |
| 13.2 | `logistic_regression` | **Logistic regression** | Sigmoid + cross-entropy + gradient descent — the atomic classifier. |
| 13.3 | `kmeans` | **k-means clustering** | Lloyd's algorithm: assign to the nearest centroid, re-center, repeat. |
| 13.4 | `knn` | **k-nearest neighbors** | Classify by majority vote of the closest points — no training phase. |
| 13.5 | `decision_tree` | **Decision trees** | Best split by Gini/entropy information gain — the atom of random forests. |
| 13.6 | `naive_bayes` | **Gaussian Naive Bayes** | Bayes' rule + a conditional-independence assumption. |
| 13.7 | `pca` | **Principal component analysis** | Covariance eigendecomposition — compress data to its top-variance axes. |
| 13.8 | `ml_metrics` | **ML metrics** | Accuracy, precision/recall/F1, confusion matrix, MSE, R², ROC-AUC. |

```bash
optimumai learn linear_regression
optimumai learn logistic_regression
optimumai learn kmeans
optimumai learn knn
optimumai learn decision_tree
optimumai learn naive_bayes
optimumai learn pca
optimumai learn ml_metrics

optimumai ml linreg "[[1],[2],[3],[4]]" "[2,4,6,8]"
optimumai ml kmeans "[[0,0],[0,1],[9,9],[9,8]]" --k 2
optimumai ml metrics
```

```python
from optimumai.ml import LinearRegression, KMeans, PCA

model = LinearRegression()
model.fit([[1],[2],[3],[4]], [2,4,6,8], explain=True)
# Normal equation: θ = (XᵀX)⁻¹Xᵀy
# θ₀ (intercept) = 0.0   θ₁ (slope) = 2.0
model.predict([[5]])   # [10.0]

km = KMeans(k=2)
km.fit([[0,0],[0,1],[9,9],[9,8]], explain=True)
# Iteration 1: centroids → [0.0, 0.5] and [9.0, 8.5]
# Converged in 2 iterations
```

---

## Track 14 · Classical AI Search

The common ancestor of pathfinding, puzzle solvers, and game-playing AI.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 14.1 | `uninformed_search` | **BFS, DFS & uniform-cost search** | Frontier order is everything: FIFO (BFS), LIFO (DFS), cheapest-g (UCS). |
| 14.2 | `informed_search` | **Greedy best-first & A\*** | Add a heuristic h(n): A* orders by f=g+h and is optimal when h is admissible. |
| 14.3 | `adversarial_search` | **Minimax & alpha-beta pruning** | Optimal play on a game tree; α-β prunes provably-irrelevant branches. |

```bash
optimumai learn uninformed_search
optimumai learn informed_search
optimumai learn adversarial_search

optimumai algo bfs
optimumai algo astar
optimumai algo minimax
```

```python
from optimumai.search import bfs, ucs, astar
from optimumai.search.problem import Graph

g = Graph()
g.add_edge("A","B",1); g.add_edge("B","C",1)
g.add_edge("C","D",1); g.add_edge("B","D",5)

bfs(g, "A", "D", explain=True)
# Frontier: [A] → [B] → [C, D] → found D
# Path: A → B → D  (fewest edges: 2)

ucs(g, "A", "D", explain=True)
# Expanded: A(0) → B(1) → C(2) → D(3)
# Path: A → B → C → D  (cheapest cost: 3)
```

---

## Track 15 · Reinforcement Learning

From the Bellman equation to PPO — the same algorithm used in RLHF.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 15.1 | `value_iteration` | **Value iteration (Bellman)** | Solve a known MDP exactly via Bellman backups until the values converge. |
| 15.2 | `q_learning` | **Q-learning & SARSA** | Learn to act from experience alone — off-policy vs on-policy TD updates. |
| 15.3 | `reinforce` | **Policy gradients (REINFORCE)** | Differentiate through sampled actions with the score-function trick. |
| 15.4 | `ppo` | **PPO — clipped surrogate objective** | The stabilized policy gradient behind RLHF's RL stage. |

```bash
optimumai learn value_iteration
optimumai learn q_learning
optimumai learn reinforce
optimumai learn ppo

optimumai rl mdp
optimumai rl q-learning
optimumai rl reinforce
optimumai rl ppo
```

---

## Track 16 · NLP

The fundamental algorithms that turn text into numbers.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 16.1 | `bpe` | **Byte-pair encoding** | Learn merges from a corpus, then tokenize a word the way an LLM does. |
| 16.2 | `tfidf` | **TF-IDF** | Weight words by how much they distinguish one document from the rest. |
| 16.3 | `ngram` | **N-gram language models** | Count your way to "what comes next," with add-k smoothing + perplexity. |
| 16.4 | `edit_distance` | **Edit distance** | Levenshtein DP — the yardstick behind spell-check and fuzzy match. |
| 16.5 | `word2vec` | **Skip-gram word2vec** | Predict-the-context SGD — the seed idea behind every learned embedding. |

```bash
optimumai learn bpe
optimumai learn tfidf
optimumai learn ngram
optimumai learn edit_distance
optimumai learn word2vec

optimumai nlp bpe lowest
optimumai nlp tfidf "the cat sat" "the dog sat"
optimumai nlp edit-distance kitten sitting
```

```python
from optimumai.nlp import BPETokenizer, edit_distance

tok = BPETokenizer(num_merges=8)
tok.train(["low", "lower", "lowest", "newer", "newest"])
tok.encode("lowest", explain=True)
# Merges applied: lo+w→low, est+</w>→est</w>, w+est</w>→west</w>
# Result: ['lo', 'west</w>']

edit_distance("kitten", "sitting", explain=True)
# DP matrix filled (7×8) ...
# Operations: substitute k→s, substitute e→i, insert g
# Distance: 3
```

---

## Track 17 · Computer Vision

How a CNN sees — convolutions, pooling, and the tiny-CNN forward pass.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 17.1 | `conv2d` | **2D convolution** | Slide a kernel over an image to detect local patterns. |
| 17.2 | `pooling` | **Max & average pooling** | Downsample a feature map for translation tolerance. |
| 17.3 | `sobel_edges` | **Sobel edge detection** | Two fixed convolutions that find image boundaries. |
| 17.4 | `tiny_cnn` | **A tiny CNN forward pass** | Watch the tensor shapes flow: conv → relu → pool → dense → softmax. |

```bash
optimumai learn conv2d
optimumai learn pooling
optimumai learn sobel_edges
optimumai learn tiny_cnn

optimumai vision conv
optimumai vision pool
optimumai vision sobel
optimumai vision cnn --level engineer
```

```python
from optimumai.vision.convolution import conv2d_trace
import numpy as np

image  = np.arange(36).reshape(6, 6).astype(float)
kernel = np.array([[1, 0], [0, -1]], dtype=float)

conv2d_trace(image, kernel, stride=1).render("beginner")
# Input  shape: (6, 6)   Kernel shape: (2, 2)   Stride: 1
# Output shape: (5, 5)   (no padding)
# Output[0,0] = 1×0 + 0×1 + 0×6 + (-1)×7 = -7
```

---

## Track 18 · LLM Evaluation

How do you know if your LLM is any good?

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 18.1 | `bleu_rouge` | **Text overlap metrics (BLEU / ROUGE / F1)** | Score generated text against a reference — and see why paraphrase breaks them. |
| 18.2 | `perplexity` | **Perplexity** | Cross-entropy of a token sequence, exponentiated into a branching factor. |
| 18.3 | `calibration` | **Calibration (ECE)** | Bin predictions by confidence and check it matches empirical accuracy. |
| 18.4 | `hallucination` | **Hallucination / faithfulness heuristic** | A claim-overlap grounding proxy — plus why real detection is unsolved. |

```bash
optimumai learn bleu_rouge
optimumai learn perplexity
optimumai learn calibration
optimumai learn hallucination

optimumai eval bleu "the quick brown fox jumps" "the quick brown fox leaps" --max-n 1
optimumai eval perplexity "[0.5,0.25,0.8]"
optimumai eval calibration
optimumai eval faithfulness
```

```python
from optimumai.evaluation import bleu, perplexity

bleu("the quick brown fox jumps", "the quick brown fox leaps", max_n=1, explain=True)
# 1-gram precision: 4/5 = 0.80  BP=1.0
# BLEU-1: 0.80

perplexity([0.5, 0.25, 0.8], explain=True)
# token probs: [0.5, 0.25, 0.8]
# log-probs:   [-0.693, -1.386, -0.223]
# mean neg log-prob: 0.767
# Perplexity = exp(0.767) = 2.15
```

---

## Track 19 · Prompt Engineering

Standard patterns for getting more out of a frozen LLM by changing what you
send it — not the model itself.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 19.1 | `zero_shot` | **Zero-shot prompting** | Role + instruction + task, with no worked examples. |
| 19.2 | `few_shot` | **Few-shot prompting** | In-context learning from K labeled exemplars. |
| 19.3 | `chain_of_thought` | **Chain-of-thought prompting** | Elicit intermediate reasoning before the final answer. |
| 19.4 | `react` | **ReAct prompting** | Interleave Thought / Action / Observation with a tool. |
| 19.5 | `self_consistency` | **Self-consistency** | Sample N chain-of-thought paths and majority-vote the answer. |
| 19.6 | `structured_output` | **Structured output** | Constrain generation to a validated JSON schema. |

```bash
optimumai learn zero_shot
optimumai learn few_shot
optimumai learn chain_of_thought
optimumai learn react
optimumai learn self_consistency
optimumai learn structured_output

optimumai prompt chain-of-thought
optimumai prompt react
```

```python
from optimumai.prompting import chain_of_thought, self_consistency

chain_of_thought(
    "If a train travels 60 miles in 2 hours, what is its speed?",
    explain=True
)
# Assembled prompt:
# [System] You are a careful reasoning assistant.
# [User]   Think step by step, then give your final answer.
#          Q: If a train travels 60 miles in 2 hours, what is its speed?
# Chain-of-thought elicits: "60 miles / 2 hours = 30 mph"

self_consistency("What is 2+2?", sampled_answers=["4","4","5"], explain=True)
# Votes: {"4": 2, "5": 1}
# Majority answer: "4"
```

---

## Track 20 · Augmented RNNs

The distill.pub lineage from 2016: the pre-transformer ideas that made
attention mainstream. NTM's read head is the direct ancestor of transformer
attention.

| # | ID | Lesson | What you learn |
|---|---|---|---|
| 20.1 | `augmented_attention` | **Attention as differentiable memory** | Content-based soft attention as a memory read: score → softmax → blend. |
| 20.2 | `ntm` | **Neural Turing Machines** | External read/write memory addressed by cosine similarity — erase then add. |
| 20.3 | `act` | **Adaptive Computation Time** | Learned, variable compute per input via a halting probability + ponder cost. |

```bash
optimumai learn augmented_attention
optimumai learn ntm
optimumai learn act

optimumai augrnn attention
optimumai augrnn ntm
optimumai augrnn act
```

```python
from optimumai.augmented_rnns import attention_read, ntm_read, adaptive_computation_time
import numpy as np

memory = np.array([[1.,0.,-1.],[0.,1.,0.],[-1.,0.,1.],[.5,.5,.5]])
query  = np.array([1., 0., -1.])

attention_read(query, memory, explain=True)
# cos similarities: [1.00, 0.00, -1.00, 0.00]
# softmax weights:  [0.576, 0.212, 0.000, 0.212]
# Read vector:      [0.682, 0.106, -0.576]

adaptive_computation_time(np.array([0.5, 1.2, 2.0, -0.3, 3.0]), eps=0.01)
# Step 1: halting_prob=0.62  cumulative=0.62
# Step 2: halting_prob=0.77  cumulative=1.00  → halt
# Ponder cost: 1.38
```

---

## Progress tracking

```bash
optimumai progress            # bar chart + percentage done + what's next
optimumai progress --reset    # clear all progress
```

Progress is stored at `~/.optimumai/progress.json`.
Override with `OPTIMUMAI_PROGRESS_PATH`.

```python
from optimumai import COURSE, ProgressTracker

tracker = ProgressTracker()
tracker.mark_complete("attention")
print(tracker.summary())
# {'completed': 1, 'total': 76, 'pct': 1.3}

# Iterate every lesson
for lesson in COURSE:
    print(f"{lesson.track:40s}  {lesson.id:25s}  {lesson.title}")
```

---

## Active recall — quiz

Studying → passive. Testing yourself → active recall, ~2× better retention.

```bash
optimumai quiz                   # list all 20 quiz topics (57 questions total)
optimumai quiz softmax           # answer, get graded + explained
optimumai quiz attention
optimumai quiz backprop
optimumai quiz transformer
```

```python
from optimumai import Quiz

q = Quiz("softmax")
q.ask()
# Q: What does the temperature parameter T control in softmax?
# Your answer: › sharpness
# ✓ Correct. T < 1 sharpens toward a one-hot; T > 1 flattens toward uniform.
```

---

## Spaced repetition — review

Quiz scores automatically feed an SM-2 scheduler.

```bash
optimumai review    # whatever's due today (SM-2 schedule)
```

---

## Compute-the-answer exercises

Numerical fill-in-the-blank, tolerance-graded.

```bash
optimumai exercise              # list exercise topics
optimumai exercise backprop     # compute the gradient, enter a number
optimumai exercise attention
optimumai exercise softmax
```

---

## Streamlit dashboard

```bash
pip install "optimumai[dashboard]"
optimumai dashboard             # localhost:8501
optimumai dashboard --port 8888
```

Shows per-track completion, overall percentage, quiz history, and SM-2 review
schedule. Deploy publicly via [Streamlit Community Cloud or HF Spaces](deploy.md).

---

## LLM tutor (optional)

```bash
pip install "optimumai[llm]"
export ANTHROPIC_API_KEY="..."
optimumai ask "why LayerNorm before attention, not after?"
optimumai ask "explain the difference between RoPE and sinusoidal positional encoding"
```

Degrades gracefully without the key — prints a friendly message and reminds
you that the entire 76-lesson course works fully offline.
