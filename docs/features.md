# Features

Every capability in OptimumAI follows one pattern: **the same code computes
the real answer and can explain itself.** Call anything with `explain=True`
(or render a `*_trace(...)` result) and you get a step-by-step `Trace` — the
exact arithmetic, the formula, and *why AI actually uses this* — at a detail
level you choose with `level=` (`beginner → intermediate → engineer →
researcher`).

!!! note "The shared mechanic"
    Almost everything here is one of two shapes:

    ```python
    Thing(...).op(args, explain=True, level="engineer")   # prints + returns result
    op_trace(args).render("engineer")                      # build Trace, then render
    ```

    `explain=False` (the Python default) skips printing and returns just the
    numeric result — the *same* computation path, so the trace can never lie
    about what ran.

---

## Foundations & math

### Vectors & matrices — `optimumai.algebra`

The dot product is "how much do two things point the same way, scaled by their
sizes" — the single operation underneath cosine similarity, attention scores,
and every matrix multiply.

**Formulas.** `a · b = Σᵢ aᵢbᵢ`, `‖a‖ = √(Σᵢ aᵢ²)`,
`cos θ = (a·b)/(‖a‖‖b‖)`, `C[i,j] = Σₖ A[i,k]·B[k,j]`

```python
from optimumai import Vector, Matrix

Vector([1, 2, 3]).dot(Vector([4, 5, 6]), explain=True)                    # 32.0
Vector([1, 2, 3]).cosine_similarity(Vector([2, 4, 6]), explain=True)      # 1.0
Matrix([[1, 2], [3, 4]]).matmul(Matrix([[5, 6], [7, 8]]), explain=True)
```

```bash
optimumai algebra dot "[1,2,3]" "[4,5,6]"
optimumai algebra dot -i                        # type vectors at a prompt
optimumai algebra cosine "[1,2,3]" "[2,4,6]"
optimumai algebra matmul "[[1,2],[3,4]]" "[[5,6],[7,8]]"
```

### Softmax — `optimumai.probability`

Turns arbitrary scores ("logits") into a probability distribution. Temperature
controls sharpness: low temperature nearly one-hots the max; high temperature
flattens toward uniform.

**Formula.** `softmax(xᵢ) = e^(xᵢ/T) / Σⱼ e^(xⱼ/T)` (max-subtraction trick for numerical stability)

```python
from optimumai import softmax
softmax([2.0, 1.0, 0.1], temperature=0.5, explain=True)
```

```bash
optimumai softmax "[2,1,0.1]" --temperature 0.5
optimumai softmax -i                            # type logits at a prompt
```

### Calculus — `optimumai.calculus`

A derivative is a slope. Central-difference estimation nudges the input and
watches the output move — the same check every framework's gradient checker is
built on.

**Formula.** `f'(x) ≈ (f(x+h) − f(x−h)) / 2h`

```python
from optimumai import derivative, gradient

derivative(lambda x: x**2, 3.0, explain=True)                          # ≈ 6.0
gradient(lambda p: p[0]**2 + p[1]**2, [3.0, 4.0], explain=True)       # ≈ [6.0, 8.0]
```

```bash
optimumai learn derivative
optimumai learn chain_rule
optimumai diff "x**3 + 2*x" --at 3             # symbolic derivative (needs [symbolic])
```

### Numerical integration — `optimumai.foundations`

An expectation *is* an integral. Two ways to approximate it: trapezoid rule
(slice into trapezoids) or Monte Carlo (average random samples).

**Formulas.** Trapezoid: `Σ (f(xᵢ)+f(xᵢ₊₁))/2 · Δx`;
Monte Carlo: `(b−a) · mean(f(X))`

```python
from optimumai import integrate

integrate(lambda x: x**2, 0, 1, method="trapezoid")       # ≈ 0.333
integrate(lambda x: x**2, 0, 1, method="monte_carlo")     # ≈ 0.33 (seeded)
```

```bash
optimumai learn integration
```

---

## Autograd & neural nets

### Value — scalar autograd engine — `optimumai.autograd`

Every neural net trains by the chain rule. `Value` is a scalar that remembers
how it was built, so calling `.backward()` fills in every gradient
automatically. This is exactly what PyTorch's `requires_grad=True` does —
just for one number at a time (inspired by Karpathy's micrograd).

**Formula.** Reverse-mode chain rule: `∂L/∂x = Σ (∂L/∂out)·(∂out/∂x)`

```python
from optimumai import Value

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
L = (a * b).tanh()
L.backprop(explain=True)      # chain rule flows backwards; prints a.grad, b.grad
```

```bash
optimumai backprop
```

### Optimizers — `optimumai.optimization`

SGD steps opposite the gradient. Adam adds momentum and adaptive scaling so it
takes confident steps in flat directions and cautious steps where the surface
is noisy.

**Formulas.** SGD: `w ← w − lr·∂L/∂w`; Adam: `w ← w − lr·m̂/(√v̂ + ε)`

```python
from optimumai import Value, Adam, minimize

x, y = Value(0.0), Value(0.0)
minimize(
    lambda: (x - 3.0)**2 + (y + 1.0)**2,
    [x, y], Adam([x, y], lr=0.3),
    steps=30, explain=True
)   # converges toward x≈3, y≈-1
```

```bash
optimumai train --steps 150 --lr 0.05
optimumai sweep softmax --values "[0.25,0.5,1,2]"
```

### Neural networks — `optimumai.neural_networks`

A neuron is `activation(weighted sum of inputs + bias)`. An MLP stacks layers.
Training is just calling `minimize` on the network's parameters.

**Formula.** `output = φ(Σᵢ wᵢxᵢ + b)` per neuron, with φ ∈ {tanh, relu, linear}

```python
from optimumai import MLP
from optimumai.neural_networks import train_demo

mlp = MLP(3, [4, 4, 1], activation="tanh", seed=0)
mlp([2.0, 3.0, -1.0])                         # forward pass
train_demo(steps=150).render("intermediate")   # loss falls to ~0
```

```bash
optimumai train --steps 150 --lr 0.05
```

---

## Transformers & attention

### Scaled dot-product attention — `optimumai.transformers`

Attention answers "for this query, how much should I weight each value, based
on how well its key matches?" Scaling by √dₖ stops dot products from growing
so large that softmax saturates.

**Formula.** `Attention(Q,K,V) = softmax(QKᵀ/√dₖ)·V`

```python
from optimumai import Attention
Attention.demo().render("engineer")
```

```bash
optimumai attention --demo --level engineer
optimumai attention --demo --seed 1 --level researcher
```

### Multi-head attention & transformer block — `optimumai.transformers`

Multi-head attention runs several smaller attention operations in parallel
subspaces. A causal mask hides future tokens, making the decoder autoregressive.

**Formula.** Per head: `softmax(XₕXₕᵀ/√d_head)·Xₕ`, concatenated across heads.
`TransformerBlock`: `x += MHA(LN(x))`, then `x += FFN(LN(x))`.

```python
from optimumai import MultiHeadAttention, TransformerBlock

MultiHeadAttention.demo().render("engineer")
TransformerBlock.demo().render("researcher")
```

```bash
optimumai attention --demo --level researcher
```

### Positional encoding — `optimumai.transformers`

Injects position by adding a unique wave pattern to each position's embedding.

**Formula.** `PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(...)`

```python
from optimumai.transformers.positional import positional_encoding
positional_encoding(seq_len=4, d_model=8, explain=True)
```

```bash
optimumai learn positional
```

### TextPipeline — full mini-LLM forward pass — `optimumai.transformers`

Tokenize → embed → add positional encoding → transformer blocks → next-token
distribution. The same pipeline as GPT, in miniature.

```python
from optimumai import TextPipeline
TextPipeline("attention is all you need", layers=2).forward(explain=True)
```

```bash
optimumai trace-text "why is the sky blue" --layers 2 --level engineer
optimumai trace-text "hello world" --layers 3 --level researcher
```

---

## World models & interpretability

### JEPA — `optimumai.world_models`

LeCun's argument: predict in *representation space* instead of pixel space.
JEPA is energy-based — energy is low when a predicted embedding matches the
target's embedding.

**Formula.** `E(x, y) = ‖g(f(x)) − f(y)‖²`

```python
from optimumai import JEPA
JEPA.demo().render("engineer")
```

```bash
optimumai jepa --demo --level engineer
```

### Superposition — `optimumai.interpretability`

Anthropic's toy model of polysemanticity: if you have more features than
neurons and features are sparse, the model can pack them into overlapping
directions and mostly get away with it.

**Formula.** Encode `x = W·h`, decode `ĥ = Wᵀ·x`, L2-normalized columns.
Off-diagonal terms of `WᵀW` measure interference.

```python
from optimumai import superposition
superposition(n_features=5, n_neurons=2, explain=True)
```

```bash
optimumai superposition --features 5 --neurons 2
optimumai superposition --features 8 --neurons 3
```

---

## Frontier — FlashAttention, quantization, LoRA, DPO

### FlashAttention — `optimumai.frontier`

IO-aware tiled attention. Never materializes the full N×N matrix. Rescales an
online softmax as it tiles through Q/K/V. Algebraically identical to standard
attention — verified error ~1e-16.

```python
from optimumai.frontier import flash_attention
import numpy as np

Q, K, V = (np.random.default_rng(0).normal(size=(6, 4)) for _ in range(3))
flash_attention(Q, K, V, block_size=2, explain=True)
```

```bash
optimumai learn flash_attention
optimumai kernel flash_attention
```

### Quantization — `optimumai.frontier`

Store weights in fewer bits for cheaper inference.

**Formula.** `q = round(x/scale) + zero_point`

```bash
optimumai quantize "[0.1,-2.3,4.5,3.14]" --bits 4
optimumai learn quantization
```

### LoRA — `optimumai.frontier`

Freeze the base model; train a tiny low-rank update instead of the full weight
matrix.

**Formula.** `W = W₀ + BA`, with `B` initialized to zero, rank `r ≪ d`.

```bash
optimumai learn lora
```

### DPO — `optimumai.frontier`

Align a model to human preferences from preference pairs — no separate reward
model, no RL rollout.

**Formula.** `L = −log σ(β · [(log π(chosen) − log π_ref(chosen)) − (log π(rejected) − log π_ref(rejected))])`

```bash
optimumai learn dpo
```

---

## Applied AI — embeddings, RAG, diffusion

### Embeddings — `optimumai.embeddings`

A lookup table mapping discrete tokens to dense vectors. Nearest-neighbor
cosine search finds "similar" tokens.

```python
from optimumai import embedding_lookup, nearest_neighbors

embedding_lookup(["cat", "dog", "car"], dim=4, explain=True)
nearest_neighbors("cat", vocab=["dog", "car", "kitten"], dim=4, k=2, explain=True)
```

```bash
optimumai learn embeddings
optimumai plot embeddings --out emb.png
```

### Retrieval-augmented generation — `optimumai.rag`

Embed the query, cosine-search a document store, stuff top-k matches into the
prompt as context, then answer from *that* context.

**Formula.** `prompt = stuff(top-k by cos(E[query], E[docᵢ])) ⊕ query`

```python
from optimumai import RAGPipeline
RAGPipeline().forward("how do neural networks learn?", k=2, explain=True)
```

```bash
optimumai learn rag
```

### Diffusion — `optimumai.diffusion`

Learns to reverse a noising process. The forward process has a closed form so
you can jump straight to noise level `t`.

**Formula.** `xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε`, with `ᾱₜ = ∏ₛ(1−βₛ)`

```python
from optimumai import forward_diffusion
import numpy as np
forward_diffusion(np.array([1.0, 2, 3, 4, 5, 6]), timesteps=10, explain=True)
```

```bash
optimumai learn diffusion
optimumai animate diffusion --out diffusion.gif
```

---

## Classical ML — `optimumai.ml`

Pure-NumPy explainable implementations of the models every practitioner needs.

| Model | Formula / idea |
|---|---|
| `LinearRegression` | OLS: `θ = (XᵀX)⁻¹Xᵀy` |
| `LogisticRegression` | `ŷ = σ(Xθ)`, cross-entropy loss, gradient descent |
| `KMeans` | Lloyd's: assign to nearest centroid, recompute, repeat |
| `KNN` | Majority vote among `k` nearest Euclidean neighbors |
| `DecisionTree` | Greedy split maximizing Gini/entropy information gain |
| `GaussianNB` | Bayes' rule with per-feature Gaussian likelihood |
| `PCA` | Eigendecomposition of the covariance matrix |
| `metrics` | `accuracy`, `precision_recall_f1`, `confusion_matrix`, `mse`, `r2_score`, `roc_auc` |

```python
from optimumai.ml import LinearRegression, KMeans

LinearRegression().fit([[1], [2], [3], [4]], [2, 4, 6, 8])
KMeans(k=2).fit([[0, 0], [0, 1], [9, 9], [9, 8]])
```

```bash
optimumai ml linreg "[[1],[2],[3],[4]]" "[2,4,6,8]"
optimumai ml kmeans "[[0,0],[0,1],[9,9],[9,8]]" --k 2
optimumai ml logreg
optimumai ml knn
optimumai ml tree
optimumai ml nb
optimumai ml pca
optimumai ml metrics
```

---

## Classical AI search — `optimumai.search`

Explores a state space to find a goal. Uninformed (BFS/DFS/UCS) vs. informed
(greedy, A*) vs. adversarial (minimax).

**Formula.** A* minimizes `f(n) = g(n) + h(n)` (cost so far + estimated to
goal). Alpha-beta prunes branches where `alpha ≥ beta`.

```python
from optimumai.search import bfs, astar, alpha_beta
from optimumai.search.problem import Graph

g = Graph()
g.add_edge("A", "B", 1); g.add_edge("B", "C", 1)
g.add_edge("C", "D", 1); g.add_edge("B", "D", 5)
bfs(g, "A", "D")   # fewest edges: ['A', 'B', 'D']
```

```bash
optimumai algo bfs
optimumai algo astar
optimumai algo minimax
```

!!! note "`algo` vs `search`"
    `optimumai algo` is classical-AI-search (BFS/A*/minimax).
    `optimumai search <query>` is full-text search over the course.

---

## Reinforcement learning — `optimumai.rl`

MDPs formalize "agent acting in a world with delayed reward." Value iteration
solves a known MDP exactly; Q-learning is model-free; PPO clips the surrogate
objective for stable training.

**Formulas.** Bellman: `V(s) ← maxₐ Σₛ' P(s'|s,a)[R + γV(s')]`;
PPO clip: `L = E[min(r·A, clip(r, 1−ε, 1+ε)·A)]`

```python
from optimumai.rl import value_iteration, q_learning, ppo_clip
```

```bash
optimumai rl mdp
optimumai rl q-learning
optimumai rl reinforce
optimumai rl ppo
```

---

## NLP — `optimumai.nlp`

The fundamental algorithms that turn text into numbers.

| Operation | What it does |
|---|---|
| BPE tokenizer | Merges most-frequent adjacent pairs — how GPT/LLaMA tokenizers are built |
| TF-IDF | `tf · log(N/df)` — distinctiveness score |
| N-gram LM | Predict next token from last `n−1`; perplexity = `exp(-(1/N)Σ log P(wᵢ))` |
| Edit distance | Levenshtein DP, O(mn) — backbone of spell-checkers and WER/CER |
| Skip-gram word2vec | Embeddings from local context via negative-sampling SGD |

```python
from optimumai.nlp import BPETokenizer, edit_distance

tok = BPETokenizer(num_merges=8)
tok.train(["low", "lower", "lowest", "newer", "newest"])
tok.encode("lowest")                   # -> ['lo', 'west</w>']

edit_distance("kitten", "sitting")     # -> 3
```

```bash
optimumai nlp bpe lowest
optimumai nlp bpe --merges 12 lowest
optimumai nlp tfidf "the cat sat" "the dog sat"
optimumai nlp ngram
optimumai nlp edit-distance kitten sitting
optimumai nlp word2vec
```

---

## Computer vision — `optimumai.vision`

A convolution slides a small filter over an image, detecting local patterns
regardless of position. Stack conv → nonlinearity → pool to build a CNN.

**Formula.** Output size: `⌊(W−K+2P)/S⌋ + 1`. Sobel: `magnitude = √(Gₓ² + Gᵧ²)`.

```python
from optimumai.vision.convolution import conv2d_trace
import numpy as np

image = np.arange(36).reshape(6, 6).astype(float)
kernel = np.array([[1, 0], [0, -1]], dtype=float)
conv2d_trace(image, kernel).render("beginner")    # output shape (5, 5)
```

```bash
optimumai vision conv
optimumai vision conv "[[1,2],[3,4]]" "[[1,0],[0,-1]]" --stride 1
optimumai vision pool
optimumai vision sobel
optimumai vision cnn --level engineer
```

---

## LLM evaluation — `optimumai.evaluation`

| Metric | What it measures |
|---|---|
| BLEU | N-gram precision with brevity penalty |
| ROUGE-N/L | N-gram recall — summarization quality |
| Perplexity | Model surprise on held-out text — lower is better |
| Calibration (ECE) | When model says "90%", is it right 90% of the time? |
| Faithfulness | Heuristic: is the answer supported by retrieved context? |
| Exact match / token-F1 | Token-level accuracy for QA benchmarks |

```python
from optimumai.evaluation import bleu, rouge_l, perplexity, ece
from optimumai.evaluation.text_metrics import bleu_trace

bleu("the quick brown fox jumps", "the quick brown fox leaps", max_n=1)
bleu_trace("the quick brown fox jumps", "the quick brown fox leaps", max_n=1).render("beginner")
```

```bash
optimumai eval bleu "the quick brown fox jumps" "the quick brown fox leaps" --max-n 1
optimumai eval rouge "the quick brown fox" "the quick brown fox jumps" -n 1
optimumai eval perplexity "[0.5,0.25,0.8]"
optimumai eval calibration
optimumai eval faithfulness
```

!!! warning "Short strings can score BLEU = 0"
    With `--max-n 4`, a short candidate/reference pair may have no overlapping
    4-grams and score `0.0`. Use `--max-n 1` or longer text.

---

## Prompt engineering — `optimumai.prompting`

Standard patterns for getting more out of a frozen LLM by changing *what you
send it*, not the model itself. Each is a deterministic offline trace — no API
key needed.

| Pattern | What it does |
|---|---|
| `zero-shot` | Role + instruction + task, no examples |
| `few-shot` | In-context learning from K exemplars |
| `chain-of-thought` | Elicit reasoning steps before the final answer |
| `react` | Interleave Thought / Action / Observation with a tool |
| `self-consistency` | Sample N chains, majority-vote the answer |
| `structured-output` | Constrain to a validated JSON schema |

```python
from optimumai.prompting import chain_of_thought, self_consistency

chain_of_thought("If a train travels 60 miles in 2 hours, what is its speed?", explain=True)
self_consistency("2+2?", sampled_answers=["4", "4", "5"], explain=True)   # -> "4"
```

```bash
optimumai prompt zero-shot
optimumai prompt few-shot
optimumai prompt chain-of-thought
optimumai prompt react
optimumai prompt self-consistency
optimumai prompt structured-output
```

---

## Augmented RNNs — `optimumai.augmented_rnns`

The pre-transformer ideas that made attention mainstream. NTM's read head is
the direct ancestor of transformer attention.

| Component | What it does |
|---|---|
| `attention` | Content-based soft attention as differentiable memory read |
| `ntm` | Neural Turing Machine: cosine-addressed read + erase/add write |
| `act` | Adaptive Computation Time: halting probability and ponder cost |

```python
from optimumai.augmented_rnns import attention_read, ntm_read, adaptive_computation_time
import numpy as np

memory = np.array([[1., 0., -1.], [0., 1., 0.], [-1., 0., 1.], [.5, .5, .5]])
attention_read(np.array([1., 0., -1.]), memory)
adaptive_computation_time(np.array([0.5, 1.2, 2.0, -0.3, 3.0]), eps=0.01)
```

```bash
optimumai augrnn attention
optimumai augrnn ntm
optimumai augrnn act
```

---

## Systems — KV cache & VRAM — `optimumai.foundations`

"Will it fit?" answered with precise formulas.

**Formula.** KV cache bytes: `2 × n_layers × n_kv_heads × d_head × S × B × bytes`

```python
from optimumai import kv_cache_size, vram_estimate

kv_cache_size(n_layers=32, n_heads=32, head_dim=128, seq_len=4096)
vram_estimate(params_billions=7, training=True)
```

```bash
optimumai kvcache --seq-len 8192
optimumai kvcache --heads 32 --kv-heads 4        # GQA: fewer KV heads than Q
optimumai vram --params 70
optimumai vram --params 7 --inference
optimumai learn cuda_matmul
optimumai learn pytorch
optimumai learn jax
```

---

## GPU kernels from scratch — `optimumai.kernels`

Write per-thread kernels, run on a pure-Python CUDA simulator, check against
NumPy, and get graded feedback.

```python
from optimumai.kernels import KernelWorkbench

wb = KernelWorkbench()
print(wb.get_challenge("vector_add").prompt)

def my_kernel(ctx, inp, out):
    i = ctx.idx.global_id
    if i < out.size:
        out[i] = ctx.gload(inp["a"], i) + ctx.gload(inp["b"], i)

print(wb.submit("vector_add", my_kernel).feedback)   # ✓ correct
```

```bash
optimumai kernel                       # list kernels
optimumai kernel matmul                # tiled matmul + shared-memory tiling
optimumai kernel flash_attention       # fused online-softmax, provably exact
optimumai kernel --backends            # numba / cupy / triton, auto-detected
```

Progression: `scalar_add → vector_add → tiled matmul → softmax → flash attention`

---

## Visualization, playgrounds & circuits

See [Visualization & circuits](visualization.md) for the full guide.

```bash
# Concept registry — 21+ concepts in PNG and/or GIF
optimumai visualize                              # list all concepts + formats
optimumai visualize attention --fmt gif --out attn.gif
optimumai visualize kmeans --fmt png --out km.png

# Matplotlib plots
optimumai plot activation --name gelu --out gelu.png
optimumai plot softmax --out temps.png
optimumai plot attention --text "the cat sat" --out att.png
optimumai plot embeddings --out emb.png
optimumai plot training --out curve.png
optimumai landscape rosenbrock --out land.png
optimumai landscape bowl --kind contour --out bowl.png

# Animated GIFs
optimumai animate descent --out descent.gif
optimumai animate diffusion --out diffusion.gif
optimumai animate softmax --out softmax.gif

# Interactive HTML (no server, works offline)
optimumai editor "a*x^2 + b*x + c"              # editable equation ↔ graph
optimumai circuit "(a*b + c) * f" --vars "a=2,b=-3,c=10,f=-2" --fmt html
optimumai playground attention                   # hover token, drag temperature
optimumai playground kmeans                      # click to add points
optimumai playground astar                       # draw walls, watch A*
optimumai playground softmax                     # drag logits live
optimumai playground backprop                    # drag inputs, watch gradients
```

---

## Interactive analysis

```bash
optimumai repl                                   # interactive session
optimumai compare relu gelu --input "[-2,-1,0,1,2]"
optimumai sweep softmax --values "[0.25,0.5,1,2]"
```

---

## Token generation — `optimumai.llm`

See [Token generation](generation.md) for the full guide.

```python
from optimumai import generate
print(generate("The key insight behind attention is", max_tokens=32))
```

```bash
optimumai providers
optimumai generate "The math behind attention is" --max-tokens 32
optimumai generate "Explain softmax" --provider ollama --model llama3.2
optimumai ask "why LayerNorm after attention?"   # optional LLM tutor (needs [llm])
```

Providers tried in order: **Ollama** (local, zero keys) → **Hugging Face** (`HF_TOKEN`)
→ **Anthropic** (`ANTHROPIC_API_KEY` + `[llm]`) → **toy bigram** (always works offline).

---

## Course & retention — `optimumai.curriculum`

See [Learning path](course.md) for the full guide.

```python
from optimumai import COURSE, ProgressTracker

for lesson in COURSE:
    print(lesson.track, lesson.id, "—", lesson.summary)

COURSE.get("rag").run("engineer")
ProgressTracker().mark_complete("rag")
```

```bash
optimumai course              # full path — 76 lessons across 20 tracks
optimumai learn attention     # run a lesson (auto-marks complete)
optimumai progress            # progress bar + what's next
optimumai search embedding    # find lessons by keyword
optimumai quiz softmax        # active recall
optimumai review              # spaced repetition (SM-2)
optimumai exercise backprop   # compute-the-answer exercises
optimumai dashboard           # Streamlit progress dashboard (needs [dashboard])
```
