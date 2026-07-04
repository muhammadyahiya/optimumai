# Features — the complete tour

Every capability in OptimumAI follows one pattern: **the same code computes the
real answer and can explain itself.** Call anything with `explain=True` (or
render a `*_trace(...)` result) and you get a step-by-step `Trace` — the exact
arithmetic, the formula, and *why AI actually uses this* — at a detail level you
choose with `level=` (`beginner → intermediate → engineer → researcher`; see
[`ExplainLevel`](cli.md#explain-levels-and-the-explaintrue-philosophy)).

This page is organized by theme. Each entry gives you the intuition first, then
the formula, then a runnable Python snippet and its CLI equivalent. Every
snippet on this page was executed against the installed package while writing
these docs.

!!! note "The shared mechanic"
    Almost everything here is one of two shapes:

    ```python
    Thing(...).op(args, explain=True, level="engineer")   # prints + returns the result
    op_trace(args).render("engineer")                      # build the Trace, then render it
    ```

    `explain=False` (the default) skips the printing and returns just the numeric
    result — the *same* computation path, so the trace can never lie about what
    ran.

---

## Foundations & math

### Vectors & matrices — `optimumai.algebra`

**Intuition.** A dot product is "how much do these two things point the same
way, scaled by their sizes" — it's the single operation underneath cosine
similarity, attention scores, and every matrix multiply. Once you can see a dot
product as a sum of products, matrix multiplication is just a grid of dot
products.

**Formula.** `a · b = Σᵢ aᵢ·bᵢ`, `‖a‖ = √(Σᵢ aᵢ²)`, `cos(θ) = (a·b)/(‖a‖·‖b‖)`,
and `C[i,j] = Σₖ A[i,k]·B[k,j]` for matrix multiply (`O(m·k·n)`).

```python
from optimumai import Vector, Matrix

Vector([1, 2, 3]).dot(Vector([4, 5, 6]), explain=True)                    # 32.0
Vector([1, 2, 3]).cosine_similarity(Vector([2, 4, 6]), explain=True)      # 1.0
Matrix([[1, 2], [3, 4]]).matmul(Matrix([[5, 6], [7, 8]]), explain=True)
```

```bash
optimumai algebra dot "[1,2,3]" "[4,5,6]"
optimumai algebra cosine "[1,2,3]" "[2,4,6]"
optimumai algebra matmul "[[1,2],[3,4]]" "[[5,6],[7,8]]"
```

### Softmax — `optimumai.probability`

**Intuition.** Softmax turns arbitrary scores ("logits") into a probability
distribution — bigger scores get more probability mass, but everything stays
positive and sums to 1. Temperature controls how sharp that distribution is:
low temperature nearly one-hots the max; high temperature flattens toward
uniform.

**Formula.** `softmax(xᵢ) = e^(xᵢ/T) / Σⱼ e^(xⱼ/T)`, computed with the
numerically-stable max-subtraction trick.

```python
from optimumai import softmax
softmax([2.0, 1.0, 0.1], temperature=0.5, explain=True)
```

```bash
optimumai softmax "[2,1,0.1]" --temperature 0.5
```

### Calculus — `optimumai.calculus`

**Intuition.** A derivative is a slope; if you can't compute it symbolically,
you can still *estimate* it by nudging the input a tiny bit and watching the
output move. That's the central-difference trick every framework's
finite-difference gradient checker is built on.

**Formula.** `f'(x) ≈ (f(x+h) − f(x−h)) / 2h`; the gradient applies this per
dimension.

```python
from optimumai import derivative, gradient
derivative(lambda x: x**2, 3.0, explain=True)                 # ≈ 6.0
gradient(lambda p: p[0]**2 + p[1]**2, [3.0, 4.0], explain=True)  # ≈ [6.0, 8.0]
```

```bash
optimumai learn derivative
optimumai learn chain_rule
```

### Numerical integration — `optimumai.foundations`

**Intuition.** An expectation *is* an integral — "average value of f weighted
by a distribution" is exactly what `∫f(x)p(x)dx` computes. Two ways to
approximate it without calculus: slice the area into trapezoids, or throw
random darts and average (Monte Carlo) — the same idea behind sampling-based
RL and diffusion training.

**Formula.** Trapezoid rule sums `(f(xᵢ)+f(xᵢ₊₁))/2 · Δx`; Monte Carlo computes
`(b−a) · mean(f(X))` for `X` uniform on `[a, b]`.

```python
from optimumai import integrate
integrate(lambda x: x**2, 0, 1, method="trapezoid")      # ≈ 0.333
integrate(lambda x: x**2, 0, 1, method="monte_carlo")    # ≈ 0.33 (seeded, deterministic)
```

```bash
optimumai learn integration
```

---

## Autograd & neural nets

### Value — a scalar autograd engine — `optimumai.autograd`

**Intuition.** Every neural net trains by the chain rule: build a graph of
operations as you go forward, then walk it backward multiplying local
derivatives. `Value` is a single scalar that remembers how it was built, so
calling `.backward()` on the end of the graph fills in every gradient
automatically — this *is* what PyTorch's `requires_grad=True` does under the
hood, just for one number at a time (Karpathy's micrograd).

**Formula.** Reverse-mode chain rule: `∂L/∂x = Σ (∂L/∂out) · (∂out/∂x)` summed
over every path from `x` to the loss `L`.

```python
from optimumai import Value

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
L = (a * b).tanh()
L.label = "L"
L.backprop(explain=True)   # watch the chain rule flow backwards; a.grad, b.grad
```

```bash
optimumai backprop
```

### Optimizers — `optimumai.optimization`

**Intuition.** Gradient descent takes a step opposite the gradient — downhill.
Adam is the same idea with memory: it tracks a running average of the gradient
(momentum) and of its squared magnitude (adaptive scaling per-parameter), so it
takes bigger, more confident steps in flat directions and smaller steps where
the loss surface is noisy.

**Formula.** SGD: `w ← w − lr·∂L/∂w`. Adam adds bias-corrected first/second
moment estimates `m̂, v̂` and updates `w ← w − lr·m̂/(√v̂ + ε)`.

```python
from optimumai import Value, Adam, minimize

x, y = Value(0.0), Value(0.0)
minimize(lambda: (x - 3.0) ** 2 + (y + 1.0) ** 2, [x, y], Adam([x, y], lr=0.3),
          steps=30, explain=True)   # converges toward x≈3, y≈-1
```

```bash
optimumai train --steps 150
```

### Neural networks — `optimumai.neural_networks`

**Intuition.** A neuron is `activation(weighted sum of inputs + bias)`. A
layer is several neurons run in parallel; an MLP stacks layers. There is no
separate "neural network math" here — it's the `Value` autograd engine
composed a few thousand times, which is why training an `MLP` is just calling
`minimize` on its parameters.

**Formula.** `output = φ(Σᵢ wᵢxᵢ + b)` per neuron, where `φ` is `tanh`, `relu`,
or linear (the output layer is forced linear).

```python
from optimumai import MLP

mlp = MLP(3, [4, 4, 1], activation="tanh", seed=0)
mlp([2.0, 3.0, -1.0])          # a forward pass

from optimumai.neural_networks import train_demo
train_demo(steps=150).render("intermediate")   # loss falls to ~0
```

```bash
optimumai train --steps 150 --lr 0.05
```

---

## Transformers & attention

### Scaled dot-product attention — `optimumai.transformers`

**Intuition.** Attention answers "for this query, how much should I weight
each value, based on how well its key matches?" — it's cosine-similarity-style
scoring (a dot product), turned into weights with softmax, then used to blend
the values. Scaling by `√dₖ` keeps the dot products from growing so large that
softmax saturates.

**Formula.** `Attention(Q,K,V) = softmax(QKᵀ/√dₖ)·V`.

```python
from optimumai import Attention
Attention.demo().render("engineer")
```

```bash
optimumai attention --demo --level engineer
```

### Multi-head attention & the transformer block — `optimumai.transformers`

**Intuition.** One attention head can only learn one notion of "relatedness."
Multi-head attention runs several smaller attention operations in parallel
subspaces (same total compute as one big head) so different heads can specialize
— syntax, coreference, position. A causal mask hides future tokens so the model
can't cheat by looking ahead — that's what makes it an autoregressive decoder
(GPT-style).

**Formula.** Per head: `softmax(XₕXₕᵀ/√d_head)·Xₕ`, concatenated across heads;
the causal mask sets future positions to `-∞` before the softmax. A
`TransformerBlock` wraps this pre-norm: `x += MHA(LN(x))`, then `x += FFN(LN(x))`.

```python
from optimumai import MultiHeadAttention, TransformerBlock
MultiHeadAttention.demo().render("engineer")
```

```bash
optimumai attention --demo --level researcher
```

### Positional encoding — `optimumai.transformers`

**Intuition.** Attention has no notion of order — it treats a sequence like a
bag of tokens. Sinusoidal positional encodings inject position by adding a
unique wave pattern to each position's embedding, at frequencies that let the
model infer relative distance between positions via linear combinations.

**Formula.** `PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(...)`.

```python
from optimumai.transformers.positional import positional_encoding
positional_encoding(seq_len=4, d_model=8, explain=True)
```

```bash
optimumai learn positional
```

### Give it your own text — `TextPipeline`

**Intuition.** This is the whole LLM forward pass in miniature: tokenize your
sentence, embed each token, add positional encoding, run it through a stack of
transformer blocks, and read off a probability distribution over the next
token. It's the same pipeline as GPT — only the vocabulary size, model
dimension, depth, and training separate this from a real model.

```python
from optimumai import TextPipeline
TextPipeline("attention is all you need", layers=2).forward(explain=True)
```

```bash
optimumai trace-text "why is the sky blue" --layers 2 --level engineer
```

---

## World models & interpretability

### JEPA — `optimumai.world_models`

**Intuition.** Yann LeCun's argument: don't train a model to reconstruct every
pixel of what happens next — predict *in representation space* instead. JEPA
is an energy-based model: energy is low when a predicted embedding matches the
real target's embedding, and high when it doesn't. Understanding meaning, not
grain.

**Formula.** `E(x, y) = ‖g(f(x)) − f(y)‖²`.

```python
from optimumai import JEPA
JEPA.demo().render("engineer")   # energy = ‖predicted embedding − target embedding‖²
```

```bash
optimumai jepa --demo --level engineer
```

### Superposition — `optimumai.interpretability`

**Intuition.** Anthropic's toy model of why individual neurons often respond
to several unrelated concepts (polysemanticity): if you have more features
than neurons, and the features are sparse (rarely active together), the model
can pack them into overlapping directions and mostly get away with it. This
toy model reproduces that interference directly.

**Formula.** Encode `x = W·h`, decode `ĥ = Wᵀ·x`, with `W`'s columns
L2-normalized — off-diagonal terms of `WᵀW` measure the interference between
features sharing a neuron.

```python
from optimumai import superposition
superposition(n_features=5, n_neurons=2, explain=True)
```

```bash
optimumai superposition --features 5 --neurons 2
```

---

## Applied AI — embeddings, RAG, diffusion

### Embeddings — `optimumai.embeddings`

**Intuition.** An embedding is a lookup table mapping discrete tokens to dense
vectors — the first step of every neural NLP model. Nearest-neighbor search
over those vectors (cosine similarity) is how you find "similar" tokens
without ever training a full model.

```python
from optimumai import embedding_lookup, nearest_neighbors
embedding_lookup(["cat", "dog", "car"], dim=4, explain=True)
nearest_neighbors("cat", vocab=["dog", "car", "kitten"], dim=4, k=2, explain=True)
```

```bash
optimumai learn embeddings
```

### Retrieval-augmented generation — `optimumai.rag`

**Intuition.** RAG grounds a model's answer in documents it never saw during
training: embed the query, cosine-search a document store, stuff the top-k
matches into the prompt as context, and let the model answer from *that*
context instead of its parametric memory alone.

**Formula.** `prompt = stuff(top-k by cos(E[query], E[docᵢ])) ⊕ query`.

```python
from optimumai import RAGPipeline
RAGPipeline().forward("how do neural networks learn?", k=2, explain=True)
```

```bash
optimumai learn rag
```

### Diffusion — `optimumai.diffusion`

**Intuition.** Diffusion models learn to *reverse* a noising process. The
forward process has a closed form — you can jump straight to noise level `t`
without simulating every step in between — which is exactly what makes
training efficient: sample a random `t`, add the matching noise, and train the
network to predict the noise that was added.

**Formula.** `xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε`, with a linear β schedule and
`ᾱₜ = ∏ₛ(1−βₛ)`.

```python
from optimumai import forward_diffusion
import numpy as np
forward_diffusion(np.array([1.0, 2, 3, 4, 5, 6]), timesteps=10, explain=True)
```

```bash
optimumai learn diffusion
```

---

## Classical ML — `optimumai.ml`

**Intuition.** Before deep learning, these were (and often still are) the
right tool: a closed-form line fit, a majority vote among neighbors, a tree of
yes/no questions, clustering by nearest centroid. Every one of them still
shows up as a baseline, a building block (logistic regression *is* the output
layer of a classifier), or a diagnostic.

| Model | Formula / idea |
|---|---|
| `LinearRegression` | OLS via the normal equation: `θ = (XᵀX)⁻¹Xᵀy` |
| `LogisticRegression` | `ŷ = σ(Xθ)`, cross-entropy loss, gradient descent |
| `KMeans` | Lloyd's algorithm: assign to nearest centroid, recompute centroids, repeat |
| `KNN` | Classify by majority vote among the k nearest (Euclidean) neighbors |
| `DecisionTree` | Greedy split search maximizing Gini/entropy information gain |
| `GaussianNB` | Bayes' rule with a per-feature Gaussian likelihood |
| `PCA` | Eigendecomposition of the covariance matrix; project onto the top components |
| `optimumai.ml.metrics` | `accuracy`, `precision_recall_f1`, `confusion_matrix`, `mse`, `r2_score`, `roc_auc` |

```python
from optimumai.ml import LinearRegression, KMeans

LinearRegression().fit([[1], [2], [3], [4]], [2, 4, 6, 8])   # θ ≈ [0, 2]
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

!!! note "A docstring bug worth knowing about"
    `DecisionTree`'s own module docstring shows a `max_depth=1` example
    claiming a split happens — in the current code, the depth check fires
    before any split at `max_depth=1`, so you need `max_depth=2` to see the
    tree actually split (verified by running it).

---

## Classical AI search — `optimumai.search`

**Intuition.** "Search" here means exploring a space of states to find a goal
— the common ancestor of pathfinding, puzzle solvers, and game-playing AI.
Uninformed search (BFS/DFS/UCS) explores blindly; informed search (greedy,
A\*) uses a heuristic to focus the search toward the goal; adversarial search
(minimax) reasons about an opponent trying to minimize your score.

**Formula.** A\* expands the node minimizing `f(n) = g(n) + h(n)` (cost so far
+ estimated cost to goal); it's optimal when `h` never overestimates
(admissible). Alpha-beta pruning cuts minimax branches once
`alpha >= beta` — provably safe, same result, less work.

```python
from optimumai.search import bfs, astar, alpha_beta
from optimumai.search.problem import Graph

g = Graph()
g.add_edge("A", "B", 1); g.add_edge("B", "C", 1)
g.add_edge("C", "D", 1); g.add_edge("B", "D", 5)
bfs(g, "A", "D")   # fewest edges, not necessarily cheapest: ['A', 'B', 'D']
```

```bash
optimumai algo bfs
optimumai algo astar
optimumai algo minimax
```

!!! note "BFS vs. UCS"
    BFS finds the path with the *fewest edges*; uniform-cost search (Dijkstra)
    finds the path with the *lowest total cost* — they can disagree, exactly
    as in the snippet above (`bfs` picks the 2-hop, cost-6 path; UCS would
    pick the 3-hop, cost-4 path).

---

## Reinforcement learning — `optimumai.rl`

**Intuition.** An MDP formalizes "an agent acting in a world with delayed
reward." Value iteration solves a *known* MDP exactly by repeatedly applying
the Bellman backup until values stop changing — the yardstick every model-free
method (Q-learning) is checked against. Policy-gradient methods (REINFORCE,
PPO) skip modeling the environment and directly nudge the policy toward
actions that led to higher reward, with PPO adding a clip so a single update
can't move the policy too far and destabilize training.

**Formula.** Bellman optimality backup:
`V(s) ← maxₐ Σₛ' P(s'|s,a)[R(s,a,s') + γV(s')]`. PPO's clipped objective:
`L = E[min(r·A, clip(r, 1−ε, 1+ε)·A)]` where `r = exp(new_logprob − old_logprob)`.

```python
from optimumai.rl import value_iteration, q_learning, ppo_clip
from optimumai.rl.mdp import MDP

# see `optimumai rl mdp` for a ready-made demo MDP
ppo_clip()   # the clipped surrogate objective on a demo batch
```

```bash
optimumai rl mdp
optimumai rl q-learning
optimumai rl reinforce
optimumai rl ppo
```

---

## NLP — `optimumai.nlp`

**Intuition.** Before a transformer ever sees a sentence, it has to become
numbers. BPE learns a vocabulary by repeatedly merging the most frequent
adjacent symbol pair — exactly how GPT/LLaMA-style tokenizers are built.
TF-IDF scores a word by how distinctive it is to a document versus the whole
corpus. N-gram models are the simplest possible language model: predict the
next token from the last `n-1`. Edit distance measures how many single-character
edits separate two strings — the backbone of spell-checkers and WER/CER.

**Formula.** TF-IDF: `tfidf = (count/|d|) · (log(N/df) + 1)`. Edit distance:
classic Levenshtein dynamic program, `O(mn)`. N-gram perplexity:
`PPL = exp(-(1/N)Σ log P(wᵢ))`.

```python
from optimumai.nlp import BPETokenizer, edit_distance

tok = BPETokenizer(num_merges=8)
tok.train(["low", "lower", "lowest", "newer", "newest"])
tok.encode("lowest")             # -> ['lo', 'west</w>'] (merges applied in order)

edit_distance("kitten", "sitting")   # -> 3
```

```bash
optimumai nlp bpe lowest
optimumai nlp tfidf "the cat sat" "the dog sat"
optimumai nlp ngram
optimumai nlp edit-distance kitten sitting
optimumai nlp word2vec
```

!!! note "Reaching n-gram perplexity"
    The top-level `optimumai.perplexity` name is owned by
    `optimumai.evaluation.perplexity` (which takes raw token probabilities).
    The n-gram-model variant lives at `optimumai.nlp.perplexity` — a
    deliberate naming choice documented in `optimumai/__init__.py`, not an
    oversight.

---

## Computer vision — `optimumai.vision`

**Intuition.** A convolution slides a small learned filter over an image,
computing a weighted sum at each position — it's how a CNN detects local
patterns (edges, textures) regardless of where they appear. Pooling shrinks
the spatial size while keeping the strongest signal. Stack
conv → nonlinearity → pool a few times, flatten, and finish with a dense layer
+ softmax, and you have LeNet-5's skeleton — the same shape underlying AlexNet,
VGG, and ResNet.

**Formula.** Output size: `⌊(size − k + 2·padding) / stride⌋ + 1`. Sobel edges:
`magnitude = √(Gₓ² + Gᵧ²)` from fixed 3×3 gradient kernels.

```python
from optimumai.vision.convolution import conv2d_trace
import numpy as np

image = np.arange(36).reshape(6, 6).astype(float)
kernel = np.array([[1, 0], [0, -1]], dtype=float)
conv2d_trace(image, kernel).render("beginner")   # -> shape (5, 5)
```

```bash
optimumai vision conv
optimumai vision pool
optimumai vision sobel
optimumai vision cnn --level engineer
```

---

## LLM evaluation — `optimumai.evaluation`

**Intuition.** How do you score generated text against a reference? BLEU
rewards n-gram overlap but penalizes short, "safe" outputs (the brevity
penalty). ROUGE is BLEU's recall-oriented cousin, used for summarization.
Perplexity measures how "surprised" a language model is by held-out text —
lower means better fit. Calibration (ECE) asks a different question: when the
model says "90% confident," is it actually right 90% of the time? Faithfulness
is a first (imperfect) attempt at asking whether an answer is actually
supported by its retrieved context.

**Formula.** BLEU: `BP · exp((1/N)Σ log pₙ)` over clipped n-gram precisions
`pₙ`. ECE: `Σ (nᵦ/N)|confidenceᵦ − accuracyᵦ|` over confidence bins.

```python
from optimumai.evaluation import bleu, rouge_l, perplexity, ece, faithfulness_score
from optimumai.evaluation.text_metrics import bleu_trace

bleu("the quick brown fox jumps", "the quick brown fox leaps", max_n=1)   # -> 0.8 (plain call)
bleu_trace("the quick brown fox jumps", "the quick brown fox leaps", max_n=1).render("beginner")
```

```bash
optimumai eval bleu "the quick brown fox jumps" "the quick brown fox leaps" --max-n 1
optimumai eval rouge "the quick brown fox" "the quick brown fox jumps"
optimumai eval perplexity "[0.5,0.25,0.8]"
optimumai eval calibration
optimumai eval faithfulness
```

!!! warning "Short-sentence BLEU can look like zero"
    `optimumai eval bleu "a quick brown fox" "the quick brown fox" --max-n 4`
    (the exact strings from earlier README versions) scores **0.0** — a
    4-word candidate has no clean 4-grams to match, so BLEU-4 correctly
    collapses to zero. That's expected n-gram sparsity on short text, not a
    bug; try `--max-n 1` or longer sentences to see a nonzero score.

!!! note "Faithfulness is a heuristic, not a solved problem"
    `faithfulness_score`'s own docstring is explicit: hallucination detection
    is an unsolved research problem, and this function implements a
    transparent claim-token-overlap **proxy** — useful for teaching the shape
    of the problem, not a production-grade detector.

---

## Prompt engineering — `optimumai.prompting`

**Intuition.** These are the standard patterns for getting more out of a
frozen LLM by changing *what you send it*, not the model itself. Each one is
implemented here as a deterministic, offline prompt-assembly trace — no API
key needed to see exactly how the final prompt string is built and why it
tends to help (or where it can fail).

- **zero-shot** — role + instruction + task, no examples
- **few-shot** — in-context learning from K examples
- **chain-of-thought** — elicit reasoning steps before the final answer
- **react** — interleave Thought / Action / Observation with a tool
- **self-consistency** — sample N reasoning chains, majority-vote the answer
- **structured-output** — constrain the output to a validated JSON schema

```python
from optimumai.prompting import chain_of_thought, self_consistency

chain_of_thought("If a train travels 60 miles in 2 hours, what is its speed?",
                  explain=True)
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

**Intuition.** Before transformers, researchers were already attaching
external memory to RNNs — this is the lineage
[distill.pub traces](https://distill.pub/2016/augmented-rnns/) from
content-based attention to Neural Turing Machines to adaptive compute. The
"read" half of a Neural Turing Machine head is the direct ancestor of
transformer attention: both score memory/keys by similarity and blend by
softmax weights.

**Formula.** NTM addressing: cosine similarity between a key and each memory
row, softened by `softmax(β · cos)`. Adaptive Computation Time: halt at the
first step where the cumulative halting probability crosses `1 − ε`, paying a
"ponder cost" for how long it took.

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

## Systems & GPU/CUDA — `optimumai.foundations`, `optimumai.kernels`

**Intuition.** Two questions dominate real-world LLM engineering: "will this
fit in memory?" and "why is this slow?" The KV cache and VRAM calculators
answer the first; the GPU-kernel simulator answers the second by making you
write the per-thread body yourself and watch the memory-access pattern that
results.

**Formula.** KV cache bytes:
`2 · n_layers · kv_heads · head_dim · seq_len · batch · bytes_per_elem`
(the leading 2 is for K and V; fewer `kv_heads` than attention heads is GQA,
`kv_heads=1` is MQA).

```python
from optimumai import kv_cache_size, vram_estimate

kv_cache_size(n_layers=32, n_heads=32, head_dim=128, seq_len=4096)   # bytes
vram_estimate(params_billions=7, training=True)                      # ≈ 80 GB
```

```bash
optimumai kvcache --seq-len 8192
optimumai vram --params 70
optimumai learn cuda_matmul
optimumai learn pytorch
optimumai learn jax
```

### GPU kernels from scratch — `optimumai.kernels`

**Intuition.** A GPU runs one thread per output element (SIMT). This package
gives you a pure-Python simulator that models the thread grid and memory
hierarchy — write the per-thread body, launch it, and see exactly which
memory accesses coalesce and which don't, with zero GPU required.

```python
from optimumai.kernels import KernelWorkbench

wb = KernelWorkbench()

def my_kernel(ctx, inp, out):
    i = ctx.idx.global_id
    if i < out.size:
        out[i] = ctx.gload(inp["a"], i) + ctx.gload(inp["b"], i)

wb.submit("vector_add", my_kernel).feedback   # "✓ Correct — your kernel matches the reference."
```

```bash
optimumai kernel                 # list: scalar_add, vector_add, matmul, softmax, flash_attention
optimumai kernel matmul          # tiled matmul + the shared-memory tiling win
optimumai kernel flash_attention # fused online-softmax attention — provably exact
optimumai kernel --backends      # numba / cupy / triton, auto-detected
```

---

## Frontier: quantization, LoRA, DPO, FlashAttention — `optimumai.frontier`

**Intuition.** This is how today's large models are actually trained and
served cheaply. FlashAttention never materializes the full N×N attention
matrix — it tiles Q/K/V through fast on-chip memory and rescales an online
softmax as it goes, giving the *exact same numbers* as standard attention for
a fraction of the memory traffic. Quantization stores weights in fewer bits.
LoRA freezes the base model and trains a tiny low-rank update instead of the
full weight matrix. DPO aligns a model to human preferences directly from
preference pairs, with no separate reward model or RL loop.

**Formula.**

- FlashAttention: tiled online-softmax attention, algebraically identical to
  `softmax(QKᵀ/√dₖ)·V` (verified error ~1e-16 against the naive computation).
- Quantization: `q = round(x/scale) + zero_point`, dequantize with the inverse.
- LoRA: `W = W₀ + BA`, with `B` initialized to zero (so training starts
  exactly at the base model) and rank `r ≪ d`.
- DPO: `L = −log σ(β · [(logπ_chosen − logπ_ref,chosen) − (logπ_rejected − logπ_ref,rejected)])`
  — no reward model, no RL rollout.

```python
from optimumai.frontier import flash_attention, quantize, lora, dpo
import numpy as np

Q, K, V = (np.random.default_rng(0).normal(size=(6, 4)) for _ in range(3))
flash_attention(Q, K, V, block_size=2, explain=True)   # exact vs standard attention
```

```bash
optimumai learn flash_attention
optimumai learn lora
optimumai learn dpo
optimumai quantize "[0.1,-2.3,4.5,3.14]" --bits 4
```

!!! note "The '10,000× fewer params' LoRA headline"
    That figure is from the original LoRA paper's *whole-model* aggregate
    (e.g. across all of GPT-3's attention matrices). `optimumai.frontier.lora`
    reports the reduction for a *single* matrix — verified to range from
    roughly 768× to 6,144× at GPT-3-sized dimensions depending on the rank you
    choose. The mechanism is identical; the headline number is a
    whole-model claim, not a single-call return value.

---

## Visualization, playgrounds & flows — `optimumai.visualization`, `optimumai.circuit`

**Intuition.** Traces are great for one call; sometimes you want to *see* a
whole concept — a loss surface, an attention heatmap, an animated descent — or
drag an input and watch the math respond live. This layer covers static
plots, GIFs, self-contained interactive HTML (no server, no build), and the
computation graph rendered as a literal circuit with data and gradients
flowing the wires.

```python
from optimumai.visualization.concepts import render_concept, list_concepts
list_concepts()                                    # 21 concepts, png and/or gif
render_concept("attention", fmt="png", out="attn.png")

from optimumai.circuit import build_from_expression, to_terminal
value, graph = build_from_expression("(a*b + c) * f", {"a": 2, "b": -3, "c": 10, "f": -2})
to_terminal(graph)   # value=-8; leaf grads a=6, b=-4, c=-2, f=4
```

```bash
optimumai visualize                          # list every concept + its formats
optimumai visualize attention --fmt gif --out attn.gif
optimumai plot attention --text "the cat sat" --out att.png
optimumai landscape rosenbrock --out land.png
optimumai editor "a*x^2 + b*x + c"           # editable equation ↔ graph, in the browser
optimumai playground attention                # hover a query token, drag temperature
optimumai playground kmeans                   # click to add points, watch Lloyd's iterate
optimumai playground astar                    # draw walls, watch the frontier expand
optimumai playground softmax                  # drag the logits, watch probs move
optimumai playground backprop                 # drag a/b/c/f, watch gradients update
optimumai circuit "(a*b + c) * f" --vars "a=2,b=-3,c=10,f=-2" --fmt html --out circuit.html
optimumai animate descent --out descent.gif
```

!!! note "`visualize` has grown to 21 concepts"
    Earlier release notes mention "14 concepts" (the v1.0 static+animated
    registry); v1.2 added 7 more (`kmeans`, `decision_boundary`, `astar`,
    `value_iteration`, `conv2d`, `calibration`, `ppo_clip`) for the classical
    ML/AI packages. Run `optimumai visualize` with no argument to see the
    live, authoritative list.

!!! note "The circuit expression sandbox"
    `optimumai.circuit.build_from_expression` walks the parsed AST against an
    explicit allow-list (`Name`, `Constant`, `BinOp`, `UnaryOp`, and the
    arithmetic operators only) before evaluating — no function calls,
    attribute access, or subscripts are permitted, so `__import__(...)`,
    `open(...)`, and similar are rejected with `ValueError` even though the
    expression ultimately reaches `eval()`.

---

## The course & spaced-repetition — `optimumai.curriculum`, `optimumai.quiz`, `optimumai.review`

**Intuition.** Reading isn't remembering. Active recall (testing yourself)
roughly doubles retention versus re-reading, and spaced repetition schedules
reviews right before you'd forget. OptimumAI bundles a first-principles
76-lesson course across 20 tracks — linear algebra → calculus & autograd →
transformers → the whole field → frontier concepts — with both built in.

**Formula.** Spaced repetition uses SM-2: after a review graded 0–5, the next
interval grows by the current "ease factor" (itself nudged up or down based on
how well you did), so easy concepts get reviewed less often and hard ones come
back sooner.

```python
from optimumai import COURSE, ProgressTracker

for lesson in COURSE:
    print(lesson.track, lesson.id, "—", lesson.summary)

COURSE.get("rag").run("engineer")        # retrieval-augmented generation, traced
ProgressTracker().mark_complete("rag")   # track your progress
```

```bash
optimumai course              # the full path, grouped by track, with progress
optimumai learn attention     # run a lesson (auto-marks it complete)
optimumai progress            # a progress bar + what's next
optimumai search embedding    # find lessons by keyword
optimumai quiz softmax        # active recall — answer, get graded + explained
optimumai review              # spaced repetition (SM-2): whatever's due
optimumai exercise backprop   # compute-the-answer exercises, tolerance-graded
optimumai dashboard           # a visual Streamlit dashboard (needs [dashboard])
```

!!! note "The numbers, verified"
    76 lessons across 20 tracks, 20 quizzes totaling 57 questions — both
    counted directly from `COURSE` and `available_quizzes()` at doc-writing
    time, matching the README exactly.

---

## Token generation & the tutor — `optimumai.llm`, `optimumai.tutor`

**Intuition.** Generation is autoregressive decoding: predict a distribution
over the next token, pick one, append it, repeat. `generate` tries a chain of
providers so a demo always produces real tokens — a local Ollama server if
you have one (zero API keys), then Hugging Face, then Anthropic (via the
tutor), and finally a tiny built-in bigram sampler as a last resort so nothing
ever hard-fails offline.

```python
from optimumai import generate
generate("The math behind attention is", max_tokens=32)
```

```bash
optimumai providers                                  # what's available on this machine
optimumai generate "The math behind attention is"    # real tokens, streamed live
optimumai ask "why LayerNorm after attention?"        # optional LLM tutor
```

!!! note "The tutor degrades gracefully"
    Without `optimumai[llm]` and an API key, `Tutor().ask(...)` never raises —
    it returns a friendly message explaining exactly what's missing (which
    extra to install, which env var to set) and reminds you that none of this
    is required for the core math, which works fully offline.
