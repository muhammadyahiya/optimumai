# Classical AI, ML & RL

Six pure-NumPy modules that broaden OptimumAI from the deep-learning/LLM stack
to the whole field. Every concept is an explainable [`Trace`](api.md) — run it
with `explain=True` in Python, from the CLI, or as a course lesson.

---

## Machine learning — `optimumai.ml`

Linear & logistic regression, k-means, KNN, decision trees, Gaussian naive
Bayes, PCA, and a complete metrics module.

| Class / function | CLI | What it teaches |
|---|---|---|
| `LinearRegression` | `optimumai ml linreg` | OLS normal equation: `θ = (XᵀX)⁻¹Xᵀy` |
| `LogisticRegression` | `optimumai ml logreg` | `ŷ = σ(Xθ)`, cross-entropy, gradient descent |
| `KMeans` | `optimumai ml kmeans` | Lloyd's algorithm: assign → recompute → repeat |
| `KNN` | `optimumai ml knn` | Majority vote among k nearest Euclidean neighbors |
| `DecisionTree` | `optimumai ml tree` | Greedy information-gain splits (Gini/entropy) |
| `GaussianNB` | `optimumai ml nb` | Bayes' rule with per-feature Gaussian likelihood |
| `PCA` | `optimumai ml pca` | Eigendecomposition of the covariance matrix |
| `metrics` | `optimumai ml metrics` | `accuracy`, `precision_recall_f1`, `confusion_matrix`, `mse`, `r2_score`, `roc_auc` |

```python
from optimumai.ml import LinearRegression, LogisticRegression, KMeans, KNN
from optimumai.ml import DecisionTree, GaussianNB, PCA
from optimumai.ml.metrics import accuracy, precision_recall_f1, roc_auc

# Linear regression
model = LinearRegression()
model.fit([[1], [2], [3], [4]], [2, 4, 6, 8])
model.predict([[5]])                          # ≈ [10]

# K-means
km = KMeans(k=2)
km.fit([[0, 0], [0, 1], [9, 9], [9, 8]])
km.predict([[1, 1]])                          # -> cluster 0
```

```bash
optimumai ml linreg "[[1],[2],[3],[4]]" "[2,4,6,8]"
optimumai ml logreg
optimumai ml kmeans "[[0,0],[0,1],[9,9],[9,8]]" --k 2
optimumai ml knn
optimumai ml tree
optimumai ml nb
optimumai ml pca
optimumai ml metrics
```

::: optimumai.ml

---

## Classical AI search — `optimumai.search`

BFS, DFS, uniform-cost (Dijkstra), greedy best-first, A*, minimax, and
alpha-beta pruning over reusable `Graph` / `GridWorld` problems.

| Function | CLI | What it teaches |
|---|---|---|
| `bfs` | `optimumai algo bfs` | Fewest-edge path (uninformed, queue) |
| `dfs` | `optimumai algo bfs` | Depth-first (uninformed, stack) |
| `ucs` (Dijkstra) | `optimumai algo bfs` | Cheapest-cost path (uniform cost) |
| `astar` | `optimumai algo astar` | `f = g + h`, optimal when h is admissible |
| `minimax` | `optimumai algo minimax` | Game tree search with opponent |
| `alpha_beta` | `optimumai algo minimax` | Minimax with pruning — same result, less work |

```python
from optimumai.search import bfs, ucs, astar
from optimumai.search.problem import Graph, GridWorld

g = Graph()
g.add_edge("A", "B", 1); g.add_edge("B", "C", 1)
g.add_edge("C", "D", 1); g.add_edge("B", "D", 5)

bfs(g, "A", "D")    # fewest edges: ['A', 'B', 'D']
ucs(g, "A", "D")    # cheapest cost: ['A', 'B', 'C', 'D'] (cost 3)
```

!!! note "BFS vs. UCS"
    BFS finds the path with the *fewest edges*; UCS (Dijkstra) finds the path
    with the *lowest total cost* — they can disagree on weighted graphs.

```bash
optimumai algo bfs
optimumai algo astar
optimumai algo minimax
```

::: optimumai.search

---

## Reinforcement learning — `optimumai.rl`

MDPs with value & policy iteration (the Bellman equation), tabular Q-learning /
SARSA, REINFORCE, and the PPO clipped surrogate objective.

| Component | CLI | What it teaches |
|---|---|---|
| `value_iteration` | `optimumai rl mdp` | Bellman backup to exact V* |
| `policy_iteration` | `optimumai rl mdp` | Alternating policy eval + improvement |
| `q_learning` | `optimumai rl q-learning` | Off-policy TD control |
| `sarsa` | `optimumai rl q-learning` | On-policy TD control |
| `reinforce` | `optimumai rl reinforce` | Policy gradient: REINFORCE on a bandit |
| `ppo_clip` | `optimumai rl ppo` | Clipped surrogate objective |

```python
from optimumai.rl import value_iteration, q_learning, ppo_clip
from optimumai.rl.mdp import MDP

mdp = MDP.demo()           # a small gridworld
V, pi = value_iteration(mdp, gamma=0.9, explain=True)
```

```bash
optimumai rl mdp
optimumai rl q-learning
optimumai rl reinforce
optimumai rl ppo
```

::: optimumai.rl

---

## NLP — `optimumai.nlp`

Byte-pair encoding, TF-IDF, n-gram language models, Levenshtein edit distance,
and skip-gram word2vec.

| Component | CLI | What it teaches |
|---|---|---|
| `BPETokenizer` | `optimumai nlp bpe` | Vocabulary learning via pair merges |
| `tfidf` | `optimumai nlp tfidf` | Term distinctiveness: `tf · log(N/df)` |
| `NGramLM` | `optimumai nlp ngram` | N-gram LM, add-k smoothing, perplexity |
| `edit_distance` | `optimumai nlp edit-distance` | Levenshtein DP, O(mn) |
| `word2vec` | `optimumai nlp word2vec` | Skip-gram, negative sampling, one SGD step |

```python
from optimumai.nlp import BPETokenizer, edit_distance

tok = BPETokenizer(num_merges=8)
tok.train(["low", "lower", "lowest", "newer", "newest"])
tok.encode("lowest")                   # -> ['lo', 'west</w>']

edit_distance("kitten", "sitting", explain=True)   # -> 3
```

```bash
optimumai nlp bpe lowest
optimumai nlp bpe --merges 12 lowest
optimumai nlp tfidf "the cat sat" "the dog sat"
optimumai nlp ngram
optimumai nlp edit-distance kitten sitting
optimumai nlp word2vec
```

::: optimumai.nlp

---

## Computer vision — `optimumai.vision`

2-D convolution, pooling, Sobel edge detection, and a tiny CNN forward pass.

| Component | CLI | What it teaches |
|---|---|---|
| `conv2d` | `optimumai vision conv` | Sliding filter, output size formula |
| `pool2d` | `optimumai vision pool` | Max & average pooling |
| `sobel` | `optimumai vision sobel` | Edge detection via gradient magnitude |
| `cnn_forward` | `optimumai vision cnn` | Conv → relu → pool stack, shape narrated |

```python
from optimumai.vision.convolution import conv2d_trace
from optimumai.vision.pooling import pool2d_trace
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

::: optimumai.vision

---

## LLM evaluation — `optimumai.evaluation`

BLEU, ROUGE-N/L, exact match, token-F1, perplexity, calibration (ECE), and a
candid faithfulness/hallucination heuristic.

| Metric | CLI | What it measures |
|---|---|---|
| `bleu` | `optimumai eval bleu` | N-gram precision + brevity penalty |
| `rouge_n`, `rouge_l` | `optimumai eval rouge` | N-gram / LCS recall |
| `perplexity` | `optimumai eval perplexity` | Model surprise — lower is better |
| `ece` | `optimumai eval calibration` | Calibration: confidence ≈ accuracy? |
| `faithfulness_score` | `optimumai eval faithfulness` | Claim-context overlap heuristic |

```python
from optimumai.evaluation import bleu, rouge_l, perplexity, ece, faithfulness_score
from optimumai.evaluation.text_metrics import bleu_trace

bleu("the quick brown fox jumps", "the quick brown fox leaps", max_n=1)
bleu_trace("the quick brown fox jumps", "the quick brown fox leaps").render("beginner")
```

```bash
optimumai eval bleu "the quick brown fox jumps" "the quick brown fox leaps" --max-n 1
optimumai eval rouge "the quick brown fox" "the quick brown fox jumps" -n 1
optimumai eval perplexity "[0.5,0.25,0.8]"
optimumai eval calibration
optimumai eval faithfulness
```

!!! warning "Short strings can score BLEU = 0"
    With `--max-n 4`, a short pair may have no 4-gram overlap and score `0.0`.
    Use `--max-n 1` or longer text.

::: optimumai.evaluation
