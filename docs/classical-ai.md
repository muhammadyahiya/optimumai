# Classical AI, ML & RL (v1.1)

Six pure-NumPy packages that broaden OptimumAI from the deep-learning/LLM stack to
the whole field. Every concept is an explainable [`Trace`](api.md) — run it with
`explain=True` in Python, from the CLI (`optimumai ml|algo|rl|nlp|vision|eval …`),
or as a course lesson.

## Machine learning — `optimumai.ml`

Linear & logistic regression, k-means, KNN, decision trees, Gaussian naive Bayes,
PCA, and a metrics module (accuracy, precision/recall/F1, confusion matrix, MSE,
R², ROC-AUC).

::: optimumai.ml

## Classical search — `optimumai.search`

BFS, DFS, uniform-cost (Dijkstra), greedy best-first, A\*, minimax, and
alpha-beta pruning over reusable `Graph` / `GridWorld` problems.

::: optimumai.search

## Reinforcement learning — `optimumai.rl`

MDPs with value & policy iteration (the Bellman equation), tabular Q-learning /
SARSA, REINFORCE, and the PPO clipped surrogate objective.

::: optimumai.rl

## NLP — `optimumai.nlp`

Byte-pair encoding, TF-IDF, n-gram language models (+ perplexity), Levenshtein
edit distance, and skip-gram word2vec.

::: optimumai.nlp

## Computer vision — `optimumai.vision`

2-D convolution, pooling, Sobel edge detection, and a tiny CNN forward pass.

::: optimumai.vision

## LLM evaluation — `optimumai.evaluation`

BLEU, ROUGE-N/L, exact match, token-F1, perplexity, calibration (ECE), and a
candid faithfulness/hallucination heuristic.

::: optimumai.evaluation
