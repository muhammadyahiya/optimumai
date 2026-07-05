# CLI reference

The `optimumai` command wraps every capability in the library. Run
`optimumai --help` or `optimumai <command> --help` at any time.

## The `--level` flag and `explain=True`

Every command explains itself — the same step-by-step `Trace` as Python's
`explain=True`. Use `--level` to control the depth:

| Level | Adds |
|---|---|
| `beginner` | Steps + plain-English "why AI uses this" |
| `intermediate` | Per-step detail notes **(CLI default)** |
| `engineer` | Intermediate values + algorithmic complexity |
| `researcher` | Everything: formulas, complexity, references |

```bash
optimumai algebra dot "[1,2,3]" "[4,5,6]" --level beginner
optimumai algebra dot "[1,2,3]" "[4,5,6]" --level researcher
```

The same enum is `optimumai.core.explain.ExplainLevel` in Python (`level="engineer"`).

!!! tip "Omitting arguments runs a built-in demo"
    Most commands (`algebra dot`, `ml linreg`, `nlp bpe`, `vision conv`, ...)
    run a built-in demo when you omit the data arguments.
    Supply your own data to operate on real numbers/text.

---

## Onboarding & tour

```bash
optimumai start                        # 30-second guided tour — start here
optimumai --version                    # print installed version
```

---

## The course, progress & retention

```bash
optimumai course                       # full path — 76 lessons, 20 tracks, ✓/○ marks
optimumai learn                        # list every topic
optimumai learn dot                    # run a lesson (auto-marks it complete)
optimumai learn transformer --level researcher
optimumai learn attention --no-track   # run without recording completion
optimumai progress                     # progress bar + percentage + what's next
optimumai progress --reset             # clear all recorded progress
optimumai search attention             # find lessons by keyword (id/title/summary/track)
```

```bash
# Active recall
optimumai quiz                         # list all quiz topics (20 topics, 57 questions)
optimumai quiz softmax                 # answer a question, get graded + explained
optimumai quiz backprop
optimumai quiz attention

# Spaced repetition
optimumai review                       # SM-2: review whatever is due today

# Exercises
optimumai exercise                     # list exercise topics
optimumai exercise backprop            # compute-the-answer, tolerance-graded

# Dashboard & tutor
optimumai dashboard                    # Streamlit dashboard (needs [dashboard])
optimumai dashboard --port 8888
optimumai ask "why LayerNorm after attention?"   # LLM tutor (needs [llm])
```

Progress persists to `~/.optimumai/progress.json` (override with
`OPTIMUMAI_PROGRESS_PATH`). Quiz scores feed the SM-2 spaced-repetition
scheduler automatically.

---

## Algebra & probability

```bash
optimumai algebra dot "[1,2,3]" "[4,5,6]"
optimumai algebra dot -i                         # interactive: type vectors at a prompt
optimumai algebra cosine "[1,2,3]" "[2,4,6]"
optimumai algebra matmul "[[1,2],[3,4]]" "[[5,6],[7,8]]"
optimumai softmax "[2,1,0.1]" --temperature 0.5
optimumai softmax -i                             # interactive: type logits at a prompt
```

---

## Autograd, training & transformers

```bash
optimumai backprop                     # chain rule through a scalar graph
optimumai train --steps 150 --lr 0.05  # train a tiny MLP, watch loss fall
optimumai attention --demo             # scaled dot-product attention
optimumai attention --demo --seed 1 --level researcher
```

---

## World models & interpretability

```bash
optimumai jepa --demo                  # LeCun's JEPA energy-based world model
optimumai jepa --demo --level engineer
optimumai superposition                # Anthropic-style polysemantic neurons
optimumai superposition --features 8 --neurons 3
```

---

## Systems & foundations

```bash
optimumai kvcache --seq-len 8192       # KV-cache VRAM for a config
optimumai kvcache --heads 32 --kv-heads 4   # GQA: fewer KV heads than Q
optimumai vram --params 70             # VRAM to train a 70B model
optimumai vram --params 7 --inference  # inference-only VRAM
optimumai learn tensors
optimumai learn cuda_matmul
optimumai learn pytorch
optimumai learn jax
```

---

## Interactive input & analysis

```bash
optimumai repl                               # interactive session (needs [repl] for arrow keys)
optimumai trace-text "why is the sky blue"   # words → tokens → transformer → next token
optimumai trace-text "hello world" --layers 3 --level researcher
optimumai diff "x**3 + 2*x" --at 3          # symbolic derivative (needs [symbolic])
optimumai compare relu gelu --input "[-2,-1,0,1,2]"
optimumai sweep softmax --values "[0.25,0.5,1,2]"
```

---

## Plots, landscapes & the concept gallery (needs `[viz]`)

```bash
# Matplotlib figures
optimumai plot activation --name gelu --out gelu.png
optimumai plot softmax --out temps.png
optimumai plot attention --text "the cat sat" --out att.png
optimumai plot embeddings --out emb.png
optimumai plot training --out curve.png

# 3-D loss landscapes
optimumai landscape rosenbrock --out land.png
optimumai landscape bowl --kind contour --out bowl.png

# Concept registry — 21+ concepts
optimumai visualize                              # list every concept + its formats
optimumai visualize attention --fmt png --out attn.png
optimumai visualize kmeans --fmt gif --out kmeans.gif
optimumai visualize gradient_descent --fmt gif --out gd.gif

# Animated GIFs
optimumai animate descent --out descent.gif
optimumai animate diffusion --out diffusion.gif
optimumai animate softmax --out softmax.gif
```

---

## Circuits, editor & playgrounds (interactive HTML)

```bash
# Computation graph as a circuit
optimumai circuit "(a*b + c) * f" --vars "a=2,b=-3,c=10,f=-2" --fmt html --out circuit.html
optimumai circuit "(a*b + c) * f" --vars "a=2,b=-3,c=10,f=-2" --fmt dot
optimumai circuit "(a*b + c) * f" --vars "a=2,b=-3,c=10,f=-2" --fmt terminal

# Editable equation ↔ graph in the browser
optimumai editor "a*x^2 + b*x + c"              # → editable_plot.html

# Drag-the-inputs playgrounds (self-contained HTML, no server)
optimumai playground attention                   # hover token, drag temperature slider
optimumai playground kmeans                      # click to add points, Lloyd's iterates
optimumai playground astar                       # draw walls, A* expands the frontier
optimumai playground softmax                     # drag logits, distribution recomputes
optimumai playground backprop                    # drag a/b/c/f, gradients update live
```

---

## Frontier — quantization & GPU kernels

```bash
optimumai quantize "[0.1,-2.3,4.5,3.14]" --bits 4
optimumai learn flash_attention
optimumai learn lora
optimumai learn dpo

optimumai kernel                       # list kernels
optimumai kernel matmul                # tiled matmul + shared-memory tiling
optimumai kernel flash_attention       # fused online-softmax attention
optimumai kernel --backends            # list available backends (numba/cupy/triton)
```

---

## Classical ML — `optimumai ml`

```bash
optimumai ml linreg                    # linear regression (normal equation)
optimumai ml linreg "[[1],[2],[3],[4]]" "[2,4,6,8]"
optimumai ml logreg                    # logistic regression
optimumai ml kmeans                    # k-means clustering (Lloyd's algorithm)
optimumai ml kmeans "[[0,0],[0,1],[9,9],[9,8]]" --k 2
optimumai ml knn                       # k-nearest neighbors
optimumai ml tree                      # decision tree (Gini/entropy)
optimumai ml nb                        # Gaussian naive Bayes
optimumai ml pca                       # principal component analysis
optimumai ml metrics                   # accuracy, F1, MSE, R², ROC-AUC
```

---

## Classical AI search — `optimumai algo`

```bash
optimumai algo bfs                     # BFS/DFS/UCS on a demo graph
optimumai algo astar                   # greedy best-first & A* on a demo grid
optimumai algo minimax                 # minimax + alpha-beta pruning on a demo tree
```

!!! note "`algo` vs `search`"
    `optimumai algo` is classical-AI-search (BFS/A*/minimax).
    `optimumai search <query>` is full-text search over the course.

---

## Reinforcement learning — `optimumai rl`

```bash
optimumai rl mdp                       # value iteration — the Bellman equation
optimumai rl q-learning                # tabular Q-learning / SARSA on a demo gridworld
optimumai rl reinforce                 # REINFORCE policy gradient on a demo bandit
optimumai rl ppo                       # PPO clipped surrogate objective
```

---

## NLP — `optimumai nlp`

```bash
optimumai nlp bpe lowest               # BPE merges on a demo corpus, then tokenize "lowest"
optimumai nlp bpe --merges 12 lowest   # learn more merge rules first
optimumai nlp bpe                      # demo mode (omit word for the training demo)
optimumai nlp tfidf "the cat sat" "the dog sat"
optimumai nlp ngram                    # n-gram LM + add-k smoothing + perplexity
optimumai nlp edit-distance kitten sitting
optimumai nlp word2vec                 # skip-gram word2vec on a tiny corpus
```

---

## Computer vision — `optimumai vision`

```bash
optimumai vision conv                  # 2-D convolution demo
optimumai vision conv "[[1,2],[3,4]]" "[[1,0],[0,-1]]" --stride 1
optimumai vision pool                  # max & average pooling
optimumai vision sobel                 # Sobel edge detection
optimumai vision cnn --level engineer  # tiny CNN forward pass, shapes narrated
```

---

## LLM evaluation — `optimumai eval`

```bash
optimumai eval bleu "the quick brown fox jumps" "the quick brown fox leaps" --max-n 1
optimumai eval bleu                    # demo mode
optimumai eval rouge "the quick brown fox" "the quick brown fox jumps" -n 1
optimumai eval perplexity "[0.5,0.25,0.8]"
optimumai eval calibration             # Expected Calibration Error demo
optimumai eval faithfulness            # hallucination proxy demo
```

!!! warning "Short strings can score BLEU = 0"
    With `--max-n 4`, a short pair may have no 4-gram overlap and correctly score
    `0.0`. Use `--max-n 1` or longer text for a more illustrative score.

---

## Prompt engineering & augmented RNNs

```bash
optimumai prompt zero-shot
optimumai prompt few-shot
optimumai prompt chain-of-thought
optimumai prompt react
optimumai prompt self-consistency
optimumai prompt structured-output

optimumai augrnn attention             # content-based attention as differentiable memory
optimumai augrnn ntm                   # Neural Turing Machine (cosine addressing + erase/add)
optimumai augrnn act                   # Adaptive Computation Time (halting + ponder cost)
```

---

## Token generation

```bash
optimumai providers                              # what's available on this machine
optimumai generate "The math behind attention is"
optimumai generate "Explain softmax" --provider ollama --model llama3.2
optimumai generate "..." --max-tokens 32 --temperature 0.7
```

Providers tried in order — **Ollama** (local, zero keys) → **Hugging Face**
(`HF_TOKEN`) → **Anthropic** (`ANTHROPIC_API_KEY` + `[llm]`) → **toy bigram**
(always works offline).

---

## Notebooks

```bash
optimumai notebooks                    # copy bundled notebooks + launch Jupyter
optimumai notebooks --dir my-notebooks # choose destination directory
optimumai notebooks --no-launch        # copy only, don't launch Jupyter
```

Needs `optimumai[notebooks]` (JupyterLab) to launch; copying works regardless.

---

## Full command index

| Command | Purpose |
|---|---|
| `start` | 30-second guided tour |
| `course`, `learn`, `progress`, `search` | The learning path |
| `quiz`, `review`, `exercise` | Active recall & spaced repetition |
| `dashboard` | Streamlit progress dashboard |
| `ask` | Optional LLM tutor |
| `algebra dot\|cosine\|matmul` | Vectors & matrices |
| `softmax` | Softmax with temperature |
| `attention` | Scaled dot-product attention demo |
| `backprop`, `train` | Autograd & MLP training |
| `jepa`, `superposition` | World models & interpretability |
| `kvcache`, `vram` | Systems calculators |
| `repl`, `trace-text`, `diff`, `compare`, `sweep` | Interactive input & analysis |
| `plot activation\|softmax\|attention\|embeddings\|training` | Matplotlib plots (`[viz]`) |
| `landscape rosenbrock\|bowl` | 3-D loss landscapes (`[viz]`) |
| `visualize` | Any-concept PNG/GIF registry (`[viz]`) |
| `animate descent\|diffusion\|softmax` | Animated GIF export (`[viz]`) |
| `circuit` | Computation graph as circuit (HTML/DOT/terminal) |
| `editor` | Editable equation ↔ graph in browser |
| `playground attention\|kmeans\|astar\|softmax\|backprop` | Interactive HTML circuits |
| `quantize`, `kernel` | Frontier quantization & GPU kernels |
| `ml linreg\|logreg\|kmeans\|knn\|tree\|nb\|pca\|metrics` | Classical ML |
| `algo bfs\|astar\|minimax` | Classical AI search |
| `rl mdp\|q-learning\|reinforce\|ppo` | Reinforcement learning |
| `nlp bpe\|tfidf\|ngram\|edit-distance\|word2vec` | NLP fundamentals |
| `vision conv\|pool\|sobel\|cnn` | Computer vision |
| `eval bleu\|rouge\|perplexity\|calibration\|faithfulness` | LLM evaluation |
| `prompt zero-shot\|few-shot\|chain-of-thought\|react\|self-consistency\|structured-output` | Prompt patterns |
| `augrnn attention\|ntm\|act` | Augmented RNNs |
| `generate`, `providers` | Token generation |
| `notebooks` | Jupyter notebooks |

Run `optimumai --version` to check your installed version, and
`optimumai <command> --help` for any command's full option list.
