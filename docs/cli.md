# CLI reference

The `optimumai` command wraps every capability in the library. Run
`optimumai --help` or `optimumai <command> --help` any time — this page is a
complete, organized map of what's there.

## `--level` and the `explain=True` philosophy

Every op in OptimumAI computes the real answer *and* can narrate itself. In
Python that's the `explain=True` keyword; on the CLI, explanation is always on
— every command prints a step-by-step [`Trace`](features.md). What changes is
how much detail you see, via `--level`:

| Level | Adds |
|---|---|
| `beginner` | The steps and a plain-English "why" |
| `intermediate` | Per-step detail notes (the default for most commands) |
| `engineer` | Intermediate values + algorithmic complexity |
| `researcher` | Everything |

```bash
optimumai algebra dot "[1,2,3]" "[4,5,6]" --level beginner
optimumai algebra dot "[1,2,3]" "[4,5,6]" --level researcher
```

The same enum is `optimumai.core.explain.ExplainLevel` in Python
(`level="engineer"` on any op). Fast mode and teaching mode are *the same code
path* — there's no separate "explained" implementation to drift out of sync.

!!! tip "Where a command takes optional positional args"
    Many commands (`algebra dot`, `ml linreg`, `nlp bpe`, `vision conv`, ...)
    run a built-in demo if you omit the arguments, or operate on your own
    numbers/text if you supply them. Try both.

!!! warning "`a|b|c` in help text is not a shell command"
    Click's own `--help` output sometimes lists an argument's choices as
    `attention|ntm|act` for readability. That is **not** something you type
    literally — it means "pick one." Every example on this page shows a single,
    copy-safe command line (e.g. `optimumai augrnn ntm`), never the piped form.

---

## The course, progress & retention

```bash
optimumai start                 # 30-second guided tour (new here? start here)
optimumai course                # the full path, grouped by track, with ✓/○ progress
optimumai learn                 # list every topic (76 lessons across 20 tracks)
optimumai learn dot              # run a lesson (auto-marks it complete)
optimumai learn transformer --level researcher
optimumai learn attention --no-track   # run without recording completion
optimumai progress              # a progress bar + what's next
optimumai progress --reset      # clear all recorded progress
optimumai search attention      # find lessons by keyword (id/title/summary/track)
optimumai quiz                  # list quizzes (20 topics, 57 questions total)
optimumai quiz softmax          # active recall — answer, get graded + explained
optimumai review                # spaced repetition (SM-2): whatever's due
optimumai exercise               # list exercise topics
optimumai exercise backprop      # compute-the-answer exercise, tolerance-graded
optimumai dashboard              # Streamlit progress dashboard (needs [dashboard])
optimumai dashboard --port 8888
optimumai ask "why LayerNorm after attention?"   # optional LLM tutor (needs [llm])
```

Progress persists to `~/.optimumai/progress.json` (override with the
`OPTIMUMAI_PROGRESS_PATH` environment variable) and is shared between the CLI
and the dashboard. Quiz scores automatically feed the spaced-repetition
scheduler, so `optimumai review` always surfaces whatever you're most likely
to have forgotten.

---

## Algebra & probability

```bash
optimumai algebra dot "[1,2,3]" "[4,5,6]"
optimumai algebra dot -i                        # type the vectors at a prompt
optimumai algebra cosine "[1,2,3]" "[2,4,6]"
optimumai algebra matmul "[[1,2],[3,4]]" "[[5,6],[7,8]]"
optimumai softmax "[2,1,0.1]" --temperature 0.5
optimumai softmax -i                            # type the logits at a prompt
```

`-i`/`--interactive` (or simply omitting the arguments) prompts you for the
vector/matrix instead of parsing a CLI argument.

---

## Autograd, training & world models

```bash
optimumai backprop                    # chain rule through a scalar graph
optimumai train --steps 150 --lr 0.05 # train a tiny MLP, watch loss fall
optimumai attention --demo            # scaled dot-product attention
optimumai attention --demo --seed 1 --level researcher
optimumai jepa --demo                 # LeCun's world-model energy
optimumai superposition               # Anthropic's polysemantic neurons
optimumai superposition --features 8 --neurons 3
```

---

## Systems & foundations

```bash
optimumai kvcache --seq-len 8192                  # KV-cache VRAM for a config
optimumai kvcache --heads 32 --kv-heads 4          # GQA: fewer KV heads than Q heads
optimumai vram --params 70                        # VRAM to train a 70B model
optimumai vram --params 7 --inference             # inference instead of training
optimumai learn tensors
optimumai learn cuda_matmul
optimumai learn pytorch
optimumai learn jax
```

---

## Interactive input & analysis

```bash
optimumai repl                              # interactive session ([repl] extra for arrow keys)
optimumai trace-text "why is the sky blue"  # your words → tokens → transformer → next token
optimumai trace-text "hello world" --layers 3 --level researcher
optimumai diff "x**3 + 2*x" --at 3          # symbolic derivative (needs [symbolic])
optimumai compare relu gelu --input "[-2,-1,0,1,2]"
optimumai sweep softmax --values "[0.25,0.5,1,2]"
```

---

## Plots, landscapes & the concept gallery (needs `[viz]`)

```bash
optimumai plot activation --name gelu --out gelu.png
optimumai plot softmax --out temps.png
optimumai plot attention --text "the cat sat" --out att.png
optimumai plot embeddings --out emb.png
optimumai plot training --out curve.png
optimumai landscape rosenbrock --out land.png
optimumai landscape bowl --kind contour --out bowl.png
optimumai visualize                              # list every concept + its formats
optimumai visualize attention --fmt png --out attn.png
optimumai visualize kmeans --fmt gif --out kmeans.gif
optimumai animate descent --out descent.gif
optimumai animate diffusion --out diffusion.gif
optimumai animate softmax --out softmax.gif
```

`optimumai visualize` lists every registered concept and which formats it
supports (21 concepts as of v1.2 — the original 14 static/animated concepts
plus 7 more for the classical ML/AI packages: `kmeans`, `decision_boundary`,
`astar`, `value_iteration`, `conv2d`, `calibration`, `ppo_clip`).

---

## Circuits & interactive playgrounds

```bash
optimumai circuit "(a*b + c) * f" --vars "a=2,b=-3,c=10,f=-2"
optimumai circuit "(a*b + c) * f" --vars "a=2,b=-3,c=10,f=-2" --fmt html --out circuit.html
optimumai circuit "a*b + c" --fmt dot
optimumai editor "a*x^2 + b*x + c"          # editable equation ↔ graph HTML
optimumai playground softmax                 # drag the logits, watch probs move
optimumai playground backprop                # drag a/b/c/f, watch gradients update
optimumai playground attention               # hover a query token, drag temperature
optimumai playground kmeans                  # click to add points, watch Lloyd's iterate
optimumai playground astar                   # draw walls, watch the frontier expand
```

Every `playground`/`editor`/`circuit --fmt html` command writes a
self-contained `.html` file (inline vanilla JS or vis-network via CDN) — open
it in any browser, no server or build step needed.

---

## Frontier: quantization, GPU kernels

```bash
optimumai quantize "[0.1,-2.3,4.5,3.14]" --bits 4
optimumai quantize "[0.1,-2.3,4.5,3.14]" --bits 8 --scheme asymmetric
optimumai kernel                            # list kernels
optimumai kernel matmul                     # tiled matmul + the tiling win
optimumai kernel flash_attention            # fused online-softmax attention
optimumai kernel --backends                 # numba/cupy/triton auto-detection
```

---

## Classical ML — `optimumai ml`

```bash
optimumai ml linreg "[[1],[2],[3],[4]]" "[2,4,6,8]"    # OLS via the normal equation
optimumai ml linreg                                     # or omit args for the demo
optimumai ml kmeans "[[0,0],[0,1],[9,9],[9,8]]" --k 2   # Lloyd's algorithm
optimumai ml logreg                                     # sigmoid + cross-entropy + GD
optimumai ml knn                                        # k-nearest-neighbors demo
optimumai ml tree                                       # Gini/entropy split search
optimumai ml nb                                         # Gaussian Naive Bayes
optimumai ml pca                                        # covariance eigendecomposition
optimumai ml metrics                                    # accuracy, F1, MSE, R², ROC-AUC
```

## Classical AI search — `optimumai algo`

!!! note "`algo` vs `search`"
    `optimumai algo` is the classical-AI-search command group (BFS/A\*/minimax).
    `optimumai search <query>` (no group) is unrelated — it's full-text search
    over the course.

```bash
optimumai algo bfs        # uninformed search (BFS/DFS/UCS) on a demo graph
optimumai algo astar      # greedy best-first & A* on a demo grid
optimumai algo minimax    # minimax + alpha-beta pruning on a demo tree
```

## Reinforcement learning — `optimumai rl`

```bash
optimumai rl mdp          # value iteration — the Bellman equation in action
optimumai rl q-learning   # tabular Q-learning / SARSA on a demo gridworld
optimumai rl reinforce    # policy-gradient REINFORCE on a demo bandit
optimumai rl ppo          # the PPO clipped surrogate objective
```

## NLP — `optimumai nlp`

```bash
optimumai nlp bpe lowest              # BPE merges, then tokenize "lowest"
optimumai nlp bpe --merges 12 lowest  # learn more merge rules first
optimumai nlp bpe                     # omit the word for the demo
optimumai nlp tfidf "the cat sat" "the dog sat"
optimumai nlp ngram                   # n-gram LM + add-k smoothing + perplexity
optimumai nlp edit-distance kitten sitting
optimumai nlp word2vec                # skip-gram, one SGD step on a tiny corpus
```

## Computer vision — `optimumai vision`

```bash
optimumai vision conv                 # 2D convolution demo
optimumai vision conv "[[1,2],[3,4]]" "[[1,0],[0,-1]]" --stride 1
optimumai vision pool                 # max & average pooling demo
optimumai vision sobel                # Sobel edge detection demo
optimumai vision cnn --level engineer # tiny CNN forward pass, shapes narrated
```

## LLM evaluation — `optimumai eval`

```bash
optimumai eval bleu "the quick brown fox jumps" "the quick brown fox leaps" --max-n 1
optimumai eval bleu                                    # omit args for the demo
optimumai eval rouge "the quick brown fox" "the quick brown fox jumps" -n 1
optimumai eval perplexity "[0.5,0.25,0.8]"
optimumai eval calibration                             # Expected Calibration Error
optimumai eval faithfulness                            # a candid hallucination proxy
```

!!! warning "Short strings can score BLEU = 0"
    With the default `--max-n 4`, a short candidate/reference pair may have no
    overlapping 4-grams and correctly score `0.0` (verified: `optimumai eval
    bleu "a quick brown fox" "the quick brown fox"` → `0`). Use `--max-n 1` or
    longer text to see a more illustrative score.

---

## Prompt engineering & augmented RNNs

```bash
optimumai prompt zero-shot
optimumai prompt few-shot
optimumai prompt chain-of-thought
optimumai prompt react
optimumai prompt self-consistency
optimumai prompt structured-output
optimumai augrnn attention   # content-based attention as differentiable memory
optimumai augrnn ntm         # Neural Turing Machine head (cosine addressing + erase/add)
optimumai augrnn act         # Adaptive Computation Time (halting + ponder cost)
```

---

## Token generation

```bash
optimumai providers                                   # what's available on this machine
optimumai generate "The math behind attention is"     # real tokens, streamed live
optimumai generate "Explain softmax" --provider ollama --model llama3.2
optimumai generate "..." --max-tokens 32 --temperature 0.7
```

Providers are tried in order — local **Ollama** (auto-detected, zero keys),
then **Hugging Face** (`HF_TOKEN`), then **Anthropic** (via the tutor, needs
`optimumai[llm]` + `ANTHROPIC_API_KEY`), then a built-in **toy** bigram sampler
that always produces tokens so a demo never hard-fails offline.

---

## Notebooks

```bash
optimumai notebooks                          # copies bundled notebooks + launches Jupyter
optimumai notebooks --dir my-notebooks       # choose the destination directory
optimumai notebooks --no-launch              # copy only, don't launch Jupyter
```

Needs `optimumai[notebooks]` (JupyterLab) to actually launch; copying works
regardless.

---

## Full command index

| Command | Purpose |
|---|---|
| `course`, `learn`, `progress`, `search`, `start` | The learning path |
| `quiz`, `review`, `exercise` | Active recall & spaced repetition |
| `dashboard` | Streamlit visual dashboard |
| `ask` | Optional LLM tutor |
| `algebra dot\|cosine\|matmul` | Vectors & matrices |
| `softmax` | Softmax with temperature |
| `attention` | Scaled dot-product attention demo |
| `backprop`, `train` | Autograd & MLP training |
| `jepa`, `superposition` | World models & interpretability |
| `kvcache`, `vram` | Systems calculators |
| `repl`, `trace-text`, `diff`, `compare`, `sweep` | Interactive input & analysis |
| `plot`, `landscape` | Matplotlib graphs (`[viz]`) |
| `visualize` | Any-concept PNG/GIF registry (`[viz]`) |
| `circuit`, `editor`, `playground` | Circuits & interactive HTML |
| `quantize`, `kernel` | Frontier quantization & GPU kernels |
| `animate` | Animated GIF export (`[viz]`) |
| `ml linreg\|kmeans\|logreg\|knn\|tree\|nb\|pca\|metrics` | Classical ML |
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
