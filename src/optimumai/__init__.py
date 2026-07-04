"""OptimumAI — unlock the math behind AI.

Every operation, from a dot product to a full transformer block, can be run with
``explain=True`` to produce a step-by-step computation trace, a terminal
visualization, and the context for *why* AI uses it.

    >>> from optimumai import Vector
    >>> Vector([1, 2, 3]).dot(Vector([4, 5, 6]), explain=True)
    32.0

v0.2 adds the fundamentals behind modern AI: a micrograd-style autograd engine,
calculus, optimizers, neural networks with real backprop, multi-head attention,
transformer blocks, LeCun's JEPA world model, and Anthropic-style superposition.

v0.3 turns it into a learning path: a structured Course, progress tracking, a
Streamlit dashboard, plus embeddings, RAG, diffusion, and an optional LLM tutor.

v0.4 adds the foundations of the stack: tensors & integration, the PyTorch and
JAX programming models, and the systems layer — the CUDA execution & memory
model, tiled matmul kernels, the KV cache, and a VRAM budget calculator.

v0.5 makes it interactive: feed your own input and watch it flow — a REPL, a
text-to-transformer pipeline, op comparisons, parameter sweeps, and symbolic
differentiation of your own equations.

v0.6 adds graphs: matplotlib figures (via ``optimumai.visualization``) for
activation curves, attention heatmaps, embedding scatter, and a 3D loss
landscape with the gradient-descent trajectory carved across it.

v0.7 adds the circuit: render any expression or Value graph as a computation
"circuit" (interactive HTML, Graphviz DOT, or terminal) via ``optimumai.circuit``,
with data and gradients flowing the wires.

v0.8 adds frontier concepts: FlashAttention (IO-aware tiling + online softmax),
quantization (int8/int4), LoRA (parameter-efficient fine-tuning), and DPO
(preference alignment) — how today's large models are actually built and run.

v0.9 makes it a learning product, grounded in cognitive science: a quiz /
active-recall engine (the testing effect), spaced-repetition review (SM-2),
guided onboarding, and course search.

v0.10 is hands-on: GPU kernels from scratch on a pure-Python simulator (write &
grade your own), an editable equation↔graph in the browser, animated GIF export,
and compute-the-answer exercises. Get your hands dirty.

v1.0 is the stable release: real token generation (Ollama / Hugging Face /
Anthropic / a toy fallback), interactive drag-the-inputs circuits, a
visualize-any-concept registry (PNG + GIF), a notebook launcher, and a docs site.

v1.1 broadens from the deep-learning stack to the whole field: classical machine
learning (regression, trees, k-means, KNN, naive Bayes, PCA, metrics), classical
AI search (BFS/DFS/UCS, A*, minimax + alpha-beta), reinforcement learning (MDPs &
the Bellman equation, Q-learning/SARSA, REINFORCE, PPO), NLP fundamentals (BPE,
TF-IDF, n-grams, edit distance, word2vec), computer vision (convolution, pooling,
Sobel edges, a tiny CNN), and LLM evaluation (BLEU/ROUGE, perplexity, calibration,
and a candid hallucination/faithfulness heuristic) — each an explainable trace.

v1.2 makes it interactive & explained: prompt-engineering patterns (zero/few-shot,
chain-of-thought, ReAct, self-consistency, structured output), the augmented-RNN
lineage from distill.pub (attention as differentiable memory, Neural Turing
Machines, Adaptive Computation Time), self-contained interactive HTML playgrounds
(a Transformer-Explainer-style attention widget, plus k-means and A*), and a
per-concept plot/GIF gallery wired into the ``visualize`` registry.

v1.3 makes it *flow*: a **Plot Studio** (feed numbers, get any chart — bar,
histogram, scatter, box, line, pie, violin — plus the exact matplotlib + numpy
code on screen), and a ``flows`` subpackage of distill.pub-style interactive
circuit-flow diagrams (a transformer forward pass, scaled dot-product attention,
TF-IDF, and word2vec) rendered as self-contained, offline HTML.

Public-API stability
---------------------
Everything exported from the top-level ``optimumai`` namespace (the names in
``__all__`` below) is considered **stable** and follows semantic versioning: no
breaking changes without a major-version bump. Submodule internals and any
name prefixed with ``_`` may change between minor releases.
"""

from optimumai.algebra.matrix import Matrix
from optimumai.algebra.vector import Vector
from optimumai.analysis.compare import compare, sweep

# --- v1.2: interactive & explained (prompting · augmented RNNs · playgrounds) ---
from optimumai.augmented_rnns import (
    NTMMemory,
    adaptive_computation_time,
    attention_read,
    ntm_read,
    ntm_write,
)
from optimumai.autograd.value import Value
from optimumai.calculus.derivative import derivative, gradient
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Step, Trace
from optimumai.curriculum import COURSE, Course, Lesson
from optimumai.diffusion.schedule import forward_diffusion
from optimumai.embeddings.lookup import embedding_lookup, nearest_neighbors

# --- v1.1: the classical-AI breadth layer (ML · search · RL · NLP · vision · eval) ---
# Package-level imports; each subpackage __init__ is the stable contract.
# Note: nlp.perplexity(model, corpus) is intentionally NOT re-exported here — the
# generic evaluation.perplexity(logprobs) owns the top-level name; reach the
# n-gram variant via `optimumai.nlp.perplexity`.
from optimumai.evaluation import (
    bleu,
    ece,
    exact_match,
    faithfulness_score,
    perplexity,
    rouge_l,
    rouge_n,
    token_f1,
)
from optimumai.exercises.engine import Workbook

# --- v1.3: see it flow (Plot Studio + interactive concept-flow diagrams) ---
from optimumai.flows import attention_flow, tfidf_flow, transformer_flow, word2vec_flow
from optimumai.foundations.kv_cache import kv_cache_size
from optimumai.foundations.math_foundations import integrate
from optimumai.foundations.vram import vram_estimate
from optimumai.frontier.flash_attention import flash_attention
from optimumai.frontier.lora import lora
from optimumai.frontier.quantization import quantize
from optimumai.frontier.rlhf import dpo
from optimumai.interactive.repl import run_repl
from optimumai.interpretability.superposition import superposition
from optimumai.kernels.exercises import KernelWorkbench
from optimumai.kernels.sim import GpuSim
from optimumai.llm.generate import generate
from optimumai.ml import (
    KNN,
    PCA,
    DecisionTree,
    GaussianNB,
    KMeans,
    LinearRegression,
    LogisticRegression,
)
from optimumai.neural_networks.mlp import MLP
from optimumai.nlp import (
    BPETokenizer,
    NGramModel,
    SkipGramModel,
    TfidfVectorizer,
    edit_distance,
)
from optimumai.optimization.optimizers import SGD, Adam, minimize
from optimumai.probability.softmax import softmax, softmax_trace
from optimumai.progress import ProgressTracker
from optimumai.prompting import (
    chain_of_thought,
    few_shot,
    react,
    self_consistency,
    structured_output,
    zero_shot,
)
from optimumai.quiz.engine import Quiz
from optimumai.rag.pipeline import RAGPipeline
from optimumai.review.scheduler import ReviewScheduler
from optimumai.rl import (
    MDP,
    policy_iteration,
    ppo_clip,
    q_learning,
    reinforce,
    sarsa,
    value_iteration,
)
from optimumai.search import (
    alpha_beta,
    astar,
    bfs,
    dfs,
    greedy_best_first,
    minimax,
    uniform_cost_search,
)
from optimumai.symbolic.differentiate import differentiate
from optimumai.transformers.attention import Attention
from optimumai.transformers.block import TransformerBlock
from optimumai.transformers.multihead import MultiHeadAttention
from optimumai.transformers.positional import positional_encoding
from optimumai.transformers.text_pipeline import TextPipeline
from optimumai.tutor import Tutor
from optimumai.vision import (
    avg_pool2d,
    cnn_forward,
    conv2d,
    max_pool2d,
    sobel_edges,
)
from optimumai.visualization.concepts import render_concept
from optimumai.visualization.interactive import editable_plot
from optimumai.visualization.playgrounds import playground
from optimumai.visualization.plotstudio import (
    describe,
    plot_code,
    plot_data,
    plot_studio_playground,
)
from optimumai.world_models.jepa import JEPA

__version__ = "1.3.0"

__all__ = [
    "COURSE",
    "MLP",
    "SGD",
    "Adam",
    "Attention",
    "Course",
    "ExplainLevel",
    "GpuSim",
    "JEPA",
    "KernelWorkbench",
    "Lesson",
    "Matrix",
    "MultiHeadAttention",
    "ProgressTracker",
    "Quiz",
    "RAGPipeline",
    "ReviewScheduler",
    "Step",
    "Workbook",
    "TextPipeline",
    "dpo",
    "flash_attention",
    "lora",
    "quantize",
    "Trace",
    "TransformerBlock",
    "Tutor",
    "Value",
    "Vector",
    "compare",
    "derivative",
    "differentiate",
    "editable_plot",
    "embedding_lookup",
    "forward_diffusion",
    "generate",
    "gradient",
    "integrate",
    "render_concept",
    "kv_cache_size",
    "minimize",
    "nearest_neighbors",
    "positional_encoding",
    "run_repl",
    "softmax",
    "softmax_trace",
    "superposition",
    "sweep",
    "vram_estimate",
    # v1.1 — classical machine learning
    "LinearRegression",
    "LogisticRegression",
    "KMeans",
    "KNN",
    "DecisionTree",
    "GaussianNB",
    "PCA",
    # v1.1 — classical AI search
    "bfs",
    "dfs",
    "uniform_cost_search",
    "greedy_best_first",
    "astar",
    "minimax",
    "alpha_beta",
    # v1.1 — reinforcement learning
    "MDP",
    "value_iteration",
    "policy_iteration",
    "q_learning",
    "sarsa",
    "reinforce",
    "ppo_clip",
    # v1.1 — NLP
    "BPETokenizer",
    "TfidfVectorizer",
    "NGramModel",
    "edit_distance",
    "SkipGramModel",
    # v1.1 — computer vision
    "conv2d",
    "max_pool2d",
    "avg_pool2d",
    "sobel_edges",
    "cnn_forward",
    # v1.1 — LLM evaluation
    "bleu",
    "rouge_n",
    "rouge_l",
    "exact_match",
    "token_f1",
    "perplexity",
    "ece",
    "faithfulness_score",
    # v1.2 — prompt engineering
    "zero_shot",
    "few_shot",
    "chain_of_thought",
    "react",
    "self_consistency",
    "structured_output",
    # v1.2 — augmented RNNs
    "attention_read",
    "ntm_read",
    "ntm_write",
    "NTMMemory",
    "adaptive_computation_time",
    # v1.2 — interactive playground dispatcher
    "playground",
    # v1.3 — Plot Studio + concept-flow diagrams
    "plot_data",
    "plot_code",
    "describe",
    "plot_studio_playground",
    "transformer_flow",
    "attention_flow",
    "tfidf_flow",
    "word2vec_flow",
    "__version__",
]
