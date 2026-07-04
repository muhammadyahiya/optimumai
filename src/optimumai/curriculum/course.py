"""The OptimumAI course — first principles of AI, learnable one step at a time.

A :class:`Course` is an ordered list of :class:`Lesson` objects grouped into
tracks that build on each other: linear algebra → calculus & autograd →
optimization & neural nets → transformers → applied AI → world models &
interpretability → systems & hardware. Each lesson knows how to render its own
runnable :class:`~optimumai.core.trace.Trace`, so "learning" and "running the
math" are the same action.

Pair this with :class:`~optimumai.progress.tracker.ProgressTracker` (and the
Streamlit dashboard) to track how far through the frontier you've come.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from optimumai.algebra import Matrix, Vector
from optimumai.analysis.compare import demo as compare_demo
from optimumai.analysis.compare import sweep_trace

# --- v1.2 demos: prompt engineering + augmented RNNs ---
from optimumai.augmented_rnns.act import demo as act_demo
from optimumai.augmented_rnns.attention import demo as augmented_attention_demo
from optimumai.augmented_rnns.ntm import demo as ntm_demo
from optimumai.autograd import Value
from optimumai.calculus.derivative import chain_rule_trace, derivative_trace, gradient_trace
from optimumai.core.trace import Trace
from optimumai.diffusion.schedule import demo as diffusion_demo
from optimumai.embeddings.lookup import demo as embeddings_demo

# --- v1.1 demos: classical ML, search, RL, NLP, vision, evaluation ---
from optimumai.evaluation.calibration import demo as calibration_demo
from optimumai.evaluation.hallucination import demo as hallucination_demo
from optimumai.evaluation.perplexity import demo as perplexity_demo
from optimumai.evaluation.text_metrics import demo as text_metrics_demo
from optimumai.foundations.cuda_kernel import tiled_matmul_trace
from optimumai.foundations.gpu_foundations import memory_hierarchy_trace, thread_hierarchy_trace
from optimumai.foundations.jax_foundations import demo as jax_demo
from optimumai.foundations.kv_cache import demo as kv_cache_demo
from optimumai.foundations.math_foundations import integrate_demo, tensor_intro_trace
from optimumai.foundations.pytorch_foundations import demo as pytorch_demo
from optimumai.foundations.vram import demo as vram_demo
from optimumai.frontier.flash_attention import demo as flash_attention_demo
from optimumai.frontier.lora import demo as lora_demo
from optimumai.frontier.quantization import demo as quantization_demo
from optimumai.frontier.rlhf import demo as dpo_demo
from optimumai.interpretability.superposition import superposition_trace
from optimumai.kernels.kernels import (
    flash_attention_kernel_trace,
    matmul_trace,
    softmax_rows_trace,
    vector_add_trace,
)
from optimumai.ml.decision_tree import demo as decision_tree_demo
from optimumai.ml.kmeans import demo as kmeans_demo
from optimumai.ml.knn import demo as knn_demo
from optimumai.ml.linear_regression import demo as linear_regression_demo
from optimumai.ml.logistic_regression import demo as logistic_regression_demo
from optimumai.ml.metrics import demo as ml_metrics_demo
from optimumai.ml.naive_bayes import demo as naive_bayes_demo
from optimumai.ml.pca import demo as pca_demo
from optimumai.neural_networks.backprop import train_demo
from optimumai.nlp.bpe import demo as bpe_demo
from optimumai.nlp.edit_distance import demo as edit_distance_demo
from optimumai.nlp.ngram import demo as ngram_demo
from optimumai.nlp.tfidf import demo as tfidf_demo
from optimumai.nlp.word2vec import demo as word2vec_demo
from optimumai.optimization.optimizers import descent_demo
from optimumai.probability.softmax import softmax_trace
from optimumai.prompting.chain_of_thought import demo as cot_demo
from optimumai.prompting.few_shot import demo as few_shot_demo
from optimumai.prompting.react import demo as react_demo
from optimumai.prompting.self_consistency import demo as self_consistency_demo
from optimumai.prompting.structured_output import demo as structured_output_demo
from optimumai.prompting.zero_shot import demo as zero_shot_demo
from optimumai.rag.pipeline import RAGPipeline
from optimumai.rl.mdp import demo as value_iteration_demo
from optimumai.rl.policy_gradient import demo as reinforce_demo
from optimumai.rl.ppo import demo as ppo_demo
from optimumai.rl.q_learning import demo as q_learning_demo
from optimumai.search.adversarial import demo as adversarial_demo
from optimumai.search.informed import demo as informed_demo
from optimumai.search.uninformed import demo as uninformed_demo
from optimumai.transformers.attention import Attention
from optimumai.transformers.block import TransformerBlock
from optimumai.transformers.multihead import MultiHeadAttention
from optimumai.transformers.positional import positional_encoding_trace
from optimumai.transformers.text_pipeline import TextPipeline
from optimumai.vision.cnn import demo as cnn_demo
from optimumai.vision.convolution import demo as conv2d_demo
from optimumai.vision.edges import demo as sobel_demo
from optimumai.vision.pooling import demo as pooling_demo
from optimumai.world_models.jepa import JEPA


def _backprop_demo() -> Trace:
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    loss = ((a * b) + c) * Value(-2.0, label="f")
    loss.label = "L"
    return loss.backward_trace()


@dataclass
class Lesson:
    """A single, runnable step in the course."""

    id: str
    title: str
    track: str
    summary: str
    demo: Callable[[], Trace] | None = None
    prerequisites: tuple[str, ...] = field(default_factory=tuple)

    def run(self, level: str = "intermediate") -> Trace | None:
        """Render this lesson's trace at ``level`` and return it."""
        if self.demo is None:
            return None
        trace = self.demo()
        trace.render(level)
        return trace


# Ordered curriculum. Track order is pedagogical: each builds on the last.
_LESSONS: tuple[Lesson, ...] = (
    # --- Linear Algebra ------------------------------------------------------
    Lesson("dot", "Dot product", "1 · Linear Algebra",
           "Similarity and the atom of every matrix multiply.",
           lambda: Vector([1, 2, 3]).dot_trace(Vector([4, 5, 6]))),
    Lesson("cosine", "Cosine similarity", "1 · Linear Algebra",
           "Angle between vectors — the RAG / semantic-search score.",
           lambda: Vector([1, 2, 3]).cosine_similarity_trace(Vector([2, 4, 6])),
           ("dot",)),
    Lesson("matmul", "Matrix multiplication", "1 · Linear Algebra",
           "Every dense layer, computed cell by cell.",
           lambda: Matrix([[1, 2], [3, 4]]).matmul_trace(Matrix([[5, 6], [7, 8]])),
           ("dot",)),
    # --- Calculus & Autograd -------------------------------------------------
    Lesson("derivative", "Derivatives", "2 · Calculus & Autograd",
           "The slope that tells a network which way to move.",
           lambda: derivative_trace(lambda x: x**3, 2.0, label="x³")),
    Lesson("gradient", "Gradients", "2 · Calculus & Autograd",
           "The vector of partial derivatives.",
           lambda: gradient_trace(lambda p: p[0] ** 2 + p[1] ** 2, [3.0, 4.0]),
           ("derivative",)),
    Lesson("chain_rule", "The chain rule", "2 · Calculus & Autograd",
           "The single idea behind all of backpropagation.",
           chain_rule_trace, ("derivative",)),
    Lesson("backprop", "Backpropagation", "2 · Calculus & Autograd",
           "Reverse-mode autodiff on a scalar graph (micrograd).",
           _backprop_demo, ("chain_rule",)),
    # --- Optimization & Neural Nets -----------------------------------------
    Lesson("descent", "Gradient descent", "3 · Optimization & Neural Nets",
           "Adam walking a loss bowl to its minimum.",
           lambda: descent_demo("adam", steps=60), ("gradient",)),
    Lesson("train", "Training an MLP", "3 · Optimization & Neural Nets",
           "The full loop: predict → loss → backprop → step.",
           lambda: train_demo(steps=120), ("backprop", "descent")),
    # --- Probability & Transformers -----------------------------------------
    Lesson("softmax", "Softmax", "4 · Probability & Transformers",
           "Logits into a probability distribution.",
           lambda: softmax_trace([2.0, 1.0, 0.1])),
    Lesson("attention", "Scaled dot-product attention", "4 · Probability & Transformers",
           "softmax(QKᵀ/√dₖ)·V — the transformer core.",
           Attention.demo, ("matmul", "softmax")),
    Lesson("multihead", "Multi-head attention", "4 · Probability & Transformers",
           "Parallel heads + a causal mask (nanoGPT).",
           MultiHeadAttention.demo, ("attention",)),
    Lesson("positional", "Positional encoding", "4 · Probability & Transformers",
           "Injecting word order into permutation-invariant attention.",
           lambda: positional_encoding_trace(6, 8), ("attention",)),
    Lesson("transformer", "The transformer block", "4 · Probability & Transformers",
           "LayerNorm → attention → FFN, with residuals.",
           TransformerBlock.demo, ("multihead", "positional")),
    # --- Applied AI ----------------------------------------------------------
    Lesson("embeddings", "Embeddings", "5 · Applied AI",
           "Turning discrete tokens into learnable dense vectors.",
           embeddings_demo, ("cosine",)),
    Lesson("rag", "Retrieval-augmented generation", "5 · Applied AI",
           "Query → embed → cosine search → top-k → prompt.",
           RAGPipeline.demo, ("embeddings",)),
    Lesson("diffusion", "Diffusion", "5 · Applied AI",
           "Forward noising schedule and the reverse denoising idea.",
           diffusion_demo, ("softmax",)),
    # --- World Models & Interpretability ------------------------------------
    Lesson("jepa", "JEPA world models", "6 · World Models & Interpretability",
           "LeCun: predict in latent space, not pixels (energy-based).",
           JEPA.demo, ("cosine",)),
    Lesson("superposition", "Superposition", "6 · World Models & Interpretability",
           "Anthropic: why single neurons are polysemantic.",
           lambda: superposition_trace(5, 2), ("matmul",)),
    # --- Math Foundations ----------------------------------------------------
    Lesson("tensors", "Tensors", "7 · Math Foundations",
           "Scalars → vectors → matrices → n-D arrays, shapes, broadcasting.",
           tensor_intro_trace),
    Lesson("integration", "Numerical integration", "7 · Math Foundations",
           "Trapezoid & Monte Carlo — every expectation is an integral.",
           integrate_demo, ("derivative",)),
    # --- Framework Internals -------------------------------------------------
    Lesson("pytorch", "PyTorch autograd", "8 · Framework Internals",
           "What torch.autograd does under the hood — it's the Value engine.",
           pytorch_demo, ("backprop",)),
    Lesson("jax", "JAX transforms", "8 · Framework Internals",
           "grad / jit / vmap / pytrees — transforming pure functions.",
           jax_demo, ("backprop",)),
    # --- Systems & Hardware --------------------------------------------------
    Lesson("gpu_threads", "GPU thread hierarchy", "9 · Systems & Hardware",
           "Grid → block → warp → thread; the SIMT execution model.",
           thread_hierarchy_trace),
    Lesson("memory_hierarchy", "GPU memory hierarchy", "9 · Systems & Hardware",
           "Registers → shared → global; why kernels are memory-bound.",
           memory_hierarchy_trace),
    Lesson("cuda_matmul", "Tiled matmul kernel", "9 · Systems & Hardware",
           "Naive vs tiled + coalescing — the canonical GPU optimization.",
           tiled_matmul_trace, ("matmul",)),
    Lesson("kv_cache", "The KV cache", "9 · Systems & Hardware",
           "Why context length eats VRAM; MHA vs GQA vs MQA.",
           kv_cache_demo, ("attention",)),
    Lesson("vram", "VRAM budget", "9 · Systems & Hardware",
           "Weights + gradients + optimizer states + activations + KV cache.",
           vram_demo, ("train",)),
    # --- Interactive Playground ---------------------------------------------
    Lesson("trace_text", "Text → transformer", "10 · Interactive Playground",
           "Watch your own words flow to a next-token distribution.",
           TextPipeline.demo, ("transformer",)),
    Lesson("compare", "Compare activations", "10 · Interactive Playground",
           "ReLU vs GELU side by side — how the activation shapes gradients.",
           compare_demo, ("train",)),
    Lesson("sweep", "Temperature sweep", "10 · Interactive Playground",
           "How softmax sharpens (T→0) or flattens (T→∞).",
           lambda: sweep_trace("softmax", "temperature", [0.25, 0.5, 1.0, 2.0]), ("softmax",)),
    # --- Frontier -----------------------------------------------------------
    Lesson("flash_attention", "FlashAttention", "11 · Frontier",
           "IO-aware tiling + online softmax — exact attention, linear memory.",
           flash_attention_demo, ("multihead",)),
    Lesson("quantization", "Quantization", "11 · Frontier",
           "int8/int4 with scale + zero-point — how models shrink to fit.",
           quantization_demo, ("matmul",)),
    Lesson("lora", "LoRA", "11 · Frontier",
           "Low-rank adapters (W₀ + BA) — fine-tune ~10,000× fewer parameters.",
           lora_demo, ("matmul",)),
    Lesson("dpo", "DPO / RLHF", "11 · Frontier",
           "Align to human preferences without a reward model or RL.",
           dpo_demo, ("softmax",)),
    # --- GPU Kernels from scratch -------------------------------------------
    Lesson("kernel_vector_add", "Kernel: vector add", "12 · GPU Kernels",
           "One thread per element — the GPU 'hello world', run on a simulator.",
           vector_add_trace),
    Lesson("kernel_matmul", "Kernel: tiled matmul", "12 · GPU Kernels",
           "A thread per output cell + the shared-memory tiling win.",
           matmul_trace, ("matmul",)),
    Lesson("kernel_softmax", "Kernel: softmax", "12 · GPU Kernels",
           "One thread per row, with the numerically-stable max trick.",
           softmax_rows_trace, ("softmax",)),
    Lesson("kernel_flash", "Kernel: flash attention", "12 · GPU Kernels",
           "Fused online-softmax attention — exact, no N×N matrix in VRAM.",
           flash_attention_kernel_trace, ("flash_attention",)),
    # --- Classical Machine Learning -----------------------------------------
    Lesson("linear_regression", "Linear regression", "13 · Classical ML",
           "OLS via the normal equation θ=(XᵀX)⁻¹Xᵀy — the calculus behind the fit.",
           linear_regression_demo, ("matmul",)),
    Lesson("logistic_regression", "Logistic regression", "13 · Classical ML",
           "Sigmoid + cross-entropy + gradient descent — the atomic classifier.",
           logistic_regression_demo, ("linear_regression", "gradient")),
    Lesson("kmeans", "k-means clustering", "13 · Classical ML",
           "Lloyd's algorithm: assign to the nearest centroid, re-center, repeat.",
           kmeans_demo),
    Lesson("knn", "k-nearest neighbors", "13 · Classical ML",
           "Classify by majority vote of the closest points — no training phase.",
           knn_demo),
    Lesson("decision_tree", "Decision trees", "13 · Classical ML",
           "Best split by Gini / entropy information gain — the atom of forests.",
           decision_tree_demo),
    Lesson("naive_bayes", "Gaussian Naive Bayes", "13 · Classical ML",
           "Bayes' rule + a conditional-independence assumption — the text baseline.",
           naive_bayes_demo),
    Lesson("pca", "Principal component analysis", "13 · Classical ML",
           "Covariance eigendecomposition — compress data to its top-variance axes.",
           pca_demo, ("matmul",)),
    Lesson("ml_metrics", "Classification & regression metrics", "13 · Classical ML",
           "Accuracy, precision/recall/F1, confusion matrix, MSE, R², ROC-AUC.",
           ml_metrics_demo),
    # --- Classical AI Search ------------------------------------------------
    Lesson("uninformed_search", "BFS, DFS & uniform-cost search", "14 · Classical AI Search",
           "Frontier order is everything: FIFO (BFS), LIFO (DFS), cheapest-g (UCS).",
           uninformed_demo),
    Lesson("informed_search", "Greedy best-first & A*", "14 · Classical AI Search",
           "Add a heuristic h(n): A* orders by f=g+h and is optimal when h is admissible.",
           informed_demo, ("uninformed_search",)),
    Lesson("adversarial_search", "Minimax & alpha-beta pruning", "14 · Classical AI Search",
           "Optimal play on a game tree; α-β prunes provably-irrelevant branches.",
           adversarial_demo),
    # --- Reinforcement Learning ---------------------------------------------
    Lesson("value_iteration", "Value iteration (Bellman)", "15 · Reinforcement Learning",
           "Solve a known MDP exactly via Bellman backups until the values converge.",
           value_iteration_demo),
    Lesson("q_learning", "Q-learning & SARSA", "15 · Reinforcement Learning",
           "Learn to act from experience alone — off-policy vs on-policy TD updates.",
           q_learning_demo, ("value_iteration",)),
    Lesson("reinforce", "Policy gradients (REINFORCE)", "15 · Reinforcement Learning",
           "Differentiate through sampled actions with the score-function trick.",
           reinforce_demo, ("q_learning", "softmax")),
    Lesson("ppo", "PPO — the clipped surrogate objective", "15 · Reinforcement Learning",
           "The stabilized policy gradient behind RLHF's RL stage; contrast with DPO.",
           ppo_demo, ("reinforce",)),
    # --- NLP -----------------------------------------------------------------
    Lesson("bpe", "Byte-pair encoding", "16 · NLP",
           "Learn merges from a corpus, then tokenize a word the way an LLM does.",
           bpe_demo),
    Lesson("tfidf", "TF-IDF", "16 · NLP",
           "Weight words by how much they distinguish one document from the rest.",
           tfidf_demo),
    Lesson("ngram", "N-gram language models", "16 · NLP",
           "Count your way to 'what comes next,' with add-k smoothing + perplexity.",
           ngram_demo, ("tfidf",)),
    Lesson("edit_distance", "Edit distance", "16 · NLP",
           "Levenshtein DP — the yardstick behind spell-check and fuzzy match.",
           edit_distance_demo),
    Lesson("word2vec", "Skip-gram word2vec", "16 · NLP",
           "Predict-the-context SGD — the seed idea behind every learned embedding.",
           word2vec_demo, ("embeddings", "softmax")),
    # --- Computer Vision -----------------------------------------------------
    Lesson("conv2d", "2D convolution", "17 · Computer Vision",
           "Slide a kernel over an image to detect local patterns.",
           conv2d_demo, ("matmul",)),
    Lesson("pooling", "Max & average pooling", "17 · Computer Vision",
           "Downsample a feature map for translation tolerance.",
           pooling_demo, ("conv2d",)),
    Lesson("sobel_edges", "Sobel edge detection", "17 · Computer Vision",
           "Two fixed convolutions that find image boundaries.",
           sobel_demo, ("conv2d",)),
    Lesson("tiny_cnn", "A tiny CNN forward pass", "17 · Computer Vision",
           "Watch the tensor shapes flow: conv → relu → pool → dense → softmax.",
           cnn_demo, ("conv2d", "pooling", "softmax")),
    # --- LLM Evaluation ------------------------------------------------------
    Lesson("bleu_rouge", "Text overlap metrics (BLEU / ROUGE / F1)", "18 · LLM Evaluation",
           "Score generated text against a reference — and see why paraphrase breaks it.",
           text_metrics_demo),
    Lesson("perplexity", "Perplexity", "18 · LLM Evaluation",
           "Cross-entropy of a token sequence, exponentiated into a branching factor.",
           perplexity_demo, ("softmax",)),
    Lesson("calibration", "Calibration (ECE)", "18 · LLM Evaluation",
           "Bin predictions by confidence and check it matches empirical accuracy.",
           calibration_demo, ("softmax",)),
    Lesson("hallucination", "Hallucination / faithfulness heuristic", "18 · LLM Evaluation",
           "A claim-overlap grounding proxy — plus why real detection is unsolved.",
           hallucination_demo, ("rag",)),
    # --- Prompt Engineering --------------------------------------------------
    Lesson("zero_shot", "Zero-shot prompting", "19 · Prompt Engineering",
           "Role + instruction + task, with no worked examples.",
           zero_shot_demo),
    Lesson("few_shot", "Few-shot prompting", "19 · Prompt Engineering",
           "In-context learning from K labeled exemplars.",
           few_shot_demo, ("zero_shot",)),
    Lesson("chain_of_thought", "Chain-of-thought prompting", "19 · Prompt Engineering",
           "Elicit intermediate reasoning before the final answer.",
           cot_demo, ("few_shot",)),
    Lesson("react", "ReAct prompting", "19 · Prompt Engineering",
           "Interleave Thought / Action / Observation with a tool.",
           react_demo, ("chain_of_thought",)),
    Lesson("self_consistency", "Self-consistency", "19 · Prompt Engineering",
           "Sample N chain-of-thought paths and majority-vote the answer.",
           self_consistency_demo, ("chain_of_thought",)),
    Lesson("structured_output", "Structured output", "19 · Prompt Engineering",
           "Constrain generation to a validated JSON schema.",
           structured_output_demo, ("zero_shot",)),
    # --- Augmented RNNs (distill.pub) ---------------------------------------
    Lesson("augmented_attention", "Attention as differentiable memory", "20 · Augmented RNNs",
           "Content-based soft attention as a memory read: score → softmax → blend.",
           augmented_attention_demo, ("softmax",)),
    Lesson("ntm", "Neural Turing Machines", "20 · Augmented RNNs",
           "External read/write memory addressed by cosine similarity — erase then add.",
           ntm_demo, ("augmented_attention",)),
    Lesson("act", "Adaptive Computation Time", "20 · Augmented RNNs",
           "Learned, variable compute per input via a halting probability + ponder cost.",
           act_demo, ("softmax",)),
)


class Course:
    """Navigable view over the ordered curriculum."""

    def __init__(self, lessons: tuple[Lesson, ...] = _LESSONS):
        self.lessons = lessons
        self._by_id = {lesson.id: lesson for lesson in lessons}

    def __len__(self) -> int:
        return len(self.lessons)

    def __iter__(self):
        return iter(self.lessons)

    def ids(self) -> list[str]:
        return [lesson.id for lesson in self.lessons]

    def get(self, lesson_id: str) -> Lesson:
        if lesson_id not in self._by_id:
            raise KeyError(lesson_id)
        return self._by_id[lesson_id]

    def tracks(self) -> OrderedDict[str, list[Lesson]]:
        """Lessons grouped by track, preserving order."""
        grouped: OrderedDict[str, list[Lesson]] = OrderedDict()
        for lesson in self.lessons:
            grouped.setdefault(lesson.track, []).append(lesson)
        return grouped

    def next_incomplete(self, completed: set[str]) -> Lesson | None:
        """The first lesson (in course order) not yet in ``completed``."""
        for lesson in self.lessons:
            if lesson.id not in completed:
                return lesson
        return None


# The default, shared course instance.
COURSE = Course()
