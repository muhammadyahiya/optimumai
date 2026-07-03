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
from optimumai.autograd import Value
from optimumai.calculus.derivative import chain_rule_trace, derivative_trace, gradient_trace
from optimumai.core.trace import Trace
from optimumai.diffusion.schedule import demo as diffusion_demo
from optimumai.embeddings.lookup import demo as embeddings_demo
from optimumai.foundations.cuda_kernel import tiled_matmul_trace
from optimumai.foundations.gpu_foundations import memory_hierarchy_trace, thread_hierarchy_trace
from optimumai.foundations.jax_foundations import demo as jax_demo
from optimumai.foundations.kv_cache import demo as kv_cache_demo
from optimumai.foundations.math_foundations import integrate_demo, tensor_intro_trace
from optimumai.foundations.pytorch_foundations import demo as pytorch_demo
from optimumai.foundations.vram import demo as vram_demo
from optimumai.interpretability.superposition import superposition_trace
from optimumai.neural_networks.backprop import train_demo
from optimumai.optimization.optimizers import descent_demo
from optimumai.probability.softmax import softmax_trace
from optimumai.rag.pipeline import RAGPipeline
from optimumai.transformers.attention import Attention
from optimumai.transformers.block import TransformerBlock
from optimumai.transformers.multihead import MultiHeadAttention
from optimumai.transformers.positional import positional_encoding_trace
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
