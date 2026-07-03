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
"""

from optimumai.algebra.matrix import Matrix
from optimumai.algebra.vector import Vector
from optimumai.analysis.compare import compare, sweep
from optimumai.autograd.value import Value
from optimumai.calculus.derivative import derivative, gradient
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Step, Trace
from optimumai.curriculum import COURSE, Course, Lesson
from optimumai.diffusion.schedule import forward_diffusion
from optimumai.embeddings.lookup import embedding_lookup, nearest_neighbors
from optimumai.foundations.kv_cache import kv_cache_size
from optimumai.foundations.math_foundations import integrate
from optimumai.foundations.vram import vram_estimate
from optimumai.frontier.flash_attention import flash_attention
from optimumai.frontier.lora import lora
from optimumai.frontier.quantization import quantize
from optimumai.frontier.rlhf import dpo
from optimumai.interactive.repl import run_repl
from optimumai.interpretability.superposition import superposition
from optimumai.neural_networks.mlp import MLP
from optimumai.optimization.optimizers import SGD, Adam, minimize
from optimumai.probability.softmax import softmax, softmax_trace
from optimumai.progress import ProgressTracker
from optimumai.rag.pipeline import RAGPipeline
from optimumai.symbolic.differentiate import differentiate
from optimumai.transformers.attention import Attention
from optimumai.transformers.block import TransformerBlock
from optimumai.transformers.multihead import MultiHeadAttention
from optimumai.transformers.positional import positional_encoding
from optimumai.transformers.text_pipeline import TextPipeline
from optimumai.tutor import Tutor
from optimumai.world_models.jepa import JEPA

__version__ = "0.8.0"

__all__ = [
    "COURSE",
    "MLP",
    "SGD",
    "Adam",
    "Attention",
    "Course",
    "ExplainLevel",
    "JEPA",
    "Lesson",
    "Matrix",
    "MultiHeadAttention",
    "ProgressTracker",
    "RAGPipeline",
    "Step",
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
    "embedding_lookup",
    "forward_diffusion",
    "gradient",
    "integrate",
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
    "__version__",
]
