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
"""

from optimumai.algebra.matrix import Matrix
from optimumai.algebra.vector import Vector
from optimumai.autograd.value import Value
from optimumai.calculus.derivative import derivative, gradient
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Step, Trace
from optimumai.interpretability.superposition import superposition
from optimumai.neural_networks.mlp import MLP
from optimumai.optimization.optimizers import SGD, Adam, minimize
from optimumai.probability.softmax import softmax, softmax_trace
from optimumai.transformers.attention import Attention
from optimumai.transformers.block import TransformerBlock
from optimumai.transformers.multihead import MultiHeadAttention
from optimumai.transformers.positional import positional_encoding
from optimumai.world_models.jepa import JEPA

__version__ = "0.2.0"

__all__ = [
    "MLP",
    "SGD",
    "Adam",
    "Attention",
    "ExplainLevel",
    "JEPA",
    "Matrix",
    "MultiHeadAttention",
    "Step",
    "Trace",
    "TransformerBlock",
    "Value",
    "Vector",
    "derivative",
    "gradient",
    "minimize",
    "positional_encoding",
    "softmax",
    "softmax_trace",
    "superposition",
    "__version__",
]
