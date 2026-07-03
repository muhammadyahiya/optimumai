"""OptimumAI — unlock the math behind AI.

Every operation, from a dot product to a full attention block, can be run with
``explain=True`` to produce a step-by-step computation trace, a terminal
visualization, and the context for *why* AI uses it.

    >>> from optimumai import Vector
    >>> Vector([1, 2, 3]).dot(Vector([4, 5, 6]), explain=True)
    32.0
"""

from optimumai.algebra.matrix import Matrix
from optimumai.algebra.vector import Vector
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Step, Trace
from optimumai.probability.softmax import softmax, softmax_trace
from optimumai.transformers.attention import Attention

__version__ = "0.1.0"

__all__ = [
    "Attention",
    "ExplainLevel",
    "Matrix",
    "Step",
    "Trace",
    "Vector",
    "softmax",
    "softmax_trace",
    "__version__",
]
