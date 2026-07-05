# API Reference

Auto-generated from the docstrings of the public `optimumai` package.

## Top-level exports

Everything in `optimumai.__all__` is the public API and follows
[semantic versioning](stability.md):

```python
# Linear algebra
from optimumai import Vector, Matrix

# Probability & calculus
from optimumai import softmax, derivative, gradient, integrate

# Autograd
from optimumai import Value

# Neural networks
from optimumai import MLP, Adam, minimize

# Transformers
from optimumai import Attention, MultiHeadAttention, TransformerBlock, TextPipeline

# World models & interpretability
from optimumai import JEPA, superposition

# Embeddings, RAG, diffusion
from optimumai import embedding_lookup, nearest_neighbors, RAGPipeline, forward_diffusion

# Systems
from optimumai import kv_cache_size, vram_estimate

# Generation & tutor
from optimumai import generate

# Course & progress
from optimumai import COURSE, ProgressTracker, Quiz, ReviewScheduler

# Visualization
from optimumai import render_concept, editable_plot

# Kernels
from optimumai import KernelWorkbench, GpuSim

# Trace types
from optimumai import Trace, ExplainLevel, Lesson, Course, Workbook
```

---

## Module reference

::: optimumai
    options:
      show_root_heading: true
      members_order: source

---

## Submodule reference

::: optimumai.algebra
::: optimumai.probability
::: optimumai.calculus
::: optimumai.autograd
::: optimumai.optimization
::: optimumai.neural_networks
::: optimumai.transformers
::: optimumai.world_models
::: optimumai.interpretability
::: optimumai.frontier
::: optimumai.embeddings
::: optimumai.rag
::: optimumai.diffusion
::: optimumai.foundations
::: optimumai.kernels
::: optimumai.ml
::: optimumai.search
::: optimumai.rl
::: optimumai.nlp
::: optimumai.vision
::: optimumai.evaluation
::: optimumai.prompting
::: optimumai.augmented_rnns
::: optimumai.curriculum
::: optimumai.quiz
::: optimumai.review
::: optimumai.llm
::: optimumai.visualization
::: optimumai.circuit
