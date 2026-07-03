"""OptimumAI v0.2 — the fundamentals, from a gradient to a world model.

Run me:  python examples/fundamentals.py
"""

from optimumai import Value
from optimumai.calculus import chain_rule_trace
from optimumai.interpretability import superposition_trace
from optimumai.neural_networks import train_demo
from optimumai.optimization import descent_demo
from optimumai.transformers import MultiHeadAttention, TransformerBlock
from optimumai.world_models import JEPA


def main() -> None:
    print("\n### 1. Autograd — the chain rule, flowing backwards (micrograd) ###")
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    loss = (a * b + c).tanh()
    loss.label = "L"
    loss.backward_trace().render("engineer")

    print("\n### 2. Chain rule — exact autograd vs numeric finite difference ###")
    chain_rule_trace().render("intermediate")

    print("\n### 3. Optimization — Adam walking a loss bowl to its minimum ###")
    descent_demo("adam", steps=60).render("intermediate")

    print("\n### 4. Neural network — real backprop, loss falling to ~0 ###")
    train_demo(steps=150).render("intermediate")

    print("\n### 5. Multi-head attention — parallel heads + causal mask (nanoGPT) ###")
    MultiHeadAttention.demo().render("engineer")

    print("\n### 6. Transformer block — LayerNorm → attention → FFN + residuals ###")
    TransformerBlock.demo().render("engineer")

    print("\n### 7. JEPA — LeCun: predict in representation space, not pixels ###")
    JEPA.demo().render("engineer")

    print("\n### 8. Superposition — Anthropic: why neurons are polysemantic ###")
    superposition_trace(5, 2).render("engineer")


if __name__ == "__main__":
    main()
