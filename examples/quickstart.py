"""OptimumAI quickstart — the full arc, from a dot product to attention.

Run me:  python examples/quickstart.py
"""

import numpy as np

from optimumai import Attention, Matrix, Vector, softmax


def main() -> None:
    print("\n### 1. Dot product — similarity & the atom of matmul ###")
    Vector([1, 2, 3]).dot(Vector([4, 5, 6]), explain=True)

    print("\n### 2. Cosine similarity — the RAG ranking score ###")
    Vector([1, 2, 3]).cosine_similarity(Vector([2, 4, 6]), explain=True)

    print("\n### 3. Matrix multiply — every dense layer, cell by cell ###")
    Matrix([[1, 2], [3, 4]]).matmul(Matrix([[5, 6], [7, 8]]), explain=True)

    print("\n### 4. Softmax — logits into a probability distribution ###")
    softmax([2.0, 1.0, 0.1], temperature=0.5, explain=True)

    print("\n### 5. Attention — softmax(QKᵀ/√dₖ)·V, the transformer core ###")
    rng = np.random.default_rng(0)
    Q, K, V = (rng.normal(size=(3, 4)).round(2) for _ in range(3))
    Attention(d_k=4).forward(Q, K, V, explain=True, level="engineer")


if __name__ == "__main__":
    main()
