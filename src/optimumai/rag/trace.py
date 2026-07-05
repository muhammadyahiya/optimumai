"""Build a :class:`FlowTrace` for the RAG pipeline using real computed values.

This module is the *only* place that knows what "RAG" means.  It calls the
actual :class:`~optimumai.rag.pipeline.RAGPipeline` to compute real cosine
scores, then encodes those numbers into a :class:`FlowTrace`.  The renderer
(``rag_explainer.py``) never sees the word "RAG" — it only sees nodes, edges,
and steps.

Proof of the abstraction
-------------------------
Swap this file for a ``mdp_trace.py`` that walks value-iteration on the
:class:`~optimumai.rl.mdp.MDP` module and emits the same ``FlowTrace`` JSON
shape, and the D3 renderer in ``rag_explainer.py`` renders that instead,
**unmodified**.
"""

from __future__ import annotations

import numpy as np

from optimumai.core.flow_trace import DataRef, FlowEdge, FlowNode, FlowStep, FlowTrace
from optimumai.rag.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# default corpus and query (mirrors the reference demo)
# ---------------------------------------------------------------------------

_DOC_TEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was designed by Gustave Eiffel's company and completed in 1889. "
    "The tower stands 330 meters tall and was the tallest man-made "
    "structure in the world for 41 years."
)

_CHUNKS = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "It was designed by Gustave Eiffel's company and completed in 1889.",
    "The tower stands 330 meters tall and was the tallest man-made structure "
    "in the world for 41 years.",
]

_DEFAULT_QUERY = "How tall is the Eiffel Tower and when was it built?"


def build_rag_trace(
    query: str = _DEFAULT_QUERY,
    k: int = 2,
    rerank_noise: float = 0.04,
    seed: int = 0,
) -> FlowTrace:
    """Build a :class:`FlowTrace` for a RAG pipeline run.

    Parameters
    ----------
    query:
        The user question to retrieve context for.
    k:
        Top-k chunks to retrieve.
    rerank_noise:
        Small perturbation added to cosine scores to simulate a cross-encoder
        reranker (real re-ranking shifts scores without changing the embedding).
    seed:
        Random seed for the embedding model (for reproducibility).

    Returns
    -------
    FlowTrace
        A validated trace with real cosine similarity scores computed by the
        actual :class:`~optimumai.rag.pipeline.RAGPipeline`.
    """
    # ------------------------------------------------------------------
    # Compute real scores from the actual pipeline
    # ------------------------------------------------------------------
    pipeline = RAGPipeline(corpus=_CHUNKS, seed=seed)
    q_vec = pipeline._embed(query)
    sim_scores: list[float] = [float(s) for s in pipeline._doc_embeddings @ q_vec]

    # Simulate a cross-encoder reranker: deterministic nudge so the ranking
    # can subtly shift without depending on a real rerank model.
    rng = np.random.default_rng(seed + 42)
    rerank_scores: list[float] = [
        max(0.0, min(1.0, s + rng.uniform(-rerank_noise, rerank_noise)))
        for s in sim_scores
    ]

    q_preview = "[" + ", ".join(f"{v:.3f}" for v in q_vec[:4]) + f", …] (dim={pipeline.dim})"

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    nodes = [
        FlowNode(id="doc",       label="Source Document",     kind="document", group="ingestion"),
        *[
            FlowNode(
                id=f"chunk_{i}",
                label=f"Chunk {i + 1}",
                kind="chunk",
                group="ingestion",
                meta={"text": _CHUNKS[i]},
            )
            for i in range(3)
        ],
        FlowNode(id="query",     label="User Query",          kind="text",   group="query"),
        FlowNode(id="query_vec", label="Query Embedding",     kind="vector", group="query"),
        *[
            FlowNode(
                id=f"chunk_vec_{i}",
                label=f"Chunk {i + 1} Embedding",
                kind="vector",
                group="ingestion",
            )
            for i in range(3)
        ],
        FlowNode(id="index",     label="Vector Index",        kind="store",  group="retrieval"),
        FlowNode(id="retrieved", label="Top-K Retrieved",     kind="text",   group="retrieval"),
        FlowNode(id="reranked",  label="Reranked Context",    kind="text",   group="retrieval"),
        FlowNode(id="context",   label="Assembled Context",   kind="text",   group="generation"),
        FlowNode(id="llm",       label="LLM",                 kind="model",  group="generation"),
        FlowNode(id="answer",    label="Generated Answer",    kind="text",   group="generation"),
    ]

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    edges = [
        *[
            FlowEdge(
                id=f"e_doc_chunk_{i}",
                source="doc",
                target=f"chunk_{i}",
                active_from_step="s1_chunk",
            )
            for i in range(3)
        ],
        *[
            FlowEdge(
                id=f"e_chunk_vec_{i}",
                source=f"chunk_{i}",
                target=f"chunk_vec_{i}",
                active_from_step="s2_embed_chunks",
            )
            for i in range(3)
        ],
        *[
            FlowEdge(
                id=f"e_vec_index_{i}",
                source=f"chunk_vec_{i}",
                target="index",
                active_from_step="s3_index",
            )
            for i in range(3)
        ],
        FlowEdge(
            id="e_query_vec",
            source="query",
            target="query_vec",
            active_from_step="s4_embed_query",
        ),
        FlowEdge(
            id="e_query_index",
            source="query_vec",
            target="index",
            active_from_step="s5_retrieve",
            label="cosine similarity",
        ),
        FlowEdge(
            id="e_index_retrieved",
            source="index",
            target="retrieved",
            active_from_step="s5_retrieve",
            label=f"top-k={k}",
        ),
        FlowEdge(
            id="e_retrieved_reranked",
            source="retrieved",
            target="reranked",
            active_from_step="s6_rerank",
        ),
        FlowEdge(
            id="e_reranked_context",
            source="reranked",
            target="context",
            active_from_step="s7_assemble",
        ),
        FlowEdge(
            id="e_context_llm",
            source="context",
            target="llm",
            active_from_step="s8_generate",
        ),
        FlowEdge(
            id="e_llm_answer",
            source="llm",
            target="answer",
            active_from_step="s8_generate",
        ),
    ]

    # ------------------------------------------------------------------
    # Steps — real scores baked in
    # ------------------------------------------------------------------
    top_idx = sorted(range(3), key=lambda i: rerank_scores[i], reverse=True)[:k]

    steps = [
        FlowStep(
            id="s1_chunk",
            index=1,
            stage="chunking",
            op="split",
            title="Split document into chunks",
            narration=(
                "The source document is split into overlapping passages small enough "
                "to embed individually and retrieve independently."
            ),
            inputs=[
                DataRef(
                    id="doc_in",
                    label="Document",
                    kind="text",
                    preview=_DOC_TEXT[:64] + "…",
                )
            ],
            outputs=[
                DataRef(
                    id=f"chunk_out_{i}",
                    label=f"Chunk {i + 1}",
                    kind="text",
                    preview=_CHUNKS[i][:44] + "…",
                )
                for i in range(3)
            ],
            highlight_nodes=["doc", "chunk_0", "chunk_1", "chunk_2"],
            highlight_edges=[f"e_doc_chunk_{i}" for i in range(3)],
        ),
        FlowStep(
            id="s2_embed_chunks",
            index=2,
            stage="embedding",
            op="embed",
            title="Embed each chunk",
            narration=(
                "Each chunk is passed through the embedding model, turning text into "
                "a fixed-length vector that captures its meaning."
            ),
            formula=r"\vec{v}_i = \text{Embed}(\text{chunk}_i)",
            highlight_nodes=[f"chunk_vec_{i}" for i in range(3)],
            highlight_edges=[f"e_chunk_vec_{i}" for i in range(3)],
        ),
        FlowStep(
            id="s3_index",
            index=3,
            stage="indexing",
            op="index",
            title="Add vectors to the index",
            narration=(
                "Chunk embeddings are added to a vector index, which organises them "
                "for fast approximate nearest-neighbour search."
            ),
            highlight_nodes=["index"],
            highlight_edges=[f"e_vec_index_{i}" for i in range(3)],
        ),
        FlowStep(
            id="s4_embed_query",
            index=4,
            stage="embedding",
            op="embed",
            title="Embed the user query",
            narration=(
                "The incoming question is embedded with the same model, so it lands "
                "in the same vector space as the chunks."
            ),
            formula=r"\vec{q} = \text{Embed}(\text{query})",
            inputs=[
                DataRef(
                    id="query_in",
                    label="Query",
                    kind="text",
                    preview=query,
                ),
                DataRef(
                    id="query_vec_out",
                    label="Query vector",
                    kind="vector",
                    preview=q_preview,
                    shape=[pipeline.dim],
                ),
            ],
            highlight_nodes=["query", "query_vec"],
            highlight_edges=["e_query_vec"],
        ),
        FlowStep(
            id="s5_retrieve",
            index=5,
            stage="retrieval",
            op="topk",
            title=f"Retrieve top-{k} by cosine similarity",
            narration=(
                "The query vector is compared against every chunk vector; the "
                f"{k} most similar chunks are pulled back as candidate context."
            ),
            formula=(
                r"\text{sim}(\vec{q},\,\vec{v}_i)"
                r"= \frac{\vec{q} \cdot \vec{v}_i}{\|\vec{q}\|\,\|\vec{v}_i\|}"
            ),
            metrics={f"chunk_{i}_score": round(sim_scores[i], 4) for i in range(3)},
            highlight_nodes=["query_vec", "index", "retrieved"],
            highlight_edges=["e_query_index", "e_index_retrieved"],
        ),
        FlowStep(
            id="s6_rerank",
            index=6,
            stage="retrieval",
            op="rerank",
            title="Rerank with cross-encoder",
            narration=(
                "A slower but more accurate cross-encoder rescores the (query, chunk) "
                "pairs directly — this can reorder the initial top-k."
            ),
            metrics={f"chunk_{i}_rerank_score": round(rerank_scores[i], 4) for i in range(3)},
            highlight_nodes=["reranked"],
            highlight_edges=["e_retrieved_reranked"],
        ),
        FlowStep(
            id="s7_assemble",
            index=7,
            stage="generation",
            op="concat",
            title="Assemble final context",
            narration=(
                "The top reranked chunks are concatenated into a single context block, "
                "ordered by relevance, ready to hand to the LLM."
            ),
            outputs=[
                DataRef(
                    id=f"ctx_chunk_{rank}",
                    label=f"Context chunk {rank + 1}",
                    kind="text",
                    preview=_CHUNKS[idx][:48] + "…",
                )
                for rank, idx in enumerate(top_idx)
            ],
            highlight_nodes=["context"],
            highlight_edges=["e_reranked_context"],
        ),
        FlowStep(
            id="s8_generate",
            index=8,
            stage="generation",
            op="generate",
            title="Generate answer conditioned on context",
            narration=(
                "The LLM produces an answer conditioned on the retrieved context — "
                "this is the step your faithfulness metric later checks for grounding."
            ),
            outputs=[
                DataRef(
                    id="answer_out",
                    label="Answer",
                    kind="text",
                    preview=(
                        "The Eiffel Tower stands 330 metres tall and was "
                        "completed in 1889."
                    ),
                )
            ],
            highlight_nodes=["context", "llm", "answer"],
            highlight_edges=["e_context_llm", "e_llm_answer"],
        ),
    ]

    trace = FlowTrace(
        concept="rag_pipeline",
        title="RAG: Retrieval-Augmented Generation",
        description=(
            "document → chunks → embeddings → vector index → "
            "cosine retrieval → rerank → context → LLM → answer"
        ),
        nodes=nodes,
        edges=edges,
        steps=steps,
        meta={
            "source_module": "optimumai.rag",
            "query": query,
            "n_chunks": len(_CHUNKS),
            "embedding_dim": pipeline.dim,
            "k": k,
        },
    )

    problems = trace.validate()
    if problems:
        raise RuntimeError("FlowTrace validation failed:\n" + "\n".join(problems))

    return trace
