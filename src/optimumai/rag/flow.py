"""Entry point for the RAG concept-flow diagram.

    >>> from optimumai.rag.flow import rag_flow
    >>> path = rag_flow(out="rag_explainer.html")   # → self-contained D3 HTML
    >>> path = rag_flow(query="What year did the tower open?")
"""

from __future__ import annotations

from optimumai.rag.explainer import RAG_LAYOUT, render_flow_trace_html
from optimumai.rag.trace import _DEFAULT_QUERY, build_rag_trace


def rag_flow(
    query: str = _DEFAULT_QUERY,
    k: int = 2,
    out: str | None = "rag_explainer.html",
) -> str:
    """Build a RAG :class:`~optimumai.core.flow_trace.FlowTrace` and render it
    to a self-contained D3 + KaTeX HTML file.

    Parameters
    ----------
    query:
        The question to retrieve context for.  Cosine scores are computed from
        the real :class:`~optimumai.rag.pipeline.RAGPipeline` — not toy values.
    k:
        Top-k chunks to retrieve.
    out:
        Output HTML path.  Defaults to ``"rag_explainer.html"`` in the current
        directory.

    Returns
    -------
    str
        The path to the generated HTML file.

    Examples
    --------
    CLI::

        optimumai flow rag
        optimumai flow rag --out my_rag.html

    Python::

        from optimumai.rag.flow import rag_flow
        rag_flow(query="How tall is the Eiffel Tower?", out="rag.html")
    """
    trace = build_rag_trace(query=query, k=k)
    return render_flow_trace_html(trace, RAG_LAYOUT, out=out)
