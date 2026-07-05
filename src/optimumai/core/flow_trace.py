"""optimumai.core.flow_trace — the narrated-pipeline schema.

Design principle
----------------
Every concept module (rag.py, mdp.py, quantize.py, attention.py …) that has
a *pipeline* shape — a directed graph of entities that activates stage by stage
— can produce one of these :class:`FlowTrace` objects.  Every renderer (the D3
web widget, a future Manim exporter, a Rich terminal summary) consumes the
*same* object.  Add a new concept → every renderer gets it for free.  Add a new
renderer → every concept gets it for free.

Two design decisions matter most
---------------------------------
1. **Nodes/edges are separate from steps.**
   Entities that persist across the pipeline (a document, a chunk, a vector)
   are :class:`FlowNode`\\ s.  Connections between them are :class:`FlowEdge`\\ s.
   Steps don't recompute the graph — they say "which edges become active, and
   with what data, at this point in time."  This makes a Sankey-style
   *progressive reveal* (à la Transformer Explainer) possible: the renderer
   draws the graph **once**, then a step index drives opacity/highlight state.

2. **DataRef never inlines large tensors.**
   A :class:`DataRef` carries a ``preview`` (first few values) and a ``shape``,
   never the full array.  This keeps the serialised trace small enough to embed
   directly in an ``<script>`` block (no fetch, no build step) and keeps the
   renderer's job purely visual.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

SCHEMA_VERSION = "1.0"

DataKind = Literal["scalar", "vector", "matrix", "tensor", "text", "list"]


@dataclass
class DataRef:
    """A reference to a value flowing through the pipeline.

    Never carries a full large tensor — just enough to render (shape + preview).
    """

    id: str
    label: str
    kind: DataKind
    preview: Any  # e.g. 0.83, [0.1, -2.3, …], "chunk text…"
    shape: Optional[list[int]] = None
    full_value_ref: Optional[str] = None  # optional pointer for later fetch

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FlowNode:
    """A persistent entity in the pipeline graph.

    Drawn once by the renderer; steps only toggle its opacity/highlight state.
    """

    id: str
    label: str
    kind: str  # "document" | "chunk" | "vector" | "model" | "store" | "text"
    group: Optional[str] = None  # colour-coding group, e.g. "retrieval"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FlowEdge:
    """A connection between two :class:`FlowNode`\\ s.

    ``active_from_step`` tells the renderer at which step this edge should
    first appear.  Past edges dim to grey rather than disappearing.
    """

    id: str
    source: str  # FlowNode.id
    target: str  # FlowNode.id
    active_from_step: str  # FlowStep.id at which this edge becomes visible
    label: Optional[str] = None
    weight: Optional[float] = None  # e.g. similarity score, attention weight

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FlowStep:
    """One narrated computation step in the pipeline."""

    id: str
    index: int
    stage: str  # pipeline-stage grouping, e.g. "chunking" | "retrieval"
    title: str  # short label shown in the side panel
    op: str  # operation type → lets renderer pick a visual template
    # e.g. "split" | "embed" | "cosine_sim" | "topk" | "rerank" | "concat" | "generate"
    narration: str  # one plain-English sentence (the "why" line)
    formula: Optional[str] = None  # LaTeX string rendered via KaTeX
    inputs: list[DataRef] = field(default_factory=list)
    outputs: list[DataRef] = field(default_factory=list)
    highlight_nodes: list[str] = field(default_factory=list)  # FlowNode ids
    highlight_edges: list[str] = field(default_factory=list)  # FlowEdge ids
    metrics: dict[str, float] = field(default_factory=dict)  # e.g. {"top_score": 0.83}
    duration_hint_ms: int = 900  # pacing hint for animation export

    def to_dict(self) -> dict:
        d = asdict(self)
        d["inputs"] = [i.to_dict() for i in self.inputs]
        d["outputs"] = [o.to_dict() for o in self.outputs]
        return d


@dataclass
class FlowTrace:
    """The full narrated trace of a pipeline's execution.

    Examples::

        trace = FlowTrace(concept="rag_pipeline", title="RAG", ...)
        html  = render_flow_trace_html(trace, layout)
        trace.to_json("rag_trace.json")
    """

    concept: str  # e.g. "rag_pipeline", "value_iteration", "quantization"
    title: str
    description: str
    nodes: list[FlowNode]
    edges: list[FlowEdge]
    steps: list[FlowStep]
    meta: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "concept": self.concept,
            "title": self.title,
            "description": self.description,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "steps": [s.to_dict() for s in self.steps],
            "meta": {**self.meta, "generated_at": time.time()},
        }

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w") as fh:
                fh.write(payload)
        return payload

    def validate(self) -> list[str]:
        """Cheap sanity checks; returns a list of problem descriptions (empty = OK)."""
        problems: list[str] = []
        node_ids = {n.id for n in self.nodes}
        step_ids = {s.id for s in self.steps}
        for e in self.edges:
            if e.source not in node_ids:
                problems.append(f"Edge {e.id}: source '{e.source}' not a known node")
            if e.target not in node_ids:
                problems.append(f"Edge {e.id}: target '{e.target}' not a known node")
            if e.active_from_step not in step_ids:
                problems.append(
                    f"Edge {e.id}: active_from_step '{e.active_from_step}' not a known step"
                )
        for s in self.steps:
            for nid in s.highlight_nodes:
                if nid not in node_ids:
                    problems.append(
                        f"Step {s.id}: highlight_nodes references unknown node '{nid}'"
                    )
            for eid in s.highlight_edges:
                edge_ids = {edge.id for edge in self.edges}
                if eid not in edge_ids:
                    problems.append(
                        f"Step {s.id}: highlight_edges references unknown edge '{eid}'"
                    )
        return problems
