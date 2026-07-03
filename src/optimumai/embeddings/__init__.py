"""Embeddings — turning discrete tokens into dense vectors."""

from optimumai.embeddings.lookup import (
    embedding_lookup,
    embedding_lookup_trace,
    nearest_neighbors,
    nearest_neighbors_trace,
)

__all__ = [
    "embedding_lookup",
    "embedding_lookup_trace",
    "nearest_neighbors",
    "nearest_neighbors_trace",
]
