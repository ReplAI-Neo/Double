"""
Graph-based retrieval augmented generation utilities for ReplAI.

This package provides a light-weight GraphRAG pipeline that can ingest the
conversation data we already parse (see `docs/CONVERSATION_SCHEMA.md`),
construct a heterogeneous graph, and expose helpers for retrieval, labeling,
and memory lookups for downstream agents.
"""

from .builder import GraphRAGBuilder, GraphRAGGraph
from .retriever import GraphMemory, RetrievalRequest, RetrievalResult
from .labeler import GraphTagger, TaggingConfig

__all__ = [
    "GraphRAGBuilder",
    "GraphRAGGraph",
    "GraphMemory",
    "RetrievalRequest",
    "RetrievalResult",
    "GraphTagger",
    "TaggingConfig",
]

