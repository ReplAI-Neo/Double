from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Neo4jConnectionConfig:
    uri: str
    username: str
    password: str
    database: Optional[str] = None
    encrypted: bool = False


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None
    batch_size: int = 32
    normalize_embeddings: bool = True


__all__ = ["Neo4jConnectionConfig", "EmbeddingConfig"]

