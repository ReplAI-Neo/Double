from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer, util

from .config import EmbeddingConfig


class SentenceTransformerEmbedder:
    """
    Thin wrapper around the project's sentence-transformers dependency.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.model = SentenceTransformer(
            self.config.model_name, device=self.config.device
        )
        self._dim = int(self.model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> List[float]:
        array = self.embed([text])
        if array.size == 0:
            return []
        return array[0].tolist()


__all__ = ["SentenceTransformerEmbedder"]

