"""Exact cosine search over all stored, L2-normalized image embeddings."""

from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np

PROCESSED_PATH = Path(__file__).resolve().parents[2] / "data/processed"
EMBEDDING_DIMENSION = 2048


class SequentialSearchService:
    """Load artifacts once and reuse them for subsequent queries."""

    def __init__(
        self,
        embeddings_path: str | Path = PROCESSED_PATH / "embeddings.npy",
        image_ids_path: str | Path = PROCESSED_PATH / "image_ids.npy",
    ) -> None:
        embeddings = np.load(embeddings_path, allow_pickle=False)
        self.image_ids = np.load(image_ids_path, allow_pickle=False)

        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError("Embeddings must have shape (N, 2048).")
        if self.image_ids.ndim != 1 or len(self.image_ids) != len(embeddings):
            raise ValueError("image_ids must have shape (N,) matching the embeddings row count.")
        if len(embeddings) == 0:
            raise ValueError("The embedding dataset must not be empty.")
        if embeddings.dtype.hasobject or self.image_ids.dtype.hasobject:
            raise ValueError("Artifacts must not use object dtype.")
        if not np.issubdtype(embeddings.dtype, np.number) or np.iscomplexobj(embeddings):
            raise ValueError("Embeddings must contain real numeric values.")

        self.embeddings = embeddings.astype(np.float32, copy=False)
        if not np.isfinite(self.embeddings).all():
            raise ValueError("Embeddings must be finite, without NaN or Inf.")
        norms = np.linalg.norm(self.embeddings, axis=1)
        if not np.allclose(norms, 1.0, rtol=0, atol=1e-5):
            raise ValueError("Stored embeddings must be L2 normalized.")

        self.last_search_latency_ms: float | None = None

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[dict]:
        """Return Top-K results; latency measures only similarity and ranking."""
        if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= len(self.embeddings):
            raise ValueError(f"K must be an integer between 1 and {len(self.embeddings)}.")

        query = np.asarray(query_vector)
        if np.iscomplexobj(query):
            raise ValueError("The query must contain real numeric values.")
        query = np.asarray(query, dtype=np.float32)
        if query.shape == (1, EMBEDDING_DIMENSION):
            query = query[0]
        if query.shape != (EMBEDDING_DIMENSION,):
            raise ValueError("Query must have shape (2048,) or (1, 2048).")
        if not np.isfinite(query).all():
            raise ValueError("Query must be finite, without NaN or Inf.")
        if not np.isclose(np.linalg.norm(query), 1.0, rtol=0, atol=1e-5):
            raise ValueError("Query must be L2 normalized.")

        started = perf_counter()
        scores = self.embeddings @ query
        # Stable sorting keeps catalog order when two scores are exactly tied.
        top_indices = np.argsort(-scores, kind="stable")[:k]
        self.last_search_latency_ms = (perf_counter() - started) * 1000

        return [
            {
                "rank": rank,
                "image_id": self.image_ids[index].item(),
                "score": float(scores[index]),
            }
            for rank, index in enumerate(top_indices, start=1)
        ]
