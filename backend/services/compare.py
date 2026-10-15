"""Compare ANN retrieval with the existing exact Sequential Search baseline."""

from numbers import Integral

import numpy as np
from faiss import METRIC_INNER_PRODUCT

from backend.services.faiss_search import DEFAULT_NPROBE, FaissSearchService
from backend.services.sequential_search import SequentialSearchService


def _validate_results(results: list[dict], k: int, engine: str) -> None:
    if len(results) != k:
        raise RuntimeError(f"{engine} returned {len(results)} results; expected {k}.")
    image_ids = [row["image_id"] for row in results]
    if len(set(image_ids)) != k:
        raise RuntimeError(f"{engine} returned duplicate image IDs.")
    scores = [row["score"] for row in results]
    if not np.isfinite(scores).all():
        raise RuntimeError(f"{engine} returned non-finite scores.")
    if scores != sorted(scores, reverse=True):
        raise RuntimeError(f"{engine} results must be sorted by descending score.")


class CompareService:
    """Reuse both search services with the same query, dataset, and K."""

    def __init__(
        self,
        sequential: SequentialSearchService | None = None,
        faiss_service: FaissSearchService | None = None,
    ) -> None:
        self.sequential = sequential if sequential is not None else SequentialSearchService()
        self.faiss = faiss_service if faiss_service is not None else FaissSearchService()
        self.dataset_size, self.dimension = self.sequential.embeddings.shape
        if (self.dataset_size, self.dimension) != (self.faiss.index.ntotal, self.faiss.index.d):
            raise ValueError("Both engines must use the same dataset size and dimension.")
        if self.faiss.index.metric_type != METRIC_INNER_PRODUCT:
            raise ValueError("Both engines must use inner product on L2-normalized vectors.")
        if not np.array_equal(self.sequential.image_ids, self.faiss.image_ids):
            raise ValueError("Both engines must use the same image IDs in the same row order.")

    def compare(
        self, query_embedding: np.ndarray, k: int = 5, nprobe: int = DEFAULT_NPROBE
    ) -> dict:
        if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= self.dataset_size:
            raise ValueError(f"K must be an integer between 1 and {self.dataset_size}.")
        if (
            isinstance(nprobe, bool)
            or not isinstance(nprobe, Integral)
            or not 1 <= nprobe <= self.faiss.index.nlist
        ):
            raise ValueError(f"nprobe must be an integer between 1 and {self.faiss.index.nlist}.")
        k, nprobe = int(k), int(nprobe)

        # Pass the same embedding unchanged; encoding belongs to the caller.
        sequential_results = self.sequential.search(query_embedding, k=k)
        sequential_latency = self.sequential.last_search_latency_ms
        _validate_results(sequential_results, k, "Sequential")

        faiss_results = self.faiss.search(query_embedding, k=k, nprobe=nprobe)
        faiss_latency = self.faiss.last_search_latency_ms
        _validate_results(faiss_results, k, "FAISS")

        # Use only the engines' search timers, without timing this wrapper.
        for engine, latency in (("Sequential", sequential_latency), ("FAISS", faiss_latency)):
            if latency is None or not np.isfinite(latency) or latency < 0:
                raise RuntimeError(f"{engine} must provide a finite, non-negative latency.")

        sequential_ids = {row["image_id"] for row in sequential_results}
        faiss_ids = {row["image_id"] for row in faiss_results}
        overlap_count = len(sequential_ids & faiss_ids)
        speedup = sequential_latency / faiss_latency if faiss_latency > 0 else None

        return {
            "config": {
                "dataset_size": self.dataset_size,
                "dimension": self.dimension,
                "k": k,
                "nlist": self.faiss.index.nlist,
                "nprobe": nprobe,
            },
            "sequential": {"latency_ms": sequential_latency, "results": sequential_results},
            "faiss": {"latency_ms": faiss_latency, "results": faiss_results},
            "metrics": {
                "overlap_count": overlap_count,
                # ANN Recall@K against exact search, not semantic/classification accuracy.
                "recall_at_k": overlap_count / k,
                "speedup": speedup,
            },
        }
