"""CPU FAISS IVF search over the stored ResNet50 embeddings."""

from numbers import Integral
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = REPO_ROOT / "data/processed"
INDEXES_PATH = REPO_ROOT / "indexes"
EMBEDDING_DIMENSION = 2048
NLIST = 64
DEFAULT_NPROBE = 8


class FaissSearchService:
    """Load the IVF index and ID mapping once per service instance."""

    def __init__(
        self,
        index_path: str | Path = INDEXES_PATH / "faiss_ivf.index",
        image_ids_path: str | Path = PROCESSED_PATH / "image_ids.npy",
    ) -> None:
        self.index = faiss.read_index(str(index_path))
        self.image_ids = np.load(image_ids_path, allow_pickle=False)

        if not isinstance(self.index, faiss.IndexIVFFlat):
            raise ValueError("The index must be IndexIVFFlat.")
        if self.index.d != EMBEDDING_DIMENSION:
            raise ValueError("Index dimension must be 2048.")
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("The IVF metric must be inner product.")
        if not isinstance(faiss.downcast_index(self.index.quantizer), faiss.IndexFlatIP):
            raise ValueError("The IVF quantizer must be IndexFlatIP.")
        if self.index.nlist != NLIST:
            raise ValueError(f"Index nlist must be {NLIST}.")
        if not self.index.is_trained:
            raise ValueError("The IVF index must be trained before searching.")
        if self.image_ids.ndim != 1 or len(self.image_ids) != self.index.ntotal:
            raise ValueError("image_ids must have shape (N,) matching index.ntotal.")
        if self.image_ids.dtype.hasobject:
            raise ValueError("image_ids must not use object dtype.")
        if self.index.ntotal == 0:
            raise ValueError("The IVF index must contain at least one vector.")

        self.last_search_latency_ms: float | None = None

    def search(
        self, query_vector: np.ndarray, k: int = 5, nprobe: int = DEFAULT_NPROBE
    ) -> list[dict]:
        """Return mapped Top-K results and record only index.search() latency."""
        if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= self.index.ntotal:
            raise ValueError(f"K must be an integer between 1 and {self.index.ntotal}.")
        if (
            isinstance(nprobe, bool)
            or not isinstance(nprobe, Integral)
            or not 1 <= nprobe <= self.index.nlist
        ):
            raise ValueError(f"nprobe must be an integer between 1 and {self.index.nlist}.")

        query = np.asarray(query_vector)
        if np.iscomplexobj(query):
            raise ValueError("Query must contain real numeric values.")
        query = np.asarray(query, dtype=np.float32)
        if query.shape == (1, EMBEDDING_DIMENSION):
            query = query[0]
        if query.shape != (EMBEDDING_DIMENSION,):
            raise ValueError("Query must have shape (2048,) or (1, 2048).")
        if not np.isfinite(query).all():
            raise ValueError("Query must be finite, without NaN or Inf.")
        if not np.isclose(np.linalg.norm(query), 1.0, rtol=0, atol=1e-5):
            raise ValueError("Query must be L2 normalized.")
        query = np.ascontiguousarray(query.reshape(1, EMBEDDING_DIMENSION))

        self.index.nprobe = int(nprobe)
        started = perf_counter()
        scores, positions = self.index.search(query, int(k))
        self.last_search_latency_ms = (perf_counter() - started) * 1000

        # FAISS uses -1 for missing neighbors; never map it to image_ids[-1].
        if (positions < 0).any():
            found = int(np.count_nonzero(positions >= 0))
            raise RuntimeError(
                f"FAISS found only {found} of {k} neighbors with nprobe={nprobe}; "
                "increase nprobe or reduce K."
            )
        if (positions >= len(self.image_ids)).any():
            raise RuntimeError("FAISS returned a position outside the image ID mapping.")
        if not np.isfinite(scores).all():
            raise RuntimeError("FAISS returned NaN or Inf scores.")

        return [
            {
                "rank": rank,
                "image_id": self.image_ids[position].item(),
                "score": float(score),
            }
            for rank, (position, score) in enumerate(zip(positions[0], scores[0]), start=1)
        ]
