"""Build the CPU IVF64 inner-product index from all catalog embeddings."""

import json
import sys
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.faiss_search import (
    DEFAULT_NPROBE,
    EMBEDDING_DIMENSION,
    INDEXES_PATH,
    NLIST,
    PROCESSED_PATH,
)


def build_index() -> None:
    embeddings = np.load(PROCESSED_PATH / "embeddings.npy", allow_pickle=False)
    image_ids = np.load(PROCESSED_PATH / "image_ids.npy", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError("Embeddings must have shape (N, 2048).")
    if embeddings.dtype != np.float32:
        raise ValueError("Embeddings must have dtype float32.")
    if image_ids.ndim != 1 or len(image_ids) != len(embeddings):
        raise ValueError("image_ids must have shape (N,) matching the embeddings row count.")
    if image_ids.dtype.hasobject:
        raise ValueError("image_ids must not use object dtype.")
    if len(embeddings) < NLIST:
        raise ValueError(f"At least {NLIST} embeddings are required to train IVF{NLIST}.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be finite, without NaN or Inf.")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("Embeddings must already be L2 normalized.")
    embeddings = np.ascontiguousarray(embeddings)

    print(f"FAISS version: {faiss.__version__}", flush=True)
    print(f"Training IVF{NLIST} on {len(embeddings)} vectors...", flush=True)
    # Offline build time covers index creation, training, and adding vectors.
    started = perf_counter()
    quantizer = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index = faiss.IndexIVFFlat(
        quantizer, EMBEDDING_DIMENSION, NLIST, faiss.METRIC_INNER_PRODUCT
    )
    index.train(embeddings)
    if not index.is_trained:
        raise RuntimeError("IVF training did not produce a trained index.")
    # Plain add() assigns row positions 0..N-1, preserving the image_ids mapping.
    index.add(embeddings)
    if index.ntotal != len(embeddings):
        raise RuntimeError("Index vector count does not match the embeddings row count.")
    build_seconds = perf_counter() - started
    index.nprobe = DEFAULT_NPROBE

    config = {
        "index_type": "IndexIVFFlat",
        "metric": "inner_product",
        "dimension": EMBEDDING_DIMENSION,
        "nlist": NLIST,
        "default_nprobe": DEFAULT_NPROBE,
        "num_vectors": index.ntotal,
        "build_seconds": build_seconds,
        "faiss_version": faiss.__version__,
    }
    INDEXES_PATH.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEXES_PATH / "faiss_ivf.index"))
    (INDEXES_PATH / "faiss_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Dimension: {index.d}")
    print(f"nlist: {index.nlist}")
    print(f"Default nprobe: {index.nprobe}")
    print(f"index.is_trained: {index.is_trained}")
    print(f"index.ntotal: {index.ntotal}")
    print(f"Index build time: {build_seconds:.3f} seconds (load/save excluded)")
    print("Saved indexes/faiss_ivf.index and indexes/faiss_config.json")


def main() -> int:
    try:
        build_index()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
