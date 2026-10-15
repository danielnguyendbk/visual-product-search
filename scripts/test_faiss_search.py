"""Check IVF search end-to-end using the first real catalog image."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    from backend.services.encoder import encode_image
    from backend.services.faiss_search import DEFAULT_NPROBE, FaissSearchService

    try:
        catalog = pd.read_csv(
            REPO_ROOT / "data/processed/catalog.csv", dtype=str, keep_default_na=False
        )
        if catalog.empty:
            raise ValueError("The catalog is empty.")
        service = FaissSearchService()
        catalog_ids = catalog["image_id"].to_numpy(dtype=service.image_ids.dtype)
        if not np.array_equal(catalog_ids, service.image_ids):
            raise ValueError("Catalog image IDs must match the complete artifact ID order.")

        row = catalog.iloc[0]
        query_id = catalog_ids[0].item()
        print(f"Query image_id: {query_id}", flush=True)
        print(f"nlist: {service.index.nlist}")
        print(f"nprobe: {DEFAULT_NPROBE}", flush=True)
        try:
            with Image.open(REPO_ROOT / row.image_path) as image:
                query = encode_image(image)
        except Exception as error:
            raise ValueError(
                f"Cannot encode image_id={row.image_id!r}, image_path={row.image_path!r}: {error}"
            ) from error

        results = service.search(query, k=5, nprobe=DEFAULT_NPROBE)
        print("Top-5 FAISS results:")
        for result in results:
            print(
                f"{result['rank']}. image_id={result['image_id']} "
                f"score={result['score']:.8f}"
            )
        print(f"Search-only latency: {service.last_search_latency_ms:.3f} ms")
        if len(results) != 5:
            raise ValueError("FAISS did not return exactly five results.")
        scores = [result["score"] for result in results]
        if scores != sorted(scores, reverse=True):
            raise ValueError("FAISS scores are not sorted in descending order.")
        if results[0]["image_id"] != query_id:
            raise ValueError(
                f"Top-1 image_id={results[0]['image_id']} differs from query image_id={query_id}; "
                "the dataset has not been changed."
            )
        if not np.isclose(results[0]["score"], 1.0, rtol=0, atol=1e-5):
            raise ValueError("The Top-1 score is not approximately 1.0.")
        print("PASS: five descending results, query image is Top-1 with score approximately 1.0.")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
