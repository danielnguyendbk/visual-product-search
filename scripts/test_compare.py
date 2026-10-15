"""Inspect one query's Sequential/FAISS results and nprobe observations."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def format_speedup(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def main() -> int:
    from backend.services.compare import CompareService
    from backend.services.encoder import encode_image

    try:
        catalog = pd.read_csv(
            REPO_ROOT / "data/processed/catalog.csv", dtype=str, keep_default_na=False
        )
        if catalog.empty:
            raise ValueError("The catalog is empty.")
        service = CompareService()
        catalog_ids = catalog["image_id"].to_numpy(dtype=service.sequential.image_ids.dtype)
        if not np.array_equal(catalog_ids, service.sequential.image_ids):
            raise ValueError("Catalog IDs must match the complete search artifact row order.")

        row = catalog.iloc[0]
        query_id = catalog_ids[0].item()
        print(f"Query ID: {query_id}", flush=True)
        try:
            with Image.open(REPO_ROOT / row.image_path) as image:
                query = encode_image(image)
        except Exception as error:
            raise ValueError(
                f"Cannot encode image_id={row.image_id!r}, image_path={row.image_path!r}: {error}"
            ) from error

        comparison = service.compare(query, k=5, nprobe=8)
        config = comparison["config"]
        print(f"N: {config['dataset_size']}")
        print(f"Dimension: {config['dimension']}")
        print(f"K: {config['k']}")
        print(f"nlist: {config['nlist']}")
        print(f"nprobe: {config['nprobe']}")
        for engine, label in (("sequential", "Sequential"), ("faiss", "FAISS")):
            result = comparison[engine]
            print(f"\n{label}: latency={result['latency_ms']:.3f} ms")
            for item in result["results"]:
                print(
                    f"{item['rank']}. image_id={item['image_id']} "
                    f"score={item['score']:.8f}"
                )

        metrics = comparison["metrics"]
        print(f"\nOverlap count: {metrics['overlap_count']}")
        print(f"ANN Recall@5: {metrics['recall_at_k']:.3f}")
        print(f"Speedup: {format_speedup(metrics['speedup'])}")

        # Encode once and reuse the same services/query for each separate observation.
        print("\nnprobe | FAISS latency (ms) | Recall@5 | overlap | speedup")
        for nprobe in (1, 4, 8, 16, 32, 64):
            observed = service.compare(query, k=5, nprobe=nprobe)
            metrics = observed["metrics"]
            print(
                f"{nprobe:6d} | {observed['faiss']['latency_ms']:18.3f} | "
                f"{metrics['recall_at_k']:8.3f} | {metrics['overlap_count']:7d} | "
                f"{format_speedup(metrics['speedup'])}"
            )
        print(
            "\nThese single-query observations are not a formal benchmark. "
            "Recall and latency need not change monotonically in individual measurements."
        )
        print("Recall measures overlap with exact Top-K, not semantic correctness.")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
