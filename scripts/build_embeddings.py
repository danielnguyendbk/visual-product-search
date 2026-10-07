"""Build ResNet50 embeddings in the exact row order of catalog.csv."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = REPO_ROOT / "data/processed"

# Support running this file directly from the repository or another directory.
sys.path.insert(0, str(REPO_ROOT))


def build_embeddings(batch_size: int, limit: int | None) -> None:
    from backend.services.encoder import get_encoder

    catalog = pd.read_csv(
        PROCESSED_PATH / "catalog.csv", dtype=str, keep_default_na=False
    )
    missing_columns = {"image_id", "image_path"}.difference(catalog.columns)
    if missing_columns:
        raise ValueError(f"Missing catalog columns: {', '.join(sorted(missing_columns))}")
    if catalog.empty:
        raise ValueError("The catalog must contain at least one image.")
    if not catalog["image_id"].is_unique or catalog["image_id"].eq("").any():
        raise ValueError("Catalog image_id values must be non-empty and unique.")

    catalog_rows = len(catalog)
    if limit is not None:
        catalog = catalog.iloc[:limit]

    ids = catalog["image_id"].tolist()
    if catalog["image_id"].str.fullmatch(r"[+-]?[0-9]+").all():
        image_ids = np.asarray(ids, dtype=np.int64)
    else:
        image_ids = np.asarray(ids, dtype=np.str_)

    print("Loading ResNet50 with IMAGENET1K_V2 weights...", flush=True)
    encoder = get_encoder()
    print(f"Device: {encoder.device}", flush=True)
    print(f"Model: {encoder.model_name}")
    print(f"Weights: ResNet50_Weights.{encoder.weights.name}")
    print(f"Embedding dimension: {encoder.embedding_dimension}")
    print(f"Batch size: {batch_size}")
    print(f"Images: {len(catalog)} of {catalog_rows} catalog rows", flush=True)

    embeddings = np.empty((len(catalog), encoder.embedding_dimension), dtype=np.float32)
    started = perf_counter()
    for start in range(0, len(catalog), batch_size):
        end = min(start + batch_size, len(catalog))
        batch_rows = list(catalog.iloc[start:end].itertuples(index=False))
        images = []
        try:
            for row in batch_rows:
                try:
                    relative_path = Path(row.image_path)
                    if not row.image_path or relative_path.is_absolute() or relative_path.drive:
                        raise ValueError("image_path must be relative to the repository root.")
                    with Image.open(REPO_ROOT / relative_path) as source:
                        images.append(source.convert("RGB"))
                except Exception as error:
                    raise ValueError(
                        f"Cannot read image_id={row.image_id!r}, "
                        f"image_path={row.image_path!r}: {error}"
                    ) from error

            try:
                embeddings[start:end] = encoder.encode_batch(images)
            except Exception as error:
                affected = "; ".join(
                    f"image_id={row.image_id!r}, image_path={row.image_path!r}"
                    for row in batch_rows
                )
                raise RuntimeError(f"Cannot encode batch [{affected}]: {error}") from error
        finally:
            for image in images:
                image.close()
        print(f"Encoded: {end}/{len(catalog)}", flush=True)

    encode_seconds = perf_counter() - started
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or Inf.")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise ValueError("Embeddings must have unit L2 norms.")

    config = {
        "model": encoder.model_name,
        "weights": f"ResNet50_Weights.{encoder.weights.name}",
        "embedding_dimension": encoder.embedding_dimension,
        "normalization": "L2",
        "preprocessing": "ResNet50_Weights.IMAGENET1K_V2.transforms()",
        "preprocessing_details": str(encoder.preprocess),
        "device": str(encoder.device),
        "batch_size": batch_size,
        "image_count": len(catalog),
        "catalog_rows": catalog_rows,
        "encode_seconds": encode_seconds,
    }

    # Write only after every selected catalog image has encoded successfully.
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_PATH / "embeddings.npy", embeddings, allow_pickle=False)
    np.save(PROCESSED_PATH / "image_ids.npy", image_ids, allow_pickle=False)
    (PROCESSED_PATH / "encoder_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Encode time: {encode_seconds:.2f} seconds")
    print("Saved embeddings.npy, image_ids.npy, encoder_config.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16, help="Images per batch (default: 16).")
    parser.add_argument("--limit", type=int, default=None, help="Encode only the first N catalog rows.")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer.")

    try:
        build_embeddings(args.batch_size, args.limit)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
