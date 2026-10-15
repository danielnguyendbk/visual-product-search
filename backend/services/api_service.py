"""Shared core services and catalog metadata used by the API."""

import json
from pathlib import Path
from threading import Lock
from time import perf_counter

import pandas as pd
from fastapi import Request
from PIL import Image

from backend.core.errors import APIError
from backend.schemas.common import ErrorCode
from backend.services.compare import CompareService
from backend.services.encoder import get_encoder
from backend.services.faiss_search import FaissSearchService
from backend.services.sequential_search import SequentialSearchService

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPOSITORY_ROOT / "data/processed/catalog.csv"
BENCHMARK_DIRECTORY = REPOSITORY_ROOT / "benchmark_results"
CATALOG_COLUMNS = (
    "image_id", "product_id", "image_path", "master_category", "sub_category",
    "article_type", "base_colour", "product_display_name",
)


def catalog_id(value: str) -> int | str:
    return int(value) if value.isascii() and value.isdecimal() else value


class APIServices:
    """Load once at startup; reuse existing core implementations for every query."""

    def __init__(self) -> None:
        catalog = pd.read_csv(CATALOG_PATH, dtype=str, keep_default_na=False)
        if not set(CATALOG_COLUMNS).issubset(catalog.columns):
            raise ValueError("Catalog is missing required metadata columns.")
        if catalog.empty or not catalog["image_id"].is_unique:
            raise ValueError("Catalog must have unique, nonempty image IDs.")
        if (catalog["image_id"] == "").any() or not catalog["product_id"].is_unique:
            raise ValueError("Catalog contains invalid IDs.")

        self.catalog = {}
        for record in catalog[list(CATALOG_COLUMNS)].to_dict(orient="records"):
            image_id = record["image_id"]
            record["image_id"] = catalog_id(image_id)
            record["product_id"] = catalog_id(record["product_id"])
            self.catalog[image_id] = record

        self.sequential = SequentialSearchService()
        self.faiss = FaissSearchService()
        self.compare = CompareService(self.sequential, self.faiss)
        if catalog["image_id"].tolist() != [str(value) for value in self.sequential.image_ids]:
            raise ValueError("Catalog and search artifacts have different image ID ordering.")
        self.encoder = get_encoder()
        # FAISS nprobe and each core service's last latency are mutable.
        self.lock = Lock()

    def encode(self, image: Image.Image):
        start = perf_counter()
        try:
            embedding = self.encoder.encode_image(image)
        except Exception as exc:
            raise APIError(503, ErrorCode.MODEL_NOT_READY, "Query encoder is unavailable.") from exc
        return embedding, (perf_counter() - start) * 1000

    def enrich(self, results: list[dict]) -> list[dict]:
        return [
            {**self.catalog[str(result["image_id"])], **result}
            for result in results
        ]


def get_services(request: Request) -> APIServices:
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise APIError(
            503, ErrorCode.SEARCH_ENGINE_NOT_READY,
            "Search resources are unavailable. Check the model, catalog and artifacts.",
        )
    return services


def read_benchmark_results() -> dict:
    """Read saved aggregates only; never invoke the benchmark runner."""
    paths = {
        "nprobe_benchmark": BENCHMARK_DIRECTORY / "nprobe_benchmark.csv",
        "scalability_benchmark": BENCHMARK_DIRECTORY / "scalability_benchmark.csv",
        "metadata": BENCHMARK_DIRECTORY / "benchmark_metadata.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise APIError(
                503, ErrorCode.BENCHMARK_NOT_READY,
                f"Saved benchmark file is missing: {path.name}.",
            )
    try:
        result = {}
        for name, path in paths.items():
            if name != "metadata":
                frame = pd.read_csv(path)
                # Preserve saved float precision and serialize missing values as null.
                result[name] = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
        result["metadata"] = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if not isinstance(result["metadata"], dict):
            raise ValueError("Benchmark metadata must be an object.")
        return result
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise APIError(
            500, ErrorCode.BENCHMARK_NOT_READY, "Saved benchmark files could not be read.",
        ) from exc
