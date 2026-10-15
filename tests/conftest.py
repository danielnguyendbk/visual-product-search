"""API fixtures: real integration layer and CompareService, lightweight core doubles."""

from types import SimpleNamespace
from unittest.mock import Mock

import faiss
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import api_service
from backend.services.catalog_service import catalog_service


@pytest.fixture
def client(tmp_path, monkeypatch):
    image_ids = np.arange(101, 111, dtype=np.int64)
    query = np.zeros(2048, dtype=np.float32)
    query[0] = 1
    records = [
        {
            "image_id": int(image_id), "product_id": int(image_id),
            "image_path": f"data/raw/images/{image_id}.jpg",
            "master_category": "Apparel", "sub_category": "Topwear",
            "article_type": "Shirts", "base_colour": "Blue",
            "product_display_name": f"Product {image_id}",
        }
        for image_id in image_ids
    ]
    catalog_path = tmp_path / "catalog.csv"
    pd.DataFrame(records).to_csv(catalog_path, index=False)
    monkeypatch.setattr(api_service, "CATALOG_PATH", catalog_path)

    sequential = SimpleNamespace(
        embeddings=np.tile(query, (10, 1)), image_ids=image_ids, last_search_latency_ms=None,
    )
    ann = SimpleNamespace(
        index=SimpleNamespace(ntotal=10, d=2048, metric_type=faiss.METRIC_INNER_PRODUCT, nlist=64),
        image_ids=image_ids.copy(), last_search_latency_ms=None,
    )

    def results(k):
        return [
            {"rank": i + 1, "image_id": int(image_ids[i]), "score": 1.0 - i / 10}
            for i in range(k)
        ]

    def sequential_search(embedding, k=5):
        sequential.last_search_latency_ms = 2.0
        return results(k)

    def faiss_search(embedding, k=5, nprobe=8):
        ann.last_search_latency_ms = 1.0
        return results(k)

    sequential.search = Mock(side_effect=sequential_search)
    ann.search = Mock(side_effect=faiss_search)
    encoder = SimpleNamespace(encode_image=Mock(return_value=query))
    factories = [Mock(return_value=sequential), Mock(return_value=ann), Mock(return_value=encoder)]
    monkeypatch.setattr(api_service, "SequentialSearchService", factories[0])
    monkeypatch.setattr(api_service, "FaissSearchService", factories[1])
    monkeypatch.setattr(api_service, "get_encoder", factories[2])

    benchmark_directory = tmp_path / "benchmark_results"
    benchmark_directory.mkdir()
    (benchmark_directory / "nprobe_benchmark.csv").write_text(
        "nprobe,recall_at_k\n8,1.0\n", encoding="utf-8",
    )
    (benchmark_directory / "scalability_benchmark.csv").write_text(
        "dataset_size,nlist,nprobe\n1000,25,3\n", encoding="utf-8",
    )
    (benchmark_directory / "benchmark_metadata.json").write_text(
        '{"seed": 42, "num_queries": 100}', encoding="utf-8",
    )
    monkeypatch.setattr(api_service, "BENCHMARK_DIRECTORY", benchmark_directory)
    catalog_service.clear()
    with TestClient(app) as test_client:
        test_client.service_factories = factories
        yield test_client
    catalog_service.clear()
