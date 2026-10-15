"""FastAPI integration contract, using shared lightweight service fixtures."""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.services import api_service


def image_bytes(format="PNG", mode="RGB") -> bytes:
    buffer = BytesIO()
    Image.new(mode, (4, 4)).save(buffer, format=format)
    return buffer.getvalue()


def upload(format="PNG", mode="RGB"):
    suffix, mime = ("jpg", "jpeg") if format == "JPEG" else ("png", "png")
    return {"file": (f"query.{suffix}", image_bytes(format, mode), f"image/{mime}")}


METADATA_FIELDS = set(api_service.CATALOG_COLUMNS) | {"rank", "score"}


@pytest.mark.parametrize("engine", ["sequential", "faiss"])
@pytest.mark.parametrize("format,mode", [("JPEG", "RGB"), ("PNG", "RGBA")])
@pytest.mark.parametrize("k", [1, 5])
def test_search_returns_ranked_metadata_and_core_latency(client, engine, format, mode, k):
    response = client.post("/api/search", data={"engine": engine, "k": k}, files=upload(format, mode))
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == engine
    assert data["k"] == k
    assert data["nprobe"] == (8 if engine == "faiss" else None)
    assert data["encoding_ms"] >= 0
    assert data["search_latency_ms"] == (1.0 if engine == "faiss" else 2.0)
    assert len(data["results"]) == k
    assert [row["rank"] for row in data["results"]] == list(range(1, k + 1))
    assert [row["score"] for row in data["results"]] == sorted(
        [row["score"] for row in data["results"]], reverse=True,
    )
    for row in data["results"]:
        assert METADATA_FIELDS == set(row)
        assert row["product_id"] == row["image_id"]
        assert row["product_display_name"] == f"Product {row['image_id']}"
    encoded_image = app.state.services.encoder.encode_image.call_args.args[0]
    assert encoded_image.mode == "RGB"


def test_invalid_image_uses_shared_error_format(client) -> None:
    response = client.post(
        "/api/search",
        data={"engine": "sequential", "k": 5},
        files={"file": ("query.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"
    assert response.json()["error"]["message"]


@pytest.mark.parametrize("content", [b"not a jpeg", b"", image_bytes("JPEG")[:40], image_bytes("GIF")])
def test_spoofed_empty_or_broken_image_is_rejected(client, content) -> None:
    response = client.post(
        "/api/search",
        data={"engine": "sequential", "k": 5},
        files={"file": ("query.jpg", content, "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


@pytest.mark.parametrize("endpoint", ["search", "compare"])
@pytest.mark.parametrize("k", [0, -1, 11, "invalid"])
def test_invalid_k_returns_400(client, endpoint, k) -> None:
    response = client.post(
        f"/api/{endpoint}", data={"k": k}, files=upload(),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_K"


def test_invalid_engine_returns_400(client):
    response = client.post("/api/search", data={"engine": "unknown"}, files=upload())
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_ENGINE"


@pytest.mark.parametrize("endpoint", ["search", "compare"])
@pytest.mark.parametrize("nprobe", [0, -1, 65, "invalid"])
def test_invalid_nprobe_returns_400(client, endpoint, nprobe):
    response = client.post(
        f"/api/{endpoint}", data={"engine": "faiss", "nprobe": nprobe}, files=upload(),
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_NPROBE"


def test_sequential_ignores_nprobe(client):
    response = client.post("/api/search", data={"nprobe": -100}, files=upload())
    assert response.status_code == 200
    assert response.json()["nprobe"] is None
    assert app.state.services.faiss.search.call_count == 0


@pytest.mark.parametrize("endpoint", ["search", "compare"])
def test_missing_upload_returns_400(client, endpoint):
    response = client.post(f"/api/{endpoint}", data={"k": 5})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_search_returns_503_when_engine_is_not_ready(client) -> None:
    app.state.services = None
    response = client.post(
        "/api/search",
        data={"engine": "faiss", "k": 5, "nprobe": 4},
        files=upload(),
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "SEARCH_ENGINE_NOT_READY"


def test_compare_returns_503_when_engines_are_not_ready(client) -> None:
    app.state.services = None
    response = client.post(
        "/api/compare",
        data={"k": 5, "nprobe": 4},
        files=upload(),
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "SEARCH_ENGINE_NOT_READY"


def test_benchmark_returns_503_when_file_is_missing(client):
    (api_service.BENCHMARK_DIRECTORY / "nprobe_benchmark.csv").unlink()
    response = client.get("/api/benchmark")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "BENCHMARK_NOT_READY"
    assert "nprobe_benchmark.csv" in response.json()["error"]["message"]


def test_compare_encodes_once_and_reuses_same_embedding(client):
    response = client.post("/api/compare", files=upload())
    assert response.status_code == 200
    data = response.json()
    services = app.state.services
    services.encoder.encode_image.assert_called_once()
    assert services.sequential.search.call_args.args[0] is services.faiss.search.call_args.args[0]
    assert data["query"]["encoding_ms"] >= 0
    assert data["config"] == {"dataset_size": 10, "dimension": 2048, "k": 5, "nlist": 64, "nprobe": 8}
    for engine, latency in [("sequential", 2.0), ("faiss", 1.0)]:
        assert data[engine]["latency_ms"] == latency
        assert len(data[engine]["results"]) == 5
        assert METADATA_FIELDS == set(data[engine]["results"][0])
    assert data["metrics"] == {"overlap_count": 5, "recall_at_k": 1.0, "speedup": 2.0}


def test_services_and_catalog_are_reused_across_requests(client, monkeypatch):
    monkeypatch.setattr(api_service.pd, "read_csv", lambda *a, **kw: pytest.fail("Catalog reloaded"))
    for endpoint in ["search", "search", "compare"]:
        assert client.post(f"/api/{endpoint}", files=upload()).status_code == 200
    for factory in client.service_factories:
        factory.assert_called_once()


def test_benchmark_reads_saved_aggregates_without_search_resources(client):
    app.state.services = None
    response = client.get("/api/benchmark")
    assert response.status_code == 200
    assert response.json() == {
        "nprobe_benchmark": [{"nprobe": 8, "recall_at_k": 1.0}],
        "scalability_benchmark": [{"dataset_size": 1000, "nlist": 25, "nprobe": 3}],
        "metadata": {"seed": 42, "num_queries": 100},
    }


def test_benchmark_bad_metadata_has_clear_error(client):
    (api_service.BENCHMARK_DIRECTORY / "benchmark_metadata.json").write_text("bad json", encoding="utf-8")
    response = client.get("/api/benchmark")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "BENCHMARK_NOT_READY"


def test_benchmark_preserves_saved_statistics_and_missing_values(client):
    (api_service.BENCHMARK_DIRECTORY / "nprobe_benchmark.csv").write_text(
        "nprobe,mean_ms,optional_ms\n8,1.2345678901234567,\n", encoding="utf-8",
    )
    response = client.get("/api/benchmark")
    assert response.status_code == 200
    assert response.json()["nprobe_benchmark"] == [
        {"nprobe": 8, "mean_ms": 1.2345678901234567, "optional_ms": None},
    ]


def test_encoder_failure_returns_safe_503(client):
    app.state.services.encoder.encode_image.side_effect = RuntimeError("private model path")
    response = client.post("/api/search", files=upload())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_READY"
    assert "private" not in response.text


def test_too_few_faiss_candidates_returns_400(client):
    app.state.services.faiss.search.side_effect = RuntimeError("FAISS found only 2 of 5 neighbors")
    response = client.post("/api/search", data={"engine": "faiss"}, files=upload())
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_NPROBE"


def test_unexpected_error_does_not_expose_traceback(client):
    with TestClient(app, raise_server_exceptions=False) as safe_client:
        app.state.services.sequential.search.side_effect = RuntimeError("private artifact path")
        response = safe_client.post("/api/search", files=upload())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "private" not in response.text


def test_startup_without_artifacts_keeps_health_and_benchmark_available(client, monkeypatch):
    import backend.main as main
    def unavailable():
        raise FileNotFoundError("missing local artifacts")
    monkeypatch.setattr(main, "APIServices", unavailable)
    with TestClient(app) as unavailable_client:
        health = unavailable_client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert health.json()["faiss_ready"] is False
        assert unavailable_client.get("/api/benchmark").status_code == 200
        assert unavailable_client.post("/api/search", files=upload()).status_code == 503


def test_index_rebuild_returns_503_when_resources_are_not_ready(client) -> None:
    response = client.post("/api/v1/index/rebuild")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "INDEX_NOT_READY"


def test_catalog_crud_uses_in_memory_contract(client) -> None:
    record = {
        "image_id": "img_001",
        "product_id": "P001",
        "image_path": "data/processed/img_001.jpg",
        "category": "shoe",
    }

    assert client.get("/api/v1/catalog").json() == []

    created = client.post("/api/v1/catalog", json=record)
    assert created.status_code == 201
    assert created.json() == record

    listed = client.get("/api/v1/catalog")
    assert listed.status_code == 200
    assert listed.json() == [record]

    fetched = client.get("/api/v1/catalog/img_001")
    assert fetched.status_code == 200
    assert fetched.json() == record

    updated = client.patch(
        "/api/v1/catalog/img_001",
        json={"category": "sneaker"},
    )
    assert updated.status_code == 200
    assert updated.json()["category"] == "sneaker"
    assert updated.json()["product_id"] == "P001"

    deleted = client.delete("/api/v1/catalog/img_001")
    assert deleted.status_code == 200
    assert deleted.json()["image_id"] == "img_001"

    missing = client.get("/api/v1/catalog/img_001")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "CATALOG_NOT_FOUND"
