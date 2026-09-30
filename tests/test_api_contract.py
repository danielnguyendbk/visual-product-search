"""Milestone 0 API contract tests."""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.services.catalog_service import catalog_service

client = TestClient(app)


@pytest.fixture(autouse=True)
def empty_catalog():
    catalog_service.clear()
    yield
    catalog_service.clear()


def valid_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_invalid_image_uses_shared_error_format() -> None:
    response = client.post(
        "/api/v1/search",
        data={"engine": "sequential", "k": 5},
        files={"image": ("query.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"
    assert response.json()["error"]["message"]


def test_spoofed_image_content_is_rejected() -> None:
    response = client.post(
        "/api/v1/search",
        data={"engine": "sequential", "k": 5},
        files={"image": ("query.jpg", b"not a jpeg", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_invalid_k_returns_4xx() -> None:
    response = client.post(
        "/api/v1/search",
        data={"engine": "sequential", "k": 0},
        files={"image": ("query.png", valid_png(), "image/png")},
    )

    assert 400 <= response.status_code < 500
    assert response.json()["error"]["code"] == "INVALID_K"


def test_search_returns_503_when_engine_is_not_ready() -> None:
    response = client.post(
        "/api/v1/search",
        data={"engine": "faiss", "k": 5, "nprobe": 4},
        files={"image": ("query.png", valid_png(), "image/png")},
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "SEARCH_ENGINE_NOT_READY",
            "message": "Search engine is not ready.",
        }
    }


def test_compare_returns_503_when_engines_are_not_ready() -> None:
    response = client.post(
        "/api/v1/compare",
        data={"k": 5, "nprobe": 4},
        files={"image": ("query.png", valid_png(), "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "SEARCH_ENGINE_NOT_READY"


def test_benchmark_returns_503_when_resources_are_not_ready() -> None:
    response = client.post(
        "/api/v1/benchmark",
        json={
            "dataset_size": 2000,
            "nprobe": 4,
            "k": 5,
            "num_queries": 30,
            "repeats": 3,
        },
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "BENCHMARK_NOT_READY"


def test_index_rebuild_returns_503_when_resources_are_not_ready() -> None:
    response = client.post("/api/v1/index/rebuild")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "INDEX_NOT_READY"


def test_catalog_crud_uses_in_memory_contract() -> None:
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
