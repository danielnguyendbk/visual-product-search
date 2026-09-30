"""Health and application discovery tests."""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_returns_readiness_state() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_ready": False,
        "embeddings_ready": False,
        "index_ready": False,
        "catalog_size": 0,
    }


def test_openapi_loads_and_lists_v1_routes() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert {
        "/api/v1/health",
        "/api/v1/search",
        "/api/v1/compare",
        "/api/v1/benchmark",
        "/api/v1/catalog",
        "/api/v1/catalog/{image_id}",
        "/api/v1/index/rebuild",
    }.issubset(paths)
