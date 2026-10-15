"""Health and application discovery tests."""

def test_health_returns_readiness_state(client) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_ready": True,
        "embeddings_ready": True,
        "index_ready": True,
        "catalog_size": 10,
        "embedding_dimension": 2048,
        "faiss_ready": True,
    }


def test_openapi_loads_and_lists_v1_routes(client) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert {
        "/api/health",
        "/api/search",
        "/api/compare",
        "/api/benchmark",
        "/api/v1/health",
        "/api/v1/search",
        "/api/v1/compare",
        "/api/v1/benchmark",
        "/api/v1/catalog",
        "/api/v1/catalog/{image_id}",
        "/api/v1/index/rebuild",
    }.issubset(paths)
    assert "get" in paths["/api/benchmark"]
    assert "post" not in paths["/api/benchmark"]
    assert client.get("/api/v1/health").json() == client.get("/api/health").json()
