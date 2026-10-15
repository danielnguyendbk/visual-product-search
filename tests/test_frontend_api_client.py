"""HTTP contracts and user-facing failures for the frontend client."""

from unittest.mock import Mock

import pytest
import requests

from frontend import api_client


@pytest.fixture
def http_request(monkeypatch):
    response = Mock(ok=True, status_code=200)
    response.json.return_value = {}
    send = Mock(return_value=response)
    monkeypatch.setattr(api_client.requests, "request", send)
    monkeypatch.setattr(api_client, "BACKEND_URL", "http://localhost:8000")
    return send


@pytest.mark.parametrize("engine", ["sequential", "faiss"])
def test_search_upload_contract_and_nprobe(http_request, engine):
    payload = {
        "engine": engine, "k": 5, "nprobe": 8 if engine == "faiss" else None,
        "encoding_ms": 100.0, "search_latency_ms": 1.0, "results": [],
    }
    http_request.return_value.json.return_value = payload
    assert api_client.search_image("query.png", b"pixels", "image/png", engine=engine) == payload
    http_request.assert_called_once_with(
        "POST", "http://localhost:8000/api/search", timeout=api_client.TIMEOUT,
        files={"file": ("query.png", b"pixels", "image/png")},
        data={"engine": engine, "k": 5, **({"nprobe": 8} if engine == "faiss" else {})},
    )


def test_compare_sends_one_upload_in_one_request(http_request):
    payload = {"query": {}, "config": {}, "sequential": {}, "faiss": {}, "metrics": {}}
    http_request.return_value.json.return_value = payload
    assert api_client.compare_image("query.jpg", b"pixels", "image/jpeg") == payload
    http_request.assert_called_once_with(
        "POST", "http://localhost:8000/api/compare", timeout=api_client.TIMEOUT,
        files={"file": ("query.jpg", b"pixels", "image/jpeg")}, data={"k": 5, "nprobe": 8},
    )


def test_benchmark_only_reads_saved_results(http_request):
    payload = {"nprobe_benchmark": [], "scalability_benchmark": [], "metadata": {}}
    http_request.return_value.json.return_value = payload
    assert api_client.get_benchmark() == payload
    http_request.assert_called_once_with("GET", "http://localhost:8000/api/benchmark", timeout=api_client.TIMEOUT)


def test_health_returns_readiness(http_request):
    payload = {"status": "ok", "catalog_size": 5000, "embedding_dimension": 2048, "faiss_ready": True}
    http_request.return_value.json.return_value = payload
    assert api_client.health() == payload


@pytest.mark.parametrize("error", [requests.Timeout, requests.ConnectionError, requests.RequestException])
def test_network_errors_have_safe_messages(http_request, error):
    http_request.side_effect = error("private connection details")
    with pytest.raises(api_client.BackendError) as caught:
        api_client.health()
    assert "private" not in str(caught.value)
    assert str(caught.value)


@pytest.mark.parametrize("status", [400, 503])
def test_backend_error_message_is_preserved(http_request, status):
    http_request.return_value.ok = False
    http_request.return_value.status_code = status
    http_request.return_value.json.return_value = {"error": {"code": "INVALID_IMAGE", "message": "Image is invalid."}}
    with pytest.raises(api_client.BackendError, match="Image is invalid"):
        api_client.health()


def test_non_json_response_is_handled(http_request):
    http_request.return_value.json.side_effect = ValueError("private invalid body")
    with pytest.raises(api_client.BackendError, match="JSON"):
        api_client.health()


@pytest.mark.parametrize("payload", [[], None, {"unrelated": True}])
def test_unexpected_json_shape_is_handled(http_request, payload):
    http_request.return_value.json.return_value = payload
    with pytest.raises(api_client.BackendError):
        api_client.health()


def test_http_error_without_shared_envelope_is_handled(http_request):
    http_request.return_value.ok = False
    http_request.return_value.status_code = 502
    http_request.return_value.json.return_value = {"error": "unavailable"}
    with pytest.raises(api_client.BackendError, match="502"):
        api_client.health()
