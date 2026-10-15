"""Compare the real search services and verify comparison-specific safeguards."""

from types import SimpleNamespace
from unittest.mock import Mock

import faiss
import numpy as np
import pytest

from backend.services import compare as compare_module
from backend.services.compare import CompareService
from backend.services.faiss_search import INDEXES_PATH


@pytest.fixture(scope="module")
def service():
    if not (INDEXES_PATH / "faiss_ivf.index").is_file():
        pytest.skip("Build the offline FAISS index before Compare integration tests.")
    return CompareService()


@pytest.fixture(scope="module")
def query(service):
    return service.sequential.embeddings[0].copy()


def test_output_structure(service, query):
    result = service.compare(query, k=5, nprobe=8)
    assert set(result) == {"config", "sequential", "faiss", "metrics"}
    assert result["config"] == {
        "dataset_size": 5000,
        "dimension": 2048,
        "k": 5,
        "nlist": 64,
        "nprobe": 8,
    }
    for engine in ("sequential", "faiss"):
        assert set(result[engine]) == {"latency_ms", "results"}
    assert set(result["metrics"]) == {"overlap_count", "recall_at_k", "speedup"}


@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("nprobe", [1, 8, 64])
def test_real_results_and_metrics(service, query, k, nprobe):
    result = service.compare(query, k=k, nprobe=nprobe)
    id_sets = []
    for engine in ("sequential", "faiss"):
        rows = result[engine]["results"]
        assert len(rows) == k
        scores = [row["score"] for row in rows]
        assert scores == sorted(scores, reverse=True)
        ids = {row["image_id"] for row in rows}
        assert len(ids) == k
        id_sets.append(ids)
    metrics = result["metrics"]
    assert 0 <= metrics["overlap_count"] <= k
    assert metrics["overlap_count"] == len(id_sets[0] & id_sets[1])
    assert 0 <= metrics["recall_at_k"] <= 1
    assert metrics["recall_at_k"] == metrics["overlap_count"] / k
    sequential_ms = result["sequential"]["latency_ms"]
    faiss_ms = result["faiss"]["latency_ms"]
    if sequential_ms > 0 and faiss_ms > 0:
        assert metrics["speedup"] > 0
        assert metrics["speedup"] == pytest.approx(sequential_ms / faiss_ms)


def test_same_query_and_k_are_passed_to_both_engines(service, query, monkeypatch):
    original_query = query.copy()
    sequential_search = Mock(wraps=service.sequential.search)
    faiss_search = Mock(wraps=service.faiss.search)
    monkeypatch.setattr(service.sequential, "search", sequential_search)
    monkeypatch.setattr(service.faiss, "search", faiss_search)
    service.compare(query, k=5, nprobe=8)
    assert sequential_search.call_count == faiss_search.call_count == 1
    assert sequential_search.call_args.args[0] is query
    assert faiss_search.call_args.args[0] is query
    assert sequential_search.call_args.kwargs == {"k": 5}
    assert faiss_search.call_args.kwargs == {"k": 5, "nprobe": 8}
    np.testing.assert_array_equal(query, original_query)


def test_reused_compare_does_not_initialize_engines_again(service, query, monkeypatch):
    sequential_constructor = Mock(side_effect=AssertionError("Unexpected engine reload"))
    faiss_constructor = Mock(side_effect=AssertionError("Unexpected engine reload"))
    monkeypatch.setattr(compare_module, "SequentialSearchService", sequential_constructor)
    monkeypatch.setattr(compare_module, "FaissSearchService", faiss_constructor)
    service.compare(query, nprobe=1)
    service.compare(query, nprobe=8)
    sequential_constructor.assert_not_called()
    faiss_constructor.assert_not_called()


@pytest.mark.parametrize("k", [0, -1, 5001, 1.5, True])
def test_invalid_k(service, query, k):
    with pytest.raises(ValueError, match="K must be an integer"):
        service.compare(query, k=k)


@pytest.mark.parametrize("nprobe", [0, -1, 65, 1.5, True])
def test_invalid_nprobe(service, query, nprobe):
    with pytest.raises(ValueError, match="nprobe must be an integer"):
        service.compare(query, nprobe=nprobe)


@pytest.mark.parametrize("shape", [(2047,), (2, 2048)])
def test_invalid_query_is_rejected_by_search_service(service, shape):
    with pytest.raises(ValueError, match="Query must have shape"):
        service.compare(np.zeros(shape, dtype=np.float32))


def make_results(image_ids):
    return [
        {"rank": rank, "image_id": image_id, "score": 1.0 - rank / 10}
        for rank, image_id in enumerate(image_ids, start=1)
    ]


@pytest.fixture
def controlled_results(service, monkeypatch):
    # Controlled outputs isolate metric math from noisy timing and ANN ranking.
    ids = service.sequential.image_ids[:6].tolist()
    sequential_rows = make_results(ids[:5])
    faiss_rows = make_results([ids[2], ids[0], ids[1], ids[5], ids[4]])
    monkeypatch.setattr(service.sequential, "search", Mock(return_value=sequential_rows))
    monkeypatch.setattr(service.faiss, "search", Mock(return_value=faiss_rows))
    monkeypatch.setattr(service.sequential, "last_search_latency_ms", 2.0)
    monkeypatch.setattr(service.faiss, "last_search_latency_ms", 1.0)
    return sequential_rows, faiss_rows


def test_overlap_ignores_rank_and_uses_engine_latencies(service, query, controlled_results):
    result = service.compare(query)
    assert result["metrics"] == {"overlap_count": 4, "recall_at_k": 0.8, "speedup": 2.0}
    assert result["sequential"]["latency_ms"] == 2.0
    assert result["faiss"]["latency_ms"] == 1.0


@pytest.mark.parametrize("sequential_ms, faiss_ms", [(2.0, 0.0), (0.0, 0.0), (0.0, 1.0)])
def test_zero_latency_is_safe(
    service, query, controlled_results, monkeypatch, sequential_ms, faiss_ms
):
    monkeypatch.setattr(service.sequential, "last_search_latency_ms", sequential_ms)
    monkeypatch.setattr(service.faiss, "last_search_latency_ms", faiss_ms)
    result = service.compare(query)
    expected = None if faiss_ms == 0 else 0.0
    assert result["metrics"]["speedup"] == expected


@pytest.mark.parametrize("engine", ["sequential", "faiss"])
@pytest.mark.parametrize("case", ["count", "duplicates", "unsorted", "non_finite"])
def test_inconsistent_results_are_rejected(service, query, controlled_results, engine, case):
    sequential_rows, faiss_rows = controlled_results
    rows = sequential_rows if engine == "sequential" else faiss_rows
    if case == "count":
        rows.pop()
    elif case == "duplicates":
        rows[-1]["image_id"] = rows[0]["image_id"]
    elif case == "unsorted":
        rows[1]["score"] = rows[0]["score"] + 0.1
    else:
        rows[0]["score"] = np.nan
    with pytest.raises(RuntimeError, match="Sequential|FAISS"):
        service.compare(query)


@pytest.mark.parametrize("latency", [None, -1.0, np.nan, np.inf])
def test_invalid_latency_is_rejected(service, query, controlled_results, monkeypatch, latency):
    monkeypatch.setattr(service.faiss, "last_search_latency_ms", latency)
    with pytest.raises(RuntimeError, match="finite, non-negative latency"):
        service.compare(query)


@pytest.mark.parametrize("case", ["dataset_size", "dimension", "metric", "mapping"])
def test_mismatched_engines_are_rejected(service, case):
    # Metadata-only doubles avoid mutating the real index or embeddings.
    sequential = SimpleNamespace(
        embeddings=service.sequential.embeddings, image_ids=service.sequential.image_ids
    )
    faiss_service = SimpleNamespace(
        index=SimpleNamespace(ntotal=5000, d=2048, metric_type=faiss.METRIC_INNER_PRODUCT),
        image_ids=service.faiss.image_ids,
    )
    if case == "dataset_size":
        faiss_service.index.ntotal -= 1
    elif case == "dimension":
        faiss_service.index.d -= 1
    elif case == "metric":
        faiss_service.index.metric_type = faiss.METRIC_L2
    else:
        faiss_service.image_ids = faiss_service.image_ids[::-1]
    with pytest.raises(ValueError, match="Both engines must"):
        CompareService(sequential=sequential, faiss_service=faiss_service)
