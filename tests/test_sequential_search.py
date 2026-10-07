"""Exact ranking, validation, and artifact reuse without model downloads."""

import numpy as np
import pytest

from backend.services.sequential_search import SequentialSearchService


@pytest.fixture
def artifacts(tmp_path):
    # Known angles give independently predictable cosine scores for query [1, 0].
    angles = np.array([np.pi, np.pi / 3, 0, 2 * np.pi / 3, np.pi / 6, np.pi / 2, 5 * np.pi / 6])
    embeddings = np.zeros((7, 2048), dtype=np.float32)
    embeddings[:, 0] = np.cos(angles)
    embeddings[:, 1] = np.sin(angles)
    image_ids = np.array([71, 42, 900, 16, 5, 88, 34], dtype=np.int64)
    embeddings_path = tmp_path / "embeddings.npy"
    image_ids_path = tmp_path / "image_ids.npy"
    np.save(embeddings_path, embeddings)
    np.save(image_ids_path, image_ids)
    return embeddings_path, image_ids_path


@pytest.fixture
def service(artifacts):
    return SequentialSearchService(*artifacts)


@pytest.mark.parametrize("k", [1, 5, 7])
def test_exact_top_k_is_complete_and_sorted(service, k):
    results = service.search(service.embeddings[2], k=k)
    assert len(results) == k
    assert [result["rank"] for result in results] == list(range(1, k + 1))
    assert [result["image_id"] for result in results] == [900, 5, 42, 88, 16, 34, 71][:k]
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)
    assert scores == pytest.approx([1, np.sqrt(3) / 2, 0.5, 0, -0.5, -np.sqrt(3) / 2, -1][:k], abs=1e-6)
    assert all(set(result) == {"rank", "image_id", "score"} for result in results)


def test_existing_embedding_is_top_one_even_at_last_row(service):
    result = service.search(service.embeddings[-1], k=1)[0]
    assert result["image_id"] == service.image_ids[-1]
    assert result["score"] == pytest.approx(1.0, abs=1e-5)
    assert service.last_search_latency_ms is not None
    assert np.isfinite(service.last_search_latency_ms)
    assert service.last_search_latency_ms >= 0


@pytest.mark.parametrize("k", [0, -1, 8, 1.5, True])
def test_invalid_k_is_rejected(service, k):
    with pytest.raises(ValueError, match="K must be an integer"):
        service.search(service.embeddings[2], k=k)


@pytest.mark.parametrize("shape", [(2047,), (2049,), (2, 2048), (2048, 1)])
def test_wrong_query_dimension_is_rejected(service, shape):
    with pytest.raises(ValueError, match="Query must have shape"):
        service.search(np.zeros(shape, dtype=np.float32))


def test_float64_row_query_is_accepted(service):
    query = service.embeddings[2].astype(np.float64).reshape(1, 2048)
    assert service.search(query, k=1)[0]["image_id"] == 900


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_query_is_rejected(service, value):
    query = service.embeddings[2].copy()
    query[0] = value
    with pytest.raises(ValueError, match="Query must be finite"):
        service.search(query)


@pytest.mark.parametrize("scale", [0, 2])
def test_unnormalized_query_is_rejected(service, scale):
    with pytest.raises(ValueError, match="Query must be L2 normalized"):
        service.search(service.embeddings[2] * scale)


def test_artifacts_are_loaded_only_at_initialization(artifacts, monkeypatch):
    original_load = np.load
    calls = []

    def counted_load(*args, **kwargs):
        calls.append(args[0])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)
    service = SequentialSearchService(*artifacts)
    service.search(service.embeddings[2], k=1)
    service.search(service.embeddings[2], k=5)
    assert calls == list(artifacts)


def test_float64_embeddings_are_converted(artifacts):
    embeddings_path, _ = artifacts
    np.save(embeddings_path, np.load(embeddings_path).astype(np.float64))
    service = SequentialSearchService(*artifacts)
    assert service.embeddings.dtype == np.float32
    assert service.search(service.embeddings[2], k=1)[0]["image_id"] == 900


@pytest.mark.parametrize("case", ["row_count", "dimension", "ids_shape", "object_ids", "object_embeddings", "nan", "inf", "not_normalized"])
def test_invalid_artifacts_are_rejected(artifacts, case):
    embeddings_path, ids_path = artifacts
    embeddings = np.load(embeddings_path)
    image_ids = np.load(ids_path)
    if case == "row_count":
        image_ids = image_ids[:-1]
    elif case == "dimension":
        embeddings = embeddings[:, :-1]
    elif case == "ids_shape":
        image_ids = image_ids.reshape(1, -1)
    elif case == "object_ids":
        image_ids = image_ids.astype(object)
    elif case == "object_embeddings":
        embeddings = embeddings.astype(object)
    elif case == "nan":
        embeddings[0, 0] = np.nan
    elif case == "inf":
        embeddings[0, 0] = np.inf
    elif case == "not_normalized":
        embeddings[0] *= 2
    np.save(embeddings_path, embeddings)
    np.save(ids_path, image_ids)
    with pytest.raises(ValueError):
        SequentialSearchService(*artifacts)
