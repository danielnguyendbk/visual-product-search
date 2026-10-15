"""FAISS artifact validation and searches against the offline 5000-vector index."""

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from backend.services.faiss_search import (
    INDEXES_PATH,
    PROCESSED_PATH,
    FaissSearchService,
)
from scripts import build_faiss_index as builder


@pytest.fixture(scope="module")
def service():
    if not (INDEXES_PATH / "faiss_ivf.index").is_file():
        pytest.skip("Run python scripts/build_faiss_index.py before FAISS integration tests.")
    return FaissSearchService()


@pytest.fixture(scope="module")
def query(service):
    return np.load(PROCESSED_PATH / "embeddings.npy", allow_pickle=False)[0].copy()


def test_loaded_index_and_config(service):
    assert isinstance(service.index, faiss.IndexIVFFlat)
    assert isinstance(faiss.downcast_index(service.index.quantizer), faiss.IndexFlatIP)
    assert service.index.is_trained
    assert service.index.ntotal == len(service.image_ids) == 5000
    assert service.index.d == 2048
    assert service.index.nlist == 64
    assert service.index.metric_type == faiss.METRIC_INNER_PRODUCT
    config = json.loads((INDEXES_PATH / "faiss_config.json").read_text(encoding="utf-8"))
    assert config["index_type"] == "IndexIVFFlat"
    assert config["metric"] == "inner_product"
    assert config["dimension"] == 2048 and config["num_vectors"] == 5000
    assert config["nlist"] == 64 and config["default_nprobe"] == 8
    assert config["build_seconds"] >= 0


@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("nprobe", [1, 8, 64])
def test_top_k_and_existing_embedding(service, query, k, nprobe):
    results = service.search(query, k=k, nprobe=nprobe)
    assert len(results) == k
    assert [row["rank"] for row in results] == list(range(1, k + 1))
    scores = [row["score"] for row in results]
    assert scores == sorted(scores, reverse=True)
    assert np.isfinite(scores).all()
    assert results[0]["image_id"] == service.image_ids[0].item()
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert all(set(row) == {"rank", "image_id", "score"} for row in results)
    assert service.index.nprobe == nprobe
    assert service.last_search_latency_ms is not None
    assert np.isfinite(service.last_search_latency_ms) and service.last_search_latency_ms >= 0


@pytest.mark.parametrize("k", [0, -1, 5001, 1.5, True])
def test_invalid_k(service, query, k):
    with pytest.raises(ValueError, match="K must be an integer"):
        service.search(query, k=k)


@pytest.mark.parametrize("nprobe", [0, -1, 65, 1.5, True])
def test_invalid_nprobe(service, query, nprobe):
    with pytest.raises(ValueError, match="nprobe must be an integer"):
        service.search(query, nprobe=nprobe)


@pytest.mark.parametrize("shape", [(2047,), (2049,), (2, 2048), (2048, 1)])
def test_invalid_query_dimension(service, shape):
    with pytest.raises(ValueError, match="Query must have shape"):
        service.search(np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_query(service, query, value):
    invalid = query.copy()
    invalid[0] = value
    with pytest.raises(ValueError, match="Query must be finite"):
        service.search(invalid)


@pytest.mark.parametrize("scale", [0, 2])
def test_unnormalized_query(service, query, scale):
    with pytest.raises(ValueError, match="Query must be L2 normalized"):
        service.search(query * scale)


def test_float64_row_query(service, query):
    results = service.search(query.astype(np.float64).reshape(1, 2048))
    assert results[0]["image_id"] == service.image_ids[0].item()


def test_default_nprobe_is_restored_on_reuse(service, query):
    service.search(query, nprobe=64)
    service.search(query)
    assert service.index.nprobe == 8


def test_index_and_ids_are_loaded_once(query, monkeypatch):
    original_read, original_load = faiss.read_index, np.load
    index_reads, id_reads = [], []

    def read_index(path):
        index_reads.append(path)
        return original_read(path)

    def load_ids(path, **kwargs):
        id_reads.append(path)
        assert kwargs["allow_pickle"] is False
        return original_load(path, **kwargs)

    monkeypatch.setattr(faiss, "read_index", read_index)
    monkeypatch.setattr(np, "load", load_ids)
    reused = FaissSearchService()
    reused.search(query, k=1)
    reused.search(query, k=5)
    assert len(index_reads) == len(id_reads) == 1


def test_missing_neighbors_never_map_to_last_image_id(service, query):
    # Searching only one list cannot return all 5000 stored vectors.
    with pytest.raises(RuntimeError, match="increase nprobe or reduce K"):
        service.search(query, k=service.index.ntotal, nprobe=1)


@pytest.mark.parametrize("case", ["untrained", "dimension", "metric", "nlist", "flat"])
def test_invalid_index_is_rejected(tmp_path, case):
    dimension = 16 if case == "dimension" else 2048
    nlist = 32 if case == "nlist" else 64
    if case == "flat":
        index = faiss.IndexFlatIP(dimension)
    elif case == "metric":
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(dimension), dimension, nlist, faiss.METRIC_L2
        )
    else:
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(dimension), dimension, nlist, faiss.METRIC_INNER_PRODUCT
        )
    index_path, ids_path = tmp_path / "invalid.index", tmp_path / "image_ids.npy"
    faiss.write_index(index, str(index_path))
    np.save(ids_path, np.empty(0, dtype=np.int64))
    with pytest.raises(ValueError):
        FaissSearchService(index_path, ids_path)


@pytest.mark.parametrize("case", ["row_count", "shape", "object"])
def test_invalid_id_mapping_is_rejected(service, tmp_path, case):
    image_ids = service.image_ids.copy()
    if case == "row_count":
        image_ids = image_ids[:-1]
    elif case == "shape":
        image_ids = image_ids.reshape(1, -1)
    else:
        image_ids = image_ids.astype(object)
    ids_path = tmp_path / "image_ids.npy"
    np.save(ids_path, image_ids)
    with pytest.raises(ValueError):
        FaissSearchService(image_ids_path=ids_path)


@pytest.mark.parametrize(
    "case", ["dimension", "dtype", "too_few", "id_count", "nan", "inf", "not_normalized"]
)
def test_builder_rejects_invalid_artifacts_before_writing(tmp_path, monkeypatch, case):
    embeddings = np.zeros((64, 2048), dtype=np.float32)
    embeddings[:, 0] = 1
    image_ids = np.arange(64, dtype=np.int64)
    if case == "dimension":
        embeddings = embeddings[:, :-1]
    elif case == "dtype":
        embeddings = embeddings.astype(np.float64)
    elif case == "too_few":
        embeddings, image_ids = embeddings[:-1], image_ids[:-1]
    elif case == "id_count":
        image_ids = image_ids[:-1]
    elif case == "nan":
        embeddings[0, 0] = np.nan
    elif case == "inf":
        embeddings[0, 0] = np.inf
    else:
        embeddings[0] *= 2
    np.save(tmp_path / "embeddings.npy", embeddings)
    np.save(tmp_path / "image_ids.npy", image_ids)
    output = tmp_path / "indexes"
    monkeypatch.setattr(builder, "PROCESSED_PATH", tmp_path)
    monkeypatch.setattr(builder, "INDEXES_PATH", output)
    with pytest.raises(ValueError):
        builder.build_index()
    assert not output.exists()
