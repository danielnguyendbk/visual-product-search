"""Test benchmark logic only, without training an index or measuring performance."""

from types import SimpleNamespace

import numpy as np
import pytest

from backend.services import benchmark


def test_query_selection_is_deterministic_and_without_replacement():
    first = benchmark.select_query_indices(5000)
    second = benchmark.select_query_indices(5000)
    np.testing.assert_array_equal(first, second)
    assert len(first) == len(set(first)) == 100
    assert first.min() >= 0 and first.max() < 5000
    assert not np.array_equal(first, benchmark.select_query_indices(5000, seed=43))


@pytest.mark.parametrize("dataset_size, count", [(5, 6), (5, 0), (0, 1)])
def test_invalid_query_count(dataset_size, count):
    with pytest.raises(ValueError):
        benchmark.select_query_indices(dataset_size, count)


@pytest.mark.parametrize("position_order", [[0, 1, 2, 3, 4, 5], [1, 2, 0, 3, 4, 5]])
def test_self_match_is_removed_by_id_even_when_not_first(position_order):
    ids = np.array([90, 15, 70, 3, 61, 8], dtype=np.int64)
    assert benchmark.remove_self_matches(position_order, ids, query_id=90, k=5) == [15, 70, 3, 61, 8]


def test_missing_ann_neighbor_never_maps_to_last_id():
    ids = np.array([10, 20, 30, 40, 50, 60])
    result = benchmark.remove_self_matches([0, 1, -1, 2], ids, 10, k=5, require_k=False)
    assert result == [20, 30]
    with pytest.raises(ValueError, match="non-self results"):
        benchmark.remove_self_matches([0, 1, -1, 2], ids, 10, k=5)


@pytest.mark.parametrize("positions", [[-2], [8], [1, 1, 2, 3, 4]])
def test_invalid_neighbor_positions(positions):
    with pytest.raises(ValueError):
        benchmark.remove_self_matches(positions, np.arange(6), 0, 5)


@pytest.mark.parametrize("approximate, expected", [([5, 3, 2, 9, 1], 0.8), ([1, 2], 0.4), ([], 0.0)])
def test_recall_ignores_rank_and_counts_missing_neighbors_as_misses(approximate, expected):
    assert benchmark.recall_at_k([1, 2, 3, 4, 5], approximate, 5) == expected


def test_recall_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        benchmark.recall_at_k([1, 2, 3, 4, 5], [1, 1, 2], 5)


def test_latency_statistics_use_linear_percentiles():
    stats = benchmark.latency_statistics([1, 2, 3, 4, 5])
    assert stats == pytest.approx({"mean_ms": 3, "p50_ms": 3, "p95_ms": 4.8})


@pytest.mark.parametrize("values", [[], [-1], [np.nan], [np.inf], [[1, 2]]])
def test_invalid_latency_statistics(values):
    with pytest.raises(ValueError):
        benchmark.latency_statistics(values)


@pytest.mark.parametrize("sequential, approximate, expected", [(4, 2, 2), (4, 0, None), (0, 2, 0)])
def test_speedup_uses_matching_p50_statistics(sequential, approximate, expected):
    assert benchmark.speedup_from_p50(sequential, approximate) == expected


@pytest.mark.parametrize("sequential, approximate", [(-1, 2), (2, -1), (np.nan, 1), (1, np.inf)])
def test_invalid_speedup_inputs(sequential, approximate):
    with pytest.raises(ValueError):
        benchmark.speedup_from_p50(sequential, approximate)


@pytest.mark.parametrize("dataset_size, expected", [(1000, (25, 3)), (2000, (45, 6)), (5000, (71, 9))])
def test_subset_configuration_matches_requested_formula(dataset_size, expected):
    nlist, nprobe = benchmark.subset_index_config(dataset_size)
    assert (nlist, nprobe) == expected
    assert 1 <= nprobe <= nlist <= dataset_size
    assert 40 * nlist <= dataset_size


def test_small_subset_configuration_is_rejected():
    with pytest.raises(ValueError):
        benchmark.subset_index_config(7)


def mock_rows(monkeypatch):
    """Exercise warm-up and raw-record logic with a mocked search kernel."""
    index = SimpleNamespace(d=2048, nlist=8, nprobe=1)
    baseline = {
        "queries": np.zeros((2, 2048), dtype=np.float32),
        "top_k": [[1, 2, 3, 4, 5], [0, 2, 3, 4, 5]],
        "latencies_ms": [3.0, 5.0],
    }
    calls = []

    def fake_kernel(current_index, query, candidate_count):
        assert current_index is index and query.shape == (1, 2048)
        assert candidate_count == 6
        calls.append(query)
        return np.arange(6), 2.0

    monkeypatch.setattr(benchmark, "_faiss_kernel", fake_kernel)
    rows = benchmark.collect_faiss_rows(index, np.arange(8), np.array([0, 1]), baseline, 4, "nprobe")
    assert len(calls) == 4  # two warm-up calls plus two retained measurements
    assert len(rows) == 2 and index.nprobe == 4
    return rows


def test_raw_and_aggregate_output_schema_and_metric_math(monkeypatch):
    rows = mock_rows(monkeypatch)
    result = benchmark.aggregate_rows(rows)
    assert set(result) == set(benchmark.SUMMARY_FIELDS)
    assert result["query_count"] == 2
    assert result["sequential_mean_ms"] == result["sequential_p50_ms"] == 4.0
    assert result["sequential_p95_ms"] == pytest.approx(4.9)
    assert result["faiss_mean_ms"] == result["faiss_p50_ms"] == result["faiss_p95_ms"] == 2.0
    assert result["recall_at_5"] == 1.0 and result["speedup"] == 2.0
    assert rows[0]["query_image_id"] == 0 and rows[1]["query_image_id"] == 1
    assert rows[0]["faiss_result_count"] == 5 and rows[0]["overlap_count"] == 5
    scaled = benchmark.aggregate_rows(rows, index_build_ms=12.5)
    assert set(scaled) == set(benchmark.SUMMARY_FIELDS) | {"index_build_ms"}
    assert scaled["index_build_ms"] == 12.5


def test_aggregate_rejects_mixed_configs_and_empty_rows(monkeypatch):
    rows = mock_rows(monkeypatch)
    rows[1]["nprobe"] += 1
    with pytest.raises(ValueError, match="same experiment configuration"):
        benchmark.aggregate_rows(rows)
    with pytest.raises(ValueError):
        benchmark.aggregate_rows([])
