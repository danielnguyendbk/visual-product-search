"""Reproducible search-only experiments with catalog self-matches excluded."""

import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np
import pandas as pd

from backend.services.faiss_search import FaissSearchService
from backend.services.sequential_search import SequentialSearchService

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
QUERY_COUNT = 100
K = 5
WARMUP_COUNT = 10
NPROBES = (1, 4, 8, 16, 32, 64)
DATASET_SIZES = (1000, 2000, 5000)
CONFIG_FIELDS = ("dataset_size", "dimension", "nlist", "nprobe", "k", "seed")
SUMMARY_FIELDS = (
    "dataset_size", "dimension", "nlist", "nprobe", "query_count", "k", "seed",
    "sequential_mean_ms", "sequential_p50_ms", "sequential_p95_ms",
    "faiss_mean_ms", "faiss_p50_ms", "faiss_p95_ms", "recall_at_5", "speedup",
)


def select_query_indices(dataset_size: int, query_count: int = QUERY_COUNT, seed: int = SEED):
    if not 1 <= query_count <= dataset_size:
        raise ValueError("Query count must be between 1 and the dataset size.")
    return np.random.default_rng(seed).choice(dataset_size, size=query_count, replace=False)


def subset_index_config(dataset_size: int) -> tuple[int, int]:
    if dataset_size < 8:
        raise ValueError("At least 8 vectors are required for the subset IVF configuration.")
    nlist = min(round(np.sqrt(dataset_size)), max(8, dataset_size // 40))
    nprobe = max(1, round(nlist / 8))
    return nlist, nprobe


def remove_self_matches(positions, image_ids, query_id, k: int, require_k: bool = True) -> list:
    """Map valid positions, exclude the query ID, and retain the first K neighbors."""
    if k < 1:
        raise ValueError("K must be positive.")
    results = []
    for position in positions:
        if position == -1:
            continue
        if not 0 <= position < len(image_ids):
            raise ValueError("Search returned a position outside the image ID mapping.")
        image_id = image_ids[position].item()
        if image_id != query_id:
            results.append(image_id)
        if len(results) == k:
            break
    if len(set(results)) != len(results):
        raise ValueError("Search returned duplicate neighbor IDs.")
    if require_k and len(results) != k:
        raise ValueError(f"Query {query_id} has only {len(results)} non-self results; expected {k}.")
    return results


def recall_at_k(exact_ids: list, approximate_ids: list, k: int) -> float:
    if k < 1 or len(exact_ids) != k or len(approximate_ids) > k:
        raise ValueError("Recall requires exactly K exact IDs and at most K approximate IDs.")
    if len(set(exact_ids)) != k or len(set(approximate_ids)) != len(approximate_ids):
        raise ValueError("Recall inputs must have unique IDs.")
    # Missing ANN neighbors count as misses; never reduce the denominator.
    return len(set(exact_ids) & set(approximate_ids)) / k


def latency_statistics(latencies) -> dict:
    values = np.asarray(latencies, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Latencies must be a non-empty sequence of finite, non-negative values.")
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def speedup_from_p50(sequential_p50_ms: float, faiss_p50_ms: float) -> float | None:
    if not np.isfinite([sequential_p50_ms, faiss_p50_ms]).all() or min(sequential_p50_ms, faiss_p50_ms) < 0:
        raise ValueError("P50 latencies must be finite and non-negative.")
    return sequential_p50_ms / faiss_p50_ms if faiss_p50_ms > 0 else None


def aggregate_rows(rows: list[dict], index_build_ms: float | None = None) -> dict:
    if not rows:
        raise ValueError("Cannot aggregate an empty set of measurements.")
    config = {key: rows[0][key] for key in CONFIG_FIELDS}
    if any(any(row[key] != config[key] for key in CONFIG_FIELDS) for row in rows):
        raise ValueError("All measurements must belong to the same experiment configuration.")
    sequential = latency_statistics([row["sequential_latency_ms"] for row in rows])
    approximate = latency_statistics([row["faiss_latency_ms"] for row in rows])
    result = {**config, "query_count": len(rows)}
    result.update({f"sequential_{key}": value for key, value in sequential.items()})
    result.update({f"faiss_{key}": value for key, value in approximate.items()})
    result["recall_at_5"] = float(np.mean([row["recall_at_5"] for row in rows]))
    result["speedup"] = speedup_from_p50(sequential["p50_ms"], approximate["p50_ms"])
    if index_build_ms is not None:
        result["index_build_ms"] = index_build_ms
    return result


def _exact_kernel(embeddings, query, candidate_count):
    # These are the same operations timed by SequentialSearchService.search().
    started = perf_counter()
    scores = embeddings @ query
    positions = np.argsort(-scores, kind="stable")[:candidate_count]
    latency_ms = (perf_counter() - started) * 1000
    return positions, latency_ms


def _faiss_kernel(index, query, candidate_count):
    started = perf_counter()
    _, positions = index.search(query, candidate_count)
    latency_ms = (perf_counter() - started) * 1000
    return positions[0], latency_ms


def measure_exact_baseline(embeddings, image_ids, query_rows) -> dict:
    queries = np.ascontiguousarray(embeddings[query_rows])
    for query in queries[:WARMUP_COUNT]:
        _exact_kernel(embeddings, query, K + 1)
    results, latencies = [], []
    for query_row, query in zip(query_rows, queries):
        positions, latency = _exact_kernel(embeddings, query, K + 1)
        results.append(remove_self_matches(positions, image_ids, image_ids[query_row].item(), K))
        latencies.append(latency)
    return {"queries": queries, "top_k": results, "latencies_ms": latencies}


def collect_faiss_rows(index, image_ids, query_rows, baseline, nprobe, experiment) -> list[dict]:
    if not 1 <= nprobe <= index.nlist:
        raise ValueError("nprobe must be between 1 and nlist.")
    index.nprobe = int(nprobe)
    # Views of the same float32 query vectors used by the exact baseline.
    queries = baseline["queries"].reshape(len(query_rows), 1, index.d)
    for query in queries[:WARMUP_COUNT]:
        _faiss_kernel(index, query, K + 1)
    rows = []
    for ordinal, (query_row, query) in enumerate(zip(query_rows, queries)):
        positions, latency = _faiss_kernel(index, query, K + 1)
        query_id = image_ids[query_row].item()
        # Preserve the chosen nprobe even if ANN finds fewer than K neighbors.
        neighbors = remove_self_matches(positions, image_ids, query_id, K, require_k=False)
        exact_ids = baseline["top_k"][ordinal]
        overlap_count = len(set(exact_ids) & set(neighbors))
        rows.append({
            "experiment": experiment, "dataset_size": len(image_ids), "dimension": index.d,
            "nlist": index.nlist, "nprobe": nprobe, "k": K, "seed": SEED,
            "query_number": ordinal + 1, "query_row": int(query_row), "query_image_id": query_id,
            "sequential_latency_ms": baseline["latencies_ms"][ordinal],
            "faiss_latency_ms": latency,
            "sequential_top_k": json.dumps(exact_ids), "faiss_top_k": json.dumps(neighbors),
            "faiss_result_count": len(neighbors), "overlap_count": overlap_count,
            "recall_at_5": recall_at_k(exact_ids, neighbors, K),
        })
    return rows


def build_temporary_index(embeddings, nlist: int):
    """Train/add in RAM; never write or replace the application's saved index."""
    if not 1 <= nlist <= len(embeddings):
        raise ValueError("nlist must be between 1 and the number of vectors.")
    started = perf_counter()
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])
    index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    index.cp.seed = SEED
    index.train(embeddings)
    index.add(embeddings)
    build_ms = (perf_counter() - started) * 1000
    if not index.is_trained or index.ntotal != len(embeddings):
        raise RuntimeError("Temporary IVF index training/add failed.")
    return index, build_ms


def run_benchmark(output_directory: str | Path = REPO_ROOT / "benchmark_results") -> dict:
    started = perf_counter()
    sequential = SequentialSearchService()
    faiss_service = FaissSearchService()
    embeddings, image_ids = sequential.embeddings, sequential.image_ids
    if embeddings.shape != (5000, 2048) or faiss_service.index.ntotal != 5000:
        raise ValueError("The official experiments require 5000 vectors of dimension 2048.")
    if not np.array_equal(image_ids, faiss_service.image_ids) or len(set(image_ids.tolist())) != 5000:
        raise ValueError("Both engines must use the same unique IDs in the same order.")

    artifacts = (
        "data/processed/catalog.csv", "data/processed/embeddings.npy",
        "data/processed/image_ids.npy", "indexes/faiss_ivf.index",
    )
    fingerprints = {
        name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in artifacts
    }
    raw_rows, nprobe_results, scalability_results, warnings = [], [], [], []
    baselines = {}
    query_sets = {}

    print("Experiment A: exact baseline, 100 queries after 10 warm-up queries...", flush=True)
    full_rows = select_query_indices(5000)
    query_sets[5000] = full_rows
    baselines[5000] = measure_exact_baseline(embeddings, image_ids, full_rows)
    for nprobe in NPROBES:
        rows = collect_faiss_rows(
            faiss_service.index, image_ids, full_rows, baselines[5000], nprobe, "nprobe"
        )
        nprobe_results.append(aggregate_rows(rows))
        raw_rows.extend(rows)
        print(f"  Completed nprobe={nprobe}", flush=True)

    for dataset_size in DATASET_SIZES:
        subset = embeddings[:dataset_size]
        subset_ids = image_ids[:dataset_size]
        nlist, nprobe = subset_index_config(dataset_size)
        print(f"Experiment B: N={dataset_size}, nlist={nlist}, nprobe={nprobe}...", flush=True)
        if dataset_size not in baselines:
            query_sets[dataset_size] = select_query_indices(dataset_size)
            baselines[dataset_size] = measure_exact_baseline(subset, subset_ids, query_sets[dataset_size])
        index, build_ms = build_temporary_index(subset, nlist)
        recommended = index.cp.min_points_per_centroid * nlist
        if dataset_size < recommended:
            warnings.append(
                f"N={dataset_size}, nlist={nlist}: FAISS recommends at least {recommended} "
                "training vectors. The requested scalability formula was kept unchanged."
            )
        rows = collect_faiss_rows(
            index, subset_ids, query_sets[dataset_size], baselines[dataset_size], nprobe, "scalability"
        )
        scalability_results.append(aggregate_rows(rows, index_build_ms=build_ms))
        raw_rows.extend(rows)

    for experiment_rows in (nprobe_results, scalability_results):
        for config in experiment_rows:
            incomplete = sum(
                row["faiss_result_count"] < K for row in raw_rows
                if row["experiment"] == ("scalability" if "index_build_ms" in config else "nprobe")
                and row["dataset_size"] == config["dataset_size"] and row["nprobe"] == config["nprobe"]
            )
            if incomplete:
                warnings.append(
                    f"N={config['dataset_size']}, nlist={config['nlist']}, nprobe={config['nprobe']}: "
                    f"{incomplete} queries returned fewer than {K} non-self neighbors; "
                    "missing results counted as misses, without changing nprobe."
                )
    if nprobe_results[-1]["recall_at_5"] < 1.0:
        warnings.append("Full-probe Recall@5 is below 1; inspect raw IDs for exact-ranking/tie differences.")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(nprobe_results, columns=SUMMARY_FIELDS).to_csv(output / "nprobe_benchmark.csv", index=False)
    pd.DataFrame(scalability_results, columns=(*SUMMARY_FIELDS, "index_build_ms")).to_csv(
        output / "scalability_benchmark.csv", index=False
    )
    pd.DataFrame(raw_rows).to_csv(output / "raw_query_results.csv", index=False)
    total_seconds = perf_counter() - started
    metadata = {
        "seed": SEED, "query_count": QUERY_COUNT, "k": K, "warmup_queries": WARMUP_COUNT,
        "query_selection": "NumPy default_rng(42), without replacement, separately for each prefix subset",
        "self_exclusion": "Retrieve K+1; remove query image_id; keep first K; missing ANN results are misses",
        "sequential_baseline": "One measured pass per N; reused across A nprobes and B at N=5000",
        "sequential_timing": "embeddings @ query plus stable full argsort for K+1",
        "faiss_timing": "index.search for one query and K+1, excluding nprobe setting and postprocessing",
        "build_timing": "Temporary index creation, training and add; no artifact I/O",
        "experiment_a_index": "Existing IVF64 index; not rebuilt",
        "subset_nlist": "min(round(sqrt(N)), max(8, N // 40))",
        "subset_nprobe": "max(1, round(nlist / 8))",
        "training_seed": SEED, "metric": "inner_product", "dimension": 2048,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "faiss_version": faiss.__version__, "faiss_omp_threads": faiss.omp_get_max_threads(),
        "platform": platform.platform(), "processor": platform.processor(),
        "artifact_sha256": fingerprints, "total_seconds": total_seconds, "warnings": warnings,
    }
    (output / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "nprobe": nprobe_results, "scalability": scalability_results,
        "total_seconds": total_seconds, "warnings": warnings,
    }
