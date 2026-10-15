"""Run both official 100-query experiments and print their measured summaries."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def speedup_text(value):
    return "n/a" if value is None else f"{value:.2f}x"


def main() -> int:
    from backend.services.benchmark import run_benchmark

    try:
        result = run_benchmark()
        print("\nEXPERIMENT A: N=5000, dimension=2048, nlist=64, K=5")
        print("nprobe | FAISS p50 ms | FAISS p95 ms | Recall@5 | Speedup")
        for row in result["nprobe"]:
            print(
                f"{row['nprobe']:6d} | {row['faiss_p50_ms']:12.4f} | {row['faiss_p95_ms']:12.4f} | "
                f"{row['recall_at_5']:8.3f} | {speedup_text(row['speedup'])}"
            )
        print(f"Sequential p50: {result['nprobe'][0]['sequential_p50_ms']:.4f} ms")
        print("\nEXPERIMENT B: K=5, dimension=2048")
        print("N | nlist | nprobe | Sequential p50 ms | FAISS p50 ms | Recall@5 | Speedup | Build ms")
        for row in result["scalability"]:
            print(
                f"{row['dataset_size']:4d} | {row['nlist']:5d} | {row['nprobe']:6d} | "
                f"{row['sequential_p50_ms']:17.4f} | {row['faiss_p50_ms']:12.4f} | "
                f"{row['recall_at_5']:8.3f} | {speedup_text(row['speedup']):>7s} | "
                f"{row['index_build_ms']:.3f}"
            )
        print(f"\nTotal benchmark time: {result['total_seconds']:.3f} seconds")
        print("Saved aggregate CSVs, raw_query_results.csv and benchmark_metadata.json in benchmark_results/.")
        print("Recall@5 is ANN overlap with exact Top-5 after self-exclusion, not semantic accuracy.")
        for warning in result["warnings"]:
            print(f"WARNING: {warning}")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
