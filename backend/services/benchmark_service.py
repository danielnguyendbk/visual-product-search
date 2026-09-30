"""Benchmark interface reserved for a later milestone."""


def run_benchmark(
    dataset_size: int,
    nprobe: int,
    k: int,
    num_queries: int,
    repeats: int,
):
    raise NotImplementedError
