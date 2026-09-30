"""Benchmark API schemas."""

from pydantic import BaseModel, Field


class BenchmarkRequest(BaseModel):
    dataset_size: int = Field(gt=0)
    nprobe: int = Field(gt=0)
    k: int = Field(gt=0)
    num_queries: int = Field(gt=0)
    repeats: int = Field(gt=0)


class BenchmarkResponse(BaseModel):
    dataset_size: int
    nlist: int | None
    nprobe: int
    k: int
    sequential_p50_ms: float | None
    sequential_p95_ms: float | None
    faiss_p50_ms: float | None
    faiss_p95_ms: float | None
    recall_at_k: float | None
    speedup: float | None
    index_build_ms: float | None
