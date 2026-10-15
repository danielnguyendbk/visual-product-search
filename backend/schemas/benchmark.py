"""Benchmark API schemas."""

from pydantic import BaseModel, Field


class BenchmarkRequest(BaseModel):
    dataset_size: int = Field(gt=0)
    nprobe: int = Field(gt=0)
    k: int = Field(gt=0)
    num_queries: int = Field(gt=0)
    repeats: int = Field(gt=0)


class BenchmarkResponse(BaseModel):
    nprobe_benchmark: list[dict]
    scalability_benchmark: list[dict]
    metadata: dict
