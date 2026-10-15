"""Sequential-versus-FAISS comparison schemas."""

from pydantic import BaseModel

from backend.schemas.search import SearchResult


class CompareConfig(BaseModel):
    k: int
    nprobe: int
    dataset_size: int
    dimension: int
    nlist: int


class CompareEngineResult(BaseModel):
    latency_ms: float
    results: list[SearchResult]


class QueryTiming(BaseModel):
    encoding_ms: float


class CompareMetrics(BaseModel):
    overlap_count: int
    recall_at_k: float
    speedup: float | None


class CompareResponse(BaseModel):
    query: QueryTiming
    config: CompareConfig
    sequential: CompareEngineResult
    faiss: CompareEngineResult
    metrics: CompareMetrics
