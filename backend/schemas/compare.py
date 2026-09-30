"""Sequential-versus-FAISS comparison schemas."""

from pydantic import BaseModel

from backend.schemas.search import SearchResult


class CompareConfig(BaseModel):
    k: int
    nprobe: int
    dataset_size: int | None
    dimension: int | None


class CompareEngineResult(BaseModel):
    latency_ms: float | None
    results: list[SearchResult]


class CompareResponse(BaseModel):
    config: CompareConfig
    sequential: CompareEngineResult
    faiss: CompareEngineResult
    overlap_at_k: float | None
    recall_at_k: float | None
    speedup: float | None
