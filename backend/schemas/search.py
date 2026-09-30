"""Search API schemas."""

from enum import Enum

from pydantic import BaseModel


class SearchEngine(str, Enum):
    SEQUENTIAL = "sequential"
    FAISS = "faiss"


class SearchResult(BaseModel):
    rank: int
    image_id: str
    product_id: str
    image_path: str
    score: float


class SearchResponse(BaseModel):
    engine: SearchEngine
    k: int
    latency_ms: float | None
    results: list[SearchResult]
