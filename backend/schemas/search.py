"""Search API schemas."""

from enum import Enum

from pydantic import BaseModel


class SearchEngine(str, Enum):
    SEQUENTIAL = "sequential"
    FAISS = "faiss"


class ProductSearchResult(BaseModel):
    rank: int
    image_id: int | str
    product_id: int | str
    image_path: str
    score: float
    master_category: str
    sub_category: str
    article_type: str
    base_colour: str
    product_display_name: str


SearchResult = ProductSearchResult


class SearchResponse(BaseModel):
    engine: SearchEngine
    k: int
    nprobe: int | None
    encoding_ms: float
    search_latency_ms: float
    results: list[ProductSearchResult]
