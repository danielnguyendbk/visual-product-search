"""Schemas shared by all API modules."""

from enum import Enum

from pydantic import BaseModel


class ErrorCode(str, Enum):
    INVALID_IMAGE = "INVALID_IMAGE"
    INVALID_K = "INVALID_K"
    INVALID_NPROBE = "INVALID_NPROBE"
    CATALOG_NOT_FOUND = "CATALOG_NOT_FOUND"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    EMBEDDINGS_NOT_READY = "EMBEDDINGS_NOT_READY"
    INDEX_NOT_READY = "INDEX_NOT_READY"
    SEARCH_ENGINE_NOT_READY = "SEARCH_ENGINE_NOT_READY"
    BENCHMARK_NOT_READY = "BENCHMARK_NOT_READY"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    embeddings_ready: bool
    index_ready: bool
    catalog_size: int
