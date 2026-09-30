"""Benchmark endpoint contract."""

from fastapi import APIRouter, status

from backend.core.errors import APIError
from backend.schemas.benchmark import BenchmarkRequest, BenchmarkResponse
from backend.schemas.common import ErrorCode, ErrorResponse

router = APIRouter(tags=["benchmark"])


@router.post(
    "/benchmark",
    response_model=BenchmarkResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def run_benchmark(payload: BenchmarkRequest) -> BenchmarkResponse:
    raise APIError(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCode.BENCHMARK_NOT_READY,
        "Benchmark resources are not ready.",
    )
