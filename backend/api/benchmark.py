"""Read-only access to saved benchmark aggregates."""

from fastapi import APIRouter

from backend.schemas.benchmark import BenchmarkResponse
from backend.schemas.common import ErrorResponse
from backend.services.api_service import read_benchmark_results

router = APIRouter(tags=["benchmark"])


@router.get(
    "/benchmark",
    response_model=BenchmarkResponse,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def get_benchmark() -> BenchmarkResponse:
    return BenchmarkResponse(**read_benchmark_results())
