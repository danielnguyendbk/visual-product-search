"""Comparison endpoint contract."""

from fastapi import APIRouter, File, Form, UploadFile, status

from backend.api.search import validate_image, validate_search_parameters
from backend.core.errors import APIError
from backend.schemas.common import ErrorCode, ErrorResponse
from backend.schemas.compare import CompareResponse

router = APIRouter(tags=["compare"])


@router.post(
    "/compare",
    response_model=CompareResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def compare_engines(
    image: UploadFile = File(...),
    k: int = Form(5),
    nprobe: int = Form(4),
) -> CompareResponse:
    validate_image(image)
    validate_search_parameters(k, nprobe)

    raise APIError(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCode.SEARCH_ENGINE_NOT_READY,
        "Search engines are not ready for comparison.",
    )
