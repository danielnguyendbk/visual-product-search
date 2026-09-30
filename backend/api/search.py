"""Search endpoint contract for the not-yet-implemented engines."""

from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, status
from PIL import Image, UnidentifiedImageError

from backend.core.errors import APIError
from backend.schemas.common import ErrorCode, ErrorResponse
from backend.schemas.search import SearchEngine, SearchResponse

router = APIRouter(tags=["search"])

ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def validate_image(image: UploadFile) -> None:
    content_type = (image.content_type or "").lower()
    suffix = Path(image.filename or "").suffix.lower()
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES or suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise APIError(
            status.HTTP_400_BAD_REQUEST,
            ErrorCode.INVALID_IMAGE,
            "Image must be a JPG or PNG file.",
        )

    try:
        Image.open(image.file).verify()
        image.file.seek(0)
    except (OSError, UnidentifiedImageError, ValueError):
        image.file.seek(0)
        raise APIError(
            status.HTTP_400_BAD_REQUEST,
            ErrorCode.INVALID_IMAGE,
            "Uploaded file is not a valid JPG or PNG image.",
        ) from None


def validate_search_parameters(k: int, nprobe: int | None = None) -> None:
    if k <= 0:
        raise APIError(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            ErrorCode.INVALID_K,
            "k must be greater than zero.",
        )
    if nprobe is not None and nprobe <= 0:
        raise APIError(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            ErrorCode.INVALID_NPROBE,
            "nprobe must be greater than zero.",
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def search_products(
    image: UploadFile = File(...),
    engine: SearchEngine = Form(SearchEngine.SEQUENTIAL),
    k: int = Form(5),
    nprobe: int | None = Form(None),
) -> SearchResponse:
    validate_image(image)
    validate_search_parameters(k, nprobe)

    raise APIError(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCode.SEARCH_ENGINE_NOT_READY,
        "Search engine is not ready.",
    )
