"""Index management endpoint contract."""

from fastapi import APIRouter, status

from backend.core.errors import APIError
from backend.schemas.common import ErrorCode, ErrorResponse

router = APIRouter(prefix="/index", tags=["index"])


@router.post(
    "/rebuild",
    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    responses={503: {"model": ErrorResponse}},
)
def rebuild_index() -> None:
    raise APIError(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCode.INDEX_NOT_READY,
        "Index resources are not ready.",
    )
