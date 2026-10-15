"""Encode once and delegate comparison to the existing core service."""

from fastapi import APIRouter, File, Form, Request, UploadFile

from backend.api.search import handle_search_error, validate_image, validate_search_parameters
from backend.schemas.common import ErrorResponse
from backend.schemas.compare import CompareResponse
from backend.services.api_service import get_services

router = APIRouter(tags=["compare"])


@router.post(
    "/compare",
    response_model=CompareResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def compare_engines(
    request: Request,
    file: UploadFile = File(...),
    k: int = Form(5),
    nprobe: int = Form(8),
) -> CompareResponse:
    with validate_image(file) as image:
        services = get_services(request)
        validate_search_parameters(services, k, nprobe)
        with services.lock:
            embedding, encoding_ms = services.encode(image)
            try:
                result = services.compare.compare(embedding, k=k, nprobe=nprobe)
            except RuntimeError as exc:
                handle_search_error(exc)
    for engine in ("sequential", "faiss"):
        result[engine]["results"] = services.enrich(result[engine]["results"])
    return CompareResponse(query={"encoding_ms": encoding_ms}, **result)
