"""Image search using shared encoder and existing search services."""

from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.core.errors import APIError
from backend.schemas.common import ErrorCode, ErrorResponse
from backend.schemas.search import SearchEngine, SearchResponse
from backend.services.api_service import APIServices, get_services

router = APIRouter(tags=["search"])
ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def validate_image(file: UploadFile) -> Image.Image:
    content_type = (file.content_type or "").lower()
    suffix = Path(file.filename or "").suffix.lower()
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES or suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise APIError(400, ErrorCode.INVALID_IMAGE, "Image must be a JPG or PNG file.")
    data = file.file.read()
    if not data:
        raise APIError(400, ErrorCode.INVALID_IMAGE, "Uploaded image must not be empty.")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Unsupported image format.")
            image.verify()
        with Image.open(BytesIO(data)) as image:
            # Conversion decodes the pixels and detaches them from the upload stream.
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise APIError(
            400, ErrorCode.INVALID_IMAGE, "Uploaded file is not a readable JPG or PNG image.",
        ) from None


def validate_search_parameters(
    services: APIServices, k: int, nprobe: int | None = None,
) -> None:
    if not 1 <= k <= services.compare.dataset_size:
        raise APIError(
            400, ErrorCode.INVALID_K,
            f"k must be between 1 and {services.compare.dataset_size}.",
        )
    if nprobe is not None and not 1 <= nprobe <= services.faiss.index.nlist:
        raise APIError(
            400, ErrorCode.INVALID_NPROBE,
            f"nprobe must be between 1 and {services.faiss.index.nlist}.",
        )


def handle_search_error(exc: RuntimeError) -> None:
    # An IVF search can have fewer than K candidates with a small nprobe.
    if str(exc).startswith("FAISS found only "):
        raise APIError(
            400, ErrorCode.INVALID_NPROBE,
            "FAISS found fewer than k candidates; increase nprobe or reduce k.",
        ) from exc
    raise exc


@router.post(
    "/search", response_model=SearchResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def search_products(
    request: Request,
    file: UploadFile = File(...),
    engine: SearchEngine = Form(SearchEngine.SEQUENTIAL),
    k: int = Form(5),
    nprobe: int = Form(8),
) -> SearchResponse:
    with validate_image(file) as image:
        services = get_services(request)
        validate_search_parameters(services, k, nprobe if engine == SearchEngine.FAISS else None)
        with services.lock:
            embedding, encoding_ms = services.encode(image)
            service = services.faiss if engine == SearchEngine.FAISS else services.sequential
            try:
                results = (
                    service.search(embedding, k=k, nprobe=nprobe)
                    if engine == SearchEngine.FAISS else service.search(embedding, k=k)
                )
            except RuntimeError as exc:
                handle_search_error(exc)
            latency_ms = service.last_search_latency_ms
    return SearchResponse(
        engine=engine, k=k, nprobe=nprobe if engine == SearchEngine.FAISS else None,
        encoding_ms=encoding_ms, search_latency_ms=latency_ms,
        results=services.enrich(results),
    )
