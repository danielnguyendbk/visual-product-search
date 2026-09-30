"""FastAPI application entrypoint."""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api import benchmark, catalog, compare, health, index, search
from backend.core.config import settings
from backend.core.errors import APIError
from backend.schemas.common import ErrorCode

app = FastAPI(title=settings.app_name, version=settings.app_version)


def error_response(status_code: int, code: ErrorCode, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code.value, "message": message}},
    )


@app.exception_handler(APIError)
async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
    return error_response(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    invalid_fields = {str(error["loc"][-1]) for error in exc.errors()}

    if "image" in invalid_fields:
        return error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            ErrorCode.INVALID_IMAGE,
            "A JPG or PNG image is required.",
        )
    if "k" in invalid_fields:
        return error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            ErrorCode.INVALID_K,
            "k must be a positive integer.",
        )
    if "nprobe" in invalid_fields:
        return error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            ErrorCode.INVALID_NPROBE,
            "nprobe must be a positive integer.",
        )

    return error_response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        ErrorCode.INTERNAL_ERROR,
        "Request validation failed.",
    )


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return error_response(exc.status_code, ErrorCode.INTERNAL_ERROR, message)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    return error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorCode.INTERNAL_ERROR,
        "An internal error occurred.",
    )


for api_router in (
    health.router,
    search.router,
    compare.router,
    benchmark.router,
    catalog.router,
    index.router,
):
    app.include_router(api_router, prefix=settings.api_v1_prefix)
