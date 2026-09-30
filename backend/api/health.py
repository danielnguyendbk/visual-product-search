"""Health endpoint."""

from fastapi import APIRouter

from backend.schemas.common import HealthResponse
from backend.services.catalog_service import catalog_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_ready=False,
        embeddings_ready=False,
        index_ready=False,
        catalog_size=catalog_service.count(),
    )
