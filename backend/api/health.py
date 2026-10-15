"""Health endpoint."""

from fastapi import APIRouter, Request

from backend.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> HealthResponse:
    services = getattr(request.app.state, "services", None)
    ready = services is not None
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_ready=ready,
        embeddings_ready=ready,
        index_ready=ready,
        catalog_size=len(services.catalog) if ready else 0,
        embedding_dimension=services.compare.dimension if ready else None,
        faiss_ready=ready,
    )
