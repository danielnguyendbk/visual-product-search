"""Temporary in-memory catalog CRUD endpoints."""

from fastapi import APIRouter, status

from backend.core.errors import APIError
from backend.schemas.catalog import CatalogCreate, CatalogRecord, CatalogUpdate
from backend.schemas.common import ErrorCode, ErrorResponse
from backend.services.catalog_service import catalog_service

router = APIRouter(prefix="/catalog", tags=["catalog"])


def get_record_or_404(image_id: str) -> CatalogRecord:
    record = catalog_service.get(image_id)
    if record is None:
        raise APIError(
            status.HTTP_404_NOT_FOUND,
            ErrorCode.CATALOG_NOT_FOUND,
            f"Catalog record '{image_id}' was not found.",
        )
    return record


@router.get("", response_model=list[CatalogRecord])
def list_catalog() -> list[CatalogRecord]:
    return catalog_service.list()


@router.get(
    "/{image_id}",
    response_model=CatalogRecord,
    responses={404: {"model": ErrorResponse}},
)
def get_catalog_record(image_id: str) -> CatalogRecord:
    return get_record_or_404(image_id)


@router.post(
    "",
    response_model=CatalogRecord,
    status_code=status.HTTP_201_CREATED,
    responses={422: {"model": ErrorResponse}},
)
def create_catalog_record(payload: CatalogCreate) -> CatalogRecord:
    return catalog_service.create(payload)


@router.patch(
    "/{image_id}",
    response_model=CatalogRecord,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
def update_catalog_record(image_id: str, payload: CatalogUpdate) -> CatalogRecord:
    record = catalog_service.update(image_id, payload)
    if record is None:
        return get_record_or_404(image_id)
    return record


@router.delete(
    "/{image_id}",
    response_model=CatalogRecord,
    responses={404: {"model": ErrorResponse}},
)
def delete_catalog_record(image_id: str) -> CatalogRecord:
    record = catalog_service.delete(image_id)
    if record is None:
        return get_record_or_404(image_id)
    return record
