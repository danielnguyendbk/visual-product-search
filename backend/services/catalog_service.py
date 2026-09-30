"""Temporary in-memory catalog used only by the Milestone 0 skeleton."""

from backend.schemas.catalog import CatalogCreate, CatalogRecord, CatalogUpdate


class CatalogService:
    """Store catalog records in process memory until persistence is designed."""

    def __init__(self) -> None:
        self._records: dict[str, CatalogRecord] = {}

    def list(self) -> list[CatalogRecord]:
        return list(self._records.values())

    def get(self, image_id: str) -> CatalogRecord | None:
        return self._records.get(image_id)

    def create(self, payload: CatalogCreate) -> CatalogRecord:
        record = CatalogRecord(**payload.model_dump())
        self._records[record.image_id] = record
        return record

    def update(self, image_id: str, payload: CatalogUpdate) -> CatalogRecord | None:
        current = self.get(image_id)
        if current is None:
            return None

        changes = payload.model_dump(exclude_none=True)
        updated = current.model_copy(update=changes)
        self._records[image_id] = updated
        return updated

    def delete(self, image_id: str) -> CatalogRecord | None:
        return self._records.pop(image_id, None)

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()


catalog_service = CatalogService()
