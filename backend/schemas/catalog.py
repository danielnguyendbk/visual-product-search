"""Catalog CRUD schemas."""

from pydantic import BaseModel, Field


class CatalogRecord(BaseModel):
    image_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    category: str = Field(min_length=1)


class CatalogCreate(CatalogRecord):
    pass


class CatalogUpdate(BaseModel):
    product_id: str | None = Field(default=None, min_length=1)
    image_path: str | None = Field(default=None, min_length=1)
    category: str | None = Field(default=None, min_length=1)
