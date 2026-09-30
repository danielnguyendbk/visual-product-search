"""Application settings for the Milestone 0 skeleton."""

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Visual Product Search API"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"


settings = Settings()
