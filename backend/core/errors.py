"""Application exceptions rendered with the shared API error envelope."""

from backend.schemas.common import ErrorCode


class APIError(Exception):
    def __init__(self, status_code: int, code: ErrorCode, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)
