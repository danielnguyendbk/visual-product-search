"""Small HTTP client for the local FastAPI backend."""

import os

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = (5, 90)  # Connect timeout, response timeout (encoding can take longer on CPU).


class BackendError(Exception):
    """A backend problem that can be shown directly in the UI."""


def _request(method: str, endpoint: str, **kwargs) -> dict:
    try:
        response = requests.request(
            method, f"{BACKEND_URL}/api/{endpoint}", timeout=TIMEOUT, **kwargs,
        )
    except requests.Timeout:
        raise BackendError("Backend phản hồi quá lâu. Vui lòng thử lại.") from None
    except requests.ConnectionError:
        raise BackendError("Không kết nối được backend. Hãy kiểm tra FastAPI đang chạy.") from None
    except requests.RequestException:
        raise BackendError("Không gửi được yêu cầu tới backend. Vui lòng thử lại.") from None

    try:
        payload = response.json()
    except ValueError:
        raise BackendError(
            f"Backend trả phản hồi JSON không hợp lệ (HTTP {response.status_code})."
        ) from None
    if not response.ok:
        error = payload.get("error") if isinstance(payload, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        raise BackendError(
            message if isinstance(message, str) else f"Backend trả lỗi HTTP {response.status_code}."
        )
    if not isinstance(payload, dict):
        raise BackendError("Backend trả phản hồi JSON không đúng định dạng.")
    return payload


def _require_fields(payload: dict, fields: tuple[str, ...]) -> dict:
    if not all(field in payload for field in fields):
        raise BackendError("Phản hồi backend thiếu thông tin cần thiết.")
    return payload


def health() -> dict:
    return _require_fields(
        _request("GET", "health"),
        ("status", "catalog_size", "embedding_dimension", "faiss_ready"),
    )


def search_image(
    filename: str, image_bytes: bytes, content_type: str,
    engine: str = "sequential", k: int = 5, nprobe: int = 8,
) -> dict:
    data = {"engine": engine, "k": k}
    if engine == "faiss":
        data["nprobe"] = nprobe
    return _require_fields(
        _request("POST", "search", data=data, files={"file": (filename, image_bytes, content_type)}),
        ("engine", "k", "nprobe", "encoding_ms", "search_latency_ms", "results"),
    )


def compare_image(
    filename: str, image_bytes: bytes, content_type: str, k: int = 5, nprobe: int = 8,
) -> dict:
    return _require_fields(
        _request(
            "POST", "compare", data={"k": k, "nprobe": nprobe},
            files={"file": (filename, image_bytes, content_type)},
        ),
        ("query", "config", "sequential", "faiss", "metrics"),
    )


def get_benchmark() -> dict:
    return _require_fields(
        _request("GET", "benchmark"), ("nprobe_benchmark", "scalability_benchmark", "metadata"),
    )
