"""Shared image preview and result rendering for the demo pages."""

from io import BytesIO
from pathlib import Path, PureWindowsPath

from PIL import Image, UnidentifiedImageError
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def number(value, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def resolve_image_path(image_path: str) -> Path | None:
    """Accept repository-relative files only, including when running on Windows."""
    try:
        path = Path(image_path)
        windows_path = PureWindowsPath(image_path)
        if path.is_absolute() or windows_path.drive or windows_path.root:
            return None
        resolved = (PROJECT_ROOT / image_path.replace("\\", "/")).resolve()
        resolved.relative_to(PROJECT_ROOT.resolve())
        return resolved if resolved.is_file() else None
    except (OSError, ValueError, TypeError):
        return None


def preview_query(uploaded_image) -> bool:
    if uploaded_image is None:
        st.info("Chọn một ảnh JPG, JPEG hoặc PNG để bắt đầu.")
        return False
    try:
        data = uploaded_image.getvalue()
        if not data or Path(uploaded_image.name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("Invalid upload.")
        with Image.open(BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Unsupported format.")
            with image.convert("RGB") as preview:
                st.image(preview, caption="Query image", width=220)
        return True
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError):
        st.error("Không đọc được ảnh upload. Hãy chọn ảnh JPG, JPEG hoặc PNG hợp lệ.")
        return False


def show_result(result: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**Rank {result['rank']}**")
        path = resolve_image_path(result.get("image_path", ""))
        if path is None:
            st.caption("Ảnh sản phẩm không có sẵn.")
        else:
            try:
                with Image.open(path) as image:
                    with image.convert("RGB") as preview:
                        st.image(preview, width=180)
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
                st.caption("Không đọc được ảnh sản phẩm.")
        st.write(result.get("product_display_name") or "Sản phẩm chưa có tên")
        st.caption(f"image_id: {result['image_id']} · product_id: {result['product_id']}")
        st.write(f"Similarity: **{number(result['score'], 4)}**")
        st.caption(f"{result.get('article_type', '')} · {result.get('base_colour', '')}")


def show_results(results: list[dict], columns: int = 1) -> None:
    if not results:
        st.info("Không có kết quả tìm kiếm.")
        return
    grid = st.columns(columns)
    for i, result in enumerate(results):
        with grid[i % columns]:
            show_result(result)
