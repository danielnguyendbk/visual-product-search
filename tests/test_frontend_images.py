"""Result images must stay inside the repository and tolerate missing files."""

import pytest

from frontend import ui


def test_repository_relative_image_is_resolved(tmp_path, monkeypatch):
    image = tmp_path / "data" / "image.jpg"
    image.parent.mkdir()
    image.write_bytes(b"placeholder")
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    assert ui.resolve_image_path("data/image.jpg") == image
    assert ui.resolve_image_path("data\\image.jpg") == image


def test_existing_file_outside_project_cannot_be_displayed(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"private file")
    monkeypatch.setattr(ui, "PROJECT_ROOT", root)
    for path in ["../outside.jpg", "..\\outside.jpg", str(outside)]:
        assert ui.resolve_image_path(path) is None


@pytest.mark.parametrize("path", [
    "../outside.jpg", "..\\outside.jpg", "/outside.jpg", "C:\\Users\\outside.jpg",
    "C:outside.jpg", "\\\\server\\share\\image.jpg", "missing.jpg", "", None,
])
def test_unsafe_or_missing_paths_are_rejected(tmp_path, monkeypatch, path):
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    assert ui.resolve_image_path(path) is None
