"""Shared test fixtures for mmdoc tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import pytest


@pytest.fixture
def test_image_png(tmp_path: Path) -> str:
    """Create a small test PNG image (1x1 pixel)."""
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    path = tmp_path / "test.png"
    img.save(path)
    return str(path)


@pytest.fixture
def test_image_jpg(tmp_path: Path) -> str:
    """Create a small test JPG image."""
    img = Image.new("RGB", (50, 50), color=(255, 0, 0))
    path = tmp_path / "test.jpg"
    img.save(path, "JPEG")
    return str(path)


@pytest.fixture
def test_image_gif(tmp_path: Path) -> str:
    """Create a small test GIF with 3 frames."""
    frames: list[Image.Image] = []
    for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
        img = Image.new("RGB", (30, 30), color=color)
        frames.append(img)
    path = tmp_path / "test.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, format="GIF")
    return str(path)


@pytest.fixture
def test_pdf(tmp_path: Path) -> str:
    """Create a minimal test PDF with one page via pymupdf."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello mmdoc - this is a test document.", fontsize=12)
    path = tmp_path / "test.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def nonexistent_path(tmp_path: Path) -> str:
    """Return a path that does not exist."""
    return str(tmp_path / "nonexistent.pdf")


@pytest.fixture
def mock_vl_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the VLModel so tests don't download the real model (~2GB)."""

    class MockVLModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def query(
            self, image: Any, prompt: str, *, system: str | None = None, temperature: float = 0.0
        ) -> str:
            return f"Mocked response for: {prompt[:50]}"

        def describe(self, image: Any, temperature: float = 0.0) -> str:
            return "A test document page."

        def extract_json(self, image: Any, schema_description: str, temperature: float = 0.0) -> dict[str, Any]:
            return {"extracted": "data", "source": schema_description[:30]}

    # Patch VLModel in every module that imported it at module level.
    # `from mmdoc.vl_model import VLModel` creates a local binding in each
    # module, so we must patch each one individually.
    import mmdoc.api
    import mmdoc.extractor

    monkeypatch.setattr("mmdoc.vl_model.VLModel", MockVLModel)
    monkeypatch.setattr(mmdoc.api, "VLModel", MockVLModel)
    monkeypatch.setattr(mmdoc.extractor, "VLModel", MockVLModel)
