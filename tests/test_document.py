"""Tests for mmdoc.document — document loading and parsing."""

from __future__ import annotations

import pytest
from mmdoc.document import load_document, Document, DocumentPage


class TestLoadDocument:
    def test_load_png(self, test_image_png: str) -> None:
        doc = load_document(test_image_png)
        assert isinstance(doc, Document)
        assert doc.format == "png"
        assert len(doc.pages) == 1
        assert isinstance(doc.pages[0], DocumentPage)
        assert doc.pages[0].index == 0
        assert doc.pages[0].extracted_text == ""

    def test_load_jpg(self, test_image_jpg: str) -> None:
        doc = load_document(test_image_jpg)
        assert doc.format == "jpg"
        assert len(doc.pages) == 1

    def test_load_gif(self, test_image_gif: str) -> None:
        doc = load_document(test_image_gif)
        assert doc.format == "gif"
        assert len(doc.pages) == 3
        for i, page in enumerate(doc.pages):
            assert page.index == i
            assert page.metadata.get("frame") == i

    def test_load_pdf(self, test_pdf: str) -> None:
        doc = load_document(test_pdf)
        assert doc.format == "pdf"
        assert len(doc.pages) == 1
        assert doc.pages[0].extracted_text == "Hello mmdoc - this is a test document."

    def test_nonexistent_file(self, nonexistent_path: str) -> None:
        with pytest.raises(FileNotFoundError):
            load_document(nonexistent_path)

    def test_unsupported_format(self, tmp_path) -> None:
        path = str(tmp_path / "test.xyz")
        with open(path, "w") as f:
            f.write("not a real file")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_document(path)
