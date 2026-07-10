"""Tests for mmdoc.extractor — extraction pipeline orchestration."""

from __future__ import annotations

from mmdoc.extractor import (
    ExtractionResult,
    ask_document,
    describe_document,
    extract_json,
    extract_structured,
    batch_extract,
)


class TestExtractJson:
    def test_returns_list_of_dicts(self, test_image_png: str, mock_vl_model: None) -> None:
        results = extract_json(test_image_png, "Extract data")
        assert isinstance(results, list)
        assert len(results) == 1
        assert "page" in results[0]
        assert "data" in results[0]
        assert results[0]["page"] == 1

    def test_multiple_pages(self, test_pdf: str, mock_vl_model: None) -> None:
        """A PDF with multiple pages should return one result per page."""
        import fitz

        doc = fitz.open()
        doc.new_page()
        doc.new_page()
        doc.new_page()
        path = test_pdf.replace("test.pdf", "multi.pdf")
        doc.save(path)
        doc.close()

        results = extract_json(path, "Extract data")
        assert len(results) == 3

    def test_nonexistent_file_raises(self, nonexistent_path: str, mock_vl_model: None) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            extract_json(nonexistent_path, "Extract data")


class TestDescribeDocument:
    def test_returns_string(self, test_image_png: str, mock_vl_model: None) -> None:
        result = describe_document(test_image_png)
        assert isinstance(result, str)
        assert "[Frame 1]" in result

    def test_pdf_labeled_as_page(self, test_pdf: str, mock_vl_model: None) -> None:
        result = describe_document(test_pdf)
        assert "[Page 1]" in result


class TestExtractStructured:
    def test_returns_extraction_result(self, test_image_png: str, mock_vl_model: None) -> None:
        result = extract_structured(test_image_png, "Extract all info")
        assert isinstance(result, ExtractionResult)
        assert result.format == "png"
        assert len(result.pages) == 1
        assert result.pages[0]["page"] == 1
        assert "content" in result.pages[0]
        assert "has_extracted_text" in result.pages[0]

    def test_combined_text(self, test_image_png: str, mock_vl_model: None) -> None:
        result = extract_structured(test_image_png, "Extract text")
        assert "--- Page 1 ---" in result.combined


class TestAskDocument:
    def test_returns_answer(self, test_image_png: str, mock_vl_model: None) -> None:
        answer = ask_document(test_image_png, "What is in this document?")
        assert isinstance(answer, str)
        assert "[Page 1]" in answer

    def test_multi_page_pdf(self, test_pdf: str, mock_vl_model: None) -> None:
        answer = ask_document(test_pdf, "What is on page 1?")
        assert "[Page 1]" in answer


class TestBatchExtract:
    def test_processes_multiple_files(self, test_image_png: str, test_image_jpg: str, mock_vl_model: None) -> None:
        results = batch_extract([test_image_png, test_image_jpg], "Extract data")
        assert len(results) == 2
        assert all(isinstance(r, ExtractionResult) for r in results)
