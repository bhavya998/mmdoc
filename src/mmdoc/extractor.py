"""Extraction pipeline — document → VL model → structured JSON.

Supports three modes:
  extract  — VL model reads every page and extracts structured data
  ask      — question answering over the document
  smart    — uses extracted text for digital PDFs, VL only for scanned/empty pages
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mmdoc.document import load_document
from mmdoc.vl_model import VLModel


@dataclass(slots=True)
class ExtractionResult:
    """Result of processing one document."""

    path: str
    format: str
    pages: list[dict[str, Any]]
    combined: str  # merged text/JSON from all pages


_EXTRACT_SYSTEM = (
    "You are a document data extractor. Extract ALL information visible on this page."
    " Include tables as arrays, key-value pairs as objects, text as strings."
    " Do NOT summarize — include every data point."
)
_ASK_SYSTEM = (
    "You are a document analyst. Answer questions based ONLY on what is visible on the page."
    " Be precise. If the answer is not on the page, say so."
)


def extract_json(
    file_path: str,
    schema_description: str,
    vl_model: VLModel | None = None,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """Extract structured data matching the schema from each page of the document."""
    model = vl_model or VLModel()
    doc = load_document(file_path)
    results: list[dict[str, Any]] = []

    for page in doc.pages:
        extracted = model.extract_json(page.image, schema_description, temperature)
        results.append({
            "page": page.index + 1,
            "data": extracted,
            "has_text": bool(page.extracted_text),
        })

    return results


def describe_document(
    file_path: str,
    vl_model: VLModel | None = None,
    temperature: float = 0.0,
) -> str:
    """Describe what each page/frame of a document depicts in 1-2 sentences."""
    model = vl_model or VLModel()
    doc = load_document(file_path)
    descriptions: list[str] = []
    label = "page" if doc.format == "pdf" else "frame"
    for page in doc.pages:
        desc = model.describe(page.image, temperature=temperature)
        descriptions.append(f"[{label.capitalize()} {page.index + 1}] {desc}")
    return "\n".join(descriptions)


def extract_structured(
    file_path: str,
    prompt: str,
    vl_model: VLModel | None = None,
    temperature: float = 0.0,
) -> ExtractionResult:
    """Extract all information from every page. Returns ExtractionResult."""
    model = vl_model or VLModel()
    doc = load_document(file_path)
    pages: list[dict[str, Any]] = []

    for page in doc.pages:
        text = model.query(page.image, prompt, system=_EXTRACT_SYSTEM, temperature=temperature)
        pages.append({
            "page": page.index + 1,
            "content": text,
            "has_extracted_text": bool(page.extracted_text),
        })

    combined = _combine(pages)
    return ExtractionResult(path=file_path, format=doc.format, pages=pages, combined=combined)


def ask_document(
    file_path: str,
    question: str,
    vl_model: VLModel | None = None,
    temperature: float = 0.0,
) -> str:
    """Ask a question about the document. Returns the answer."""
    model = vl_model or VLModel()
    doc = load_document(file_path)
    answers: list[str] = []

    for page in doc.pages:
        answer = model.query(page.image, question, system=_ASK_SYSTEM, temperature=temperature)
        answers.append(f"[Page {page.index + 1}] {answer}")

    return "\n\n".join(answers)


def batch_extract(
    file_paths: list[str],
    prompt: str,
    vl_model: VLModel | None = None,
    temperature: float = 0.0,
) -> list[ExtractionResult]:
    """Process multiple files. Model is loaded once, shared across files."""
    model = vl_model or VLModel()
    results: list[ExtractionResult] = []
    for fp in file_paths:
        results.append(extract_structured(fp, prompt, vl_model=model, temperature=temperature))
    return results


def _combine(pages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for p in pages:
        parts.append(f"--- Page {p['page']} ---\n{p['content']}")
    return "\n\n".join(parts)
