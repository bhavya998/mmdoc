"""FastAPI application — upload documents, get structured extraction / description / Q&A.

POST /extract   — upload file + prompt → structured JSON per page
POST /describe  — upload file → 1-2 sentence description per page
POST /ask       — upload file + question → answer
GET  /health    — liveness check
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mmdoc.extractor import ask_document, describe_document, extract_structured
from mmdoc.vl_model import VLModel

_vl: VLModel | None = None


def _get_model() -> VLModel:
    global _vl  # noqa: PLW0603
    if _vl is None:
        _vl = VLModel()
    return _vl


class PageResult(BaseModel):
    page: int
    content: str
    has_extracted_text: bool


class ExtractResponse(BaseModel):
    path: str
    format: str
    pages: list[PageResult]


def create_app() -> FastAPI:
    app = FastAPI(title="mmdoc — Multi-modal Document Understanding", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/extract", response_model=ExtractResponse)
    async def extract(
        file: UploadFile = File(...),
        prompt: str = Form("Extract all information from this document page"),
        temperature: float = Form(0.0),
    ) -> ExtractResponse:
        tmp = _save_upload(file)
        result = extract_structured(tmp, prompt, vl_model=_get_model(), temperature=temperature)
        Path(tmp).unlink(missing_ok=True)
        return ExtractResponse(
            path=file.filename or "upload",
            format=result.format,
            pages=[
                PageResult(page=p["page"], content=p["content"], has_extracted_text=p["has_extracted_text"])
                for p in result.pages
            ],
        )

    @app.post("/describe")
    async def describe(
        file: UploadFile = File(...),
        temperature: float = Form(0.0),
    ) -> dict[str, Any]:
        tmp = _save_upload(file)
        result = describe_document(tmp, vl_model=_get_model(), temperature=temperature)
        Path(tmp).unlink(missing_ok=True)
        return {"filename": file.filename, "description": result}

    @app.post("/ask")
    async def ask(
        file: UploadFile = File(...),
        question: str = Form(...),
        temperature: float = Form(0.0),
    ) -> dict[str, Any]:
        tmp = _save_upload(file)
        result = ask_document(tmp, question, vl_model=_get_model(), temperature=temperature)
        Path(tmp).unlink(missing_ok=True)
        return {"filename": file.filename, "question": question, "answer": result}

    return app


def _save_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "upload").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read())
    tmp.close()
    return tmp.name


app = create_app()
