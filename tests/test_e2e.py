"""End-to-end test: real HTTP server + real file parsing + mocked VL model.

Boots a live Uvicorn server on a random port, sends real multipart uploads
to every endpoint, and validates the full response contract.
The only thing mocked is the VL model (avoids downloading 2GB).
"""

from __future__ import annotations

import io
import socket
import threading
import time
from typing import Any

import pytest
import requests
from PIL import Image
from uvicorn import Config, Server


def _free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _StubVLModel:
    """Drop-in replacement for VLModel that returns deterministic responses."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def query(
        self, image: Any, prompt: str, *, system: str | None = None, temperature: float = 0.0
    ) -> str:
        return f"Stub response for prompt: {prompt[:40]}"

    def describe(self, image: Any, temperature: float = 0.0) -> str:
        return "A test image with solid color."

    def extract_json(self, image: Any, schema_description: str, temperature: float = 0.0) -> dict[str, Any]:
        return {"stub": True, "schema": schema_description[:30]}


@pytest.fixture(scope="module")
def e2e_server() -> str:
    """Boot a real Uvicorn server with the VL model stubbed. Returns base URL."""
    import mmdoc.api as api_mod
    import mmdoc.extractor as ext_mod

    # Stub the model in every module that references VLModel
    api_mod.VLModel = _StubVLModel  # type: ignore[assignment]
    ext_mod.VLModel = _StubVLModel  # type: ignore[assignment]
    api_mod._vl = _StubVLModel()  # pre-populate the singleton

    port = _free_port()
    config = Config(app=api_mod.app, host="127.0.0.1", port=port, log_level="warning")
    server = Server(config=config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            r = requests.get(f"{base_url}/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("Server did not start in time")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


def _make_png_bytes() -> bytes:
    """Create a small PNG image as raw bytes."""
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_pdf_bytes() -> bytes:
    """Create a minimal PDF as raw bytes."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Test invoice - total: $42.00", fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


class TestE2EHealth:
    def test_health(self, e2e_server: str) -> None:
        resp = requests.get(f"{e2e_server}/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestE2EDescribe:
    def test_describe_png(self, e2e_server: str) -> None:
        resp = requests.post(
            f"{e2e_server}/describe",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "test.png"
        assert "description" in data
        assert isinstance(data["description"], str)

    def test_describe_pdf(self, e2e_server: str) -> None:
        resp = requests.post(
            f"{e2e_server}/describe",
            files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "doc.pdf"
        assert "[Page 1]" in data["description"]


class TestE2EExtract:
    def test_extract_png(self, e2e_server: str) -> None:
        resp = requests.post(
            f"{e2e_server}/extract",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
            data={"prompt": "Extract all text and numbers"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["path"] == "test.png"
        assert data["format"] == "png"
        assert len(data["pages"]) == 1
        assert data["pages"][0]["page"] == 1
        assert "content" in data["pages"][0]
        assert "has_extracted_text" in data["pages"][0]


class TestE2EAsk:
    def test_ask_pdf(self, e2e_server: str) -> None:
        resp = requests.post(
            f"{e2e_server}/ask",
            files={"file": ("doc.pdf", _make_pdf_bytes(), "application/pdf")},
            data={"question": "What is the total amount?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "doc.pdf"
        assert data["question"] == "What is the total amount?"
        assert isinstance(data["answer"], str)
        assert "[Page 1]" in data["answer"]


class TestE2ECORS:
    def test_cors_headers_present(self, e2e_server: str) -> None:
        """Verify CORS middleware allows cross-origin requests from the frontend."""
        resp = requests.options(
            f"{e2e_server}/describe",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*"


class TestE2EErrorHandling:
    def test_missing_file_returns_422(self, e2e_server: str) -> None:
        resp = requests.post(f"{e2e_server}/describe")
        assert resp.status_code == 422

    def test_ask_missing_question_returns_422(self, e2e_server: str) -> None:
        resp = requests.post(
            f"{e2e_server}/ask",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 422
