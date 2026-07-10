"""Tests for mmdoc.api — FastAPI endpoints.

Uses FastAPI TestClient and mocks the VLModel to avoid model downloads.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mmdoc.api import app


@pytest.fixture(autouse=True)
def reset_globals() -> None:
    """Reset the global VL singleton between tests."""
    import mmdoc.api as api

    api._vl = None


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestDescribe:
    def test_describe_png(self, client: TestClient, test_image_png: str, mock_vl_model: None) -> None:
        with open(test_image_png, "rb") as f:
            resp = client.post("/describe", files={"file": ("test.png", f, "image/png")})

        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "test.png"
        assert isinstance(data["description"], str)

    def test_describe_without_file_returns_422(self, client: TestClient) -> None:
        resp = client.post("/describe")
        assert resp.status_code == 422


class TestExtract:
    def test_extract_png(self, client: TestClient, test_image_png: str, mock_vl_model: None) -> None:
        with open(test_image_png, "rb") as f:
            resp = client.post(
                "/extract",
                files={"file": ("test.png", f, "image/png")},
                data={"prompt": "Extract all text"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["path"] == "test.png"
        assert data["format"] == "png"
        assert isinstance(data["pages"], list)
        assert len(data["pages"]) == 1
        assert data["pages"][0]["page"] == 1
        assert "content" in data["pages"][0]

    def test_extract_no_file_returns_422(self, client: TestClient) -> None:
        resp = client.post("/extract")
        assert resp.status_code == 422


class TestAsk:
    def test_ask_png(self, client: TestClient, test_image_png: str, mock_vl_model: None) -> None:
        with open(test_image_png, "rb") as f:
            resp = client.post(
                "/ask",
                files={"file": ("test.png", f, "image/png")},
                data={"question": "What is in this image?"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "test.png"
        assert data["question"] == "What is in this image?"
        assert isinstance(data["answer"], str)

    def test_ask_without_question_returns_422(self, client: TestClient, test_image_png: str) -> None:
        with open(test_image_png, "rb") as f:
            resp = client.post("/ask", files={"file": ("test.png", f, "image/png")})

        assert resp.status_code == 422

    def test_ask_with_temperature(self, client: TestClient, test_image_png: str, mock_vl_model: None) -> None:
        with open(test_image_png, "rb") as f:
            resp = client.post(
                "/ask",
                files={"file": ("test.png", f, "image/png")},
                data={"question": "What?", "temperature": "0.5"},
            )

        assert resp.status_code == 200
