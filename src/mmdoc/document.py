"""Document handler — PDF, PNG, JPG, GIF → consistent page/frame objects.

Renders any input into a list of PIL images (one per page/frame) + extracted text.
Supports: PDF (digital + scanned), PNG, JPG, JPEG, GIF, TIFF, BMP, WebP.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # pymupdf


@dataclass(slots=True)
class DocumentPage:
    """One page/frame of a document, ready for VL inference."""

    index: int
    image: Any  # PIL.Image
    extracted_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    """A loaded document: list of pages/frames with source metadata."""

    path: str
    pages: list[DocumentPage]
    format: str  # "pdf", "png", "jpg", "gif", etc.


# Image formats that PIL can open directly (no PDF needed)
_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".tiff", ".tif", ".bmp", ".webp"}


def load_document(file_path: str) -> Document:
    """Load any supported file into a Document with page/frame images.

    PDFs are rendered page-by-page via pymupdf. Text from digital PDFs is
    extracted alongside the page image so cheap extraction paths can skip VL.
    Images and GIFs are loaded via PIL — GIFs become one frame per page.
    """
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = src.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix in _IMAGE_FORMATS:
        return _load_image(file_path, suffix)
    raise ValueError(f"Unsupported format: {suffix}. Supported: PDF, PNG, JPG, GIF, TIFF, BMP, WebP")


def _load_pdf(path: str) -> Document:
    doc = fitz.open(path)
    pages: list[DocumentPage] = []
    for i in range(len(doc)):
        page = doc[i]
        # render page as image
        pix = page.get_pixmap(dpi=200)
        img_data = pix.tobytes("png")
        from PIL import Image

        image = Image.open(io.BytesIO(img_data))

        # extract text from digital PDFs
        text = page.get_text().strip()

        pages.append(
            DocumentPage(
                index=i,
                image=image,
                extracted_text=text,
                metadata={"page_num": i + 1, "width": page.rect.width, "height": page.rect.height},
            )
        )
    doc.close()
    return Document(path=path, pages=pages, format="pdf")


def _load_image(path: str, suffix: str) -> Document:
    from PIL import Image

    suffix_stripped = suffix.lstrip(".")

    if suffix_stripped == "gif":
        gif = Image.open(path)
        frames: list[DocumentPage] = []
        for i in range(getattr(gif, "n_frames", 1)):
            gif.seek(i)
            frame = gif.copy().convert("RGB")
            frames.append(DocumentPage(index=i, image=frame, extracted_text="", metadata={"frame": i}))
        return Document(path=path, pages=frames, format="gif")

    image = Image.open(path).convert("RGB")
    return Document(
        path=path,
        pages=[DocumentPage(index=0, image=image, extracted_text="", metadata={})],
        format=suffix_stripped,
    )
