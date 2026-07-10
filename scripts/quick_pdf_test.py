"""Quick single-call PDF extract test (reduced tokens for speed)."""

from __future__ import annotations

import time

from mmdoc.document import load_document
from mmdoc.vl_model import VLModel


def main() -> None:
    doc = load_document("test_invoice.pdf")
    print(f"PDF loaded: {len(doc.pages)} pages, format={doc.format}")
    print(f"Page 1 digital text: {doc.pages[0].extracted_text!r}")
    print(f"Page 2 digital text: {doc.pages[1].extracted_text!r}")
    print(f"Page 1 image size: {doc.pages[0].image.size}")

    model = VLModel(max_tokens=200)
    model._load()

    print("\n[EXTRACT - Page 1]")
    t0 = time.time()
    out = model.query(
        doc.pages[0].image,
        "List all items and their prices",
        system="Be brief. List items only.",
    )
    print(f"({time.time() - t0:.0f}s)")
    print(out)


if __name__ == "__main__":
    main()
