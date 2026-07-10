"""Full real E2E on multi-page PDF with GPU — describe + extract + ask."""

from __future__ import annotations

import time

from mmdoc.extractor import ask_document, describe_document, extract_structured
from mmdoc.vl_model import VLModel

PDF_PATH = "test_invoice.pdf"


def main() -> None:
    print("Loading model on GPU (4-bit)...")
    model = VLModel()
    model._load()
    print("Model ready.\n")

    print("=" * 60)
    print("[1/3] DESCRIBE")
    print("=" * 60)
    t0 = time.time()
    desc = describe_document(PDF_PATH, vl_model=model)
    print(f"({time.time() - t0:.0f}s)")
    print(desc)

    print("\n" + "=" * 60)
    print("[2/3] EXTRACT")
    print("=" * 60)
    t0 = time.time()
    result = extract_structured(
        PDF_PATH, "Extract all text, items, prices, and payment details", vl_model=model
    )
    print(f"({time.time() - t0:.0f}s)")
    for page in result.pages:
        has_text = page["has_extracted_text"]
        print(f"\n--- Page {page['page']} (digital_text={has_text}) ---")
        print(page["content"])

    print("\n" + "=" * 60)
    print("[3/3] ASK")
    print("=" * 60)
    t0 = time.time()
    answer = ask_document(
        PDF_PATH, "What is the grand total and what are the payment terms?", vl_model=model
    )
    print(f"({time.time() - t0:.0f}s)")
    print(answer)


if __name__ == "__main__":
    main()
