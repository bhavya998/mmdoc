"""Real end-to-end test: live server + real model + real image OCR."""

from __future__ import annotations

import io
import socket
import threading
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont
from uvicorn import Config, Server

from mmdoc.api import app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    port = _free_port()
    config = Config(app=app, host="127.0.0.1", port=port, log_level="warning")
    server = Server(config=config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            r = requests.get(f"{base}/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("Server did not start")

    print(f"Server up on port {port}")

    # Create a clear test image (receipt)
    img = Image.new("RGB", (500, 350), color="white")
    draw = ImageDraw.Draw(img)
    f1 = ImageFont.truetype("arial.ttf", 32)
    f2 = ImageFont.truetype("arial.ttf", 22)
    draw.text((40, 20), "RECEIPT", fill="black", font=f1)
    draw.text((40, 80), "Store: TechMart", fill="black", font=f2)
    draw.text((40, 120), "Mouse x2 = 30 USD", fill="black", font=f2)
    draw.text((40, 160), "Keyboard x1 = 50 USD", fill="black", font=f2)
    draw.text((40, 200), "Cable x3 = 15 USD", fill="black", font=f2)
    draw.text((40, 260), "TOTAL: 95 USD", fill="blue", font=f1)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # TEST 1: Health
    r = requests.get(f"{base}/health")
    print(f"\n[HEALTH] {r.json()}")

    # TEST 2: Describe
    print("\n[DESCRIBE] sending receipt image to /describe ...")
    t0 = time.time()
    r = requests.post(f"{base}/describe", files={"file": ("receipt.png", img_bytes, "image/png")})
    elapsed = time.time() - t0
    data = r.json()
    print(f"  ({elapsed:.0f}s) status={r.status_code}")
    print(f"  filename: {data['filename']}")
    print(f"  description: {data['description']}")

    # TEST 3: Extract
    print("\n[EXTRACT] sending receipt image to /extract ...")
    t0 = time.time()
    r = requests.post(
        f"{base}/extract",
        files={"file": ("receipt.png", img_bytes, "image/png")},
        data={"prompt": "Extract all items and prices"},
    )
    elapsed = time.time() - t0
    data = r.json()
    print(f"  ({elapsed:.0f}s) status={r.status_code}")
    print(f"  path: {data['path']}")
    print(f"  format: {data['format']}")
    print(f"  pages: {len(data['pages'])}")
    content = data["pages"][0]["content"]
    print(f"  page 1 content:\n{content[:500]}")

    # TEST 4: Ask
    print("\n[ASK] sending receipt image to /ask ...")
    t0 = time.time()
    r = requests.post(
        f"{base}/ask",
        files={"file": ("receipt.png", img_bytes, "image/png")},
        data={"question": "What items were purchased?"},
    )
    elapsed = time.time() - t0
    data = r.json()
    print(f"  ({elapsed:.0f}s) status={r.status_code}")
    print(f"  question: {data['question']}")
    print(f"  answer: {data['answer']}")

    server.should_exit = True
    print("\nAll real E2E API tests complete.")


if __name__ == "__main__":
    main()
