<div align="center">

# mmdoc — Multi-modal Document Understanding

**Upload any document (PDF, image, screenshot) and get structured JSON, descriptions, and answers — all from a local 0.8B vision-language model. No API keys, no cloud, runs entirely on your GPU.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/Tests-42%20passing-22c55e)]()
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

</div>

---

## What It Does

Feed it a PDF, PNG, JPG, or GIF. It reads every page using a vision-language model and gives you back:

| Mode | What you get |
|---|---|
| **Describe** | 1-2 sentence summary of what each page depicts |
| **Extract** | Structured JSON — tables, key-value pairs, all data points |
| **Ask** | Natural-language Q&A over the document content |
| **Batch** | Process an entire folder of files at once |

```
Product label (PNG)  →  {"product":"Parle-G","batch":"8472-AB","sugar":"15g/100g"}
Invoice (PDF)        →  {"vendor":"ABC Corp","total":"₹12,450","items":[...]}
Research paper (PDF) →  Full text + tables as arrays + figure captions
Screenshot           →  "A login form with username and password fields"
```

## Quick Start

### Prerequisites

- **Python 3.12+** and [`uv`](https://docs.astral.sh/uv/) (package manager)
- **NVIDIA GPU** with CUDA support (tested on RTX 3070 Ti, 8GB VRAM)
- **Node.js 18+** for the frontend

> CPU-only mode works but is ~50x slower (~400s/page vs ~7s/page on GPU).

```bash
git clone https://github.com/bhavya998/mmdoc.git
cd mmdoc

# Backend (PyTorch CUDA + model deps)
uv sync                         # installs torch with CUDA 12.8 automatically
uv run mmdoc serve              # FastAPI on :8000

# Frontend (separate terminal)
cd ui && npm install && npm run dev   # Next.js on :3000
```

Open `http://localhost:3000`, drop a file, get results.

> Model downloads on first use (~2GB). Cached thereafter.

## CLI Commands

| Command | Description |
|---|---|
| `mmdoc serve` | FastAPI server on :8000 |
| `mmdoc extract file.pdf` | Extract structured data (CLI) |
| `mmdoc describe file.png` | Describe what each page depicts |
| `mmdoc ask file.pdf "question"` | Ask a question about a document |
| `mmdoc batch "*.pdf"` | Batch process files → JSON |

## API Endpoints

```bash
POST /extract   — upload file + prompt → structured JSON per page
POST /describe  — upload file → 1-2 sentence description per page
POST /ask       — upload file + question → answer
GET  /health    — liveness check
```

## How It Works

```
PDF / PNG / JPG / GIF
       ↓
  pymupdf / PIL          ← renders pages → PIL images
       ↓
  Qwen3.5-0.8B           ← 0.8B unified vision-language model (4-bit, ~1GB VRAM)
  (local GPU)                reads each image: extracts, describes, answers
       ↓
  Structured JSON / text ← clean output
```

## Tech Stack

| Layer | Technology |
|---|---|
| Vision-Language Model | `Qwen3.5-0.8B` (unified multimodal, 4-bit, ~1GB VRAM) |
| PDF Rendering | pymupdf (200 DPI page → image) |
| Image Handling | PIL / Pillow (PNG, JPG, GIF, TIFF, BMP, WebP) |
| Backend | FastAPI + Uvicorn |
| Frontend | Next.js 16 + React 19 + Tailwind 4 |
| Inference | HuggingFace transformers + bitsandbytes |
| CLI | Typer + Rich |

## Project Structure

```
mmdoc/
├── src/mmdoc/
│   ├── vl_model.py       Qwen3.5-0.8B loader (GPU 4-bit + CPU fallback)
│   ├── document.py       PDF/image handler (pymupdf + PIL)
│   ├── extractor.py      Pipeline: extract / describe / ask / batch
│   ├── api.py            FastAPI app (4 endpoints)
│   └── cli.py            CLI (serve / extract / describe / ask / batch)
├── tests/                42 tests (unit + API + live server E2E)
├── scripts/              Real model E2E test scripts
├── ui/                   Next.js 16 UI (drag-drop upload + results)
├── Makefile              test, test-unit, test-e2e, lint, serve, dev
├── pyproject.toml        Project config + CUDA torch index
└── README.md
```

## License

MIT

---

## Testing

```bash
# Run all tests (42 tests: unit + API + end-to-end)
make test

# Unit tests only (document parsing, VL model, extractor, API)
make test-unit

# End-to-end tests (live server + real file uploads)
make test-e2e

# Lint (Python + TypeScript)
make lint
```

| Test Suite | What it covers |
|---|---|
| `test_document.py` | PDF/image/GIF loading, page extraction, error handling |
| `test_vl_model.py` | Model init, query/describe/extract_json with mocked model |
| `test_extractor.py` | Full extraction pipeline with mocked VL model |
| `test_api.py` | FastAPI endpoints via TestClient (health, extract, describe, ask) |
| `test_e2e.py` | Live Uvicorn server, real multipart uploads, CORS, error handling |

## Verified With Real Model

Tested end-to-end on RTX 3070 Ti (8GB VRAM) with a real 2-page PDF (image invoice + digital text page):

| Operation | Time | Result |
|---|---|---|
| Describe | 9s | Both pages captioned accurately |
| Extract | 42s | All items, prices, GSTIN, bank details as JSON |
| Ask | 13s | "Grand Total: 2100 INR" + "Payment Terms: Net 30" |

Handles both scanned/image-based PDF pages (VL model reads the image) and digital-text PDF pages (text extracted via pymupdf).
