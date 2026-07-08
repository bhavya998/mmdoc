<div align="center">

# 📄 mmdoc — Multi-modal Document Understanding

### Upload any document — get structured JSON, descriptions, and answers. Powered by Qwen3.5-VL. 100% local GPU.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

**Zero API keys. Qwen3.5-0.8B runs on your machine.**

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

```bash
git clone https://github.com/bhavya998/mmdoc.git
cd mmdoc

# Backend
uv sync
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
├── ui/                   Next.js 16 UI (drag-drop upload + results)
├── pyproject.toml
└── README.md
```

## License

MIT
