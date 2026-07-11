# ADR: Architecture Decision Record — mmdoc

## Status
Accepted

## Context
Build a document understanding system that reads any PDF/image and returns structured JSON, descriptions, or answers — running entirely on local GPU with zero API keys.

## Decisions

### D1: Qwen3.5-0.8B as the VL Model
**Decision:** Use `Qwen/Qwen3.5-0.8B` (architecture: `Qwen3_5ForConditionalGeneration`) in 4-bit quantization.

**Rationale:** 0.8B params fits in ~1GB VRAM (4-bit), making it viable on consumer GPUs (RTX 3060+). Despite small size, Qwen3.5 handles OCR, structured extraction, and QA adequately for a portfolio demo. The hybrid linear+full attention architecture (qwen3_5) provides good throughput.

**Critical bug found during testing:** Original code used `Qwen3VLForConditionalGeneration` (wrong class). The model's actual architecture is `Qwen3_5ForConditionalGeneration` — loading with the wrong class causes `RuntimeError` (MISSING/UNEXPECTED/MISMATCH weights). Fixed and verified with real model.

**Alternatives considered:**
- Qwen2-VL-2B: larger, better accuracy, but needs 2GB+ VRAM
- Llama 3.2 Vision: good but heavier
- Cloud OCR (Tesseract + GPT-4): rejected — defeats the "100% local" goal

**Tradeoffs:** 0.8B model has limited accuracy on complex documents (struggles with `$` symbols in small images). Acceptable for portfolio — the architecture matters more than model size.

### D2: PyMuPDF for PDF Rendering
**Decision:** Render PDFs at 200 DPI via `pymupdf` (fitz), extract digital text alongside images.

**Rationale:** 200 DPI balances quality and memory. Digital text extraction is free (no model needed) and enables a "smart" path: skip VL model for pages with extractable text, use VL only for scanned/image pages. PyMuPDF is the fastest Python PDF library.

### D3: Per-Page Processing Architecture
**Decision:** Process each page independently — render to PIL image → VL model → collect results.

**Rationale:** Enables future batching, parallelism, and selective processing (skip pages with digital text). Also makes the API stateless — each call is independent.

**Tradeoffs:** No cross-page context (can't answer "what was discussed on the previous page"). Acceptable for document extraction use cases.

### D4: FastAPI + Next.js (Same Pattern as All Projects)
**Decision:** Python FastAPI backend + Next.js 16 frontend, direct browser-to-API calls (CORS `*`).

**Rationale:** Consistent stack across portfolio projects. FastAPI is async, typed, generates OpenAPI docs automatically. Next.js App Router with Tailwind 4 is the current standard for React frontends.

### D5: Docker + CI from Day One
**Decision:** Multi-stage Dockerfile, docker-compose, GitHub Actions CI on every PR.

**Rationale:** Proves the system is deployable, not just runnable on a dev machine. CI catches regressions. Docker enables reproducible builds.

## Consequences
- System runs 100% locally with zero API keys
- Handles PDF, PNG, JPG, GIF, TIFF, BMP, WebP
- GPU inference: ~7s per page (describe), ~42s per page (extract)
- Docker-ready, CI-tested, 42 tests passing
