# LokumAI — Technical Documentation (Architecture + GUI + RAG + Fine-tune)

This document explains how the LokumAI app works end-to-end: architecture, GUI structure, RAG pipeline, fine-tuning pipeline, storage, and integration points. It is written to be usable for demos and technical presentations.

---

## 1) High-level architecture

LokumAI is a single-process desktop application built with PyQt.

Core subsystems:
- **GUI (PyQt)**: sidebar chats + main chat view + developer tools panel.
- **Inference (MLX / mlx_lm)**: local model loading + token streaming.
- **RAG (FAISS + sentence-transformers)**: optional local retrieval over indexed files.
- **Fine-tuning (MLX LoRA)**: optional LoRA training via a subprocess, logs streamed into UI.
- **Persistence (SQLite)**: chat history stored in `app.db` (project folder).

Everything runs locally on the same machine. No HTTP server is required.

---

## 2) GUI implementation

### 2.1 Main window layout

The main window uses splitters and panels:
- **Left sidebar**: app title + “New chat”, search, chat list, hardware stats, settings, dev entry.
- **Main area**:
  - header (“Service: …” etc.)
  - chat transcript area (scrollable)
  - input box (text input + tools + send/stop)
- **Right dev sidebar**: embedded developer panel (RAG indexer, fine-tune, model, testing, unrestricted).

### 2.2 Chat transcript rendering

Chat transcript is rendered as real widgets inside a scroll area:
- **User messages**: right-aligned bubble (auto-wrapping, max-width cap).
- **Assistant messages**:
  - header “Lokum AI”
  - collapsible “Thought for X.XX seconds”
  - final answer area
  - optional metadata (tokens/sec etc.)

This widget-based approach avoids CSS/HTML rendering inconsistencies seen with QTextBrowser.

### 2.3 Developer Panel (Dev Mode)

Dev Mode is a right sidebar inside the main window (password-gated):
- **RAG Indexer**: index a folder or download+index Python docs, reset index.
- **Fine-tune**: start/stop LoRA training; stream logs.
- **Model**: select and load a model path.
- **Testing**: benchmark/stress tools.
- **Unrestricted**: toggles an alternate system prompt mode.

---

## 3) Inference & streaming

### 3.1 Model loading

The app loads an MLX model using `mlx_lm.load(...)` in a background thread.

### 3.2 Streaming tokens to UI

Generation uses `mlx_lm.stream_generate(...)` in a worker thread:
- Each incremental token delta is emitted to the UI.
- The UI appends answer text as tokens arrive.

---

## 4) Thinking vs final answer separation

LokumAI separates “thinking” text from final answers using:
1) **Streaming parser**: detects `<think>...</think>` / `<analysis>...</analysis>` blocks and routes them into a hidden/collapsible “Thought” section.
2) **Finalization logic**: when the model doesn’t provide clean tags or a final answer, the app can run a short second pass to request “final answer only”.

Goal:
- user sees clean final answers
- reasoning stays hidden/collapsible (demo-friendly)

---

## 5) RAG (Retrieval-Augmented Generation)

### 5.1 What RAG does

RAG builds a local searchable index of your files and retrieves the most relevant chunks during chat.

Pipeline:
1) Extract text from files
2) Chunk text into overlapping segments
3) Embed chunks with sentence-transformers
4) Store vectors in FAISS index
5) Query → nearest neighbors → retrieved chunks injected into the prompt

### 5.2 Supported file formats for RAG

Text/code:
- `.py`, `.cpp`, `.c`, `.h`, `.hpp`
- `.html`, `.css`, `.js`, `.ts`
- `.txt`, `.md`, `.json`, `.xml`, `.yaml` …

Documents:
- `.pdf` (PyMuPDF)
- `.docx` (python-docx)

Archives:
- `.zim` (python-zim)

Images (optional):
- `.jpg/.png/...` via OCR (pillow + pytesseract + system tesseract)

### 5.3 RAG runtime integration

At send-time:
- user message is used to query RAG
- retrieved text is injected as “Background info: …” into the prompt
- model generates response with that extra context

RAG is optional:
- if dependencies aren’t installed, the app disables indexing and warns with install instructions.

---

## 6) Fine-tuning (LoRA) — MLX

### 6.1 What fine-tuning does

Fine-tuning changes the model’s behavior (tone, formatting, rules) using LoRA adapters.
It is not a replacement for RAG knowledge retrieval; it’s best used for stable behavior changes.

### 6.2 Training pipeline

The app starts training via:
- `python -m mlx_lm lora ...`

The training runs in a subprocess:
- logs are streamed into the UI
- Stop button terminates the subprocess

### 6.3 Training data sources

Supported sources in the UI:
- **JSONL** file or folder
- **SQLite** dataset table (`dataset.db` with `dataset(instruction, output)`)

Extra helper:
- “Build Training Dataset From Files” can export a JSONL dataset from a folder containing files like `.py/.cpp/.html/.pdf/.zim/.jpg` (text extraction + chunking).

---

## 7) Storage and persistence

### 7.1 Chat history

Chats and messages are persisted to SQLite:
- `app.db` in the project folder

### 7.2 RAG index artifacts

RAG index is persisted as:
- `faiss_index.bin`
- `docs_metadata.npy`

### 7.3 Fine-tune artifacts

Training exports:
- datasets under `lora_data/`
- adapters saved by mlx_lm

---

## 8) Demo script (presentation-friendly)

Suggested demo flow:
1) Start app, show “Service: ready”
2) Create a new chat and ask a simple question (show streaming)
3) Open Dev Mode → index a small folder in RAG
4) Ask a question referencing a unique string in that folder (show that answer uses it)
5) Open Fine-tune → run a short training demo (show logs, stop button)
6) Mention persistence: restart app and show chats still exist

---

## 9) Dependencies summary

Core:
- PyQt5
- mlx_lm
- sqlite3 (stdlib)

RAG:
- sentence-transformers
- faiss-cpu

Optional file format support:
- pymupdf (fitz) for PDF
- python-docx for DOCX
- zim for ZIM
- pillow + pytesseract (+ system tesseract) for OCR on images
