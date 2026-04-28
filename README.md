# LokumAI — Local AI Chat Studio

**LokumAI** is a desktop-first AI chat app built with **PyQt** that runs locally on your machine: chat UI, local model inference, optional **RAG** (Retrieval-Augmented Generation) over your files, and optional **MLX LoRA** fine-tuning.

## Highlights

| Area | What it does |
|---|---|
| Chat UI | WhatsApp-like layout, streaming responses, Stop button, per-message menu |
| Dev Mode | Password-gated right sidebar (no separate dev window) |
| RAG | Index files → retrieve relevant chunks at chat-time (FAISS + sentence embeddings) |
| Fine-tune | Launches MLX LoRA training via `python -m mlx_lm lora …` with live logs + Stop |
| Persistence | Chats/messages saved in SQLite so history survives restarts |
| Multi-format ingest | Code/text + PDF + DOCX + ZIM + optional image OCR |

## Quickstart (macOS)

### 1) Create + activate a virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

Optional system dependency for OCR (images):

```bash
brew install tesseract
```

### 3) Run the app

```bash
python3 -u main.py
```

Wait until the UI shows `Service: ready`.

## Usage Examples

### Create a chat (auto-naming)
- Click **+ New**
- Send your first message
- The chat title is auto-named from that first message (and updates immediately)

### Use RAG (Retrieval-Augmented Generation)
1. Open **Dev Mode** (left sidebar) → enter the developer password
2. Dev sidebar → **RAG Indexer**
3. Choose a folder → **Index Project Files**
4. Ask a question that can only be answered from that folder’s content

RAG dependencies:
- Required: `sentence-transformers`, `faiss-cpu`
- Optional: `pymupdf` (PDF), `python-docx` (DOCX), `libzim`/`python-zim` (ZIM), `pytesseract` stack (images)

### Fine-tune (MLX LoRA)
1. Dev sidebar → **Fine-tune**
2. Select a dataset source:
   - JSONL file/folder (`train.jsonl` + `valid.jsonl`), or
   - SQLite dataset (`dataset.db`, table `dataset(instruction, output)`)
3. Click **Start Training** and watch logs in real time
4. Click **Stop** to terminate the training subprocess

Dataset helper:
- Dev → Fine-tune → **Build Training Dataset From Files** exports JSONL from a folder (extract → chunk → export).

## Project Files

| File/Folder | Purpose |
|---|---|
| `main.py` | App entrypoint: UI, streaming, chat persistence, Dev sidebar |
| `rag_engine.py` | RAG ingestion + indexing + retrieval |
| `file_ingest.py` | Shared extraction/chunking pipeline for multiple file formats |
| `finetune_engine.py` | MLX LoRA training subprocess orchestration |
| `app.db` | Chat history database |
| `lora_data/` | LoRA working/export outputs |

## Documentation

- Training + Dev guide: [TRAINING_GUIDE.txt](./TRAINING_GUIDE.txt)
- Architecture + demo reference: [TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md)
- Model ownership/data governance: [MODEL_OWNERSHIP_AND_DATA_GOVERNANCE.txt](./MODEL_OWNERSHIP_AND_DATA_GOVERNANCE.txt)

## Troubleshooting

### `Service: model path not found`
- The app auto-detects a local MLX model folder (LM Studio default location).
- Set a valid model directory in `prompts.json` (`model_path`) if auto-detect can’t find one.

### `RAG engine not available`
- Ensure dependencies are installed:
  - `pip install -r requirements.txt`
  - RAG needs `sentence-transformers` and `faiss-cpu`

### OCR returns empty text
- Install system `tesseract` and re-run:
  - `brew install tesseract`

## Contributing

1. Create a branch
2. Make changes
3. Run checks:

```bash
python3 -m py_compile main.py rag_engine.py file_ingest.py finetune_engine.py
python3 -m unittest discover -s tests -v
```

4. Open a PR (include screenshots for UI changes)

## Security Notes (Local-First)

- RAG indexes file content. Avoid indexing secrets (`.env`, API keys, credentials).
- Treat `app.db` and RAG index files as sensitive if your chats/docs contain private data.
