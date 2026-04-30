# Persistent RAG Data Management (LokumAI)

This document describes the persistent/cumulative RAG storage behavior implemented in the local RAG store under `~/.lokumai/rag/`.

## Goals

- Keep successfully indexed data permanently (“yığılmak”) so the AI can use it even if the original source files are later deleted.
- Avoid destructive resets when indexing a new folder/file; new successful data is appended to the existing store.
- Prevent partially written/half-indexed data when an indexing operation fails.
- Track indexing state per file and validate store integrity before writing.

## Storage Layout

Files under `~/.lokumai/rag/`:
- `faiss_index.bin`: vector index for all chunks (cumulative).
- `docs_metadata.npy`: stored chunk texts aligned with FAISS vector order.
- `chunks_meta.npy`: per-chunk metadata aligned with `docs_metadata.npy` (currently includes `file_id` and `source_path`).
- `rag_state.json`: per-file indexing state and chunk ranges.
- `rag_meta.json`: UI convenience metadata (last indexed folder).
- `staging/`: ephemeral per-run staging directory; deleted automatically after each run.

## Behavior

### Cumulative indexing (“yığılmak”)

- Indexing a new folder does not delete existing indexed data.
- If a file is unchanged since the last successful index (based on size+mtime), it is skipped.
- If a file is missing (deleted from disk), previously indexed chunks remain available because they are stored in `docs_metadata.npy`.

### Failure handling (delete only last generated RAG data)

- Indexing is commit-safe at the file level:
  - A file’s chunks are extracted and embedded first.
  - Only after embeddings are ready are vectors+chunks appended to the persistent store.
- If extraction/embedding fails for a file, nothing for that file is committed to the persistent store.
- Each indexing run creates a directory under `staging/` and always deletes it at the end (success or failure).

### State tracking

- Each indexed file has a record in `rag_state.json` keyed by a stable `file_id` (hash of absolute path).
- Records store:
  - `status`: `ok` or `failed`
  - `source_path`, `size`, `mtime`
  - `chunk_start` / `chunk_end` for successfully indexed files
  - `error` (when failed)
  - timestamps (`indexed_at`, `last_seen_at`)

### Validation / integrity checks

- Before indexing, the engine validates:
  - `FAISS.ntotal == len(docs_metadata)`
  - `len(chunks_meta) == len(docs_metadata)` (when `chunks_meta.npy` exists)
  - stored chunk ranges in `rag_state.json` are in bounds
- If integrity checks fail, the store files are quarantined by renaming them with a `.corrupt.<timestamp>` suffix and a new empty store is started (data is preserved on disk via the renamed files).

