# LokumAI Code Audit & Fix Report

This report documents issues found during a systematic code review of the repository, along with the implemented fixes and the tests that verify them.

## Summary

Focus areas:
- Crash-at-import risks (optional dependencies)
- Exception-prone logic in RAG chunking
- Non-functional UI artifacts in `roadmapp.html`
- Dependency manifest correctness

All changes are backward compatible: defaults remain unchanged, but the app now degrades more gracefully when optional dependencies are missing.

## Issues & Fixes

### 1) Crash-at-import when optional dependencies are missing

- **Location**
  - [main.py](file:///Users/fox/Documents/PROJECTS/foxAI/main.py)
- **Severity**: High
- **Impact**
  - If `mlx_lm` is missing/incompatible, the entire application fails to start at import time.
  - If `psutil` is missing, hardware monitoring crashes.
  - If `libzim` or `file_ingest` is missing, features fail with unclear error messages.
- **Root cause**
  - Unconditional top-level imports for optional libraries.
  - Missing dependency error strings for `file_ingest` import.
- **Fix**
  - Converted `mlx_lm`, `psutil`, `libzim` imports to optional imports with feature flags and captured error strings:
    - `HAS_MLX_LM`, `MLX_IMPORT_ERROR`
    - `HAS_PSUTIL`, `PSUTIL_IMPORT_ERROR`
    - `HAS_LIBZIM`, `LIBZIM_IMPORT_ERROR`
    - `INGEST_IMPORT_ERROR`
  - Added explicit runtime checks in worker threads so missing `mlx_lm` fails with a clear error message rather than a `NoneType is not callable` crash:
    - `ModelLoaderWorker`, `AIWorker`, `BenchmarkWorker`, `FinalAnswerWorker`
  - Made RAM monitoring degrade gracefully when `psutil` is unavailable (reports `N/A` instead of crashing).
- **Tests**
  - [tests/test_main_optional_imports.py](file:///Users/fox/Documents/PROJECTS/foxAI/tests/test_main_optional_imports.py)

### 2) Potential `range()` step bug in RAG chunking

- **Location**
  - [rag_engine.py](file:///Users/fox/Documents/PROJECTS/foxAI/rag_engine.py)
- **Severity**: High
- **Impact**
  - If `overlap >= chunk_size`, `range(…, chunk_size - overlap)` can become zero/negative and raise `ValueError` (or behave incorrectly), breaking RAG ingestion.
- **Root cause**
  - Missing normalization/guard for `chunk_size` and `overlap` parameters in `RAGEngine.chunk_text`.
- **Fix**
  - Normalized `chunk_size`/`overlap` and ensured the chunking step is always positive.
  - Preserved prior behavior for normal values.
- **Tests**
  - [tests/test_rag_chunk_text_guard.py](file:///Users/fox/Documents/PROJECTS/foxAI/tests/test_rag_chunk_text_guard.py)

### 3) Non-functional / malformed Phase Tabs in roadmap HTML

- **Location**
  - [roadmapp.html](file:///Users/fox/Documents/PROJECTS/foxAI/roadmapp.html)
- **Severity**: Medium
- **Impact**
  - Duplicate tab elements created duplicate `id` attributes (invalid HTML).
  - A tab referenced `switchPhase(3)` without a matching panel, causing a non-functional tab and potential JS/UI issues.
- **Root cause**
  - Leftover/duplicated tab markup outside the phase tabs container.
- **Fix**
  - Removed the duplicated tab block so the document has a single `#phase-tabs` container and consistent phase indices.
- **Tests**
  - [tests/test_roadmapp_html_structure.py](file:///Users/fox/Documents/PROJECTS/foxAI/tests/test_roadmapp_html_structure.py)

### 4) Incomplete dependency manifest (installation may be broken)

- **Location**
  - [requirements.txt](file:///Users/fox/Documents/PROJECTS/foxAI/requirements.txt)
- **Severity**: High
- **Impact**
  - Fresh installs may fail because core runtime dependencies were missing from `requirements.txt` even though the code imports them.
- **Root cause**
  - `PyQt5` (core GUI) and `numpy` (required by `rag_engine.py`) were not listed.
- **Fix**
  - Added `numpy` and `PyQt5` to `requirements.txt`.
- **Tests**
  - Covered indirectly via existing unit tests and compile checks in this audit run.

## Validation Performed

- Python syntax check:
  - `python3 -m py_compile main.py rag_engine.py file_ingest.py finetune_engine.py`
- Full test suite:
  - `python3 -m unittest discover -s tests -v`

