"""
RAG (Retrieval-Augmented Generation) Engine for FoxAI Studio.

This module handles the RAG pipeline:
1. PROCESSING: Load various file formats (PDF, DOCX, Code, ZIM)
2. CHUNKING: Split documents into manageable chunks
3. EMBEDDING: Convert text chunks into vector representations
4. INDEXING: Store vectors in FAISS for fast similarity search
5. RETRIEVAL: Find relevant chunks given a user query

KEY CONCEPTS FOR LEARNING:
- Embeddings: Numerical vector representations of text. Similar texts have similar vectors.
- FAISS: Facebook AI Search Similarity - fast nearest-neighbor search on vectors
- Chunking: Splitting large documents into smaller pieces for better retrieval
- L2 Distance: How we measure "closeness" of vectors in embedding space

USAGE:
    from rag_engine import RAGEngine
    engine = RAGEngine()
    engine.ingest_documents(["path/to/file.pdf", "path/to/code.py"])
    context = engine.query("What is this project about?")
"""

import os
import glob
import json
import numpy as np
from typing import List, Dict, Any, Optional

# FAISS for vector similarity search
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False
    print("Warning: faiss not installed. Run: pip install faiss-cpu")

SentenceTransformer = None
HAS_SENTENCE_TRANSFORMERS = False

# PDF processing library - extracts text from PDF files
try:
    import fitz  # PyMuPDF - reads PDFs and extracts text/images
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not installed. Run: pip install pymupdf")

# DOCX processing library - extracts text from Word documents
try:
    import docx  # python-docx - reads .docx files
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("Warning: python-docx not installed. Run: pip install python-docx")

# Image OCR (optional) - extracts text from images (JPG/PNG/etc.)
try:
    from PIL import Image  # pillow
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False
    print("Warning: pillow not installed. Run: pip install pillow")

try:
    import pytesseract  # requires system tesseract installed
    HAS_TESSERACT = True
except ImportError:
    pytesseract = None
    HAS_TESSERACT = False
    print("Warning: pytesseract not installed. Run: pip install pytesseract (also requires 'tesseract' installed on your OS)")

DEFAULT_RAG_DIR = os.path.join(os.path.expanduser("~"), ".lokumai", "rag")
DEFAULT_INDEX_NAME = "faiss_index.bin"
DEFAULT_DOCS_NAME = "docs_metadata.npy"
DEFAULT_META_NAME = "rag_meta.json"

# ============================================================================
# RAG ENGINE CLASS
# ============================================================================
class RAGEngine:
    """
    Main RAG engine class. Handles:
    - Loading files in various formats
    - Chunking text into manageable pieces
    - Creating embeddings using sentence-transformers
    - Storing and searching vectors using FAISS

    STREAMLINED VERSION WITHOUT LANGCHAIN:
    We use raw FAISS + sentence-transformers instead of Langchain because:
    - Fewer dependencies to manage
    - More control over the pipeline
    - Faster for our specific use case
    """

    def __init__(self, storage_dir: str | None = None):
        # Check if we have all required dependencies
        global SentenceTransformer, HAS_SENTENCE_TRANSFORMERS
        if not HAS_SENTENCE_TRANSFORMERS:
            try:
                from sentence_transformers import SentenceTransformer as _SentenceTransformer
                SentenceTransformer = _SentenceTransformer
                HAS_SENTENCE_TRANSFORMERS = True
            except Exception as e:
                HAS_SENTENCE_TRANSFORMERS = False
                SentenceTransformer = None
                print(f"Warning: sentence-transformers not available ({e}). Install: pip install sentence-transformers")

        self.enabled = bool(HAS_SENTENCE_TRANSFORMERS and HAS_FAISS)
        if not self.enabled:
            return

        self.storage_dir = os.path.abspath(storage_dir or DEFAULT_RAG_DIR)
        os.makedirs(self.storage_dir, exist_ok=True)
        self.index_path = os.path.join(self.storage_dir, DEFAULT_INDEX_NAME)
        self.docs_path = os.path.join(self.storage_dir, DEFAULT_DOCS_NAME)
        self.meta_path = os.path.join(self.storage_dir, DEFAULT_META_NAME)
        self.indexed_folder: str = ""

        # Load the embedding model
        # This model is downloaded on first use (~90MB) and cached
        # 'all-MiniLM-L6-v2' creates 384-dimensional vectors
        # It maps any text to a point in 384D space where similar texts are close
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # FAISS index - will be initialized when we have vectors
        # None means no index loaded yet
        self.index: Optional[faiss.Index] = None

        # Store original text chunks so we can return them on query
        # This list is parallel to the FAISS index:
        # - self.documents[0] corresponds to vector at index 0 in FAISS
        self.documents: List[str] = []
        self.last_error: str = ""

        # Load existing index if available (persistent across restarts)
        self.load_index()

    def _set_last_error(self, msg: str) -> None:
        try:
            self.last_error = (msg or "").strip()
        except Exception:
            pass

    def load_index(self) -> None:
        """
        Load previously saved FAISS index from disk.

        HOW IT WORKS:
        - FAISS index contains all the vectors for our documents
        - numpy file contains the original text (needed to return context)
        - Both files must exist and match for valid loading

        WHY PERSIST?
        - Creating embeddings is slow (one forward pass per chunk)
        - We only need to do it once, then reuse the index
        """
        meta_folder = ""
        try:
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
                meta_folder = str(meta.get("folder") or "").strip()
        except Exception:
            meta_folder = ""

        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            try:
                # Read FAISS index from binary file
                self.index = faiss.read_index(self.index_path)

                # Read document chunks from numpy file
                # allow_pickle=True needed because numpy can't store plain lists
                self.documents = np.load(self.docs_path, allow_pickle=True).tolist()
                self.indexed_folder = meta_folder

                print(f"[RAG] Loaded index with {len(self.documents)} chunks.")
            except Exception as e:
                print(f"[RAG] Error loading index: {e}")
                # Reset to empty state if loading fails
                self.index = None
                self.documents = []
                self.indexed_folder = ""

    def save_index(self) -> None:
        """
        Save current FAISS index and documents to disk.

        FILES CREATED:
        - faiss_index.bin: Binary FAISS index file
        - docs_metadata.npy: NumPy file with original text chunks

        NOTE: This overwrites any existing index!
        Call this after adding new documents to persist them.
        """
        if self.index is not None:
            # Write FAISS index to binary file
            faiss.write_index(self.index, self.index_path)

            # Save document chunks as numpy array
            # dtype=object needed for list of strings
            np.save(self.docs_path, np.array(self.documents, dtype=object))
            try:
                with open(self.meta_path, "w", encoding="utf-8") as f:
                    json.dump({"folder": self.indexed_folder}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            print(f"[RAG] Saved {len(self.documents)} chunks to index.")

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 800,
        overlap: int = 100
    ) -> List[str]:
        """
        Split long text into overlapping chunks.

        ARGS:
            text: The full text to chunk
            chunk_size: Target size of each chunk (in characters)
            overlap: How many characters to overlap between chunks

        RETURNS:
            List of text chunks

        WHY OVERLAP?
        - Ensures context isn't cut mid-sentence
        - If a concept spans two chunks, overlap helps capture it

        EXAMPLE:
            chunk_text("Hello world test", chunk_size=6, overlap=2)
            -> ["Hello world", "world test"]
        """
        chunks = []

        # Walk through text with step of (chunk_size - overlap)
        # This creates overlapping windows
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk:  # Skip empty chunks
                chunks.append(chunk.strip())

        return chunks

    # =========================================================================
    # FILE FORMAT HANDLERS
    # Each format needs its own extraction method
    # =========================================================================

    def extract_from_pdf(self, file_path: str) -> str:
        """
        Extract all text from a PDF file.

        HOW PDFs WORK:
        - PDFs store text as positioned characters or as embedded images
        - PyMuPDF (fitz) extracts visible text, ignoring formatting
        - Images containing text (scanned PDFs) won't be extracted

        ARGS:
            file_path: Path to .pdf file

        RETURNS:
            Extracted text as string, or empty string if failed
        """
        if not HAS_PYMUPDF:
            return ""

        try:
            text_parts = []

            # Open PDF with PyMuPDF
            doc = fitz.open(file_path)

            # Iterate through each page
            for page_num, page in enumerate(doc):
                # Get text from this page
                page_text = page.get_text()
                text_parts.append(page_text)

            doc.close()  # Always close the document

            # Join all pages with page break markers
            full_text = "\n\n--- Page Break ---\n\n".join(text_parts)
            return full_text

        except Exception as e:
            print(f"[RAG] PDF extraction error for {file_path}: {e}")
            return ""

    def extract_from_docx(self, file_path: str) -> str:
        """
        Extract all text from a Word (.docx) document.

        HOW DOCX WORKS:
        - DOCX is a ZIP archive containing XML files
        - python-docx parses the XML to extract paragraphs
        - It ignores images, formatting (mostly), and some complex elements

        ARGS:
            file_path: Path to .docx file

        RETURNS:
            Extracted text as string, or empty string if failed
        """
        if not HAS_DOCX:
            return ""

        try:
            doc = docx.Document(file_path)
            paragraphs = []

            # Each paragraph is a text element in Word
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)

            # Join paragraphs with double newlines (like in the original doc)
            return "\n\n".join(paragraphs)

        except Exception as e:
            print(f"[RAG] DOCX extraction error for {file_path}: {e}")
            return ""

    def extract_from_zim(self, file_path: str) -> str:
        """
        Extract text content from a ZIM archive.

        WHAT IS ZIM?
        - ZIM is a format for storing Wikipedia and other offline content
        - Used by Kiwix to create offline knowledge bases
        - Contains article text, images, and metadata

        HOW IT WORKS:
        - ZIM files are organized as article entries
        - Each entry has a URL, title, and content
        - We iterate through entries and extract text

        NOTE: This requires the 'zim' library. If not available, returns a
        placeholder message. See: https://github.com/openzim/python-zim

        ARGS:
            file_path: Path to .zim file

        RETURNS:
            Extracted text or placeholder message
        """
        text_parts = []
        try:
            pyzim_err = None
            pyzim_mod = None
            try:
                import pyzim as _pyzim  # published on PyPI as python-zim
                pyzim_mod = _pyzim
            except Exception as e:
                pyzim_err = str(e)

            libzim_err = None
            LibZimArchive = None
            try:
                from libzim.reader import Archive as _LibZimArchive
                LibZimArchive = _LibZimArchive
            except Exception as e:
                libzim_err = str(e)

            if LibZimArchive is not None:
                try:
                    zf = LibZimArchive(file_path)
                    n = getattr(zf, "entry_count", None)
                    if callable(n):
                        n = n()
                    if not isinstance(n, int):
                        n = 0
                    scanned = 0
                    kept = 0
                    skipped_ns = 0
                    skipped_nontext = 0
                    skipped_resource = 0
                    read_fail = 0

                    def is_article_entry(entry) -> bool:
                        ns = getattr(entry, "namespace", None)
                        if isinstance(ns, str) and ns:
                            return ns.upper() == "A"
                        url = ""
                        for key in ("url", "path", "full_url"):
                            v = getattr(entry, key, None)
                            if isinstance(v, str) and v:
                                url = v
                                break
                        if url:
                            low = url.lower()
                            if low.startswith("a/") or low.startswith("a:") or low.startswith("a"):
                                return True
                            return False
                        return True

                    max_scan = min(max(500, n), 20000) if n > 0 else 20000
                    for i in range(max_scan):
                        try:
                            entry = None
                            if hasattr(zf, "get_entry_by_id"):
                                entry = zf.get_entry_by_id(i)
                            elif hasattr(zf, "get_entry"):
                                entry = zf.get_entry(i)
                            if entry is None:
                                continue
                            scanned += 1
                            title = (getattr(entry, "title", "") or "").strip()
                            if not title:
                                continue
                            if not is_article_entry(entry):
                                skipped_ns += 1
                                continue
                            mimetype = (getattr(entry, "mimetype", "") or "").lower()
                            if title.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp")):
                                skipped_resource += 1
                                continue
                            if mimetype and not mimetype.startswith("text/") and "xml" not in mimetype and "html" not in mimetype:
                                skipped_nontext += 1
                                continue
                            item = entry.get_item() if hasattr(entry, "get_item") else None
                            raw = bytes(item.content) if (item is not None and hasattr(item, "content")) else b""
                            content = raw.decode("utf-8", errors="ignore").strip()
                            if content:
                                text_parts.append(f"## {title}\n\n{content}")
                                kept += 1
                            if len(text_parts) >= 200:
                                break
                        except Exception:
                            read_fail += 1
                            continue
                    out = "\n\n".join(text_parts).strip()
                    if out:
                        return out
                    self._set_last_error(
                        f"ZIM (libzim) extracted 0 text entries (scanned={scanned}, skipped_ns={skipped_ns}, skipped_nontext={skipped_nontext}, skipped_resource={skipped_resource}, read_fail={read_fail})."
                    )
                    return ""
                except Exception as e:
                    self._set_last_error(f"ZIM (libzim) read failed: {e}")
                    return ""

            if pyzim_mod is not None:
                try:
                    with pyzim_mod.Zim.open(file_path) as zf:
                        it = None
                        if hasattr(zf, "iter_entries"):
                            it = zf.iter_entries()
                        elif hasattr(zf, "iter_content_entries"):
                            it = zf.iter_content_entries()
                        elif hasattr(zf, "entries"):
                            it = getattr(zf, "entries")
                        if it is None:
                            self._set_last_error("ZIM (pyzim) could not iterate entries (unsupported API).")
                            return ""

                        scanned = 0
                        kept = 0
                        skipped_ns = 0
                        skipped_nontext = 0
                        skipped_resource = 0
                        read_fail = 0

                        def is_article_entry(entry) -> bool:
                            ns = getattr(entry, "namespace", None)
                            if isinstance(ns, str) and ns:
                                return ns.upper() == "A"
                            url = getattr(entry, "url", None)
                            if isinstance(url, str) and url:
                                low = url.lower()
                                if low.startswith("a/") or low.startswith("a:") or low.startswith("a"):
                                    return True
                                return False
                            return True

                        for entry in it:
                            scanned += 1
                            title = (getattr(entry, "title", "") or "").strip()
                            mimetype = (getattr(entry, "mimetype", "") or "").lower()
                            if not title:
                                continue
                            if not is_article_entry(entry):
                                skipped_ns += 1
                                continue
                            if title.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp")):
                                skipped_resource += 1
                                continue
                            if mimetype and not mimetype.startswith("text/"):
                                skipped_nontext += 1
                                continue
                            try:
                                content = entry.read()
                                if isinstance(content, bytes):
                                    content = content.decode("utf-8", errors="ignore")
                                content = (content or "").strip()
                            except Exception:
                                read_fail += 1
                                content = ""
                            if content:
                                text_parts.append(f"## {title}\n\n{content}")
                                kept += 1
                            if len(text_parts) >= 200:
                                break
                    out = "\n\n".join(text_parts).strip()
                    if out:
                        return out
                    self._set_last_error(
                        f"ZIM (pyzim) extracted 0 text entries (scanned={scanned}, skipped_ns={skipped_ns}, skipped_nontext={skipped_nontext}, skipped_resource={skipped_resource}, read_fail={read_fail})."
                    )
                    return ""
                except Exception as e:
                    self._set_last_error(f"ZIM (pyzim) read failed: {e}")
                    return ""

            detail = []
            if libzim_err:
                detail.append(f"libzim import error: {libzim_err}")
            if pyzim_err:
                detail.append(f"pyzim import error: {pyzim_err}")
            msg = "ZIM support not available. Install: pip install libzim OR pip install 'python-zim[all]'."
            if detail:
                msg += " " + " | ".join(detail)
            self._set_last_error(msg)
            return ""
        except Exception as e:
            self._set_last_error(f"ZIM extraction error: {e}")
            print(f"[RAG] ZIM extraction error for {file_path}: {e}")
            return ""

    def extract_from_image(self, file_path: str) -> str:
        """
        Extract text from images using OCR.

        Supported formats:
        - .jpg, .jpeg, .png, .webp, .bmp, .tif, .tiff

        Requirements:
        - pip install pillow pytesseract
        - Install 'tesseract' on the system (macOS: brew install tesseract)
        """
        if not (HAS_PIL and HAS_TESSERACT):
            return ""
        try:
            img = Image.open(file_path)
            txt = pytesseract.image_to_string(img)
            return (txt or "").strip()
        except Exception as e:
            print(f"[RAG] Image OCR error for {file_path}: {e}")
            return ""

    def extract_from_code(self, file_path: str) -> str:
        """
        Extract text from code/source files.

        SUPPORTED FORMATS:
        - .py (Python)
        - .cpp, .c (C/C++)
        - .h, .hpp (Header files)
        - .js (JavaScript)
        - .html, .htm (HTML)
        - .css (Stylesheets)
        - .txt, .md (Plain text/Markdown)
        - .json (JSON - treat as text)
        - .xml (XML)
        - .yaml, .yml (YAML)
        - .sh (Shell scripts)
        - Any other text-based file

        WHY TREAT CODE DIFFERENTLY?
        - Code has its own structure (functions, classes)
        - We preserve line breaks to maintain code structure
        - Comments can be valuable for understanding intent

        ARGS:
            file_path: Path to code file

        RETURNS:
            File content as string, or empty string if failed
        """
        try:
            # Try common encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue

            # If all encodings fail
            print(f"[RAG] Could not decode file: {file_path}")
            return ""

        except Exception as e:
            print(f"[RAG] Code extraction error for {file_path}: {e}")
            return ""

    # =========================================================================
    # MAIN FILE PROCESSING
    # =========================================================================

    def process_file(self, file_path: str) -> List[str]:
        """
        Load a single file and return its text chunks.

        ROUTING LOGIC:
        This method determines which extractor to use based on file extension.
        For unsupported formats, returns empty list (won't be indexed).

        HOW TO ADD A NEW FORMAT:
        1. Add the extension to the appropriate extractor
        2. Or create a new extractor method and add it here

        ARGS:
            file_path: Path to the file to process

        RETURNS:
            List of text chunks from this file
        """
        if not self.enabled:
            return []

        # Get lowercase file extension (e.g., '.pdf')
        ext = os.path.splitext(file_path)[1].lower()

        self._set_last_error("")
        # Route to appropriate extractor based on file type
        try:
            if ext == '.pdf':
                # PDF files - use PyMuPDF
                content = self.extract_from_pdf(file_path)

            elif ext in ['.docx', '.doc']:
                # Word documents
                content = self.extract_from_docx(file_path)

            elif ext == '.zim':
                # Offline Wikipedia / knowledge base archives
                content = self.extract_from_zim(file_path)

            elif ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff']:
                content = self.extract_from_image(file_path)

            elif ext in [
                '.py', '.cpp', '.c', '.h', '.hpp', '.js', '.ts',
                '.html', '.htm', '.css', '.scss', '.sass', '.less',
                '.txt', '.md', '.markdown', '.rst',
                '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg',
                '.sh', '.bash', '.zsh', '.csh', '.ps1',
                '.r', '.java', '.kt', '.swift', '.go', '.rs', '.rb',
                '.php', '.pl', '.pm', '.lua', '.scala', '.clj', '.ex', '.exs',
                '.sql', '.graphql', '.gql',
                '.vim', '.editorconfig', '.gitignore', '.dockerfile',
                '.makefile', '.cmake',
            ]:
                # Code and text files - direct read
                content = self.extract_from_code(file_path)

            else:
                # Unknown format - try as plain text anyway
                content = self.extract_from_code(file_path)

            # Split content into chunks for better retrieval
            if content:
                return self.chunk_text(content)

            return []

        except Exception as e:
            self._set_last_error(f"Error processing {file_path}: {e}")
            print(f"[RAG] Error processing {file_path}: {e}")
            return []

    # =========================================================================
    # BULK INDEXING
    # =========================================================================

    def ingest_documents(self, file_paths: List[str]) -> bool:
        """
        Add multiple files to the RAG index.

        PROCESS:
        1. Process each file -> get text chunks
        2. Create embeddings for all chunks (batch processing)
        3. Add vectors to FAISS index
        4. Extend document list
        5. Save to disk

        ARGS:
            file_paths: List of file paths to index

        RETURNS:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        added = 0
        failures: List[str] = []
        pending_save = 0

        def encode_batch(texts: List[str]):
            try:
                return self.embedding_model.encode(texts, batch_size=32, show_progress_bar=False)
            except TypeError:
                return self.embedding_model.encode(texts)

        for path in file_paths:
            chunks = self.process_file(path)
            if not chunks:
                ext = os.path.splitext(path)[1].lower()
                if ext == ".zim":
                    le = getattr(self, "last_error", "") or ""
                    failures.append(f"{os.path.basename(path)}: {le or 'No text extracted from this ZIM.'}")
                continue

            ext = os.path.splitext(path)[1].lower()
            max_per_file = 2500 if ext == ".zim" else 1200
            if len(chunks) > max_per_file:
                chunks = chunks[:max_per_file]

            for i in range(0, len(chunks), 128):
                batch = chunks[i:i + 128]
                if not batch:
                    continue
                emb = encode_batch(batch)
                emb = np.array(emb).astype("float32")
                if emb.ndim != 2 or emb.shape[0] != len(batch):
                    raise RuntimeError("Embedding model returned invalid shape.")
                dim = int(emb.shape[1])
                if self.index is None:
                    self.index = faiss.IndexFlatL2(dim)
                self.index.add(emb)
                self.documents.extend(batch)
                added += len(batch)
                pending_save += len(batch)
                if pending_save >= 2000:
                    self.save_index()
                    pending_save = 0

        if added <= 0:
            msg = "No content extracted from files."
            if failures:
                msg += "\n\n" + "\n".join(failures[:6])
            raise RuntimeError(msg)

        if pending_save > 0:
            self.save_index()

        return True

    def ingest_folder(self, folder_path: str, recursive: bool = True) -> bool:
        """
        Index all supported files in a folder.

        ARGS:
            folder_path: Path to folder to scan
            recursive: If True, scan subdirectories too

        RETURNS:
            True if successful
        """
        if not os.path.isdir(folder_path):
            print(f"[RAG] Invalid folder: {folder_path}")
            return False

        folder_abs = os.path.abspath(folder_path)
        if self.indexed_folder and os.path.abspath(self.indexed_folder) != folder_abs:
            self.reset_database()
        if not self.indexed_folder:
            self.indexed_folder = folder_abs
        if os.path.abspath(self.indexed_folder) == folder_abs and self.index is not None and self.documents:
            return True

        # Collect all files we can process
        file_paths = []

        # Extensions we support
        extensions = [
            '*.pdf', '*.docx', '*.doc', '*.zim',
            '*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.tif', '*.tiff',
            '*.py', '*.cpp', '*.c', '*.h', '*.hpp', '*.js', '*.ts',
            '*.html', '*.htm', '*.css', '*.txt', '*.md',
            '*.json', '*.xml', '*.yaml', '*.yml', '*.sh', '*.ino',
        ]

        for ext in extensions:
            if recursive:
                pattern = os.path.join(folder_abs, '**', ext)
                file_paths.extend(glob.glob(pattern, recursive=True))
            else:
                pattern = os.path.join(folder_abs, ext)
                file_paths.extend(glob.glob(pattern))

        print(f"[RAG] Found {len(file_paths)} files in {folder_abs}")
        return self.ingest_documents(file_paths)

    # =========================================================================
    # QUERY / RETRIEVAL
    # =========================================================================

    def query(self, query_text: str, k: int = 3) -> str:
        """
        Find the k most relevant document chunks for a query.

        HOW IT WORKS:
        1. Embed the query text (same model used for indexing)
        2. Search FAISS for k nearest vectors
        3. Retrieve original text for those vectors
        4. Join into context string

        ARGS:
            query_text: The user's question/query
            k: Number of chunks to retrieve (default: 3)

        RETURNS:
            Context string with retrieved chunks, or empty string if no results
        """
        if not self.enabled or self.index is None:
            return ""

        try:
            # Embed the query using the same model
            query_vector = self.embedding_model.encode([query_text])
            query_vector = np.array(query_vector).astype('float32')

            # Search FAISS for k nearest neighbors
            # Returns both distances and indices
            # distances: L2 distance to each result (lower = better match)
            # indices: Position of result in our documents list
            distances, indices = self.index.search(query_vector, k)

            # Retrieve original text for each matched index
            results = []
            for idx in indices[0]:
                # FAISS returns -1 for "no result"
                if idx != -1 and idx < len(self.documents):
                    results.append(self.documents[idx])

            if not results:
                return ""

            # Join results with separators for context
            # This becomes the RAG context injected into the prompt
            context_str = "\n\n---\n\n".join(results)
            return context_str

        except Exception as e:
            print(f"[RAG] Query error: {e}")
            return ""

    def query_with_sources(self, query_text: str, k: int = 3) -> Dict[str, Any]:
        """
        Find relevant chunks AND return metadata about matches.

        USEFUL FOR:
        - Debugging retrieval quality
        - Showing users which documents were used
        - Building citations/references

        Returns a dict with:
        - 'context': The combined context string
        - 'chunks': List of individual chunks
        - 'distances': L2 distances for each chunk
        - 'count': Number of chunks found
        """
        if not self.enabled or self.index is None:
            return {"context": "", "chunks": [], "distances": [], "count": 0}

        try:
            query_vector = self.embedding_model.encode([query_text])
            query_vector = np.array(query_vector).astype('float32')

            distances, indices = self.index.search(query_vector, k)

            results = []
            dists = []

            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.documents):
                    results.append(self.documents[idx])
                    dists.append(float(dist))

            return {
                "context": "\n\n---\n\n".join(results),
                "chunks": results,
                "distances": dists,
                "count": len(results)
            }

        except Exception as e:
            print(f"[RAG] Query error: {e}")
            return {"context": "", "chunks": [], "distances": [], "count": 0}

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.

        Returns:
            Dict with index statistics
        """
        if self.index is None:
            return {
                "enabled": self.enabled,
                "chunk_count": 0,
                "index_size": 0,
                "embedding_dim": 0,
                "indexed": False,
                "folder": self.indexed_folder,
            }

        return {
            "enabled": self.enabled,
            "chunk_count": len(self.documents),
            "index_size": self.index.ntotal,
            "embedding_dim": self.index.d if hasattr(self.index, 'd') else 384,
            "indexed": True,
            "folder": self.indexed_folder,
        }

    def reset_database(self) -> None:
        """
        Clear all indexed data and delete saved files.

        WARNING: This is destructive! All indexed content will be lost.
        Use with caution in production.
        """
        # Remove persisted files
        try:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
        except Exception:
            pass
        try:
            if os.path.exists(self.docs_path):
                os.remove(self.docs_path)
        except Exception:
            pass
        try:
            if os.path.exists(self.meta_path):
                os.remove(self.meta_path)
        except Exception:
            pass

        # Clear in-memory data
        self.index = None
        self.documents = []
        self.indexed_folder = ""

        print("[RAG] Index reset. All data cleared.")

    def get_relevant_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """
        Simple interface to just get the text chunks without formatting.

        ARGS:
            query: Search query
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant text chunks
        """
        result = self.query_with_sources(query, top_k)
        return result.get("chunks", [])


# ============================================================================
# STANDALONE TEST
# ============================================================================
if __name__ == "__main__":
    # Test the RAG engine standalone
    print("Testing RAG Engine...")
    print("-" * 40)

    engine = RAGEngine()

    if engine.enabled:
        print(f"Engine enabled: {engine.enabled}")
        print(f"Stats: {engine.get_stats()}")

        # Example: Index a folder
        # engine.ingest_folder("./sample_docs")

        # Example: Query
        # context = engine.query("What is machine learning?")
        # print(f"Context: {context[:200]}...")
    else:
        print("RAG engine not available. Install dependencies:")
        print("  pip install sentence-transformers faiss-cpu pymupdf python-docx")
