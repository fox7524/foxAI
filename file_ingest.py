import glob
import os
from typing import Iterable, List


try:
    import fitz  # type: ignore
    HAS_PYMUPDF = True
except Exception:
    fitz = None
    HAS_PYMUPDF = False

try:
    import docx  # type: ignore
    HAS_DOCX = True
except Exception:
    docx = None
    HAS_DOCX = False

try:
    import pyzim  # type: ignore
    HAS_PYZIM = True
except Exception:
    pyzim = None
    HAS_PYZIM = False

try:
    from libzim.reader import Archive as LibZimArchive  # type: ignore
    HAS_LIBZIM = True
except Exception:
    LibZimArchive = None
    HAS_LIBZIM = False

try:
    from PIL import Image  # type: ignore
    HAS_PIL = True
except Exception:
    Image = None
    HAS_PIL = False

try:
    import pytesseract  # type: ignore
    HAS_TESSERACT = True
except Exception:
    pytesseract = None
    HAS_TESSERACT = False


DEFAULT_EXT_GLOBS = [
    "*.py", "*.cpp", "*.c", "*.h", "*.hpp", "*.ino",
    "*.html", "*.htm", "*.css", "*.js", "*.ts",
    "*.md", "*.txt", "*.json", "*.xml", "*.yaml", "*.yml",
    "*.pdf", "*.docx", "*.doc", "*.zim",
    "*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tif", "*.tiff",
]


def iter_files(folder: str, recursive: bool = True, patterns: List[str] | None = None) -> List[str]:
    root = os.path.abspath(folder or "")
    if not root or not os.path.isdir(root):
        return []
    pats = patterns or DEFAULT_EXT_GLOBS
    out: List[str] = []
    for pat in pats:
        if recursive:
            out.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))
        else:
            out.extend(glob.glob(os.path.join(root, pat)))
    return sorted(set(out))


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    chunk_size = max(100, int(chunk_size))
    overlap = max(0, int(overlap))
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: List[str] = []
    start = 0
    n = len(s)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(s[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def extract_text(file_path: str) -> str:
    p = os.path.abspath(file_path or "")
    if not p or not os.path.isfile(p):
        return ""
    ext = os.path.splitext(p)[1].lower()

    if ext == ".pdf" and HAS_PYMUPDF and fitz is not None:
        try:
            doc = fitz.open(p)
            parts = []
            for i in range(len(doc)):
                parts.append(doc.load_page(i).get_text("text"))
            return "\n".join(parts).strip()
        except Exception:
            return ""

    if ext in (".docx", ".doc") and HAS_DOCX and docx is not None:
        try:
            d = docx.Document(p)
            return "\n".join((para.text or "") for para in d.paragraphs).strip()
        except Exception:
            return ""

    if ext == ".zim":
        texts: List[str] = []
        try:
            if HAS_PYZIM and pyzim is not None:
                with pyzim.Zim.open(p) as zf:
                    it = None
                    if hasattr(zf, "iter_entries"):
                        it = zf.iter_entries()
                    elif hasattr(zf, "iter_content_entries"):
                        it = zf.iter_content_entries()
                    elif hasattr(zf, "entries"):
                        it = getattr(zf, "entries")
                    if it is None:
                        return ""
                    for entry in it:
                        title = (getattr(entry, "title", "") or "").strip()
                        mimetype = (getattr(entry, "mimetype", "") or "").lower()
                        if not title:
                            continue
                        if title.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp")):
                            continue
                        if mimetype and not mimetype.startswith("text/"):
                            continue
                        try:
                            content = entry.read()
                            if isinstance(content, bytes):
                                content = content.decode("utf-8", errors="ignore")
                            content = (content or "").strip()
                        except Exception:
                            content = ""
                        if content:
                            texts.append(content)
                        if len(texts) >= 200:
                            break
                return "\n\n".join(texts).strip()

            if HAS_LIBZIM and LibZimArchive is not None:
                zf = LibZimArchive(p)
                n = getattr(zf, "entry_count", None)
                if callable(n):
                    n = n()
                if not isinstance(n, int):
                    n = None
                if n is None:
                    return ""
                for i in range(min(n, 500)):
                    try:
                        entry = None
                        if hasattr(zf, "get_entry_by_id"):
                            entry = zf.get_entry_by_id(i)
                        elif hasattr(zf, "get_entry"):
                            entry = zf.get_entry(i)
                        if entry is None:
                            continue
                        title = (getattr(entry, "title", "") or "").strip()
                        if not title:
                            continue
                        item = entry.get_item() if hasattr(entry, "get_item") else None
                        raw = bytes(item.content) if (item is not None and hasattr(item, "content")) else b""
                        content = raw.decode("utf-8", errors="ignore").strip()
                        if content:
                            texts.append(content)
                        if len(texts) >= 200:
                            break
                    except Exception:
                        continue
                return "\n\n".join(texts).strip()
        except Exception:
            return ""

    if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff") and HAS_PIL and HAS_TESSERACT and Image is not None and pytesseract is not None:
        try:
            img = Image.open(p)
            txt = pytesseract.image_to_string(img)
            return (txt or "").strip()
        except Exception:
            return ""

    try:
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                with open(p, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
    except Exception:
        return ""
    return ""


def build_text_chunks_from_paths(paths: Iterable[str], chunk_size: int = 800, overlap: int = 100) -> List[str]:
    chunks: List[str] = []
    for fp in paths:
        txt = extract_text(fp)
        if not txt:
            continue
        chunks.extend(chunk_text(txt, chunk_size=chunk_size, overlap=overlap))
    return chunks
