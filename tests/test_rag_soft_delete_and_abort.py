import os
import tempfile
import unittest

import numpy as np

import rag_engine


class _StubEmbedder:
    def encode(self, texts, batch_size=32, show_progress_bar=False):
        return np.zeros((len(texts), 3), dtype="float32")


class _StubIndex:
    def __init__(self, indices, distances):
        self._indices = np.array(indices, dtype="int64")
        self._distances = np.array(distances, dtype="float32")
        self.ntotal = int(self._indices.size)

    def search(self, vec, k):
        return self._distances[:, :k], self._indices[:, :k]


class TestRagSoftDeleteAndAbort(unittest.TestCase):
    def _make_engine(self):
        eng = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
        eng.enabled = True
        eng.embedding_model = _StubEmbedder()
        eng.index = _StubIndex(indices=[[0, 1]], distances=[[0.1, 0.2]])
        eng.documents = ["chunkA", "chunkB"]
        eng.chunk_meta = [{"file_id": "a"}, {"file_id": "b"}]
        eng.state = {"version": 1, "files": {"a": {"deleted": True}, "b": {"deleted": False}}}
        eng.last_error = ""
        eng._abort = False
        eng._set_last_error = rag_engine.RAGEngine._set_last_error.__get__(eng, rag_engine.RAGEngine)
        eng._check_abort = rag_engine.RAGEngine._check_abort.__get__(eng, rag_engine.RAGEngine)
        eng._is_file_deleted = rag_engine.RAGEngine._is_file_deleted.__get__(eng, rag_engine.RAGEngine)
        return eng

    def test_soft_deleted_chunks_are_filtered(self):
        eng = self._make_engine()
        res = rag_engine.RAGEngine.query_with_sources(eng, "q", k=1)
        self.assertEqual(res.get("chunks"), ["chunkB"])

    def test_abort_makes_query_return_empty(self):
        eng = self._make_engine()
        eng._abort = True
        out = rag_engine.RAGEngine.query(eng, "q", k=1)
        self.assertEqual(out, "")

    def test_abort_raises_in_ingest(self):
        if not getattr(rag_engine, "HAS_FAISS", False):
            self.skipTest("faiss not available")
        with tempfile.TemporaryDirectory() as td:
            storage = os.path.join(td, "ragstore")
            os.makedirs(storage, exist_ok=True)

            p = os.path.join(td, "a.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("hello world")

            eng = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
            eng.enabled = True
            eng.embedding_model = _StubEmbedder()
            eng.index = None
            eng.documents = []
            eng.chunk_meta = []
            eng.state = {"version": 1, "files": {}}
            eng.last_error = ""
            eng._abort = True
            eng.storage_dir = storage
            eng.index_path = os.path.join(storage, "faiss_index.bin")
            eng.docs_path = os.path.join(storage, "docs_metadata.npy")
            eng.meta_path = os.path.join(storage, "rag_meta.json")
            eng.chunks_meta_path = os.path.join(storage, "chunks_meta.npy")
            eng.state_path = os.path.join(storage, "rag_state.json")
            eng.staging_dir = os.path.join(storage, "staging")
            os.makedirs(eng.staging_dir, exist_ok=True)
            eng._set_last_error = rag_engine.RAGEngine._set_last_error.__get__(eng, rag_engine.RAGEngine)
            eng._check_abort = rag_engine.RAGEngine._check_abort.__get__(eng, rag_engine.RAGEngine)
            eng._file_id_for = rag_engine.RAGEngine._file_id_for.__get__(eng, rag_engine.RAGEngine)
            eng._load_state = rag_engine.RAGEngine._load_state.__get__(eng, rag_engine.RAGEngine)
            eng._validate_or_quarantine_existing_store = rag_engine.RAGEngine._validate_or_quarantine_existing_store.__get__(eng, rag_engine.RAGEngine)

            with self.assertRaises(RuntimeError):
                eng.ingest_documents([p])
