import unittest

import rag_engine


class TestRagChunkTextGuard(unittest.TestCase):
    def test_chunk_text_overlap_ge_chunk_size_does_not_crash(self):
        eng = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
        chunks = eng.chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=10)
        self.assertTrue(chunks)
        self.assertTrue(all(isinstance(c, str) and c for c in chunks))

    def test_chunk_text_step_never_zero(self):
        eng = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
        chunks = eng.chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_size=5, overlap=999)
        self.assertTrue(chunks)
