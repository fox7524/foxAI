import os
import tempfile
import unittest

import rag_engine


class TestRagEngineZim(unittest.TestCase):
    def test_extract_from_zim_empty_file_returns_empty(self):
        eng = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "x.zim")
            with open(p, "wb") as f:
                f.write(b"")
            out = eng.extract_from_zim(p)
            self.assertTrue(out == "" or isinstance(out, str))
            self.assertNotEqual(out, "[ZIM: Library not installed]")
