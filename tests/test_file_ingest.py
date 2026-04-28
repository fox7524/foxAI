import os
import tempfile
import unittest

import file_ingest


class TestFileIngest(unittest.TestCase):
    def test_chunk_text_basic(self):
        s = "a" * 250
        chunks = file_ingest.chunk_text(s, chunk_size=100, overlap=20)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(len(c) <= 100 for c in chunks))

    def test_iter_files_finds_supported(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = os.path.join(td, "a.py")
            p2 = os.path.join(td, "b.cpp")
            p3 = os.path.join(td, "c.unknown")
            with open(p1, "w", encoding="utf-8") as f:
                f.write("print('x')\n")
            with open(p2, "w", encoding="utf-8") as f:
                f.write("int main() {}\n")
            with open(p3, "w", encoding="utf-8") as f:
                f.write("nope\n")
            paths = file_ingest.iter_files(td, recursive=False)
            self.assertIn(os.path.abspath(p1), paths)
            self.assertIn(os.path.abspath(p2), paths)
            self.assertNotIn(os.path.abspath(p3), paths)

    def test_extract_text_plain(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "x.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("hello\nworld\n")
            out = file_ingest.extract_text(p)
            self.assertIn("hello", out)

    def test_extract_text_missing_returns_empty(self):
        out = file_ingest.extract_text("/no/such/file.xyz")
        self.assertEqual(out, "")

