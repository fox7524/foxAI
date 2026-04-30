import os
import sys
import tempfile
import types
import unittest

import rag_engine


class TestZimLibzimFallback(unittest.TestCase):
    def test_libzim_entries_attr_non_iterable_falls_back_to_id_scan(self):
        class _Entry:
            def __init__(self, title: str, mimetype: str, content: bytes):
                self.title = title
                self.mimetype = mimetype
                self._content = content

            def read(self):
                return self._content

        class _Archive:
            def __init__(self, path: str):
                self._entries = [
                    _Entry("home", "text/html", b"hello"),
                    _Entry("about", "text/plain", b"world"),
                ]
                self.entries = object()

            def entry_count(self):
                return len(self._entries)

            def get_entry_by_id(self, i: int):
                if 0 <= i < len(self._entries):
                    return self._entries[i]
                return None

        libzim_mod = types.ModuleType("libzim")
        reader_mod = types.ModuleType("libzim.reader")
        reader_mod.Archive = _Archive
        libzim_mod.reader = reader_mod

        prev_libzim = sys.modules.get("libzim")
        prev_libzim_reader = sys.modules.get("libzim.reader")
        sys.modules["libzim"] = libzim_mod
        sys.modules["libzim.reader"] = reader_mod
        try:
            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "x.zim")
                with open(p, "wb") as f:
                    f.write(b"not a real zim")
                eng = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
                eng.enabled = True
                eng.last_error = ""
                eng._set_last_error = rag_engine.RAGEngine._set_last_error.__get__(eng, rag_engine.RAGEngine)
                out = rag_engine.RAGEngine.extract_from_zim(eng, p)
                self.assertIn("hello", out)
                self.assertIn("world", out)
        finally:
            if prev_libzim is None:
                sys.modules.pop("libzim", None)
            else:
                sys.modules["libzim"] = prev_libzim
            if prev_libzim_reader is None:
                sys.modules.pop("libzim.reader", None)
            else:
                sys.modules["libzim.reader"] = prev_libzim_reader

    def test_libzim_iterable_but_empty_falls_back_to_id_scan(self):
        class _Entry:
            def __init__(self, title: str, mimetype: str, content: bytes):
                self.title = title
                self.mimetype = mimetype
                self._content = content

            def read(self):
                return self._content

        class _Archive:
            def __init__(self, path: str):
                self._entries = [
                    _Entry("home", "text/html", b"hello"),
                    _Entry("about", "text/plain", b"world"),
                ]

            def iterByPath(self):
                if False:
                    yield None
                return iter(())

            def entry_count(self):
                return len(self._entries)

            def get_entry_by_id(self, i: int):
                if 0 <= i < len(self._entries):
                    return self._entries[i]
                return None

        libzim_mod = types.ModuleType("libzim")
        reader_mod = types.ModuleType("libzim.reader")
        reader_mod.Archive = _Archive
        libzim_mod.reader = reader_mod

        prev_libzim = sys.modules.get("libzim")
        prev_libzim_reader = sys.modules.get("libzim.reader")
        sys.modules["libzim"] = libzim_mod
        sys.modules["libzim.reader"] = reader_mod
        try:
            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "x.zim")
                with open(p, "wb") as f:
                    f.write(b"not a real zim")
                eng = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
                eng.enabled = True
                eng.last_error = ""
                eng._set_last_error = rag_engine.RAGEngine._set_last_error.__get__(eng, rag_engine.RAGEngine)
                out = rag_engine.RAGEngine.extract_from_zim(eng, p)
                self.assertIn("hello", out)
                self.assertIn("world", out)
        finally:
            if prev_libzim is None:
                sys.modules.pop("libzim", None)
            else:
                sys.modules["libzim"] = prev_libzim
            if prev_libzim_reader is None:
                sys.modules.pop("libzim.reader", None)
            else:
                sys.modules["libzim.reader"] = prev_libzim_reader

    def test_libzim_random_entry_fallback(self):
        class _Entry:
            def __init__(self, title: str, mimetype: str, content: bytes, path: str):
                self.title = title
                self.mimetype = mimetype
                self._content = content
                self.path = path

            def read(self):
                return self._content

        class _Archive:
            def __init__(self, path: str):
                self._entry = _Entry("home", "text/html", b"hello", "home")

            def iterByPath(self):
                return iter(())

            def entry_count(self):
                return 0

            def get_entry_by_id(self, i: int):
                return None

            def get_random_entry(self):
                return self._entry

        libzim_mod = types.ModuleType("libzim")
        reader_mod = types.ModuleType("libzim.reader")
        reader_mod.Archive = _Archive
        libzim_mod.reader = reader_mod

        prev_libzim = sys.modules.get("libzim")
        prev_libzim_reader = sys.modules.get("libzim.reader")
        sys.modules["libzim"] = libzim_mod
        sys.modules["libzim.reader"] = reader_mod
        try:
            with tempfile.TemporaryDirectory() as td:
                p = os.path.join(td, "x.zim")
                with open(p, "wb") as f:
                    f.write(b"not a real zim")
                eng = rag_engine.RAGEngine.__new__(rag_engine.RAGEngine)
                eng.enabled = True
                eng.last_error = ""
                eng._set_last_error = rag_engine.RAGEngine._set_last_error.__get__(eng, rag_engine.RAGEngine)
                out = rag_engine.RAGEngine.extract_from_zim(eng, p)
                self.assertIn("hello", out)
        finally:
            if prev_libzim is None:
                sys.modules.pop("libzim", None)
            else:
                sys.modules["libzim"] = prev_libzim
            if prev_libzim_reader is None:
                sys.modules.pop("libzim.reader", None)
            else:
                sys.modules["libzim.reader"] = prev_libzim_reader
