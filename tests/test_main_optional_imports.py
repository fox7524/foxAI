import builtins
import importlib.util
import os
import types
import unittest
from unittest import mock


class TestMainOptionalImports(unittest.TestCase):
    def _load_isolated_main(self, block_modules: set[str]) -> types.ModuleType:
        here = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(here, os.pardir))
        main_path = os.path.join(repo_root, "main.py")
        spec = importlib.util.spec_from_file_location("main_isolated_optional_imports", main_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)

        orig_import = builtins.__import__

        def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
            top = (name or "").split(".", 1)[0]
            if top in block_modules:
                raise ImportError(f"blocked import for test: {name}")
            return orig_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=hooked_import):
            spec.loader.exec_module(mod)
        return mod

    def test_import_main_without_mlx_lm_psutil_libzim_file_ingest(self):
        try:
            import PyQt5  # noqa: F401
        except Exception:
            self.skipTest("PyQt5 not available")

        mod = self._load_isolated_main({"mlx_lm", "psutil", "libzim", "file_ingest"})
        self.assertFalse(getattr(mod, "HAS_MLX_LM", True))
        self.assertTrue(getattr(mod, "MLX_IMPORT_ERROR", ""))
        self.assertFalse(getattr(mod, "HAS_PSUTIL", True))
        self.assertTrue(getattr(mod, "PSUTIL_IMPORT_ERROR", ""))
        self.assertFalse(getattr(mod, "HAS_LIBZIM", True))
        self.assertTrue(getattr(mod, "LIBZIM_IMPORT_ERROR", ""))
        self.assertTrue(getattr(mod, "INGEST_IMPORT_ERROR", ""))
