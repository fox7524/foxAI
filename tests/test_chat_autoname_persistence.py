import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

import main


class TestChatAutoNameAndPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "app.db")
        self.gui = main.ChatbotGUI(None, None, "", db_path=self.db_path, start_service=False, start_monitor=False)

    def tearDown(self):
        try:
            self.gui.close()
        except Exception:
            pass
        try:
            self.gui.deleteLater()
        except Exception:
            pass
        try:
            QApplication.processEvents()
        except Exception:
            pass
        self._tmp.cleanup()

    def test_auto_name_then_persist_then_reload(self):
        self.gui.new_chat()
        old_name = self.gui.active_chat
        self.assertTrue(old_name.startswith("New Chat"))

        first_msg = "Hello world from test suite"
        self.gui._auto_name_active_chat(first_msg)
        new_name = self.gui.active_chat
        self.assertNotEqual(old_name, new_name)
        self.assertIn(new_name, self.gui.chats)

        self.gui.chats[new_name].append({"role": "user", "content": first_msg})
        self.gui.chat_ui[new_name].append({"role": "user", "content": first_msg})
        self.gui._persist_message(new_name, "user", first_msg)

        gui2 = main.ChatbotGUI(None, None, "", db_path=self.db_path, start_service=False, start_monitor=False)
        try:
            self.assertIn(new_name, gui2.chat_ui)
            texts = [m.get("content", "") for m in gui2.chat_ui[new_name] if m.get("role") == "user"]
            self.assertIn(first_msg, texts)
        finally:
            try:
                gui2.close()
            except Exception:
                pass
            try:
                gui2.deleteLater()
            except Exception:
                pass
            try:
                QApplication.processEvents()
            except Exception:
                pass

    def test_pending_chat_updates_on_rename(self):
        self.gui.new_chat()
        old_name = self.gui.active_chat
        self.gui.chat_ui[old_name].append(
            {"role": "assistant", "answer": "", "think": "", "think_open": False, "thought_s": None, "meta": None}
        )
        self.gui._pending_chat = old_name
        self.gui._pending_msg_index = len(self.gui.chat_ui[old_name]) - 1

        self.gui._rename_chat(old_name, "Renamed Chat", render_after=False)
        self.assertEqual(self.gui._pending_chat, "Renamed Chat")

    def test_delete_clears_pending_state(self):
        self.gui.new_chat()
        chat_name = self.gui.active_chat
        self.gui._pending_chat = chat_name
        self.gui._pending_msg_index = 0

        self.gui._on_chat_deleted(chat_name, True, "", 0.0)
        self.assertIsNone(self.gui._pending_chat)
        self.assertIsNone(self.gui._pending_msg_index)

