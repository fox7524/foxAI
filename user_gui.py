import sys
import os
import time
import logging
import sqlite3
import uuid
import subprocess
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QHBoxLayout, 
    QLabel, QSplitter, QDialog, QFormLayout, QComboBox, QMessageBox, 
    QRadioButton, QDialogButtonBox, QButtonGroup, QStackedLayout, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect
from docx import Document

import database
import backend

logger = logging.getLogger('thunderbird')

class EmbeddingWorker(QThread):
    finished = pyqtSignal(str)
    def __init__(self, parent, file_path, file_type):
        super().__init__(parent)
        self.file_path = file_path
        self.file_type = file_type
        self.parent = parent

    def run(self):
        try:
            self.finished.emit("success")
        except Exception as e:
            self.finished.emit(f"error: {str(e)}")


class ChatbotGUI(QWidget):
    def style_button(self, button, color="#007ACC", text_color=None):
        try: cls_name = button.__class__.__name__
        except Exception: cls_name = ""

        if cls_name == 'QRadioButton':
            if text_color is None:
                text_color = "#FFFFFF" if getattr(self, 'current_theme', 'system') == 'dark' else "#000000"
            button.setStyleSheet(f"QRadioButton {{ color: {text_color}; background-color: transparent; font-weight: bold; }}")
            return

        if text_color is None: text_color = "white"
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: {text_color};
                border: 2px solid #005F8C;
                border-radius: 6px;
                padding: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2892D7;
            }}
        """)

    def __init__(self):
        self.layout_yuklendi = False
        super().__init__()
        self.setWindowTitle("Thunderbird AI Volume Alpha")
        self.setGeometry(100, 100, 800, 500)
        
        self.context_history = []
        self.max_context_length = 10
        self.conversations = []
        self.max_conversations = 5
        self.current_conversation_id = None
        self.settings_mode = False
        self.current_theme = "system"
        self.memory_mode = "aktif" 
        self.emotion_mode = "neutral" 
        self.model_version = "BIRD AI 1.5 (Local)"
        self.logged_in_email = None
        
        def dummy_shimmer(label):
            label.setStyleSheet("color: red; font-weight: bold;")
        self.shimmer_label = dummy_shimmer
        self.dev_mode_enabled = False
        
        self.ensure_login()
        self.init_ui()
        self.apply_saved_theme_and_language()

    def ensure_login(self):
        while not self.logged_in_email:
            dialog = QDialog(self)
            dialog.setWindowTitle("Login")
            layout = QVBoxLayout(dialog)

            selection_layout = QHBoxLayout()
            login_select_btn = QPushButton("Log In")
            register_select_btn = QPushButton("Sıgn Up")
            self.style_button(login_select_btn)
            self.style_button(register_select_btn)
            selection_layout.addWidget(login_select_btn)
            selection_layout.addWidget(register_select_btn)
            
            exit_btn = QPushButton("Exit")
            self.style_button(exit_btn)
            exit_btn.clicked.connect(lambda: sys.exit())
            selection_layout.addWidget(exit_btn)
            layout.addLayout(selection_layout)

            dev_mode_button = QPushButton("Dev Mode")
            self.style_button(dev_mode_button, color="red")
            
            def handle_dev_login():
                self.dev_mode_enabled = True
                self.logged_in_email = "developer@local"
                dialog.accept()
                
            dev_mode_button.clicked.connect(handle_dev_login)
            layout.addWidget(dev_mode_button)

            login_widget = QWidget()
            login_form = QVBoxLayout(login_widget)
            login_email_input = QLineEdit()
            login_email_input.setPlaceholderText("Email")
            login_password_input = QLineEdit()
            login_password_input.setEchoMode(QLineEdit.Password)
            
            login_form.addWidget(QLabel("Email:"))
            login_form.addWidget(login_email_input)
            login_form.addWidget(QLabel("Şifre:"))
            login_form.addWidget(login_password_input)
            
            login_btn = QPushButton("Giriş Yap")
            self.style_button(login_btn)
            login_form.addWidget(login_btn)
            layout.addWidget(login_widget)

            def attempt_login():
                self.logged_in_email = login_email_input.text().strip()
                dialog.accept()

            login_btn.clicked.connect(attempt_login)
            dialog.setLayout(layout)
            dialog.exec_()

    def init_ui(self):
        if self.layout_yuklendi: return
        layout = QHBoxLayout(self)

        self.sidebar = QVBoxLayout()
        self.sidebar_label = QLabel("BIRD AI")
        self.sidebar_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        
        header_layout = QHBoxLayout()
        header_layout.addWidget(self.sidebar_label)

        self.settings_button = QPushButton("⚙")
        self.settings_button.setFixedSize(24, 24)
        self.style_button(self.settings_button, color="#555", text_color="white")
        self.settings_button.clicked.connect(self.open_settings_panel)
        header_layout.addWidget(self.settings_button)

        self.newchat_button = QPushButton("＋")
        self.newchat_button.setFixedSize(24, 24)
        self.style_button(self.newchat_button, color="#2ecc71", text_color="white")
        self.newchat_button.clicked.connect(self.new_conversation)
        header_layout.addWidget(self.newchat_button)

        self.sidebar.addLayout(header_layout)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(self.sidebar)
        sidebar_widget.setFixedWidth(150)

        self.chat_layout = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.soru_sor)
        
        self.chat_layout.addWidget(self.chat_display)
        self.chat_layout.addWidget(self.input_field)

        self.suggestion_label = QLabel("")
        self.chat_layout.addWidget(self.suggestion_label)

        chat_widget = QWidget()
        chat_widget.setLayout(self.chat_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(sidebar_widget)
        splitter.addWidget(chat_widget)
        layout.addWidget(splitter)

        self.hamburger_button = QPushButton("☰")
        self.style_button(self.hamburger_button, color="#333", text_color="white")
        def toggle_sidebar():
            sidebar_widget.setVisible(not sidebar_widget.isVisible())
        self.hamburger_button.clicked.connect(toggle_sidebar)
        layout.addWidget(self.hamburger_button)

        self.setLayout(layout)
        self.layout_yuklendi = True

    def soru_sor(self):
        soru = self.input_field.text().strip()
        if not soru: return
            
        profile = self.get_user_profile()
        self.chat_display.append(f"<div style='border:1px solid #ccc; padding:5px; border-radius:8px;'><b>SEN:</b> {soru}</div>")
        self.input_field.clear()

        context_text = ""
        for sender, msg in self.context_history[-self.max_context_length:]:
            context_text += f"{sender}: {msg}\n"

        # CALLING YOUR LOCAL BACKEND ONLY
        try:
            cevap = backend.get_response(
                user_input=soru, 
                context=context_text, 
                tone=profile["tone"], 
                emotion=self.emotion_mode,
                interest=profile["interests"]
            )
        except Exception as e:
            logger.error(f"Yerel Backend Hatası: {e}")
            cevap = f"(Yerel AI Hatası: {e})"

        if not cevap:
            cevap = "(Kendi modelimden anlamlı bir cevap alınamadı.)"

        self.chat_display.append(f"<div style='border:1px solid #ccc; padding:5px; border-radius:8px;'><b>BIRD AI:</b> {cevap}</div>")

        if self.memory_mode == "aktif":
            self.context_history.append(("Kullanıcı", soru))
            self.context_history.append(("BIRD AI", cevap))
            
        database.log_to_memory(soru, cevap, "local_ai")

    def new_conversation(self):
        title = f"Sohbet {len(self.conversations)+1}"
        self.conversations.append([])
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("INSERT INTO conversations (title, created_at) VALUES (?, ?)", (title, now))
        self.current_conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()

    def open_settings_panel(self):
        splash = QDialog(self, Qt.FramelessWindowHint | Qt.Dialog)
        splash.setAttribute(Qt.WA_TranslucentBackground)
        splash.setModal(True)
        splash.resize(520, 620)
        splash.exec_()

    def get_user_profile(self, email=None):
        if email is None: email = self.logged_in_email
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_profile WHERE email = ?", (email,))
        result = cursor.fetchone()
        cols = [c[0] for c in cursor.description] if cursor.description else []
        conn.close()
        
        def col_val(name, default=""):
            if name in cols:
                try: return result[cols.index(name)]
                except Exception: return default
            return default

        return {
            "tone": col_val('tone', 'formal'),
            "interests": col_val('interests', ''),
            "theme": col_val('theme', 'system'),
            "language": col_val('language', 'tr'),
            "first_name": col_val('first_name', ''),
            "last_name": col_val('last_name', ''),
            "age": col_val('age', 0),
            "email": col_val('email', '')
        }

    def apply_saved_theme_and_language(self):
        profile = self.get_user_profile()
        theme = profile.get("theme", "system")
        if theme == "dark":
            self.setStyleSheet("background-color: #121212; color: #E0E0E0;")
        elif theme == "light":
            self.setStyleSheet("background-color: #FFFFFF; color: #000000;")
        else:
            self.setStyleSheet("")   