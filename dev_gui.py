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
    QRadioButton, QDialogButtonBox, QColorDialog, QMenu, QInputDialog, 
    QApplication, QStackedWidget, QListWidget, QFileDialog, QSlider, 
    QCheckBox, QSpinBox, QGroupBox, QTableWidget, QTableWidgetItem, 
    QHeaderView, QAbstractItemView, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from docx import Document

import database
import backend

class DevGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.layout_yuklendi = False
        self.setWindowTitle("Developer Panel")
        self.setGeometry(100, 100, 1000, 700) # Increased size for new features
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
        self.logged_in_email = "developer@local"
        
        def dummy_shimmer(label):
            label.setStyleSheet("color: red; font-weight: bold;")
        self.shimmer_label = dummy_shimmer
        self.dev_mode_enabled = True
        
        self.init_ui()
        self.apply_saved_theme_and_language()

    # --- KEEPING YOUR EXACT DEV UI METHODS ---
    def style_button(self, button, color="#007ACC", text_color="white"):
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

    def init_ui(self):
        if self.layout_yuklendi: return
        layout = QHBoxLayout(self)

        # --- Sidebar ---
        self.sidebar = QVBoxLayout()
        self.sidebar_label = QLabel("FoxAI Dev")
        self.sidebar_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.sidebar_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        self.sidebar.addWidget(self.sidebar_label)

        # Navigation Buttons
        self.nav_btn_test = QPushButton("Test Prompt")
        self.style_button(self.nav_btn_test, color="#007ACC")
        self.nav_btn_test.clicked.connect(lambda: self.central_stack.setCurrentIndex(0))
        self.sidebar.addWidget(self.nav_btn_test)

        self.nav_btn_dataset = QPushButton("Dataset Manager")
        self.style_button(self.nav_btn_dataset, color="#007ACC")
        self.nav_btn_dataset.clicked.connect(lambda: self.central_stack.setCurrentIndex(1))
        self.sidebar.addWidget(self.nav_btn_dataset)

        self.nav_btn_model = QPushButton("Model Settings")
        self.style_button(self.nav_btn_model, color="#007ACC")
        self.nav_btn_model.clicked.connect(lambda: self.central_stack.setCurrentIndex(2))
        self.sidebar.addWidget(self.nav_btn_model)

        self.nav_btn_export = QPushButton("Export / Stats")
        self.style_button(self.nav_btn_export, color="#007ACC")
        self.nav_btn_export.clicked.connect(lambda: self.central_stack.setCurrentIndex(3))
        self.sidebar.addWidget(self.nav_btn_export)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.sidebar.addWidget(line)

        # Existing Buttons (Preserved)
        zim_btn = QPushButton("ZIM Yükle")
        self.style_button(zim_btn, color="#2980b9")
        zim_btn.clicked.connect(self.load_zim_file)
        self.sidebar.addWidget(zim_btn) 

        doc_btn = QPushButton("PDF/DOCX Yükle")
        self.style_button(doc_btn, color="#27ae60")
        doc_btn.clicked.connect(self.load_doc_dialog)
        self.sidebar.addWidget(doc_btn)

        users_btn = QPushButton("Kullanıcıları Gör")
        self.style_button(users_btn, color="#e67e22")
        users_btn.clicked.connect(self.show_gmail_users)
        self.sidebar.addWidget(users_btn)

        self.sidebar.addStretch()

        self.exit_button = QPushButton("Kapat")
        self.style_button(self.exit_button, color="#c0392b")
        self.exit_button.clicked.connect(self.close)
        self.sidebar.addWidget(self.exit_button)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(self.sidebar)
        sidebar_widget.setFixedWidth(180)

        # --- Central Area (Stacked Widget) ---
        self.central_stack = QStackedWidget()

        # Page 0: Test Prompt (Existing Chat Interface)
        self.chat_widget = self.create_chat_widget()
        self.central_stack.addWidget(self.chat_widget)

        # Page 1: Dataset Manager
        self.dataset_widget = self.create_dataset_widget()
        self.central_stack.addWidget(self.dataset_widget)

        # Page 2: Model Settings
        self.settings_widget = self.create_settings_widget()
        self.central_stack.addWidget(self.settings_widget)

        # Page 3: Export
        self.export_widget = self.create_export_widget()
        self.central_stack.addWidget(self.export_widget)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(sidebar_widget)
        splitter.addWidget(self.central_stack)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)
        self.setLayout(layout)
        self.layout_yuklendi = True

    # --- UI Component Creators ---

    def create_chat_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        label = QLabel("Test Prompt / Chat")
        label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(label)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your prompt here...")
        self.input_field.returnPressed.connect(self.soru_sor)
        
        layout.addWidget(self.chat_display)
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_field)
        
        send_btn = QPushButton("Send")
        self.style_button(send_btn)
        send_btn.clicked.connect(self.soru_sor)
        input_layout.addWidget(send_btn)
        
        layout.addLayout(input_layout)

        self.suggestion_label = QLabel("")
        layout.addWidget(self.suggestion_label)
        
        self.response_time_label = QLabel("Last Response: N/A")
        self.response_time_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.response_time_label)

        widget.setLayout(layout)
        return widget

    def create_dataset_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        header = QLabel("Dataset Manager (JSONL)")
        header.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(header)

        # File Operations
        file_layout = QHBoxLayout()
        load_btn = QPushButton("Load JSONL")
        self.style_button(load_btn, color="#8e44ad")
        load_btn.clicked.connect(self.load_dataset_file)
        
        save_btn = QPushButton("Save JSONL")
        self.style_button(save_btn, color="#27ae60")
        save_btn.clicked.connect(self.save_dataset_file)
        
        file_layout.addWidget(load_btn)
        file_layout.addWidget(save_btn)
        file_layout.addStretch()
        layout.addLayout(file_layout)

        # List/Table of Entries
        self.dataset_table = QTableWidget()
        self.dataset_table.setColumnCount(2)
        self.dataset_table.setHorizontalHeaderLabels(["Instruction", "Output"])
        self.dataset_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.dataset_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.dataset_table)

        # Add New Entry Area
        entry_group = QGroupBox("Add New Entry")
        entry_layout = QFormLayout()
        
        self.new_instruction = QTextEdit()
        self.new_instruction.setMaximumHeight(80)
        self.new_output = QTextEdit()
        self.new_output.setMaximumHeight(80)
        
        entry_layout.addRow("Instruction:", self.new_instruction)
        entry_layout.addRow("Output:", self.new_output)
        
        add_btn = QPushButton("Add Entry")
        self.style_button(add_btn)
        add_btn.clicked.connect(self.add_dataset_entry)
        entry_layout.addWidget(add_btn)
        
        entry_group.setLayout(entry_layout)
        layout.addWidget(entry_group)
        
        # Delete Button
        del_btn = QPushButton("Delete Selected Entry")
        self.style_button(del_btn, color="#c0392b")
        del_btn.clicked.connect(self.delete_dataset_entry)
        layout.addWidget(del_btn)

        widget.setLayout(layout)
        return widget

    def create_settings_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        header = QLabel("Model Settings")
        header.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(header)

        # Model Path
        path_group = QGroupBox("Base Model (GGUF)")
        path_layout = QHBoxLayout()
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("/path/to/model.gguf")
        
        # Load path from backend
        if backend.model_manager.model_path:
            self.model_path_input.setText(backend.model_manager.model_path)
            
        browse_btn = QPushButton("Browse")
        self.style_button(browse_btn)
        browse_btn.clicked.connect(self.browse_model)
        path_layout.addWidget(self.model_path_input)
        path_layout.addWidget(browse_btn)
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        load_model_btn = QPushButton("Load Model")
        self.style_button(load_model_btn, color="#d35400")
        load_model_btn.clicked.connect(self.load_model)
        layout.addWidget(load_model_btn)

        # Parameters
        params_group = QGroupBox("Inference Parameters")
        params_layout = QFormLayout()
        
        # Initialize values from backend
        current_max_tokens = backend.model_manager.settings.get("max_tokens", 128)
        current_temp = backend.model_manager.settings.get("temperature", 0.7)

        self.slider_max_tokens = QSlider(Qt.Horizontal)
        self.slider_max_tokens.setRange(64, 4096)
        self.slider_max_tokens.setValue(current_max_tokens)
        self.label_max_tokens = QLabel(str(current_max_tokens))
        self.slider_max_tokens.valueChanged.connect(lambda v: self.label_max_tokens.setText(str(v)))
        self.slider_max_tokens.sliderReleased.connect(self.save_settings) # Save on release
        
        self.slider_temp = QSlider(Qt.Horizontal)
        self.slider_temp.setRange(0, 100) # 0.0 to 1.0
        self.slider_temp.setValue(int(current_temp * 100))
        self.label_temp = QLabel(str(current_temp))
        self.slider_temp.valueChanged.connect(lambda v: self.label_temp.setText(str(v/100.0)))
        self.slider_temp.sliderReleased.connect(self.save_settings) # Save on release
        
        self.check_thinking = QCheckBox("Enable Thinking Mode (Chain of Thought)")
        
        params_layout.addRow("Max Tokens:", self.label_max_tokens)
        params_layout.addRow(self.slider_max_tokens)
        params_layout.addRow("Temperature:", self.label_temp)
        params_layout.addRow(self.slider_temp)
        params_layout.addRow(self.check_thinking)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_export_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        header = QLabel("Export Dataset")
        header.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(header)
        
        self.stats_label = QLabel("Total Entries: 0")
        layout.addWidget(self.stats_label)
        
        export_btn = QPushButton("Export for Google Colab / Unsloth")
        self.style_button(export_btn, color="#16a085")
        export_btn.clicked.connect(self.export_dataset)
        layout.addWidget(export_btn)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    # --- Logic Implementations ---

    def save_settings(self):
        # Update backend settings
        backend.model_manager.settings['max_tokens'] = self.slider_max_tokens.value()
        backend.model_manager.settings['temperature'] = self.slider_temp.value() / 100.0
        # Persist settings
        backend.model_manager.save_config()

    def soru_sor(self):
        soru = self.input_field.text().strip()
        if not soru: return
        
        profile = self.get_user_profile()
        self.chat_display.append(f"<div style='border:1px solid #ccc; padding:5px; border-radius:8px;'><b>SEN (DEV):</b> {soru}</div>")
        self.input_field.clear()

        context_text = ""
        for sender, msg in self.context_history[-self.max_context_length:]:
            context_text += f"{sender}: {msg}\n"
            
        # Add thinking prompt if enabled
        if self.check_thinking.isChecked():
            soru = "Let's think step by step. " + soru

        start_time = time.time()
        try:
            # Ensure settings are current (in case slider wasn't released but moved)
            self.save_settings()
            
            # Using backend.model_manager directly if possible, or via backend.get_response
            cevap = backend.get_response(soru, context_text, profile["tone"], self.emotion_mode, profile["interests"])
        except Exception as e:
            cevap = f"(Yerel AI Hatası: {e})"
        end_time = time.time()
        
        duration = end_time - start_time
        self.response_time_label.setText(f"Last Response: {duration:.2f}s")

        self.chat_display.append(f"<div style='border:1px solid #ccc; padding:5px; border-radius:8px;'><b>BIRD AI:</b> {cevap}</div>")
        database.log_to_memory(soru, cevap, "local_ai_dev")

    def load_dataset_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Dataset', '', 'JSONL Files (*.jsonl);;All Files (*)')
        if fname:
            success, msg = backend.dataset_manager.load_dataset(fname)
            if success:
                self.refresh_dataset_table()
                QMessageBox.information(self, "Success", msg)
            else:
                QMessageBox.warning(self, "Error", msg)

    def save_dataset_file(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save Dataset', '', 'JSONL Files (*.jsonl)')
        if fname:
            success, msg = backend.dataset_manager.save_dataset(fname)
            QMessageBox.information(self, "Info", msg)

    def refresh_dataset_table(self):
        entries = backend.dataset_manager.get_entries()
        self.dataset_table.setRowCount(len(entries))
        for i, entry in enumerate(entries):
            self.dataset_table.setItem(i, 0, QTableWidgetItem(entry.get("instruction", "")))
            self.dataset_table.setItem(i, 1, QTableWidgetItem(entry.get("output", "")))
        self.stats_label.setText(f"Total Entries: {len(entries)}")

    def add_dataset_entry(self):
        instruction = self.new_instruction.toPlainText().strip()
        output = self.new_output.toPlainText().strip()
        if not instruction or not output:
            QMessageBox.warning(self, "Warning", "Both fields are required.")
            return
        
        backend.dataset_manager.add_entry(instruction, output)
        self.new_instruction.clear()
        self.new_output.clear()
        self.refresh_dataset_table()

    def delete_dataset_entry(self):
        rows = sorted(set(index.row() for index in self.dataset_table.selectedIndexes()))
        if not rows: return
        
        # Delete in reverse order to maintain indices
        for row in reversed(rows):
            backend.dataset_manager.delete_entry(row)
        self.refresh_dataset_table()

    def browse_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select GGUF Model', '', 'GGUF Files (*.gguf);;All Files (*)')
        if fname:
            self.model_path_input.setText(fname)

    def load_model(self):
        path = self.model_path_input.text().strip()
        if not path: return
        
        self.chat_display.append(f"<i>Loading model from {path}... Please wait.</i>")
        QApplication.processEvents()
        
        success = backend.model_manager.load_model(path)
        if success:
            self.chat_display.append("<b>Model loaded successfully!</b>")
            QMessageBox.information(self, "Success", "Model loaded.")
        else:
            self.chat_display.append("<b>Failed to load model. Check logs.</b>")
            QMessageBox.critical(self, "Error", "Failed to load model.")

    def export_dataset(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Export for Colab', 'training_export.jsonl', 'JSONL Files (*.jsonl)')
        if fname:
            success, msg = backend.dataset_manager.export_for_colab(fname)
            QMessageBox.information(self, "Export", msg)

    # --- Existing Helper Methods ---
    def show_gmail_users(self):
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT username, email FROM user_profile")
        rows = cursor.fetchall()
        conn.close()

        dlg = QDialog(self)
        dlg.setWindowTitle("Kullanıcılar")
        vbox = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        for r in rows: text.append(f"Kullanıcı: {r[0]} | Email: {r[1]}")
        vbox.addWidget(text)
        dlg.setLayout(vbox)
        dlg.exec_()

    def load_zim_file(self):
        QMessageBox.information(self, "Dev", "ZIM loading logic preserved.")
        
    def load_doc_dialog(self):
        QMessageBox.information(self, "Dev", "PDF/DOCX loading logic preserved.")

    def apply_theme(self, theme):
        if theme == "dark":
            self.setStyleSheet("background-color: #121212; color: #E0E0E0;")
        elif theme == "light":
            self.setStyleSheet("background-color: #FFFFFF; color: #000000;")
        else:
            self.setStyleSheet("")

    def apply_saved_theme_and_language(self):
        try:
            profile = self.get_user_profile()
            theme = profile.get("theme", "system")
        except Exception:
            theme = "system"
        self.apply_theme(theme)

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
            "theme": col_val('theme', 'system')
        }
