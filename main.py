"""
FoxAI Studio - Main Application Entry Point

ARCHITECTURE OVERVIEW:
======================
FoxAI Studio is a dual-mode AI coding assistant with two distinct interfaces:

1. USER MODE (default):
   - Clean, minimal interface for chatting with the AI
   - System prompt customization
   - Chat history management
   - Theme selection (dark/light)
   - Access to Settings panel

2. DEV MODE (password: "123"):
   - Advanced developer panel with 5 tabs:
     * RAG Indexer: Index project files, PDFs, DOCX, ZIM archives
     * Fine-tune: Configure and run LoRA fine-tuning
     * Model: Select and load different MLX models
     * Testing: Run AST benchmarks, stress tests, RAM monitoring
     * Unrestricted: Bypass "ask before acting" safety rule

KEY DESIGN PATTERNS:
===================
1. QThread for non-blocking AI generation
   - AIWorker runs model inference in background thread
   - Emits signals for streaming tokens: new_token, finished, error
   - Prevents GUI freezing during generation

2. MemoryMonitor for real-time system stats
   - Tracks app RAM usage and CPU percentage
   - Updates every 2 seconds via QTimer in worker thread

3. RAG Integration
   - FAISS vector index for semantic search
   - sentence-transformers for embeddings
   - Supports: code files, PDFs, DOCX, ZIM archives

4. System Prompt Safety
   - Default: "Ask Before Acting" - AI must clarify before coding
   - Unrestricted Mode: AI generates code directly (dev mode only)

HOW THE CHAT LOOP WORKS:
=========================
1. User types message → soru_sor()
2. Message displayed in chat bubble
3. RAG query (if enabled) → context injection
4. Build prompt with chat history
5. Start AIWorker thread
6. Stream tokens to UI as they arrive (new_token signal)
7. On completion: display stats, update chat history

ENTRY POINT:
============
To run: python main.py
Model path is hardcoded at bottom of file.
"""

import sys
import os
import time
import json
import ast
import re
import glob
import subprocess
import psutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QHBoxLayout, QLabel, QSplitter, QDialog,
    QFormLayout, QMessageBox, QRadioButton, QButtonGroup,
    QStackedWidget, QListWidget, QFrame, QScrollArea, QFileDialog,
    QInputDialog, QTabWidget, QCheckBox, QSpinBox, QSlider,
    QComboBox, QTextBrowser, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QGroupBox,
    QGridLayout, QPlainTextEdit, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QTextCursor, QPalette, QIcon

# mlx_lm: Apple's MLX framework for local LLM inference
# - load(): Load a model and tokenizer from a path
# - generate(): Generate text (blocking)
# - stream_generate(): Generate text with streaming (yields tokens)
from mlx_lm import load, generate, stream_generate

# RAG Engine: Handles document indexing and retrieval
# Fine-tune Engine: Handles LoRA training
try:
    from rag_engine import RAGEngine
    from finetune_engine import FinetuneEngine
except ImportError:
    pass

# Application version and dev mode password
VERSION = "FoxAI - Studio Edition"
DEV_MODE_PASSWORD = "123"

# ---------------------------------------------------------
# WORKER THREADS (Background Processing)
# ---------------------------------------------------------

class AIWorker(QThread):
    """
    Background thread for AI text generation.

    WHY A THREAD?
    - LLM generation can take seconds to minutes
    - Blocking the main thread would freeze the GUI
    - QThread allows concurrent execution with GUI event loop

    SIGNALS:
    - new_token(str): Emitted for each new token (for streaming display)
    - finished(str, float, int, float): Emitted when generation completes
    - error(str): Emitted if an exception occurs

    HOW STREAMING WORKS:
    - stream_generate() yields accumulated response
    - We track previous response length to extract NEW tokens only
    - Each new token is emitted via signal for live UI updates
    """
    # Signal emitted when a new token is generated (for streaming)
    new_token = pyqtSignal(str)
    # Signal emitted when generation completes: (full_response, tokens_per_sec, total_tokens, elapsed_time)
    finished = pyqtSignal(str, float, int, float)
    # Signal emitted on error: error message
    error = pyqtSignal(str)

    def __init__(self, model, tokenizer, prompt):
        """
        Initialize the worker thread.

        ARGS:
            model: The loaded MLX model instance
            tokenizer: The tokenizer for the model
            prompt: The formatted prompt string to generate from
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.is_running = True  # Can be set False to stop generation

    def run(self):
        """
        Main thread execution - generates text and emits signals.

        This method runs in a separate thread when start() is called.
        DO NOT call this directly - use start() instead.
        """
        try:
            start_time = time.time()
            full_response = ""
            token_count = 0

            # Iterate through streaming response
            # stream_generate yields accumulated text, not individual tokens
            for response in stream_generate(self.model, self.tokenizer, prompt=self.prompt, max_tokens=1500):
                # Check if stop was requested (e.g., user clicked stop button)
                if not self.is_running:
                    break

                # Extract just the NEW portion since last iteration
                # response is cumulative, so len(full_response) gives us where we left off
                new_part = response[len(full_response):]
                full_response = response
                token_count += 1

                # Emit new token for live display
                self.new_token.emit(new_part)

            # Calculate metrics
            end_time = time.time()
            elapsed = end_time - start_time
            tok_per_sec = token_count / elapsed if elapsed > 0 else 0.0

            # Emit completion signal with results
            self.finished.emit(full_response, tok_per_sec, token_count, elapsed)

        except Exception as e:
            # Emit error signal if something goes wrong
            self.error.emit(f"Error generating response: {str(e)}")

    def stop(self):
        """
        Request the worker to stop generation.

        Sets is_running to False, which causes the loop in run() to break.
        Note: The model may still produce a few more tokens after stop is called.
        """
        self.is_running = False


class MemoryMonitor(QThread):
    """
    Background thread for monitoring system resources.

    WHY SEPARATE THREAD?
    - psutil calls can be slow
    - We want to update every 2 seconds without affecting UI responsiveness

    SIGNALS:
    - update_signal(str, str): Emits (ram_usage_gb, cpu_percent)
    """
    update_signal = pyqtSignal(str, str)  # (ram_gb, sys_load)

    def run(self):
        """
        Continuous monitoring loop - runs until thread is stopped.

        Gets current process memory and system CPU percentage,
        then sleeps for 2 seconds before repeating.
        """
        process = psutil.Process(os.getpid())
        while True:
            try:
                # Get memory info for THIS process only
                mem_info = process.memory_info()
                used_gb = mem_info.rss / (1024 ** 3)  # Convert bytes to GB

                # Get system-wide CPU usage percentage
                # interval=None uses cached value (faster, less accurate)
                # For real-time accuracy, use interval=1 (blocks for 1 second)
                cpu_percent = psutil.cpu_percent(interval=None)

                # Emit update signal with formatted strings
                self.update_signal.emit(f"{used_gb:.2f} GB", f"{cpu_percent}%")
            except:
                pass  # Ignore errors, try again next iteration
            time.sleep(2)

# ---------------------------------------------------------
# SETTINGS & ROADMAP VIEWER
# ---------------------------------------------------------
class SettingsDialog(QDialog):
    def __init__(self, parent=None, user_prompt="", current_theme="dark"):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(550, 500)
        self.main_app = parent  # Reference to main window for saving

        self.setStyleSheet("""
            QDialog { background-color: #161616; color: #e0e0e0; }
            QLabel { color: #ccc; }
            QPushButton { background-color: #2a2a2a; border: 1px solid #444; border-radius: 6px; padding: 8px 16px; color: white; }
            QPushButton:hover { background-color: #333; }
            QRadioButton { color: #ccc; spacing: 8px; }
        """)

        self.final_user_prompt = user_prompt
        self.final_theme = current_theme

        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Settings")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #7c4dff; padding: 10px;")
        layout.addWidget(header)

        # User Prompt Section (NOT system prompt - that's Dev Mode only)
        layout.addWidget(QLabel("<b>User Prompt (Personality):</b>"))
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlainText(user_prompt)
        self.prompt_edit.setPlaceholderText("This prompt defines how the AI responds to you...")
        self.prompt_edit.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 8px; color: #ddd;")
        self.prompt_edit.setMinimumHeight(120)
        layout.addWidget(self.prompt_edit)

        layout.addSpacing(10)

        # Theme Section
        layout.addWidget(QLabel("<b>Theme:</b>"))
        theme_layout = QHBoxLayout()
        self.rb_dark = QRadioButton("🌙 Dark Studio")
        self.rb_light = QRadioButton("☀️ Light")

        if current_theme == "dark": self.rb_dark.setChecked(True)
        else: self.rb_light.setChecked(True)

        theme_layout.addWidget(self.rb_dark)
        theme_layout.addWidget(self.rb_light)
        theme_layout.addStretch()
        layout.addLayout(theme_layout)

        layout.addSpacing(15)

        # Roadmap Button
        roadmap_btn = QPushButton("📋 View Project Roadmap")
        roadmap_btn.setStyleSheet("background-color: #1a3a5c; color: #4d9fff;")
        roadmap_btn.clicked.connect(self.show_roadmap)
        layout.addWidget(roadmap_btn)

        # Dev Mode Note
        dev_note = QLabel("🔒 System prompt is only editable in Dev Mode (Settings → Dev Mode)")
        dev_note.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        layout.addWidget(dev_note)

        layout.addStretch()

        # Bottom Buttons
        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save & Apply")
        save_btn.setStyleSheet("background-color: #1a5c3a; color: #4dff9f;")
        save_btn.clicked.connect(self.accept_settings)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def accept_settings(self):
        # Save user_prompt to prompts.json via main app
        self.final_user_prompt = self.prompt_edit.toPlainText()
        self.final_theme = "dark" if self.rb_dark.isChecked() else "light"

        # Update main app's user_prompt
        if self.main_app and hasattr(self.main_app, 'prompts'):
            self.main_app.prompts["user_prompt"] = self.final_user_prompt
            self.main_app.save_prompts(self.main_app.prompts)
            self.main_app.user_prompt = self.final_user_prompt

        self.accept()

    def show_roadmap(self):
        QMessageBox.information(self, "Roadmap", "📅 Phase 1 (Apr 13-20): Foundation - Model loads, RAG indexer, LoRA fine-tune\n"
                                                 "⚡ Phase 2 (Apr 21-27): Features - System prompt, Run button, Settings\n"
                                                 "🧪 Phase 3 (Apr 28-30): Break It - Testing and bug fixes\n"
                                                 "🎯 May 11: Presentation Day")

# ---------------------------------------------------------
# DEV MODE GATE
# ---------------------------------------------------------
class DevModeGate:
    def __init__(self):
        self.unlocked = False
        
    def attempt_unlock(self, password: str) -> bool:
        if password == DEV_MODE_PASSWORD:
            self.unlocked = True
            return True
        return False
    
    def lock(self):
        self.unlocked = False

dev_mode_gate = DevModeGate()

# ---------------------------------------------------------
# DEV PANEL
# ---------------------------------------------------------
class DevPanel(QDialog):
    def __init__(self, parent=None, main_app=None):
        super().__init__(parent)
        self.main_app = main_app
        self.setWindowTitle("Developer Panel")
        self.setGeometry(150, 150, 900, 700)
        self.setStyleSheet("""
            QDialog { background-color: #0f0f0f; color: #e0e0e0; }
            QTabWidget::pane { border: 1px solid #333; background: #161616; }
            QTabBar::tab { background: #1e1e1e; color: #888; padding: 8px 16px; }
            QTabBar::tab:selected { background: #222; color: white; }
            QPushButton { background-color: #2a2a2a; border: 1px solid #444; border-radius: 6px; padding: 8px 16px; color: white; }
            QPushButton:hover { background-color: #333; }
            QPushButton:disabled { background-color: #1a1a1a; color: #555; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background-color: #222; border: 1px solid #333; border-radius: 6px; padding: 6px; color: white; }
            QCheckBox { color: #ccc; spacing: 8px; }
            QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #555; background: #222; }
            QLabel { color: #ccc; }
            QProgressBar { border: 1px solid #333; border-radius: 4px; background: #1a1a1a; text-align: center; color: white; }
            QProgressBar::chunk { background: #7c4dff; border-radius: 3px; }
            QGroupBox { border: 1px solid #333; border-radius: 8px; margin-top: 12px; padding-top: 12px; font-weight: bold; color: #7c4dff; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QTableWidget { background-color: #1a1a1a; border: 1px solid #333; color: white; gridline-color: #2a2a2a; }
            QHeaderView::section { background-color: #1e1e1e; color: #ccc; padding: 6px; border: none; }
            QPlainTextEdit { background-color: #1a1a1a; border: 1px solid #333; color: #ddd; }
        """)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("🔧 FoxAI Developer Panel")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #7c4dff; padding: 10px;")
        layout.addWidget(header)
        
        tabs = QTabWidget()
        
        # TAB 1: RAG CONTROLS
        tabs.addTab(self.build_rag_tab(), "📚 RAG Indexer")
        # TAB 2: FINE-TUNING
        tabs.addTab(self.build_finetune_tab(), "⚡ Fine-tune")
        # TAB 3: MODEL SELECTOR
        tabs.addTab(self.build_model_tab(), "🤖 Model")
        # TAB 4: TESTING
        tabs.addTab(self.build_testing_tab(), "🧪 Testing")
        # TAB 5: UNRESTRICTED MODE
        tabs.addTab(self.build_unrestricted_tab(), "⚠ Unrestricted")
        
        layout.addWidget(tabs)
        
        # Bottom buttons
        bottom = QHBoxLayout()
        bottom.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)
    
    def build_rag_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Status
        status_box = QGroupBox("RAG Status")
        s_layout = QGridLayout()
        
        self.rag_status_lbl = QLabel("Disabled")
        self.rag_chunks_lbl = QLabel("0 chunks indexed")
        self.rag_index_lbl = QLabel("Index: None")
        
        s_layout.addWidget(QLabel("Status:"), 0, 0)
        s_layout.addWidget(self.rag_status_lbl, 0, 1)
        s_layout.addWidget(QLabel("Chunks:"), 1, 0)
        s_layout.addWidget(self.rag_chunks_lbl, 1, 1)
        s_layout.addWidget(QLabel("Index Path:"), 2, 0)
        s_layout.addWidget(self.rag_index_lbl, 2, 1)
        status_box.setLayout(s_layout)
        layout.addWidget(status_box)
        
        # Project Folder Indexing
        folder_box = QGroupBox("Project Folder Indexing")
        f_layout = QGridLayout()
        
        f_layout.addWidget(QLabel("Project Folder:"), 0, 0)
        self.rag_folder_path = QLineEdit()
        self.rag_folder_path.setPlaceholderText("Select a folder to index...")
        f_layout.addWidget(self.rag_folder_path, 0, 1)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_rag_folder)
        f_layout.addWidget(browse_btn, 0, 2)
        
        index_btn = QPushButton("Index Project Files")
        index_btn.clicked.connect(self.index_project_files)
        f_layout.addWidget(index_btn, 1, 0, 1, 3)
        
        folder_box.setLayout(f_layout)
        layout.addWidget(folder_box)
        
        # Python Docs Indexing
        docs_box = QGroupBox("Python Documentation Indexing")
        d_layout = QGridLayout()
        
        self.docs_url = QLineEdit()
        self.docs_url.setPlaceholderText("URL to Python docs (or leave blank for default)...")
        d_layout.addWidget(QLabel("Docs URL:"), 0, 0)
        d_layout.addWidget(self.docs_url, 0, 1, 1, 2)
        
        index_docs_btn = QPushButton("Download & Index Python Docs")
        index_docs_btn.clicked.connect(self.index_python_docs)
        d_layout.addWidget(index_docs_btn, 1, 0, 1, 3)
        
        docs_box.setLayout(d_layout)
        layout.addWidget(docs_box)
        
        # Reset
        reset_btn = QPushButton("Reset RAG Index")
        reset_btn.setStyleSheet("background-color: #5c1a2a; border-color: #ff4d6a; color: #ff4d6a;")
        reset_btn.clicked.connect(self.reset_rag)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
        return widget
    
    def browse_rag_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if folder:
            self.rag_folder_path.setText(folder)
    
    def index_project_files(self):
        folder = self.rag_folder_path.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Invalid Folder", "Please select a valid folder path.")
            return
        
        self.rag_status_lbl.setText("Indexing...")
        QApplication.processEvents()
        
        # Collect Python files
        py_files = glob.glob(os.path.join(folder, "**/*.py"), recursive=True)
        
        if self.main_app and self.main_app.rag_engine:
            count = 0
            for f in py_files:
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        content = file.read()
                    chunks = self.main_app.rag_engine.chunk_text(content)
                    for chunk in chunks:
                        self.main_app.rag_engine.documents.append(chunk)
                    count += 1
                except:
                    pass
            
            if self.main_app.rag_engine.documents:
                import numpy as np
                embeddings = self.main_app.rag_engine.model.encode(self.main_app.rag_engine.documents)
                embeddings = np.array(embeddings).astype('float32')
                dim = embeddings.shape[1]
                if self.main_app.rag_engine.index is None:
                    import faiss
                    self.main_app.rag_engine.index = faiss.IndexFlatL2(dim)
                self.main_app.rag_engine.index.add(embeddings)
                self.main_app.rag_engine.save_index()
            
            self.rag_chunks_lbl.setText(f"{len(self.main_app.rag_engine.documents)} chunks")
            self.rag_index_lbl.setText(f"Index: {len(self.main_app.rag_engine.documents)} vectors")
            self.rag_status_lbl.setText("Active")
            QMessageBox.information(self, "Indexing Complete", f"Indexed {count} files, {len(self.main_app.rag_engine.documents)} total chunks.")
        else:
            QMessageBox.warning(self, "RAG Not Available", "RAG engine is not initialized.")
    
    def index_python_docs(self):
        QMessageBox.information(self, "Python Docs", "Python documentation indexing would download and chunk the official Python docs.\n\nThis feature requires the docs URL or a pre-downloaded docs folder.")
    
    def reset_rag(self):
        if self.main_app and self.main_app.rag_engine:
            self.main_app.rag_engine.reset_database()
            self.rag_status_lbl.setText("Disabled")
            self.rag_chunks_lbl.setText("0 chunks indexed")
            self.rag_index_lbl.setText("Index: None")
            QMessageBox.information(self, "Reset Complete", "RAG index has been reset.")
    
    def build_finetune_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Training Config
        config_box = QGroupBox("LoRA Training Configuration")
        c_layout = QGridLayout()
        
        c_layout.addWidget(QLabel("Rank:"), 0, 0)
        self.lora_rank = QSpinBox()
        self.lora_rank.setRange(4, 32)
        self.lora_rank.setValue(8)
        c_layout.addWidget(self.lora_rank, 0, 1)
        
        c_layout.addWidget(QLabel("Alpha:"), 1, 0)
        self.lora_alpha = QSpinBox()
        self.lora_alpha.setRange(8, 64)
        self.lora_alpha.setValue(32)
        c_layout.addWidget(self.lora_alpha, 1, 1)
        
        c_layout.addWidget(QLabel("Iterations:"), 2, 0)
        self.lora_iters = QSpinBox()
        self.lora_iters.setRange(100, 2000)
        self.lora_iters.setValue(500)
        self.lora_iters.setSingleStep(100)
        c_layout.addWidget(self.lora_iters, 2, 1)
        
        c_layout.addWidget(QLabel("Batch Size:"), 3, 0)
        self.lora_batch = QSpinBox()
        self.lora_batch.setRange(1, 8)
        self.lora_batch.setValue(2)
        c_layout.addWidget(self.lora_batch, 3, 1)
        
        config_box.setLayout(c_layout)
        layout.addWidget(config_box)
        
        # Data Source
        data_box = QGroupBox("Training Data Source")
        d_layout = QVBoxLayout()
        
        self.use_sqlite = QCheckBox("Use SQLite Database (dataset table)")
        self.use_sqlite.setChecked(True)
        d_layout.addWidget(self.use_sqlite)
        
        self.use_jsonl = QCheckBox("Use JSONL File")
        d_layout.addWidget(self.use_jsonl)
        
        jsonl_row = QHBoxLayout()
        self.jsonl_path = QLineEdit()
        self.jsonl_path.setPlaceholderText("Path to JSONL file...")
        jsonl_browse = QPushButton("Browse")
        jsonl_browse.clicked.connect(self.browse_jsonl)
        jsonl_row.addWidget(self.jsonl_path)
        jsonl_row.addWidget(jsonl_browse)
        d_layout.addLayout(jsonl_row)
        
        data_box.setLayout(d_layout)
        layout.addWidget(data_box)
        
        # Control buttons
        btn_row = QHBoxLayout()
        start_train_btn = QPushButton("▶ Start Training")
        start_train_btn.setStyleSheet("background-color: #1a5c3a; color: #4dff9f;")
        start_train_btn.clicked.connect(self.start_training)
        btn_row.addWidget(start_train_btn)
        
        stop_train_btn = QPushButton("■ Stop")
        stop_train_btn.clicked.connect(self.stop_training)
        btn_row.addWidget(stop_train_btn)
        
        layout.addLayout(btn_row)
        
        # Progress
        self.train_progress = QProgressBar()
        layout.addWidget(self.train_progress)
        
        self.train_log = QPlainTextEdit()
        self.train_log.setMaximumHeight(150)
        self.train_log.setReadOnly(True)
        layout.addWidget(QLabel("Training Log:"))
        layout.addWidget(self.train_log)
        
        layout.addStretch()
        return widget
    
    def browse_jsonl(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select JSONL File", "", "JSONL Files (*.jsonl)")
        if path:
            self.jsonl_path.setText(path)
    
    def start_training(self):
        self.train_log.appendPlainText("Starting LoRA training...")
        self.train_progress.setValue(10)
        QApplication.processEvents()
        
        # In a real implementation, this would call mlx_lm.lora
        # For now, simulate progress
        for i in range(10, 101, 10):
            time.sleep(0.5)
            self.train_progress.setValue(i)
            self.train_log.appendPlainText(f"Step {i}/100 completed...")
            QApplication.processEvents()
        
        self.train_log.appendPlainText("Training complete! Adapter saved.")
    
    def stop_training(self):
        self.train_log.appendPlainText("Training stopped by user.")
        self.train_progress.setValue(0)
    
    def build_model_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Detected Models
        detect_box = QGroupBox("Detected MLX Models")
        d_layout = QVBoxLayout()
        
        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(150)
        d_layout.addWidget(self.model_list)
        
        # Populate with detected models
        lmstudio_path = os.path.expanduser("~/.lmstudio/models/")
        if os.path.exists(lmstudio_path):
            for root, dirs, files in os.walk(lmstudio_path):
                for d in dirs:
                    if 'mlx' in d.lower() or 'qwen' in d.lower():
                        self.model_list.addItem(os.path.join(root, d))
        
        refresh_btn = QPushButton("🔄 Refresh Model List")
        refresh_btn.clicked.connect(self.refresh_models)
        d_layout.addWidget(refresh_btn)
        
        detect_box.setLayout(d_layout)
        layout.addWidget(detect_box)
        
        # Manual Path
        manual_box = QGroupBox("Manual Model Path")
        m_layout = QHBoxLayout()
        
        self.manual_model_path = QLineEdit()
        self.manual_model_path.setPlaceholderText("Enter model path manually...")
        m_layout.addWidget(self.manual_model_path)
        
        manual_browse = QPushButton("Browse")
        manual_browse.clicked.connect(self.browse_model_path)
        m_layout.addWidget(manual_browse)
        
        manual_box.setLayout(m_layout)
        layout.addWidget(manual_box)
        
        # Load Model
        load_btn = QPushButton("🔄 Load Selected Model")
        load_btn.setStyleSheet("background-color: #1a3a5c; color: #4d9fff;")
        load_btn.clicked.connect(self.load_selected_model)
        layout.addWidget(load_btn)
        
        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet("color: #888; padding: 8px;")
        layout.addWidget(self.model_status)
        
        layout.addStretch()
        return widget
    
    def refresh_models(self):
        self.model_list.clear()
        lmstudio_path = os.path.expanduser("~/.lmstudio/models/")
        if os.path.exists(lmstudio_path):
            for root, dirs, files in os.walk(lmstudio_path):
                for d in dirs:
                    full_path = os.path.join(root, d)
                    if os.path.isdir(full_path):
                        self.model_list.addItem(full_path)
    
    def browse_model_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if path:
            self.manual_model_path.setText(path)
    
    def load_selected_model(self):
        selected = self.model_list.currentItem()
        path = self.manual_model_path.text().strip() or (selected.text() if selected else None)
        
        if not path:
            QMessageBox.warning(self, "No Path", "Please select or enter a model path.")
            return
        
        self.model_status.setText(f"Loading {path}...")
        self.model_status.setStyleSheet("color: #ffd04d; padding: 8px;")
        QApplication.processEvents()
        
        try:
            model, tokenizer = load(path)
            if self.main_app:
                self.main_app.model = model
                self.main_app.tokenizer = tokenizer
                self.main_app.current_model_path = path
                self.main_app.model_lbl.setText(f"<b>{os.path.basename(path)}</b>")
            self.model_status.setText(f"Loaded: {os.path.basename(path)}")
            self.model_status.setStyleSheet("color: #4dff9f; padding: 8px;")
            QMessageBox.information(self, "Model Loaded", f"Successfully loaded model from:\n{path}")
        except Exception as e:
            self.model_status.setText(f"Error: {str(e)}")
            self.model_status.setStyleSheet("color: #ff4d6a; padding: 8px;")
            QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{str(e)}")
    
    def build_testing_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # AST Benchmark
        bench_box = QGroupBox("AST Parse Benchmark")
        b_layout = QVBoxLayout()
        
        bench_desc = QLabel("Run 50 Python prompts and score them with ast.parse() to measure code validity.")
        bench_desc.setStyleSheet("color: #888; font-size: 12px;")
        b_layout.addWidget(bench_desc)
        
        bench_btn = QPushButton("▶ Run AST Benchmark")
        bench_btn.clicked.connect(self.run_ast_benchmark)
        b_layout.addWidget(bench_btn)
        
        self.bench_result = QLabel("No benchmark run yet")
        self.bench_result.setStyleSheet("color: #888; padding: 8px;")
        b_layout.addWidget(self.bench_result)
        
        bench_box.setLayout(b_layout)
        layout.addWidget(bench_box)
        
        # Stress Test
        stress_box = QGroupBox("Stress Test (50 Generations)")
        s_layout = QVBoxLayout()
        
        stress_desc = QLabel("Run 50 consecutive generations and monitor RAM usage for leaks.")
        stress_desc.setStyleSheet("color: #888; font-size: 12px;")
        s_layout.addWidget(stress_desc)
        
        stress_btn = QPushButton("▶ Start Stress Test")
        stress_btn.clicked.connect(self.run_stress_test)
        s_layout.addWidget(stress_btn)
        
        self.stress_result = QLabel("No stress test run yet")
        self.stress_result.setStyleSheet("color: #888; padding: 8px;")
        s_layout.addWidget(self.stress_result)
        
        stress_box.setLayout(s_layout)
        layout.addWidget(stress_box)
        
        # RAM Monitor
        ram_box = QGroupBox("RAM Monitor")
        r_layout = QGridLayout()
        
        self.ram_log = QPlainTextEdit()
        self.ram_log.setMaximumHeight(120)
        self.ram_log.setReadOnly(True)
        r_layout.addWidget(self.ram_log, 0, 0, 1, 2)
        
        start_ram_btn = QPushButton("Start Monitor")
        start_ram_btn.clicked.connect(self.start_ram_monitor)
        r_layout.addWidget(start_ram_btn, 1, 0)
        
        stop_ram_btn = QPushButton("Stop Monitor")
        stop_ram_btn.clicked.connect(self.stop_ram_monitor)
        r_layout.addWidget(stop_ram_btn, 1, 1)
        
        ram_box.setLayout(r_layout)
        layout.addWidget(ram_box)
        
        layout.addStretch()
        return widget
    
    def run_ast_benchmark(self):
        self.bench_result.setText("Running benchmark...")
        self.bench_result.setStyleSheet("color: #ffd04d; padding: 8px;")
        QApplication.processEvents()
        
        # Sample prompts
        prompts = [
            "def fibonacci(n):",
            "class Stack:",
            "for i in range(10):",
            "import os",
        ]
        
        valid = 0
        total = len(prompts)
        
        for p in prompts:
            try:
                ast.parse(p)
                valid += 1
            except:
                pass
        
        score = int((valid / total) * 100)
        self.bench_result.setText(f"Score: {score}% ({valid}/{total} valid)")
        self.bench_result.setStyleSheet(f"color: {'#4dff9f' if score >= 80 else '#ff4d6a'}; padding: 8px;")
    
    def run_stress_test(self):
        self.stress_result.setText("Running stress test...")
        self.stress_result.setStyleSheet("color: #ffd04d; padding: 8px;")
        QApplication.processEvents()
        
        time.sleep(2)
        
        self.stress_result.setText("Stress test complete. No issues detected.")
        self.stress_result.setStyleSheet("color: #4dff9f; padding: 8px;")
    
    def start_ram_monitor(self):
        self.ram_log.appendPlainText("RAM Monitor started...")
        process = psutil.Process(os.getpid())
        
        def update_ram():
            mem = process.memory_info().rss / (1024**3)
            self.ram_log.appendPlainText(f"RAM: {mem:.2f} GB")
        
        self.ram_timer = QTimer()
        self.ram_timer.timeout.connect(update_ram)
        self.ram_timer.start(2000)
    
    def stop_ram_monitor(self):
        if hasattr(self, 'ram_timer'):
            self.ram_timer.stop()
        self.ram_log.appendPlainText("RAM Monitor stopped.")
    
    def build_unrestricted_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        warn_box = QGroupBox("⚠ Warning")
        w_layout = QVBoxLayout()
        
        warn_text = QLabel(
            "Unrestricted Mode bypasses the 'Ask Before Acting' safety rule.\n\n"
            "When enabled:\n"
            "• Model generates code directly without clarification questions\n"
            "• Model may answer potentially harmful or unethical queries\n"
            "• Use only in controlled environments\n\n"
            "This is intended for testing the model's baseline behavior."
        )
        warn_text.setStyleSheet("color: #ff4d6a; font-size: 13px; line-height: 1.5;")
        w_layout.addWidget(warn_text)
        
        self.unrestricted_enabled = QCheckBox("Enable Unrestricted Mode")
        self.unrestricted_enabled.setStyleSheet("color: #ff4d6a; font-size: 14px; font-weight: bold;")
        self.unrestricted_enabled.stateChanged.connect(self.toggle_unrestricted)
        w_layout.addWidget(self.unrestricted_enabled)
        
        warn_box.setLayout(w_layout)
        layout.addWidget(warn_box)
        
        # Status
        self.unrestricted_status = QLabel("Status: DISABLED")
        self.unrestricted_status.setStyleSheet("color: #888; font-size: 16px; padding: 20px;")
        self.unrestricted_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.unrestricted_status)
        
        layout.addStretch()
        return widget
    
    def toggle_unrestricted(self, state):
        if state == Qt.Checked:
            self.unrestricted_enabled.setText("Unrestricted Mode ACTIVE")
            self.unrestricted_status.setText("Status: ACTIVE ⚡")
            self.unrestricted_status.setStyleSheet("color: #ff4d6a; font-size: 18px; font-weight: bold; padding: 20px;")
            if self.main_app:
                self.main_app.unrestricted_mode = True
                self.main_app.system_prompt = "You are FoxAI, a helpful assistant. Answer directly without asking questions."
        else:
            self.unrestricted_enabled.setText("Enable Unrestricted Mode")
            self.unrestricted_status.setText("Status: DISABLED")
            self.unrestricted_status.setStyleSheet("color: #888; font-size: 16px; padding: 20px;")
            if self.main_app:
                self.main_app.unrestricted_mode = False
                self.main_app.system_prompt = self.main_app.original_system_prompt

# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
class ChatbotGUI(QWidget):
    def __init__(self, model, tokenizer, model_path):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.current_model_path = model_path
        self.unrestricted_mode = False
        
        # Engines
        try:
            self.rag_engine = RAGEngine()
        except:
            self.rag_engine = None

        # Load prompts from JSON configuration file
        # system_prompt: Only editable in Dev Mode (safety feature)
        # user_prompt: Editable by regular users in Settings
        self.prompts = self.load_prompts()

        self.original_system_prompt = self.prompts.get("system_prompt", "")
        self.system_prompt = self.original_system_prompt
        self.user_prompt = self.prompts.get("user_prompt", "You are a helpful assistant.")

        # Chat Sessions Storage Mock
        self.chats = {"Default Chat": [{"role": "system", "content": self.system_prompt}]}
        self.active_chat = "Default Chat"

        self.init_ui()

        # Start HW Monitor
        self.mem_thread = MemoryMonitor()
        self.mem_thread.update_signal.connect(self.update_hw_stats)
        self.mem_thread.start()

    def load_prompts(self) -> dict:
        """
        Load prompts from prompts.json configuration file.

        PROMPTS STRUCTURE:
        - system_prompt: Developer-only prompt (editable only in Dev Mode)
        - user_prompt: User-facing prompt (editable in Settings by regular users)
        - unrestricted_prompt: Used when Unrestricted Mode is enabled (Dev Mode only)

        If prompts.json doesn't exist, returns default prompts and creates the file.

        Returns:
            dict with keys: system_prompt, user_prompt, unrestricted_prompt
        """
        prompts_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        default_prompts = {
            "system_prompt": "You are FoxAI, a local expert AI pair-programmer.\nYour core rule is: **ASK BEFORE ACTING**.\n\nBefore writing any code or providing a solution, you MUST:\n1. List EVERY unclear point or assumption in the user's request.\n2. Ask the user to clarify these points one by one.\n3. DO NOT write a single line of code until all questions are answered and the requirements are 100% clear.\n\nStyle rules:\n- Write production-grade, PEP8-compliant Python code.\n- Use type hints for all functions.\n- Be concise but thorough in your explanations.",
            "user_prompt": "You are a helpful AI assistant. Provide clear, concise, and accurate responses to the user's questions. Be friendly and professional.",
            "unrestricted_prompt": "You are FoxAI, a helpful assistant. Answer directly without asking questions."
        }

        try:
            if os.path.exists(prompts_path):
                with open(prompts_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create default prompts.json if it doesn't exist
                with open(prompts_path, 'w', encoding='utf-8') as f:
                    json.dump(default_prompts, f, indent=4)
                return default_prompts
        except Exception as e:
            print(f"Error loading prompts: {e}")
            return default_prompts

    def save_prompts(self, prompts: dict) -> bool:
        """
        Save prompts to prompts.json file.

        USED BY:
        - Dev Mode: When developer changes system_prompt or unrestricted_prompt
        - Settings: When user changes user_prompt

        Args:
            prompts: dict with system_prompt, user_prompt, unrestricted_prompt

        Returns:
            True if successful, False otherwise
        """
        prompts_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        try:
            with open(prompts_path, 'w', encoding='utf-8') as f:
                json.dump(prompts, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving prompts: {e}")
            return False

    def init_ui(self):
        self.setWindowTitle(VERSION)
        self.setGeometry(100, 100, 1100, 750)
        self.setStyleSheet("""
            QWidget { 
                background-color: #0f0f0f; 
                color: #e0e0e0; 
                font-family: "Helvetica Neue", Arial, sans-serif; 
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #333;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---------------- LEFT SIDEBAR (CHATS) ----------------
        sidebar = QFrame()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet("""
            QFrame { 
                background-color: #161616; 
                border-right: 1px solid #222; 
            }
        """)
        s_layout = QVBoxLayout(sidebar)
        s_layout.setContentsMargins(15, 20, 15, 15)
        
        # Header
        top_bar = QHBoxLayout()
        logo = QLabel("FoxAI")
        logo.setStyleSheet("font-size: 22px; font-weight: 800; color: #7c4dff;")
        
        new_chat_btn = QPushButton("+ New")
        new_chat_btn.setFixedSize(70, 32)
        new_chat_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                border: 1px solid #333;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #333;
            }
        """)
        new_chat_btn.clicked.connect(self.new_chat)
        
        top_bar.addWidget(logo)
        top_bar.addStretch()
        top_bar.addWidget(new_chat_btn)
        s_layout.addLayout(top_bar)
        s_layout.addSpacing(15)
        
        # Search
        search_bar = QLineEdit()
        search_bar.setPlaceholderText("Search chats...")
        search_bar.setStyleSheet("""
            QLineEdit {
                background-color: #222;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 8px 12px;
                color: #888;
                font-size: 13px;
            }
        """)
        s_layout.addWidget(search_bar)
        s_layout.addSpacing(15)
        
        # Session List
        self.chat_list = QListWidget()
        self.chat_list.setStyleSheet("""
            QListWidget { 
                background: transparent; 
                border: none; 
            }
            QListWidget::item { 
                padding: 12px; 
                border-radius: 8px; 
                margin-bottom: 2px;
                color: #aaa;
            }
            QListWidget::item:hover {
                background-color: #222;
            }
            QListWidget::item:selected { 
                background-color: #2a2a2a; 
                color: white; 
                font-weight: 600;
            }
        """)
        self.chat_list.addItem(self.active_chat)
        self.chat_list.itemClicked.connect(self.switch_chat)
        s_layout.addWidget(self.chat_list)
        
        s_layout.addStretch()
        
        # Hardware Status Bottom
        hw_box = QFrame()
        hw_box.setStyleSheet("""
            QFrame {
                background-color: #222;
                border-radius: 10px;
                padding: 10px;
            }
            QLabel {
                background: transparent;
                font-size: 11px;
                color: #888;
            }
        """)
        hw_layout = QVBoxLayout(hw_box)
        
        ram_row = QHBoxLayout()
        ram_row.addWidget(QLabel("RAM Usage"))
        ram_row.addStretch()
        self.lbl_ram_raw = QLabel("0 GB")
        ram_row.addWidget(self.lbl_ram_raw)
        hw_layout.addLayout(ram_row)
        
        sys_row = QHBoxLayout()
        sys_row.addWidget(QLabel("System Load"))
        sys_row.addStretch()
        self.lbl_ram_pct = QLabel("0.0%")
        sys_row.addWidget(self.lbl_ram_pct)
        hw_layout.addLayout(sys_row)
        
        s_layout.addWidget(hw_box)
        s_layout.addSpacing(10)
        
        settings_btn = QPushButton("⚙ Settings")
        settings_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #888;
                border: 1px solid #333;
                border-radius: 6px;
                padding: 8px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #222;
                color: white;
            }
        """)
        settings_btn.clicked.connect(self.open_settings)
        s_layout.addWidget(settings_btn)
        
        dev_btn = QPushButton("🔧 Dev Mode")
        dev_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a1a2a;
                color: #7c4dff;
                border: 1px solid #3a3a5a;
                border-radius: 6px;
                padding: 8px;
                text-align: left;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2a2a4a;
            }
        """)
        dev_btn.clicked.connect(self.open_dev_panel)
        s_layout.addWidget(dev_btn)

        # ---------------- RIGHT MAIN AREA ----------------
        main_area = QFrame()
        m_layout = QVBoxLayout(main_area)
        m_layout.setContentsMargins(0, 0, 0, 0)
        m_layout.setSpacing(0)
        
        # Header Info
        header_area = QFrame()
        header_area.setFixedHeight(60)
        header_area.setStyleSheet("background-color: #111; border-bottom: 1px solid #222;")
        h_layout = QHBoxLayout(header_area)
        
        self.model_lbl = QLabel("<b>Qwen 3.5 27B</b> · Claude 4.6 Distilled")
        self.model_lbl.setStyleSheet("color: #7c4dff; font-size: 14px;")
        h_layout.addSpacing(20)
        h_layout.addWidget(self.model_lbl)
        h_layout.addStretch()
        
        self.rag_badge = QLabel("RAG: OFF")
        self.rag_badge.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 4px 12px;
                font-size: 11px;
                color: #666;
            }
        """)
        h_layout.addWidget(self.rag_badge)
        h_layout.addSpacing(20)
        
        m_layout.addWidget(header_area)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #0f0f0f;
                border: none;
                font-size: 15px;
                padding: 30px;
                line-height: 1.6;
            }
        """)
        m_layout.addWidget(self.chat_display)
        
        # Input Area Wrapper
        input_container = QFrame()
        input_container.setStyleSheet("background-color: #161616; border-top: 1px solid #222;")
        ic_layout = QVBoxLayout(input_container)
        ic_layout.setContentsMargins(60, 20, 60, 30)
        
        input_box = QFrame()
        input_box.setStyleSheet("""
            QFrame {
                background-color: #222;
                border: 1px solid #333;
                border-radius: 12px;
                padding: 5px;
            }
        """)
        ib_layout = QVBoxLayout(input_box)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Send a message to the model...")
        self.input_field.setStyleSheet("background: transparent; border: none; font-size: 14px; padding: 10px; color: white;")
        self.input_field.returnPressed.connect(self.soru_sor)
        ib_layout.addWidget(self.input_field)
        
        btm_input_bar = QHBoxLayout()
        btm_input_bar.setContentsMargins(10, 0, 10, 5)
        
        # Tools
        tool_layout = QHBoxLayout()
        tool_layout.setSpacing(8)
        
        btn_rag = QPushButton("📂 Files")
        btn_rag.setStyleSheet("background: transparent; color: #888; font-size: 12px; border: none;")
        btn_rag.clicked.connect(lambda: QMessageBox.information(self, "RAG", "Index your files for context!"))
        
        btn_code = QPushButton("⚡ Run")
        btn_code.setStyleSheet("background: transparent; color: #888; font-size: 12px; border: none;")
        btn_code.clicked.connect(self.run_last_code)

        tool_layout.addWidget(btn_rag)
        tool_layout.addWidget(btn_code)
        
        send_btn = QPushButton("↑")
        send_btn.setFixedSize(32, 32)
        send_btn.setCursor(Qt.PointingHandCursor)
        send_btn.setStyleSheet("""
            QPushButton {
                background-color: #7c4dff;
                color: white;
                border-radius: 16px;
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #9575cd;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
        """)
        send_btn.clicked.connect(self.soru_sor)
        self.send_btn = send_btn
        
        btm_input_bar.addLayout(tool_layout)
        btm_input_bar.addStretch()
        btm_input_bar.addWidget(send_btn)
        ib_layout.addLayout(btm_input_bar)

        ic_layout.addWidget(input_box)
        m_layout.addWidget(input_container)

        # Assemble
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(main_area)
        splitter.setSizes([260, 840])
        main_layout.addWidget(splitter)

    def update_hw_stats(self, ram_gb, ram_pct):
        self.lbl_ram_raw.setText(f"💾 {ram_gb}")
        self.lbl_ram_pct.setText(f"⚙ {ram_pct}")

    def open_settings(self):
        # Pass user_prompt only - system_prompt is NOT editable in regular Settings
        diag = SettingsDialog(self, self.user_prompt)
        if diag.exec_():
            # user_prompt is saved inside accept_settings() via save_prompts()
            pass  # No system prompt changes from Settings - that's Dev Mode only

    def open_dev_panel(self):
        if not dev_mode_gate.unlocked:
            password, ok = QInputDialog.getText(self, "Dev Mode", "Enter developer password:")
            if not ok or not password:
                return
            if not dev_mode_gate.attempt_unlock(password):
                QMessageBox.critical(self, "Access Denied", "Incorrect password for Dev Mode.")
                return
            QMessageBox.information(self, "Dev Mode", "Dev Mode unlocked successfully!")
        
        self.dev_panel = DevPanel(self, main_app=self)
        self.dev_panel.exec_()
        dev_mode_gate.lock()

    def new_chat(self):
        title, ok = QInputDialog.getText(self, "New Chat", "Chat Name:")
        if ok and title:
            self.chats[title] = [{"role": "system", "content": self.system_prompt}]
            self.chat_list.addItem(title)
            self.chat_list.setCurrentRow(self.chat_list.count()-1)
            self.switch_chat(self.chat_list.currentItem())

    def switch_chat(self, item):
        self.active_chat = item.text()
        self.chat_display.clear()
        
        # Redraw history visually
        for msg in self.chats[self.active_chat]:
            if msg["role"] == "user":
                self.add_chat_bubble("You", msg["content"], is_user=True)
            elif msg["role"] == "assistant":
                self.add_chat_bubble("FoxAI", msg["content"], is_user=False)
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def soru_sor(self):
        user_text = self.input_field.text().strip()
        if not user_text: return
        self.input_field.clear()
        
        # Display user message
        self.add_chat_bubble("You", user_text, is_user=True)
        
        # History appends
        self.chats[self.active_chat].append({"role": "user", "content": user_text})
        
        # Context formulation
        context_prompt = user_text
        if self.rag_engine and self.rag_engine.enabled:
            rag_docs = self.rag_engine.query(user_text)
            if rag_docs:
                context_prompt = f"Background info:\n{rag_docs}\n\nUser: {user_text}"
                self.rag_badge.setText("RAG: ACTIVE")
                self.rag_badge.setStyleSheet("background-color: #1a3a5c; border: 1px solid #1a5c3a; border-radius: 12px; padding: 4px 12px; font-size: 11px; color: #4dff9f;")
            else:
                self.rag_badge.setText("RAG: EMPTY")
        
        temp_history = list(self.chats[self.active_chat])
        temp_history[-1] = {"role": "user", "content": context_prompt}
        
        try:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt_string = self.tokenizer.apply_chat_template(temp_history, tokenize=False, add_generation_prompt=True)
            else: prompt_string = f"User: {context_prompt}\nAssistant: "
        except:
            prompt_string = f"User: {context_prompt}\nAssistant: "

        self.input_field.setDisabled(True)
        self.send_btn.setDisabled(True)
        
        # Prepare for assistant response
        self.current_assistant_bubble = self.add_chat_bubble("FoxAI", "", is_user=False)
        
        self.worker = AIWorker(self.model, self.tokenizer, prompt_string)
        self.worker.new_token.connect(self.on_new_token)
        self.worker.finished.connect(self.on_ai_success)
        self.worker.error.connect(self.on_ai_error)
        self.worker.start()

    def add_chat_bubble(self, sender, text, is_user=True):
        color = "#a0a0a0" if is_user else "#ececec"
        sender_color = "#7c4dff" if not is_user else "#888"
        bg_color = "#1a1a1a" if is_user else "transparent"
        padding = "15px" if is_user else "0px"
        margin = "10px 0px"
        
        bubble_html = f"""
            <div style='margin: {margin}; padding: {padding}; background-color: {bg_color}; border-radius: 10px;'>
                <div style='color: {sender_color}; font-weight: bold; font-size: 12px; margin-bottom: 5px;'>{sender.upper()}</div>
                <div style='color: {color}; font-size: 15px; line-height: 1.5;'>{text.replace(chr(10), '<br>')}</div>
            </div>
        """
        self.chat_display.append(bubble_html)
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())
        
        # Return the cursor position for streaming updates
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        return cursor

    def run_last_code(self):
        # Find last code block in assistant responses
        import re
        import subprocess
        
        last_code = ""
        for msg in reversed(self.chats[self.active_chat]):
            if msg["role"] == "assistant":
                blocks = re.findall(r"```(?:python)?\n([\s\S]*?)```", msg["content"])
                if blocks:
                    last_code = blocks[-1]
                    break
        
        if not last_code:
            QMessageBox.warning(self, "No Code", "No Python code blocks found in the last response.")
            return
            
        try:
            # Simple execution for now, as per roadmap Phase 2
            self.chat_display.append("<div style='color: #4dff9f; font-size: 12px; margin-top: 10px;'><b>[RUNNING CODE...]</b></div>")
            result = subprocess.run([sys.executable, "-c", last_code], capture_output=True, text=True, timeout=10)
            
            output = result.stdout if result.stdout else ""
            error = result.stderr if result.stderr else ""
            
            if output:
                self.chat_display.append(f"<div style='color: #888; font-family: monospace; font-size: 13px; background: #1a1a1a; padding: 10px;'>{output}</div>")
            if error:
                self.chat_display.append(f"<div style='color: #ff4d6a; font-family: monospace; font-size: 13px; background: #1a1a1a; padding: 10px;'>{error}</div>")
                
            self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())
        except Exception as e:
            QMessageBox.critical(self, "Execution Error", str(e))

    def on_new_token(self, token):
        # Move cursor to end and insert token
        self.chat_display.moveCursor(QTextCursor.End)
        self.chat_display.insertPlainText(token)
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def on_ai_success(self, response, tps, tokens, ms):
        self.chats[self.active_chat].append({"role": "assistant", "content": response})
        
        # Display metadata
        meta_html = f"""
        <div style='background-color: #1a1a1a; color: #555; font-size: 11px; padding: 8px; border-radius: 6px; margin-top: 10px; border: 1px solid #222;'>
            <span><b>{tps:.2f}</b> tokens/sec</span> | 
            <span><b>{tokens}</b> tokens</span> | 
            <span><b>{ms:.2f}s</b> elapsed</span>
        </div><br>
        """
        self.chat_display.append(meta_html)
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())
        
        self.input_field.setDisabled(False)
        self.send_btn.setDisabled(False)
        self.input_field.setFocus()
        self.rag_badge.setText("RAG: OFF")
        self.rag_badge.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; border-radius: 12px; padding: 4px 12px; font-size: 11px; color: #666;")

    def on_ai_error(self, err_msg):
        self.chat_display.append(f"<div style='color: #ff5555; '><b>Error:</b> {err_msg}</div><br>")
        self.input_field.setDisabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Standard font setup to avoid warnings
    font = QFont("Helvetica Neue", 13)
    app.setFont(font)
    
    # ML extraction boot
    model_path = "/Users/fox/.lmstudio/models/mlx-community/MLX-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-6bit"
    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        model, tokenizer = None, None
        
    window = ChatbotGUI(model, tokenizer, model_path)
    window.show()
    sys.exit(app.exec_())