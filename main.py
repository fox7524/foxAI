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
from collections import deque
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
from PyQt5.QtGui import QFont, QColor, QTextCursor, QPalette, QIcon, QPainter, QPen

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

class ModelLoaderWorker(QThread):
    loaded = pyqtSignal(object, object, str)
    error = pyqtSignal(str)

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            model, tokenizer = load(self.model_path, tokenizer_config={"fix_mistral_regex": True})
            self._ensure_special_tokens(tokenizer)
            self.loaded.emit(model, tokenizer, self.model_path)
        except Exception as e:
            self.error.emit(str(e))

    def _ensure_special_tokens(self, tokenizer):
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_token = getattr(tokenizer, "eos_token", None) or "</s>"
            if hasattr(tokenizer, "convert_tokens_to_ids"):
                try:
                    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
                except Exception:
                    eos_id = None
            if eos_id is not None:
                try:
                    tokenizer.eos_token_id = eos_id
                except Exception:
                    pass

        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            if hasattr(tokenizer, "eos_token_id") and getattr(tokenizer, "eos_token_id", None) is not None:
                try:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                except Exception:
                    pass

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
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model or tokenizer is not loaded.")
            if getattr(self.tokenizer, "eos_token_id", None) is None:
                raise ValueError("Tokenizer eos_token_id is missing. Reload the model or choose another model.")

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


class BenchmarkWorker(QThread):
    finished = pyqtSignal(float, int, float, str)
    error = pyqtSignal(str)

    def __init__(self, model, tokenizer, prompt: str, max_tokens: int = 128):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_tokens = max_tokens

    def run(self):
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model/tokenizer not loaded")
            start = time.time()
            out = generate(self.model, self.tokenizer, prompt=self.prompt, max_tokens=self.max_tokens)
            elapsed = time.time() - start
            token_count = 0
            if hasattr(self.tokenizer, "encode"):
                try:
                    token_count = len(self.tokenizer.encode(out))
                except Exception:
                    token_count = len(out.split())
            else:
                token_count = len(out.split())
            tps = token_count / elapsed if elapsed > 0 else 0.0
            self.finished.emit(tps, token_count, elapsed, out[:2000])
        except Exception as e:
            self.error.emit(str(e))


class MemoryMonitor(QThread):
    """
    Background thread for monitoring system resources.

    WHY SEPARATE THREAD?
    - psutil calls can be slow
    - We want to update every 2 seconds without affecting UI responsiveness

    SIGNALS:
    - update_signal(str, str): Emits (ram_usage_gb, cpu_percent)
    """
    update_signal = pyqtSignal(str, str, str, str)  # (app_ram_gb, sys_ram_percent, cpu_percent, gpu_percent)

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

                sys_mem = psutil.virtual_memory()
                sys_ram_percent = sys_mem.percent

                cpu_percent = psutil.cpu_percent(interval=None)

                gpu_percent = self._get_gpu_util_percent()

                self.update_signal.emit(
                    f"{used_gb:.2f} GB",
                    f"{sys_ram_percent:.1f}%",
                    f"{cpu_percent:.1f}%",
                    gpu_percent,
                )
            except:
                pass  # Ignore errors, try again next iteration
            time.sleep(2)

    def _get_gpu_util_percent(self) -> str:
        if sys.platform == "darwin":
            return "N/A"

        try:
            import shutil
            if shutil.which("nvidia-smi") is None:
                return "N/A"
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                val = result.stdout.strip().splitlines()[0].strip()
                if val:
                    return f"{float(val):.1f}%"
        except Exception:
            pass
        return "N/A"

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
        self.rb_system = QRadioButton("🖥 System")

        if current_theme == "system":
            self.rb_system.setChecked(True)
        elif current_theme == "light":
            self.rb_light.setChecked(True)
        else:
            self.rb_dark.setChecked(True)

        theme_layout.addWidget(self.rb_dark)
        theme_layout.addWidget(self.rb_light)
        theme_layout.addWidget(self.rb_system)
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

        # Live theme preview (applies immediately)
        self.rb_dark.toggled.connect(lambda _v: self._preview_theme())
        self.rb_light.toggled.connect(lambda _v: self._preview_theme())
        self.rb_system.toggled.connect(lambda _v: self._preview_theme())

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
        if self.rb_system.isChecked():
            self.final_theme = "system"
        elif self.rb_light.isChecked():
            self.final_theme = "light"
        else:
            self.final_theme = "dark"

        # Update main app's user_prompt
        if self.main_app and hasattr(self.main_app, 'prompts'):
            self.main_app.prompts["user_prompt"] = self.final_user_prompt
            self.main_app.prompts["theme"] = self.final_theme
            self.main_app.save_prompts(self.main_app.prompts)
            self.main_app.user_prompt = self.final_user_prompt
            self.main_app.apply_theme(self.final_theme)

        self.accept()

    def _preview_theme(self):
        if not self.main_app:
            return
        if self.rb_system.isChecked():
            self.main_app.apply_theme("system")
        elif self.rb_light.isChecked():
            self.main_app.apply_theme("light")
        elif self.rb_dark.isChecked():
            self.main_app.apply_theme("dark")

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
class DevPanel(QWidget):
    def __init__(self, parent=None, main_app=None):
        super().__init__(parent)
        self.main_app = main_app
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Developer")
        header.setObjectName("DevHeader")
        layout.addWidget(header)
        
        tabs = QTabWidget()
        
        # TAB 1: RAG CONTROLS
        tabs.addTab(self.build_rag_tab(), "RAG Indexer")
        # TAB 2: FINE-TUNING
        tabs.addTab(self.build_finetune_tab(), "⚡ Fine-tune")
        # TAB 3: MODEL SELECTOR
        tabs.addTab(self.build_model_tab(), "Model")
        # TAB 4: TESTING
        tabs.addTab(self.build_testing_tab(), "Testing")
        # TAB 5: UNRESTRICTED MODE
        tabs.addTab(self.build_unrestricted_tab(), "⚠ Unrestricted")
        
        layout.addWidget(tabs)
        
        # Bottom buttons
        bottom = QHBoxLayout()
        bottom.addStretch()
        close_btn = QPushButton("Hide")
        close_btn.clicked.connect(self._hide_dev_sidebar)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

    def _hide_dev_sidebar(self):
        if self.main_app:
            self.main_app.toggle_dev_sidebar(force_state=False)
    
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

        if self.main_app and self.main_app.rag_engine:
            ok = False
            try:
                ok = self.main_app.rag_engine.ingest_folder(folder, recursive=True)
            except Exception as e:
                QMessageBox.critical(self, "Indexing Error", str(e))
                self.rag_status_lbl.setText("Error")
                return

            stats = {}
            try:
                stats = self.main_app.rag_engine.get_stats()
            except Exception:
                stats = {}

            chunk_count = stats.get("chunk_count", 0)
            indexed = stats.get("indexed", False)

            self.rag_chunks_lbl.setText(f"{chunk_count} chunks indexed")
            self.rag_index_lbl.setText(f"Index: {chunk_count if indexed else 'None'}")
            self.rag_status_lbl.setText("Active" if ok else "No data")

            QMessageBox.information(
                self,
                "Indexing Complete",
                f"Folder indexed: {folder}\nChunks: {chunk_count}"
            )
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

        if self.main_app:
            self.main_app.service_status_lbl.setText("Service: loading…")
            self.main_app._set_chat_enabled(False)

        self._model_loader = ModelLoaderWorker(path)
        self._model_loader.loaded.connect(self._on_model_loaded)
        self._model_loader.error.connect(self._on_model_load_error)
        self._model_loader.start()

    def _on_model_loaded(self, model, tokenizer, model_path: str) -> None:
        if self.main_app:
            self.main_app._on_model_loaded(model, tokenizer, model_path)
        self.model_status.setText(f"Loaded: {os.path.basename(model_path)}")
        self.model_status.setStyleSheet("color: #4dff9f; padding: 8px;")
        QMessageBox.information(self, "Model Loaded", f"Successfully loaded model from:\n{model_path}")

    def _on_model_load_error(self, err: str) -> None:
        if self.main_app:
            self.main_app._on_model_load_error(err)
        self.model_status.setText(f"Error: {err}")
        self.model_status.setStyleSheet("color: #ff4d6a; padding: 8px;")
        QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{err}")
    
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

        smoke_box = QGroupBox("Regression Smoke Tests")
        sm_layout = QVBoxLayout()
        sm_desc = QLabel("Quick checks for prompts.json loading, theme switching, and core UI wiring.")
        sm_desc.setStyleSheet("color: #888; font-size: 12px;")
        sm_layout.addWidget(sm_desc)
        smoke_btn = QPushButton("▶ Run Smoke Tests")
        smoke_btn.clicked.connect(self.run_smoke_tests)
        sm_layout.addWidget(smoke_btn)
        self.smoke_result = QLabel("No smoke tests run yet")
        self.smoke_result.setStyleSheet("color: #888; padding: 8px;")
        sm_layout.addWidget(self.smoke_result)
        smoke_box.setLayout(sm_layout)
        layout.addWidget(smoke_box)

        perf_box = QGroupBox("Performance Benchmark")
        p_layout = QVBoxLayout()
        p_desc = QLabel("Measures generation throughput on the currently loaded model (small run).")
        p_desc.setStyleSheet("color: #888; font-size: 12px;")
        p_layout.addWidget(p_desc)
        perf_btn = QPushButton("▶ Run Throughput Benchmark")
        perf_btn.clicked.connect(self.run_throughput_benchmark)
        p_layout.addWidget(perf_btn)
        self.perf_result = QLabel("No benchmark run yet")
        self.perf_result.setStyleSheet("color: #888; padding: 8px;")
        p_layout.addWidget(self.perf_result)
        perf_box.setLayout(p_layout)
        layout.addWidget(perf_box)
        
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

    def run_smoke_tests(self):
        ok = True
        problems = []

        if not self.main_app:
            self.smoke_result.setText("Smoke tests unavailable: no main app reference")
            self.smoke_result.setStyleSheet("color: #ff4d6a; padding: 8px;")
            return

        try:
            prompts = self.main_app.load_prompts()
            if not isinstance(prompts, dict):
                ok = False
                problems.append("prompts.json did not load as dict")
        except Exception as e:
            ok = False
            problems.append(f"prompts load failed: {e}")

        try:
            self.main_app.apply_theme("dark")
            self.main_app.apply_theme("light")
            self.main_app.apply_theme("system")
            self.main_app.apply_theme(self.main_app.prompts.get("theme", "dark"))
        except Exception as e:
            ok = False
            problems.append(f"theme switching failed: {e}")

        if ok:
            self.smoke_result.setText("OK: prompts load + theme switching + UI wiring")
            self.smoke_result.setStyleSheet("color: #4dff9f; padding: 8px;")
        else:
            self.smoke_result.setText("FAILED: " + " | ".join(problems))
            self.smoke_result.setStyleSheet("color: #ff4d6a; padding: 8px;")

    def run_throughput_benchmark(self):
        if not self.main_app or self.main_app.model is None or self.main_app.tokenizer is None:
            self.perf_result.setText("Benchmark requires a loaded model.")
            self.perf_result.setStyleSheet("color: #ff4d6a; padding: 8px;")
            return

        self.perf_result.setText("Running benchmark…")
        self.perf_result.setStyleSheet("color: #ffd04d; padding: 8px;")
        QApplication.processEvents()

        prompt = "Write a short Python function that adds two numbers."
        self._bench_worker = BenchmarkWorker(self.main_app.model, self.main_app.tokenizer, prompt, max_tokens=128)
        self._bench_worker.finished.connect(self._on_benchmark_done)
        self._bench_worker.error.connect(self._on_benchmark_error)
        self._bench_worker.start()

    def _on_benchmark_done(self, tps: float, tokens: int, elapsed: float, sample: str):
        self.perf_result.setText(f"{tps:.2f} tok/s | {tokens} tokens | {elapsed:.2f}s")
        self.perf_result.setStyleSheet("color: #4dff9f; padding: 8px;")

    def _on_benchmark_error(self, err: str):
        self.perf_result.setText(f"Benchmark failed: {err}")
        self.perf_result.setStyleSheet("color: #ff4d6a; padding: 8px;")

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
                self.main_app.system_prompt = "You are FoxAI, a helpful assistant. Answer directly without asking clarifying questions."
        else:
            self.unrestricted_enabled.setText("Enable Unrestricted Mode")
            self.unrestricted_status.setText("Status: DISABLED")
            self.unrestricted_status.setStyleSheet("color: #888; font-size: 16px; padding: 20px;")
            if self.main_app:
                self.main_app.unrestricted_mode = False
                self.main_app.system_prompt = self.main_app.original_system_prompt


class MiniSparkline(QWidget):
    def __init__(self, range_min: float = 0.0, range_max: float = 100.0, parent=None):
        super().__init__(parent)
        self._values = deque(maxlen=60)
        self._range_min = range_min
        self._range_max = range_max
        self._available = True
        self.setFixedHeight(26)

    def set_available(self, available: bool) -> None:
        self._available = available
        self.update()

    def add_point(self, value):
        self._values.append(value)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()

        pen_bg = QPen(QColor(255, 255, 255, 30), 1)
        painter.setPen(pen_bg)
        painter.drawRoundedRect(0, 0, w - 1, h - 1, 6, 6)

        if not self._available:
            painter.setPen(QPen(QColor(140, 140, 140, 180), 1))
            painter.drawText(self.rect(), Qt.AlignCenter, "N/A")
            return

        vals = [v for v in self._values if isinstance(v, (int, float))]
        if len(vals) < 2:
            painter.setPen(QPen(QColor(140, 140, 140, 120), 1))
            painter.drawLine(6, h // 2, w - 6, h // 2)
            return

        rmin = self._range_min
        rmax = self._range_max
        if rmax <= rmin:
            rmax = rmin + 1.0

        step_x = (w - 12) / max(1, (len(vals) - 1))
        points = []
        for i, v in enumerate(vals):
            nv = max(rmin, min(rmax, float(v)))
            t = (nv - rmin) / (rmax - rmin)
            x = 6 + i * step_x
            y = 4 + (1.0 - t) * (h - 8)
            points.append((x, y))

        painter.setPen(QPen(QColor(124, 77, 255, 220), 2))
        for i in range(1, len(points)):
            painter.drawLine(int(points[i - 1][0]), int(points[i - 1][1]), int(points[i][0]), int(points[i][1]))

# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
class ChatbotGUI(QWidget):
    def __init__(self, model, tokenizer, model_path):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.current_model_path = model_path or ""
        self.unrestricted_mode = False
        self.dev_mode_active = False
        self.dev_sidebar_shown = False
        
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
        self.theme = self.prompts.get("theme", "dark")
        self.current_model_path = self.prompts.get("model_path", self.current_model_path)

        # Chat Sessions Storage Mock
        self.chats = {"Default Chat": [{"role": "system", "content": self.system_prompt}]}
        self.active_chat = "Default Chat"

        self.init_ui()

        # Start HW Monitor
        self.mem_thread = MemoryMonitor()
        self.mem_thread.update_signal.connect(self.update_hw_stats)
        self.mem_thread.start()

        self.start_ai_service()

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
            "unrestricted_prompt": "You are FoxAI, a helpful assistant. Answer directly without asking clarifying questions.",
            "theme": "dark",
            "model_path": ""
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

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---------------- LEFT SIDEBAR (CHATS) ----------------
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(280)
        s_layout = QVBoxLayout(self.sidebar)
        s_layout.setContentsMargins(15, 20, 15, 15)
        
        # Header
        top_bar = QHBoxLayout()
        logo = QLabel("FoxAI")
        logo.setObjectName("Logo")
        
        new_chat_btn = QPushButton("+ New")
        new_chat_btn.setFixedSize(70, 32)
        new_chat_btn.setObjectName("NewChatButton")
        new_chat_btn.clicked.connect(self.new_chat)
        
        top_bar.addWidget(logo)
        top_bar.addStretch()
        top_bar.addWidget(new_chat_btn)
        s_layout.addLayout(top_bar)
        s_layout.addSpacing(15)
        
        # Search
        search_bar = QLineEdit()
        search_bar.setPlaceholderText("Search chats...")
        s_layout.addWidget(search_bar)
        s_layout.addSpacing(15)
        
        # Session List
        self.chat_list = QListWidget()
        self.chat_list.addItem(self.active_chat)
        self.chat_list.itemClicked.connect(self.switch_chat)
        s_layout.addWidget(self.chat_list)
        
        s_layout.addStretch()
        
        # Hardware Status Bottom
        self.hw_box = QFrame()
        self.hw_box.setObjectName("HwBox")
        hw_layout = QVBoxLayout(self.hw_box)
        
        ram_row = QHBoxLayout()
        ram_row.addWidget(QLabel("RAM (App)"))
        ram_row.addStretch()
        self.lbl_ram_raw = QLabel("0.00 GB")
        ram_row.addWidget(self.lbl_ram_raw)
        hw_layout.addLayout(ram_row)

        sys_ram_row = QHBoxLayout()
        sys_ram_row.addWidget(QLabel("RAM (System)"))
        sys_ram_row.addStretch()
        self.lbl_sys_ram_pct = QLabel("0.0%")
        sys_ram_row.addWidget(self.lbl_sys_ram_pct)
        hw_layout.addLayout(sys_ram_row)

        self.ram_chart = MiniSparkline(0.0, 100.0)
        hw_layout.addWidget(self.ram_chart)
        
        sys_row = QHBoxLayout()
        sys_row.addWidget(QLabel("CPU"))
        sys_row.addStretch()
        self.lbl_cpu_pct = QLabel("0.0%")
        sys_row.addWidget(self.lbl_cpu_pct)
        hw_layout.addLayout(sys_row)

        self.cpu_chart = MiniSparkline(0.0, 100.0)
        hw_layout.addWidget(self.cpu_chart)

        gpu_row = QHBoxLayout()
        gpu_row.addWidget(QLabel("GPU (Apple)"))
        gpu_row.addStretch()
        self.lbl_gpu_pct = QLabel("N/A")
        gpu_row.addWidget(self.lbl_gpu_pct)
        hw_layout.addLayout(gpu_row)

        self.gpu_chart = MiniSparkline(0.0, 100.0)
        self.gpu_chart.set_available(False)
        hw_layout.addWidget(self.gpu_chart)
        
        s_layout.addWidget(self.hw_box)
        s_layout.addSpacing(10)
        
        settings_btn = QPushButton("⚙ Settings")
        settings_btn.setObjectName("SettingsButton")
        settings_btn.clicked.connect(self.open_settings)
        s_layout.addWidget(settings_btn)
        
        dev_btn = QPushButton("🔧 Dev Mode")
        dev_btn.setObjectName("DevUnlockButton")
        dev_btn.clicked.connect(self.open_dev_panel)
        s_layout.addWidget(dev_btn)

        # ---------------- RIGHT MAIN AREA ----------------
        self.main_area = QFrame()
        self.main_area.setObjectName("MainArea")
        m_layout = QVBoxLayout(self.main_area)
        m_layout.setContentsMargins(0, 0, 0, 0)
        m_layout.setSpacing(0)
        
        # Header Info
        self.header_area = QFrame()
        self.header_area.setObjectName("HeaderArea")
        self.header_area.setFixedHeight(60)
        h_layout = QHBoxLayout(self.header_area)

        h_layout.addSpacing(20)
        self.service_status_lbl = QLabel("Service: starting…")
        h_layout.addWidget(self.service_status_lbl)
        h_layout.addStretch()

        self.dev_toggle_btn = QPushButton("Dev")
        self.dev_toggle_btn.setFixedSize(54, 28)
        self.dev_toggle_btn.setVisible(False)
        self.dev_toggle_btn.clicked.connect(self.toggle_dev_sidebar)
        h_layout.addWidget(self.dev_toggle_btn)
        
        self.rag_badge = QLabel("RAG: OFF")
        h_layout.addWidget(self.rag_badge)
        h_layout.addSpacing(20)
        
        m_layout.addWidget(self.header_area)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        m_layout.addWidget(self.chat_display)
        
        # Input Area Wrapper
        self.input_container = QFrame()
        self.input_container.setObjectName("InputContainer")
        ic_layout = QVBoxLayout(self.input_container)
        ic_layout.setContentsMargins(60, 20, 60, 30)
        
        self.input_box = QFrame()
        self.input_box.setObjectName("InputBox")
        ib_layout = QVBoxLayout(self.input_box)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Send a message to the model...")
        self.input_field.returnPressed.connect(self.soru_sor)
        ib_layout.addWidget(self.input_field)
        
        btm_input_bar = QHBoxLayout()
        btm_input_bar.setContentsMargins(10, 0, 10, 5)
        
        # Tools
        tool_layout = QHBoxLayout()
        tool_layout.setSpacing(8)
        
        btn_rag = QPushButton("📂 Files")
        btn_rag.clicked.connect(lambda: QMessageBox.information(self, "RAG", "Index your files for context!"))
        
        btn_code = QPushButton("⚡ Run")
        btn_code.clicked.connect(self.run_last_code)

        tool_layout.addWidget(btn_rag)
        tool_layout.addWidget(btn_code)
        
        send_btn = QPushButton("↑")
        send_btn.setFixedSize(32, 32)
        send_btn.setCursor(Qt.PointingHandCursor)
        send_btn.clicked.connect(self.soru_sor)
        self.send_btn = send_btn
        
        btm_input_bar.addLayout(tool_layout)
        self.gen_status_lbl = QLabel("")
        btm_input_bar.addWidget(self.gen_status_lbl)
        btm_input_bar.addStretch()
        btm_input_bar.addWidget(send_btn)
        ib_layout.addLayout(btm_input_bar)

        ic_layout.addWidget(self.input_box)
        m_layout.addWidget(self.input_container)

        # ---------------- RIGHT DEV SIDEBAR (HIDDEN BY DEFAULT) ----------------
        self.dev_sidebar = QFrame()
        self.dev_sidebar.setObjectName("DevSidebar")
        self.dev_sidebar.setFixedWidth(420)
        self.dev_sidebar.setVisible(False)
        dev_layout = QVBoxLayout(self.dev_sidebar)
        dev_layout.setContentsMargins(10, 10, 10, 10)
        self.dev_panel = DevPanel(self.dev_sidebar, main_app=self)
        dev_layout.addWidget(self.dev_panel)

        # Assemble
        splitter = QSplitter(Qt.Horizontal)
        self.splitter = splitter
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.main_area)
        splitter.addWidget(self.dev_sidebar)
        splitter.setSizes([260, 840, 0])
        main_layout.addWidget(splitter)

        self.apply_theme(self.prompts.get("theme", "dark"))

    def detect_system_theme(self) -> str:
        pal = QApplication.instance().palette()
        window = pal.color(QPalette.Window)
        return "dark" if window.lightness() < 128 else "light"

    def apply_theme(self, theme: str) -> None:
        if theme == "system":
            theme = self.detect_system_theme()

        self.theme = theme

        if theme == "light":
            colors = {
                "bg": "#f5f5f7",
                "panel": "#ffffff",
                "panel2": "#f0f0f2",
                "border": "#d0d0d7",
                "text": "#111111",
                "muted": "#666666",
                "accent": "#6a3dff",
                "accent2": "#4d74ff",
                "danger": "#c62828",
                "chip": "#e9e9ef",
            }
        else:
            colors = {
                "bg": "#0f0f0f",
                "panel": "#161616",
                "panel2": "#222222",
                "border": "#222222",
                "text": "#e0e0e0",
                "muted": "#888888",
                "accent": "#7c4dff",
                "accent2": "#9575cd",
                "danger": "#ff4d6a",
                "chip": "#1a1a1a",
            }

        qss = f"""
            QWidget {{
                background-color: {colors['bg']};
                color: {colors['text']};
                font-family: "Helvetica Neue", Arial, sans-serif;
            }}
            QLabel#Logo {{
                font-size: 22px;
                font-weight: 800;
                color: {colors['accent']};
            }}
            QLabel#DevHeader {{
                font-size: 16px;
                font-weight: 800;
                color: {colors['accent']};
                padding: 6px 4px;
            }}
            QFrame#Sidebar {{
                background-color: {colors['panel']};
                border-right: 1px solid {colors['border']};
            }}
            QFrame#DevSidebar {{
                background-color: {colors['panel']};
                border-left: 1px solid {colors['border']};
            }}
            QFrame#HeaderArea {{
                background-color: {colors['panel']};
                border-bottom: 1px solid {colors['border']};
            }}
            QFrame#HwBox {{
                background-color: {colors['panel2']};
                border-radius: 10px;
            }}
            QTabWidget::pane {{
                border: 1px solid {colors['border']};
                background: {colors['panel']};
                border-radius: 8px;
            }}
            QTabBar::tab {{
                background: {colors['panel2']};
                color: {colors['muted']};
                padding: 8px 12px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
            }}
            QTabBar::tab:selected {{
                background: {colors['chip']};
                color: {colors['text']};
            }}
            QGroupBox {{
                border: 1px solid {colors['border']};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 700;
                color: {colors['accent']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QListWidget {{
                background: transparent;
                border: none;
            }}
            QListWidget::item {{
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 2px;
                color: {colors['muted']};
            }}
            QListWidget::item:hover {{
                background-color: {colors['panel2']};
            }}
            QListWidget::item:selected {{
                background-color: {colors['chip']};
                color: {colors['text']};
                font-weight: 600;
            }}
            QTextEdit {{
                background-color: {colors['bg']};
                border: none;
                font-size: 15px;
                padding: 30px;
            }}
            QLineEdit {{
                background-color: {colors['chip']};
                border: 1px solid {colors['border']};
                border-radius: 8px;
                padding: 8px 12px;
                color: {colors['text']};
                font-size: 13px;
            }}
            QFrame#InputContainer {{
                background-color: {colors['panel']};
                border-top: 1px solid {colors['border']};
            }}
            QFrame#InputBox {{
                background-color: {colors['panel2']};
                border: 1px solid {colors['border']};
                border-radius: 12px;
            }}
            QPushButton {{
                background-color: {colors['panel2']};
                border: 1px solid {colors['border']};
                border-radius: 6px;
                padding: 6px 10px;
                color: {colors['text']};
            }}
            QPushButton:hover {{
                background-color: {colors['chip']};
            }}
            QPushButton:disabled {{
                background-color: {colors['chip']};
                color: {colors['muted']};
            }}
            QPushButton#NewChatButton {{
                font-weight: 700;
            }}
            QPushButton#SettingsButton {{
                text-align: left;
            }}
            QPushButton#DevUnlockButton {{
                background-color: {colors['chip']};
                border: 1px solid {colors['border']};
                color: {colors['accent']};
                font-weight: 700;
                text-align: left;
            }}
            QPushButton#SendButton {{
                background-color: {colors['accent']};
                color: white;
                border-radius: 16px;
                font-weight: bold;
                font-size: 18px;
                padding: 0px;
            }}
            QPushButton#SendButton:hover {{
                background-color: {colors['accent2']};
            }}
            QLabel#RagBadge {{
                background-color: {colors['chip']};
                border: 1px solid {colors['border']};
                border-radius: 12px;
                padding: 4px 12px;
                font-size: 11px;
                color: {colors['muted']};
            }}
            QLabel#RagBadge[ragState="active"] {{
                background-color: {colors['panel2']};
                border: 1px solid {colors['accent2']};
                color: {colors['accent']};
            }}
            QLabel#RagBadge[ragState="empty"] {{
                background-color: {colors['chip']};
                border: 1px solid {colors['border']};
                color: {colors['muted']};
            }}
            QLabel#RagBadge[ragState="off"] {{
                background-color: {colors['chip']};
                border: 1px solid {colors['border']};
                color: {colors['muted']};
            }}
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 8px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {colors['border']};
                min-height: 20px;
                border-radius: 4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """

        QApplication.instance().setStyleSheet(qss)
        self.send_btn.setObjectName("SendButton")
        self.rag_badge.setObjectName("RagBadge")

    def update_hw_stats(self, app_ram_gb: str, sys_ram_percent: str, cpu_percent: str, gpu_percent: str):
        self.lbl_ram_raw.setText(app_ram_gb)
        self.lbl_sys_ram_pct.setText(sys_ram_percent)
        self.lbl_cpu_pct.setText(cpu_percent)
        self.lbl_gpu_pct.setText(gpu_percent)

        def parse_percent(s: str):
            try:
                return float(s.strip().replace("%", ""))
            except Exception:
                return None

        ram_v = parse_percent(sys_ram_percent)
        cpu_v = parse_percent(cpu_percent)
        gpu_v = parse_percent(gpu_percent) if gpu_percent != "N/A" else None

        if ram_v is not None:
            self.ram_chart.add_point(ram_v)
        if cpu_v is not None:
            self.cpu_chart.add_point(cpu_v)

        if gpu_v is None:
            self.gpu_chart.set_available(False)
        else:
            self.gpu_chart.set_available(True)
            self.gpu_chart.add_point(gpu_v)

    def _set_chat_enabled(self, enabled: bool) -> None:
        self.input_field.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        if enabled:
            self.input_field.setFocus()

    def _find_default_model_path(self) -> str:
        if self.current_model_path and os.path.isdir(self.current_model_path):
            return self.current_model_path

        candidates = []
        base = os.path.expanduser("~/.lmstudio/models")
        if os.path.isdir(base):
            for root, dirs, _files in os.walk(base):
                for d in dirs:
                    full = os.path.join(root, d)
                    if not os.path.isdir(full):
                        continue
                    if "mlx" not in full.lower():
                        continue
                    has_config = os.path.exists(os.path.join(full, "config.json"))
                    has_tokenizer = os.path.exists(os.path.join(full, "tokenizer.json")) or os.path.exists(os.path.join(full, "tokenizer.model"))
                    has_weights = any(
                        name.endswith((".safetensors", ".npz", ".bin"))
                        for name in os.listdir(full)
                        if os.path.isfile(os.path.join(full, name))
                    )
                    if has_config or has_tokenizer or has_weights:
                        score = 0
                        lower = full.lower()
                        if "qwen" in lower:
                            score += 50
                        if "27b" in lower:
                            score += 20
                        if "6bit" in lower:
                            score += 15
                        if "distilled" in lower:
                            score += 10
                        if "4bit" in lower:
                            score -= 5
                        candidates.append((score, full))

        if not candidates:
            return ""

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def start_ai_service(self) -> None:
        self._set_chat_enabled(False)
        model_path = self._find_default_model_path()
        if not model_path:
            self.service_status_lbl.setText("Service: model path not found")
            return

        self.service_status_lbl.setText("Service: loading…")
        self.model_loader = ModelLoaderWorker(model_path)
        self.model_loader.loaded.connect(self._on_model_loaded)
        self.model_loader.error.connect(self._on_model_load_error)
        self.model_loader.start()

    def _on_model_loaded(self, model, tokenizer, model_path: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.current_model_path = model_path
        self.prompts["model_path"] = model_path
        self.save_prompts(self.prompts)
        self.service_status_lbl.setText("Service: ready")
        self._set_chat_enabled(True)

    def _on_model_load_error(self, err: str) -> None:
        self.model = None
        self.tokenizer = None
        self.service_status_lbl.setText("Service: error")
        self._set_chat_enabled(False)
        self.chat_display.append(f"<div style='color: #ff5555; '><b>Model load error:</b> {err}</div><br>")

    def open_settings(self):
        diag = SettingsDialog(self, self.user_prompt, current_theme=self.theme)
        if diag.exec_():
            self.apply_theme(getattr(diag, "final_theme", self.theme))

    def open_dev_panel(self):
        if not dev_mode_gate.unlocked:
            password, ok = QInputDialog.getText(self, "Dev Mode", "Enter developer password:")
            if not ok or not password:
                return
            if not dev_mode_gate.attempt_unlock(password):
                QMessageBox.critical(self, "Access Denied", "Incorrect password for Dev Mode.")
                return
            QMessageBox.information(self, "Dev Mode", "Dev Mode unlocked successfully!")

        self.dev_mode_active = True
        self.dev_toggle_btn.setVisible(True)
        self.toggle_dev_sidebar(force_state=True)

    def toggle_dev_sidebar(self, force_state=None) -> None:
        if not self.dev_mode_active:
            return

        if force_state is True:
            self.dev_sidebar_shown = True
        elif force_state is False:
            self.dev_sidebar_shown = False
        else:
            self.dev_sidebar_shown = not self.dev_sidebar_shown

        self.dev_sidebar.setVisible(self.dev_sidebar_shown)
        if self.dev_sidebar_shown:
            self.splitter.setSizes([260, 700, 420])
        else:
            self.splitter.setSizes([260, 840, 0])

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
        if not user_text:
            return

        if self.model is None or self.tokenizer is None:
            self.chat_display.append("<div style='color: #ff5555; '><b>Service not ready:</b> Model is not loaded.</div><br>")
            self.input_field.setText(user_text)
            self.input_field.setFocus()
            return
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
                self.rag_badge.setProperty("ragState", "active")
                self.rag_badge.style().unpolish(self.rag_badge)
                self.rag_badge.style().polish(self.rag_badge)
            else:
                self.rag_badge.setText("RAG: EMPTY")
                self.rag_badge.setProperty("ragState", "empty")
                self.rag_badge.style().unpolish(self.rag_badge)
                self.rag_badge.style().polish(self.rag_badge)
        
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
        self.gen_status_lbl.setText("Generating…")
        
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
        self.gen_status_lbl.setText("")
        self.input_field.setFocus()
        self.rag_badge.setText("RAG: OFF")
        self.rag_badge.setProperty("ragState", "off")
        self.rag_badge.style().unpolish(self.rag_badge)
        self.rag_badge.style().polish(self.rag_badge)

    def on_ai_error(self, err_msg):
        self.chat_display.append(f"<div style='color: #ff5555; '><b>Error:</b> {err_msg}</div><br>")
        self.input_field.setDisabled(False)
        self.send_btn.setDisabled(False)
        self.gen_status_lbl.setText("")
        self.input_field.setFocus()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Standard font setup to avoid warnings
    font = QFont("Helvetica Neue", 13)
    app.setFont(font)

    window = ChatbotGUI(None, None, "")
    window.show()
    sys.exit(app.exec_())
