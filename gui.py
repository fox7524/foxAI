import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QHBoxLayout, QLabel, QSplitter,
    QDialog, QFormLayout, QMessageBox, QRadioButton,
    QButtonGroup, QStackedWidget, QMenu, QInputDialog,
    QSplashScreen, QProgressBar, QFrame, QListWidget, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from mlx_lm import load, generate

# Import our new engines
try:
    from rag_engine import RAGEngine
    from finetune_engine import FinetuneEngine
except ImportError:
    pass # Wait to instantiate or mock if missing

VERSION = "Thunderbird AI Volume Alpha - DEV Edition"

# ---------------------------------------------------------
# WORKER THREADS
# ---------------------------------------------------------

class AIWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, model, tokenizer, prompt):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt

    def run(self):
        try:
            response = generate(self.model, self.tokenizer, prompt=self.prompt, max_tokens=1500)
            self.finished.emit(response)
        except Exception as e:
            self.finished.emit(f"Error generating response: {str(e)}")

class TaskWorker(QThread):
    """Generic worker for background tasks like RAG Ingestion or data prep"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.task_func(*self.args, **self.kwargs)
        except Exception as e:
            self.log_signal.emit(f"Error during task: {str(e)}")
        finally:
            self.finished_signal.emit()

# ---------------------------------------------------------
# MAIN UI CLASS
# ---------------------------------------------------------
class ChatbotGUI(QWidget):
    def __init__(self, model, tokenizer, model_path):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        
        # Initialize Engines
        try:
            self.rag_engine = RAGEngine()
        except:
            self.rag_engine = None
        self.finetune_engine = FinetuneEngine(self.model_path)
        
        self.rag_enabled = True # Toggle for using RAG context

        self.system_instruction = "You are a highly intelligent and helpful Assistant named FoxAI developed by Kayra and Ahmet. You answer questions concisely and accurately."
        self.messages = [{"role": "system", "content": self.system_instruction}]
        
        self.selected_files = [] # Files chosen in dev mode

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(VERSION)
        self.setGeometry(100, 100, 1000, 650)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
        """)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---------------- SIDEBAR ----------------
        sidebar = QFrame()
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border-radius: 0px;
                padding: 20px;
            }
            QLabel { color: #ecf0f1; }
        """)
        sidebar.setFixedWidth(240)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(15)

        logo = QLabel("FoxAI")
        logo.setFont(QFont("SF Pro Display", 22, QFont.Bold))
        sidebar_layout.addWidget(logo, alignment=Qt.AlignTop | Qt.AlignHCenter)

        # Navigation Buttons
        self.btn_chat_mode = QPushButton("💬 Chat Mode")
        self.btn_dev_mode = QPushButton("⚙️ Developer Mode")
        
        for btn in [self.btn_chat_mode, self.btn_dev_mode]:
            self.style_nav_button(btn)
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()

        self.exit_button = QPushButton("🚪 Exit")
        self.style_nav_button(self.exit_button, hover_color="#c0392b")
        self.exit_button.clicked.connect(self.close)
        sidebar_layout.addWidget(self.exit_button)

        # ---------------- MAIN CONTENT AREA ----------------
        self.stack = QStackedWidget()
        
        # 1. Chat Page
        self.chat_page = QWidget()
        self.setup_chat_page()
        self.stack.addWidget(self.chat_page)

        # 2. Dev Page
        self.dev_page = QWidget()
        self.setup_dev_page()
        self.stack.addWidget(self.dev_page)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.stack)

        # Connections
        self.btn_chat_mode.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.btn_dev_mode.clicked.connect(lambda: self.stack.setCurrentIndex(1))

    def style_nav_button(self, button, hover_color="#34495e"):
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: #ecf0f1;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """)

    def style_action_button(self, button, bg_color="#3498db"):
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {bg_color}aa;
            }}
        """)

    # ---------------- CHAT PAGE ----------------
    def setup_chat_page(self):
        layout = QVBoxLayout(self.chat_page)
        
        chat_area = QFrame()
        chat_area.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 12px;
            }
        """)
        chat_layout = QVBoxLayout(chat_area)
        
        # Top bar in chat
        top_bar = QHBoxLayout()
        header = QLabel("Chat Dashboard")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setStyleSheet("color: #2c3e50;")
        
        self.status_lbl = QLabel("")
        
        top_bar.addWidget(header)
        top_bar.addStretch()
        top_bar.addWidget(self.status_lbl)
        
        chat_layout.addLayout(top_bar)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #fafafa;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
                padding: 15px;
                font-size: 14px;
            }
        """)
        chat_layout.addWidget(self.chat_display)

        # Input Area
        input_frame = QFrame()
        input_frame.setStyleSheet("QFrame { background-color: #ffffff; }")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(0,0,0,0)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a message...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                border: 2px solid #e0e0e0;
                border-radius: 20px;
                padding: 10px 15px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
            }
        """)
        self.input_field.returnPressed.connect(self.soru_sor)
        
        send_btn = QPushButton("Send")
        send_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border-radius: 20px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #27ae60; }
        """)
        send_btn.clicked.connect(self.soru_sor)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(send_btn)

        chat_layout.addWidget(input_frame)
        layout.addWidget(chat_area)

    def soru_sor(self):
        user_text = self.input_field.text().strip()
        if not user_text: return
        
        self.input_field.clear()
        self.input_field.setDisabled(True)
        self.chat_display.append(f"<div style='margin: 5px 0;'><b style='color:#3498db;'>You:</b> {user_text}</div>")
        
        # Handle RAG Context insertion temporarily into the prompt
        rag_context = ""
        if self.rag_enabled and self.rag_engine and self.rag_engine.enabled:
            self.status_lbl.setText("<i>Searching knowledge base...</i>")
            rag_context = self.rag_engine.query(user_text)

        # Build messages for this specific prompt
        temp_messages = list(self.messages)
        
        augmented_prompt = user_text
        if rag_context:
            augmented_prompt = f"Using the following retrieved context, answer the user's question.\n\nContext: {rag_context}\n\nQuestion: {user_text}"
            self.chat_display.append("<br><i><small style='color:gray;'>* Context retrieved from RAG db.</small></i>")

        temp_messages.append({"role": "user", "content": augmented_prompt})
        
        # Add to actual history without the massive context
        self.messages.append({"role": "user", "content": user_text})
        
        self.status_lbl.setText("<i>Thinking...</i>")
        
        prompt = self.tokenizer.apply_chat_template(temp_messages, tokenize=False, add_generation_prompt=True)
        
        self.worker = AIWorker(self.model, self.tokenizer, prompt)
        self.worker.finished.connect(self.on_ai_response)
        self.worker.start()

    def on_ai_response(self, response):
        self.messages.append({"role": "assistant", "content": response})
        formatted = response.replace('\n', '<br>')
        self.chat_display.append(f"<div style='margin: 5px 0; background:#f0f3f4; padding:10px; border-radius:8px;'><b style='color:#e67e22;'>FoxAI:</b><br>{formatted}</div>")
        self.chat_display.append("<br>")
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())
        
        self.status_lbl.setText("")
        self.input_field.setDisabled(False)
        self.input_field.setFocus()


    # ---------------- DEV PAGE ----------------
    def setup_dev_page(self):
        layout = QHBoxLayout(self.dev_page)
        
        # Left Panel (Controls)
        left_panel = QFrame()
        left_panel.setStyleSheet("background-color: #ffffff; border-radius: 12px;")
        l_layout = QVBoxLayout(left_panel)
        
        header = QLabel("AI Training & Data Console")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        l_layout.addWidget(header)
        
        l_layout.addWidget(QLabel("1. Select Documents (PDF, DOCX, C++, PY, ZIM, etc.)"))
        self.btn_select_files = QPushButton("Browse Files")
        self.style_action_button(self.btn_select_files, "#8e44ad")
        self.btn_select_files.clicked.connect(self.select_files)
        l_layout.addWidget(self.btn_select_files)
        
        self.file_list = QListWidget()
        self.file_list.setStyleSheet("background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 4px;")
        l_layout.addWidget(self.file_list)
        
        l_layout.addWidget(QLabel("2. RAG Upload (Vectorize Docs into ChromaDB)"))
        self.btn_rag = QPushButton("Ingest to RAG")
        self.style_action_button(self.btn_rag, "#27ae60")
        self.btn_rag.clicked.connect(self.ingest_rag)
        l_layout.addWidget(self.btn_rag)
        
        self.btn_clear_rag = QPushButton("Reset RAG Database")
        self.style_action_button(self.btn_clear_rag, "#e74c3c")
        self.btn_clear_rag.clicked.connect(self.reset_rag)
        l_layout.addWidget(self.btn_clear_rag)
        
        l_layout.addSpacing(20)
        
        l_layout.addWidget(QLabel("3. Fine-Tune (Train Model with Files)"))
        self.btn_finetune = QPushButton("Start LoRA Fine-Tuning")
        self.style_action_button(self.btn_finetune, "#d35400")
        self.btn_finetune.clicked.connect(self.start_finetuning)
        l_layout.addWidget(self.btn_finetune)

        # Right Panel (Console Output)
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: #1e1e1e; border-radius: 12px;")
        r_layout = QVBoxLayout(right_panel)
        
        console_lbl = QLabel("Developer Console")
        console_lbl.setStyleSheet("color: #00ff00; font-weight: bold;")
        r_layout.addWidget(console_lbl)
        
        self.dev_console = QTextEdit()
        self.dev_console.setReadOnly(True)
        self.dev_console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; border: none; font-family: monospace;")
        r_layout.addWidget(self.dev_console)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        layout.addWidget(splitter)
        
    def log_dev(self, text):
        self.dev_console.append(f"> {text}")
        self.dev_console.verticalScrollBar().setValue(self.dev_console.verticalScrollBar().maximum())

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Documents", "", "All Files (*);;PDFs (*.pdf);;Code (*.py *.cpp *.c);;Docs (*.docx *.txt)")
        if files:
            for f in files:
                if f not in self.selected_files:
                    self.selected_files.append(f)
                    self.file_list.addItem(os.path.basename(f))
            self.log_dev(f"Selected {len(files)} new files.")

    def ingest_rag(self):
        if not self.selected_files:
            self.log_dev("ERROR: No files selected.")
            return
        if not self.rag_engine or not self.rag_engine.enabled:
            self.log_dev("ERROR: RAG Engine not initialized. Missing dependencies?")
            return
            
        self.log_dev("Starting RAG ingestion in background...")
        self.btn_rag.setDisabled(True)
        
        def run_ingest():
            success = self.rag_engine.ingest_documents(self.selected_files)
            if success:
                self.tworker.log_signal.emit("RAG Ingestion completed successfully.")
            else:
                self.tworker.log_signal.emit("RAG Ingestion encountered errors or yielded no data.")

        self.tworker = TaskWorker(run_ingest)
        self.tworker.log_signal.connect(self.log_dev)
        self.tworker.finished_signal.connect(lambda: self.btn_rag.setDisabled(False))
        self.tworker.start()

    def reset_rag(self):
        if self.rag_engine:
            self.rag_engine.reset_database()
            self.log_dev("RAG Database has been totally reset.")

    def start_finetuning(self):
        if not self.selected_files:
            self.log_dev("ERROR: No files selected to train on.")
            return
        
        self.log_dev("Starting background data preparation for Fine-Tuning...")
        self.btn_finetune.setDisabled(True)

        def run_ft():
            # Mock extraction, ideally we use generic loaders from rag_engine
            # to extract text out of the files.
            chunks = []
            if self.rag_engine and self.rag_engine.enabled:
                for f in self.selected_files:
                    try:
                        self.tworker.log_signal.emit(f"Extracting text from {f}")
                        docs = self.rag_engine.process_file(f)
                        chunks.extend([d.page_content for d in docs])
                    except:
                        pass
                        
            if not chunks:
                self.tworker.log_signal.emit("No valid text extracted to train on.")
                return
                
            train_p, valid_p = self.finetune_engine.prepare_dataset(chunks)
            self.tworker.log_signal.emit(f"Dataset generated at {train_p} and {valid_p}")
            self.tworker.log_signal.emit("Launching subprocess for mlx_lm.lora...")
            
            # This starts the subprocess. In a real desktop app we'd attach a QProcess 
            # to continuously read stdout, but for now we note it launched.
            proc = self.finetune_engine.start_training()
            self.tworker.log_signal.emit(f"Fine-Tuning started with PID {proc.pid}. Watch terminal for details.")
            
        self.tworker = TaskWorker(run_ft)
        self.tworker.log_signal.connect(self.log_dev)
        self.tworker.finished_signal.connect(lambda: self.btn_finetune.setDisabled(False))
        self.tworker.start()


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Simple splash
    splash = QSplashScreen()
    splash.setFixedSize(400, 200)
    splash.setWindowFlags(Qt.FramelessWindowHint)   
    splash.setStyleSheet("background-color: #2c3e50; color: white;")
    
    layout = QVBoxLayout(splash)
    v_lbl = QLabel(VERSION)
    v_lbl.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
    v_lbl.setStyleSheet("font-size: 16pt; font-weight: bold;")
    layout.addWidget(v_lbl)
    
    s_lbl = QLabel("Loading ML Model (this takes a moment)...")
    s_lbl.setAlignment(Qt.AlignCenter)
    layout.addWidget(s_lbl)
    
    splash.setLayout(layout)
    splash.show()
    app.processEvents()

    model_path = "/Users/fox/.lmstudio/models/Jackrong/MLX-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-6bit"
    
    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Make mock for UI testing if user doesn't have model running currently
        class MockObj:
            def apply_chat_template(self, *args, **kwargs): return ""
        model, tokenizer = None, MockObj()

    splash.close()
    
    window = ChatbotGUI(model, tokenizer, model_path)
    window.show()
    
    sys.exit(app.exec_())