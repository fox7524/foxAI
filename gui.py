import sys
import os
import time
import psutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, 
    QPushButton, QHBoxLayout, QLabel, QSplitter, QDialog, 
    QFormLayout, QMessageBox, QRadioButton, QButtonGroup, 
    QStackedWidget, QListWidget, QFrame, QScrollArea, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

from mlx_lm import load, generate

try:
    from rag_engine import RAGEngine
    from finetune_engine import FinetuneEngine
except ImportError:
    pass

VERSION = "FoxAI - Studio Edition"

# ---------------------------------------------------------
# WORKERS
# ---------------------------------------------------------
class AIWorker(QThread):
    """Handles text generation and calculates performance metrics."""
    # signal returns: (response_text, tok_per_sec, total_tokens, time_elapsed)
    finished = pyqtSignal(str, float, int, float)
    error = pyqtSignal(str)

    def __init__(self, model, tokenizer, prompt):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt

    def run(self):
        try:
            start_time = time.time()
            response = generate(self.model, self.tokenizer, prompt=self.prompt, max_tokens=1500)
            end_time = time.time()
            
            elapsed = end_time - start_time
            # Rough token count metric via tokenizer length
            token_count = len(self.tokenizer.encode(response)) if hasattr(self.tokenizer, 'encode') else len(response.split())
            tok_per_sec = token_count / elapsed if elapsed > 0 else 0.0

            self.finished.emit(response, tok_per_sec, token_count, elapsed)
        except Exception as e:
            self.error.emit(f"Error generating response: {str(e)}")

class MemoryMonitor(QThread):
    update_signal = pyqtSignal(str, str) # ram_gb, ram_percent
    
    def run(self):
        process = psutil.Process(os.getpid())
        while True:
            try:
                # App memory usage
                mem_info = process.memory_info()
                used_gb = mem_info.rss / (1024 ** 3)
                
                # System overall RAM percentage
                sys_mem = psutil.virtual_memory()
                sys_percent = sys_mem.percent
                
                self.update_signal.emit(f"{used_gb:.2f} GB", f"{sys_percent}%")
            except:
                pass
            time.sleep(2)

# ---------------------------------------------------------
# SETTINGS & ROADMAP VIEWER
# ---------------------------------------------------------
class SettingsDialog(QDialog):
    def __init__(self, parent=None, current_prompt="", current_theme="dark"):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(500, 450)
        self.setStyleSheet("background-color: #2c2c2c; color: white;")
        
        self.final_prompt = current_prompt
        self.final_theme = current_theme
        
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("<b>System / Personality Prompt:</b>"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText(current_prompt)
        self.prompt_edit.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444; padding: 5px;")
        layout.addWidget(self.prompt_edit)
        
        layout.addWidget(QLabel("<b>Theme Setup (Mock):</b>"))
        theme_layout = QHBoxLayout()
        self.rb_dark = QRadioButton("Dark Theme (Studio)")
        self.rb_light = QRadioButton("Light ")
        
        if current_theme == "dark": self.rb_dark.setChecked(True)
        else: self.rb_light.setChecked(True)
            
        theme_layout.addWidget(self.rb_dark)
        theme_layout.addWidget(self.rb_light)
        theme_layout.addStretch()
        layout.addLayout(theme_layout)
        
        layout.addSpacing(20)
        layout.addWidget(QLabel("<b>Roadmap & Architecture References:</b>"))
        roadmap_btn = QPushButton("View Original Roadmap Tabs")
        roadmap_btn.setStyleSheet("background-color: #3f51b5; padding: 8px; border-radius: 4px; font-weight: bold;")
        roadmap_btn.clicked.connect(self.show_mock_roadmap)
        layout.addWidget(roadmap_btn)
        
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save & Apply")
        save_btn.setStyleSheet("background-color: #2ecc71; padding: 8px; border-radius: 4px; font-weight: bold;")
        save_btn.clicked.connect(self.accept_settings)
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def accept_settings(self):
        self.final_prompt = self.prompt_edit.toPlainText()
        self.final_theme = "dark" if self.rb_dark.isChecked() else "light"
        self.accept()

    def show_mock_roadmap(self):
        QMessageBox.information(self, "Roadmap Specs", "Phase 1: Ask Before Acting (Ahmet trains 50 pairs)\nPhase 2: RAG / Finetuning Core\nPhase 3: GUI Dev\nPhase 4: Optimization")

# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
class ChatbotGUI(QWidget):
    def __init__(self, model, tokenizer, model_path):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
        # Engines
        try:
            self.rag_engine = RAGEngine()
        except:
            self.rag_engine = None
        
        self.system_prompt = "You are FoxAI, a local assistant."
        
        # Chat Sessions Storage Mock
        self.chats = {"Default Chat": [{"role": "system", "content": self.system_prompt}]}
        self.active_chat = "Default Chat"
        
        self.init_ui()
        
        # Start HW Monitor
        self.mem_thread = MemoryMonitor()
        self.mem_thread.update_signal.connect(self.update_hw_stats)
        self.mem_thread.start()

    def init_ui(self):
        self.setWindowTitle(VERSION)
        self.setGeometry(100, 100, 1100, 750)
        self.setStyleSheet("""
            QWidget { background-color: #1a1a1a; color: #d4d4d4; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto; }
        """)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---------------- LEFT SIDEBAR (CHATS) ----------------
        sidebar = QFrame()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("background-color: #242424; border-right: 1px solid #333;")
        s_layout = QVBoxLayout(sidebar)
        
        # Header
        top_bar = QHBoxLayout()
        logo = QLabel("👾 Chats")
        logo.setFont(QFont("Arial", 14, QFont.Bold))
        new_chat_btn = QPushButton("📝")
        new_chat_btn.setFixedSize(30, 30)
        new_chat_btn.setStyleSheet("border: none; background: transparent; font-size: 16px;")
        new_chat_btn.clicked.connect(self.new_chat)
        
        top_bar.addWidget(logo)
        top_bar.addStretch()
        top_bar.addWidget(new_chat_btn)
        s_layout.addLayout(top_bar)
        
        # Search
        search_bar = QLineEdit()
        search_bar.setPlaceholderText("Search chats...")
        search_bar.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444; border-radius: 4px; padding: 6px;")
        s_layout.addWidget(search_bar)
        
        # Session List
        self.chat_list = QListWidget()
        self.chat_list.setStyleSheet("""
            QListWidget { background: transparent; border: none; }
            QListWidget::item { padding: 10px; border-radius: 6px; }
            QListWidget::item:selected { background-color: #3b3b3b; color: white; }
        """)
        self.chat_list.addItem(self.active_chat)
        self.chat_list.itemClicked.connect(self.switch_chat)
        s_layout.addWidget(self.chat_list)
        
        s_layout.addStretch()
        
        # Hardware Status Bottom
        hw_layout = QHBoxLayout()
        self.lbl_ram_raw = QLabel("💾 0 GB")
        self.lbl_ram_raw.setStyleSheet("font-size: 11px; color: #888; background: #1e1e1e; padding: 4px; border-radius:4px;")
        self.lbl_ram_pct = QLabel("⚙ 0.0%")
        self.lbl_ram_pct.setStyleSheet("font-size: 11px; color: #888; background: #1e1e1e; padding: 4px; border-radius:4px;")
        
        settings_btn = QPushButton("⚙")
        settings_btn.setStyleSheet("background: transparent; color: #888; border: none; font-size: 16px;")
        settings_btn.clicked.connect(self.open_settings)
        
        hw_layout.addWidget(settings_btn)
        hw_layout.addStretch()
        hw_layout.addWidget(self.lbl_ram_raw)
        hw_layout.addWidget(self.lbl_ram_pct)
        s_layout.addLayout(hw_layout)

        # ---------------- RIGHT MAIN AREA ----------------
        main_area = QFrame()
        m_layout = QVBoxLayout(main_area)
        
        # Header Info
        header_area = QHBoxLayout()
        self.model_lbl = QLabel("🤖 <b>Active Model:</b> mlx_lm.load() Weights | FoxAI Core")
        self.model_lbl.setAlignment(Qt.AlignCenter)
        self.model_lbl.setStyleSheet("background-color: #2a2a2a; padding: 8px; border-radius: 6px; color: #bbb;")
        header_area.addStretch()
        header_area.addWidget(self.model_lbl)
        header_area.addStretch()
        m_layout.addLayout(header_area)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                border: none;
                font-size: 15px;
                padding: 20px;
                line-height: 1.5;
            }
        """)
        m_layout.addWidget(self.chat_display)
        
        # Input Area Wrapper
        input_container = QFrame()
        input_container.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333; border-radius: 12px; margin: 10px 40px;")
        ic_layout = QVBoxLayout(input_container)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Send a message to the model...")
        self.input_field.setStyleSheet("background: transparent; border: none; font-size: 14px; padding: 5px;")
        self.input_field.returnPressed.connect(self.soru_sor)
        ic_layout.addWidget(self.input_field)
        
        btm_input_bar = QHBoxLayout()
        # Mock buttons for Dev Panel / Tools
        btn_dev = QPushButton("🔧 RAG / Train Tools")
        btn_dev.setStyleSheet("background-color: #2b3b55; color: #a4c2f4; border-radius: 6px; padding: 4px 8px; font-size: 12px;")
        btn_dev.clicked.connect(lambda: QMessageBox.information(self, "Dev Panel", "Opens RAG/Fine-tune manager!"))
        
        send_btn = QPushButton("↑")
        send_btn.setFixedSize(28, 28)
        send_btn.setStyleSheet("background-color: #444; color: white; border-radius: 14px; font-weight: bold;")
        send_btn.clicked.connect(self.soru_sor)
        
        btm_input_bar.addWidget(btn_dev)
        btm_input_bar.addStretch()
        btm_input_bar.addWidget(send_btn)
        ic_layout.addLayout(btm_input_bar)

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
        diag = SettingsDialog(self, self.system_prompt)
        if diag.exec_():
            self.system_prompt = diag.final_prompt
            # Update history of active chat to adhere to new prompt
            if self.chats[self.active_chat] and self.chats[self.active_chat][0]["role"] == "system":
                self.chats[self.active_chat][0]["content"] = self.system_prompt

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
                self.chat_display.append(f"<div style='margin-bottom: 10px; color: #ececec; font-size: 16px;'><b>You</b><br><span style='color: #a0a0a0;'>{msg['content']}</span></div>")
            elif msg["role"] == "assistant":
                self.chat_display.append(f"<div style='margin-bottom: 10px; color: #ececec; font-size: 16px;'><b>FoxAI</b><br>{msg['content'].replace(chr(10), '<br>')}</div><br>")
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def soru_sor(self):
        user_text = self.input_field.text().strip()
        if not user_text: return
        self.input_field.clear()
        
        # Display user bubble
        self.chat_display.append(f"<div style='margin-bottom: 10px; color: #ececec; font-size: 16px;'><b>You</b><br><span style='color: #a0a0a0;'>{user_text}</span></div>")
        
        # History appends
        self.chats[self.active_chat].append({"role": "user", "content": user_text})
        
        # Context formulation
        context_prompt = user_text
        if self.rag_engine and self.rag_engine.enabled:
            rag_docs = self.rag_engine.query(user_text)
            if rag_docs:
                context_prompt = f"Background info:\n{rag_docs}\n\nUser: {user_text}"
                
        temp_history = list(self.chats[self.active_chat])
        temp_history[-1] = {"role": "user", "content": context_prompt}
        
        try:
            # We mock the prompt string since it handles the backend logic cleanly 
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt_string = self.tokenizer.apply_chat_template(temp_history, tokenize=False, add_generation_prompt=True)
            else: prompt_string = f"User: {context_prompt}\nAssistant: "
        except:
            prompt_string = f"User: {context_prompt}\nAssistant: "

        self.input_field.setDisabled(True)
        self.worker = AIWorker(self.model, self.tokenizer, prompt_string)
        self.worker.finished.connect(self.on_ai_success)
        self.worker.error.connect(self.on_ai_error)
        self.worker.start()

    def on_ai_success(self, response, tps, tokens, ms):
        self.chats[self.active_chat].append({"role": "assistant", "content": response})
        formatted = response.replace('\n', '<br>')
        
        # Display response
        self.chat_display.append(f"<div style='color: #ececec; font-size: 16px;'><b>FoxAI</b><br>{formatted}</div>")
        
        # Display LM-studio style metadata
        meta_html = f"""
        <div style='background-color: #242424; color: #888; font-size: 11px; padding: 5px; border-radius: 4px; display: inline-block; margin-top: 8px;'>
            💡 <span>⏱ {tps:.2f} tok/sec</span> | <span>🧱 {tokens} tokens</span> | <span>🕒 {ms:.2f}s</span> | <b>Stop reason: EOS Token Found</b>
        </div><br><br>
        """
        self.chat_display.append(meta_html)
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())
        
        self.input_field.setDisabled(False)
        self.input_field.setFocus()

    def on_ai_error(self, err_msg):
        self.chat_display.append(f"<div style='color: #ff5555; '><b>Error:</b> {err_msg}</div><br>")
        self.input_field.setDisabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Simple ML extraction boot
    try:
        model_path = "/Users/fox/.lmstudio/models/Jackrong/MLX-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-6bit"
        model, tokenizer = load(model_path)
    except:
        model, tokenizer = None, None
        
    window = ChatbotGUI(model, tokenizer, "")
    window.show()
    sys.exit(app.exec_())