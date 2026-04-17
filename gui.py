import sys
import os
import time
import psutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, 
    QPushButton, QHBoxLayout, QLabel, QSplitter, QDialog, 
    QFormLayout, QMessageBox, QRadioButton, QButtonGroup, 
    QStackedWidget, QListWidget, QFrame, QScrollArea, QFileDialog,
    QInputDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QTextCursor

from mlx_lm import load, generate, stream_generate

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
    """Handles text generation with token streaming."""
    new_token = pyqtSignal(str)
    finished = pyqtSignal(str, float, int, float)
    error = pyqtSignal(str)

    def __init__(self, model, tokenizer, prompt):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.is_running = True

    def run(self):
        try:
            start_time = time.time()
            full_response = ""
            token_count = 0
            
            for response in stream_generate(self.model, self.tokenizer, prompt=self.prompt, max_tokens=1500):
                if not self.is_running:
                    break
                
                # In mlx_lm.stream_generate, the response is typically the accumulated text
                # We want just the new part
                new_part = response[len(full_response):]
                full_response = response
                token_count += 1
                self.new_token.emit(new_part)
            
            end_time = time.time()
            elapsed = end_time - start_time
            tok_per_sec = token_count / elapsed if elapsed > 0 else 0.0

            self.finished.emit(full_response, tok_per_sec, token_count, elapsed)
        except Exception as e:
            self.error.emit(f"Error generating response: {str(e)}")

    def stop(self):
        self.is_running = False

class MemoryMonitor(QThread):
    update_signal = pyqtSignal(str, str) # ram_gb, sys_load
    
    def run(self):
        process = psutil.Process(os.getpid())
        while True:
            try:
                # App memory usage
                mem_info = process.memory_info()
                used_gb = mem_info.rss / (1024 ** 3)
                
                # System overall CPU percentage
                cpu_percent = psutil.cpu_percent(interval=None)
                
                self.update_signal.emit(f"{used_gb:.2f} GB", f"{cpu_percent}%")
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
        
        self.system_prompt = """You are FoxAI, a local expert AI pair-programmer. 
Your core rule is: **ASK BEFORE ACTING**.

Before writing any code or providing a solution, you MUST:
1. List EVERY unclear point or assumption in the user's request.
2. Ask the user to clarify these points one by one.
3. DO NOT write a single line of code until all questions are answered and the requirements are 100% clear.

Style rules:
- Write production-grade, PEP8-compliant Python code.
- Use type hints for all functions.
- Be concise but thorough in your explanations.
"""
        
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