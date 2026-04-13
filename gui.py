import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QHBoxLayout, QLabel, QSplitter,
    QDialog, QFormLayout, QMessageBox, QRadioButton,
    QButtonGroup, QStackedLayout, QMenu, QInputDialog,
    QSplashScreen, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from mlx_lm import load, generate

VERSION = "Thunderbird AI Volume Alpha"

# ---------------------------------------------------------
# AI WORKER THREAD (Prevents UI Freezing)
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
            # Run the heavy generation process in the background
            response = generate(self.model, self.tokenizer, prompt=self.prompt, max_tokens=1500)
            self.finished.emit(response)
        except Exception as e:
            self.finished.emit(f"Error generating response: {str(e)}")

# ---------------------------------------------------------
# MAIN UI CLASS
# ---------------------------------------------------------
class ChatbotGUI(QWidget):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
        # Load your giant system instruction
        self.system_instruction = """You are a smart and helpful assistant. You give direct and logical answers to questions. You are a helpful and precise assistant for answering questions. Your name is FoxAI. You are developed by Kayra (Fox) and Ahmet (Callisto). You are a virtual assistant designed to help users with their questions and tasks. You are knowledgeable in various topics and can provide accurate and concise information. You are a senior software developer with expertise in Python and AI technologies. You have experience working on various projects, including web development, machine learning, and natural language processing. You are passionate about coding and enjoy solving complex problems. You remember the conversation history and use it to provide contextually relevant responses. You are a senior Developer for ultra complex algorithms and also an ultra complex, high priority code builder for other developers. You help developers with understanding what they actually want. You never move on without clarifying the exact situation for the code-error-algorithm. You are meant to be helpful and useful. You ask before moving on. You ask every unclear point to the user; when all the question marks are clarified, you can start assisting them with the error/code/algorithm. You are a smart, helpful, mindful, and understanding AI agent for firms, companies, developers, and vibecoders. You always try to put the user in the work, not do all the work on your own, so the user understands the logic for his/her code basics. Most importantly, the user learns that ability to use the function or whatever you teach to them, and the user needs to learn that skill. You need to push the user to learn new skills and things. You explain and teach the user if the user asks you to explain how to use that function, what that system does, what that algorithm does, why we use this, etc."""
        
        # Initialize conversation history
        self.messages = [{"role": "system", "content": self.system_instruction}]

        self.layout_yuklendi = False
        self.setWindowTitle(VERSION)
        self.setGeometry(100, 100, 800, 500)
        
        self.conversations = []
        self.max_conversations = 5
        self.current_theme = "system"
        
        self.init_ui()

    def style_button(self, button, color="#007ACC", text_color=None):
        cls_name = button.__class__.__name__
        if cls_name == 'QRadioButton':
            if text_color is None:
                text_color = "#FFFFFF" if self.current_theme == 'dark' else "#000000"
            button.setStyleSheet(f"QRadioButton {{ color: {text_color}; background-color: transparent; font-weight: bold; }}")
            return

        if text_color is None:
            text_color = "white"
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

        header_layout.setAlignment(Qt.AlignTop)
        self.sidebar.addLayout(header_layout)

        self.conversation_buttons_layout = QVBoxLayout()
        self.sidebar.addLayout(self.conversation_buttons_layout)
        self.sidebar.addStretch()

        self.exit_button = QPushButton("Exit")
        self.style_button(self.exit_button)
        self.exit_button.clicked.connect(self.close)
        self.sidebar.addWidget(self.exit_button)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(self.sidebar)
        sidebar_widget.setFixedWidth(160)

        self.chat_layout = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Mesajınızı yazın...")
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
        self.hamburger_button.clicked.connect(lambda: sidebar_widget.setVisible(not sidebar_widget.isVisible()))
        layout.addWidget(self.hamburger_button)

        self.setLayout(layout)
        self.layout_yuklendi = True
        self.new_conversation()

    def soru_sor(self):
        soru = self.input_field.text().strip()
        if not soru: return
        
        # 1. Update History & UI
        self.messages.append({"role": "user", "content": soru})
        self.chat_display.append(f"<div style='border:1px solid #ccc; padding:5px; border-radius:8px;'><b>SEN:</b> {soru}</div>")
        
        # 2. Lock input while thinking
        self.input_field.clear()
        self.input_field.setDisabled(True)
        self.suggestion_label.setText("<i>FoxAI is thinking...</i>")
        
        # 3. Format Prompt
        prompt = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        
        # 4. Fire up the background thread
        self.worker = AIWorker(self.model, self.tokenizer, prompt)
        self.worker.finished.connect(self.on_ai_response)
        self.worker.start()

    def on_ai_response(self, response):
        # 1. Save response to history
        self.messages.append({"role": "assistant", "content": response})
        
        # 2. Format linebreaks for HTML display in QTextEdit
        formatted_response = response.replace('\n', '<br>')
        
        # 3. Update UI
        self.chat_display.append(f"<div style='border:1px solid #3498db; padding:5px; border-radius:8px; margin-top:5px; background-color: #ecf0f1; color: #2c3e50;'><b>BIRD AI:</b><br>{formatted_response}</div>")
        
        # 4. Unlock input
        self.suggestion_label.setText("")
        self.input_field.setDisabled(False)
        self.input_field.setFocus()

    def new_conversation(self):
        if len(self.conversations) >= self.max_conversations:
            QMessageBox.warning(self, "Limit", "Maksimum 5 sohbet oluşturabilirsiniz.")
            return

        title = f"Sohbet {len(self.conversations)+1}"
        self.conversations.append(title)
        
        btn = QPushButton(title)
        self.style_button(btn)

        def load_conv():
            self.chat_display.clear()
            self.chat_display.append(f"<i>--- {title} yüklendi ---</i>")
            # In a real app, you would swap self.messages out here based on the selected chat.
            # For now, it just resets the view.
        btn.clicked.connect(load_conv)
        
        self.conversation_buttons_layout.addWidget(btn)
        load_conv()

    def get_mock_profile(self):
        return {"kullanıcı": "Fox", "rol": "Admin", "versiyon": VERSION}

    def open_settings_panel(self):
        # Your existing settings logic...
        splash = QDialog(self, Qt.FramelessWindowHint | Qt.Dialog)
        splash.setAttribute(Qt.WA_TranslucentBackground)
        splash.setModal(True)

        panel = QWidget()
        panel.setStyleSheet("QWidget { background-color: #f5f5f5; border-radius: 12px; }")
        panel.resize(520, 620)
        stack = QStackedLayout()

        main_page = QWidget()
        main_layout = QVBoxLayout(main_page)
        main_layout.setSpacing(25)
        
        theme_btn = QPushButton("Tema Seç")
        lang_btn = QPushButton("Dil Seç")
        prof_btn = QPushButton("Profil Ayarları")
        for b in (theme_btn, lang_btn, prof_btn): self.style_button(b)

        main_layout.addWidget(theme_btn)
        main_layout.addWidget(lang_btn)
        main_layout.addWidget(prof_btn)
        main_layout.addStretch()

        stack.addWidget(main_page)

        prof_btn.clicked.connect(lambda: [splash.close(), self.open_profile_settings()])

        back_btn = QPushButton("✖")
        back_btn.setFixedSize(28, 28)
        self.style_button(back_btn, color="#bdc3c7", text_color="black")
        back_btn.clicked.connect(splash.close)

        panel_vbox = QVBoxLayout(panel)
        back_row = QHBoxLayout()
        back_row.addWidget(back_btn)
        back_row.addStretch()
        panel_vbox.addLayout(back_row)
        panel_vbox.addLayout(stack)

        outer = QVBoxLayout(splash)
        hbox = QHBoxLayout()
        hbox.addWidget(panel)
        outer.addLayout(hbox)
        splash.setLayout(outer)
        
        screen_geo = QApplication.primaryScreen().geometry()
        splash.resize(screen_geo.size())
        splash.exec_()

    def open_profile_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Profil Ayarları")
        layout = QFormLayout(dialog)
        prof = self.get_mock_profile()

        for key, val in prof.items():
            le = QLineEdit(str(val))
            le.setReadOnly(True)
            layout.addRow(f"{key.capitalize()}:", le)

        close_button = QPushButton("Kapat")
        self.style_button(close_button)
        close_button.clicked.connect(dialog.accept)
        layout.addRow(close_button)
        dialog.setLayout(layout)
        dialog.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Setup Splash Screen
    splash = QSplashScreen()
    splash.setFixedSize(300, 180)
    splash.setWindowFlags(Qt.FramelessWindowHint)   
    splash.setStyleSheet("background-color: black; color: white;")
    
    layout = QVBoxLayout(splash)
    version_label = QLabel(VERSION)
    version_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
    version_label.setStyleSheet("color: white; font-size: 14pt;")
    layout.addWidget(version_label)

    status_label = QLabel("Loading MLX Model...\n(This will take a moment)")
    status_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(status_label)

    progress = QProgressBar()
    progress.setMaximum(0) # Indeterminate progress bar while loading
    progress.setStyleSheet("QProgressBar {background-color: #444; color: white; text-align: center;}")
    layout.addWidget(progress)

    splash.setLayout(layout)
    splash.show()

    # Move to center of screen
    screen = QApplication.primaryScreen()
    if screen:
        splash.move(screen.geometry().center() - splash.rect().center())
        
    app.processEvents() # Force UI to update before heavy loading

    # Load the Model while splash screen is active
    model_yolu = "/Users/fox/.lmstudio/models/Jackrong/MLX-Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-6bit"
    
    try:
        model, tokenizer = load(model_yolu)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # Close Splash and Open Main App
    splash.close()
    
    pencere = ChatbotGUI(model, tokenizer)
    pencere.show()
    
    sys.exit(app.exec_())
    