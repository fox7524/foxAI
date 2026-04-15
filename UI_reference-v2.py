import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox,
                            QFrame, QListWidget, QMessageBox)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class ElegantChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):  
        # Main window configuration
        self.setWindowTitle("Thunderbird AI - Elegant UI")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
        """)
        
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar with elegant design
        sidebar = QFrame()
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 20px;
            }
        """)
        sidebar.setFixedWidth(240)
        
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(15)
        
        # Logo/title
        logo = QLabel("Thunderbird AI")
        logo.setFont(QFont("SF Pro Display", 18, QFont.Bold))
        logo.setStyleSheet("color: #2c3e50;")
        sidebar_layout.addWidget(logo, alignment=Qt.AlignTop)
        
        # Navigation buttons
        nav_buttons = [
            ("🏠 Home", "#007AFF"),
            ("⚙ Settings", "#3498db"),
            ("🚪 Logout", "#e74c3c")
        ]
        
        for text, color in nav_buttons:
            btn = QPushButton(text)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border-radius: 8px;
                    padding: 12px;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: {self.darken_color(color, 0.8)};
                }}
            """)
            sidebar_layout.addWidget(btn)
        
        # Chat area
        chat_area = QFrame()
        chat_area.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 20px;
            }
        """)
        
        chat_layout = QVBoxLayout(chat_area)
        chat_layout.setSpacing(15)
        
        # Chat messages
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
                padding: 15px;
                font-size: 14px;
            }
        """)
        chat_layout.addWidget(self.chat_display)
        
        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet("QFrame { background-color: #ffffff; border-radius: 12px; }")
        input_layout = QHBoxLayout(input_frame)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            }
        """)
        input_layout.addWidget(self.input_field, 3)
        
        send_btn = QPushButton("Send")
        send_btn.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #005fa3;
            }
        """)
        input_layout.addWidget(send_btn, 1)
        
        chat_layout.addWidget(input_frame)
        
        # Add components to main layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(chat_area)
        
        # Connect signals
        send_btn.clicked.connect(self.send_message)
        
    def darken_color(self, hex_color, factor):
        """Darken a hex color by a given factor"""
        if hex_color.startswith("#"):
            hex_color = hex_color[1:]
            
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def send_message(self):
        message = self.input_field.text().strip()
        if not message:
            return
            
        # Add message to chat
        self.chat_display.append(f"<div style='margin-bottom: 10px;'><strong>You:</strong> {message}</div>")
        self.input_field.clear()
        
        # Simulate bot response
        self.chat_display.append(f"<div style='margin-bottom: 10px;'><strong>Bot:</strong> This is a simulated response to your message: '{message}'</div>")
        
        # Scroll to bottom
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ElegantChatApp()
    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec_())

