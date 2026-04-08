# Thunderbird AI - A Python-based AI chatbot application
# This file contains the main application logic, user interface, and database management.
#by Fox


# Import necessary libraries for GUI and threading
import sqlite3
from datetime import datetime
import requests
from urllib.parse import quote
from docx import Document
import subprocess
import uuid
from PyQt5.QtWidgets import QSplashScreen
from PyQt5.QtCore import QTimer
import os
import time
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QHBoxLayout, QLabel, QSplitter,
    QDialog, QFormLayout, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal

# Version 
VERSION = "Thunderbird AI Volume Alpha"

# Database initialization functions
def initialize_memory_db():
    # Creates or initializes the memory database and tables.
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_input TEXT NOT NULL,
            bot_reply TEXT NOT NULL,
            source TEXT NOT NULL
        )
    """)
    # New conversations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            speaker TEXT,
            message TEXT,
            timestamp TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()

# Function to initialize the user profile table
# This function creates the user profile table with default values.
# It also adds new columns if they do not exist, ensuring backward compatibility.
def initialize_user_profile():
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            language TEXT DEFAULT 'tr',
            tone TEXT DEFAULT 'formal',
            interests TEXT DEFAULT '',
            sources TEXT DEFAULT 'json,wikipedia,chatgpt',
            email TEXT DEFAULT '',
            first_name TEXT DEFAULT '',
            last_name TEXT DEFAULT '',
            age INTEGER DEFAULT 0,
            birthdate TEXT DEFAULT '',
            chat_count INTEGER DEFAULT 0,
            password TEXT DEFAULT ''
        )
    """)
    # Add new columns if not exist (for backward compatibility)
    try:
        cursor.execute("ALTER TABLE user_profile ADD COLUMN email TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE user_profile ADD COLUMN first_name TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE user_profile ADD COLUMN last_name TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE user_profile ADD COLUMN age INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE user_profile ADD COLUMN chat_count INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE user_profile ADD COLUMN birthdate TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE user_profile ADD COLUMN password TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE user_profile ADD COLUMN theme TEXT DEFAULT 'system'")
    except sqlite3.OperationalError:
        pass 
    # E‑posta alanı benzersiz olsun
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_user_profile_email ON user_profile(email)")
    conn.commit()
    conn.close()

# Function to get user profile from the database
def log_to_memory(user_input, bot_reply, source):
    # Logs a user-bot interaction to the memory database.
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    # Use session timestamp format for consistency
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO memory_log (timestamp, user_input, bot_reply, source) VALUES (?, ?, ?, ?)",
                   (timestamp, user_input, bot_reply, source))
    conn.commit()
    conn.close()

# Function to get user profile from the database
def wikipedia_bilgi_getir(soru, dil='tr'):
    # Fetches a summary from Wikipedia for a given question and logs if found.
    anahtar_soru = quote(soru.replace(' ', '_'))
    url = f"https://{dil}.wikipedia.org/api/rest_v1/page/summary/{anahtar_soru}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            extract = data.get("extract", "")
            if extract and len(extract.split()) > 10:
                specific = filtrele_spesifik_bilgi(soru, extract)
                if specific:
                    log_to_memory(soru, specific, "wikipedia")
                return specific
            else:
                return None
    except:
        return None

# Function to extract text from a ZIM file
def filtrele_spesifik_bilgi(soru, metin):
    # Helper function: Filters for more specific information in Wikipedia text.
    soru = soru.lower()
    if "nerededir" in soru or "nerede" in soru:
        for satir in metin.split("."):
            if "il" in satir or "ilç" in satir or "bölge" in satir:
                return satir.strip() + "."
    return metin

# Function to query the ChatGPT API for a check
def chatgpt_api_sorgula(soru, dil='tr'):
    # Queries the ChatGPT API for a response to the given question.
    api_key = "YOUR_API_KEY_HERE"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": soru}],
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return "(GPT yanıtı alınamadı)"
    except Exception as e:
        return f"ChatGPT API Hatası: {e}"

# Function to create a prompt based on the user's question and context
def tahmini_prompt_olustur(soru, context_list):
    # Generates an extended prompt by guessing context from previous conversation.
    soru = soru.strip().lower()
    if any(zamir in soru for zamir in ["bu", "bunu", "o", "onun", "kim", "ne", "hangi"]):
        son_konular = []
        for sender, msg in reversed(context_list):
            if sender == "Kullanıcı":
                continue
            for kelime in msg.split():
                if kelime.istitle() and kelime not in son_konular:
                    son_konular.append(kelime)
            if len(son_konular) >= 2:
                break
        if son_konular:
            tahmin_konu = ", ".join(son_konular[:2])
            return f"{tahmin_konu} hakkında konuşuluyordu. {soru}"
    return soru


# Function to extract text from a ZIM file
# This function uses the `zim` command-line tool to extract text.
class EmbeddingWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, parent, file_path, file_type):
        super().__init__(parent)
        self.file_path = file_path
        self.file_type = file_type
        self.parent = parent

    def run(self):
        # Runs the embedding extraction and emits the result.
        try:
            if self.file_type == "zim":
                text = self.parent.extract_text_from_zim(self.file_path)
            elif self.file_type == "pdf":
                text = self.parent.extract_text_from_pdf(self.file_path)
            elif self.file_type == "docx":
                text = self.parent.extract_text_from_docx(self.file_path)
            else:
                self.finished.emit("unsupported")
                return

            if text:
                # Split document and create embeddings
                self.parent.pdf_chunks = self.parent.split_text_into_chunks(text)
                self.parent.pdf_vectorizer, self.parent.pdf_embeddings = self.parent.create_embeddings(self.parent.pdf_chunks)
                self.finished.emit("success")
            else:
                self.finished.emit("empty")
        except Exception as e:
            self.finished.emit(f"error: {str(e)}")

# User Interface of Thunderbird AI
# This class handles the main GUI components and user interactions.
class ChatbotGUI(QWidget):

    def style_button(self, button, color="#007ACC", text_color="white"):
        # Styles a QPushButton with the given colors.
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
    def show_recent_memory(self, hours=1):
        # Shows memory logs from the last specified hours in a dialog.
        from datetime import datetime, timedelta
        threshold = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, user_input, bot_reply FROM memory_log WHERE timestamp > ?", (threshold,))
        rows = cursor.fetchall()
        conn.close()
        dialog = QDialog(self)
        dialog.setWindowTitle("Son Bellek Kayıtları")
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        for row in rows:
            text.append(f"[{row[0]}]\nSoru: {row[1]}\nCevap: {row[2]}\n")
        layout.addWidget(text)
        close_btn = QPushButton("Kapat")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.setLayout(layout)
        dialog.exec_()
    def __init__(self):
        # Initializes the ChatbotGUI and sets up the UI and user session.
        self.layout_yuklendi = False
        super().__init__()
        self.setWindowTitle(VERSION)
        self.setGeometry(100, 100, 800, 500)
        initialize_memory_db()
        initialize_user_profile()
        self.context_history = []
        self.max_context_length = 10
        self.conversations = []
        self.max_conversations = 5
        self.current_conversation_id = None
        self.settings_mode = False
        self.current_theme = "system"
        self.memory_mode = "aktif"  # aktif | geçici
        self.emotion_mode = "neutral"  # neutral, happy, sad, angry
        self.model_version = "BIRD AI 1.5"
        self.logged_in_email = None
        # Disable shimmer effect for all labels in init
        # (No shimmer logic, but ensure shimmer_label does nothing)
        def dummy_shimmer(label):
            label.setStyleSheet("color: red; font-weight: bold;")
        self.shimmer_label = dummy_shimmer
        self.dev_mode_enabled = False
        self.ensure_login()
        self.init_ui()
        self.apply_saved_theme_and_language()



    def apply_saved_theme_and_language(self):
        # Applies the saved theme and language from the user profile.
        profile = self.get_user_profile()
            # Apply saved theme
        if profile.get("theme", "system") == "dark":
            self.apply_theme("dark")
        elif profile.get("theme", "light") == "light":
            self.apply_theme("light")
        elif profile.get("theme", "system") == "system":
            self.apply_theme("system")
                # Language kaydı zaten girişte fetch ediliyor
    def ensure_login(self):
        # Ensures the user is logged in or registered before using the app.
        import re
        import sys
        from PyQt5.QtWidgets import QInputDialog
        while not self.logged_in_email:
            dialog = QDialog(self)
            dialog.setWindowTitle(VERSION)
            dialog.resize(500, 600)
            layout = QVBoxLayout(dialog)

            # Step 1: Giriş/Kayıt seçimi
            selection_layout = QHBoxLayout()
            login_select_btn = QPushButton("Log In")
            register_select_btn = QPushButton("Sıgn Up")
            self.style_button(login_select_btn)
            self.style_button(register_select_btn)
            selection_layout.addWidget(login_select_btn)
            selection_layout.addWidget(register_select_btn)
            # Add Çıkış button
            exit_btn = QPushButton("Exit")
            self.style_button(exit_btn)
            exit_btn.clicked.connect(lambda: sys.exit())
            selection_layout.addWidget(exit_btn)
            layout.addLayout(selection_layout)

            # --- Developer Mode Button ---
            dev_mode_button = QPushButton("Dev Mode")
            self.style_button(dev_mode_button, color="red")
            def handle_dev_login():
                dev_dialog = QDialog(self)
                dev_dialog.setWindowTitle("Dev Mode")
                dev_layout = QVBoxLayout(dev_dialog)

                username_input = QLineEdit()
                username_input.setPlaceholderText("Auth Name")
                password_input = QLineEdit()
                password_input.setEchoMode(QLineEdit.Password)
                password_input.setPlaceholderText("Pass")

                dev_layout.addWidget(QLabel("AuthName:"))
                dev_layout.addWidget(username_input)
                dev_layout.addWidget(QLabel("Pass:"))
                dev_layout.addWidget(password_input)

                status_label = QLabel("")
                dev_layout.addWidget(status_label)

                def attempt_dev_login():
                    if username_input.text() == "Thunderbird" and password_input.text() == "Thunderbird123":
                        print("Developer mode credentials accepted.")
                        # Close both dev login and main login dialogs
                        dev_dialog.accept()
                        dialog.accept()
                        # Activate developer mode flag
                        self.dev_mode_enabled = True
                        # Mark a pseudo‑login so ensure_login loop terminates
                        self.logged_in_email = "developer@local"
                        # Launch the developer interface
                        self.enable_dev_mode()
                        # Hide the main Chatbot window – only the Dev panel should remain visible
                        self.hide()
                    else:
                        status_label.setText("Invalid Log.")
                        self.shimmer_label(status_label)

                login_button = QPushButton("Login")
                self.style_button(login_button, color="red")
                login_button.clicked.connect(attempt_dev_login)
                dev_layout.addWidget(login_button)

                dev_dialog.setLayout(dev_layout)
                dev_dialog.exec_()
            dev_mode_button.clicked.connect(handle_dev_login)
            layout.addWidget(dev_mode_button)

            # Placeholders for stacked widgets
            login_widget = QWidget()
            login_form = QVBoxLayout(login_widget)
            reg_widget = QWidget()
            reg_form = QVBoxLayout(reg_widget)
            login_widget.setVisible(False)
            reg_widget.setVisible(False)

            # --- Giriş Formu ---
            login_email_input = QLineEdit()
            login_email_input.setPlaceholderText("Email adresinizi girin")
            login_password_input = QLineEdit()
            login_password_input.setPlaceholderText("Şifre")
            login_password_input.setEchoMode(QLineEdit.Password)
            login_form.addWidget(QLabel("Email:"))
            login_form.addWidget(login_email_input)
            login_form.addWidget(QLabel("Şifre:"))
            login_form.addWidget(login_password_input)
            login_status_label = QLabel("")
            login_form.addWidget(login_status_label)
            login_btn = QPushButton("Giriş Yap")
            self.style_button(login_btn)
            login_form.addWidget(login_btn)
            # --- Şifremi Unuttum Butonu ---
            forgot_btn = QPushButton("Şifremi Unuttum")
            self.style_button(forgot_btn, color="#e67e22")
            login_form.addWidget(forgot_btn)

            import random
            def forgot_password_popup():
                popup = QDialog(self)
                popup.setWindowTitle("Şifre Sıfırlama")
                popup_layout = QVBoxLayout(popup)
                email_input = QLineEdit()
                email_input.setPlaceholderText("Kayıtlı e-posta adresinizi girin")
                popup_layout.addWidget(QLabel("Şifrenizi sıfırlamak için e-posta adresinizi girin:"))
                popup_layout.addWidget(email_input)

                # --- Email Verification Code logic ---
                code_sent = [None]  # store code in list for closure mutability
                code_verified = [False]

                def send_code():
                    email = email_input.text().strip()
                    if not email or '@' not in email:
                        QMessageBox.warning(popup, "Hata", "Geçersiz e-posta adresi giriniz.")
                        return
                    # Generate 6-digit code
                    code = str(random.randint(100000, 999999))
                    code_sent[0] = code
                    from email.message import EmailMessage
                    import smtplib

                    smtp_server = "smtp.gmail.com"
                    smtp_port = 587
                    sender_email = "Thunderbirdai@gmail.com"
                    sender_password = "twethyvvyzrvvxfm".replace(" ", "")

                    try:
                        msg = EmailMessage()
                        msg["Subject"] = "BIRD AI - Şifre Sıfırlama Kodunuz"
                        msg["From"] = sender_email
                        msg["To"] = email
                        msg.set_content(f"Şifre sıfırlama kodunuz: {code}")

                        QMessageBox.information(popup, "Kod Göster", f"Simülasyon kodu: {code}")

                    except Exception as e:
                        QMessageBox.warning(popup, "E-posta Hatası", f"E-posta gönderilemedi: {e}")

                send_code_btn = QPushButton("Kodu Gönder")
                self.style_button(send_code_btn, color="#27ae60")
                send_code_btn.clicked.connect(send_code)
                popup_layout.addWidget(send_code_btn)

                code_input = QLineEdit()
                code_input.setPlaceholderText("E-posta ile gelen doğrulama kodu")
                popup_layout.addWidget(QLabel("Doğrulama Kodu:"))
                popup_layout.addWidget(code_input)

                verify_btn = QPushButton("Kodu Doğrula")
                self.style_button(verify_btn, color="#2980b9")
                def verify_code():
                    if code_sent[0] is None:
                        QMessageBox.warning(popup, "Hata", "Önce kod göndermelisiniz.")
                        return
                    if code_input.text().strip() == code_sent[0]:
                        code_verified[0] = True
                        QMessageBox.information(popup, "Başarılı", "Kod doğrulandı! Şifre sıfırlamaya devam edebilirsiniz.")
                    else:
                        QMessageBox.warning(popup, "Hata", "Kod yanlış.")
                verify_btn.clicked.connect(verify_code)
                popup_layout.addWidget(verify_btn)

                new_pass_input = QLineEdit()
                new_pass_input.setPlaceholderText("Yeni şifre")
                new_pass_input.setEchoMode(QLineEdit.Password)
                popup_layout.addWidget(QLabel("Yeni Şifre:"))
                popup_layout.addWidget(new_pass_input)
                new_pass2_input = QLineEdit()
                new_pass2_input.setPlaceholderText("Yeni şifre (tekrar)")
                new_pass2_input.setEchoMode(QLineEdit.Password)
                popup_layout.addWidget(QLabel("Yeni Şifre (tekrar):"))
                popup_layout.addWidget(new_pass2_input)
                status_label = QLabel("")
                popup_layout.addWidget(status_label)
                save_btn = QPushButton("Şifreyi Sıfırla")
                self.style_button(save_btn)
                def do_reset():
                    if not code_verified[0]:
                        status_label.setText("Önce e-posta doğrulaması yapmalısınız.")
                        return
                    email = email_input.text().strip()
                    new_pass = new_pass_input.text()
                    new_pass2 = new_pass2_input.text()
                    if not email or '@' not in email:
                        status_label.setText("Geçersiz e-posta.")
                        return
                    if not new_pass or not new_pass2:
                        status_label.setText("Yeni şifre alanı boş olamaz.")
                        return
                    if new_pass != new_pass2:
                        status_label.setText("Şifreler eşleşmiyor.")
                        return
                    conn = sqlite3.connect("memory.db")
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM user_profile WHERE email = ?", (email,))
                    if not cursor.fetchone():
                        status_label.setText("Bu e-posta ile kayıtlı bir kullanıcı bulunamadı.")
                        conn.close()
                        return
                    cursor.execute("UPDATE user_profile SET password = ? WHERE email = ?", (new_pass, email))
                    conn.commit()
                    conn.close()
                    status_label.setText("Şifreniz başarıyla güncellendi.")
                    popup.accept()
                save_btn.clicked.connect(do_reset)
                popup_layout.addWidget(save_btn)
                popup.setLayout(popup_layout)
                popup.exec_()
            forgot_btn.clicked.connect(forgot_password_popup)

            # --- Kayıt Formu ---
            reg_email_input = QLineEdit()
            reg_email_input.setPlaceholderText("Email adresinizi girin")
            reg_password_input = QLineEdit()
            reg_password_input.setPlaceholderText("Şifre")
            reg_password_input.setEchoMode(QLineEdit.Password)
            reg_password2_input = QLineEdit()
            reg_password2_input.setPlaceholderText("Şifreyi tekrar girin")
            reg_password2_input.setEchoMode(QLineEdit.Password)
            reg_fname_input = QLineEdit()
            reg_fname_input.setPlaceholderText("Adınızı girin")
            reg_lname_input = QLineEdit()
            reg_lname_input.setPlaceholderText("Soyadınızı girin")
            reg_birthdate_input = QLineEdit()
            reg_birthdate_input.setPlaceholderText("Doğum tarihi (gg/aa/yyyy)")
            reg_form.addWidget(QLabel("Email:"))
            reg_form.addWidget(reg_email_input)
            reg_form.addWidget(QLabel("Şifre:"))
            reg_form.addWidget(reg_password_input)
            reg_form.addWidget(QLabel("Şifre (tekrar):"))
            reg_form.addWidget(reg_password2_input)
            # --- Şifre uyuşma kontrolü (anlık) ---
            def check_password_match():
                if reg_password_input.text() != reg_password2_input.text():
                    reg_status_label.setText("Şifreler uyuşmuyor.")
                else:
                    reg_status_label.setText("")

            reg_password_input.textChanged.connect(check_password_match)
            reg_password2_input.textChanged.connect(check_password_match)
            # Enter key in first password field triggers registration
            reg_password_input.returnPressed.connect(attempt_register)
            # Enter key in second password field already triggers via reg_btn if focused and default button
            reg_form.addWidget(QLabel("Ad:"))
            reg_form.addWidget(reg_fname_input)
            reg_form.addWidget(QLabel("Soyad:"))
            reg_form.addWidget(reg_lname_input)
            reg_form.addWidget(QLabel("Doğum Tarihi (gg/aa/yyyy):"))
            reg_form.addWidget(reg_birthdate_input)
            reg_status_label = QLabel("")
            reg_form.addWidget(reg_status_label)
            reg_btn = QPushButton("Kayıt Ol")
            self.style_button(reg_btn)
            reg_form.addWidget(reg_btn)

            layout.addWidget(login_widget)
            layout.addWidget(reg_widget)

            def show_login():
                login_widget.setVisible(True)
                reg_widget.setVisible(False)
            def show_register():
                login_widget.setVisible(False)
                reg_widget.setVisible(True)
            login_select_btn.clicked.connect(show_login)
            register_select_btn.clicked.connect(show_register)

            # --- Giriş işlemi ---
            def attempt_login():
                email = login_email_input.text().strip()
                password = login_password_input.text()
                if not email or '@' not in email:
                    login_status_label.setText("Lütfen geçerli bir e-posta adresi giriniz.")
                    self.shimmer_label(login_status_label)
                    return
                if not password:
                    login_status_label.setText("Lütfen şifrenizi giriniz.")
                    self.shimmer_label(login_status_label)
                    return
                try:
                    conn = sqlite3.connect("memory.db")
                    cursor = conn.cursor()
                    cursor.execute("SELECT password FROM user_profile WHERE email = ?", (email,))
                    row = cursor.fetchone()
                    if not row:
                        login_status_label.setText("Böyle bir hesap bulunamadı. Lütfen kayıt olun.")
                        self.shimmer_label(login_status_label)
                        conn.close()
                        return
                    db_password = row[0] if row else ""
                    if password != db_password:
                        login_status_label.setText("Şifre yanlış. Lütfen tekrar deneyiniz.")
                        self.shimmer_label(login_status_label)
                        conn.close()
                        return
                    # Set user as default for session
                    conn.commit()
                    conn.close()
                    self.logged_in_email = email
                    dialog.accept()
                except Exception as e:
                    login_status_label.setText(f"Hata oluştu: {e}")
                    self.shimmer_label(login_status_label)

            login_btn.clicked.connect(attempt_login)

            # --- Kayıt işlemi ---
            def attempt_register():
                import re
                import random
                email = reg_email_input.text().strip()
                password = reg_password_input.text()
                password2 = reg_password2_input.text()
                fname = reg_fname_input.text().strip()
                lname = reg_lname_input.text().strip()
                birthdate = reg_birthdate_input.text().strip()
                # ---- E‑posta doğrulama adım 1 : benzersizlik ----
                conn = sqlite3.connect("memory.db")
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM user_profile WHERE email=?", (email,))
                if cursor.fetchone():
                    reg_status_label.setText("Bu e-posta zaten kayıtlı.")
                    self.shimmer_label(reg_status_label)
                    conn.close()
                    return
                conn.close()

                # ---- E‑posta doğrulama adım 2 : sadece Gmail ----
                if not email.lower().endswith("@gmail.com"):
                    reg_status_label.setText("Şu anda yalnızca Gmail adresleri destekleniyor.")
                    self.shimmer_label(reg_status_label)
                    return
                # Validate
                if '@' not in email or not re.match(r".+@.+\.(com|net|org)$", email):
                    reg_status_label.setText("Geçersiz email (eksik @ veya uzantı hatalı).")
                    self.shimmer_label(reg_status_label)
                    return
                if not password or not password2:
                    reg_status_label.setText("Şifre alanı boş olamaz.")
                    self.shimmer_label(reg_status_label)
                    return
                if password != password2:
                    reg_status_label.setText("Şifreler eşleşmiyor.")
                    self.shimmer_label(reg_status_label)
                    return
                if not fname:
                    reg_status_label.setText("Ad alanı boş olamaz.")
                    self.shimmer_label(reg_status_label)
                    return
                if not lname:
                    reg_status_label.setText("Soyad alanı boş olamaz.")
                    self.shimmer_label(reg_status_label)
                    return
                if not re.match(r"^\d{2}/\d{2}/\d{4}$", birthdate):
                    reg_status_label.setText("Doğum tarihi formatı hatalı (gg/aa/yyyy olmalı).")
                    self.shimmer_label(reg_status_label)
                    return
                day, month, year = birthdate.split("/")
                if not re.match(r"^\d{2}/\d{2}/\d{4}$", birthdate):
                    reg_status_label.setText("Lütfen doğum tarihinizi gg/aa/yyyy formatında giriniz.")
                    self.shimmer_label(reg_status_label)
                    return
                try:
                    import datetime
                    datetime.datetime(int(year), int(month), int(day))
                except Exception:
                    reg_status_label.setText("Geçersiz doğum tarihi.")
                    self.shimmer_label(reg_status_label)
                    return
                # --- Email Verification Code logic for register ---
                code = str(random.randint(100000, 999999))
                # Real email sending for registration
                from email.message import EmailMessage
                import smtplib
                smtp_server = "smtp.gmail.com"
                smtp_port = 587
                sender_email = "seningmailadresin@gmail.com"
                sender_password = "UYGULAMA_SIFRESI_BURAYA"  # Replace with actual app password
                try:
                    msg = EmailMessage()
                    msg["Subject"] = "BIRD AI - Kayıt Doğrulama Kodunuz"
                    msg["From"] = sender_email
                    msg["To"] = email
                    msg.set_content(f"Kayıt doğrulama kodunuz: {code}")
                    QMessageBox.information(self, "Kod Göster", f"Simülasyon kodu: {code}")
                except Exception as e:
                    QMessageBox.warning(self, "E-posta Hatası", f"E-posta gönderilemedi: {e}")
                    reg_status_label.setText("Kayıt iptal edildi.")
                    self.shimmer_label(reg_status_label)
                    return
                # Prompt user to enter code
                from PyQt5.QtWidgets import QInputDialog
                for _ in range(3):
                    user_code, ok = QInputDialog.getText(self, "Kod Doğrulama", "E-posta ile gelen doğrulama kodunu girin:")
                    if not ok:
                        reg_status_label.setText("Kayıt işlemi iptal edildi.")
                        self.shimmer_label(reg_status_label)
                        return
                    if user_code.strip() == code:
                        break
                    else:
                        QMessageBox.warning(self, "Hata", "Kod yanlış. Lütfen tekrar deneyin.")
                else:
                    reg_status_label.setText("Doğrulama başarısız. Kayıt iptal edildi.")
                    self.shimmer_label(reg_status_label)
                    return
                try:
                    conn = sqlite3.connect("memory.db")
                    cursor = conn.cursor()
                    # Insert new user with unique username to avoid duplicate username values
                    unique_username = str(uuid.uuid4())
                    cursor.execute(
                        "INSERT INTO user_profile (username, email, password, first_name, last_name, birthdate) VALUES (?, ?, ?, ?, ?, ?)",
                        (unique_username, email, password, fname, lname, birthdate)
                    )
                    # First set all usernames to non-default
                    self.logged_in_email = email
            
                    conn.commit()
                    conn.close()
                    self.logged_in_email = email
                    dialog.accept()
                except Exception as e:
                    reg_status_label.setText(f"Hata oluştu: {e}")
                    self.shimmer_label(reg_status_label)

            # shimmer_label is disabled (dummy) in __init__ to prevent shimmer anywhere, including Booting...

            reg_btn.clicked.connect(attempt_register)

            dialog.setLayout(layout)
            dialog.exec_()

    def init_ui(self):
        # Initializes the main user interface components and layouts.
        if self.layout_yuklendi:
            return
        layout = QHBoxLayout(self)

        # Sidebar
        self.sidebar = QVBoxLayout()
        self.sidebar_label = QLabel("BIRD AI")
        self.sidebar_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        # Sidebar label and header buttons row
        header_layout = QHBoxLayout()
        header_layout.addWidget(self.sidebar_label)

        # Tiny Settings button (gear)
        self.settings_button = QPushButton("⚙")
        self.settings_button.setFixedSize(24, 24)
        self.style_button(self.settings_button, color="#555", text_color="white")
        self.settings_button.clicked.connect(self.open_settings_panel)
        header_layout.addWidget(self.settings_button)

        # Tiny New‑Chat button (plus)
        self.newchat_button = QPushButton("＋")
        self.newchat_button.setFixedSize(24, 24)
        self.style_button(self.newchat_button, color="#2ecc71", text_color="white")
        self.newchat_button.clicked.connect(self.new_conversation)
        header_layout.addWidget(self.newchat_button)

        header_layout.setAlignment(Qt.AlignTop)
        self.sidebar.addLayout(header_layout)



        self.exit_button = QPushButton("Exit")
        self.style_button(self.exit_button)
        self.exit_button.clicked.connect(self.close)
        self.sidebar.addWidget(self.exit_button)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(self.sidebar)
        sidebar_widget.setFixedWidth(150)

        # Chat Area
        self.chat_layout = QVBoxLayout()

        # Remove the toggle sidebar button at the top (toggle_btn)
        # Instead, always keep hamburger_button visible, placed in main layout

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.soru_sor)
        self.chat_layout.addWidget(self.chat_display)
        self.chat_layout.addWidget(self.input_field)

        # Add suggestion label below the input field
        self.suggestion_label = QLabel("")
        self.chat_layout.addWidget(self.suggestion_label)

        chat_widget = QWidget()
        chat_widget.setLayout(self.chat_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(sidebar_widget)
        splitter.addWidget(chat_widget)

        layout.addWidget(splitter)

        # Always visible hamburger button (bottom-left)
        self.hamburger_button = QPushButton("☰")
        self.style_button(self.hamburger_button, color="#333", text_color="white")
        def toggle_sidebar():
            sidebar_widget.setVisible(not sidebar_widget.isVisible())
        self.hamburger_button.clicked.connect(toggle_sidebar)
        layout.addWidget(self.hamburger_button)

        self.setLayout(layout)
        self.layout_yuklendi = True

    def enter_developer_mode(self):
        # Prompts for developer mode password and enables dev mode if correct.
        auth_dialog = QDialog(self)
        auth_dialog.setWindowTitle("Developer Girişi")
        layout = QVBoxLayout()

        password_input = QLineEdit()
        password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(QLabel("Şifre:"))
        layout.addWidget(password_input)

        status_label = QLabel("")
        layout.addWidget(status_label)

        def check_password():
            if password_input.text() == "kyra1453":
                QMessageBox.information(self, "Başarılı", "Developer moduna giriş yapılıyor...")
                auth_dialog.accept()
                self.enable_dev_mode()
            else:
                status_label.setText("ACCESS DENIED")
                auth_dialog.reject()

        login_btn = QPushButton("Giriş Yap")
        login_btn.clicked.connect(check_password)
        self.style_button(login_btn, color="red")

        layout.addWidget(login_btn)
        auth_dialog.setLayout(layout)
        auth_dialog.exec_()

    def enable_dev_mode(self):
        """
        Enables developer mode, launching the developer interface.
        """
        # Mark the main window as running in developer mode
        self.dev_mode_enabled = True
        from PyQt5.QtWidgets import QDialog
        self.dev_window = DevGUI()
        self.dev_window.setWindowTitle("Developer Panel")
        self.dev_window.setGeometry(200, 200, 700, 500)
        self.dev_window.show()

    def load_zim_file(self):
        # Loads and processes a ZIM file for embedding search.
        self.pdf_chunks = []
        self.pdf_vectorizer = None
        self.pdf_embeddings = None
        from PyQt5.QtWidgets import QFileDialog
        try:
            zim_path, _ = QFileDialog.getOpenFileName(self, "ZIM Dosyası Seç", "", "ZIM Dosyaları (*.zim)")
            if zim_path:
                QMessageBox.information(self, "ZIM Yükleniyor", "ZIM dosyası yükleniyor ve işleniyor. Lütfen bekleyin...")
                self.worker = EmbeddingWorker(self, zim_path, "zim")
                self.worker.finished.connect(self.handle_embedding_result)
                self.worker.start()
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"ZIM dosyası yüklenemedi: {e}")

    def load_pdf_document(self):
        # Loads and processes a PDF or DOCX file for embedding search.
        self.pdf_chunks = []
        self.pdf_vectorizer = None
        self.pdf_embeddings = None
        from PyQt5.QtWidgets import QFileDialog
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Belge Seç", "", "PDF veya Word Dosyaları (*.pdf *.docx)")
            if file_name:
                if file_name.endswith(".pdf"):
                    file_type = "pdf"
                elif file_name.endswith(".docx"):
                    file_type = "docx"
                else:
                    QMessageBox.warning(self, "Hata", "Sadece PDF veya DOCX dosyası seçebilirsiniz.")
                    return
                QMessageBox.information(self, "Belge Yükleniyor", f"{file_type.upper()} dosyası yükleniyor ve işleniyor. Lütfen bekleyin...")
                self.worker = EmbeddingWorker(self, file_name, file_type)
                self.worker.finished.connect(self.handle_embedding_result)
                self.worker.start()
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Belge yüklenemedi: {e}")

    def handle_embedding_result(self, result):
        # Handles the result of the embedding worker and notifies the user.
        if result == "success":
            QMessageBox.information(self, "Başarılı", f"{len(self.pdf_chunks)} parça oluşturuldu.")
        elif result == "empty":
            QMessageBox.warning(self, "Başarısız", "Dosyadan içerik çıkarılamadı.")
        elif result == "unsupported":
            QMessageBox.warning(self, "Hata", "Geçersiz dosya türü.")
        elif result.startswith("error:"):
            QMessageBox.warning(self, "Hata", f"İşlem hatası: {result}")
        else:
            QMessageBox.warning(self, "Hata", f"İşlem sırasında bilinmeyen bir hata oluştu: {result}")
        self.worker = None

    def show_memory_log(self):
        # Shows the last 50 memory log entries in a dialog.
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, user_input, bot_reply, source FROM memory_log ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()

        dialog = QDialog(self)
        dialog.setWindowTitle("Bellek Kayıtları")
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        for row in rows:
            text.append(f"[{row[0]}] ({row[3]})\nSoru: {row[1]}\nCevap: {row[2]}\n")
        layout.addWidget(text)

        close_btn = QPushButton("Kapat")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec_()

    def add_qa_popup(self):
        # Opens a dialog to manually add a question-answer pair to memory.
        dialog = QDialog(self)
        dialog.setWindowTitle("Soru-Cevap Ekle")
        layout = QVBoxLayout(dialog)

        soru_input = QTextEdit()
        soru_input.setPlaceholderText("Soru")
        cevap_input = QTextEdit()
        cevap_input.setPlaceholderText("Cevap")
        layout.addWidget(QLabel("Soru:"))
        layout.addWidget(soru_input)
        layout.addWidget(QLabel("Cevap:"))
        layout.addWidget(cevap_input)

        def kaydet():
            soru = soru_input.toPlainText().strip()
            cevap = cevap_input.toPlainText().strip()
            if soru and cevap:
                log_to_memory(soru, cevap, "manual")
                QMessageBox.information(dialog, "Başarılı", "Soru-Cevap kaydedildi.")
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Hata", "Boş bırakılamaz.")

        save_btn = QPushButton("Kaydet")
        save_btn.clicked.connect(kaydet)
        layout.addWidget(save_btn)

        cancel_btn = QPushButton("İptal")
        cancel_btn.clicked.connect(dialog.reject)
        layout.addWidget(cancel_btn)

        dialog.setLayout(layout)
        dialog.exec_()

    def show_json_popup(self):
        # Shows all manual Q&A pairs from memory as JSON in a dialog.
        dialog = QDialog(self)
        dialog.setWindowTitle("JSON Görüntüle (veritabanından alınan kayıtlar)")
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)

        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT user_input, bot_reply FROM memory_log WHERE source = 'manual'")
        rows = cursor.fetchall()
        conn.close()

        json_entries = [{"soru": row[0], "cevap": row[1]} for row in rows]
        import json
        text.setText(json.dumps(json_entries, indent=2, ensure_ascii=False))
        layout.addWidget(text)

        close_btn = QPushButton("Kapat")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec_()

    def soru_sor(self):
        # Handles user input, generates a response, and updates memory and UI.
        soru = self.input_field.text().strip()
        if not soru:
            return
        profile = self.get_user_profile()
        if profile["chat_count"] >= 5:
            QMessageBox.warning(self, "Limit Aşıldı", "Maksimum 5 sohbet hakkınızı kullandınız.")
            return
        self.chat_display.append(f"<div style='border:1px solid #ccc; padding:5px; border-radius:8px;'><b>SEN:</b> {soru}</div>")
        self.input_field.clear()

        context_text = ""
        for sender, msg in self.context_history[-self.max_context_length:]:
            context_text += f"{sender}: {msg}\n"
        genişletilmiş = tahmini_prompt_olustur(soru, self.context_history)
        # Add emotional and tone cues
        tone_text = ""
        if profile["tone"] == "formal":
            tone_text += "Cevapların resmi bir üslupla olsun. "
        elif profile["tone"] == "casual":
            tone_text += "Cevapların samimi ve günlük dilde olsun. "

        if self.emotion_mode == "happy":
            tone_text += "Cevapların neşeli ve pozitif bir şekilde verilsin."
        elif self.emotion_mode == "sad":
            tone_text += "Cevapların biraz daha düşünceli ve ciddi olsun."
        elif self.emotion_mode == "angry":
            tone_text += "Cevapların biraz daha sert ve kısa olsun."
        else:
            tone_text += "Cevapların dengeli ve tarafsız olsun."

        context_text += f"Kullanıcı: {genişletilmiş}\nYönerge: {tone_text}"
        genişletilmiş_soru = context_text

        # Kişisel ilgi alanı odaklı prompt ekle
        if profile["interests"]:
            interest_hint = profile["interests"].split(",")[0].strip()
            genişletilmiş_soru = f"Kullanıcının ilgilendiği konu: {interest_hint}\n{genişletilmiş_soru}"

        cevap = None
        try:
            # Try Embedding Search first
            # Embedding similarity search
            if self.dev_mode_enabled:
                if hasattr(self, 'pdf_vectorizer') and hasattr(self, 'pdf_embeddings') and self.pdf_embeddings is not None:
                    from sklearn.metrics.pairwise import cosine_similarity
                    user_embedding = self.pdf_vectorizer.transform([soru])
                    scores = cosine_similarity(user_embedding, self.pdf_embeddings)
                    max_score = scores.max()
                    if max_score >= 0.88649827630184394:
                        best_idx = scores.argmax()
                        cevap = self.pdf_chunks[best_idx]
                    else:
                        cevap = None
                else:
                    cevap = None
        except Exception as e:
            QMessageBox.warning(self, "Embedding Hatası", f"Embedding işlemi başarısız: {e}")
            cevap = None

        try:
            # Only if cevap is still None, proceed with JSON, Wikipedia, ChatGPT
            # Manual Q&A (JSON) lookup
            if cevap is None:
                if "json" in profile["sources"]:
                    conn = sqlite3.connect("memory.db")
                    cursor = conn.cursor()
                    cursor.execute("SELECT bot_reply FROM memory_log WHERE user_input = ? AND source = 'manual'", (soru,))
                    result = cursor.fetchone()
                    conn.close()
                    if result:
                        cevap = result[0]
        except Exception as e:
            QMessageBox.warning(self, "Bellek Hatası", f"JSON kaynağından veri alınamadı: {e}")

        try:
            # Wikipedia search
            if cevap is None and "wikipedia" in profile["sources"]:
                cevap = wikipedia_bilgi_getir(soru, profile["language"])
        except Exception as e:
            QMessageBox.warning(self, "Wikipedia Hatası", f"Wikipedia'dan veri alınamadı: {e}")

        try:
            # ChatGPT API call
            if cevap is None and "chatgpt" in profile["sources"]:
                cevap = chatgpt_api_sorgula(genişletilmiş_soru, profile["language"])
        except Exception as e:
            QMessageBox.warning(self, "ChatGPT Hatası", f"ChatGPT'den veri alınamadı: {e}")

        if cevap and "GPT" in cevap:
            log_to_memory(soru, cevap, "chatgpt")

        if not cevap:
            cevap = "(Hiçbir kaynaktan anlamlı cevap alınamadı.)"

        self.chat_display.append(f"<div style='border:1px solid #ccc; padding:5px; border-radius:8px;'><b>BIRD AI:</b> {cevap}</div>")

        if self.memory_mode == "aktif":
            self.context_history.append(("Kullanıcı", soru))
            self.context_history.append(("BIRD AI", cevap))
            if len(self.context_history) > self.max_context_length:
                self.context_history = self.context_history[-self.max_context_length:]
        # Add user profile memory to context
        self.context_history.append(("Profil", f"{profile['first_name']} {profile['last_name']} - {profile['email']} Yaş:{profile['age']}"))
        # Conversation history for multi-conversation
        if self.conversations:
            self.conversations[-1].append(("Kullanıcı", soru))
            self.conversations[-1].append(("BIRD AI", cevap))
        # Log conversation to memory
        log_to_memory(soru, cevap, "chat")

        # Save chat to conversation_logs if conversation_id exists
        try:
            # Insert chat into conversation_logs table
            if self.current_conversation_id:
                from datetime import datetime
                conn = sqlite3.connect("memory.db")
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                cursor.execute("INSERT INTO conversation_logs (conversation_id, speaker, message, timestamp) VALUES (?, ?, ?, ?)",
                               (self.current_conversation_id, "Kullanıcı", soru, now))
                cursor.execute("INSERT INTO conversation_logs (conversation_id, speaker, message, timestamp) VALUES (?, ?, ?, ?)",
                               (self.current_conversation_id, "BIRD AI", cevap, now))
                conn.commit()
                conn.close()
        except Exception as e:
            QMessageBox.warning(self, "Sohbet Kaydı Hatası", f"Sohbet kaydedilemedi: {e}")

        try:
            # Update user's chat count
            conn = sqlite3.connect("memory.db")
            cursor = conn.cursor()
            cursor.execute("UPDATE user_profile SET chat_count = chat_count + 1 WHERE username = 'default'")
            conn.commit()
            conn.close()
        except Exception as e:
            QMessageBox.warning(self, "Kullanıcı Güncelleme Hatası", f"Kullanıcı sohbet sayısı güncellenemedi: {e}")

        # Suggest follow-up questions based on interests, emotion and tone
        suggestion = ""
        if profile["interests"]:
            interest = profile["interests"].split(",")[0]
            suggestion = f"{interest.strip().capitalize()} hakkında daha fazla bilgi ister misin?"
        elif self.emotion_mode == "happy":
            suggestion = "Yeni bir şey öğrenmek ister misin?"
        elif self.emotion_mode == "sad":
            suggestion = "İstersen moralini yükseltecek bir bilgi verebilirim."
        elif self.emotion_mode == "angry":
            suggestion = "Sakinleşmek için istersen bir fıkra anlatayım mı?"
        else:
            suggestion = "Başka bir konuda yardımcı olabilir miyim?"
        self.suggestion_label.setText(f"<i>Öneri: {suggestion}</i>")

    def clear_memory_log(self):
        # Clears all memory log entries from the database.
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memory_log")
        conn.commit()
        conn.close()
        QMessageBox.information(self, "Silindi", "Bellek verisi temizlendi.")

    def clear_context_history(self):
        # Clears the conversation context history.
        self.context_history.clear()
        self.chat_display.append("<i>Konuşma geçmişi temizlendi.</i>")

    def clear_sidebar(self):
        # Removes all widgets from the sidebar.
        for i in reversed(range(self.sidebar.count())):
            widget = self.sidebar.itemAt(i).widget()
            if widget:
                widget.setParent(None)

    def open_settings_panel(self):
        """
        Persistent Settings overlay:
        - Frameless dark overlay
        - Central white panel with internal stacked pages
        - Back button (←) at top‑left of the panel
        - Theme / Language pages use radio buttons
        """
        from PyQt5.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout,
                                     QRadioButton, QButtonGroup, QStackedLayout)
        from PyQt5.QtCore import Qt, QRect, QPoint

        # ---------- Outer overlay ----------
        splash = QDialog(self, Qt.FramelessWindowHint | Qt.Dialog)
        splash.setAttribute(Qt.WA_TranslucentBackground)
        splash.setModal(True)

        # ---------- Inner panel ----------
        panel = QWidget()
        panel.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-radius: 12px;
            }
        """)
        panel.resize(520, 620)

        # StackedLayout to swap pages
        stack = QStackedLayout()

        # ----- 0) MAIN MENU PAGE -----
        main_page = QWidget()
        main_layout = QVBoxLayout(main_page)
        main_layout.setContentsMargins(40, 60, 40, 40)
        main_layout.setSpacing(25)

        theme_btn = QPushButton("Tema Seç")
        lang_btn  = QPushButton("Dil Seç")
        dev_btn   = QPushButton("Geliştirici Modu")
        prof_btn  = QPushButton("Profil Ayarları")

        for b in (theme_btn, lang_btn, dev_btn, prof_btn):
            self.style_button(b)

        dev_btn.setStyleSheet("background-color:#e74c3c; color:white;")

        main_layout.addWidget(theme_btn)
        main_layout.addWidget(lang_btn)
        main_layout.addWidget(dev_btn)
        main_layout.addWidget(prof_btn)
        main_layout.addStretch()

        # ----- 1) THEME PAGE -----
        theme_page = QWidget()
        t_layout = QVBoxLayout(theme_page)
        t_layout.setContentsMargins(40, 60, 40, 40)
        t_layout.setSpacing(15)

        theme_group = QButtonGroup(theme_page)
        rb_light  = QRadioButton("Açık Tema")
        rb_dark   = QRadioButton("Koyu Tema")
        rb_system = QRadioButton("Sistem Teması")
        for rb in (rb_light, rb_dark, rb_system):
            theme_group.addButton(rb)
            t_layout.addWidget(rb)

        # Pre‑select current theme
        if self.current_theme == "light":
            rb_light.setChecked(True)
        elif self.current_theme == "dark":
            rb_dark.setChecked(True)
        else:
            rb_system.setChecked(True)

        save_theme_btn = QPushButton("Kaydet")
        self.style_button(save_theme_btn, color="#27ae60")
        t_layout.addWidget(save_theme_btn)
        t_layout.addStretch()

        # ----- 2) LANGUAGE PAGE -----
        lang_page = QWidget()
        l_layout = QVBoxLayout(lang_page)
        l_layout.setContentsMargins(40, 60, 40, 40)
        l_layout.setSpacing(15)

        lang_group = QButtonGroup(lang_page)
        rb_tr = QRadioButton("Türkçe")
        rb_en = QRadioButton("English")
        for rb in (rb_tr, rb_en):
            lang_group.addButton(rb)
            l_layout.addWidget(rb)

        profile = self.get_user_profile()
        if profile["language"] == "en":
            rb_en.setChecked(True)
        else:
            rb_tr.setChecked(True)

        save_lang_btn = QPushButton("Kaydet")
        self.style_button(save_lang_btn, color="#27ae60")
        l_layout.addWidget(save_lang_btn)
        l_layout.addStretch()

        # ---------- Back button ----------
        back_btn = QPushButton("←")
        back_btn.setFixedSize(28, 28)
        self.style_button(back_btn, color="#bdc3c7", text_color="black")

        def show_main():
            stack.setCurrentIndex(0)
        back_btn.clicked.connect(show_main)

        # ---------- button wiring ----------
        theme_btn.clicked.connect(lambda: stack.setCurrentIndex(1))
        lang_btn.clicked.connect(lambda: stack.setCurrentIndex(2))
        prof_btn.clicked.connect(lambda: [splash.close(), self.open_profile_settings()])
        dev_btn.clicked.connect(lambda: [splash.close(), self.enter_developer_mode()])

        def apply_theme_change():
            if rb_light.isChecked():
                self.apply_theme("light")
            elif rb_dark.isChecked():
                self.apply_theme("dark")
            else:
                self.apply_theme("system")
            show_main()
        save_theme_btn.clicked.connect(apply_theme_change)

        def apply_lang_change():
            if rb_tr.isChecked():
                self.change_language("tr")
            else:
                self.change_language("en")
            show_main()
        save_lang_btn.clicked.connect(apply_lang_change)

        # ---------- assemble stack ----------
        stack.addWidget(main_page)   # index 0
        stack.addWidget(theme_page)  # index 1
        stack.addWidget(lang_page)   # index 2

        # Outer layout for panel: back button top‑row, then stacked pages
        panel_vbox = QVBoxLayout(panel)
        back_row = QHBoxLayout()
        back_row.addWidget(back_btn)
        back_row.addStretch()
        panel_vbox.addLayout(back_row)
        panel_vbox.addLayout(stack)

        # ---------- Place panel inside splash ----------
        outer = QVBoxLayout(splash)
        outer.addStretch()
        hbox = QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(panel)
        hbox.addStretch()
        outer.addLayout(hbox)
        outer.addStretch()
        splash.setLayout(outer)

        # Overlay covers whole screen
        screen_geo: QRect = QApplication.desktop().screenGeometry(self)
        splash.resize(screen_geo.size())

        splash.exec_()

    def open_theme_selector(self):
        # Opens the theme selection as a dialog window instead of modifying the sidebar.
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QRadioButton, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Tema Seç")
        layout = QVBoxLayout(dialog)

        light_theme_btn = QRadioButton("Açık Tema")
        dark_theme_btn = QRadioButton("Koyu Tema")
        system_theme_btn = QRadioButton("Sistem Teması")

        layout.addWidget(light_theme_btn)
        layout.addWidget(dark_theme_btn)
        layout.addWidget(system_theme_btn)

        # Pre-select the current theme
        if self.current_theme == "light":
            light_theme_btn.setChecked(True)
        elif self.current_theme == "dark":
            dark_theme_btn.setChecked(True)
        else:
            system_theme_btn.setChecked(True)

        def set_theme():
            if light_theme_btn.isChecked():
                self.apply_theme("light")
            elif dark_theme_btn.isChecked():
                self.apply_theme("dark")
            else:
                self.apply_theme("system")

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(lambda: [set_theme(), dialog.accept()])
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)
        dialog.exec_()

    def open_lang_selector(self):
        # Opens the language selector as a dialog window instead of sidebar.
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QRadioButton, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Dil Seç")
        layout = QVBoxLayout(dialog)

        tr_lang_btn = QRadioButton("Türkçe")
        en_lang_btn = QRadioButton("İngilizce")

        layout.addWidget(tr_lang_btn)
        layout.addWidget(en_lang_btn)

        profile = self.get_user_profile()
        if profile["language"] == "en":
            en_lang_btn.setChecked(True)
        else:
            tr_lang_btn.setChecked(True)

        def set_language():
            if tr_lang_btn.isChecked():
                self.change_language("tr")
            elif en_lang_btn.isChecked():
                self.change_language("en")

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(lambda: [set_language(), dialog.accept()])
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)
        dialog.exec_()

    def open_personal_theme_settings(self):
        # Opens a color picker to set a custom theme color.
        from PyQt5.QtWidgets import QColorDialog
        color = QColorDialog.getColor()
        if color.isValid():
            self.setStyleSheet(f"background-color: {color.name()}; color: #000;")

    def change_language(self, lang):
        # Changes the user profile language setting.
        profile = self.get_user_profile()
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE user_profile SET language=? WHERE email=?", (lang, self.logged_in_email))
        conn.commit()
        conn.close()

    def update_emotion_mode(self, mood):
        # Updates the current emotion mode and sidebar label.
        self.emotion_mode = mood
        # Optionally update theme or other UI elements here if needed
        # Change sidebar label icon to reflect emotion
        if hasattr(self, 'sidebar_label'):
            emoji = {
                "happy": "🙂",
                "sad": "😕",
                "angry": "😠",
                "neutral": "😐"
            }.get(mood, "😐")
            self.sidebar_label.setText(f"BIRD AI {emoji}")

    def apply_theme(self, theme):
        # Applies the selected theme to the application and saves it.
        if theme == "dark":
            # Much darker background and lighter text
            self.setStyleSheet("background-color: #121212; color: #E0E0E0;")
        elif theme == "light":
            # Cleaner white background and darker text
            self.setStyleSheet("background-color: #FFFFFF; color: #000000;")
        else:  # system
            self.setStyleSheet("")
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE user_profile SET theme=? WHERE email=?", (theme, self.logged_in_email))
        conn.commit()
        conn.close()

    def new_conversation(self):
        """
        Starts a new conversation:
        - Adds a single button (no extra icons) to the sidebar list
        - Right‑click context menu with “Rename” and “Delete”
        """
        if len(self.conversations) >= self.max_conversations:
            QMessageBox.warning(self, "Limit Aşıldı", "Maksimum 5 sohbet oluşturabilirsiniz.")
            return

        title = f"Sohbet {len(self.conversations)+1}"
        self.conversations.append([])

        # ---------- DB insert ----------
        from datetime import datetime
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("INSERT INTO conversations (title, created_at) VALUES (?, ?)", (title, now))
        conv_id = cursor.lastrowid
        self.current_conversation_id = conv_id
        conn.commit()
        conn.close()

        # ---------- Sidebar button ----------
        btn = QPushButton(title)
        self.style_button(btn)

        # Ensure conversation_buttons_layout exists
        if not hasattr(self, "conversation_buttons_layout"):
            self.conversation_buttons_layout = QVBoxLayout()
            self.sidebar.addLayout(self.conversation_buttons_layout)

        self.conversation_buttons_layout.addWidget(btn)

        # ---------- Left‑click: load conversation ----------
        def load_conv():
            # Clear display and reload messages from DB
            self.chat_display.clear()
            conn = sqlite3.connect("memory.db")
            cursor = conn.cursor()
            cursor.execute("SELECT speaker, message FROM conversation_logs WHERE conversation_id=? ORDER BY id", (conv_id,))
            rows = cursor.fetchall()
            conn.close()
            for speaker, msg in rows:
                prefix = "<b>SEN:</b>" if speaker == "Kullanıcı" else "<b>BIRD AI:</b>"
                self.chat_display.append(f"{prefix} {msg}")
            self.current_conversation_id = conv_id
        btn.clicked.connect(load_conv)

        # ---------- Right‑click (context menu) ----------
        btn.setContextMenuPolicy(Qt.CustomContextMenu)

        def context_menu(point):
            menu = QMenu()
            act_rename = menu.addAction("Yeniden Adlandır")
            act_delete = menu.addAction("Sil")
            chosen = menu.exec_(btn.mapToGlobal(point))
            if chosen == act_rename:
                new_title, ok = QInputDialog.getText(self, "Yeniden Adlandır", "Yeni sohbet başlığı:")
                if ok and new_title:
                    btn.setText(new_title)
                    conn = sqlite3.connect("memory.db")
                    cursor = conn.cursor()
                    cursor.execute("UPDATE conversations SET title=? WHERE id=?", (new_title, conv_id))
                    conn.commit()
                    conn.close()
            elif chosen == act_delete:
                if QMessageBox.question(self, "Sil", "Bu sohbeti silmek istiyor musunuz?") == QMessageBox.Yes:
                    btn.setParent(None)
                    conn = sqlite3.connect("memory.db")
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
                    cursor.execute("DELETE FROM conversation_logs WHERE conversation_id=?", (conv_id,))
                    conn.commit()
                    conn.close()

        from PyQt5.QtWidgets import QMenu, QInputDialog
        btn.customContextMenuRequested.connect(context_menu)

        # Auto‑open the new conversation
        load_conv()

    def export_conversations(self):
        # Exports all conversations and logs to a JSON file.
        import json
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, title FROM conversations")
        convs = cursor.fetchall()
        all_data = {}
        for cid, title in convs:
            cursor.execute("SELECT speaker, message FROM conversation_logs WHERE conversation_id=?", (cid,))
            all_data[title] = cursor.fetchall()
        conn.close()
        path = f"chatlog_{self.logged_in_email.replace('@','_at_')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        QMessageBox.information(self, "Dışa Aktarıldı", f"{path} dosyasına aktarıldı.")

    def show_all_conversations(self):
        # Shows all conversations and their logs in a dialog window.
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, title FROM conversations ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        dialog = QDialog(self)
        dialog.setWindowTitle("Tüm Sohbetler")
        layout = QVBoxLayout(dialog)
        list_widget = QTextEdit()
        list_widget.setReadOnly(True)

        for row in rows:
            conv_id = row[0]
            conn = sqlite3.connect("memory.db")
            cursor = conn.cursor()
            cursor.execute("SELECT speaker, message FROM conversation_logs WHERE conversation_id=? ORDER BY timestamp", (conv_id,))
            log_entries = cursor.fetchall()
            conn.close()
            list_widget.append(f"--- {row[1]} (ID: {conv_id}) ---")
            for speaker, message in log_entries:
                list_widget.append(f"{speaker}: {message}")
            list_widget.append("\n")

        layout.addWidget(list_widget)
        close_btn = QPushButton("Kapat")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.setLayout(layout)
        dialog.exec_()

    def get_user_profile(self, email=None):
        # Fetches the current user's profile information from the database.
        if email is None:
            email = self.logged_in_email
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT username, language, tone, interests, sources,
                   email, first_name, last_name, birthdate, chat_count, theme
            FROM user_profile WHERE email = ?
        """, (email,))
        result = cursor.fetchone()
        conn.close()

        # Inserted: Calculate age from birthdate, before using in profile dict.
        birthdate = result[9] if result and result[9] else ""
        age = 0
        if birthdate:
            try:
                day, month, year = map(int, birthdate.split("/"))
                today = datetime.today()
                age = today.year - year - ((today.month, today.day) < (month, day))
            except:
                pass

        if result:
            return {
                "username": result[0],
                "language": result[1],
                "tone": result[2],
                "interests": result[3],
                "sources": result[4],
                "email": result[5],
                "first_name": result[6],
                "last_name": result[7],
                "birthdate": birthdate,
                "age": age,
                "chat_count": result[10],
                "theme": result[11]
            }

        return {
            "username": "default",
            "language": "tr",
            "tone": "formal",
            "interests": "",
            "sources": "json,wikipedia,chatgpt",
            "email": "",
            "first_name": "",
            "last_name": "",
            "birthdate": "",
            "age": 0,
            "chat_count": 0,
            "theme": "system"
        }

    def open_profile_settings(self):
        # Opens a dialog to display the user's profile settings (read-only).
        dialog = QDialog(self)
        dialog.setWindowTitle("Profil Ayarları")
        layout = QFormLayout(dialog)

        # Only display all fields as read-only (not editable)
        language_input = QComboBox()
        language_input.addItems(["tr", "en"])
        language_input.setEnabled(False)
        tone_input = QComboBox()
        tone_input.addItems(["formal", "casual"])
        tone_input.setEnabled(False)
        interest_input = QLineEdit()
        interest_input.setReadOnly(True)
        sources_input = QLineEdit()
        sources_input.setReadOnly(True)
        email_input = QLineEdit()
        email_input.setReadOnly(True)
        fname_input = QLineEdit()
        fname_input.setReadOnly(True)
        lname_input = QLineEdit()
        lname_input.setReadOnly(True)
        age_input = QLineEdit()
        age_input.setReadOnly(True)
        birthdate_input = QLineEdit()
        birthdate_input.setReadOnly(True)

        profile = self.get_user_profile()
        language_input.setCurrentText(profile["language"])
        tone_input.setCurrentText(profile["tone"])
        interest_input.setText(profile["interests"])
        sources_input.setText(profile["sources"])
        email_input.setText(profile["email"])
        fname_input.setText(profile["first_name"])
        lname_input.setText(profile["last_name"])
        age_input.setText(str(profile["age"]))
        # Get birthdate if present
        birthdate_val = ""
        try:
            conn = sqlite3.connect("memory.db")
            cursor = conn.cursor()
            cursor.execute("SELECT birthdate FROM user_profile WHERE username='default'")
            row = cursor.fetchone()
            if row and row[0]:
                birthdate_val = row[0]
            conn.close()
        except Exception:
            birthdate_val = ""
        birthdate_input.setText(birthdate_val)

        layout.addRow("Dil:", language_input)
        layout.addRow("Üslup:", tone_input)
        layout.addRow("İlgi Alanları:", interest_input)
        # layout.addRow("Kaynaklar:", sources_input)  # REMOVED as per instructions
        layout.addRow("E-posta:", email_input)
        layout.addRow("Ad:", fname_input)
        layout.addRow("Soyad:", lname_input)
        layout.addRow("Doğum Tarihi:", birthdate_input)
        layout.addRow("Yaş:", age_input)

        # Remove the Save button (no edits allowed)
        # Optionally, add a Close button
        close_button = QPushButton("Kapat")
        close_button.clicked.connect(dialog.accept)
        layout.addRow(close_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def edit_profile_dev_mode(self):
        # Opens an editable profile settings dialog for developer mode.
        # Developer-only: Editable profile settings
        dialog = QDialog(self)
        dialog.setWindowTitle("Profil Ayarları (Geliştirici Modu)")
        layout = QFormLayout(dialog)

        language_input = QComboBox()
        language_input.addItems(["tr", "en"])
        tone_input = QComboBox()
        tone_input.addItems(["formal", "casual"])
        interest_input = QLineEdit()
        sources_input = QLineEdit()
        email_input = QLineEdit()
        fname_input = QLineEdit()
        lname_input = QLineEdit()
        age_input = QLineEdit()
        birthdate_input = QLineEdit()

        profile = self.get_user_profile()
        language_input.setCurrentText(profile["language"])
        tone_input.setCurrentText(profile["tone"])
        interest_input.setText(profile["interests"])
        sources_input.setText(profile["sources"])
        email_input.setText(profile["email"])
        fname_input.setText(profile["first_name"])
        lname_input.setText(profile["last_name"])
        age_input.setText(str(profile["age"]))
        # Get birthdate if present
        birthdate_val = ""
        try:
            conn = sqlite3.connect("memory.db")
            cursor = conn.cursor()
            cursor.execute("SELECT birthdate FROM user_profile WHERE username='default'")
            row = cursor.fetchone()
            if row and row[0]:
                birthdate_val = row[0]
            conn.close()
        except Exception:
            birthdate_val = ""
        birthdate_input.setText(birthdate_val)

        layout.addRow("Dil:", language_input)
        layout.addRow("Üslup:", tone_input)
        layout.addRow("İlgi Alanları:", interest_input)
        layout.addRow("Kaynaklar:", sources_input)
        layout.addRow("E-posta:", email_input)
        layout.addRow("Ad:", fname_input)
        layout.addRow("Soyad:", lname_input)
        layout.addRow("Doğum Tarihi:", birthdate_input)
        layout.addRow("Yaş:", age_input)

        save_button = QPushButton("Kaydet")
        def save_profile():
            conn = sqlite3.connect("memory.db")
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE user_profile SET language=?, tone=?, interests=?, sources=?,
                email=?, first_name=?, last_name=?, age=?, birthdate=?
                WHERE username='default'
            """, (
                language_input.currentText(),
                tone_input.currentText(),
                interest_input.text(),
                sources_input.text(),
                email_input.text(),
                fname_input.text(),
                lname_input.text(),
                int(age_input.text()) if age_input.text().isdigit() else 0,
                birthdate_input.text()
            ))
            conn.commit()
            conn.close()
            QMessageBox.information(dialog, "Başarılı", "Profil güncellendi (Geliştirici Modu).")
            dialog.accept()

        save_button.clicked.connect(save_profile)
        layout.addRow(save_button)

        dialog.setLayout(layout)
        dialog.exec_()

# Developer GUI class for advanced features
# This class is separate from ChatbotGUI to keep the main interface clean.
class DevGUI(QWidget):
    def apply_theme(self, theme):
        if theme == "dark":
            self.setStyleSheet("background-color: #121212; color: #E0E0E0;")
        elif theme == "light":
            self.setStyleSheet("background-color: #FFFFFF; color: #000000;")
        else:
            self.setStyleSheet("")

    def apply_saved_theme_and_language(self):
        """
        Mirrors ChatbotGUI logic: fetch the current user's preferred theme
        (if available) and apply it. DevGUI does not need language switching
        for now, but we keep the interface consistent.
        """
        try:
            profile = self.get_user_profile()
            theme = profile.get("theme", "system")
        except Exception:
            theme = "system"
        self.apply_theme(theme)
    def load_doc_dialog(self):
        """
        Allows the developer to choose a ZIM / PDF / DOCX file and
        ingests its text for later embedding search.
        """
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        self.pdf_chunks = []
        self.pdf_vectorizer = None
        self.pdf_embeddings = None

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Belge Seç",
            "", "ZIM (*.zim);;PDF (*.pdf);;Word (*.docx)"
        )
        if not file_path:
            return

        try:
            if file_path.endswith(".zim"):
                text = self.extract_text_from_zim(file_path)
            elif file_path.endswith(".pdf"):
                text = self.extract_text_from_pdf(file_path)
            else:
                text = self.extract_text_from_docx(file_path)

            if text:
                # Basit parçalara ayırma – 1 000 karakterlik dilimler
                self.pdf_chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                QMessageBox.information(self, "Başarılı", f"{len(self.pdf_chunks)} metin parçası oluşturuldu.")
            else:
                QMessageBox.warning(self, "Boş", "Dosyadan metin çıkarılamadı.")
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Dosya işlenemedi: {e}")
    def extract_text_from_docx(self, docx_path):
        # Extracts and returns all text from a DOCX document.
        doc = Document(docx_path)
        full_text = ""
        for para in doc.paragraphs:
            full_text += para.text + "\n"
        return full_text

    def extract_text_from_zim(self, zim_path):
        # Extracts text from a ZIM file using kiwix-manage.
        """
        ZIM dosyasını doğrudan metne çevirir.
        Şu an terminal tabanlı olarak kiwix-manage ile dump alınır.
        Hata durumunda boş string döner ve log basar.
        """
        import tempfile
        import os
        import logging
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "extracted_zim.txt")
        try:
            # Kiwix-manage ile ZIM dump alınır
            subprocess.run(["kiwix-manage", zim_path, "dump", "--output", output_path], check=True)
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    text = f.read()
                logging.info(f"ZIM dosyası başarıyla işlendi: {zim_path}, {len(text)} karakter.")
                return text
            else:
                logging.warning(f"ZIM dump dosyası oluşmadı: {output_path}")
                return ""
        except Exception as e:
            logging.error(f"ZIM işlenemedi: {e}")
            return ""
    def style_button(self, button, color="#007ACC", text_color="white"):
        # Styles a QPushButton with the given colors.
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
    def show_recent_memory(self, hours=1):
        # Shows memory logs from the last specified hours in a dialog.
        from datetime import datetime, timedelta
        threshold = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, user_input, bot_reply FROM memory_log WHERE timestamp > ?", (threshold,))
        rows = cursor.fetchall()
        conn.close()
        dialog = QDialog(self)
        dialog.setWindowTitle("Son Bellek Kayıtları")
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        for row in rows:
            text.append(f"[{row[0]}]\nSoru: {row[1]}\nCevap: {row[2]}\n")
        layout.addWidget(text)
        close_btn = QPushButton("Kapat")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.setLayout(layout)
        dialog.exec_()
    def __init__(self):
        # Initializes the ChatbotGUI and sets up the UI and user session.
        self.layout_yuklendi = False
        super().__init__()
        self.setWindowTitle(VERSION)
        self.setGeometry(100, 100, 800, 500)
        initialize_memory_db()
        initialize_user_profile()
        self.context_history = []
        self.max_context_length = 10
        self.conversations = []
        self.max_conversations = 5
        self.current_conversation_id = None
        self.settings_mode = False
        self.current_theme = "system"
        self.memory_mode = "aktif"  # aktif | geçici
        self.emotion_mode = "neutral"  # neutral, happy, sad, angry
        self.model_version = "BIRD AI 1.5"
        self.logged_in_email = None
        # Disable shimmer effect for all labels in init
        # (No shimmer logic, but ensure shimmer_label does nothing)
        def dummy_shimmer(label):
            label.setStyleSheet("color: red; font-weight: bold;")
        self.shimmer_label = dummy_shimmer
        self.dev_mode_enabled = False
        self.init_ui()
        self.apply_saved_theme_and_language()



    def apply_saved_theme_and_language(self):
        # Applies the saved theme and language from the user profile.
        profile = self.get_user_profile()
            # Apply saved theme
        if profile.get("theme", "system") == "dark":
            self.apply_theme("dark")
        elif profile.get("theme", "light") == "light":
            self.apply_theme("light")
        elif profile.get("theme", "system") == "system":
            self.apply_theme("system")
                # Language kaydı her girişte fetch ediliyor

    # (No changes to DevGUI's init_ui needed for this patch)
    def show_gmail_users(self):
        """
        Displays all users with a Gmail address in a searchable dialog.
        """
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT username, email, first_name, last_name, birthdate, age, language, tone, interests, sources, chat_count FROM user_profile WHERE email LIKE '%@gmail.com'")
        rows = cursor.fetchall()
        conn.close()

        dlg = QDialog(self)
        dlg.setWindowTitle("Gmail Kullanıcıları")
        vbox = QVBoxLayout(dlg)

        search_bar = QLineEdit()
        search_bar.setPlaceholderText("Gmail adresi ara...")
        vbox.addWidget(search_bar)

        text = QTextEdit()
        text.setReadOnly(True)
        vbox.addWidget(text)

        def refresh_display(filter_text=""):
            text.clear()
            for r in rows:
                if filter_text.lower() in r[1].lower():
                    text.append(
                        f"Username: {r[0]}\n"
                        f"Email: {r[1]}\n"
                        f"Ad: {r[2]} {r[3]}\n"
                        f"Doğum Tarihi: {r[4]}  Yaş: {r[5]}\n"
                        f"Dil: {r[6]}  Üslup: {r[7]}\n"
                        f"İlgi Alanları: {r[8]}\n"
                        f"Kaynaklar: {r[9]}\n"
                        f"Sohbet Sayısı: {r[10]}\n"
                        "-----------------------------"
                    )

        refresh_display()

        search_bar.textChanged.connect(lambda t: refresh_display(t))

        close_btn = QPushButton("Kapat")
        close_btn.clicked.connect(dlg.accept)
        vbox.addWidget(close_btn)

        dlg.setLayout(vbox)
        dlg.exec_()

        # --- Zim file upload button ---
        zim_btn = QPushButton("ZIM Yükle")
        self.style_button(zim_btn, color="#2980b9")
        zim_btn.clicked.connect(self.load_zim_file)
        self.sidebar.addWidget(zim_btn) 

        # ---pdf/docx file upload button ---
        doc_btn = QPushButton("PDF/DOCX Yükle")
        self.style_button(doc_btn, color="#27ae60")
        doc_btn.clicked.connect(self.load_doc_dialog)
        self.sidebar.addWidget(doc_btn)

        self.new_chat_button = QPushButton("Yeni Sohbet")
        self.style_button(self.new_chat_button)
        self.new_chat_button.clicked.connect(self.new_conversation)
        self.sidebar.addWidget(self.new_chat_button)

        self.memory_toggle = QPushButton(f"Hafıza: {self.memory_mode.capitalize()}")
        self.style_button(self.memory_toggle)
        def toggle_memory():
            self.memory_mode = "geçici" if self.memory_mode == "aktif" else "aktif"
            self.memory_toggle.setText(f"Hafıza: {self.memory_mode.capitalize()}")
        self.memory_toggle.clicked.connect(toggle_memory)
        self.sidebar.addWidget(self.memory_toggle)

        # Emotion mode selector
        self.emotion_select = QComboBox()
        self.emotion_select.addItems(["neutral", "happy", "sad", "angry"])
        self.emotion_select.currentTextChanged.connect(self.update_emotion_mode)
        self.sidebar.addWidget(QLabel("Duygu Modu:"))
        self.sidebar.addWidget(self.emotion_select)

        # Model version selector
        self.model_select = QComboBox()
        self.model_select.addItems(["Thunderbird AI Alpha 1.0", "Thunderbird AI Beta 1.1", "BIRD AI Alpha 1.5", "Thunderbird AI Beta 2.0"])
        self.model_select.currentTextChanged.connect(lambda m: setattr(self, "model_version", m))
        self.sidebar.addWidget(QLabel("Model Seçimi:"))
        self.sidebar.addWidget(self.model_select)

        self.conversation_buttons_layout = QVBoxLayout()
        self.sidebar.addLayout(self.conversation_buttons_layout)

        # Logout button
        self.logout_button = QPushButton("Hesaptan Çıkış Yap")
        self.style_button(self.logout_button)
        def logout():
            self.logged_in_email = None
            conn = sqlite3.connect("memory.db")
            cursor = conn.cursor()
            cursor.execute("UPDATE user_profile SET email='' WHERE username='default'")
            conn.commit()
            conn.close()
        self.logout_button.clicked.connect(logout)
        self.sidebar.addWidget(self.logout_button)

        self.exit_button = QPushButton("Uygulamayı Kapat")
        self.style_button(self.exit_button)
        self.exit_button.clicked.connect(self.close)
        self.sidebar.addWidget(self.exit_button)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(self.sidebar)
        sidebar_widget.setFixedWidth(150)

        # Chat Area
        self.chat_layout = QVBoxLayout()

        # Remove the toggle sidebar button at the top (toggle_btn)
        # Instead, always keep hamburger_button visible, placed in main layout

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.input_field = QLineEdit()
        self.input_field.returnPressed.connect(self.soru_sor)
        self.chat_layout.addWidget(self.chat_display)
        self.chat_layout.addWidget(self.input_field)

        # Add suggestion label below the input field
        self.suggestion_label = QLabel("")
        self.chat_layout.addWidget(self.suggestion_label)

        chat_widget = QWidget()
        chat_widget.setLayout(self.chat_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(sidebar_widget)
        splitter.addWidget(chat_widget)

        layout.addWidget(splitter)

        # Always visible hamburger button (bottom-left)
        self.hamburger_button = QPushButton("☰")
        self.style_button(self.hamburger_button, color="#333", text_color="white")
        def toggle_sidebar():
            sidebar_widget.setVisible(not sidebar_widget.isVisible())
        self.hamburger_button.clicked.connect(toggle_sidebar)
        layout.addWidget(self.hamburger_button)

        self.setLayout(layout)
        self.layout_yuklendi = True

    def enter_developer_mode(self):
        # Prompts for developer mode password and enables dev mode if correct.
        auth_dialog = QDialog(self)
        auth_dialog.setWindowTitle("Developer Girişi")
        layout = QVBoxLayout()

        password_input = QLineEdit()
        password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(QLabel("Şifre:"))
        layout.addWidget(password_input)

        status_label = QLabel("")
        layout.addWidget(status_label)

        def check_password():
            if password_input.text() == "kyra1453":
                QMessageBox.information(self, "Başarılı", "Developer moduna giriş yapılıyor...")
                auth_dialog.accept()
                self.enable_dev_mode()
            else:
                status_label.setText("ACCESS DENIED")
                auth_dialog.reject()

        login_btn = QPushButton("Giriş Yap")
        login_btn.clicked.connect(check_password)
        self.style_button(login_btn, color="red")

        layout.addWidget(login_btn)
        auth_dialog.setLayout(layout)
        auth_dialog.exec_()


    def load_zim_file(self):
        # Loads and processes a ZIM file for embedding search.
        self.pdf_chunks = []
        self.pdf_vectorizer = None
        self.pdf_embeddings = None
        from PyQt5.QtWidgets import QFileDialog
        try:
            zim_path, _ = QFileDialog.getOpenFileName(self, "ZIM Dosyası Seç", "", "ZIM Dosyaları (*.zim)")
            if zim_path:
                QMessageBox.information(self, "ZIM Yükleniyor", "ZIM dosyası yükleniyor ve işleniyor. Lütfen bekleyin...")
                self.worker = EmbeddingWorker(self, zim_path, "zim")
                self.worker.finished.connect(self.handle_embedding_result)
                self.worker.start()
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"ZIM dosyası yüklenemedi: {e}")

    def load_pdf_document(self):
        # Loads and processes a PDF or DOCX file for embedding search.
        self.pdf_chunks = []
        self.pdf_vectorizer = None
        self.pdf_embeddings = None
        from PyQt5.QtWidgets import QFileDialog
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Belge Seç", "", "PDF veya Word Dosyaları (*.pdf *.docx)")
            if file_name:
                if file_name.endswith(".pdf"):
                    file_type = "pdf"
                elif file_name.endswith(".docx"):
                    file_type = "docx"
                else:
                    QMessageBox.warning(self, "Hata", "Sadece PDF veya DOCX dosyası seçebilirsiniz.")
                    return
                QMessageBox.information(self, "Belge Yükleniyor", f"{file_type.upper()} dosyası yükleniyor ve işleniyor. Lütfen bekleyin...")
                self.worker = EmbeddingWorker(self, file_name, file_type)
                self.worker.finished.connect(self.handle_embedding_result)
                self.worker.start()
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Belge yüklenemedi: {e}")

    def handle_embedding_result(self, result):
        # Handles the result of the embedding worker and notifies the user.
        if result == "success":
            QMessageBox.information(self, "Başarılı", f"{len(self.pdf_chunks)} parça oluşturuldu.")
        elif result == "empty":
            QMessageBox.warning(self, "Başarısız", "Dosyadan içerik çıkarılamadı.")
        elif result == "unsupported":
            QMessageBox.warning(self, "Hata", "Geçersiz dosya türü.")
        elif result.startswith("error:"):
            QMessageBox.warning(self, "Hata", f"İşlem hatası: {result}")
        else:
            QMessageBox.warning(self, "Hata", f"İşlem sırasında bilinmeyen bir hata oluştu: {result}")
        self.worker = None

    def show_memory_log(self):
        # Shows the last 50 memory log entries in a dialog.
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, user_input, bot_reply, source FROM memory_log ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()

        dialog = QDialog(self)
        dialog.setWindowTitle("Bellek Kayıtları")
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        for row in rows:
            text.append(f"[{row[0]}] ({row[3]})\nSoru: {row[1]}\nCevap: {row[2]}\n")
        layout.addWidget(text)

        close_btn = QPushButton("Kapat")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec_()

    def add_qa_popup(self):
        # Opens a dialog to manually add a question-answer pair to memory.
        dialog = QDialog(self)
        dialog.setWindowTitle("Soru-Cevap Ekle")
        layout = QVBoxLayout(dialog)

        soru_input = QTextEdit()
        soru_input.setPlaceholderText("Soru")
        cevap_input = QTextEdit()
        cevap_input.setPlaceholderText("Cevap")
        layout.addWidget(QLabel("Soru:"))
        layout.addWidget(soru_input)
        layout.addWidget(QLabel("Cevap:"))
        layout.addWidget(cevap_input)

        def kaydet():
            soru = soru_input.toPlainText().strip()
            cevap = cevap_input.toPlainText().strip()
            if soru and cevap:
                log_to_memory(soru, cevap, "manual")
                QMessageBox.information(dialog, "Başarılı", "Soru-Cevap kaydedildi.")
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Hata", "Boş bırakılamaz.")

        save_btn = QPushButton("Kaydet")
        save_btn.clicked.connect(kaydet)
        layout.addWidget(save_btn)

        cancel_btn = QPushButton("İptal")
        cancel_btn.clicked.connect(dialog.reject)
        layout.addWidget(cancel_btn)

        dialog.setLayout(layout)
        dialog.exec_()

    def show_json_popup(self):
        # Shows all manual Q&A pairs from memory as JSON in a dialog.
        dialog = QDialog(self)
        dialog.setWindowTitle("JSON Görüntüle (veritabanından alınan kayıtlar)")
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)

        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT user_input, bot_reply FROM memory_log WHERE source = 'manual'")
        rows = cursor.fetchall()
        conn.close()

        json_entries = [{"soru": row[0], "cevap": row[1]} for row in rows]
        import json
        text.setText(json.dumps(json_entries, indent=2, ensure_ascii=False))
        layout.addWidget(text)

        close_btn = QPushButton("Kapat")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec_()

    def soru_sor(self):
        # Handles user input, generates a response, and updates memory and UI.
        soru = self.input_field.text().strip()
        if not soru:
            return
        profile = self.get_user_profile()
        if profile["chat_count"] >= 5:
            QMessageBox.warning(self, "Limit Aşıldı", "Maksimum 5 sohbet hakkınızı kullandınız.")
            return
        self.chat_display.append(f"<div style='border:1px solid #ccc; padding:5px; border-radius:8px;'><b>SEN:</b> {soru}</div>")
        self.input_field.clear()

        context_text = ""
        for sender, msg in self.context_history[-self.max_context_length:]:
            context_text += f"{sender}: {msg}\n"
        genişletilmiş = tahmini_prompt_olustur(soru, self.context_history)
        # Add emotional and tone cues
        tone_text = ""
        if profile["tone"] == "formal":
            tone_text += "Cevapların resmi bir üslupla olsun. "
        elif profile["tone"] == "casual":
            tone_text += "Cevapların samimi ve günlük dilde olsun. "

        if self.emotion_mode == "happy":
            tone_text += "Cevapların neşeli ve pozitif bir şekilde verilsin."
        elif self.emotion_mode == "sad":
            tone_text += "Cevapların biraz daha düşünceli ve ciddi olsun."
        elif self.emotion_mode == "angry":
            tone_text += "Cevapların biraz daha sert ve kısa olsun."
        else:
            tone_text += "Cevapların dengeli ve tarafsız olsun."

        context_text += f"Kullanıcı: {genişletilmiş}\nYönerge: {tone_text}"
        genişletilmiş_soru = context_text

        # Kişisel ilgi alanı odaklı prompt ekle
        if profile["interests"]:
            interest_hint = profile["interests"].split(",")[0].strip()
            genişletilmiş_soru = f"Kullanıcının ilgilendiği konu: {interest_hint}\n{genişletilmiş_soru}"

        cevap = None
        try:
            # Try Embedding Search first
            # Embedding similarity search
            if self.dev_mode_enabled:
                if hasattr(self, 'pdf_vectorizer') and hasattr(self, 'pdf_embeddings') and self.pdf_embeddings is not None:
                    from sklearn.metrics.pairwise import cosine_similarity
                    user_embedding = self.pdf_vectorizer.transform([soru])
                    scores = cosine_similarity(user_embedding, self.pdf_embeddings)
                    max_score = scores.max()
                    if max_score >= 0.88649827630184394:
                        best_idx = scores.argmax()
                        cevap = self.pdf_chunks[best_idx]
                    else:
                        cevap = None
                else:
                    cevap = None
        except Exception as e:
            QMessageBox.warning(self, "Embedding Hatası", f"Embedding işlemi başarısız: {e}")
            cevap = None

        try:
            # Only if cevap is still None, proceed with JSON, Wikipedia, ChatGPT
            # Manual Q&A (JSON) lookup
            if cevap is None:
                if "json" in profile["sources"]:
                    conn = sqlite3.connect("memory.db")
                    cursor = conn.cursor()
                    cursor.execute("SELECT bot_reply FROM memory_log WHERE user_input = ? AND source = 'manual'", (soru,))
                    result = cursor.fetchone()
                    conn.close()
                    if result:
                        cevap = result[0]
        except Exception as e:
            QMessageBox.warning(self, "Bellek Hatası", f"JSON kaynağından veri alınamadı: {e}")

        try:
            # Wikipedia search
            if cevap is None and "wikipedia" in profile["sources"]:
                cevap = wikipedia_bilgi_getir(soru, profile["language"])
        except Exception as e:
            QMessageBox.warning(self, "Wikipedia Hatası", f"Wikipedia'dan veri alınamadı: {e}")

        try:
            # ChatGPT API call
            if cevap is None and "chatgpt" in profile["sources"]:
                cevap = chatgpt_api_sorgula(genişletilmiş_soru, profile["language"])
        except Exception as e:
            QMessageBox.warning(self, "ChatGPT Hatası", f"ChatGPT'den veri alınamadı: {e}")

        if cevap and "GPT" in cevap:
            log_to_memory(soru, cevap, "chatgpt")

        if not cevap:
            cevap = "(Hiçbir kaynaktan anlamlı cevap alınamadı.)"

        self.chat_display.append(f"<div style='border:1px solid #ccc; padding:5px; border-radius:8px;'><b>BIRD AI:</b> {cevap}</div>")

        if self.memory_mode == "aktif":
            self.context_history.append(("Kullanıcı", soru))
            self.context_history.append(("BIRD AI", cevap))
            if len(self.context_history) > self.max_context_length:
                self.context_history = self.context_history[-self.max_context_length:]
        # Add user profile memory to context
        self.context_history.append(("Profil", f"{profile['first_name']} {profile['last_name']} - {profile['email']} Yaş:{profile['age']}"))
        # Conversation history for multi-conversation
        if self.conversations:
            self.conversations[-1].append(("Kullanıcı", soru))
            self.conversations[-1].append(("BIRD AI", cevap))
        # Log conversation to memory
        log_to_memory(soru, cevap, "chat")

        # Save chat to conversation_logs if conversation_id exists
        try:
            # Insert chat into conversation_logs table
            if self.current_conversation_id:
                from datetime import datetime
                conn = sqlite3.connect("memory.db")
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                cursor.execute("INSERT INTO conversation_logs (conversation_id, speaker, message, timestamp) VALUES (?, ?, ?, ?)",
                               (self.current_conversation_id, "Kullanıcı", soru, now))
                cursor.execute("INSERT INTO conversation_logs (conversation_id, speaker, message, timestamp) VALUES (?, ?, ?, ?)",
                               (self.current_conversation_id, "BIRD AI", cevap, now))
                conn.commit()
                conn.close()
        except Exception as e:
            QMessageBox.warning(self, "Sohbet Kaydı Hatası", f"Sohbet kaydedilemedi: {e}")

        try:
            # Update user's chat count
            conn = sqlite3.connect("memory.db")
            cursor = conn.cursor()
            cursor.execute("UPDATE user_profile SET chat_count = chat_count + 1 WHERE username = 'default'")
            conn.commit()
            conn.close()
        except Exception as e:
            QMessageBox.warning(self, "Kullanıcı Güncelleme Hatası", f"Kullanıcı sohbet sayısı güncellenemedi: {e}")

        # Suggest follow-up questions based on interests, emotion and tone
        suggestion = ""
        if profile["interests"]:
            interest = profile["interests"].split(",")[0]
            suggestion = f"{interest.strip().capitalize()} hakkında daha fazla bilgi ister misin?"
        elif self.emotion_mode == "happy":
            suggestion = "Yeni bir şey öğrenmek ister misin?"
        elif self.emotion_mode == "sad":
            suggestion = "İstersen moralini yükseltecek bir bilgi verebilirim."
        elif self.emotion_mode == "angry":
            suggestion = "Sakinleşmek için istersen bir fıkra anlatayım mı?"
        else:
            suggestion = "Başka bir konuda yardımcı olabilir miyim?"
        self.suggestion_label.setText(f"<i>Öneri: {suggestion}</i>")

    def clear_memory_log(self):
        # Clears all memory log entries from the database.
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memory_log")
        conn.commit()
        conn.close()
        QMessageBox.information(self, "Silindi", "Bellek verisi temizlendi.")

    def clear_context_history(self):
        # Clears the conversation context history.
        self.context_history.clear()
        self.chat_display.append("<i>Konuşma geçmişi temizlendi.</i>")

    def clear_sidebar(self):
        # Removes all widgets from the sidebar.
        for i in reversed(range(self.sidebar.count())):
            widget = self.sidebar.itemAt(i).widget()
            if widget:
                widget.setParent(None)

    def open_settings_panel(self):
        # Opens the settings dialog for theme, language, and profile.
        splash = QDialog(self)
        splash.setWindowTitle("Ayarlar")
        layout = QVBoxLayout()

        theme_btn = QPushButton("Tema Seç")
        theme_btn.clicked.connect(lambda: [splash.accept(), self.open_theme_selector()])
        self.style_button(theme_btn)
        layout.addWidget(theme_btn)

        lang_btn = QPushButton("Dil Seç")
        lang_btn.clicked.connect(lambda: [splash.accept(), self.open_lang_selector()])
        self.style_button(lang_btn)
        layout.addWidget(lang_btn)

        # Add Profil Ayarları button inside the splash screen
        profile_btn = QPushButton("Profil Ayarları")
        self.style_button(profile_btn)
        profile_btn.clicked.connect(lambda: [splash.accept(), self.open_profile_settings()])
        layout.addWidget(profile_btn)

        # Removed memory mode selection to prevent changing memory mode
        # storage_label = QLabel("Cevapları Kaydet:")
        # storage_mode_select = QComboBox()
        # storage_mode_select.addItems(["SQL (varsayılan)", "Kaydetme"])
        # layout.addWidget(storage_label)
        # layout.addWidget(storage_mode_select)
        #
        # def update_storage():
        #     selected = storage_mode_select.currentText()
        #     self.memory_mode = "aktif" if "SQL" in selected else "geçici"
        # storage_mode_select.currentTextChanged.connect(update_storage)

        back_btn = QPushButton("← Geri")
        self.style_button(back_btn)
        back_btn.clicked.connect(lambda: [splash.accept()])
        layout.addWidget(back_btn)

        splash.setLayout(layout)
        splash.exec_()

    def open_theme_selector(self):
        # Opens the theme selection as a dialog window instead of modifying the sidebar.
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QRadioButton, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Tema Seç")
        layout = QVBoxLayout(dialog)

        light_theme_btn = QRadioButton("Açık Tema")
        dark_theme_btn = QRadioButton("Koyu Tema")
        system_theme_btn = QRadioButton("Sistem Teması")

        layout.addWidget(light_theme_btn)
        layout.addWidget(dark_theme_btn)
        layout.addWidget(system_theme_btn)

        # Pre-select the current theme
        if self.current_theme == "light":
            light_theme_btn.setChecked(True)
        elif self.current_theme == "dark":
            dark_theme_btn.setChecked(True)
        else:
            system_theme_btn.setChecked(True)

        def set_theme():
            if light_theme_btn.isChecked():
                self.apply_theme("light")
            elif dark_theme_btn.isChecked():
                self.apply_theme("dark")
            else:
                self.apply_theme("system")

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(lambda: [set_theme(), dialog.accept()])
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)
        dialog.exec_()

    def open_lang_selector(self):
        # Opens the language selector as a dialog window instead of sidebar.
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QRadioButton, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Dil Seç")
        layout = QVBoxLayout(dialog)

        tr_lang_btn = QRadioButton("Türkçe")
        en_lang_btn = QRadioButton("İngilizce")

        layout.addWidget(tr_lang_btn)
        layout.addWidget(en_lang_btn)

        profile = self.get_user_profile()
        if profile["language"] == "en":
            en_lang_btn.setChecked(True)
        else:
            tr_lang_btn.setChecked(True)

        def set_language():
            if tr_lang_btn.isChecked():
                self.change_language("tr")
            elif en_lang_btn.isChecked():
                self.change_language("en")

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(lambda: [set_language(), dialog.accept()])
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)
        dialog.exec_()

    def open_personal_theme_settings(self):
        # Opens a color picker to set a custom theme color.
        from PyQt5.QtWidgets import QColorDialog
        color = QColorDialog.getColor()
        if color.isValid():
            self.setStyleSheet(f"background-color: {color.name()}; color: #000;")

    def change_language(self, lang):
        # Changes the user profile language setting.
        profile = self.get_user_profile()
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE user_profile SET language=? WHERE email=?", (lang, self.logged_in_email))
        conn.commit()
        conn.close()

    def update_emotion_mode(self, mood):
        # Updates the current emotion mode and sidebar label.
        self.emotion_mode = mood
        # Optionally update theme or other UI elements here if needed
        # Change sidebar label icon to reflect emotion
        if hasattr(self, 'sidebar_label'):
            emoji = {
                "happy": "🙂",
                "sad": "😕",
                "angry": "😠",
                "neutral": "😐"
            }.get(mood, "😐")
            self.sidebar_label.setText(f"BIRD AI {emoji}")

    def apply_theme(self, theme):
        # Applies the selected theme to the application and saves it.
        if theme == "dark":
            self.setStyleSheet("background-color: #2C3E50; color: #ECF0F1;")
        elif theme == "light":
            self.setStyleSheet("background-color: #ECF0F1; color: #2C3E50;")
        else:
            self.setStyleSheet("")
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE user_profile SET theme=? WHERE email=?", (theme, self.logged_in_email))
        conn.commit()
        conn.close()

    def new_conversation(self):
        # Starts a new conversation and adds it to sidebar and database.
        if len(self.conversations) >= self.max_conversations:
            QMessageBox.warning(self, "Limit Aşıldı", "Maksimum 5 sohbet oluşturabilirsiniz.")
            return
        title = f"Sohbet {len(self.conversations)+1}"
        self.conversations.append([])
        # Store new conversation in database
        # Insert new conversation into database
        from datetime import datetime
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("INSERT INTO conversations (title, created_at) VALUES (?, ?)", (title, now))
        conv_id = cursor.lastrowid
        self.current_conversation_id = conv_id
        conn.commit()
        conn.close()

        button = QPushButton(title)
        self.style_button(button)

        def load_conv():
            self.chat_display.clear()
            for speaker, text in self.conversations[-1]:
                prefix = "<b>SEN:</b>" if speaker == "Kullanıcı" else "<b>BIRD AI:</b>"
                self.chat_display.append(f"{prefix} {text}")

        def rename_conv():
            from PyQt5.QtWidgets import QInputDialog
            new_title, ok = QInputDialog.getText(self, "Yeniden Adlandır", "Yeni sohbet başlığı:")
            if ok and new_title:
                button.setText(new_title)
                conn = sqlite3.connect("memory.db")
                cursor = conn.cursor()
                cursor.execute("UPDATE conversations SET title=? WHERE id=?", (new_title, self.current_conversation_id))
                conn.commit()
                conn.close()

        def delete_conv():
            confirm = QMessageBox.question(self, "Sil", "Bu sohbet sadece ekrandan silinecek. Emin misin?")
            if confirm == QMessageBox.Yes:
                button.setParent(None)
                self.conversations[-1].clear()

        menu_btn = QPushButton("⋮")
        menu_btn.setMaximumWidth(30)
        self.style_button(menu_btn, color="#aaa", text_color="#222")
        menu_btn.clicked.connect(rename_conv)  # could toggle menu instead
        del_btn = QPushButton("🗑")
        del_btn.setMaximumWidth(30)
        self.style_button(del_btn, color="#e74c3c", text_color="white")
        del_btn.clicked.connect(delete_conv)

        conv_layout = QHBoxLayout()
        conv_layout.addWidget(button)
        conv_layout.addWidget(menu_btn)
        conv_layout.addWidget(del_btn)
        conv_layout.setAlignment(Qt.AlignLeft)

        container = QWidget()
        container.setLayout(conv_layout)
        self.conversation_buttons_layout.addWidget(container)

        button.clicked.connect(load_conv)
        load_conv()

    def export_conversations(self):
        # Exports all conversations and logs to a JSON file.
        import json
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, title FROM conversations")
        convs = cursor.fetchall()
        all_data = {}
        for cid, title in convs:
            cursor.execute("SELECT speaker, message FROM conversation_logs WHERE conversation_id=?", (cid,))
            all_data[title] = cursor.fetchall()
        conn.close()
        path = f"chatlog_{self.logged_in_email.replace('@','_at_')}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        QMessageBox.information(self, "Dışa Aktarıldı", f"{path} dosyasına aktarıldı.")

    def show_all_conversations(self):
        # Shows all conversations and their logs in a dialog window.
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, title FROM conversations ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        dialog = QDialog(self)
        dialog.setWindowTitle("Tüm Sohbetler")
        layout = QVBoxLayout(dialog)
        list_widget = QTextEdit()
        list_widget.setReadOnly(True)

        for row in rows:
            conv_id = row[0]
            conn = sqlite3.connect("memory.db")
            cursor = conn.cursor()
            cursor.execute("SELECT speaker, message FROM conversation_logs WHERE conversation_id=? ORDER BY timestamp", (conv_id,))
            log_entries = cursor.fetchall()
            conn.close()
            list_widget.append(f"--- {row[1]} (ID: {conv_id}) ---")
            for speaker, message in log_entries:
                list_widget.append(f"{speaker}: {message}")
            list_widget.append("\n")

        layout.addWidget(list_widget)
        close_btn = QPushButton("Kapat")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.setLayout(layout)
        dialog.exec_()

    def get_user_profile(self, email=None):
        # Fetches the current user's profile information from the database.
        if email is None:
            email = self.logged_in_email
        conn = sqlite3.connect("memory.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT username, language, tone, interests, sources,
                   email, first_name, last_name, birthdate, chat_count, theme
            FROM user_profile WHERE email = ?
        """, (email,))
        result = cursor.fetchone()
        conn.close()

        # Inserted: Calculate age from birthdate, before using in profile dict.
        birthdate = result[9] if result and result[9] else ""
        age = 0
        if birthdate:
            try:
                day, month, year = map(int, birthdate.split("/"))
                today = datetime.today()
                age = today.year - year - ((today.month, today.day) < (month, day))
            except:
                pass

        if result:
            return {
                "username": result[0],
                "language": result[1],
                "tone": result[2],
                "interests": result[3],
                "sources": result[4],
                "email": result[5],
                "first_name": result[6],
                "last_name": result[7],
                "birthdate": birthdate,
                "age": age,
                "chat_count": result[10],
                "theme": result[11]
            }

        return {
            "username": "default",
            "language": "tr",
            "tone": "formal",
            "interests": "",
            "sources": "json,wikipedia,chatgpt",
            "email": "",
            "first_name": "",
            "last_name": "",
            "birthdate": "",
            "age": 0,
            "chat_count": 0,
            "theme": "system"
        }

    def open_profile_settings(self):
        # Opens a dialog to display the user's profile settings (read-only).
        dialog = QDialog(self)
        dialog.setWindowTitle("Profil Ayarları")
        layout = QFormLayout(dialog)

        # Only display all fields as read-only (not editable)
        language_input = QComboBox()
        language_input.addItems(["tr", "en"])
        language_input.setEnabled(False)
        tone_input = QComboBox()
        tone_input.addItems(["formal", "casual"])
        tone_input.setEnabled(False)
        interest_input = QLineEdit()
        interest_input.setReadOnly(True)
        sources_input = QLineEdit()
        sources_input.setReadOnly(True)
        email_input = QLineEdit()
        email_input.setReadOnly(True)
        fname_input = QLineEdit()
        fname_input.setReadOnly(True)
        lname_input = QLineEdit()
        lname_input.setReadOnly(True)
        age_input = QLineEdit()
        age_input.setReadOnly(True)
        birthdate_input = QLineEdit()
        birthdate_input.setReadOnly(True)

        profile = self.get_user_profile()
        language_input.setCurrentText(profile["language"])
        tone_input.setCurrentText(profile["tone"])
        interest_input.setText(profile["interests"])
        sources_input.setText(profile["sources"])
        email_input.setText(profile["email"])
        fname_input.setText(profile["first_name"])
        lname_input.setText(profile["last_name"])
        age_input.setText(str(profile["age"]))
        # Get birthdate if present
        birthdate_val = ""
        try:
            conn = sqlite3.connect("memory.db")
            cursor = conn.cursor()
            cursor.execute("SELECT birthdate FROM user_profile WHERE username='default'")
            row = cursor.fetchone()
            if row and row[0]:
                birthdate_val = row[0]
            conn.close()
        except Exception:
            birthdate_val = ""
        birthdate_input.setText(birthdate_val)

        layout.addRow("Dil:", language_input)
        layout.addRow("Üslup:", tone_input)
        layout.addRow("İlgi Alanları:", interest_input)
        # layout.addRow("Kaynaklar:", sources_input)  # REMOVED as per instructions
        layout.addRow("E-posta:", email_input)
        layout.addRow("Ad:", fname_input)
        layout.addRow("Soyad:", lname_input)
        layout.addRow("Doğum Tarihi:", birthdate_input)
        layout.addRow("Yaş:", age_input)

        # Remove the Save button (no edits allowed)
        # Optionally, add a Close button
        close_button = QPushButton("Kapat")
        close_button.clicked.connect(dialog.accept)
        layout.addRow(close_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def edit_profile_dev_mode(self):
        # Opens an editable profile settings dialog for developer mode.
        # Developer-only: Editable profile settings
        dialog = QDialog(self)
        dialog.setWindowTitle("Profil Ayarları (Geliştirici Modu)")
        layout = QFormLayout(dialog)

        language_input = QComboBox()
        language_input.addItems(["tr", "en"])
        tone_input = QComboBox()
        tone_input.addItems(["formal", "casual"])
        interest_input = QLineEdit()
        sources_input = QLineEdit()
        email_input = QLineEdit()
        fname_input = QLineEdit()
        lname_input = QLineEdit()
        age_input = QLineEdit()
        birthdate_input = QLineEdit()

        profile = self.get_user_profile()
        language_input.setCurrentText(profile["language"])
        tone_input.setCurrentText(profile["tone"])
        interest_input.setText(profile["interests"])
        sources_input.setText(profile["sources"])
        email_input.setText(profile["email"])
        fname_input.setText(profile["first_name"])
        lname_input.setText(profile["last_name"])
        age_input.setText(str(profile["age"]))
        # Get birthdate if present
        birthdate_val = ""
        try:
            conn = sqlite3.connect("memory.db")
            cursor = conn.cursor()
            cursor.execute("SELECT birthdate FROM user_profile WHERE username='default'")
            row = cursor.fetchone()
            if row and row[0]:
                birthdate_val = row[0]
            conn.close()
        except Exception:
            birthdate_val = ""
        birthdate_input.setText(birthdate_val)

        layout.addRow("Dil:", language_input)
        layout.addRow("Üslup:", tone_input)
        layout.addRow("İlgi Alanları:", interest_input)
        layout.addRow("Kaynaklar:", sources_input)
        layout.addRow("E-posta:", email_input)
        layout.addRow("Ad:", fname_input)
        layout.addRow("Soyad:", lname_input)
        layout.addRow("Doğum Tarihi:", birthdate_input)
        layout.addRow("Yaş:", age_input)

        save_button = QPushButton("Kaydet")
        def save_profile():
            conn = sqlite3.connect("memory.db")
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE user_profile SET language=?, tone=?, interests=?, sources=?,
                email=?, first_name=?, last_name=?, age=?, birthdate=?
                WHERE username='default'
            """, (
                language_input.currentText(),
                tone_input.currentText(),
                interest_input.text(),
                sources_input.text(),
                email_input.text(),
                fname_input.text(),
                lname_input.text(),
                int(age_input.text()) if age_input.text().isdigit() else 0,
                birthdate_input.text()
            ))
            conn.commit()
            conn.close()
            QMessageBox.information(dialog, "Başarılı", "Profil güncellendi (Geliştirici Modu).")
            dialog.accept()

        save_button.clicked.connect(save_profile)
        layout.addRow(save_button)

        dialog.setLayout(layout)
        dialog.exec_()



# Main application entry point
# This section initializes the application, displays a splash screen, and starts the main GUI.
if __name__ == "__main__":
    print("App is starting...")
    app = QApplication(sys.argv)
    # Initialize splash screen
    splash = QSplashScreen()
    splash.setFixedSize(300, 180)
    splash.setWindowFlags(Qt.FramelessWindowHint)	
    splash.setStyleSheet("background-color: black; color: white;")
    splash.setFont(QFont("Arial", 16, QFont.Bold))

    # Layout for splash
    layout = QVBoxLayout(splash)
    version_label = QLabel(VERSION)
    version_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
    version_label.setStyleSheet("color: white; font-size: 18pt;")
    layout.addWidget(version_label)

    status_label = QLabel("Booting...")
    status_label.setAlignment(Qt.AlignCenter)
    # Ensure status_label is static
    status_label.setStyleSheet("color: white; font-size: 14pt; font-weight: bold;")
    layout.addWidget(status_label)

    progress = QProgressBar()
    progress.setMaximum(100)
    progress.setValue(0)
    progress.setStyleSheet("QProgressBar {background-color: #444; color: white;}")
    layout.addWidget(progress)

    # Author label at bottom right with margin
    from PyQt5.QtWidgets import QSpacerItem, QSizePolicy
    author_label = QLabel("by FOX")
    author_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
    author_label.setStyleSheet("color: gray; font-size: 10pt; margin-right: 18px; margin-bottom: 10px;")

    # Spacer to push author_label to bottom
    spacer = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
    layout.addItem(spacer)
    layout.addWidget(author_label, alignment=Qt.AlignRight | Qt.AlignBottom)

    splash.setLayout(layout)
    splash.show()
    print("Splash screen displayed")

    # Center the splash screen using QApplication.primaryScreen() (QApplication.desktop() is deprecated)
    screen = QApplication.primaryScreen()
    if screen:
        splash.move(screen.geometry().center() - splash.rect().center())


    # Booting progress bar loop
    for i in range(0, 101, 10):
        time.sleep(0.6)
        progress.setValue(i)
        QApplication.processEvents()

    splash.close()
    print("Login window displayed")
    import traceback
    try:
        pencere = ChatbotGUI()
    except Exception as e:
        print("ChatbotGUI başlatılırken hata oluştu:")
        traceback.print_exc()
        sys.exit(1)

    # Ana pencereyi yalnızca geliştirici modu kapalıysa göster
    if not getattr(pencere, "dev_mode_enabled", False):
        pencere.show()
        print("Main window shown.")
    else:
        print("Developer mode active – main window hidden.")
    print("Main UI initialized")
    sys.exit(app.exec_())
