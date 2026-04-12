# Thunderbird AI - UI Only Mockup
# Stripped of all backend, database, and API logic.

import sys
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QHBoxLayout, QLabel, QSplitter,
    QDialog, QFormLayout, QComboBox, QMessageBox, QRadioButton,
    QButtonGroup, QStackedLayout, QDialogButtonBox, QMenu, QInputDialog,
    QSplashScreen, QProgressBar, QSpacerItem, QSizePolicy, QColorDialog
)
from PyQt5.QtCore import Qt, QRect
from PyQt5 import QtGui

VERSION = "Thunderbird AI Volume Alpha"

class DevGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Developer Panel")
        self.setGeometry(200, 200, 700, 500)
        self.current_theme = "system"
        self.memory_mode = "aktif"
        self.init_ui()

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
        layout = QHBoxLayout(self)
        self.sidebar = QVBoxLayout()
        
        self.sidebar_label = QLabel("BIRD AI (DEV)")
        self.sidebar_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.sidebar.addWidget(self.sidebar_label)

        zim_btn = QPushButton("ZIM Yükle (Mock)")
        self.style_button(zim_btn, color="#2980b9")
        zim_btn.clicked.connect(lambda: QMessageBox.information(self, "ZIM", "ZIM Yükleme UI Tetiklendi"))
        self.sidebar.addWidget(zim_btn) 

        doc_btn = QPushButton("PDF/DOCX Yükle (Mock)")
        self.style_button(doc_btn, color="#27ae60")
        doc_btn.clicked.connect(lambda: QMessageBox.information(self, "Doc", "Belge Yükleme UI Tetiklendi"))
        self.sidebar.addWidget(doc_btn)

        users_btn = QPushButton("Kullanıcıları Gör")
        self.style_button(users_btn, color="#8e44ad")
        users_btn.clicked.connect(lambda: QMessageBox.information(self, "Users", "Kullanıcı Listesi UI Tetiklendi"))
        self.sidebar.addWidget(users_btn)

        self.exit_button = QPushButton("Kapat")
        self.style_button(self.exit_button, color="#c0392b")
        self.exit_button.clicked.connect(self.close)
        self.sidebar.addWidget(self.exit_button)
        self.sidebar.addStretch()

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(self.sidebar)
        sidebar_widget.setFixedWidth(180)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.append("<i>Developer console loaded...</i>")
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(sidebar_widget)
        splitter.addWidget(self.chat_display)

        layout.addWidget(splitter)
        self.setLayout(layout)

class ChatbotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.layout_yuklendi = False
        self.setWindowTitle(VERSION)
        self.setGeometry(100, 100, 800, 500)
        
        self.conversations = []
        self.max_conversations = 5
        self.current_theme = "system"
        self.emotion_mode = "neutral"
        self.logged_in_email = None
        self.dev_mode_enabled = False

        def dummy_shimmer(label):
            label.setStyleSheet("color: red; font-weight: bold;")
        self.shimmer_label = dummy_shimmer

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
        
        self.chat_display.append(f"<div style='border:1px solid #ccc; padding:5px; border-radius:8px;'><b>SEN:</b> {soru}</div>")
        self.input_field.clear()
        
        cevap = f"(Mock Yanıt): '{soru}' sorunuza dair veritabanı veya API bağlantısı UI modunda kapalıdır."
        self.chat_display.append(f"<div style='border:1px solid #ccc; padding:5px; border-radius:8px; margin-top:5px;'><b>BIRD AI:</b> {cevap}</div>")
        self.suggestion_label.setText("<i>Öneri: UI testlerine devam edebilirsiniz.</i>")

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
        btn.clicked.connect(load_conv)
        
        def context_menu(point):
            menu = QMenu()
            act_rename = menu.addAction("Yeniden Adlandır")
            act_delete = menu.addAction("Sil")
            chosen = menu.exec_(btn.mapToGlobal(point))
            if chosen == act_rename:
                new_title, ok = QInputDialog.getText(self, "Yeniden Adlandır", "Yeni sohbet başlığı:")
                if ok and new_title: btn.setText(new_title)
            elif chosen == act_delete:
                btn.setParent(None)

        btn.setContextMenuPolicy(Qt.CustomContextMenu)
        btn.customContextMenuRequested.connect(context_menu)

        self.conversation_buttons_layout.addWidget(btn)
        load_conv()

    def open_settings_panel(self):
        splash = QDialog(self, Qt.FramelessWindowHint | Qt.Dialog)
        splash.setAttribute(Qt.WA_TranslucentBackground)
        splash.setModal(True)

        panel = QWidget()
        panel.setStyleSheet("QWidget { background-color: #f5f5f5; border-radius: 12px; }")
        panel.resize(520, 620)
        stack = QStackedLayout()

        # MAIN PAGE
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

        # THEME PAGE
        theme_page = QWidget()
        t_layout = QVBoxLayout(theme_page)
        theme_group = QButtonGroup(theme_page)
        rb_light = QRadioButton("Açık Tema")
        rb_dark = QRadioButton("Koyu Tema")
        rb_system = QRadioButton("Sistem Teması")
        
        for rb in (rb_light, rb_dark, rb_system):
            self.style_button(rb, color="#273bae", text_color="#000")
            theme_group.addButton(rb)
            t_layout.addWidget(rb)

        if self.current_theme == "light": rb_light.setChecked(True)
        elif self.current_theme == "dark": rb_dark.setChecked(True)
        else: rb_system.setChecked(True)

        save_theme_btn = QPushButton("Kaydet")
        self.style_button(save_theme_btn, color="#273bae")
        t_layout.addWidget(save_theme_btn)
        t_layout.addStretch()

        def apply_mock_theme():
            if rb_light.isChecked(): self.apply_theme("light")
            elif rb_dark.isChecked(): self.apply_theme("dark")
            else: self.apply_theme("system")
            stack.setCurrentIndex(0)
        save_theme_btn.clicked.connect(apply_mock_theme)

        theme_btn.clicked.connect(lambda: stack.setCurrentIndex(1))
        prof_btn.clicked.connect(lambda: [splash.close(), self.open_profile_settings()])

        stack.addWidget(main_page)
        stack.addWidget(theme_page)

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
        dialog.setWindowTitle("Profil Ayarları (Mock)")
        layout = QFormLayout(dialog)
        prof = self.get_mock_profile()

        for key, val in prof.items():
            if key != "theme":
                le = QLineEdit(str(val))
                le.setReadOnly(True)
                layout.addRow(f"{key.capitalize()}:", le)

        close_button = QPushButton("Kapat")
        self.style_button(close_button)
        close_button.clicked.connect(dialog.accept)
        layout.addRow(close_button)
        dialog.setLayout(layout)
        dialog.exec_()

    def apply_theme(self, theme):
        self.current_theme = theme
        if theme == "dark":
            self.setStyleSheet("background-color: #2C3E50; color: #ECF0F1;")
        elif theme == "light":
            self.setStyleSheet("background-color: #ECF0F1; color: #2C3E50;")
        else:
            self.setStyleSheet("")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    splash = QSplashScreen()
    splash.setFixedSize(300, 180)
    splash.setWindowFlags(Qt.FramelessWindowHint)	
    splash.setStyleSheet("background-color: black; color: white;")
    
    layout = QVBoxLayout(splash)
    version_label = QLabel(VERSION)
    version_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
    version_label.setStyleSheet("color: white; font-size: 14pt;")
    layout.addWidget(version_label)

    status_label = QLabel("Booting (UI Mock)...")
    status_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(status_label)

    progress = QProgressBar()
    progress.setMaximum(100)
    progress.setStyleSheet("QProgressBar {background-color: #444; color: white; text-align: center;}")
    layout.addWidget(progress)

    author_label = QLabel("by FOX (UI Render)")
    layout.addWidget(author_label, alignment=Qt.AlignRight | Qt.AlignBottom)

    splash.setLayout(layout)
    splash.show()

    screen = QApplication.primaryScreen()
    if screen:
        splash.move(screen.geometry().center() - splash.rect().center())

    for i in range(0, 101, 20):
        time.sleep(0.2)
        progress.setValue(i)
        QApplication.processEvents()

    splash.close()
    
    pencere = ChatbotGUI()
    if not pencere.dev_mode_enabled:
        pencere.show()
    
    sys.exit(app.exec_())