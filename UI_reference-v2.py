from PyQt5.QtWidgets import QApplication, QSplashScreen, QLabel, QProgressBar, QVBoxLayout, QSpacerItem, QSizePolicy, QFont
from PyQt5.QtCore import Qt
import sys
import time

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

import logging


def init_ui(self):
    # Apply Apple-style Qt stylesheet
    self.setStyleSheet("""
        /* Main window styling */
        QWidget {
            background-color: #f5f5f7;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        /* Button styling - Apple's primary button */
        QPushButton, QToolButton {
            background-color: #0071e3;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 500;
            min-width: 80px;
        }
        
        /* Hover effect */
        QPushButton:hover, QToolButton:hover {
            background-color: #005bb7;
        }
        
        /* Disabled button */
        QPushButton:disabled, QToolButton:disabled {
            background-color: #cccccc;
        }
        
        /* Sidebar styling */
        QWidget#sidebar {
            background-color: #ffffff;
            border-right: 1px solid #e0e0e0;
            box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
        }
        
        /* Chat area styling */
        QTextEdit {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 12px;
            font-size: 14px;
        }
        
        /* Input field styling */
        QLineEdit {
            background-color: #ffffff;
            border: 1px solid #d0d0d0;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 14px;
        }
        
        /* Dialog styling */
        QDialog {
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Menu styling */
        QMenuBar {
            background-color: #ffffff;
            border-bottom: 1px solid #e0e0e0;
        }
        
        QMenuBar::item {
            padding: 8px 16px;
            color: #222222;
        }
        
        QMenuBar::item:selected {
            background-color: #f0f0f0;
        }
        
        /* Tab widget styling */
        QTabWidget::pane {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 6px;
        }
        
        QTabBar {
            border-bottom: 1px solid #e0e0e0;
        }
        
        QTabBar::tab {
            background-color: #f5f5f7;
            padding: 8px 16px;
            margin-right: -1px;
        }
        
        QTabBar::tab:selected {
            background-color: #ffffff;
            border-top: 2px solid #0071e3;
        }
        
        /* Progress bar */
        QProgressBar {
            background-color: #e0e0e0;
            border-radius: 4px;
        }
        
        QProgressBar::chunk {
            background-color: #0071e3;
            border-radius: 2px;
        }
        
        /* Scrollbar */
        QScrollBar:vertical {
            background-color: #f0f0f0;
            width: 12px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #c0c0c0;
            border-radius: 6px;
        }
        
        QScrollBar::add-line, QScrollBar::sub-line {
            background-color: transparent;
        }
        
        /* Tree view */
        QTreeView, QListView {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
        }
        
        /* Context menu */
        QMenu {
            background-color: #ffffff;
            border: 1px solid #d0d0d0;
            font-size: 14px;
        }
        
        QMenu::item:selected {
            background-color: #f0f0f0;
        }
        
        /* Tooltip */
        QToolTip {
            background-color: #333333;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 4px 8px;
            font-size: 13px;
        }
    """)
    
    # Set object names for specific styling (e.g., sidebar)
    self.sidebar.setObjectName("sidebar")
    
    # Continue with your existing init_ui logic

# Main application entry point
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
    version_label = QLabel("VERSION")  # Replace "VERSION" with actual version
    version_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
    version_label.setStyleSheet("color: white; font-size: 18pt;")
    layout.addWidget(version_label)

    status_label = QLabel("Booting...")
    status_label.setAlignment(Qt.AlignCenter)
    status_label.setStyleSheet("color: white; font-size: 14pt; font-weight: bold;")
    layout.addWidget(status_label)

    progress = QProgressBar()
    progress.setMaximum(100)
    progress.setValue(0)
    progress.setStyleSheet("QProgressBar {background-color: #444; color: white;}")
    layout.addWidget(progress)

    # Author label at bottom right with margin
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

    # Center the splash screen using QApplication.primaryScreen()
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
    
    try:
        # Replace ChatbotGUI with your actual main window class
        pencere = QWidget()  # Example placeholder
    except Exception as e:
        print("Error initializing main window:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Show main window if not in dev mode
    if not getattr(pencere, "dev_mode_enabled", False):
        pencere.show()
        print("Main window shown.")
    else:
        print("Developer mode active – main window hidden.")
    
    print("Main UI initialized")
    sys.exit(app.exec_())
