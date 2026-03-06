import sys
import time
import logging
import traceback
from PyQt5.QtWidgets import QApplication, QSplashScreen, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

import database
import backend
from user_gui import ChatbotGUI
from dev_gui import DevGUI

# Basic logging setup
logger = logging.getLogger('thunderbird')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler('thunderbird_debug.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

def _global_exception_hook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error('Uncaught exception', exc_info=(exc_type, exc_value, exc_traceback))
    traceback.print_exception(exc_type, exc_value, exc_traceback)

sys.excepthook = _global_exception_hook

VERSION = "foxAI Volume Alpha"

if __name__ == "__main__":
    print("App is starting...")
    app = QApplication(sys.argv)
    
    # Initialize DB before UI
    database.initialize_memory_db()
    database.initialize_user_profile()

    # Initialize splash screen
    splash = QSplashScreen()
    splash.setFixedSize(300, 180)
    splash.setWindowFlags(Qt.FramelessWindowHint)    
    splash.setStyleSheet("background-color: black; color: white;")
    splash.setFont(QFont("Arial", 16, QFont.Bold))

    layout = QVBoxLayout(splash)
    version_label = QLabel(VERSION)
    version_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
    version_label.setStyleSheet("color: white; font-size: 14pt;")
    layout.addWidget(version_label)

    status_label = QLabel("Booting...")
    status_label.setAlignment(Qt.AlignCenter)
    status_label.setStyleSheet("color: white; font-size: 14pt; font-weight: bold;")
    layout.addWidget(status_label)

    status_label2 = QLabel("by fox & callisto")
    status_label2.setAlignment(Qt.AlignCenter)
    status_label2.setStyleSheet("color: gray; font-size: 11pt; font-weight: italic;")
    layout.addWidget(status_label2)

    progress = QProgressBar()
    progress.setMaximum(100)
    progress.setValue(0)
    progress.setStyleSheet("QProgressBar {background-color: #444; color: white;}")
    layout.addWidget(progress)

    splash.setLayout(layout)
    splash.show()

    screen = QApplication.primaryScreen()
    if screen:
        splash.move(screen.geometry().center() - splash.rect().center())

    # Booting progress bar loop
    for i in range(0, 101, 15):
        time.sleep(0.1)  # Faster boot
        progress.setValue(i)
        QApplication.processEvents()

    splash.close()

    try:
        # Launch User GUI (blocks on login)
        print("Launching User GUI...")
        user_window = ChatbotGUI()
        
        # Launch Dev GUI
        print("Launching Dev GUI...")
        dev_window = DevGUI()
        
        # Position windows if possible
        if screen:
            geom = screen.availableGeometry()
            w, h = geom.width(), geom.height()
            # User GUI on Left
            user_window.move(int(w/4 - 400), int(h/2 - 250))
            # Dev GUI on Right
            dev_window.move(int(3*w/4 - 400), int(h/2 - 350))

        user_window.show()
        dev_window.show()

    except Exception as e:
        print("Error starting application:")
        traceback.print_exc()
        sys.exit(1)
        
    sys.exit(app.exec_())
