import sys
import json
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QCheckBox, QMessageBox
)
from PySide6.QtGui import QFont, QIcon
from PySide6.QtCore import Qt

SETTINGS_PATH = os.path.join("config", "settings.json")

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shadow — Вход")
        self.setFixedSize(600, 360)
        self.setStyleSheet(self.load_styles())
        self.setWindowIcon(QIcon("assets/icon.png"))

        # ==== ЦЕНТРАЛЬНЫЙ БЛОК ====
        title = QLabel("Shadow")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Логин")
        self.username_input.setFixedWidth(300)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Пароль")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setFixedWidth(300)

        self.remember_checkbox = QCheckBox("Запомнить меня")

        self.login_button = QPushButton("ВОЙТИ")
        self.login_button.setFixedWidth(300)
        self.login_button.clicked.connect(self.handle_login)

        layout = QVBoxLayout()
        layout.addStretch()
        layout.addWidget(title, alignment=Qt.AlignCenter)
        layout.addSpacing(10)
        layout.addWidget(self.username_input, alignment=Qt.AlignCenter)
        layout.addWidget(self.password_input, alignment=Qt.AlignCenter)
        layout.addWidget(self.remember_checkbox, alignment=Qt.AlignCenter)
        layout.addWidget(self.login_button, alignment=Qt.AlignCenter)
        layout.addStretch()

        self.setLayout(layout)

        self.check_saved_user()

    def load_styles(self):
        return """
        QWidget {
            background-color: #1B1B1B;
            color: #FFFFFF;
            font-family: Arial;
            font-size: 14px;
        }
        QLabel#title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        QLineEdit {
            background-color: #2A2A2A;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 6px;
            color: white;
        }
        QLineEdit:focus {
            border: 1px solid #4BA4FF;
        }
        QPushButton {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #4BA4FF, stop:1 #2B85F8);
            color: white;
            border-radius: 4px;
            padding: 8px;
        }
        QPushButton:hover {
            background-color: #3B9CFF;
        }
        QCheckBox {
            spacing: 6px;
        }
        """

    def check_saved_user(self):
        if os.path.exists(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    username = data.get("username")
                    if username:
                        self.open_main_window(username)
            except Exception as e:
                print("Ошибка чтения settings.json:", e)

    def handle_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if not username or not password:
            QMessageBox.warning(self, "Ошибка", "Введите логин и пароль")
            return

        if self.remember_checkbox.isChecked():
            os.makedirs("config", exist_ok=True)
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump({"username": username}, f, ensure_ascii=False, indent=4)

        self.open_main_window(username)

    def open_main_window(self, username):
        QMessageBox.information(self, "Вход", f"Добро пожаловать, {username}!")
        self.close()
