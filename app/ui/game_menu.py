from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QScrollArea, QFrame
)
from PySide6.QtGui import QPixmap, QIcon, QCursor
from PySide6.QtCore import Qt, QSize
import os
import json
import subprocess


class GameMenu(QWidget):
    def __init__(self, game_data: dict, game_folder: str):
        super().__init__()
        self.setWindowTitle(f"{game_data.get("name")}")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.showMaximized()
        self.game_data = game_data
        self.game_folder = game_folder
        self.settings_path = os.path.join(game_folder, "settings.json")

        self.setStyleSheet("background-color: #1b1d1f; color: white; font-size: 16px;")

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.setup_top_bar()
        self.setup_hero_header()
        self.setup_settings_area()

    def setup_top_bar(self):
        self.top_bar = QFrame()
        self.top_bar.setFixedHeight(50)
        self.top_bar.setStyleSheet("background-color: rgba(0, 0, 0, 0.6);")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(10, 10, 10, 10)

        self.back_button = QPushButton()
        self.back_button.setIcon(QIcon("assets/back_arrow.png"))
        self.back_button.setIconSize(QSize(32, 32))
        self.back_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.back_button.setStyleSheet("background: transparent; border: none;")
        self.back_button.clicked.connect(self.go_back)

        top_layout.addWidget(self.back_button, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        self.layout.addWidget(self.top_bar)

    def setup_hero_header(self):
        self.hero = QFrame()
        self.hero.setMinimumHeight(400)
        self.hero.setStyleSheet(f"""
            QFrame {{
                background-image: url("{self.game_data.get("assets").get('hero')}");
                background-position: center;
                background-repeat: no-repeat;
            }}
        """)
        self.hero_layout = QVBoxLayout(self.hero)
        self.hero_layout.setContentsMargins(20, 20, 20, 20)
        self.hero_layout.setSpacing(20)

        # Центрированный по вертикали логотип
        logo_path = self.game_data.get("assets").get("logo")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaledToHeight(200, Qt.SmoothTransformation)
            self.logo = QLabel()
            self.logo.setPixmap(pixmap)
            self.logo.setAlignment(Qt.AlignCenter)
            self.hero_layout.addStretch()
            self.hero_layout.addWidget(self.logo, alignment=Qt.AlignHCenter)
            self.hero_layout.addStretch()

        # Нижняя панель с кнопками
        bottom_layout = QHBoxLayout()

        self.play_button = QPushButton("Играть")
        self.play_button.setFixedSize(160, 50)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                border: none;
                color: black;
                font-weight: bold;
                font-size: 18px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        self.play_button.clicked.connect(self.launch_game)

        # Центрирование play-кнопки по всей ширине
        play_layout = QHBoxLayout()
        play_layout.addStretch()
        play_layout.addWidget(self.play_button)
        play_layout.addStretch()

        # Кнопка настроек внизу справа
        self.settings_button = QPushButton()
        icon_path = "assets/gear.png"
        self.settings_button.setIcon(QIcon(icon_path))
        self.settings_button.setIconSize(QSize(32, 32))
        self.settings_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.settings_button.setStyleSheet("background: #7A7A7A;")
        self.settings_button.clicked.connect(self.open_edit_dialog)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.settings_button, alignment=Qt.AlignRight)

        self.hero_layout.addLayout(play_layout)
        self.hero_layout.addLayout(bottom_layout)

        self.layout.addWidget(self.hero)

    def go_back(self):
        from ui.main_menu import MainMenu
        self.main_menu = MainMenu()
        self.main_menu.show()
        self.close()

    def setup_settings_area(self):
        self.layout.addSpacing(10)

        title = QLabel("НАСТРОЙКИ")
        title.setAlignment(Qt.AlignHCenter)
        title.setStyleSheet("font-size: 20px; margin-top: 10px;")
        self.layout.addWidget(title)

        self.settings_container = QVBoxLayout()
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        settings_widget = QWidget()
        settings_widget.setLayout(self.settings_container)
        self.settings_scroll.setWidget(settings_widget)
        self.layout.addWidget(self.settings_scroll)

        self.add_setting_button = QPushButton("+ Добавить настройку")
        self.add_setting_button.clicked.connect(self.add_setting_row)
        self.layout.addWidget(self.add_setting_button, alignment=Qt.AlignHCenter)

        self.load_settings()

    def add_setting_row(self, action_name="", key_binding=""):
        row = QHBoxLayout()
        action_field = QLabel(action_name or "Действие")
        action_field.setStyleSheet("background-color: #2c3e50; padding: 6px; min-width: 120px;")
        key_field = QLabel(key_binding or "Клавиша")
        key_field.setStyleSheet("background-color: #34495e; padding: 6px; min-width: 80px;")

        row.addWidget(action_field)
        row.addWidget(key_field)
        self.settings_container.addLayout(row)

    def load_settings(self):
        if os.path.exists(self.settings_path):
            with open(self.settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for action, key in data.items():
                    self.add_setting_row(action, key)

    def save_settings(self):
        data = {}
        for i in range(self.settings_container.count()):
            row = self.settings_container.itemAt(i).layout()
            action = row.itemAt(0).widget().text()
            key = row.itemAt(1).widget().text()
            data[action] = key
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def launch_game(self):
        exe_path = self.game_data.get("exe", "")
        if os.path.exists(exe_path):
            subprocess.Popen([exe_path])
        else:
            print(f"Не найден exe: {exe_path}")

    def open_edit_dialog(self):
        print("Открытие окна редактирования игры... (заглушка)")
