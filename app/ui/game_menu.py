from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QScrollArea, QFrame, QStackedLayout
)
from PySide6.QtGui import QPixmap, QIcon, QCursor, QColor, QEnterEvent, QPainter
from PySide6.QtCore import Qt, QSize, Property, QPropertyAnimation, QEasingCurve
import os
import json
import subprocess

class AnimatedButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bg_color = QColor("#636363")
        self._target_color = QColor("#636363")
        self._animation = QPropertyAnimation(self, b"bgColor")
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)

        self.setFixedSize(40, 40)
        self.setIconSize(QSize(32, 32))
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setStyleSheet(self._get_stylesheet(self._bg_color))

    def _get_stylesheet(self, color):
        return f"""
            QPushButton {{
                background-color: {color.name()};
                border: none;
                border-radius: 20px;
            }}
        """

    def enterEvent(self, event: QEnterEvent):
        self._animate_to(QColor("#575656"))
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate_to(QColor("#636363"))
        super().leaveEvent(event)

    def _animate_to(self, color: QColor):
        self._target_color = color
        self._animation.stop()
        self._animation.setStartValue(self._bg_color)
        self._animation.setEndValue(color)
        self._animation.start()

    def get_bg_color(self):
        return self._bg_color

    def set_bg_color(self, color):
        self._bg_color = color
        self.setStyleSheet(self._get_stylesheet(color))

    bgColor = Property(QColor, get_bg_color, set_bg_color)

class HeroFrame(QFrame):
    def __init__(self, game_data, parent=None):
        super().__init__(parent)
        self.game_data = game_data
        self.setMinimumHeight(400)

    def paintEvent(self, event):
        painter = QPainter(self)
        # Отображаем фоновое изображение
        hero_image = QPixmap(self.game_data.get("assets").get('hero'))
        if not hero_image.isNull():
            painter.drawPixmap(self.rect(), hero_image)

        # Отображаем логотип
        logo_path = self.game_data.get("assets").get("logo")
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            logo_pixmap = logo_pixmap.scaledToHeight(200, Qt.SmoothTransformation)
            logo_rect = logo_pixmap.rect()
            logo_rect.moveCenter(self.rect().center())  # Центрируем логотип
            painter.drawPixmap(logo_rect, logo_pixmap)

        painter.end()


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
        self.hero = HeroFrame(self.game_data, self)

        self.hero_layout = QVBoxLayout(self.hero)
        self.hero_layout.setContentsMargins(20, 20, 20, 20)
        self.hero_layout.setSpacing(0)

        self.hero_layout.addStretch()
        play_layout = QHBoxLayout()
        play_layout.addStretch()
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
        play_layout.addWidget(self.play_button)
        play_layout.addStretch()
        self.hero_layout.addLayout(play_layout)

        self.layout.addWidget(self.hero)

        self.settings_button = AnimatedButton(self.hero)
        self.settings_button.setIcon(QIcon("assets/gear.png"))
        self.settings_button.clicked.connect(self.open_edit_dialog)

        self.hero.resizeEvent = self.position_gear_button

    def position_gear_button(self, event):
        # Отступ от правого и нижнего края
        margin = 20
        btn_size = self.settings_button.sizeHint()
        x = self.hero.width() - btn_size.width() - margin
        y = self.hero.height() - btn_size.height() - margin
        self.settings_button.move(x, y)

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
