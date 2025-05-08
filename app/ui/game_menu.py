import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QScrollArea, QFrame, QStackedLayout, QMessageBox, QLineEdit
)
from PySide6.QtGui import QPixmap, QIcon, QCursor, QColor, QEnterEvent, QPainter, QPalette, QLinearGradient, QBrush, \
    QKeySequence, QKeyEvent, QMouseEvent
from PySide6.QtCore import Qt, QSize, Property, QPropertyAnimation, QEasingCurve
import os
import json
import subprocess
from ui.add_game import AddGameDialog

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
        self.setMinimumHeight(200)

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

class KeyBindingLineEdit(QLineEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.pressed_keys = set()

    def keyPressEvent(self, event: QKeyEvent):
        modifiers = event.modifiers()

        key_sequence = []

        if modifiers & Qt.ShiftModifier:
            key_sequence.append("Shift")
        if modifiers & Qt.ControlModifier:
            key_sequence.append("Ctrl")
        if modifiers & Qt.AltModifier:
            key_sequence.append("Alt")

        key_code = event.key()
        key_text = event.text()

        # Обработка специальных клавиш
        if key_code == Qt.Key_Space:
            key_sequence.append("Space")  # Пробел
        elif key_code == Qt.Key_Tab:
            key_sequence.append("Tab")  # Табуляция
        elif key_code == Qt.Key_Escape:
            key_sequence.append("Esc")  # Escape
        elif key_code == Qt.Key_Enter or key_code == Qt.Key_Return:
            key_sequence.append("Enter")  # Enter или Return
        elif key_code == Qt.Key_Backspace:
            key_sequence.append("Backspace")  # Backspace
        elif key_code == Qt.Key_Delete:
            key_sequence.append("Delete")  # Delete
        elif key_code == Qt.Key_Shift:
            key_sequence.append("Shift")  # Shift
        elif key_code == Qt.Key_Control:
            key_sequence.append("Control")  # Control
        elif key_code == Qt.Key_Alt:
            key_sequence.append("Alt")  # Alt
        elif key_code == Qt.Key_CapsLock:
            key_sequence.append("CapsLock")  # CapsLock
        elif key_code == Qt.Key_NumLock:
            key_sequence.append("NumLock")  # NumLock
        elif key_code == Qt.Key_ScrollLock:
            key_sequence.append("ScrollLock")  # ScrollLock
        elif key_code == Qt.Key_F1:
            key_sequence.append("F1")  # F1
        elif key_code == Qt.Key_F2:
            key_sequence.append("F2")  # F2
        elif key_code == Qt.Key_F3:
            key_sequence.append("F3")  # F3
        elif key_code == Qt.Key_F4:
            key_sequence.append("F4")  # F4
        elif key_code == Qt.Key_F5:
            key_sequence.append("F5")  # F5
        elif key_code == Qt.Key_F6:
            key_sequence.append("F6")  # F6
        elif key_code == Qt.Key_F7:
            key_sequence.append("F7")  # F7
        elif key_code == Qt.Key_F8:
            key_sequence.append("F8")  # F8
        elif key_code == Qt.Key_F9:
            key_sequence.append("F9")  # F9
        elif key_code == Qt.Key_F10:
            key_sequence.append("F10")  # F10
        elif key_code == Qt.Key_F11:
            key_sequence.append("F11")  # F11
        elif key_code == Qt.Key_F12:
            key_sequence.append("F12")  # F12
        elif key_code == Qt.Key_Left:
            key_sequence.append("Left Arrow")  # Стрелка влево
        elif key_code == Qt.Key_Right:
            key_sequence.append("Right Arrow")  # Стрелка вправо
        elif key_code == Qt.Key_Up:
            key_sequence.append("Up Arrow")  # Стрелка вверх
        elif key_code == Qt.Key_Down:
            key_sequence.append("Down Arrow")  # Стрелка вниз
        elif key_text:
            if key_text not in key_sequence:
                key_sequence.append(key_text.upper())
        else:
            key_sequence = []
        if len(key_sequence) == 2 and key_sequence[0] == key_sequence[1]:
            key_sequence = [key_sequence[0]]

        if key_sequence:
            self.setText(" + ".join(key_sequence))
        else:
            self.clear()

    def mousePressEvent(self, event: QMouseEvent):
        # Проверка, какая кнопка была нажата
        modifiers = event.modifiers()
        button = event.button()

        mouse_sequence = []
        if modifiers & Qt.ShiftModifier:
            mouse_sequence.append("Shift")
        if modifiers & Qt.ControlModifier:
            mouse_sequence.append("Ctrl")
        if modifiers & Qt.AltModifier:
            mouse_sequence.append("Alt")

        if button == Qt.LeftButton:
            mouse_sequence.append("Left Click")
        elif button == Qt.RightButton:
            mouse_sequence.append("Right Click")
        elif button == Qt.MiddleButton:
            mouse_sequence.append("Middle Click")

        if mouse_sequence:
            self.setText(" + ".join(mouse_sequence))



class GameMenu(QWidget):
    def __init__(self, game_data: dict, game_folder: str):
        super().__init__()
        self.setWindowTitle(f"{game_data.get("name")}")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.showMaximized()
        self.game_data = game_data
        self.game_folder = game_folder
        self.settings_path = os.path.join(game_folder, "settings.json")

        self.set_dark_gradient_background()

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.setup_top_bar()
        self.setup_hero_header()
        self.setup_settings_area()

    def setup_top_bar(self):
        self.top_bar = QFrame()
        self.top_bar.setFixedHeight(50)
        self.set_dark_gradient_background()
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

    def set_dark_gradient_background(self):
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#0f0f0f"))
        gradient.setColorAt(1.0, QColor("#1a1a2e"))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setAutoFillBackground(True)
        self.setPalette(palette)

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
        self.set_dark_gradient_background()

        title = QLabel("НАСТРОЙКИ")
        title.setAlignment(Qt.AlignHCenter)
        title.setStyleSheet("font-size: 20px; margin-top: 10px;")
        self.layout.addWidget(title)

        # Контейнер для настроек
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)

        # Вложенный виджет, внутри которого вертикальный лейаут
        self.settings_widget = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_widget)
        self.settings_layout.setContentsMargins(10, 10, 10, 10)
        self.settings_layout.setSpacing(10)
        self.settings_layout.addStretch()

        self.settings_scroll.setWidget(self.settings_widget)
        self.layout.addWidget(self.settings_scroll)

        # Кнопка добавления
        self.add_setting_button = QPushButton("+ Добавить настройку")
        self.add_setting_button.clicked.connect(lambda: self.add_setting_row("", ""))
        self.layout.addWidget(self.add_setting_button, alignment=Qt.AlignHCenter)

        # self.load_settings()

    def add_setting_row(self, action_name="", key_binding=""):
        row_widget = QWidget()
        row_widget.setFixedHeight(40)

        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)

        # Поле действия — редактируемое
        action_field = QLineEdit()
        action_field.setPlaceholderText("Действие")
        if action_name != "":
            action_field.setText(action_name)
        action_field.setFixedHeight(40)
        action_field.setStyleSheet("""
            background-color: #2c3e50;
            color: white;
            padding-left: 10px;
            min-width: 200px;
            border-top-left-radius: 5px;
            border-bottom-left-radius: 5px;
            border: none;
        """)

        # Поле клавиши — специальное
        key_field = KeyBindingLineEdit()
        key_field.setPlaceholderText("Клавиша")
        if key_binding != "":
            key_field.setText(key_binding)
        key_field.setFixedHeight(40)
        key_field.setStyleSheet("""
            background-color: #34495e;
            color: white;
            padding-left: 10px;
            min-width: 100px;
            border-top-right-radius: 5px;
            border-bottom-right-radius: 5px;
            border: none;
        """)

        row_layout.addWidget(action_field, 2)
        row_layout.addWidget(key_field, 1)

        index = self.settings_layout.count() - 1
        self.settings_layout.insertWidget(index, row_widget)

    # def load_settings(self):
    #     if os.path.exists(self.settings_path):
    #         with open(self.settings_path, "r", encoding="utf-8") as f:
    #             data = json.load(f)
    #             for action, key in data.items():
    #                 self.add_setting_row(action, key)

    # def save_settings(self):
    #     data = {}
    #     for i in range(self.settings_container.count()):
    #         row = self.settings_container.itemAt(i).layout()
    #         action = row.itemAt(0).widget().text()
    #         key = row.itemAt(1).widget().text()
    #         data[action] = key
    #     with open(self.settings_path, "w", encoding="utf-8") as f:
    #         json.dump(data, f, ensure_ascii=False, indent=4)

    def launch_game(self):
        exe_path = self.game_data.get("exe", "")
        if os.path.exists(exe_path):
            subprocess.Popen([exe_path])
        else:
            print(f"Не найден exe: {exe_path}")

    def open_edit_dialog(self):
        if self.game_data.get("appid") != -1:
            QMessageBox.warning(self, "Ошибка", "Нельзя изменить параметры игр из Steam")
            return

        dialog = AddGameDialog("Изменить параметры игры", self.game_data)
        if dialog.exec():
            game_data = dialog.get_data()
            if not game_data:
                return

            game_name = game_data["name"]
            game_path = Path(self.game_folder)

            if game_name != self.game_data.get("name"):
                QMessageBox.warning(self, "Ошибка", "Указан другой .exe файл.")
                return

            for key, path in game_data["assets"].items():
                if path == None:
                    game_data["assets"][key] = self.game_data.get("assets").get(key)
                    continue

                os.unlink(self.game_data.get("assets").get(key))
                type = path.split(".")[-1]
                dst = Path(game_name) / "assets" / f"{key}.{type}"
                shutil.copy(path, game_path / "assets" / f"{key}.{type}")
                game_data["assets"][key] = str(dst).replace("\\", "/")
                self.game_data["assets"][key] = game_path / "assets" / f"{key}.{type}"

            with open(game_path / "appmanifest.json", "w", encoding="utf-8") as f:
                json.dump(game_data, f, indent=4, ensure_ascii=False)

            QMessageBox.information(self, "Успех", f"Параметры '{game_name}' изменены.")
            self.hero.update()

