import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QScrollArea, QFrame, QMessageBox, QLineEdit
)
from PySide6.QtGui import QPixmap, QIcon, QCursor, QColor, QEnterEvent, QPainter, QPalette, QLinearGradient, QBrush, \
    QKeySequence, QKeyEvent, QMouseEvent, QFocusEvent, QCloseEvent
from PySide6.QtCore import Qt, QSize, Property, QPropertyAnimation, QEasingCurve
import os
import json


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
    def __init__(self, settings_pth, index):
        super().__init__()
        self.value = self.text().strip()
        self.settings_pth = settings_pth
        self.index = index
        self.setReadOnly(True)
        self.recording_started = False
        self.awaiting_first_release = False
        self.pressed_mouse_buttons = []
        self.current_keys = set()
        self.default_style = self.styleSheet()

    def startRecording(self):
        self.recording_started = True
        self.awaiting_first_release = True
        self.value = self.text().strip()
        self.clear()
        self.setText("     ")
        self.default_style = self.styleSheet()
        self.setStyleSheet("background-color: #4a4949; border: 2px solid gray;")

    def stopRecording(self):
        self.recording_started = False
        self.awaiting_first_release = False
        self.pressed_mouse_buttons.clear()
        if self.default_style:
            self.setStyleSheet(self.default_style)
        else:
            self.setStyleSheet("")

        with open(self.settings_pth, "r", encoding="utf-8") as f:
            data = json.load(f)
            for i, action in enumerate(data.keys()):
                if i == self.index:
                    data[action] = self.text().strip()
                    break
            else:
                data[str(self.index)] = self.text().strip()

        with open(self.settings_pth, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def keyPressEvent(self, event: QKeyEvent):
        if not self.recording_started:
            return

        key = event.key()
        self.current_keys.add(key)
        self._update_key_text()

    def keyReleaseEvent(self, event: QKeyEvent):
        key = event.key()
        if key in self.current_keys:
            self.current_keys.remove(key)

        if self.recording_started and not self.current_keys:
            self.stopRecording()

    def _update_key_text(self):
        if not self.current_keys:
            return

        keys_text = []
        modifiers_map = {
            Qt.Key_Control: "Ctrl",
            Qt.Key_Shift: "Shift",
            Qt.Key_Alt: "Alt",
        }

        special_keys = {
            Qt.Key_Space: "Space",
            Qt.Key_Tab: "Tab",
            Qt.Key_Escape: "Esc",
            Qt.Key_Enter: "Enter",
            Qt.Key_Return: "Enter",
            Qt.Key_Backspace: "Backspace",
            Qt.Key_Delete: "Delete",
            Qt.Key_Left: "Left Arrow",
            Qt.Key_Right: "Right Arrow",
            Qt.Key_Up: "Up Arrow",
            Qt.Key_Down: "Down Arrow",
        }

        for key in sorted(self.current_keys):
            if key in modifiers_map:
                keys_text.append(modifiers_map[key])
            elif key in special_keys:
                keys_text.append(special_keys[key])
            else:
                text = QKeySequence(key).toString()
                if text:
                    keys_text.append(text.upper())

        self.setText(" + ".join(keys_text))

    def mousePressEvent(self, event: QMouseEvent):
        if not self.recording_started:
            self.startRecording()
            return

        if self.awaiting_first_release:
            return

        self.pressed_mouse_buttons.append(event.button())
        self._update_mouse_text(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.awaiting_first_release:
            self.awaiting_first_release = False
            return

        if event.button() in self.pressed_mouse_buttons:
            self.pressed_mouse_buttons.remove(event.button())

        if self.recording_started and not self.pressed_mouse_buttons:
            self.stopRecording()

    def _update_mouse_text(self, event: QMouseEvent):
        modifiers = []
        if event.modifiers() & Qt.ControlModifier:
            modifiers.append("Ctrl")
        if event.modifiers() & Qt.ShiftModifier:
            modifiers.append("Shift")
        if event.modifiers() & Qt.AltModifier:
            modifiers.append("Alt")

        for btn in self.pressed_mouse_buttons:
            if Qt.LeftButton == btn:
                modifiers.append("Left Click")
            if Qt.RightButton == btn:
                modifiers.append("Right Click")
            if Qt.MiddleButton == btn:
                modifiers.append("Middle Click")

        self.setText("+".join(modifiers))

    def focusOutEvent(self, event: QFocusEvent):
        if self.recording_started:
            self.stopRecording()
        super().focusOutEvent(event)

    def contextMenuEvent(self, event):
        pass



class GameMenu(QWidget):
    def __init__(self, game_data: dict, game_folder: str):
        super().__init__()
        self.setWindowTitle(f"{game_data.get("name")}")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.showMaximized()
        self.game_data = game_data
        self.game_folder = game_folder
        self.global_game_folder = Path(__file__).resolve().parent.parent / game_folder
        self.run_turn_model_pth = Path(__file__).resolve().parent.parent / "run_turn_model"
        self.settings_path = os.path.join(str(self.global_game_folder), "settings.json").replace("\\", "/")
        if os.path.exists(self.settings_path):
            with open(self.settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data == {}:
                    data = {"Идти вперед": "W", "Идти назад": "S", "Идти влево": "A",
                            "Идти вправо": "D", "Бег вперед": "Shift + W", "Прыжок": "Space", "Сесть": "Ctrl"}
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        self.set_dark_gradient_background()

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.setup_top_bar()
        self.setup_hero_header()
        self.setup_settings_area()
        self.load_settings()

    def closeEvent(self, event):
        # Сохраняем настройки перед закрытием
        self.save_settings()
        event.accept()

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
        from ui.main_menu_window import MainMenu
        self.closeEvent(QCloseEvent())
        self.main_menu = MainMenu()
        self.main_menu.show()
        self.close()

    def setup_settings_area(self):
        self.layout.addSpacing(10)
        self.set_dark_gradient_background()

        # --- Верхний layout с кнопками и заголовком ---
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(20, 10, 20, 0)

        # Кнопка "Собрать данные"
        self.collect_data_button = AnimatedButton()
        self.collect_data_button.setIcon(QIcon("assets/folder.png"))
        self.collect_data_button.setToolTip("Подготовить данные для настроек")
        self.collect_data_button.clicked.connect(self.collect_data)

        # Кнопка "Обучить модель"
        self.prepare_model_button = AnimatedButton()
        self.prepare_model_button.setIcon(QIcon("assets/brain.png"))
        self.prepare_model_button.setToolTip("Обучить модель на собранных данных")
        self.prepare_model_button.clicked.connect(self.prepare_model)

        # Заголовок
        title = QLabel("НАСТРОЙКИ")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: white;")

        # Кнопка "Добавить настройку"
        self.add_setting_button = AnimatedButton()
        self.add_setting_button.setIcon(QIcon("assets/add.png"))
        self.add_setting_button.setToolTip("Добавить настройку")
        self.add_setting_button.clicked.connect(lambda: self.add_setting_row("", ""))

        # Обёртка для title + кнопка "добавить"
        title_with_add_layout = QHBoxLayout()
        title_with_add_layout.setContentsMargins(0, 0, 0, 0)
        title_with_add_layout.setSpacing(8)
        title_with_add_layout.addWidget(title)
        title_with_add_layout.addWidget(self.add_setting_button)

        title_container = QWidget()
        title_container.setLayout(title_with_add_layout)

        # --- Финальный header ---
        header_layout.addWidget(self.collect_data_button, alignment=Qt.AlignLeft)
        header_layout.addStretch()
        header_layout.addWidget(title_container, alignment=Qt.AlignCenter)
        header_layout.addStretch()
        header_layout.addWidget(self.prepare_model_button, alignment=Qt.AlignRight)

        self.layout.addLayout(header_layout)

        # --- Контейнер для настроек ---
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)

        self.settings_widget = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_widget)
        self.settings_layout.setContentsMargins(10, 10, 10, 10)
        self.settings_layout.setSpacing(10)
        self.settings_layout.addStretch()

        self.settings_scroll.setWidget(self.settings_widget)
        self.layout.addWidget(self.settings_scroll)

    def collect_data(self):
        flag = False
        self.save_settings()
        if os.path.exists(self.settings_path):
            with open(self.settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for action, keys in data.items():
                    if action == "" or keys == "":
                        flag = True
                        break
        if flag:
            QMessageBox.warning(self, "Ошибка", "Для продолжения нужно задать имя класса и клавишу для каждой настройки")
            return

        from ui.collect_data_dialog import CollectDataDialog
        dialog = CollectDataDialog(self.settings_path)
        if dialog.exec():
            selected_classes = dialog.get_selected_classes()

            if not selected_classes:
                QMessageBox.warning(self, "Ошибка", "Нужно выбрать хотя бы один класс.")
                return

            from logic.data_collector import DataCollectionWindow
            self.data_collection_window = DataCollectionWindow(
                data_path=self.global_game_folder / "VidData",
                actions=selected_classes,
                no_sequences=50,
                sequence_length=30,
                start_folder=1
            )

            self.data_collection_window.show()


    def prepare_model(self):
        if not os.path.exists(self.global_game_folder / "VidData"):
            QMessageBox.information(self, "Подготовка модели", f"Сначала нужно собрать данные для каждого действия")
            return

        actions = []
        directories = []
        with open(self.settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for idx, (action, keys) in enumerate(data.items()):
                if idx > 6 and " + " not in action:
                    directories.append(action)

        name = "model_1"
        if os.path.exists(self.global_game_folder / "checkpoints"):
            number = len(os.listdir(self.global_game_folder / "checkpoints"))
            name = "_".join([name.split("_")[0], f"{number + 1}"])

        from logic.train_model import TrainingWindow
        self.window = TrainingWindow(
            train_name=name,
            actions=actions,
            directories=directories,
            game_path=self.global_game_folder,
            sequence_length=30,
            epochs=2000,
            num_classes=len(actions),
            batch_size=32
        )
        self.window.show()

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
        index = self.settings_layout.count() - 1
        key_field = KeyBindingLineEdit(self.settings_path, index)
        key_field.setPlaceholderText("Клавиша")
        if key_binding != "":
            key_field.setText(key_binding)
        key_field.setFixedHeight(40)
        key_field.setStyleSheet("""
            background-color: #242424;
            color: white;
            padding-left: 10px;
            min-width: 10px;
            border-top-right-radius: 5px;
            border-bottom-right-radius: 5px;
            border: none;
        """)

        row_layout.addWidget(action_field, 2)
        row_layout.addWidget(key_field, 1)

        self.settings_layout.insertWidget(index, row_widget)

    def save_settings(self):
        data = {}
        for i in range(self.settings_layout.count()):
            row_widget = self.settings_layout.itemAt(i).widget()
            if row_widget:
                action_field = row_widget.findChild(QLineEdit)
                key_field = row_widget.findChild(KeyBindingLineEdit)
                if action_field and key_field:
                    action = action_field.text().strip()
                    key = key_field.text().strip()
                    if key == "Клавиша":
                        key = ""
                    data[action] = key

        # Записываем данные в файл JSON
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load_settings(self):
        if os.path.exists(self.settings_path):
            with open(self.settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for action, keys in data.items():
                    self.add_setting_row(action, keys)

    def launch_game(self):
        if not os.path.exists(self.global_game_folder / "checkpoints"):
            QMessageBox.information(self, "Запуск игры", "Сперва нужно подготовить модель")
            return

        models = [file.name for file in (self.global_game_folder / "checkpoints").iterdir() if file.is_dir()]
        exe_files = [Path(exe).name for exe in self.game_data.get("exe")]

        from ui.launch_game_dialog import LaunchGameDialog
        dialog = LaunchGameDialog(models, exe_files)

        if dialog.exec():
            model, exe = dialog.get_selection()
            for ef in self.game_data.get("exe"):
                if Path(ef).name == exe:
                    exe = ef
                    break

            if os.path.exists(self.settings_path):
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    actions = {}
                    walk_actions = {}
                    for i, (action, key) in enumerate(data.items()):
                        if i <= 6:
                            walk_actions[action] = [key, False]
                        elif " + " not in action:
                            actions[action] = [key, False]
                    actions["Бездействие"] = ["", True]
                    walk_actions["Бездействие"] = ["", True]
            else:
                QMessageBox.information(self, "Запуск игры", "Отсутствуют настройки")
                return
            turn_actions = {"Поворот направо": ["", False], "Поворот налево": ["", False],
                            "Поворот вверх": ["", False], "Поворот вниз": ["", False], "Бездействие": ["", True]}

            from logic.game_control import GameLauncher
            self.gl = GameLauncher(parent_window=self,
                         path=Path(__file__).resolve().parent.parent / "mediamtx",
                         exe_file=exe,
                         actions=actions,
                         walk_actions=walk_actions,
                         turn_actions=turn_actions,
                         action_model_path=self.global_game_folder / "checkpoints" / model / "best_model.pth",
                         walk_model_path=self.run_turn_model_pth / "run_model7" / "best_model.pth",
                         turn_model_path=self.run_turn_model_pth / "turn_model2" / "best_model.pth")

    def open_edit_dialog(self):
        if self.game_data.get("appid") != -1:
            QMessageBox.warning(self, "Ошибка", "Нельзя изменить параметры игр из Steam")
            return

        from ui.add_game_dialog import AddGameDialog
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

