from pathlib import Path
import os
import webbrowser

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QHBoxLayout, QMessageBox
)
from PySide6.QtGui import QPixmap, QPainter, QLinearGradient, QColor, QFont, QImage, QCursor
from PySide6.QtCore import Qt, QSize


class AddGameDialog(QDialog):
    def __init__(self, type, game=None):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setFixedSize(400, 420)
        self.setStyleSheet(self.load_styles())
        self._type = type

        self.cover_path = None
        self.logo_path = None
        self.hero_path = None
        self.initial_dir = os.path.abspath(os.sep)

        layout = QVBoxLayout()

        self.exe_input = QLineEdit()
        if game:
            self.exe_input.setText(game.get("exe", "")[0])
        else:
            self.exe_input.setPlaceholderText("Путь до .exe файла")

        exe_btn = QPushButton("Выбрать файл .exe")
        exe_btn.clicked.connect(self.choose_exe)
        layout.addWidget(self.exe_input)
        layout.addWidget(exe_btn)

        cover_btn = QPushButton("Выбрать обложку")
        cover_btn.clicked.connect(lambda: self.choose_image("cover"))
        layout.addWidget(cover_btn)

        logo_btn = QPushButton("Выбрать логотип")
        logo_btn.clicked.connect(lambda: self.choose_image("logo"))
        layout.addWidget(logo_btn)

        hero_btn = QPushButton("Выбрать шапку")
        hero_btn.clicked.connect(lambda: self.choose_image("hero"))
        layout.addWidget(hero_btn)

        button_layout = QHBoxLayout()
        add_btn = QPushButton("Добавить" if self._type == "Добавить игру" else "Изменить")
        cancel_btn = QPushButton("Отмена")
        cancel_btn.setObjectName("cancelBtn")
        add_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(add_btn)
        button_layout.addWidget(cancel_btn)

        layout.addStretch()
        layout.addLayout(button_layout)

        link_label = QLabel('<a href="https://www.steamgriddb.com">Изображения можно найти здесь</a>')
        link_label.setObjectName("linkLabel")
        link_label.setAlignment(Qt.AlignCenter)
        link_label.setOpenExternalLinks(False)
        link_label.setCursor(QCursor(Qt.PointingHandCursor))
        link_label.linkActivated.connect(self.open_link)
        layout.addWidget(link_label)

        self.setLayout(layout)

    def load_styles(self):
        return """
        QDialog {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1,
                                              stop:0 #0f0f0f, stop:1 #1a1a2e);
            color: white;
            font-size: 14px;
        }

        QLabel {
            font-weight: bold;
            color: #cfcfcf;
        }

        QLineEdit, QComboBox {
            background-color: #2c2c3c;
            color: white;
            border: 1px solid #444;
            padding: 6px;
            border-radius: 6px;
        }

        QPushButton {
            background-color: #7e22ce;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #5e1c99;
        }

        QPushButton#cancelBtn {
            background-color: #444;
        }

        QPushButton#cancelBtn:hover {
            background-color: #666;
        }

        QLabel:hover {
            color: #a07ede;
        }
        QLabel#linkLabel {
            color: #cfcfcf;
            font-size: 13px;
        }

        QLabel#linkLabel:hover {
            color: #7e22ce;
            text-decoration: underline;
        }
        """

    def open_link(self, link):
        webbrowser.open(link)

    def choose_exe(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите .exe файл", self.initial_dir, "Executable (*.exe)")
        if file_path:
            self.exe_input.setText(file_path)

    def choose_image(self, image_type):
        file_path, _ = QFileDialog.getOpenFileName(self, f"Выберите изображение для {image_type}", self.initial_dir,
                                                   "Images (*.jpg *.png *jpeg)")
        if file_path:
            setattr(self, f"{image_type}_path", file_path)

    def get_data(self):
        exe_path = self.exe_input.text().strip()

        if not exe_path:
            QMessageBox.warning(self, "Ошибка", "Поле 'Путь до .exe' должно быть заполнено!")
            return None

        name = Path(exe_path).name.split(".")[0]
        return {
            "appid": -1,
            "name": name,
            "installdir": name,
            "exe": [exe_path],
            "assets": {
                "cover": self.cover_path or (self.create_placeholder(name, "cover") if self._type != "Изменить параметры игры" else None),
                "logo": self.logo_path or (self.create_placeholder(name, "logo") if self._type != "Изменить параметры игры" else None),
                "hero": self.hero_path or (self.create_placeholder(name, "hero") if self._type != "Изменить параметры игры" else None)
            }
        }

    def create_placeholder(self, text, image_type):
        size_map = {
            "cover": QSize(600, 900),
            "logo": QSize(600, 300),
            "hero": QSize(1920, 620)
        }
        type = {
            "cover": ".jpg",
            "logo": ".png",
            "hero": ".png"
        }
        size = size_map.get(image_type, QSize(600, 400))
        if image_type != "logo":
            image = QPixmap(size)
        else:
            image = QImage(size, QImage.Format_ARGB32)
        image.fill(Qt.transparent)

        painter = QPainter(image)
        if image_type != "logo":
            gradient = QLinearGradient(0, 0, size.width(), size.height())
            gradient.setColorAt(0.0, QColor("#444444"))
            gradient.setColorAt(1.0, QColor("#888888"))
            painter.fillRect(image.rect(), gradient)

        if image_type != "hero":
            painter.setPen(Qt.white)
            font = QFont("Arial", 40)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(image.rect(), Qt.AlignCenter, text)
        painter.end()

        temp_path = os.path.join(os.getcwd(), f"temp/temp_{image_type}_{text}{type.get(image_type)}")
        image.save(temp_path)
        return temp_path
