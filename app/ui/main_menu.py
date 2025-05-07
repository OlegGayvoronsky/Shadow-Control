import json
import sys
import os
from pathlib import Path
from logic.create_game_folders import CreateGameFolders
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QScrollArea, QGridLayout, QMessageBox
)
from PySide6.QtGui import QPixmap, QIcon, QPalette, QLinearGradient, QColor, QBrush
from PySide6.QtCore import Qt, QSize

GAMES_DIR = Path(__file__).resolve().parent.parent / "games"
creator = CreateGameFolders()

class GameTile(QPushButton):
    def __init__(self, game):
        super().__init__()
        game_name = game.get("name")
        cover_path = game.get("assets").get("cover")
        self.setFixedSize(150, 200)
        self.setCursor(Qt.PointingHandCursor)

        # Устанавливаем иконку обложки
        pixmap = QPixmap(cover_path).scaled(150, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setIcon(QIcon(pixmap))
        self.setIconSize(QSize(150, 200))

        # Стилизация кнопки
        self.setStyleSheet("""
            QPushButton {
                border: 2px solid transparent;
                border-radius: 8px;
                padding: 0px;
            }
            QPushButton:hover {
                border: 2px solid #00ffff;
                background-color: rgba(0, 255, 255, 30);
            }
        """)

        self.clicked.connect(lambda: self.open_game(game_name))

    def open_game(self, game_name):
        QMessageBox.information(self, "Открытие игры", f"Меню игры: {game_name}")



class MainMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shadow — Библиотека игр")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.showMaximized()

        self.set_dark_gradient_background()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Верхняя панель
        top_bar = QHBoxLayout()

        self.games_label = QLabel("Игры: (0)")
        self.games_label.setStyleSheet("color: #a855f7; font-size: 18px; font-weight: bold;")

        add_game_btn = QPushButton("Добавить новую игру")
        add_game_btn.clicked.connect(self.add_game_placeholder)
        add_game_btn.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: #7e22ce;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5e1c99;
            }
        """)

        top_bar.addWidget(self.games_label)
        top_bar.addStretch()
        top_bar.addWidget(add_game_btn)
        main_layout.addLayout(top_bar)

        # Область со скроллом
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.NoFrame)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(20)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.grid_layout.setContentsMargins(20, 20, 20, 20)

        self.scroll_area.setWidget(self.grid_widget)
        main_layout.addWidget(self.scroll_area)

        self.load_games()

    def set_dark_gradient_background(self):
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#0f0f0f"))
        gradient.setColorAt(1.0, QColor("#1a1a2e"))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setAutoFillBackground(True)
        self.setPalette(palette)

    def add_game_placeholder(self):
        QMessageBox.information(self, "Добавить игру", "Заглушка — добавление новой игры")

    def load_games(self):
        self.clear_grid()
        creator.create_folders_from_steam()

        if not GAMES_DIR.exists():
            os.makedirs(GAMES_DIR)

        games = [d for d in GAMES_DIR.iterdir() if d.is_dir()]

        col_count = 7
        row = col = 0

        for gf in games:
            game_folder = GAMES_DIR / gf
            if not game_folder.exists():
                continue

            game = creator.read_manifest(game_folder)
            if game.get('assets').get('cover') == ".":
                continue

            tile = GameTile(game)
            self.grid_layout.addWidget(tile, row, col)

            col += 1
            if col >= col_count:
                col = 0
                row += 1

        self.games_label.setText(f"Игры: ({len(games)})")

    def clear_grid(self):
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)