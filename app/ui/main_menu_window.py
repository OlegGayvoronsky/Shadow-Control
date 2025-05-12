import json
import os
from pathlib import Path
from logic.create_game_folders import CreateGameFolders
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QScrollArea, QGridLayout, QMessageBox
)
from PySide6.QtGui import QPixmap, QIcon, QPalette, QLinearGradient, QColor, QBrush
from PySide6.QtCore import Qt, QSize, QTimer
import shutil

GAMES_DIR = Path(__file__).resolve().parent.parent / "games"
creator = CreateGameFolders()

class GameTile(QPushButton):
    def __init__(self, game, parent=None):
        super().__init__(parent)
        self.game = game
        self.parent_window = parent
        self.setFixedSize(150, 200)
        self.setCursor(Qt.PointingHandCursor)

        appid = game.get("appid")
        if appid == -1:
            for k, path in game.get("assets").items():
                game["assets"][k] = str(GAMES_DIR / path).replace("\\", "/")
        cover_path = game.get("assets").get("cover")

        pixmap = QPixmap(str(cover_path)).scaled(150, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setIcon(QIcon(pixmap))
        self.setIconSize(QSize(150, 200))

        self.setStyleSheet("""
            QPushButton {
                border: 2px solid transparent;
                border-radius: 8px;
                padding: 0px;
            }
            QPushButton:hover {
                border: 10px solid #02e5e5;
                background-color: rgba(0, 255, 255, 30);
            }
        """)

        self.clicked.connect(self.open_game)

    def open_game(self):
        from ui.game_menu_window import GameMenu
        game_folder = Path("games") / self.game["installdir"]
        self.game_window = GameMenu(self.game, game_folder)
        self.game_window.show()

        if self.parent_window:
            self.parent_window.close()



class MainMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shadow — Библиотека игр")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.showMaximized()

        self.set_dark_gradient_background()
        self.tiles = []

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Верхняя панель
        top_bar = QHBoxLayout()
        self.games_label = QLabel("Игры: (0)")
        self.games_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")

        self.add_game_btn = QPushButton("Добавить новую игру")
        self.add_game_btn.clicked.connect(self.add_game)
        self.add_game_btn.setStyleSheet("""
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
        top_bar.addWidget(self.add_game_btn)
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
        QTimer.singleShot(0, self.populate_grid)

    def set_dark_gradient_background(self):
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#0f0f0f"))
        gradient.setColorAt(1.0, QColor("#1a1a2e"))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setAutoFillBackground(True)
        self.setPalette(palette)

    def add_game(self):
        from ui.add_game_dialog import AddGameDialog
        dialog = AddGameDialog("Добавить игру")
        if dialog.exec():
            game_data = dialog.get_data()
            if not game_data:
                return

            game_name = game_data["name"]
            install_dir = game_data["installdir"]
            game_path = GAMES_DIR / install_dir

            if game_path.exists():
                QMessageBox.warning(self, "Ошибка", "Игра с таким именем уже существует.")
                for key, path in game_data["assets"].items():
                    if "temp/" in path:
                        os.unlink(path)
                return

            os.makedirs(game_path / "assets", exist_ok=True)

            for key, path in game_data["assets"].items():
                type = path.split(".")[-1]

                dst = Path(install_dir) / "assets" / f"{key}.{type}"
                shutil.copy(path, GAMES_DIR / dst)
                game_data["assets"][key] = str(dst).replace("\\", "/")
                if "temp/" in path:
                    os.unlink(path)

            with open(game_path / "appmanifest.json", "w", encoding="utf-8") as f:
                json.dump(game_data, f, indent=4, ensure_ascii=False)

            QMessageBox.information(self, "Успех", f"Игра '{game_name}' добавлена.")
            self.load_games()
            QTimer.singleShot(0, self.populate_grid)

    def load_games(self):
        self.clear_grid()
        creator.create_folders_from_steam()

        if not GAMES_DIR.exists():
            os.makedirs(GAMES_DIR)

        games = [d for d in GAMES_DIR.iterdir() if d.is_dir()]
        self.tiles = []

        for gf in games:
            game_folder = GAMES_DIR / gf
            if not game_folder.exists():
                continue

            game = creator.read_manifest(game_folder)
            if game.get('assets').get('cover') == ".":
                continue

            tile = GameTile(game, self)
            self.tiles.append(tile)

        self.games_label.setText(f"Игры: ({len(self.tiles)})")

    def clear_grid(self):
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

    def populate_grid(self):
        self.clear_grid()
        if not self.tiles:
            return

        available_width = self.scroll_area.viewport().width()
        tile_width = 150
        spacing = self.grid_layout.spacing()
        columns = max(1, available_width // (tile_width + spacing))

        row = col = 0
        for tile in self.tiles:
            self.grid_layout.addWidget(tile, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "grid_layout"):
            self.populate_grid()
