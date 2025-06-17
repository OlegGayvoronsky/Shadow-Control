import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QFileDialog

class SteamPathDialog(QDialog):
    def __init__(self, on_path_selected=None):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setStyleSheet("""
            QDialog {
                background-color: #1c1b22;
                color: white;
                font-size: 14px;
            }
            QLabel#dialogLabel {
                color: white;
                font-size: 16px;
                font-weight: bold;
                qproperty-alignment: AlignCenter;
                padding: 10px;
            }
            QPushButton {
                background-color: #a855f7;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #9333ea;
            }
            QPushButton:pressed {
                background-color: #7e22ce;
            }
        """)
        self.setFixedSize(400, 150)

        self.on_path_selected = on_path_selected

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        label = QLabel("Укажите путь до папки Steam на вашем устройстве.")
        label.setObjectName("dialogLabel")
        label.setWordWrap(True)
        layout.addWidget(label)

        btn = QPushButton("Выбрать путь...")
        btn.clicked.connect(self.select_path)
        layout.addWidget(btn)

        self.setLayout(layout)

    def select_path(self):
        initial_dir = os.path.abspath(os.sep)
        path = QFileDialog.getExistingDirectory(self, "Выберите папку Steam", initial_dir)
        if path and self.on_path_selected:
            self.on_path_selected(path)
        self.accept()
