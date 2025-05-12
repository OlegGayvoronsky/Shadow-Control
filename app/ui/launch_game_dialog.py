from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout, QComboBox
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, Signal


class LaunchGameDialog(QDialog):
    model_selected = Signal(str)
    exe_selected = Signal(str)

    def __init__(self, models: list[str], exe_files: list[str]):
        super().__init__()
        self.setWindowTitle("Запуск игры")
        self.setMinimumSize(500, 300)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                              stop:0 #1a1a2e, stop:1 #4a00e0);
                color: white;
            }
            QLabel {
                font-size: 18px;
                font-weight: bold;
            }
            QComboBox {
                background-color: #2f2f4f;
                color: white;
                border: 2px solid #7f5af0;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background-color: #7f5af0;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #9b7df0;
            }
        """)

        self.models = models
        self.exe_files = exe_files

        layout = QVBoxLayout()

        # Заголовок
        title = QLabel("Выберите модель и исполняемый файл")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Выбор модели
        model_layout = QVBoxLayout()
        model_label = QLabel("Модель:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(models)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        # Выбор exe-файла
        exe_layout = QVBoxLayout()
        exe_label = QLabel("Исполняемый файл:")
        self.exe_combo = QComboBox()
        self.exe_combo.addItems(exe_files)
        exe_layout.addWidget(exe_label)
        exe_layout.addWidget(self.exe_combo)
        layout.addLayout(exe_layout)

        # Кнопки действий
        button_layout = QHBoxLayout()
        launch_btn = QPushButton("Готово")
        cancel_btn = QPushButton("Отмена")
        launch_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(launch_btn)
        button_layout.addWidget(cancel_btn)
        layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def get_selection(self):
        model = self.model_combo.currentText()
        exe = self.exe_combo.currentText()
        return model, exe
