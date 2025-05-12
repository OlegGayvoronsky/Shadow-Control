from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton,
    QHBoxLayout, QComboBox
)
from PySide6.QtGui import QFont, QLinearGradient, QPalette, QColor, QBrush
from PySide6.QtCore import Qt, Signal


class LaunchGameDialog(QDialog):
    model_selected = Signal(str)
    exe_selected = Signal(str)

    def __init__(self, models: list[str], exe_files: list[str]):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setMinimumSize(500, 300)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                              stop:0 #0f0f0f, stop:1 #1a1a2e);
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
                background-color: #6C1DB1;
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
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #6b7280;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #4b5563;
            }
        """)
        launch_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(launch_btn)
        button_layout.addWidget(cancel_btn)
        layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def set_dark_gradient_background(self):
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#0f0f0f"))
        gradient.setColorAt(1.0, QColor("#1a1a2e"))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setAutoFillBackground(True)
        self.setPalette(palette)

    def get_selection(self):
        model = self.model_combo.currentText()
        exe = self.exe_combo.currentText()
        return model, exe
