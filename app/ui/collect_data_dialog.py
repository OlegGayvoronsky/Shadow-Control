import json
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel,
    QPushButton, QScrollArea, QWidget
)
from PySide6.QtCore import Qt

class CollectDataDialog(QDialog):
    def __init__(self, class_json_path):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setStyleSheet("background-color: #2b2b2b; color: white; font-size: 14px;")
        self.setMinimumWidth(400)

        self.class_checkboxes = []

        with open(class_json_path, "r", encoding="utf-8") as f:
            self.class_data = json.load(f)

        self.class_data["Бездействие"] = ""
        layout = QVBoxLayout(self)

        title = QLabel("Выбери классы для начала сбора данных")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")
        title.setAlignment(Qt.AlignHCenter)
        layout.addWidget(title)
        layout.addSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_container = QWidget()
        scroll_layout = QVBoxLayout(scroll_container)
        scroll_layout.setSpacing(10)

        for idx, class_name in enumerate(self.class_data.keys()):
            checkbox = QCheckBox(class_name)
            if idx > 6:
                checkbox.setChecked(True)
            checkbox.setStyleSheet("color: white;")
            checkbox.stateChanged.connect(lambda state, cb=checkbox: self._update_text_color(cb))
            self.class_checkboxes.append(checkbox)
            scroll_layout.addWidget(checkbox)

        scroll_container.setLayout(scroll_layout)
        scroll.setWidget(scroll_container)
        layout.addWidget(scroll)

        # Кнопки
        buttons_layout = QHBoxLayout()
        start_button = QPushButton("Начать сбор")
        start_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addStretch()
        buttons_layout.addWidget(start_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)

    def load_styles(self):
        return self.setStyleSheet("""
            QDialog {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1,
                                              stop:0 #0f0f0f, stop:1 #1a1a2e);
                color: white;
                font-size: 14px;
            }
            QCheckBox {
                color: white;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:checked {
                image: none;
                border: 1px solid #aaa;
                background-color: #444;
            }
            QCheckBox::indicator:unchecked {
                image: none;
                border: 1px solid #aaa;
                background-color: #444;
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
        """)

    def _update_text_color(self, checkbox):
        checkbox.setStyleSheet("color: white;" if checkbox.isChecked() else "color: gray;")

    def get_selected_classes(self):
        return [cb.text() for cb in self.class_checkboxes if cb.isChecked()]

