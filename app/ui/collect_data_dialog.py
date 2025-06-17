import json
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel,
    QPushButton, QScrollArea, QWidget, QSpinBox
)
from PySide6.QtCore import Qt

class CollectDataDialog(QDialog):
    def __init__(self, class_json_path):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setMinimumWidth(400)

        self.setStyleSheet("""
            QDialog {
                background-color: #1c1b22;
                color: white;
                font-size: 14px;
            }
            QLabel#titleLabel {
                font-size: 16px;
                font-weight: bold;
                color: white;
                qproperty-alignment: AlignCenter;
                padding: 10px;
            }
            QLabel#countLabel {
                padding: 4px;
            }
            QCheckBox {
                color: white;
                spacing: 6px;
                padding: 4px;
            }
            QCheckBox:!checked {
                color: gray;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #7e22ce;
                border: 1px solid #a855f7;
            }
            QCheckBox::indicator:unchecked {
                background-color: #444;
                border: 1px solid #777;
            }
            QSpinBox {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 4px 8px;
                color: white;
                font-weight: bold;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                subcontrol-origin: border;
                width: 12px;
                background: #333;
                border: none;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background: #555;
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
                background-color: #9333ea;
            }
            QPushButton#cancelBtn {
                background-color: #444;
            }
            QPushButton#cancelBtn:hover {
                background-color: #666;
            }
        """)

        self.class_checkboxes = []

        # Загрузка классов
        with open(class_json_path, "r", encoding="utf-8") as f:
            temporary_d = json.load(f)
            self.class_data = {}
            for i, el in enumerate(temporary_d.keys()):
                if i > 6:
                    self.class_data[el] = temporary_d[el]
        self.class_data["Бездействие"] = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Заголовок
        title = QLabel("Выбери классы для начала сбора данных")
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        # Прокручиваемая область
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_container = QWidget()
        scroll_layout = QVBoxLayout(scroll_container)
        scroll_layout.setSpacing(8)

        for idx, class_name in enumerate(self.class_data.keys()):
            checkbox = QCheckBox(class_name)
            checkbox.setChecked(True)
            self.class_checkboxes.append(checkbox)
            scroll_layout.addWidget(checkbox)

        scroll.setWidget(scroll_container)
        layout.addWidget(scroll)

        # Блок выбора количества
        count_label = QLabel("Сколько примеров собрать")
        count_label.setObjectName("countLabel")
        self.count_spinbox = QSpinBox()
        self.count_spinbox.setMinimum(1)
        self.count_spinbox.setMaximum(10000)
        self.count_spinbox.setValue(100)

        count_layout = QHBoxLayout()
        count_layout.addWidget(count_label)
        count_layout.addStretch()
        count_layout.addWidget(self.count_spinbox)

        layout.addLayout(count_layout)

        # Кнопки
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        start_button = QPushButton("Начать сбор")
        cancel_button = QPushButton("Отмена")
        cancel_button.setObjectName("cancelBtn")

        start_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

        buttons_layout.addWidget(start_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)

    def get_selected_classes(self):
        return [cb.text() for cb in self.class_checkboxes if cb.isChecked()]

    def get_count(self):
        return self.count_spinbox.value()
