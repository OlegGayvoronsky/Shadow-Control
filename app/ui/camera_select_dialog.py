import cv2
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QComboBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

class CameraSelectDialog(QDialog):
    def __init__(self, on_camera_selected=None):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setFixedSize(400, 500)
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
            QComboBox {
                padding: 6px;
                border-radius: 4px;
                background-color: #2a2a33;
                color: white;
            }
        """)

        self.on_camera_selected = on_camera_selected
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        label = QLabel("Выберите камеру для захвата движения:")
        label.setObjectName("dialogLabel")
        label.setWordWrap(True)
        layout.addWidget(label)

        self.combo = QComboBox()
        self.combo.currentIndexChanged.connect(self.change_camera)
        self.populate_camera_list()
        layout.addWidget(self.combo)

        self.video_label = QLabel()
        self.video_label.setFixedSize(360, 240)
        self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #444;")
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        btn = QPushButton("Продолжить")
        btn.clicked.connect(self.select_camera)
        layout.addWidget(btn)

        self.setLayout(layout)

        self.start_preview()

    def populate_camera_list(self):
        self.combo.clear()
        for index in range(5):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.combo.addItem(f"Камера {index}", index)
                cap.release()

    def start_preview(self):
        index = self.combo.currentData()
        if index is not None:
            self.open_camera(index)

    def change_camera(self, index):
        new_index = self.combo.itemData(index)
        if new_index is not None:
            self.open_camera(new_index)

    def open_camera(self, index):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if self.cap.isOpened():
            self.timer.start(30)
        else:
            self.timer.stop()
            self.video_label.clear()

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(img)
            self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def select_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.on_camera_selected:
            self.on_camera_selected(self.combo.currentData())
        self.accept()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        super().closeEvent(event)

