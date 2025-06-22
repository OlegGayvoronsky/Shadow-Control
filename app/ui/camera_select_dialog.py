import json
import os
from pathlib import Path
import sounddevice as sd
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
        self.index = 0
        self.micro_index = 0
        cnfg_pth = str(Path(__file__).resolve().parent.parent / Path("config"))
        if os.path.exists(os.path.join(cnfg_pth, "camera_micro_index.json")):
            with open(cnfg_pth + "/camera_micro_index.json", "r") as f:
                data = json.load(f)
                self.index = data.get("camera_index")
                self.micro_index = data.get("micro_index")

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
        if self.combo.count() >= self.index:
            self.combo.setCurrentIndex(self.index)
        for i in range(self.combo.count()):
            if self.combo.itemData(i) == self.index:
                self.combo.setCurrentIndex(i)
                break

        audio_label = QLabel("Выберите микрофон для голосового управления:")
        audio_label.setObjectName("dialogLabel")
        audio_label.setWordWrap(True)
        layout.addWidget(audio_label)

        self.audio_combo = QComboBox()
        self.populate_audio_devices()
        for i in range(self.audio_combo.count()):
            if self.audio_combo.itemData(i) == self.micro_index:
                self.audio_combo.setCurrentIndex(i)
                break

        layout.addWidget(self.audio_combo)
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

    def populate_audio_devices(self):
        self.audio_combo.clear()
        devices = sd.query_devices()
        seen_names = set()

        for i, dev in enumerate(devices):
            name = dev['name'].lower()

            if any(skip in name for skip in ["переназначение", "первичный", "default"]):
                continue

            if dev['max_input_channels'] > 0:
                try:
                    with sd.InputStream(device=i, channels=1, samplerate=16000):
                        display_name = dev['name']
                        if dev['max_input_channels'] >= 2:
                            display_name += " [2ch]"

                        if display_name not in seen_names:
                            seen_names.add(display_name)
                            self.audio_combo.addItem(f"{display_name} (ID {i})", i)
                except Exception:
                    continue

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
            selected_camera = self.combo.currentData()
            selected_audio = self.audio_combo.currentData()
            self.on_camera_selected(selected_camera, selected_audio)
        self.accept()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        super().closeEvent(event)

