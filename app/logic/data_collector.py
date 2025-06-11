import os
import cv2
import time
import numpy as np
import mediapipe as mp
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import QWidget, QLabel, QProgressBar, QPushButton, QVBoxLayout, QMessageBox, QHBoxLayout

from logic.utils import extract_keypoints

class DataCollectorThread(QThread):
    update_frame = Signal(np.ndarray)
    update_progress = Signal(int, int)
    update_class_progress = Signal(int, int)
    finished = Signal()
    show_message = Signal(str)
    toggle_pause_button = Signal(bool)

    def __init__(self, actions, start_folder, no_sequences, sequence_length, data_path):
        super().__init__()
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)

        self.actions = np.array(actions) if actions else np.array([])
        self.start_folders = {action: start_folder for action in actions}
        self.no_sequences = no_sequences
        self.sequence_length = sequence_length
        self.start_folder = start_folder
        self.running = True
        self.paused = False

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.frame_size = (640, 480)
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.setup_folders()

    def setup_folders(self):
        for action in self.actions:
            os.makedirs(os.path.join(self.data_path, action), exist_ok=True)

            dirmax = len(os.listdir(os.path.join(self.data_path, action)))
            self.start_folders[action] = dirmax + 1
            for sequence in range(1, self.no_sequences + 1):
                os.makedirs(os.path.join(self.data_path, action, str(dirmax + sequence)), exist_ok=True)

    def run(self):
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        for idx, action in enumerate(self.actions):
            self.update_progress.emit(0, self.no_sequences)
            self.paused = True
            self.show_message.emit(f"Класс: '{action}' — нажми 'Возобновить' для начала")
            self.toggle_pause_button.emit(True)

            while self.paused and self.running:
                time.sleep(0.1)

            for i in range(3, 0, -1):
                self.show_message.emit(f"Сбор начнется через {i}...")
                time.sleep(1)
            for sequence in range(self.start_folders[action], self.start_folders[action] + self.no_sequences):
                if not self.running:
                    break

                self.show_message.emit(f"{action} ({sequence - 1}/{self.start_folders[action] - 1 + self.no_sequences})")
                while self.paused:
                    time.sleep(0.1)
                time.sleep(1)
                vid_path = os.path.join(self.data_path, action, str(sequence), "video.mp4")
                video_writer = cv2.VideoWriter(vid_path, self.fourcc, 30, self.frame_size)

                for frame_num in range(self.sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(frame_rgb)

                    if results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                    video_writer.write(frame)
                    self.update_frame.emit(frame)

                    keypoints = extract_keypoints(results, 1)
                    npy_path = os.path.join(self.data_path, action, str(sequence), f"{frame_num}_3d")
                    np.save(npy_path, keypoints)

                self.update_progress.emit(sequence - self.start_folders[action] + 1, self.no_sequences)

            self.update_class_progress.emit(idx + 1, len(self.actions))

        cap.release()
        self.show_message.emit("Сбор завершен")
        self.finished.emit()

    def stop(self):
        self.running = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class DataCollectionWindow(QWidget):
    def __init__(self, actions, start_folder, no_sequences, sequence_length, data_path):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.showMaximized()
        self.is_paused = False

        font_large = QFont("Arial", 27)
        font_medium = QFont("Arial", 14)

        self.image_label = QLabel("Ожидание видео...")
        self.image_label.setFixedSize(700, 500)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(font_large)

        self.status_label = QLabel("Ожидание...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(font_large)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(no_sequences)
        self.progress_bar.setStyleSheet("QProgressBar { height: 30px; font-size: 16px; }")

        self.class_progress_bar = QProgressBar()
        self.class_progress_bar.setMaximum(len(actions))
        self.class_progress_bar.setStyleSheet("QProgressBar { height: 30px; font-size: 16px; }")

        self.toggle_button = QPushButton("Остановить")
        self.toggle_button.setFont(font_medium)
        self.toggle_button.setFixedHeight(40)
        self.toggle_button.clicked.connect(self.toggle_collection)

        label_class_progress = QLabel("Прогресс текущего класса:")
        label_class_progress.setFont(font_medium)

        label_all_progress = QLabel("Прогресс по классам:")
        label_all_progress.setFont(font_medium)

        layout = QVBoxLayout()

        # Центрируем QLabel с видео по горизонтали
        image_layout = QHBoxLayout()
        image_layout.addStretch()
        image_layout.addWidget(self.image_label)
        image_layout.addStretch()

        layout.addLayout(image_layout)
        layout.addWidget(self.status_label)

        label_class_progress = QLabel("Прогресс текущего класса:")
        label_class_progress.setFont(font_medium)
        layout.addWidget(label_class_progress)
        layout.addWidget(self.progress_bar)

        label_all_progress = QLabel("Прогресс по классам:")
        label_all_progress.setFont(font_medium)
        layout.addWidget(label_all_progress)
        layout.addWidget(self.class_progress_bar)

        layout.addWidget(self.toggle_button)

        self.setLayout(layout)

        self.thread = DataCollectorThread(actions, start_folder, no_sequences, sequence_length, data_path)
        self.thread.update_frame.connect(self.update_image)
        self.thread.update_progress.connect(self.progress_bar.setValue)
        self.thread.update_class_progress.connect(self.class_progress_bar.setValue)
        self.thread.toggle_pause_button.connect(self.toggle_collection)
        self.thread.finished.connect(self.on_collection_finished)
        self.thread.show_message.connect(self.status_label.setText)
        self.thread.start()

    def on_collection_finished(self):
        self.thread.quit()
        self.thread.wait()
        self.close()

    def update_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def stop_collection(self):
        self.thread.stop()
        self.thread.wait()
        self.close()

    def toggle_collection(self):
        if self.is_paused:
            self.thread.resume()
            self.toggle_button.setText("Остановить")
            self.is_paused = False
        else:
            self.thread.pause()
            self.toggle_button.setText("Возобновить")
            self.is_paused = True