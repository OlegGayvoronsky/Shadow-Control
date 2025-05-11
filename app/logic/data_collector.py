import os
import cv2
import time
import numpy as np
import mediapipe as mp
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QProgressBar, QPushButton, QVBoxLayout, QMessageBox

from logic.utils import extract_keypoints

class DataCollectorThread(QThread):
    update_frame = Signal(np.ndarray)
    update_progress = Signal(int, int)
    update_class_progress = Signal(int, int)
    finished = Signal()

    def __init__(self, actions, start_folder, no_sequences, sequence_length, data_path):
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)

        self.actions = np.array(actions) if actions else np.array([])
        self.start_folders = {action: start_folder for action in actions}
        self.no_sequences = no_sequences
        self.sequence_length = sequence_length
        self.start_folder = start_folder
        self.running = True

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=False,
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
            for sequence in range(self.start_folders[action], self.start_folders[action] + self.no_sequences):
                if not self.running:
                    break

                self.show_message.emit(f"Готовься: {action} ({sequence}/{self.no_sequences})")
                time.sleep(2)

                for i in range(3, 0, -1):
                    self.show_message.emit(f"Сбор начнется через {i}...")
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

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(self.data_path, action, str(sequence), f"{frame_num}_3d")
                    np.save(npy_path, keypoints)

                self.update_progress.emit(sequence + 1)

            self.update_class_progress.emit(idx + 1)

        cap.release()
        self.show_message.emit("Сбор завершен")

    def stop(self):
        self.running = False


class DataCollectionWindow(QWidget):
    def __init__(self, actions, start_folder, no_sequences, sequence_length, data_path):
        super().__init__()
        self.setWindowTitle("Сбор данных")
        self.setFixedSize(700, 600)

        self.image_label = QLabel("Ожидание видео...")
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("Ожидание...")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(no_sequences)

        self.class_progress_bar = QProgressBar()
        self.class_progress_bar.setMaximum(len(actions))

        self.stop_button = QPushButton("Остановить")
        self.stop_button.clicked.connect(self.stop_collection)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        layout.addWidget(QLabel("Прогресс текущего класса:"))
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Прогресс по классам:"))
        layout.addWidget(self.class_progress_bar)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        self.thread = DataCollectorThread(actions, start_folder, no_sequences, sequence_length, data_path)
        self.thread.update_frame.connect(self.update_image)
        self.thread.update_progress.connect(self.progress_bar.setValue)
        self.thread.update_class_progress.connect(self.class_progress_bar.setValue)
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