import os
import subprocess

import torch
import cv2
import numpy as np
import time
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QMovie, QImage, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QLabel, QApplication
from logic.classification_model import LSTMModel
from logic.utils import extract_keypoints
import mediapipe as mp


class GameController(QThread):
    frame_signal = Signal(np.ndarray)

    def __init__(self, actions, walk_actions, model_path, walk_model_path, camera_index):
        super().__init__()
        self.paused = False
        self._stop_event = False
        self.model_path = model_path
        self.walk_model_path = walk_model_path
        self.actions = actions
        self.walk_actions = walk_actions
        self.label_map = {action: idx for idx, action in enumerate(self.actions.keys())}
        self.walk_label_map = {action: idx for idx, action in enumerate(self.walk_actions.keys())}
        self.invers_label_map = {idx: action for idx, action in enumerate(self.actions.keys())}
        self.invers_walk_label_map = {idx: action for idx, action in enumerate(self.walk_actions.keys())}

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.sequence = []
        self.prev_time = time.time()

    def init_models(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=False,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7
        )
        self.drawing = mp.solutions.drawing_utils
        self.model = self.load_model(self.model_path, len(self.actions))
        self.walk_model = self.load_model(self.walk_model_path, len(self.walk_actions))

    def load_model(self, path, output_dim):
        model = LSTMModel(33 * 4, hidden_dim=128, output_dim=output_dim).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    def run(self):
        while not self._stop_event:
            while self.paused:
                time.sleep(0.1)

            ret, frame = self.cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                self.drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            # keypoints = extract_keypoints(results)
            # self.sequence.append(keypoints)
            #
            # if len(self.sequence) == 30:
            #     with torch.no_grad():
            #         action_res = predict(self.model, np.expand_dims(self.sequence, axis=0), self.device)[0]
            #         walk_res = walk_predict(self.walk_model, np.expand_dims(self.sequence, axis=0), self.device)[0]
            #
            #     pred = torch.where(action_res == 1)[0].cpu().tolist()
            #     walk_pred = walk_res
            #     self.sequence = self.sequence[-20:]

                # for i, lbl in enumerate(pred):
                #     label_name = self.invers_label_map[lbl]
                #     cv2.putText(frame, f"{label_name}", (0, 100 + 200 * i),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                # if results.pose_landmarks:
                #     point = results.pose_landmarks.landmark[23]
                #     h, w, _ = frame.shape
                #     x, y = int(point.x * w), int(point.y * h)
                #     walk_label = self.invers_walk_label_map[walk_pred]
                #     cv2.putText(frame, f"{walk_label}", (x, y),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time)
            self.prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.frame_signal.emit(frame)

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self._stop_event = True
        self.quit()
        self.wait()

    def toggle_pause(self):
        self.paused = not self.paused

class ControllerThread(QThread):
    controller_ready = Signal()

    def __init__(self, action_model_path, walk_model_path, actions, walk_actions, camera_index=1):
        super().__init__()
        self.controller = GameController(
            model_path=action_model_path,
            walk_model_path=walk_model_path,
            actions=actions,
            walk_actions=walk_actions,
            camera_index=1
        )

    def run(self):
        self.controller.init_models()
        self.controller_ready.emit()
        self.controller.start()

    def stop(self):
        if self.controller:
            self.controller.stop()
        self.quit()
        self.wait()

class GameLauncher:
    def __init__(self, parent_window, exe_file, actions, walk_actions, action_model_path, walk_model_path):
        self.parent_window = parent_window
        self.exe_file = exe_file
        self.process = None
        self.camera_window = None
        self.controller_thread = ControllerThread(action_model_path=action_model_path,
                                                  walk_model_path=walk_model_path,
                                                  actions=actions,
                                                  walk_actions=walk_actions,
                                                  camera_index=1)
        self.loading_window = LoadingWindow()
        self.controller_thread.controller_ready.connect(self.on_controller_ready)

    def launch_game(self):
        if not os.path.exists(self.exe_file):
            QMessageBox.warning(self.parent_window, "Ошибка", "Файл игры не найден")
            return

        self.loading_window.show()
        self.controller_thread.start()

    def on_controller_ready(self):
        self.loading_window.close()

        self.process = subprocess.Popen([str(self.exe_file)])
        print("Игра запущена:", self.exe_file)

        self.setup_shortcuts()

        monitor_thread = QThread()
        monitor_thread.run = self.monitor_game
        monitor_thread.start()

    def setup_shortcuts(self):
        self.pause_shortcut = QShortcut(QKeySequence("Ctrl+P"), self.parent_window)
        self.pause_shortcut.activated.connect(self.toggle_pause)

        self.cam_shortcut = QShortcut(QKeySequence("Ctrl+C"), self.parent_window)
        self.cam_shortcut.activated.connect(self.toggle_camera_window)

    def toggle_pause(self):
        self.controller_thread.controller.toggle_pause()

    def toggle_camera_window(self):
        if self.camera_window and self.camera_window.isVisible():
            self.camera_window.hide()
        else:
            if not self.camera_window:
                self.camera_window = FloatingCameraWindow(self.controller_thread.controller)
            self.camera_window.show()

    def monitor_game(self):
        self.process.wait()
        print("Игра завершена")
        self.controller_thread.stop()

class LoadingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        # Используем gif для анимации загрузки
        self.movie = QMovie("assets/loading.gif")
        self.label.setMovie(self.movie)
        self.movie.start()

        layout.addWidget(self.label)
        self.setLayout(layout)
        self.resize(200, 200)
        self.center()

    def center(self):
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )

class FloatingCameraWindow(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.controller = controller
        self.label = QLabel()
        self.label.setFixedSize(320, 240)
        self.label.setStyleSheet("background-color: black;")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.controller.frame_signal.connect(self.update_frame)

    def update_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))
