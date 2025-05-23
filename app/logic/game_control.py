import os
import subprocess

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt, QSize
from PySide6.QtGui import QMovie, QImage, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QLabel, QApplication, QDialog, QHBoxLayout, QPushButton
from logic.classification_model import LSTMModel
from logic.utils import extract_keypoints
import torch
import time
import ctypes

import pydirectinput as pdi

class GameController(QThread):
    frame_signal = Signal(np.ndarray)

    def __init__(self, actions, walk_actions, model_path, walk_model_path, camera_index):
        super().__init__()
        import mediapipe as mp

        self.paused = False
        self._stop_event = False
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_connections = mp.solutions.pose.POSE_CONNECTIONS
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=False,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7
        )
        self.drawing = mp.solutions.drawing_utils

        self.actions = actions
        self.walk_actions = walk_actions
        self.label_map = {action: idx for idx, action in enumerate(self.actions.keys())}
        self.walk_label_map = {action: idx for idx, action in enumerate(self.walk_actions.keys())
                               if action != "Прыжок" and action != "Сесть"}
        self.walk_label_map["Бездействие"] = len(self.walk_label_map.keys()) - 1
        self.invers_label_map = {idx: action for idx, action in enumerate(self.actions.keys())}
        self.invers_walk_label_map = {idx: action for action, idx in self.walk_label_map.items()}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path, len(self.actions.keys()))
        self.walk_model = self.load_model(walk_model_path, len(self.walk_actions.keys()) - 2)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        self.sequence = []
        self.prev_time = time.time()

    def load_model(self, path, output_dim):
        model = LSTMModel(33 * 4, hidden_dim=128, output_dim=output_dim).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    def press_combination(self, combo: str):
        if not combo:
            return
        keys = combo.lower().split('+')

        for key in keys:
            pdi.keyDown(key)
        time.sleep(0.05)
        for key in reversed(keys):
            pdi.keyUp(key)

    def press_mouse(self, button_name: str):
        if button_name == 'left click':
            pdi.mouseDown(button='left')
            pdi.mouseUp(button='left')
        elif button_name == 'right click':
            pdi.mouseDown(button='right')
            pdi.mouseUp(button='right')

    def handle_prediction(self, action_res, walk_res):
        pred = torch.where(action_res == 1)[0].cpu().tolist()
        walk_pred = walk_res

        if len(walk_pred) > 0:
            walk_action = self.invers_walk_label_map[walk_pred[0]]
            walk_key = self.walk_actions.get(walk_action).lower()
            if walk_key and walk_key != "":
                self.press_combination(walk_key)

        for p in pred:
            action = self.invers_label_map[p]
            key = self.actions.get(action).lower()
            if key in ['left click', 'right click']:
                self.press_mouse(key)
            elif key != "":
                self.press_combination(key)

    def run(self):
        while not self._stop_event:
            while self.paused:
                time.sleep(0.1)

            ret, frame = self.cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if not results.pose_landmarks:
                continue

            self.drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_connections)
            keypoints = extract_keypoints(results)
            self.sequence.append(keypoints)

            if len(self.sequence) == 30:
                with torch.no_grad():
                    action_res = self.action_predict(np.expand_dims(self.sequence, axis=0))[0]
                    walk_res = self.walk_predict(np.expand_dims(self.sequence, axis=0))

                self.handle_prediction(action_res, walk_res)
                self.sequence = self.sequence[-20:]
                pred = torch.where(action_res == 1)[0].cpu().tolist()
                walk_pred = walk_res
                self.sequence = self.sequence[-20:]

                for i, lbl in enumerate(pred):
                    label_name = self.invers_label_map[lbl]
                    cv2.putText(frame, f"{label_name}", (0, 100 + 200 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                if results.pose_landmarks:
                    point = results.pose_landmarks.landmark[23]
                    h, w, _ = frame.shape
                    x, y = int(point.x * w), int(point.y * h)
                    walk_label = self.invers_walk_label_map[walk_pred[0]]
                    cv2.putText(frame, f"{walk_label}", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time)
            self.prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.frame_signal.emit(frame)

        self.cleanup()

    def action_predict(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        data = data.to(self.device)
        pred = torch.sigmoid(self.model(data))
        return (pred >= 0.8).int()

    def walk_predict(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        data = data.to(self.device)
        pred = torch.sigmoid(self.walk_model(data))[0]
        v, i = torch.max(pred, dim=0)
        if v >= 0.8:
            return [i.item()]
        return [6]

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

    def __init__(self, action_model_path, walk_model_path, actions, walk_actions):
        super().__init__()
        self.action_model_path = action_model_path
        self.walk_model_path = walk_model_path
        self.actions = actions
        self.walk_actions = walk_actions

    def run(self):
        self.controller = GameController(
            model_path=self.action_model_path,
            walk_model_path=self.walk_model_path,
            actions=self.actions,
            walk_actions=self.walk_actions,
            camera_index=1
        )
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
                                                  walk_actions=walk_actions)
        self.loading_window = LoadingWindow()
        self.launch_game()
        self.controller_thread.controller_ready.connect(self.on_controller_ready)
        self.exit_dialog = ExitDialog(self.controller_thread)

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

    def setup_shortcuts(self):
        self.pause_shortcut = QShortcut(QKeySequence("Ctrl+P"), self.parent_window)
        self.pause_shortcut.activated.connect(self.toggle_pause)

        self.cam_shortcut = QShortcut(QKeySequence("Ctrl+C"), self.parent_window)
        self.cam_shortcut.activated.connect(self.toggle_camera_window)

        self.exit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self.parent_window)
        self.exit_shortcut.activated.connect(self.show_exit_dialog)

    def toggle_pause(self):
        self.controller_thread.controller.toggle_pause()

    def toggle_camera_window(self):
        if self.camera_window and self.camera_window.isVisible():
            self.camera_window.hide()
        else:
            if not self.camera_window:
                self.camera_window = FloatingCameraWindow(self.controller_thread.controller)
            self.camera_window.show()

    def show_exit_dialog(self):
        if self.exit_dialog.exec() and self.camera_window and self.camera_window.isVisible():
            self.camera_window.hide()

class ExitDialog(QDialog):
    def __init__(self, controller_thread):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(300, 100)
        self.controller_thread = controller_thread

        layout = QVBoxLayout()

        button_layout = QHBoxLayout()

        exit_button = QPushButton("Выйти")
        exit_button.clicked.connect(self.exit_game)
        button_layout.addWidget(exit_button)

        cancel_button = QPushButton("Отмена")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def exit_game(self):
        print("Игра завершена")
        self.controller_thread.stop()
        self.accept()

class LoadingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        self.movie = QMovie("assets/ai_training.gif")
        self.movie.setScaledSize(QSize(200, 200))
        self.label.setMovie(self.movie)
        self.movie.start()

        layout.addWidget(self.label)
        self.setLayout(layout)
        self.resize(200, 200)

    def showEvent(self, event):
        self.center()
        super().showEvent(event)

    def center(self):
        qr = self.frameGeometry()
        screen = QApplication.primaryScreen()
        if screen:
            cp = screen.availableGeometry().center()
            qr.moveCenter(cp)
            self.move(qr.topLeft())


class FloatingCameraWindow(QWidget):
    def __init__(self, controller):
        super().__init__()

        # Окно поверх всех, без рамок
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)

        self.controller = controller
        self.label = QLabel()
        self.label.setFixedSize(640, 480)
        self.label.setStyleSheet("background-color: black;")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.controller.frame_signal.connect(self.update_frame)

        self.dragging = False
        self.drag_position = None

        self.show()  # Показать окно
        self.force_stay_on_top()  # Зафиксировать поверх всех окон

    def force_stay_on_top(self):
        """Принудительно установить окно поверх всех окон через WinAPI"""
        hwnd = int(self.winId())  # Получаем HWND окна
        HWND_TOPMOST = -1
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_SHOWWINDOW = 0x0040

        ctypes.windll.user32.SetWindowPos(
            hwnd, HWND_TOPMOST, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
        )

    def update_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.dragging and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            event.accept()

