import os
import subprocess
import threading
from math import atan2, degrees
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt, QSize
from PySide6.QtGui import QMovie, QImage, QPixmap, QShortcut, QKeySequence
from PySide6.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QLabel, QApplication, QDialog, QHBoxLayout, QPushButton
from flask import Flask, Response
from matplotlib import pyplot as plt

from logic.classification_model import LSTMModel
from logic.utils import extract_keypoints
from logic.rstp_server import download_mediamtx, start_mediamtx, stop_mediamtx, install_ffmpeg
import torch
import time
import ctypes

import pydirectinput as pdi

# app = Flask(__name__)
# last_frame = None
# lock = threading.Lock()

# def generate():
#     global last_frame
#     while True:
#         with lock:
#             if last_frame is None:
#                 continue
#             ret, jpeg = cv2.imencode('.jpg', last_frame)
#             if not ret:
#                 continue
#             frame = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
#
# @app.route('/video')
# def video_feed():
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
# def run_flask():
#     print("Starting Flask server...")
#     app.run(host='0.0.0.0', port=8080, threaded=True, debug=False, use_reloader=False)


class GameController(QThread):
    # frame_signal = Signal(np.ndarray)

    def __init__(self, path, actions, walk_actions, turn_actions, model_path, walk_model_path, turn_model_path, camera_index):
        super().__init__()
        import mediapipe as mp

        # install_ffmpeg()
        self.paused = False
        self._stop_event = False

        # exe_path = download_mediamtx()
        # self.server_proc = start_mediamtx(exe_path, path)
        # time.sleep(2)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_connections = mp.solutions.pose.POSE_CONNECTIONS
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7
        )
        self.drawing = mp.solutions.drawing_utils

        self.actions = actions
        self.walk_actions = walk_actions
        self.turn_actions = turn_actions
        self.label_map = {action: idx for idx, action in enumerate(self.actions.keys())}
        self.walk_label_map = {action: idx for idx, action in enumerate(self.walk_actions.keys())
                               if action != "Прыжок" and action != "Сесть"}
        self.turn_label_map = {action: idx for idx, action in enumerate(self.turn_actions.keys())}
        self.walk_label_map["Бездействие"] = len(self.walk_label_map.keys()) - 1
        self.invers_label_map = {idx: action for idx, action in enumerate(self.actions.keys())}
        self.invers_walk_label_map = {idx: action for action, idx in self.walk_label_map.items()}
        self.invers_turn_label_map = {idx: action for action, idx in self.turn_label_map.items()}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path, len(self.actions.keys()), 33*4, 0.3)
        self.walk_model = self.load_model(walk_model_path, len(self.walk_actions.keys()) - 2, 23*4, 0)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        self.sequence1 = []
        self.sequence2 = []
        self.prev_time = time.time()

    def load_model(self, path, output_dim, input_dim, dropout):
        model = LSTMModel(input_dim, hidden_dim=128, output_dim=output_dim, dropout=dropout).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    def press_combination(self, combo, mode):
        keys = combo.lower().split('+')
        for key in keys:
            key = key.strip()
            if mode:
                pdi.keyDown(key)
            else:
                pdi.keyUp(key)

    def press_mouse(self, button_name, mode):
        if mode:
            pdi.mouseDown(button=button_name.split()[0])
        else:
            pdi.mouseUp(button=button_name.split()[0])

    def handle_prediction(self, action_res, walk_res):
        pred = torch.where(action_res == 1)[0].cpu().tolist()
        walk_pred = walk_res

        walk_action = self.invers_walk_label_map[walk_pred]
        for w_act in self.walk_actions:
            if(w_act == "Бездействие" or w_act == "Прыжок" or w_act == "Сесть"):
                continue
            walk_key = self.walk_actions[w_act][0].lower()
            flag = self.walk_actions[w_act][1]
            mode = True if (w_act == walk_action) else False
            if flag != mode:
                self.walk_actions[w_act][1] = mode
                self.press_combination(walk_key, mode)

        if len(pred) == 0:
            return

        drop_flags = self.label_map["Бездействие"] in pred
        for act in self.actions:
            if (act == "Бездействие"):
                continue
            key = self.actions[act][0].lower()
            flag = self.actions[act][1]
            mode = True if (self.label_map[act] in pred and not drop_flags) else False
            if flag != mode:
                self.actions[act][1] = mode
                if key in ['left click', 'right click']:
                    self.press_mouse(key, mode)
                else:
                    self.press_combination(key, mode)

    def move_mouse_by_head_angles(self, vertical_angle, horizontal_angle, sens=0.1):
        if abs(horizontal_angle) < 0.8:
            horizontal_angle = 0
        else:
            horizontal_angle = 10 * horizontal_angle / abs(horizontal_angle)

        # Смещения по осям
        dx = int(horizontal_angle * sens)
        dy = int(vertical_angle * sens)

        # Плавное движение мыши
        pdi.moveRel(round(dx), round(dy))

    def is_jump_or_sit(self, distances, points):
        jump = False
        sit = False
        f1 = points[-1].y < 0.14
        f2 = abs(distances[-1] - distances[0]) <= 0.02
        if f1 and f2:
            jump = True

        f1 = points[-1].y > 0.43
        f2 = distances[-1] < 0.35
        if f1 and f2:
            sit = True
        return jump, sit

    def algebra_calculate(self, results, points, distances):
        y = np.array([0, 1, 0])

        landmarks = results.pose_world_landmarks.landmark

        x_vals = [-lm.x for lm in landmarks]
        y_vals = [-lm.z for lm in landmarks]
        z_vals = [-lm.y for lm in landmarks]

        pt = results.pose_landmarks.landmark
        points.append(pt[11])
        points = points[-15:]

        l_sock = np.array([pt[31].x, pt[31].y])
        r_sock = np.array([pt[32].x, pt[32].y])
        l_heel = np.array([pt[29].x, pt[29].y])
        r_heel = np.array([pt[30].x, pt[30].y])
        l_sh2d = np.array([pt[11].x, pt[11].y])
        r_sh2d = np.array([pt[12].x, pt[12].y])
        l_hip2d = np.array([pt[23].x, pt[23].y])
        r_hip2d = np.array([pt[24].x, pt[24].y])

        distance = (l_sh2d + r_sh2d) / 2 - (l_hip2d + r_hip2d) / 2
        distance = np.linalg.norm(distance)
        distances.append(distance)
        distances = distances[-15:]

        l_foot = l_sock - l_heel
        if np.linalg.norm(l_foot) != 0:
            l_foot /= np.linalg.norm(l_foot)

        r_foot = r_sock - r_heel
        if np.linalg.norm(r_foot) != 0:
            r_foot /= np.linalg.norm(r_foot)

        l_angle = np.dot(l_foot, [1, 0])
        r_angle = np.dot(r_foot, [1, 0])
        range = abs(l_angle - r_angle)
        angle2 = max(l_angle, r_angle) if range < 0.1 else 0
        angle1 = 0

        return points, distances, angle1, angle2


    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # global last_frame
        # threading.Thread(target=run_flask, daemon=True).start()
        os.makedirs("./scrincast", exist_ok=True)
        fps_list = []

        jump = False
        sit = False
        points = []
        distances = []
        pred = []
        walk_pred = self.walk_label_map["Бездействие"]
        prev = self.walk_label_map["Бездействие"]
        segment_number = 0
        f = 0
        frames = []
        vid_path = os.path.join("./scrincast", f"video{segment_number}.mp4")
        video_writer = cv2.VideoWriter(vid_path, fourcc, 30, (640, 480))
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
            keypoints1 = extract_keypoints(results, 1)
            keypoints2 = extract_keypoints(results, 2)
            self.sequence1.append(keypoints1)
            self.sequence2.append(keypoints2)
            points, distances, angle1, angle2 = self.algebra_calculate(results, points, distances)

            if len(distances) == 15:
                jump, sit = self.is_jump_or_sit(distances, points)
                f_jump = self.walk_actions["Прыжок"][1]
                f_sit = self.walk_actions["Сесть"][1]
                if(jump != f_jump):
                    self.walk_actions["Прыжок"][1] = jump
                    self.press_combination(self.walk_actions["Прыжок"][0], jump)
                if (sit != f_sit):
                    self.walk_actions["Сесть"][1] = sit
                    self.press_combination(self.walk_actions["Сесть"][0], sit)

            if len(self.sequence1) == 15:
                f = 1
                with torch.no_grad():
                    action_res = self.action_predict(np.expand_dims(self.sequence1, axis=0))[0]
                    walk_res, prev = self.walk_predict(prev, np.expand_dims(self.sequence2, axis=0))

                self.handle_prediction(action_res, walk_res)
                pred = torch.where(action_res == 1)[0].cpu().tolist()
                walk_pred = walk_res
                self.sequence1 = self.sequence1[-10:]
                self.sequence2 = self.sequence2[-10:]

            self.move_mouse_by_head_angles(angle1, angle2)

            for i, lbl in enumerate(pred):
                label_name = self.invers_label_map[lbl]
                cv2.putText(frame, f"{label_name}", (0, 100 + 200 * i),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

            if results.pose_landmarks:
                point = results.pose_landmarks.landmark[23]
                h, w, _ = frame.shape
                x, y = int(point.x * w), int(point.y * h)
                jp = results.pose_landmarks.landmark[11]
                sp = results.pose_landmarks.landmark[12]
                jx, sx = int(jp.x * w), int(sp.x * w)
                jy, sy = int(jp.y * h), int(sp.y * h)
                cv2.putText(frame, f"j: {jump}", (jx, jy),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
                cv2.putText(frame, f"s: {sit}", (sx, sy),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                walk_label = self.invers_walk_label_map[walk_pred]
                cv2.putText(frame, f"{walk_label}", (x, y),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time)
            self.prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frames.append(frame)
            fps_list.append(fps)

            if f:
                segment_number += 1
                print(vid_path)
                for frame in frames:
                    video_writer.write(frame)
                frames = frames[-10:]
                f = 0
                # vid_path = os.path.join("./scrincast", f"video{segment_number}.mp4")
                # video_writer = cv2.VideoWriter(vid_path, fourcc, 30, (640, 480))

            # with lock:
            #     last_frame = frame.copy()


        for k in self.actions.keys():
            if self.actions[k][1]:
                key = self.actions[k][0]
                if key in ['left click', 'right click']:
                    self.press_mouse(key, 0)
                else:
                    self.press_combination(key, 0)

        for k in self.walk_actions.keys():
            if self.walk_actions[k][1]:
                key = self.walk_actions[k][0]
                if key in ['left click', 'right click']:
                    self.press_mouse(key, 0)
                else:
                    self.press_combination(key, 0)

        plt.figure(figsize=(10, 10))
        plt.plot(fps_list)
        plt.savefig("./scrincast/graph.png")
        self.cleanup()

    def action_predict(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        data = data.to(self.device)
        pred = torch.sigmoid(self.model(data))
        return (pred >= 0.8).int()

    def walk_predict(self, prev, data):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        data = data.to(self.device)
        pred = torch.sigmoid(self.walk_model(data))[0]
        v, i = torch.max(pred, dim=0)
        i = i.item()
        if v >= 0.9:
            if i == 2 or i == 3 or i == 5:
                if prev == i or prev == 5:
                    return i, i
                else:
                    return prev, i
            else:
                return i, i
        return prev, prev

    def turn_predict(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        data = data.to(self.device)
        pred = torch.sigmoid(self.turn_model(data))[0]
        v, i = torch.max(pred, dim=0)
        if v >= 0.9:
            return [i.item()]
        return [4]

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        # stop_mediamtx(self.server_proc)

    def stop(self):
        self._stop_event = True
        self.quit()
        self.wait()

    def toggle_pause(self):
        self.paused = not self.paused




class ControllerThread(QThread):
    controller_ready = Signal()

    def __init__(self, path, action_model_path, walk_model_path, turn_model_path, actions, walk_actions, turn_actions):
        super().__init__()
        self.action_model_path = action_model_path
        self.walk_model_path = walk_model_path
        self.turn_model_path = turn_model_path
        self.actions = actions
        self.walk_actions = walk_actions
        self.turn_actions = turn_actions
        self.path = path

    def run(self):
        self.controller = GameController(
            path=self.path,
            model_path=self.action_model_path,
            walk_model_path=self.walk_model_path,
            turn_model_path=self.turn_model_path,
            actions=self.actions,
            walk_actions=self.walk_actions,
            turn_actions=self.turn_actions,
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
    def __init__(self, parent_window, path, exe_file, actions, walk_actions, turn_actions, action_model_path, walk_model_path, turn_model_path):
        self.parent_window = parent_window
        self.exe_file = exe_file
        self.process = None
        self.camera_window = None
        self.controller_thread = ControllerThread(path=path,
                                                  action_model_path=action_model_path,
                                                  walk_model_path=walk_model_path,
                                                  turn_model_path=turn_model_path,
                                                  actions=actions,
                                                  walk_actions=walk_actions,
                                                  turn_actions=turn_actions)
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

        # self.cam_shortcut = QShortcut(QKeySequence("Ctrl+C"), self.parent_window)
        # self.cam_shortcut.activated.connect(self.toggle_camera_window)

        self.exit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self.parent_window)
        self.exit_shortcut.activated.connect(self.show_exit_dialog)

    def toggle_pause(self):
        self.controller_thread.controller.toggle_pause()

    # def toggle_camera_window(self):
    #     if self.camera_window and self.camera_window.isVisible():
    #         self.camera_window.hide()
    #     else:
    #         if not self.camera_window:
    #             self.camera_window = FloatingCameraWindow(self.controller_thread.controller)
    #         self.camera_window.show()

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


# class FloatingCameraWindow(QWidget):
#     def __init__(self, controller):
#         super().__init__()
#
#         # Окно поверх всех, без рамок
#         self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
#
#         self.controller = controller
#         self.label = QLabel()
#         self.label.setFixedSize(640, 480)
#         self.label.setStyleSheet("background-color: black;")
#
#         layout = QVBoxLayout()
#         layout.setContentsMargins(0, 0, 0, 0)
#         layout.addWidget(self.label)
#         self.setLayout(layout)
#
#         self.controller.frame_signal.connect(self.update_frame)
#
#         self.dragging = False
#         self.drag_position = None
#
#         self.show()  # Показать окно
#         self.force_stay_on_top()  # Зафиксировать поверх всех окон
#
#     def force_stay_on_top(self):
#         """Принудительно установить окно поверх всех окон через WinAPI"""
#         hwnd = int(self.winId())  # Получаем HWND окна
#         HWND_TOPMOST = -1
#         SWP_NOMOVE = 0x0002
#         SWP_NOSIZE = 0x0001
#         SWP_SHOWWINDOW = 0x0040
#
#         ctypes.windll.user32.SetWindowPos(
#             hwnd, HWND_TOPMOST, 0, 0, 0, 0,
#             SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
#         )
#
#     def update_frame(self, frame):
#         rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb_image.shape
#         bytes_per_line = ch * w
#         qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         self.label.setPixmap(QPixmap.fromImage(qt_image))
#
#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.dragging = True
#             self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
#             event.accept()
#
#     def mouseMoveEvent(self, event):
#         if self.dragging and event.buttons() & Qt.LeftButton:
#             self.move(event.globalPosition().toPoint() - self.drag_position)
#             event.accept()
#
#     def mouseReleaseEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.dragging = False
#             event.accept()

