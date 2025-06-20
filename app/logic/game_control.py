import os
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt, QSize
from PySide6.QtGui import QMovie
from PySide6.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QLabel, QApplication
from flask import Flask, Response

from logic.classification_model import LSTMModel
from logic.utils import extract_keypoints
import torch
import time
import ctypes

import queue
import json
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import pydirectinput as pdi

app = Flask(__name__)
frame_queue = queue.Queue(maxsize=1)
lock = threading.Lock()

# def generate():
#     while True:
#         try:
#             frame = frame_queue.get(timeout=1.0)
#         except queue.Empty:
#             continue
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         if not ret:
#             continue
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
#
#
# @app.route('/')
# def index():
#     return '''
#     <html>
#     <body style="margin:0; overflow:hidden; background:black;">
#         <img id="video" style="width:100%; height:auto;" />
#         <script>
#         const img = document.getElementById("video");
#
#         function loadStream() {
#             img.src = "/video?" + new Date().getTime();
#         }
#
#         img.onerror = () => {
#             console.log("Ошибка загрузки потока, перезагрузить страницу через 3 секунды...");
#             setTimeout(() => location.reload(), 3000);
#         };
#
#         img.onload = () => {
#             console.log("Поток загружен");
#         };
#
#         loadStream();
#         </script>
#     </body>
#     </html>
#     '''
#
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

class VoiceCommandListener(QThread):
    command_received = Signal(str)

    def __init__(self, model_path: str):
        super().__init__()
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, 16000)
        self.q = queue.Queue()
        self._stop_event = False
        self.stream = None

    def callback(self, indata, frames, time, status):
        if self._stop_event:
            return
        self.q.put(bytes(indata))

    def run(self):
        self._stop_event = False
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype='int16',
            channels=1,
            callback=self.callback,
            device=7
        ) as self.stream:
            while not self._stop_event:
                try:
                    data = self.q.get(timeout=0.5)
                    if self.rec.AcceptWaveform(data):
                        text = json.loads(self.rec.Result()).get("text", "")
                        if text:
                            print(text)
                            self.command_received.emit(text)
                except queue.Empty:
                    continue

    def stop(self):
        self._stop_event = True
        self.quit()
        self.wait()


class GameController(QThread):
    # frame_signal = Signal(np.ndarray)
    pdi.FAILSAFE = False

    def __init__(self, actions, walk_actions, model_path, walk_model_path, camera_index, exit):
        super().__init__()
        import mediapipe as mp

        self.paused = True
        self._stop_event = False
        self.exit = exit

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
        self.label_map = {action: idx for idx, action in enumerate(self.actions.keys())}
        self.walk_label_map = {action: idx for idx, action in enumerate(self.walk_actions.keys())
                               if action != "Прыжок" and action != "Сесть"}
        self.walk_label_map["Бездействие"] = len(self.walk_label_map.keys()) - 1
        self.invers_label_map = {idx: action for idx, action in enumerate(self.actions.keys())}
        self.invers_walk_label_map = {idx: action for action, idx in self.walk_label_map.items()}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path, len(self.actions.keys()), 33*4, 0.3)
        self.walk_model = self.load_model(walk_model_path, len(self.walk_actions.keys()) - 2, 23*4, 0)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        self.sequence1 = []
        self.sequence2 = []
        self.prev_time = time.time()

    def handle_voice_command(self, text):
        if "один" in text:
            for i in range(text.count("один")):
                self.press_combination("1", True)
                time.sleep(0.5)
                self.press_combination("1", False)
        if "два" in text:
            for i in range(text.count("два")):
                self.press_combination("2", True)
                time.sleep(0.5)
                self.press_combination("2", False)
        if "три" in text:
            for i in range(text.count("три")):
                self.press_combination("3", True)
                time.sleep(0.5)
                self.press_combination("3", False)
        if "четыре" in text:
            for i in range(text.count("четыре")):
                self.press_combination("4", True)
                time.sleep(0.5)
                self.press_combination("4", False)
        if "пять" in text:
            for i in range(text.count("пять")):
                self.press_combination("5", True)
                time.sleep(0.5)
                self.press_combination("5", False)
        if "шесть" in text:
            for i in range(text.count("шесть")):
                self.press_combination("6", True)
                time.sleep(0.5)
                self.press_combination("6", False)
        if "семь" in text:
            for i in range(text.count("семь")):
                self.press_combination("7", True)
                time.sleep(0.5)
                self.press_combination("7", False)
        if "восемь" in text:
            for i in range(text.count("восемь")):
                self.press_combination("8", True)
                time.sleep(0.5)
                self.press_combination("8", False)
        if "девять" in text:
            for i in range(text.count("девять")):
                self.press_combination("9", True)
                time.sleep(0.5)
                self.press_combination("9", False)
        elif "инвентарь" in text:
            self.press_combination("I", True)
            self.press_combination("I", False)
        elif "года" in text:
            self.press_combination("Z", True)
            time.sleep(1)
            self.press_combination("Z", False)
        elif "остановить" in text:
            self.toggle_pause()
            self.press_combination("escape", True)
            self.press_combination("escape", False)
        elif "продолжить" in text:
            self.toggle_pause()
        elif "меню" in text:
            self.press_combination("escape", True)
            self.press_combination("escape", False)
        elif "конец игры" in text:
            if self.paused:
                self.toggle_pause()
            self.press_combination("escape", True)
            self.press_combination("escape", False)
            if self.exit:
                self.exit(True)

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

    def handle_prediction(self, action_res, walk_res, prev_action):
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
        if abs(horizontal_angle) < 0.7:
            horizontal_angle = 0
        else:
            horizontal_angle = -horizontal_angle * 100
        if abs(vertical_angle) < 0.7:
            vertical_angle = 0
        else:
            vertical_angle = -vertical_angle * 15

        dx = int(horizontal_angle * sens)
        dy = int(vertical_angle * sens)

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
        l_sh = np.array([x_vals[11], y_vals[11], z_vals[11]])
        r_sh = np.array([x_vals[12], y_vals[12], z_vals[12]])
        l_hip = np.array([x_vals[23], y_vals[23], z_vals[23]])
        r_hip = np.array([x_vals[24], y_vals[24], z_vals[24]])

        distance = (l_sh2d + r_sh2d) / 2 - (l_hip2d + r_hip2d) / 2
        distance = np.linalg.norm(distance)
        distances.append(distance)
        distances = distances[-15:]

        spine = (l_sh + r_sh) / 2 - (l_hip + r_hip) / 2
        if np.linalg.norm(spine) != 0:
            spine /= np.linalg.norm(spine)
        sh = l_sh2d - r_sh2d
        if np.linalg.norm(sh) != 0:
            sh /= np.linalg.norm(sh)
        is_vertical = abs(np.dot(spine, y) - 0.97) <= 0.02
        angle1 = np.dot(sh, [0, 1])

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

        return points, distances, angle1, angle2


    def run(self):
        # threading.Thread(target=run_flask, daemon=True).start()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs("./scrincast", exist_ok=True)

        jump = False
        sit = False
        points = []
        distances = []
        pred = []
        walk_pred = self.walk_label_map["Бездействие"]
        prev_action = []
        prev = self.walk_label_map["Бездействие"]
        segment_number = 0
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
                    self.press_combination(self.walk_actions["Сесть"][0], 1)
                    self.press_combination(self.walk_actions["Сесть"][0], 0)

            if len(self.sequence1) == 15:
                with torch.no_grad():
                    action_res = self.action_predict(np.expand_dims(self.sequence1, axis=0))[0]
                    walk_res, prev = self.walk_predict(prev, np.expand_dims(self.sequence2, axis=0))

                self.handle_prediction(action_res, walk_res, prev_action)
                pred = torch.where(action_res == 1)[0].cpu().tolist()
                prev_action = pred
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
                cv2.putText(frame, f"j: {angle2}", (jx, jy),
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

            video_writer.write(frame)
            # try:
            #     frame_queue.put_nowait(frame.copy())
            # except queue.Full:
            #     pass

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

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self._stop_event = True
        self.quit()
        self.wait()

    def toggle_pause(self):
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

        print(self.paused)
        self.paused = not self.paused
        print(self.paused)

class ControllerThread(QThread):
    controller_ready = Signal()

    def __init__(self, action_model_path, walk_model_path, actions, walk_actions, exit=None):
        super().__init__()
        self.action_model_path = action_model_path
        self.walk_model_path = walk_model_path
        self.actions = actions
        self.walk_actions = walk_actions
        self.exit = exit

    def run(self):
        self.controller = GameController(
            model_path=self.action_model_path,
            walk_model_path=self.walk_model_path,
            actions=self.actions,
            walk_actions=self.walk_actions,
            camera_index=1,
            exit=self.exit
        )
        self.controller.start()
        self.controller_ready.emit()
        self.exec()

    def stop(self):
        if self.controller:
            self.controller.stop()
        self.quit()
        self.wait()

class GameLauncher:
    def __init__(self, parent_window, exe_file, actions, walk_actions, action_model_path, walk_model_path, on_exit_dialog_done=None):
        self.on_exit_dialog_done = on_exit_dialog_done
        self.parent_window = parent_window
        self.exe_file = exe_file
        self.process = None
        self.controller_thread = ControllerThread(action_model_path=action_model_path,
                                                  walk_model_path=walk_model_path,
                                                  actions=actions,
                                                  walk_actions=walk_actions,
                                                  exit=self.exit)
        self.loading_window = LoadingWindow()
        self.launch_game()
        self.voice_model_pth = str(Path(__file__).resolve().parent.parent / "run_model" / "vosk-model-small-ru-0.22")
        self.voice_listener = None
        self.controller_thread.controller_ready.connect(self.on_controller_ready)
        # self.exit_dialog = ExitDialog(self.controller_thread)

    def launch_game(self):
        if not os.path.exists(self.exe_file):
            QMessageBox.warning(self.parent_window, "Ошибка", "Файл игры не найден")
            return

        self.loading_window.show()
        self.controller_thread.start()

    def on_controller_ready(self):
        self.voice_listener = VoiceCommandListener(self.voice_model_pth)

        self.voice_listener.command_received.connect(self.controller_thread.controller.handle_voice_command)
        self.voice_listener.start()
        self.loading_window.close()

        self.process = subprocess.Popen([str(self.exe_file)])
        print("Игра запущена:", self.exe_file)

    def exit(self, result):
        if(result):
            print("Игра завершена")
            self.voice_listener.stop()
            self.controller_thread.stop()
            if self.on_exit_dialog_done:
                self.on_exit_dialog_done(True)

# class ExitDialog(QDialog):
#     def __init__(self, controller_thread):
#         super().__init__()
#         self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
#         self.setFixedSize(300, 100)
#         self.controller_thread = controller_thread
#
#         layout = QVBoxLayout()
#
#         button_layout = QHBoxLayout()
#
#         exit_button = QPushButton("Выйти")
#         exit_button.clicked.connect(self.exit_game)
#         button_layout.addWidget(exit_button)
#
#         cancel_button = QPushButton("Отмена")
#         cancel_button.clicked.connect(self.reject)
#         button_layout.addWidget(cancel_button)
#
#         layout.addLayout(button_layout)
#         self.setLayout(layout)
#
#     def exit_game(self):
#         print("Игра завершена")
#         self.controller_thread.stop()
#         self.accept()

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
