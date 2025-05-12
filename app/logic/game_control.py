import os
import subprocess

import torch
import cv2
import numpy as np
import time
import mediapipe as mp
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QMessageBox


class GameController(QThread):
    update_frame_signal = Signal(np.ndarray)  # Сигнал для обновления изображения в UI
    update_fps_signal = Signal(float)  # Сигнал для обновления FPS

    def __init__(self, model_path, walk_model_path, camera_index=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=False,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7
        )
        self.drawing = mp.solutions.drawing_utils

        self.actions = np.array([
            "left weapon attack", "right weapon attack", "two-handed weapon attack", "shield block",
            "weapon block", "left attacking magic", "right attacking magic", "left use magic",
            "right use magic", "bowstring pull", "nothing"
        ])
        self.walk_actions = np.array([
            "walk forward", "walk backward", "walk left", "walk right",
            "run forward", "run backward", "nothing"
        ])
        self.label_map = {action: idx for idx, action in enumerate(self.actions)}
        self.invers_label_map = {idx: action for idx, action in enumerate(self.actions)}
        self.invers_walk_label_map = {idx: action for idx, action in enumerate(self.walk_actions)}

        self.model = self.load_model(model_path, len(self.actions))
        self.walk_model = self.load_model(walk_model_path, len(self.walk_actions))

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.sequence = []
        self.prev_time = time.time()

    def load_model(self, path, output_dim):
        model = LSTMModel(33 * 4, hidden_dim=128, output_dim=output_dim).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                self.drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            keypoints = extract_keypoints(results)
            self.sequence.append(keypoints)

            if len(self.sequence) == 30:
                with torch.no_grad():
                    action_res = predict(self.model, np.expand_dims(self.sequence, axis=0), self.device)[0]
                    walk_res = walk_predict(self.walk_model, np.expand_dims(self.sequence, axis=0), self.device)[0]

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
                    walk_label = self.invers_walk_label_map[walk_pred]
                    cv2.putText(frame, f"{walk_label}", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time)
            self.prev_time = curr_time

            # Отправляем FPS на UI
            self.update_fps_signal.emit(fps)

            # Отправляем изображение на UI
            self.update_frame_signal.emit(frame)

            time.sleep(0.01)

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.quit()  # Завершаем QThread
        self.wait()  # Дожидаемся завершения потока


class GameLauncher:
    def __init__(self, exe_file, action_model_path):
        self.exe_file = exe_file
        self.action_model_path = action_model_path

    def launch_game(self):
        if not os.path.exists(self.exe_file):
            QMessageBox.warning(None, "Ошибка", "Файл игры не найден")
            return

        self.process = subprocess.Popen([str(self.exe_file)])
        print("Игра запущена:", self.exe_file)

        self.controller = GameController(model_path=self.global_game_folder / "checkpoints" / model)
        self.control_thread = threading.Thread(target=self.controller.run)
        self.control_thread.start()

            # 🔥 Отслеживаем завершение игры
            monitor_thread = threading.Thread(target=self.monitor_game)
            monitor_thread.start()

    def monitor_game(self):
        self.process.wait()  # ждет завершения игры
        print("Игра завершена")
        if self.controller:
            self.controller.stop()
        self.control_thread.join()
