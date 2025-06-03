import cv2
import numpy as np
import torch
import mediapipe as mp
import time

from train_model_utils import extract_keypoints, LSTMModel, predict, walk_predict, turn_predict

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=False,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils


actions = np.array(["Удар левой"
                        ,"Удар правой"
                        ,"Двуручный удар"
                        ,"Блок щитом"
                        ,"Удар щитом"
                        ,"Блок оружием"
                        ,"Удар оружием"
                        ,"Атака магией с левой руки"
                        ,"Атака магией с правой руки"
                        ,"Использование магии с левой руки"
                        ,"Использование магии с правой руки"
                        ,"Выстрел из лука"
                        ,"Бездействие"])
label_map = {action: idx for idx, action in enumerate(actions)}
invers_label_map = {idx: action for idx, action in enumerate(actions)}
num_classes = len(actions)

walk_actions = np.array(
        ["Ходьба вперед", "Ходьба назад", "Ходьба влево", "Ходьба вправо", "Бег вперед", "Бездействие"])
walk_label_map = {action: idx for idx, action in enumerate(walk_actions)}
invers_walk_label_map = {idx: action for idx, action in enumerate(walk_actions)}

turn_actions = np.array(
        ["Поворот направо", "Поворот налево", "Поворот вверх", "Поворот вниз", "Бездействие"])
turn_label_map = {action: idx for idx, action in enumerate(turn_actions)}
invers_turn_label_map = {idx: action for idx, action in enumerate(turn_actions)}
non_arm_indices = [
        0,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32
    ]
INPUT_DIM = len(non_arm_indices)*4
num_walk_classes = len(walk_actions)
num_turn_classes = len(turn_actions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = LSTMModel(33*4, hidden_dim=128, output_dim=num_classes).to(device)
model.load_state_dict(torch.load("checkpoints/experiment_global3.2/best_model.pth"))
model.eval()

walk_model = LSTMModel(INPUT_DIM, hidden_dim=128, output_dim=num_walk_classes, dropout=0.1).to(device)
walk_model.load_state_dict(torch.load("checkpoints/run_model_experiment_global3.2/best_model.pth"))
walk_model.eval()

turn_model = LSTMModel(INPUT_DIM, hidden_dim=128, output_dim=num_turn_classes, dropout=0.1).to(device)
turn_model.load_state_dict(torch.load("checkpoints/turn_model_experiment_global3/best_model.pth"))
turn_model.eval()

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
sequence1 = []
sequence2 = []
prev_time = time.time()
pred = []
walk_pred = walk_label_map["Бездействие"]
turn_pred = turn_label_map["Бездействие"]
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    keypoints1 = extract_keypoints(results, 1)
    keypoints2 = extract_keypoints(results, 2)
    sequence1.append(keypoints1)
    sequence2.append(keypoints2)

    if len(sequence1) == 30:
        # cv2.putText(frame, f"O", (500, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
        with torch.no_grad():
            res = predict(model, np.expand_dims(sequence1, axis=0), device)[0]
            walk_res = walk_predict(walk_model, np.expand_dims(sequence2, axis=0), device)[0]
            turn_res = turn_predict(turn_model, np.expand_dims(sequence2, axis=0), device)[0]
        pred = torch.where(res == 1)[0].cpu()
        walk_pred = walk_res
        turn_pred = turn_res
        sequence1 = sequence1[-20:]
        sequence2 = sequence2[-20:]

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    for i, lbl in enumerate(pred):
        lbl = lbl.item()
        cv2.putText(frame, f"{invers_label_map[lbl]}", (0, 100 + 200 * i),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    if results.pose_landmarks:
        point = results.pose_landmarks.landmark[23]
        tpoint = results.pose_landmarks.landmark[12]
        h, w, _ = frame.shape
        x, tx = int(point.x * w), int(tpoint.x * w)
        y, ty = int(point.y * h), int(tpoint.y * h)
        cv2.putText(frame, f"{invers_walk_label_map[walk_pred]}", (x, y),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(frame, f"{invers_turn_label_map[turn_pred]}", (tx, ty),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("Pose Detection", frame)

    torch.cuda.empty_cache()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


del results, pred, sequence1, sequence2
torch.cuda.empty_cache()
cap.release()
cv2.destroyAllWindows()
