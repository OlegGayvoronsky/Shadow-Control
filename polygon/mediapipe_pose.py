import cv2
import numpy as np
import torch
import mediapipe as mp
import time
from train_model_utils import extract_keypoints, LSTMModel, predict

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=False,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils


actions = np.array(
        ["walking forward", "walking backward", "walking left", "walking right", "running forward", "running back",
        "sit down", "jump", "one-handed weapon attack", "two-handed weapon attack", "shield block", "weapon block",
        "attacking magic", "bowstring pull", "nothing"])
label_map = {action: idx for idx, action in enumerate(actions)}
invers_label_map = {idx: action for idx, action in enumerate(actions)}
num_classes = len(actions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = LSTMModel(33*4, hidden_dim=128, output_dim=15).to(device)
model.load_state_dict(torch.load("checkpoints/experiment_20250403-130045/best_model.pth"))
model.eval()

cap = cv2.VideoCapture(0)
sequence = []
pred = []
prev_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)

    if len(sequence) == 30:
        with torch.no_grad():
            res = predict(model, np.expand_dims(sequence, axis=0), device)[0]
        pred = torch.where(res == 1)[0].cpu()
        if len(pred) == 0:
            pred = torch.tensor([label_map["nothing"]])
        sequence = sequence[-15:]

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for i, lbl in enumerate(pred):
        lbl = lbl.item()
        cv2.putText(frame, f"{invers_label_map[lbl]}", (0, 100 + 10 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow("Pose Detection", frame)

    torch.cuda.empty_cache()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


del results, pred, sequence
torch.cuda.empty_cache()
cap.release()
cv2.destroyAllWindows()
