from math import acos, degrees
import cv2
import numpy as np
import torch
import mediapipe as mp
import time
from matplotlib import pyplot as plt
from train_model_utils import extract_keypoints, LSTMModel, predict


def set_natural_arm_pose(landmarks, hand='left'):
    if hand == 'left':
        shoulder_idx, elbow_idx, wrist_idx = 11, 13, 15
        hip_idx = 23
    else:
        shoulder_idx, elbow_idx, wrist_idx = 12, 14, 16
        hip_idx = 24

    shoulder = landmarks[shoulder_idx]
    hip = landmarks[hip_idx]

    # Рассчитываем вектор от плеча до бедра
    dx = hip.x - shoulder.x
    dy = hip.y - shoulder.y
    dz = hip.z - shoulder.z

    # Позиция локтя: чуть ниже и ближе к центру тела
    elbow = mp_pose.PoseLandmark(elbow_idx)
    landmarks[elbow_idx].x = shoulder.x + dx * 0.5
    landmarks[elbow_idx].y = shoulder.y + dy * 0.5
    landmarks[elbow_idx].z = shoulder.z + dz * 0.5
    landmarks[elbow_idx].visibility = 1.0

    # Позиция кисти: чуть ниже локтя
    landmarks[wrist_idx].x = landmarks[elbow_idx].x + dx * 0.3
    landmarks[wrist_idx].y = landmarks[elbow_idx].y + dy * 0.3 + 0.02  # чуть ниже живота
    landmarks[wrist_idx].z = landmarks[elbow_idx].z + dz * 0.3
    landmarks[wrist_idx].visibility = 1.0

def is_hand_confident(landmarks, hand='left', threshold=0.3):
    ids = [11, 13, 15] if hand == 'left' else [12, 14, 16]
    return all(landmarks[i].visibility > threshold for i in ids)


def is_jump_or_sit(distances, points):
    jump = "nothing"
    sit = "nothing"
    f1 = -(points[-1].y - points[0].y) > 0.1
    f2 = abs(distances[-1] - distances[0]) <= 0.01
    if f1 and f2:
        jump = "jump"

    if distances[-1] < 0.46:
        sit = "sit"
    return jump, sit


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = LSTMModel(33*4, hidden_dim=128, output_dim=num_classes).to(device)
model.load_state_dict(torch.load("checkpoints/experiment_global3.2/best_model.pth"))
model.eval()


prev_time = time.time()
x = np.array([1, 0, 0])
z = np.array([0, 0, 1])
spine = z
angle1 = 0
angle2 = 0

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

k_frames = 0
points = []
distances = []
distance = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        if not is_hand_confident(landmarks, hand="left"):
            set_natural_arm_pose(landmarks, hand="left")
        if not is_hand_confident(landmarks, hand="right"):
            set_natural_arm_pose(landmarks, hand="right")
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    ax.clear()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=15, azim=70)

    if results.pose_world_landmarks:
        landmarks = results.pose_world_landmarks.landmark
        x_vals = [-lm.x for lm in landmarks]
        z_vals = [-lm.y for lm in landmarks]
        y_vals = [-lm.z for lm in landmarks]

        #2d токи носа
        pt = results.pose_landmarks.landmark
        points.append(results.pose_landmarks.landmark[11])
        points = points[-20:]

        #3d токи
        nose = np.array([x_vals[0], y_vals[0], z_vals[0]])
        l_ear = np.array([x_vals[7], y_vals[7], z_vals[7]])
        r_ear = np.array([x_vals[8], y_vals[8], z_vals[8]])
        l_sh = np.array([x_vals[11], y_vals[11], z_vals[11]])
        r_sh = np.array([x_vals[12], y_vals[12], z_vals[12]])
        l_hip = np.array([x_vals[23], y_vals[23], z_vals[23]])
        r_hip = np.array([x_vals[24], y_vals[24], z_vals[24]])

        distance = (l_sh + r_sh) / 2 - (l_hip + r_hip) / 2
        distance[1] = 0
        distance = np.linalg.norm(distance)
        distances.append(distance)
        distances = distances[-20:]
        k_frames += 1
        if k_frames > 20:
            k_frames = 20

        spine = (l_sh + r_sh) / 2 - (l_hip + r_hip) / 2
        spine[1] = 0
        norms = np.linalg.norm(spine)
        if norms != 0: spine /= norms

        head = (l_ear + r_ear) / 2
        vision1, vision2 = nose - head, nose - head
        vision1[0] = 0
        vision2[2] = 0
        norm1 = np.linalg.norm(vision1)
        norm2 = np.linalg.norm(vision2)
        if norm1 != 0: vision1 /= norm1
        if norm2 != 0: vision2 /= norm2
        angle1 = degrees(acos(vision1 @ z))
        angle2 = degrees(acos(vision2 @ x))
        if abs(degrees(acos(spine @ z))) > 3:
            angle1 = 100
            angle2 = 90
        angle1 -= 100
        angle2 -= 90

        ax.scatter(x_vals, y_vals, z_vals, c='r')
        for connection in mp_pose.POSE_CONNECTIONS:
            p1, p2 = connection
            ax.plot([x_vals[p1], x_vals[p2]], [y_vals[p1], y_vals[p2]], [z_vals[p1], z_vals[p2]], 'b')

    fig.canvas.draw()
    fig.canvas.flush_events()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    jump = "nothing"
    sit = "nothing"
    if k_frames == 20:
        jump, sit = is_jump_or_sit(distances, points)
    cv2.putText(frame, f"fps:{round(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"a1: {int(angle1)}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)
    cv2.putText(frame, f"a2: {int(angle2)}", (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)
    if results.pose_landmarks:
        jp = results.pose_landmarks.landmark[11]
        sp = results.pose_landmarks.landmark[12]
        h, w, _ = frame.shape
        jx, sx = int(jp.x * w), int(sp.x * w)
        jy, sy = int(jp.y * h), int(sp.y * h)
        cv2.putText(frame, f"j: {jump}", (jx, jy),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(frame, f"s: {sit}", (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow("Pose Detection", frame)

    torch.cuda.empty_cache()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


del results
torch.cuda.empty_cache()
cap.release()
cv2.destroyAllWindows()