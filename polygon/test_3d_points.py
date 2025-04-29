import cv2
import numpy as np
import torch
import mediapipe as mp
import time

from matplotlib import pyplot as plt

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
        ["sit down", "jump", "one-handed weapon attack", "two-handed weapon attack", "shield block", "weapon block",
        "attacking magic", "bowstring pull", "nothing"])
label_map = {action: idx for idx, action in enumerate(actions)}
invers_label_map = {idx: action for idx, action in enumerate(actions)}
num_classes = len(actions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = LSTMModel(33*4, hidden_dim=128, output_dim=num_classes).to(device)
model.load_state_dict(torch.load("checkpoints/experiment_20250410-125405/best_model.pth"))
model.eval()

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
prev_time = time.time()
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame_rgb)

    if results.pose_landmarks:
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

        ax.scatter(x_vals, y_vals, z_vals, c='r')
        for connection in mp_pose.POSE_CONNECTIONS:
            p1, p2 = connection
            ax.plot([x_vals[p1], x_vals[p2]], [y_vals[p1], y_vals[p2]], [z_vals[p1], z_vals[p2]], 'b')

    plt.draw()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Pose Detection", frame)

    torch.cuda.empty_cache()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


del results
torch.cuda.empty_cache()
cap.release()
cv2.destroyAllWindows()

# # y, z = [], []
# # for hi in history:
# #     y.append(-hi.y)
# #     z.append(-hi.z)
#
# # y = np.array(y)
# # z = np.array(z)
#
# plt.ioff()
# plt.figure()
#
# plt.plot(history)
# plt.show()