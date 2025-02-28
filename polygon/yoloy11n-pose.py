import cv2
import torch
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("Pose Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
