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

    with torch.no_grad():
        results = model.predict(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("Pose Detection", annotated_frame)

    del annotated_frame
    torch.cuda.empty_cache()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

del results
torch.cuda.empty_cache()
cap.release()
cv2.destroyAllWindows()
