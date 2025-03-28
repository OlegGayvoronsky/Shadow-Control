from utils import create_segment, show
import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
start = time.time()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_size = (640, 480)

# Создаем объект для записи видео
video_writer = cv2.VideoWriter("./videos/output.mp4", fourcc, 30, frame_size)
cap = cv2.VideoCapture(0)

cadr = 0
frames = []
batch = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # with torch.no_grad():
    #      result = model.predict(frame)
    #
    # frame = result[0].plot()
    # cv2.imshow("Pose Detection", frame)
    #
    # torch.cuda.empty_cache()
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    cadr += 1
    frames.append(frame)
    end = time.time()
    if end - start >= 1:
        batch += 1
        if batch == 1:
            frames = []
            start = time.time()
            cadr = 0
            continue

        with torch.no_grad():
            results = model.predict(frames)
        segment = [result.keypoints.xyn[0].cpu().numpy() for result in results]

        orig_imgs = [result.orig_img for result in results]
        segment, ids = create_segment(segment)
        show(segment, orig_imgs, ids, video_writer)

        frames = []
        start = time.time()
        print(start - end)
        cadr = 0
        if batch == 11: break

del results
torch.cuda.empty_cache()
cap.release()
cv2.destroyAllWindows()
