import time
import cv2
import torch
from super_gradients.training import models

confidence = 0.6
model = models.get("yolo_nas_pose_n", pretrained_weights="coco_pose")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


def draw_predictions(frame, bboxes, poses, scores):
    for bbox, pose, score in zip(bboxes, poses, scores):
        if score < confidence:
            continue

        # Рисуем bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Рисуем keypoints
        for joint in pose:
            x, y, conf = joint
            if conf > confidence:  # Рисуем только уверенные точки
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    return frame


def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        predict = model.predict(frame, conf=confidence).prediction

        frame = draw_predictions(frame, predict.bboxes_xyxy, predict.poses, predict.scores)
        cv2.imshow("Pose Detection", frame)
        print(time.time() - start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()