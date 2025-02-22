import cv2
import torch
from ultralytics import YOLO

# Загружаем модель YOLO-NAS-Pose
model = YOLO("yolo11n-pose.pt")  # Можно заменить на другую версию (m/l)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Запуск веб-камеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Передача кадра в модель
    results = model.predict(frame)

    # Отображение результатов на изображении
    annotated_frame = results[0].plot()  # Метод plot() наносит ключевые точки на изображение

    cv2.imshow("Pose Detection", annotated_frame)

    # Выход из цикла по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
