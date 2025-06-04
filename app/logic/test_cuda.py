# import torch
# import tensorflow as tf
#
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

from flask import Flask, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # или путь к видеофайлу

def generate():
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
