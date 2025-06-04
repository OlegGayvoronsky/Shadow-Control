from flask import Flask, Response
import cv2
import threading

app = Flask(__name__)

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.lock = threading.Lock()
        threading.Thread(target=self.update_frame, daemon=True).start()

    def update_frame(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame

camera = Camera()

def generate():
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '''
    <html>
    <body style="margin:0; overflow:hidden; background:transparent;">
    <img src="/video" style="width:100%; height:auto;"/>
    </body>
    </html>
    '''

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
