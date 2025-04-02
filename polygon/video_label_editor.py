import sys
import cv2
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QLabel, QPushButton, QFileDialog,
                             QVBoxLayout, QHBoxLayout, QWidget, QComboBox,
                             QTableWidget, QTableWidgetItem, QSlider, QLineEdit)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt


class VideoLabelingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.video_path = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.fps = 30
        self.current_time = 0
        self.annotations = []
        self.start_time = None
        self.playing = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

    def initUI(self):
        layout = QVBoxLayout()

        self.video_label = QLabel("Видео не загружено")
        layout.addWidget(self.video_label)

        self.time_label = QLabel("Текущее время: 0.0 сек")
        layout.addWidget(self.time_label)

        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Загрузить видео")
        self.load_btn.clicked.connect(self.load_video)
        btn_layout.addWidget(self.load_btn)

        self.play_pause_btn = QPushButton("Старт")
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        btn_layout.addWidget(self.play_pause_btn)

        self.start_btn = QPushButton("Старт метки")
        self.start_btn.clicked.connect(self.set_start)
        btn_layout.addWidget(self.start_btn)

        self.end_btn = QPushButton("Конец метки")
        self.end_btn.clicked.connect(self.set_end)
        btn_layout.addWidget(self.end_btn)

        self.new_label_input = QLineEdit()
        self.new_label_input.setPlaceholderText("Добавить новую метку")
        btn_layout.addWidget(self.new_label_input)

        self.add_label_btn = QPushButton("Добавить")
        self.add_label_btn.clicked.connect(self.add_custom_label)
        btn_layout.addWidget(self.add_label_btn)

        self.action_combo = QComboBox()
        self.action_combo.addItems([])
        self.action_combo.currentIndexChanged.connect(self.on_action_changed)
        btn_layout.addWidget(self.action_combo)

        self.export_btn = QPushButton("Экспорт в CSV")
        self.export_btn.clicked.connect(self.export_csv)
        btn_layout.addWidget(self.export_btn)

        layout.addLayout(btn_layout)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderReleased.connect(self.set_video_position)
        layout.addWidget(self.slider)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Start Frame", "End Frame", "Action"])
        layout.addWidget(self.table)

        self.setLayout(layout)

    def add_custom_label(self):
        new_label = self.new_label_input.text().strip()
        if new_label and new_label not in [self.action_combo.itemText(i) for i in range(self.action_combo.count())]:
            self.action_combo.addItem(new_label)
            self.new_label_input.clear()

        self.setFocus()

    def on_action_changed(self):
        # После изменения значения в комбобоксе возвращаем фокус на окно
        self.setFocus()

    def load_video(self):
        file_dialog = QFileDialog()
        self.video_path, _ = file_dialog.getOpenFileName(self, "Выберите видео", "", "Video Files (*.mp4 *.avi)")
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.slider.setEnabled(True)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.setMaximum(total_frames)

            # Устанавливаем первый кадр
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.next_frame()  # Отображаем первый кадр, но не запускаем воспроизведение

            self.playing = False  # Убеждаемся, что видео в паузе
            self.play_pause_btn.setText("Старт")  # Обновляем кнопку
        self.setFocus()

    def toggle_playback(self):
        self.setFocus()
        if self.cap is None:
            return

        if self.playing:
            self.timer.stop()
            self.play_pause_btn.setText("Старт")
        else:
            self.timer.start(1000 // self.fps)
            self.play_pause_btn.setText("Пауза")

        self.playing = not self.playing

    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.playing = False
            self.play_pause_btn.setText("Старт")
            return

        self.current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.time_label.setText(f"Текущее время: {self.current_time:.2f} сек")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

        self.slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

    def set_video_position(self):
        if self.cap is None:
            return

        frame_number = self.slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.next_frame()

    def set_start(self):
        if self.cap:
            self.start_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.setFocus()

    def set_end(self):
        if self.cap and hasattr(self, 'start_frame'):
            end_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            action = self.action_combo.currentText()
            self.annotations.append((self.start_frame, end_frame, action))
            del self.start_frame
            self.update_table()
        self.setFocus()

    def update_table(self):
        self.table.setRowCount(len(self.annotations))
        for i, (start_frame, end_frame, action) in enumerate(self.annotations):
            self.table.setItem(i, 0, QTableWidgetItem(str(start_frame)))
            self.table.setItem(i, 1, QTableWidgetItem(str(end_frame)))
            self.table.setItem(i, 2, QTableWidgetItem(action))

    def export_csv(self):
        df = pd.DataFrame(self.annotations, columns=["start_frame", "end_frame", "action"])
        df.to_csv("annotations.csv", index=False)
        print("Разметка сохранена в annotations.csv")

        self.setFocus()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.toggle_playback()
        elif event.key() == Qt.Key.Key_Left:
            self.step_frame(-1)
        elif event.key() == Qt.Key.Key_Right:
            self.step_frame(1)

        event.accept()  # Окончательно блокируем стандартное поведение

    def step_frame(self, step):
        if self.cap is None:
            return

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Коррекция индексации
        new_frame = max(0, min(current_frame + step, total_frames - 1))  # Ограничение диапазона

        if new_frame == current_frame:
            return  # Если кадр не изменился, ничего не делаем

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)  # Перемещение без цикла
        _, frame = self.cap.read()  # Читаем один кадр
        self.update_frame_display(frame, new_frame)  # Отображение кадра

    def update_frame_display(self, frame, frame_number):
        """ Обновляет изображение и положение слайдера """
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))
            self.slider.setValue(frame_number)
            self.current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            self.time_label.setText(f"Текущее время: {self.current_time:.2f} сек")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoLabelingApp()
    window.show()
    sys.exit(app.exec())
