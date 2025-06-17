from PySide6.QtCore import QThread, Signal, Qt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PySide6.QtGui import QMovie, QPalette, QLinearGradient, QColor, QBrush, QPixmap, QIcon
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QPushButton, QHBoxLayout
from torch.utils.data import DataLoader, Dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy

from logic.classification_model import LSTMModel


class ActionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ActionModelTrainer(QThread):
    data_progress = Signal(int)
    train_progress = Signal(int)
    training_finished = Signal()

    def __init__(self, train_name, actions, directories, game_path, sequence_length, epochs, num_classes, batch_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.model = LSTMModel(33*4, hidden_dim=128, output_dim=self.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy_fn = Accuracy(task='multilabel', num_labels=num_classes).to(self.device)
        self.actions = np.array(actions) if actions else np.array([])
        self.directories = np.array(directories) if directories else np.array([])
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.label_map = {action: idx for idx, action in enumerate(actions)}
        self.batch_size = batch_size

        self.data_path = game_path / "VidData"
        self.log_dir = game_path / "runs" / train_name
        self.best_model_path = game_path / "checkpoints" / train_name
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.best_model_path, exist_ok=True)

    def run(self):
        sequences, labels = [], []
        total_actions = self.num_classes

        for i, action in enumerate(self.directories):
            action_path = self.data_path / action
            sequence_ids = sorted(map(int, os.listdir(action_path)))
            for sequence in sequence_ids:
                window = []
                for frame_num in range(self.sequence_length):
                    frame_path = action_path / str(sequence) / f"{frame_num}_3d.npy"
                    res = np.load(frame_path)
                    window.append(res)
                sequences.append(window)
                labels.append(np.zeros(total_actions))
                for lbl in action.split(" + "):
                    labels[-1][self.label_map[lbl]] = 1
            progress = int((i + 1) / total_actions * 100)
            self.data_progress.emit(progress)

        X = np.array(sequences)
        y = np.array(labels)

        X, y = shuffle(X, y, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

        X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
        y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

        train_dataset = ActionDataset(X_train, y_train)
        val_dataset = ActionDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        writer = SummaryWriter(log_dir=self.log_dir)
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)

                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)

            self.model.eval()
            total_val_loss = 0
            total_accuracy = 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)

                    val_outputs = self.model(X_val)
                    val_loss = self.criterion(val_outputs, y_val)
                    total_val_loss += val_loss.item()

                    val_outputs = torch.sigmoid(val_outputs)
                    y_pred_bin = (val_outputs > 0.8).int()
                    acc = self.accuracy_fn(y_pred_bin, y_val)
                    total_accuracy += acc.item()

            avg_val_loss = total_val_loss / len(val_loader)
            avg_accuracy = total_accuracy / len(val_loader)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
            writer.add_scalar("Accuracy", avg_accuracy, epoch + 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(self.best_model_path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.best_model_path, "best_model.pth"))

            self.train_progress.emit(int((epoch + 1) / self.epochs * 100))

        writer.close()
        self.training_finished.emit()


class TrainingWindow(QWidget):
    def __init__(self, train_name, actions, directories, game_path, sequence_length, epochs, num_classes, batch_size):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setFixedSize(700, 400)

        self.set_dark_gradient_background()

        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignCenter)

        self.gear_container = QWidget()
        self.gear_layout = QHBoxLayout(self.gear_container)
        self.gear_layout.setAlignment(Qt.AlignCenter)
        self.gear_layout.setContentsMargins(0, 0, 0, 0)
        self.gear_layout.setSpacing(0)

        self.gear_label = QLabel()
        self.gear_label.setFixedSize(256, 256)
        self.gear_label.setScaledContents(True)
        self.gear_label.setAlignment(Qt.AlignCenter)
        self.movie = QMovie("assets/ai_training.gif")
        self.gear_label.setMovie(self.movie)
        self.movie.start()

        self.gear_layout.addWidget(self.gear_label)
        self.layout.addWidget(self.gear_container)

        self.progress = QProgressBar(self)
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #444;
                border-radius: 5px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #7e22ce;
                width: 20px;
            }
        """)
        self.layout.addWidget(self.progress)

        self.final_label = QLabel("Процесс завершён", self)
        self.final_label.setAlignment(Qt.AlignCenter)
        self.final_label.setStyleSheet("font-size: 18px; color: white;")
        self.final_label.setVisible(False)
        self.layout.addWidget(self.final_label)

        self.close_button = QPushButton(self)
        self.close_button.setIcon(QIcon("assets/close.png"))
        self.close_button.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                font-size: 18px;
                color: #ffffff;
            }
            QPushButton:hover {
                color: #ff0000;
            }
        """)
        self.close_button.setFixedSize(30, 30)
        self.close_button.clicked.connect(self.close)

        self.close_button.move(self.width() - 40, 10)
        self.close_button.setVisible(False)

        self.trainer = ActionModelTrainer(train_name, actions, directories, game_path, sequence_length, epochs, num_classes,
                                          batch_size)
        self.trainer.data_progress.connect(self.update_data_progress)
        self.trainer.train_progress.connect(self.update_train_progress)
        self.trainer.training_finished.connect(self.finish_training)

        self.trainer.start()

    def set_dark_gradient_background(self):
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#0f0f0f"))
        gradient.setColorAt(1.0, QColor("#1a1a2e"))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setAutoFillBackground(True)
        self.setPalette(palette)

    def closeEvent(self, event):
        if self.trainer and self.trainer.isRunning():
            self.trainer.quit()
            self.trainer.wait()
        event.accept()

    def update_data_progress(self, value):
        self.progress.setValue(value)
        self.progress.setFormat("Загрузка данных: %p%")

    def update_train_progress(self, value):
        self.progress.setValue(value)
        self.progress.setFormat("Обучение модели: %p%")

    def finish_training(self):
        self.progress.setVisible(False)
        self.movie.stop()

        self.check_animation = QMovie("assets/complete.gif")
        self.gear_label.setMovie(self.check_animation)
        self.gear_label.setVisible(True)
        self.final_label.setVisible(True)

        self.check_animation.start()

        self.last_frame_index = self.check_animation.frameCount() - 1


        def enable_close():
            self.close_button.setVisible(True)

        def on_frame_changed(frame_number):
            if frame_number == self.last_frame_index:
                self.check_animation.stop()
                enable_close()

        self.check_animation.frameChanged.connect(on_frame_changed)