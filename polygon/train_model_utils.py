import os
import random

import cv2
import numpy as np
import torch
from sklearn.utils import shuffle
from skmultilearn.model_selection import iterative_train_test_split
from torch import nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class ActionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Определение модели
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.relu(self.fc1(lstm_out[:, -1, :]))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return pose


def data_load_and_separate(actions, DATA_PATH, SEQUENCE_LENGTH, label_map, num_classes):
    def get_shifts(label, window, lshift, rshift):
        windows, labels = [], []
        intervals = np.linspace(lshift, rshift, 21)
        for i in range(20):
            shift = random.uniform(intervals[i], intervals[i + 1])
            aug_window = window.copy()
            aug_window[:, ::4][aug_window[:, ::4] > 0] += shift
            aug_window[((aug_window[:, ::4] < 0) | (aug_window[:, ::4] > 1)).repeat(4, axis=1)] = 0
            windows.append(aug_window)
            labels.append(label)
        return windows, labels

    def show_cadr(wdfs, lbls):
        i = 0
        lbl = np.where(lbls[0] == 1)[0][0]
        while True:
            if i == 30:
                break

            frame = np.zeros((480, 640, 3))
            landmarks = wdfs[:, i]  # точки для выбранного кадра
            i += 1

            # предполагаем, что координаты нормализованы в диапазоне [0, 1], нужно перевести в пиксели
            h, w, _ = frame.shape
            landmarks = landmarks.reshape(20, 33, 4)
            for j in range(20):
                for point in landmarks[j]:
                    x, y, z, c = point
                    px = int(x * w)
                    py = int(y * h)
                    cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

            # показать изображение
            cv2.imshow(f"{lbl}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def get_aug(X, y):
        X_aug, y_aug = [], []
        for window, label in zip(X, y):
            X_aug.append(window)
            y_aug.append(label)

            left_board, right_board = (np.min(window[:, 0][window[:, 0] != 0]) if len(window[:, 0][window[:, 0] != 0]) > 0 else 0,
                                       np.max(window[:, 0]))

            windows, labels = get_shifts(label, np.array(window), 0 - left_board, 1 - right_board)
            show_cadr(np.array(windows), labels)
            X_aug.extend(windows)
            y_aug.extend(labels)
        return np.array(X_aug), np.array(y_aug)

    sequences, labels = [], []

    action_loop = tqdm(actions, desc="action loop", leave=False)
    for action in action_loop:
        sequence_loop = tqdm(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int), desc=f"{action} sequence loop", leave=False)
        for sequence in sequence_loop:
            window = []
            frame_loop = tqdm(range(SEQUENCE_LENGTH), desc=f"frame loop", leave=False)
            for frame_num in frame_loop:
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

            frame_loop.close()
        sequence_loop.close()

    X = np.array(sequences)
    y = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        y[i][labels[i]] = 1

    X, y = shuffle(X, y, random_state=42)
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.1)
    X_train, y_train = get_aug(X_train, y_train)
    print(X_train.shape, y_train.shape)

    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
    return X_train, X_test, y_train, y_test

def train_loop(EPOCHS, model, train_loader, val_loader, device, optimizer, criterion, log_dir, best_model_path, accuracy):
    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard
    best_val_loss = float("inf")  # Лучший показатель ошибки
    epoch_loop = tqdm(range(EPOCHS), desc="epoch loop", leave=True)

    for epoch in epoch_loop:
        model.train()
        total_loss = 0
        train_loop = tqdm(train_loader, desc=f"train loop", leave=False)
        val_loop = tqdm(val_loader, desc="valid loop", leave=False)
        for X_batch, y_batch in train_loop:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)  # Логируем train loss

        # ======= Оценка на валидации =======
        model.eval()
        total_val_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for X_val, y_val in val_loop:
                X_val, y_val = X_val.to(device), y_val.to(device)

                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                total_val_loss += val_loss.item()

                val_outputs = torch.sigmoid(val_outputs)
                y_pred_bin = (val_outputs > 0.7).int()
                acc = accuracy(y_pred_bin, y_val)
                total_accuracy += acc.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)  # Логируем val loss
        writer.add_scalar("Accuracy", avg_accuracy, epoch + 1)
        train_loop.close()
        val_loop.close()

        # ======= Сохранение лучшей модели =======
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(best_model_path, "best_model.pth"))  # Сохраняем веса

    writer.close()
    print("Обучение завершено! Лучшая модель сохранена в", best_model_path)


def predict(model, data, device):
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data).float()
    data = data.to(device)
    pred = torch.sigmoid(model(data))
    return (pred > 0.9).int()
