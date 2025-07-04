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


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # последний временной шаг
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def extract_keypoints(results, type):
    non_arm_indices = [
        0,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32
    ]
    visibility_threshold = 0.45

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        def process_landmark(lm):
            if lm.visibility < visibility_threshold:
                return [0.0, 0.0, 0.0, 0.0]
            return [(-lm.x + 1) / 2, (-lm.y + 1) / 2, (-lm.z + 1) / 2, lm.visibility]

        if type == 1:
            # Используем все 33 точки
            pose = np.array([process_landmark(lm) for lm in landmarks]).flatten()

        elif type == 2:
            pose = np.array([process_landmark(landmarks[i]) for i in non_arm_indices]).flatten()

    else:
        # В зависимости от типа подставляем нули
        if type == 1:
            pose = np.zeros(33 * 4)
        else:
            pose = np.zeros(len(non_arm_indices) * 4)

    return pose




def data_load_and_separate(actions, DATA_PATH, SEQUENCE_LENGTH, label_map, inverse_label_map, num_classes):
    sequences, labels = [], []

    action_loop = tqdm(actions, desc="action loop", leave=False)
    for action in action_loop:
        sequence_loop = tqdm(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int), desc=f"{action} sequence loop", leave=False)
        for sequence in sequence_loop:
            window = []
            frame_loop = tqdm(range(SEQUENCE_LENGTH), desc=f"frame loop", leave=False)
            for frame_num in frame_loop:
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}_3d.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(np.zeros(num_classes))
            for lbl in action.split(" + "):
                labels[-1][label_map[lbl]] = 1

            frame_loop.close()
        sequence_loop.close()

    X = np.array(sequences)
    y = np.array(labels)

    X, y = shuffle(X, y, random_state=42)
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.1)

    stat = dict()
    for yt in y_test:
        ln = []
        for i in range(len(yt)):
            if(yt[i] == 1): ln.append(inverse_label_map[i])
        sn = " + ".join(ln)
        stat[sn] = stat.get(sn, 0) + 1
    print(stat)

    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
    return X_train, X_test, y_train, y_test

def train_loop(EPOCHS, model, train_loader, val_loader, device, optimizer, criterion, log_dir, best_model_path, accuracy):
    writer = SummaryWriter(log_dir=log_dir)
    best_val_loss = float("inf")
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
        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)

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
                y_pred_bin = (val_outputs > 0.8).int()
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
    return (pred >= 0.8).int()


def walk_predict(model, prev, data, device):
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data).float()
    data = data.to(device)
    pred = torch.sigmoid(model(data))[0]
    v, i = torch.max(pred, dim=0)
    i = i.item()
    if v >= 0.9:
        if i == 2 or i == 3 or i == 5:
            if prev == i or prev == 5:
                return i, i
            else:
                return prev, i
        else:
            return i, i
    return prev, prev
