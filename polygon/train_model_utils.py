import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.relu(self.fc1(lstm_out[:, -1, :]))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def data_load_and_separate(actions, DATA_PATH, SEQUENCE_LENGTH, label_map, num_classes):
    _sequences, _labels = [], []
    for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            _sequences.append(window)
            _labels.append(label_map[action])

    _X = np.array(_sequences)
    _y = np.zeros((len(_labels), num_classes))
    for i in range(len(_labels)):
        _y[i][_labels[i]] = 1

    _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=0.05, random_state=42)
    _X_train, _X_test = torch.tensor(_X_train, dtype=torch.float32), torch.tensor(_X_test, dtype=torch.float32)
    _y_train, _y_test = torch.tensor(_y_train, dtype=torch.long), torch.tensor(_y_test, dtype=torch.long)
    return _X_train, _X_test, _y_train, _y_test


def train_loop(EPOCHS, model, train_loader, val_loader, device, optimizer, criterion, log_dir, best_model_path, accuracy):
    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard
    best_val_loss = float("inf")  # Лучший показатель ошибки

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)  # Логируем train loss

        # ======= Оценка на валидации =======
        model.eval()
        total_val_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                total_val_loss += val_loss.item()
                y_pred_bin = (val_outputs > 0.5).int()
                acc = accuracy(y_pred_bin, y_val)
                total_accuracy += acc

        avg_val_loss = total_val_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)  # Логируем val loss
        writer.add_scalar("Accuracy", avg_accuracy, epoch)

        # ======= Сохранение лучшей модели =======
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(best_model_path, "best_model.pth"))  # Сохраняем веса
            print(f"🔹 Эпоха {epoch+1}: Найдена новая лучшая модель! (Val Loss: {avg_val_loss:.4f})")

    writer.close()
    print("Обучение завершено! Лучшая модель сохранена в", best_model_path)

