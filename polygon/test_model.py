import os

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from train_model_utils import ActionDataset, LSTMModel, predict
from torchmetrics.classification import Accuracy
from collections import Counter


def get_label(i):
    i += 1
    labels = pd.read_csv("TestData/annotations.csv", sep=',')
    label = labels[labels['start_frame'] <= i].iloc[-1]
    if label['end_frame'] >= i: return label['action']
    return "nothing"


def make_data(data_path, label_map, sequence_length):
    sequences, labels = [], []
    data = np.load(data_path)
    action_loop = tqdm(enumerate(data), desc="action loop", leave=False)
    window_frames, window_labels = [], []
    for i, frame in action_loop:
        window_frames.append(frame)
        window_labels.append(label_map[get_label(i)])
        if len(window_frames) == sequence_length:
            sequences.append(window_frames)
            labels.append(Counter(window_labels).most_common(1)[0][0])
            window_frames = window_frames[-15:]
            window_labels = window_labels[-15:]
    action_loop.close()

    X = np.array(sequences)
    y = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        y[i][labels[i]] = 1

    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    return X, y



if __name__ == "__main__":
    DATA_PATH = 'TestData/data/sequence.npy'
    SEQUENCE_LENGTH = 30
    INPUT_DIM = 33*4
    BATCH_SIZE = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Действия
    actions = np.array(
        ["sit down", "jump", "one-handed weapon attack", "two-handed weapon attack", "shield block", "weapon block",
        "attacking magic", "bowstring pull", "nothing"])
    label_map = {action: idx for idx, action in enumerate(actions)}
    invers_label_map = {idx: action for idx, action in enumerate(actions)}
    test_actions = {}
    num_classes = len(actions)

    X, y = make_data(DATA_PATH, label_map, SEQUENCE_LENGTH)
    test_dataset = ActionDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(INPUT_DIM, hidden_dim=128, output_dim=num_classes).to(device)
    accuracy = Accuracy(task='multilabel', num_labels=num_classes).to(device)
    model.load_state_dict(torch.load("checkpoints/experiment_add_iterative_train_test_split/best_model.pth"))
    model.eval()

    train_loop = tqdm(test_loader, desc="test", leave=False)
    all_labels, all_preds = [], []
    total_accuracy = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = predict(model, X, device)
            acc = accuracy(y_pred, y)
            total_accuracy += acc.item()
            all_preds.append(y_pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    total_accuracy /= len(test_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    for lbl in all_labels:
        id = np.where(lbl == 1)[0][0]
        key = invers_label_map[id]
        test_actions[key] = test_actions.get(key, 0) + 1
    print(test_actions)

    conf_matrices = multilabel_confusion_matrix(all_preds, all_labels)

    for i, cm in enumerate(conf_matrices):
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        plt.title(f"Confusion Matrix for Class {invers_label_map[i]}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    print(f"accuracy: {total_accuracy}")



