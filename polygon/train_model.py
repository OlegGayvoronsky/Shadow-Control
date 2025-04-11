import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from train_model_utils import ActionDataset, LSTMModel, data_load_and_separate, train_loop


if __name__ == "__main__":
    DATA_PATH = 'VidData'
    log_dir = f"runs/experiment_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    best_model_path = f"checkpoints/experiment_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(best_model_path, exist_ok=True)
    SEQUENCE_LENGTH = 30
    INPUT_DIM = 33*4
    BATCH_SIZE = 32
    EPOCHS = 2000
    LEARNING_RATE = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Действия
    actions = np.array(
        ["left weapon attack", "right weapon attack", "two-handed weapon attack", "shield block", "weapon block", "left attacking magic", "right attacking magic", "left use magic", "right use magic", "bowstring pull", "nothing"])
    label_map = {action: idx for idx, action in enumerate(actions)}
    num_classes = len(actions)

    X_train, X_test, y_train, y_test = data_load_and_separate(actions, DATA_PATH, SEQUENCE_LENGTH, label_map, num_classes)

    train_dataset = ActionDataset(X_train, y_train)
    test_dataset = ActionDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(INPUT_DIM, hidden_dim=128, output_dim=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    accuracy = Accuracy(task='multilabel', num_labels=num_classes).to(device)

    train_loop(EPOCHS, model, train_loader, test_loader, device, optimizer, criterion, log_dir, best_model_path, accuracy)