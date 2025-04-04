import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from train_model_utils import ActionDataset, LSTMModel, data_load_and_separate, predict
from torchmetrics.classification import Accuracy


if __name__ == "__main__":
    DATA_PATH = 'VidData'
    SEQUENCE_LENGTH = 30
    INPUT_DIM = 33*4
    BATCH_SIZE = 32
    EPOCHS = 2000
    LEARNING_RATE = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Действия
    actions = np.array(
        ["walking forward", "walking backward", "walking left", "walking right", "running forward", "running back",
        "sit down", "jump", "one-handed weapon attack", "two-handed weapon attack", "shield block", "weapon block",
        "attacking magic", "bowstring pull", "nothing"])
    label_map = {action: idx for idx, action in enumerate(actions)}
    invers_label_map = {idx: action for idx, action in enumerate(actions)}
    test_actions = {}
    num_classes = len(actions)

    X_train, X_test, y_train, y_test = data_load_and_separate(actions, DATA_PATH, SEQUENCE_LENGTH, label_map, num_classes)
    test_dataset = ActionDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(INPUT_DIM, hidden_dim=128, output_dim=num_classes).to(device)
    accuracy = Accuracy(task='multilabel', num_labels=num_classes).to(device)
    model.load_state_dict(torch.load("checkpoints/experiment_20250403-130045/best_model.pth"))
    model.eval()

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



