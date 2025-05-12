from torch import nn
import torch

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

    def action_predict(model, data, device):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        data = data.to(device)
        pred = torch.sigmoid(model(data))
        return (pred >= 0.8).int()

    def walk_predict(model, data, device):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        data = data.to(device)
        pred = torch.sigmoid(model(data))[0]
        v, i = torch.max(pred, dim=0)
        if v >= 0.8:
            return [i.item()]
        return [6]