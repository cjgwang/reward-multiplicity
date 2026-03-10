import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DeepCNN(nn.Module):
    """
    CNN with hidden FC layer — more capacity to represent 'follow the star'
    (which requires comparing agent and star channel positions).

    Architecture: conv1 -> ReLU -> flatten -> fc1 -> ReLU -> fc_out
    """
    def __init__(self, in_channels: int = 3, hidden_channels: int = 4,
                 H: int = 7, W: int = 7, fc_hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(hidden_channels * H * W, fc_hidden)
        self.fc_out = nn.Linear(fc_hidden, 1)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc_out(x)


class GridDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.X[idx])            # float32 (C,H,W)
        label = torch.from_numpy(self.y[idx]).float()  # float32 (1,)
        return img, label


class CNN(nn.Module):
    """
    Tiny regression CNN outputting a single scalar per image.

    Architecture: conv1 -> ReLU -> flatten -> fc_out
    fc_out input size = hidden_channels * H * W (no pooling).
    """
    def __init__(self, in_channels: int = 3, hidden_channels: int = 1, H: int = 7, W: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.fc_out = nn.Linear(hidden_channels * H * W, 1)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        return self.fc_out(x)
