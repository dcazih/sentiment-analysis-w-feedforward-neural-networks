import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5000, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)  # Binary classification (0 or 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)  # raw logits, softmax is applied by loss function instead
