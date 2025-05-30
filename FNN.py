import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, use_dropout=False, dropout_rate=0.5):
        super().__init__()
        self.use_dropout = use_dropout 
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(5000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 2)  # Binary classification (neg - 0 or pos - 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.use_dropout: x = self.dropout(x)       
        x = F.relu(self.fc3(x))
        if self.use_dropout: x = self.dropout(x)
        return self.out(x) 