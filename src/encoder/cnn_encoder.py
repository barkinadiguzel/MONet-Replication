import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, 5, stride=2, padding=2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
