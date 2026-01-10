import torch
import torch.nn as nn

class MaskEncoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        h = self.net(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
