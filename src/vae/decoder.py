import torch.nn as nn

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=16, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels * 64 * 64)
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(z.size(0), 3, 64, 64)
