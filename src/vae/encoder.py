import torch

class VAEEncoder:
    def __init__(self, mask_encoder):
        self.enc = mask_encoder

    def forward(self, x, mask):
        mu, logvar = self.enc(x, mask)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
